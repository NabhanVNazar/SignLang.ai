import {
  ArrowDown,
  ArrowRight,
  Check,
  CircleHelp,
  Hand,
  Info,
  LoaderCircle,
  LockKeyhole,
  ScanLine,
  ShieldCheck,
  Sparkles,
  Video,
  Wifi,
  X,
} from "lucide-react";
import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from "@mediapipe/tasks-vision";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type MutableRefObject,
} from "react";

type ImagePrediction = {
  label: string;
  class_index: number;
  model_version: string;
  preprocessing_schema_version: number;
  status: string;
};
type FramePrediction = {
  id: number;
  frameNumber: number;
  label: string;
  capturedAt: string;
};
type ApiHealth =
  | "checking"
  | "ready"
  | "model_not_ready"
  | "auth_not_configured"
  | "unavailable";
type CameraState = "idle" | "starting" | "tracking" | "error";

const HAND_CONNECTIONS: Array<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15],
  [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17],
];
const LANDMARKER_WASM =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.2.0/wasm";
const LANDMARKER_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const LIVE_PREDICTION_INTERVAL_MS = 1200;
const MAX_VISIBLE_PREDICTIONS = 8;
const MAX_CAPTURE_WIDTH = 480;

function drawLandmarks(
  canvas: HTMLCanvasElement,
  hands: HandLandmarkerResult["landmarks"],
) {
  const context = canvas.getContext("2d");
  if (!context) return;
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  context.clearRect(0, 0, width, height);
  context.lineWidth = Math.max(2, 2.2 * ratio);
  context.lineCap = "round";
  context.strokeStyle = "#b9f36b";
  context.fillStyle = "#fbfff4";

  for (const hand of hands) {
    for (const [from, to] of HAND_CONNECTIONS) {
      const start = hand[from];
      const end = hand[to];
      if (!start || !end) continue;
      context.beginPath();
      context.moveTo(start.x * width, start.y * height);
      context.lineTo(end.x * width, end.y * height);
      context.stroke();
    }
    for (const point of hand) {
      context.beginPath();
      context.arc(point.x * width, point.y * height, 3.2 * ratio, 0, Math.PI * 2);
      context.fill();
    }
  }
}

function useHandTracking(
  videoRef: MutableRefObject<HTMLVideoElement | null>,
  canvasRef: MutableRefObject<HTMLCanvasElement | null>,
  onHandsDetected: (handsDetected: number) => void,
) {
  const detectorRef = useRef<HandLandmarker | null>(null);
  const animationRef = useRef<number | null>(null);
  const onHandsDetectedRef = useRef(onHandsDetected);
  onHandsDetectedRef.current = onHandsDetected;

  const stop = useCallback(() => {
    if (animationRef.current !== null) {
      cancelAnimationFrame(animationRef.current);
      animationRef.current = null;
    }
    detectorRef.current?.close();
    detectorRef.current = null;
    const canvas = canvasRef.current;
    if (canvas) {
      const context = canvas.getContext("2d");
      context?.clearRect(0, 0, canvas.width, canvas.height);
    }
  }, [canvasRef]);

  const start = useCallback(async () => {
    stop();
    const video = videoRef.current;
    if (!video) throw new Error("Camera preview is unavailable.");
    const vision = await FilesetResolver.forVisionTasks(LANDMARKER_WASM);
    const detector = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: LANDMARKER_MODEL },
      runningMode: "VIDEO",
      numHands: 2,
    });
    detectorRef.current = detector;

    let previousVideoTime = -1;
    const detect = () => {
      const activeVideo = videoRef.current;
      const activeDetector = detectorRef.current;
      const canvas = canvasRef.current;
      if (
        !activeVideo ||
        !activeDetector ||
        !canvas ||
        activeVideo.readyState < HTMLMediaElement.HAVE_CURRENT_DATA
      ) {
        animationRef.current = requestAnimationFrame(detect);
        return;
      }
      if (activeVideo.currentTime !== previousVideoTime) {
        previousVideoTime = activeVideo.currentTime;
        const result = activeDetector.detectForVideo(
          activeVideo,
          performance.now(),
        );
        drawLandmarks(canvas, result.landmarks);
        onHandsDetectedRef.current(result.landmarks.length);
      }
      animationRef.current = requestAnimationFrame(detect);
    };
    animationRef.current = requestAnimationFrame(detect);
  }, [canvasRef, stop, videoRef]);

  useEffect(() => stop, [stop]);
  return { start, stop };
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const liveRecognitionRef = useRef(false);
  const recognitionSessionRef = useRef(0);
  const frameNumberRef = useRef(0);
  const predictionInFlightRef = useRef(false);
  const abortControllerRef = useRef<AbortController | null>(null);
  const [cameraState, setCameraState] = useState<CameraState>("idle");
  const [cameraError, setCameraError] = useState("");
  const [trackingCount, setTrackingCount] = useState(0);
  const [apiHealth, setApiHealth] = useState<ApiHealth>("checking");
  const [predictions, setPredictions] = useState<FramePrediction[]>([]);
  const [predicting, setPredicting] = useState(false);
  const [apiError, setApiError] = useState("");
  const [liveRecognition, setLiveRecognition] = useState(false);
  const { start: startTracking, stop: stopTracking } = useHandTracking(
    videoRef,
    canvasRef,
    setTrackingCount,
  );

  useEffect(() => {
    let active = true;
    const checkHealth = async () => {
      try {
        const response = await fetch("/api/health", { cache: "no-store" });
        if (!response.ok) throw new Error("API unavailable");
        const health = (await response.json()) as {
          image_classifier_status?: string;
          image_classifier_prediction_enabled?: string;
        };
        if (active) {
          if (health.image_classifier_prediction_enabled === "true") {
            setApiHealth("ready");
          } else if (health.image_classifier_status === "ready") {
            setApiHealth("auth_not_configured");
          } else {
            setApiHealth("model_not_ready");
          }
        }
      } catch {
        if (active) setApiHealth("unavailable");
      }
    };
    void checkHealth();
    const interval = window.setInterval(checkHealth, 10000);
    return () => {
      active = false;
      window.clearInterval(interval);
    };
  }, []);

  const startCamera = async () => {
    setPredictions([]);
    setApiError("");
    setCameraError("");
    setCameraState("starting");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      streamRef.current = stream;
      if (!videoRef.current) throw new Error("Camera preview is unavailable.");
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
      await startTracking();
      setCameraState("tracking");
    } catch (error) {
      streamRef.current?.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
      stopTracking();
      setCameraState("error");
      setCameraError(
        error instanceof Error
          ? error.message
          : "Camera or hand tracking could not start.",
      );
    }
  };

  const stopLiveRecognition = () => {
    liveRecognitionRef.current = false;
    recognitionSessionRef.current += 1;
    setLiveRecognition(false);
    abortControllerRef.current?.abort();
    abortControllerRef.current = null;
  };

  const startLiveRecognition = () => {
    if (cameraState !== "tracking" || apiHealth !== "ready") return;
    setPredictions([]);
    setApiError("");
    frameNumberRef.current = 0;
    liveRecognitionRef.current = true;
    recognitionSessionRef.current += 1;
    setLiveRecognition(true);
  };

  const stopCamera = () => {
    stopLiveRecognition();
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    stopTracking();
    if (videoRef.current) videoRef.current.srcObject = null;
    setTrackingCount(0);
    setCameraState("idle");
    setPredictions([]);
    setApiError("");
  };

  const requestPrediction = useCallback(async () => {
    if (
      !liveRecognitionRef.current ||
      predictionInFlightRef.current ||
      apiHealth !== "ready" ||
      trackingCount < 1
    ) {
      return;
    }
    const recognitionSession = recognitionSessionRef.current;
    const video = videoRef.current;
    if (
      !video ||
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      setApiError("The camera frame is not ready. Try again in a moment.");
      return;
    }

    predictionInFlightRef.current = true;
    setPredicting(true);
    const capturedAt = new Date().toLocaleTimeString();
    const controller = new AbortController();
    abortControllerRef.current = controller;
    try {
      const capture = document.createElement("canvas");
      const scale = Math.min(1, MAX_CAPTURE_WIDTH / video.videoWidth);
      capture.width = Math.max(1, Math.round(video.videoWidth * scale));
      capture.height = Math.max(1, Math.round(video.videoHeight * scale));
      const context = capture.getContext("2d");
      if (!context) throw new Error("Could not capture the current camera frame.");
      context.drawImage(video, 0, 0, capture.width, capture.height);
      const imageBase64 = await new Promise<string>((resolve, reject) => {
        capture.toBlob((blob) => {
          if (!blob) {
            reject(new Error("Could not encode the captured camera frame."));
            return;
          }
          const reader = new FileReader();
          reader.onload = () => {
            if (typeof reader.result !== "string") {
              reject(new Error("Could not read the captured camera frame."));
              return;
            }
            const separator = reader.result.indexOf(",");
            if (separator < 0) {
              reject(new Error("Could not encode the captured camera frame."));
              return;
            }
            resolve(reader.result.slice(separator + 1));
          };
          reader.onerror = () =>
            reject(new Error("Could not read the captured camera frame."));
          reader.readAsDataURL(blob);
        }, "image/jpeg", 0.7);
      });
      const response = await fetch("/api/predict/image", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        signal: controller.signal,
        body: JSON.stringify({
          preprocessing_schema_version: 1,
          image_base64: imageBase64,
        }),
      });
      const result = (await response.json()) as {
        detail?: string;
        label?: string;
        class_index?: number;
        model_version?: string;
        preprocessing_schema_version?: number;
        status?: string;
      };
      if (!response.ok) {
        throw new Error(result.detail ?? `Request failed (${response.status}).`);
      }
      if (typeof result.label !== "string" || typeof result.class_index !== "number") {
        throw new Error("The inference API returned an invalid response.");
      }
      const prediction: ImagePrediction = result as ImagePrediction;
      if (
        liveRecognitionRef.current &&
        recognitionSession === recognitionSessionRef.current
      ) {
        const frameNumber = ++frameNumberRef.current;
        setPredictions((current) =>
          [
            {
              id: Date.now(),
              frameNumber,
              label: prediction.label,
              capturedAt,
            },
            ...current,
          ].slice(0, MAX_VISIBLE_PREDICTIONS),
        );
        setApiError("");
      }
    } catch (error) {
      if (error instanceof DOMException && error.name === "AbortError") return;
      setApiError(
        error instanceof Error
          ? error.message
          : "Could not classify the latest camera frame.",
      );
    } finally {
      if (abortControllerRef.current === controller) {
        abortControllerRef.current = null;
      }
      predictionInFlightRef.current = false;
      setPredicting(false);
    }
  }, [apiHealth, trackingCount]);

  useEffect(() => {
    if (
      !liveRecognition ||
      cameraState !== "tracking" ||
      apiHealth !== "ready"
    ) {
      return;
    }
    void requestPrediction();
    const interval = window.setInterval(
      () => void requestPrediction(),
      LIVE_PREDICTION_INTERVAL_MS,
    );
    return () => window.clearInterval(interval);
  }, [apiHealth, cameraState, liveRecognition, requestPrediction, trackingCount]);

  useEffect(() => {
    if (!liveRecognition || cameraState !== "tracking" || trackingCount > 0) {
      return;
    }
    const timer = window.setTimeout(() => {
      if (!liveRecognitionRef.current) return;
      recognitionSessionRef.current += 1;
      abortControllerRef.current?.abort();
      abortControllerRef.current = null;
      setPredictions([]);
      setApiError("");
    }, 750);
    return () => window.clearTimeout(timer);
  }, [cameraState, liveRecognition, trackingCount]);

  useEffect(() => {
    return () => {
      liveRecognitionRef.current = false;
      recognitionSessionRef.current += 1;
      abortControllerRef.current?.abort();
    };
  }, []);

  const healthLabel: Record<ApiHealth, string> = {
    checking: "Checking model",
    ready: "Image model ready",
    model_not_ready: "Image model not configured",
    auth_not_configured: "API authentication not configured",
    unavailable: "API offline",
  };

  useEffect(
    () => () => {
      streamRef.current?.getTracks().forEach((track) => track.stop());
    },
    [],
  );

  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="SignLearn home">
          <span className="brand-mark"><Hand size={19} strokeWidth={2.2} /></span>
          <span>signlearn<span className="brand-ai">.ai</span></span>
        </a>
        <div className="topbar-right">
          <span className="prototype-label"><span /> Learning prototype</span>
          <button className="help-button" aria-label="About this prototype" title="This prototype classifies still images into letters and digits; it does not translate signing.">
            <CircleHelp size={19} />
          </button>
          <div className="avatar" aria-label="Local practice session">S</div>
        </div>
      </header>

      <section className="page-heading">
        <div>
          <div className="eyebrow"><span className="eyebrow-line" /> YOUR PRACTICE SPACE</div>
          <h1>Learn at your own pace.</h1>
          <p>Track your hands and explore a prototype that classifies one letter or digit at a time.</p>
        </div>
        <div className="language-chip"><span className="language-dot" /> Indian Sign Language <ArrowDown size={14} /></div>
      </section>

      <section className="workspace-grid" aria-label="Sign language practice">
        <section className="camera-card card">
          <div className="card-heading">
            <div className="heading-icon"><ScanLine size={17} /></div>
            <div>
              <h2>Live hand tracking</h2>
              <p>Landmarks are processed locally in your browser</p>
            </div>
            <div className={`tracking-pill ${cameraState === "tracking" ? "is-live" : ""}`}>
              <span /> {cameraState === "tracking" ? "LIVE" : "STANDBY"}
            </div>
          </div>

          <div className={`camera-stage ${cameraState === "tracking" ? "camera-active" : ""}`}>
            <video ref={videoRef} className="camera-video" muted playsInline aria-label="Live camera preview" />
            <canvas ref={canvasRef} className="landmark-canvas" aria-hidden="true" />
            {cameraState !== "tracking" && (
              <div className="camera-placeholder">
                <div className="camera-placeholder-icon"><Video size={28} /></div>
                <strong>{cameraState === "starting" ? "Preparing your camera…" : "Your camera preview"}</strong>
                <span>
                  {cameraState === "error"
                    ? cameraError
                    : "Start the camera when you’re ready to begin."}
                </span>
                {cameraState === "error" && (
                  <button className="text-button" onClick={() => void startCamera()}>Try again <ArrowRight size={14} /></button>
                )}
              </div>
            )}
            {cameraState === "tracking" && (
              <>
                <div className="camera-overlay-top">
                  <span className="camera-state-label"><span /> Camera active</span>
                  <span className="privacy-label"><LockKeyhole size={12} /> On-device tracking</span>
                </div>
                <div className="camera-overlay-bottom">
                  <span className={`hand-detect ${trackingCount > 0 ? "detected" : ""}`}>
                    <Hand size={14} /> {trackingCount > 0 ? `${trackingCount} hand${trackingCount > 1 ? "s" : ""} detected` : "Show your hands"}
                  </span>
                </div>
              </>
            )}
          </div>

          <div className="camera-controls">
            {cameraState !== "tracking" ? (
              <button
                className="button button-primary"
                onClick={() => void startCamera()}
                disabled={cameraState === "starting"}
              >
                {cameraState === "starting" ? <LoaderCircle className="spin" size={17} /> : <Video size={17} />}
                {cameraState === "starting" ? "Starting camera…" : "Start camera"}
              </button>
            ) : (
              <>
                <button className="button button-quiet" onClick={stopCamera}>
                  <X size={16} /> Stop camera
                </button>
              </>
            )}
            <span className="recording-limit">Single-frame letter/digit demo</span>
          </div>
          <div className="privacy-note"><ShieldCheck size={15} /><span>Hand tracking stays local. After you select <b>Start live predictions</b>, a compressed camera frame is sent about once per second while a hand is detected. Stop live predictions or the camera to stop sending.</span></div>
        </section>

        <aside className="output-card card">
          <div className="output-heading">
            <div>
              <div className="eyebrow output-eyebrow"><Sparkles size={13} /> MODEL OUTPUT</div>
              <h2>Letter or digit</h2>
              <p>Live single-frame guesses, when enabled</p>
            </div>
            <div className="output-symbol"><span>0–9</span><ArrowRight size={14} /><span>A–Z</span></div>
          </div>

          <div className={`model-status status-${apiHealth}`}>
            <span className="status-indicator">{apiHealth === "ready" ? <Check size={12} /> : apiHealth === "checking" ? <LoaderCircle className="spin" size={12} /> : <Info size={12} />}</span>
            <span>{healthLabel[apiHealth]}</span>
          </div>

          <div className={`gloss-output ${predictions.length > 0 ? "has-prediction" : ""}`}>
            {predictions.length > 0 ? (
              <div className="prediction-feed" aria-live="polite">
                <span className="gloss-label">LATEST FRAME GUESSES · NEWEST FIRST</span>
                {predictions.map((frame) => (
                  <div className="prediction-frame" key={frame.id}>
                    <span className="prediction-frame-number">
                      FRAME {String(frame.frameNumber).padStart(2, "0")}
                    </span>
                    <strong>{frame.label}</strong>
                    <time>{frame.capturedAt}</time>
                  </div>
                ))}
              </div>
            ) : (
              <div className="output-empty">
                <div className="empty-speech"><span /><span /><span /></div>
                <strong>Live frame guesses will appear here</strong>
                <span>Start the camera and opt in to live predictions.</span>
              </div>
            )}
          </div>

          <div className="output-actions">
            {cameraState === "tracking" && (
              <button
                className={`button ${liveRecognition ? "button-stop" : "button-primary"} output-submit`}
                onClick={liveRecognition ? stopLiveRecognition : startLiveRecognition}
                disabled={!liveRecognition && apiHealth !== "ready"}
              >
                {liveRecognition
                  ? <X size={16} />
                  : predicting
                    ? <LoaderCircle className="spin" size={16} />
                    : <Sparkles size={16} />}
                {liveRecognition ? "Stop live predictions" : "Start live predictions"}
              </button>
            )}
            {liveRecognition && (
              <div className="live-recognition-status" role="status">
                <span className={predicting ? "live-dot is-processing" : "live-dot"} />
                {trackingCount === 0
                  ? "Waiting for a detected hand… old guesses clear when no hand is visible"
                  : predicting
                    ? "Analyzing a camera frame…"
                    : "Live predictions on · about one frame per second"}
              </div>
            )}
            {predictions.length > 0 && !liveRecognition && (
              <button
                className="text-button clear-predictions"
                onClick={() => setPredictions([])}
              >
                Clear frame results
              </button>
            )}
            {apiError && <div className="api-error" role="alert">{apiError}</div>}
            {apiHealth === "model_not_ready" && (
              <div className="not-ready-note">The still-image classifier is not configured. Configure the trained image checkpoint on the backend to enable letter/digit predictions.</div>
            )}
            {apiHealth === "auth_not_configured" && (
              <div className="not-ready-note">The model is loaded, but the backend API key is not configured for the development proxy.</div>
            )}
            {apiHealth === "unavailable" && (
              <div className="not-ready-note">Inference API is offline. Start the backend to enable live frame predictions.</div>
            )}
          </div>

          <div className="meaning-note"><Info size={14} /><span>Predictions are experimental image classes for isolated handshapes. They are not word meanings, continuous signing recognition, or verified ISL translations.</span></div>
        </aside>
      </section>

      <section className="how-it-works">
        <div className="how-heading"><span>THREE SIMPLE STEPS</span><div /></div>
        <div className="steps-row">
          <div className="step-item"><span className="step-number">01</span><div><strong>Allow camera</strong><span>Tracking runs in your browser</span></div></div>
          <ArrowRight className="step-arrow" size={17} />
          <div className="step-item"><span className="step-number">02</span><div><strong>Opt in to live guesses</strong><span>Sampled frames go to the local API</span></div></div>
          <ArrowRight className="step-arrow" size={17} />
          <div className="step-item"><span className="step-number">03</span><div><strong>Review the label</strong><span>One letter or digit at a time</span></div></div>
        </div>
      </section>

      <footer className="footer">
        <span><span className="footer-mark"><Hand size={13} /></span> SignLearn AI <span className="footer-separator">·</span> An experimental learning tool</span>
        <span><Wifi size={13} /> Camera frames stay local until you opt in to live predictions</span>
      </footer>
    </main>
  );
}

export default App;
