# SignLearn AI frontend prototype

The Vite/React page provides a browser-side webcam preview with MediaPipe hand
landmark overlay. Tracking is local. After the camera is started, the user must
separately choose **Start live predictions** to opt into sending compressed JPEG
frames to the backend. While a hand is detected, the page submits at most one
frame about every 1.2 seconds and displays each returned letter/digit guess as a
separate timestamped result. The latest eight results are kept in page state;
they are not saved to a database and are cleared shortly after the hand tracker
loses the hand. Stop live predictions or stop the camera to stop submissions.
The browser never receives the server API key.

The backend center-crops non-square captures to a square before resizing to the
checkpoint input size. Keep the hand near the center of the camera view.

The model was trained as a static-image classifier on ordered filename splits
without verified signer identities. It predicts one of `0`–`9` or `A`–`Z` per
image; it does not recognize continuous signing, words, or meanings. Accuracy
metrics in the ML README are not signer-independent recognition guarantees.

## Run locally (Windows PowerShell)

Install dependencies:

```powershell
Set-Location frontend
npm install
```

In one PowerShell window, configure the trained image checkpoint and an API key
for the backend. Do not enable synthetic smoke checkpoints for actual use.

```powershell
Set-Location ..
$env:SIGNLEARN_IMAGE_CHECKPOINT_PATH = (Resolve-Path "ml\artifacts\static_classifier\best_checkpoint.pt").Path
$env:SIGNLEARN_API_KEY = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second PowerShell window, set that same key and start Vite:

```powershell
Set-Location C:\Users\Nabhan\signlearn-ai\frontend
$env:SIGNLEARN_API_KEY = "<paste-the-same-key-used-by-the-backend>"
npm run dev -- --port 5173
```

Open the printed local Vite URL (normally `http://127.0.0.1:5173`). The Vite
development proxy adds the key on the server side and forwards `/api/health` and
`/api/predict/image` to the backend. `SIGNLEARN_API_KEY` is read by `vite.config.ts`
but is not included in the built client bundle. Never add it to a `VITE_*` setting.
If the backend has no valid image checkpoint or API key, the page shows that
configuration state and disables prediction.

MediaPipe's WASM runtime and hand-landmarker task model are loaded from their
upstream static hosting when camera tracking starts. Webcam frames are processed
locally and are not sent to those hosts by this application. Use localhost or
HTTPS for camera permissions. Each sampled still is resized to at most 480 pixels
wide and encoded as JPEG before submission, subject to the backend's request and
image-size limits. Frames are never sent until live predictions are explicitly
started.

## Build

```powershell
npm run build
```

The static production build does not provide a secure authenticated backend
proxy. Production deployment needs a trusted same-origin server-side session/BFF
that authenticates users and injects the API credential; do not expose the
development API key to the browser or publish the API key.
