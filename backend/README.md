# SignLearn AI inference API

This service exposes two separate offline research-prototype endpoints:

- `POST /predict/clip` accepts hand-landmark sequences for the BiLSTM/CTC model.
- `POST /predict/image` accepts one base64-encoded JPEG, PNG, or WebP still image
  for the static letter/digit classifier.

The clip model and image model have different input contracts and checkpoints.
The image model predicts one class per still image. Neither endpoint translates
arbitrary signing or returns a calibrated confidence score.

## Install and configure (Windows PowerShell)

From the repository root, install the ML dependencies and backend API packages:

```powershell
python -m pip install -r ml\requirements.txt
python -m pip install -r backend\requirement.txt
```

Set the server-side image-model location and a strong API key in the same
PowerShell session used to run Uvicorn. The clip model path is optional:

```powershell
$env:SIGNLEARN_IMAGE_CHECKPOINT_PATH = (Resolve-Path "ml\artifacts\static_classifier\best_checkpoint.pt").Path
$env:SIGNLEARN_API_KEY = [guid]::NewGuid().ToString("N") + [guid]::NewGuid().ToString("N")
$env:SIGNLEARN_MAX_FRAMES = "300"
$env:SIGNLEARN_MAX_REQUEST_BYTES = "2097152"
$env:SIGNLEARN_ALLOW_SYNTHETIC_CHECKPOINT = "false"
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

`SIGNLEARN_CHECKPOINT_PATH` must point to a checkpoint produced by the real
training pipeline. The checkpoint is loaded once when the application starts,
using PyTorch `weights_only=True`. Metadata, architecture, blank class,
preprocessing schema, and saved vocabulary are validated before the service
becomes ready. CPU is used unless CUDA is available. The response exposes only
a SHA-256 checkpoint identifier, not the configured path.

`SIGNLEARN_IMAGE_CHECKPOINT_PATH` is optional and points to the separate
`StaticSignCNN` image checkpoint. It is also loaded once with
`weights_only=True`; class indices, dataset fingerprint, model configuration,
and image preprocessing metadata are validated at startup. The image route is
ready only when a valid image checkpoint is loaded. Non-square image requests
are center-cropped to square before resizing; square dataset images retain their
original framing. Keep the hand near the center of the camera frame.

Synthetic CTC smoke checkpoints are rejected by default. To use one in an
isolated development smoke test only, set
`SIGNLEARN_ALLOW_SYNTHETIC_CHECKPOINT=true`. Never enable that setting in a
deployed environment. Configure only trained image checkpoints for the static
image endpoint.

Other environment settings:

| Variable | Default | Limits |
| --- | --- | --- |
| `SIGNLEARN_CHECKPOINT_PATH` | unset | Optional server-side path to trained landmark-sequence checkpoint |
| `SIGNLEARN_MAX_FRAMES` | `300` | 1 through 3000 frames per clip |
| `SIGNLEARN_MAX_REQUEST_BYTES` | `2097152` | 1 byte through 16 MiB per POST |
| `SIGNLEARN_ALLOW_SYNTHETIC_CHECKPOINT` | `false` | `true`/`false` only |
| `SIGNLEARN_IMAGE_CHECKPOINT_PATH` | unset | Server-side path to trained static-image classifier checkpoint |
| `SIGNLEARN_API_KEY` | unset | Required to make predictions; at least 32 characters |

No Supabase key is needed for this inference endpoint. The app does not load
`.env` or access database credentials.

## Test locally

With the API running, use a small JSON request. Every frame must include both
`left_hand` and `right_hand`; each is either `null` or exactly 21 `[x, y, z]`
points. At least one hand must be observed in the clip. Points must be finite
and representable in float32. Frame timestamps are non-negative and strictly
increasing.

```powershell
$points = @()
for ($i = 0; $i -lt 21; $i++) {
    $point = @(([double]$i / 21.0), ([double]$i / 42.0), 0.0)
    $points += ,$point
}
$body = @{
    preprocessing_schema_version = 1
    frames = @(
        @{ timestamp_ms = 0; left_hand = $points; right_hand = $null },
        @{ timestamp_ms = 40; left_hand = $points; right_hand = $null }
    )
} | ConvertTo-Json -Depth 8
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict/clip" `
  -Headers @{ Authorization = "Bearer $env:SIGNLEARN_API_KEY" } `
  -ContentType "application/json" `
  -Body $body
```

The clip response contains ordered `glosses`, their space-joined `sequence`,
`model_version` (checkpoint SHA-256), `preprocessing_schema_version`, and a
`prototype_prediction` status. It has no confidence field. `GET /health`
reports clip and image model readiness separately; missing or invalid model
configuration makes predictions return HTTP 503 rather than fabricating output.

### Single-image classifier

Set `SIGNLEARN_IMAGE_CHECKPOINT_PATH` to the actual trained image checkpoint.
Each request must contain `preprocessing_schema_version: 1` and an
`image_base64` field containing the raw base64 encoding of a JPEG, PNG, or WebP.
`GET /health` reports `image_classifier_status` and whether image prediction is
enabled (a valid model and API key are both required). Image responses contain
the predicted saved class label, class index, checkpoint identifier, and schema
version. No softmax score is exposed as confidence.

```powershell
$bytes = [System.IO.File]::ReadAllBytes("C:\path\to\hand-image.jpg")
$encoded = [Convert]::ToBase64String($bytes)
$body = @{
    preprocessing_schema_version = 1
    image_base64 = $encoded
} | ConvertTo-Json -Compress
Invoke-RestMethod `
  -Method Post `
  -Uri "http://127.0.0.1:8000/predict/image" `
  -Headers @{ Authorization = "Bearer $env:SIGNLEARN_API_KEY" } `
  -ContentType "application/json" `
  -Body $body
```

## Deployment and frontend security

The server binds to loopback by default. Do not expose it directly to the
internet: the API key is a service credential and must never be embedded in a
browser frontend. A deployed frontend needs a trusted backend session/auth
layer that authorizes each user and forwards requests server-side, plus TLS,
rate limits, request timeouts, access controls, and monitoring. Keep private
landmark recordings out of logs. This service does not add authentication integration with Supabase. The
development frontend uses a local Vite proxy to inject the API key server-side;
production needs a trusted same-origin session/auth layer.

Predictions are greedy CTC-decoded gloss hypotheses, not natural-language
translations or calibrated confidence estimates. Recognition quality has not
been established; evaluate only with the held-out signer-independent test
procedure documented in [the ML README](../ml/README.md).
