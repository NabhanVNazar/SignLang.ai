"""FastAPI application for offline landmark clip inference."""

from __future__ import annotations

import base64
import binascii
import hmac
import math
from collections.abc import Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictFloat,
    StrictInt,
    field_validator,
    model_validator,
)

from backend.app.config import Settings
from backend.app.image_inference import (
    MAX_IMAGE_BYTES,
    ImageClassifierInferenceService,
    ImageInferenceNotReadyError,
)
from backend.app.inference import InferenceNotReadyError, InferenceService

Coordinate = StrictFloat | StrictInt
HandLandmarkArray = list[list[Coordinate]] | None


class LandmarkFrameRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp_ms: Annotated[StrictInt, Field(ge=0)]
    left_hand: HandLandmarkArray
    right_hand: HandLandmarkArray

    @field_validator("left_hand", "right_hand")
    @classmethod
    def validate_hand(
        cls, value: HandLandmarkArray
    ) -> HandLandmarkArray:
        if value is None:
            return None
        if len(value) != 21:
            raise ValueError("an observed hand must contain exactly 21 landmarks")
        for point_index, point in enumerate(value):
            if len(point) != 3:
                raise ValueError(
                    f"landmark {point_index} must contain exactly x, y, and z"
                )
            try:
                coordinates = [float(coordinate) for coordinate in point]
            except OverflowError as exc:
                raise ValueError(
                    f"landmark {point_index} coordinates are outside float32 range"
                ) from exc
            if any(
                not math.isfinite(coordinate) or abs(coordinate) > 3.4028235e38
                for coordinate in coordinates
            ):
                raise ValueError(f"landmark {point_index} coordinates must be finite")
        return value


class ClipPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing_schema_version: Annotated[StrictInt, Field(ge=1)]
    frames: list[LandmarkFrameRequest] = Field(min_length=1)

    @model_validator(mode="after")
    def timestamps_increase(self) -> ClipPredictionRequest:
        timestamps = [frame.timestamp_ms for frame in self.frames]
        if any(
            current <= previous
            for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise ValueError("frame timestamps must be strictly increasing")
        if not any(
            frame.left_hand is not None or frame.right_hand is not None
            for frame in self.frames
        ):
            raise ValueError("clip must contain at least one observed hand")
        return self


class ClipPredictionResponse(BaseModel):
    glosses: list[str]
    sequence: str
    model_version: str
    preprocessing_schema_version: int
    status: str


class ImagePredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preprocessing_schema_version: Annotated[StrictInt, Field(ge=1)]
    image_base64: Annotated[
        str, Field(min_length=1, max_length=((MAX_IMAGE_BYTES + 2) // 3) * 4)
    ]


class ImagePredictionResponse(BaseModel):
    label: str
    class_index: int
    model_version: str
    preprocessing_schema_version: int
    status: str


class RequestBodyLimitMiddleware:
    """Read request bodies only up to the configured per-request bound."""

    def __init__(self, app: Callable[..., Any], max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope["type"] != "http" or scope.get("method") != "POST":
            await self.app(scope, receive, send)
            return

        content_length = next(
            (
                value.decode("latin-1")
                for name, value in scope.get("headers", [])
                if name.lower() == b"content-length"
            ),
            None,
        )
        if content_length is not None:
            try:
                if int(content_length) > self.max_bytes:
                    await self._send_too_large(send)
                    return
            except ValueError:
                await self._send_too_large(send)
                return

        chunks: list[bytes] = []
        total_bytes = 0
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            chunk = message.get("body", b"")
            total_bytes += len(chunk)
            if total_bytes > self.max_bytes:
                await self._send_too_large(send)
                return
            chunks.append(chunk)
            if not message.get("more_body", False):
                break

        body = b"".join(chunks)
        replayed = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal replayed
            if not replayed:
                replayed = True
                return {"type": "http.request", "body": body, "more_body": False}
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _send_too_large(send: Callable[..., Any]) -> None:
        body = b'{"detail":"Request body exceeds the configured limit."}'
        await send(
            {
                "type": "http.response.start",
                "status": HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send({"type": "http.response.body", "body": body})


def create_app(
    settings: Settings | None = None,
    inference_service: InferenceService | None = None,
    image_inference_service: ImageClassifierInferenceService | None = None,
) -> FastAPI:
    app_settings = settings or Settings.from_env()
    service = inference_service or InferenceService(
        app_settings.checkpoint_path,
        allow_synthetic_checkpoint=app_settings.allow_synthetic_checkpoint,
    )
    image_service = image_inference_service or ImageClassifierInferenceService(
        app_settings.image_checkpoint_path
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        service.load()
        image_service.load()
        yield

    app = FastAPI(
        title="SignLearn AI Inference API",
        description="Offline prototype API for landmark clip gloss-sequence inference.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = app_settings
    app.state.inference_service = service
    app.add_middleware(
        RequestBodyLimitMiddleware, max_bytes=app_settings.max_request_bytes
    )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        # Do not echo landmark payloads in validation responses.
        del request, exc
        return JSONResponse(
            status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
            content={"detail": "Invalid inference request."},
        )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ready" if service.ready else "model_not_ready",
            "service": "clip_inference",
            "image_classifier_status": (
                "ready" if image_service.ready else "model_not_ready"
            ),
            "image_classifier_prediction_enabled": str(
                image_service.ready and app_settings.api_key is not None
            ).lower(),
        }

    def authorize(authorization: str | None) -> None:
        expected_key = app_settings.api_key
        if expected_key is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Inference API authentication is not configured.",
            )
        supplied_key = (
            authorization.removeprefix("Bearer ").strip()
            if authorization is not None and authorization.startswith("Bearer ")
            else ""
        )
        if not hmac.compare_digest(supplied_key, expected_key):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid bearer token.",
                headers={"WWW-Authenticate": "Bearer"},
            )

    @app.post(
        "/predict/clip",
        response_model=ClipPredictionResponse,
        responses={
            401: {"description": "Missing or invalid API key"},
            413: {"description": "Request or clip exceeds configured limits"},
            503: {"description": "Model or API authentication is not configured"},
        },
    )
    def predict_clip(
        payload: ClipPredictionRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> ClipPredictionResponse:
        authorize(authorization)
        if len(payload.frames) > app_settings.max_frames:
            raise HTTPException(
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                detail="Clip exceeds the configured frame limit.",
            )
        if payload.preprocessing_schema_version != 1:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Unsupported preprocessing schema version.",
            )
        if not service.ready:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No valid inference model is configured.",
            )
        try:
            glosses, sequence = service.predict(
                [(frame.left_hand, frame.right_hand) for frame in payload.frames]
            )
        except InferenceNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="No valid inference model is configured.",
            ) from exc
        return ClipPredictionResponse(
            glosses=glosses,
            sequence=sequence,
            model_version=service.checkpoint_id or "unknown",
            preprocessing_schema_version=1,
            status="prototype_prediction",
        )

    @app.post(
        "/predict/image",
        response_model=ImagePredictionResponse,
        responses={
            401: {"description": "Missing or invalid API key"},
            413: {"description": "Image or request exceeds configured limits"},
            503: {"description": "Model or API authentication is not configured"},
        },
    )
    def predict_image(
        payload: ImagePredictionRequest,
        authorization: Annotated[str | None, Header()] = None,
    ) -> ImagePredictionResponse:
        authorize(authorization)
        if payload.preprocessing_schema_version != 1:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Unsupported preprocessing schema version.",
            )
        try:
            image_bytes = base64.b64decode(payload.image_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Image must be valid base64-encoded image data.",
            ) from exc
        if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
            raise HTTPException(
                status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                detail="Image exceeds the configured size limit.",
            )
        if not image_service.ready:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="No valid image-classifier model is configured.",
            )
        try:
            label, class_index = image_service.predict(image_bytes)
        except ImageInferenceNotReadyError as exc:
            raise HTTPException(
                status_code=HTTPStatus.SERVICE_UNAVAILABLE,
                detail="No valid image-classifier model is configured.",
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=HTTPStatus.UNPROCESSABLE_ENTITY,
                detail="Image data is invalid or uses an unsupported format.",
            ) from exc
        return ImagePredictionResponse(
            label=label,
            class_index=class_index,
            model_version=image_service.checkpoint_id or "unknown",
            preprocessing_schema_version=1,
            status="prototype_image_classification",
        )

    return app


app = create_app()
