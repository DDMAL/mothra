"""
paco-classifier-service — thin FastAPI wrapper around the Paco_classifier
submodule's recognition_engine.process_image_msae(), exposed over HTTP so
the landing-page worker (PyTorch/ultralytics) never has to import a
TensorFlow/Keras dependency into its own process. Mirrors text-service's
shape (a standalone service wrapping a sibling submodule) rather than
staff-finding's in-process pip-package pattern.

POST /classify takes one page image and returns two RGBA PNGs (background,
stafflines), alpha=0 outside each layer's own mask — the exact derivation
below is ported from Paco_classifier's own Classifiers/run_classifier.py
(the standalone, rodan-independent CLI script — NOT evaluation.py's
evaluateRodan(), which is the old Rodan-job wrapper and pulls in a Rodan-
specific ConfigParser.loadConfig() we don't want anywhere near this
service).
"""
import base64
import os
import sys
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from tensorflow.keras.models import load_model

PACO_DIR = Path(__file__).resolve().parent.parent / "paco-classifier"
sys.path.insert(0, str(PACO_DIR))

from Paco_classifier import recognition_engine  # noqa: E402

# No CORS middleware: this service has no authentication of its own and is
# only ever called server-to-server, from landing-page's backend/worker over
# the Compose/k8s internal network (see paco_api.py) -- never directly from a
# browser. Adding a permissive CORS policy here would let any web page that
# can reach this port (e.g. if it were ever accidentally published, see
# docker-compose.yml's `expose` vs `ports` note) post arbitrary images to it.
app = FastAPI()

def _resolve_staffline_models_dir() -> Path:
    """Priority order mirrors medieval_models.py's own
    resolve_medieval_model_paths(): env override -> the paco-classifier
    submodule's own models_v4/ dir. The weights live INSIDE the submodule
    (checked into DDMAL/Paco_classifier itself, branch
    gianna/calvo-training-script) -- not bundled separately in landing-page's
    assets dir, so no Docker COPY step is needed for them either: the
    Dockerfile's `COPY paco-classifier/ ./paco-classifier/` already brings
    models_v4/ along with the rest of the submodule tree.
    """
    env_override = os.environ.get("STAFFLINE_MODELS_DIR")
    if env_override:
        return Path(env_override)
    return PACO_DIR / "models_v4"


MODELS_DIR = _resolve_staffline_models_dir()
BACKGROUND_MODEL_PATH = MODELS_DIR / "model_0.h5"
STAFFLINES_MODEL_PATH = MODELS_DIR / "model_1.h5"
PATCH_HEIGHT = 256
PATCH_WIDTH = 256

BACKGROUND_LABEL = 0
STAFFLINES_LABEL = 1

# This endpoint has no auth of its own (see the CORS comment on `app` above)
# and reads the full upload into memory before decoding, so an unbounded
# upload -- or a small, highly-compressed one that decodes to a huge pixel
# grid -- can exhaust the container's memory limit before decoding even
# fails. Bound both: 50 MiB comfortably covers a single high-res page scan,
# and 6000x6000 covers real manuscript-page resolutions while still keeping
# the classifier's several in-flight copies (input image, label map, two
# RGBA PNG outputs) well under the container's 4 GiB limit.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
MAX_DECODED_PIXELS = 6000 * 6000

if not BACKGROUND_MODEL_PATH.exists() or not STAFFLINES_MODEL_PATH.exists():
    # Fail fast at startup (mirrors resolve_medieval_model_paths()'s own
    # RuntimeError-on-missing-weights philosophy) rather than on first request.
    raise RuntimeError(
        f"staffline classifier weights not found in {MODELS_DIR} "
        f"(expected model_0.h5 and model_1.h5) -- set STAFFLINE_MODELS_DIR, "
        f"or check that the paco-classifier submodule is checked out "
        f"(git submodule update --init --recursive)"
    )


def _validate_classifier_models() -> str | None:
    """One-time, startup-only load of both weight files as real Keras
    models (mirrors recognition_engine.process_image_msae()'s own bare
    load_model(path) call, so this fails on exactly the same conditions
    inference would). The existence check above only confirms the files
    are present, not that they're valid/loadable/version-compatible --
    a corrupt or incompatible .h5 passes that check but fails every
    /classify call. Returns the error string on failure, or None if both
    models loaded successfully. Deliberately NOT re-run per request or
    per health poll: load_model() is exactly as expensive as inference
    (same reason process_image_msae() itself reloads fresh every call
    instead of caching -- see health()'s docstring), so this result is
    computed once here and cached for /ready to report cheaply.
    """
    try:
        for path in (BACKGROUND_MODEL_PATH, STAFFLINES_MODEL_PATH):
            load_model(str(path))
    except Exception as exc:  # noqa: BLE001 - any load failure means "not ready"
        return str(exc)
    return None


_classifier_ready_error = _validate_classifier_models()

def _layer_to_rgba_png(image: np.ndarray, label_map: np.ndarray, id_label: int) -> bytes:
    """One label's pixels, alpha-masked — ported verbatim from
    Paco_classifier/Classifiers/run_classifier.py's per-label loop (the
    rodan-independent CLI; same algorithm as evaluation.py's Rodan-coupled
    evaluateRodan(), but this is the version with no Rodan dependency to
    accidentally inherit). `image` is BGR (cv2 convention); masked-out
    pixels are forced to pure white on ALL THREE colour channels (not just
    alpha=0) — the Mothra side (tasks_predict.py) relies on that when it
    later drops the alpha channel to feed this into the stave YOLO model.
    """
    label_range = np.array(id_label, dtype=np.uint8)
    mask = cv2.inRange(label_map, label_range, label_range)
    masked = cv2.bitwise_and(image, image, mask=mask)
    masked[mask == 0] = (255, 255, 255)
    alpha_channel = np.ones(mask.shape, dtype=mask.dtype) * 255
    alpha_channel[mask == 0] = 0
    b, g, r = cv2.split(masked)
    rgba = cv2.merge((b, g, r, alpha_channel))
    ok, buf = cv2.imencode(".png", rgba)
    if not ok:
        raise RuntimeError("PNG encode failed")
    return buf.tobytes()

@app.get("/health")
def health():
    """Liveness signal only (see docker-compose.yml / k8s's livenessProbe).
    Deliberately does NOT exercise recognition_engine.process_image_msae()
    -- it calls tensorflow.keras.models.load_model() fresh on every single
    request (no caching in the vendored submodule), so there is no
    persistent "model warmed" state to probe for on every poll; the
    meaningful, cheap thing to confirm here is that this process is
    actually serving HTTP (not just that the OS has a listener on the
    port, which a bare TCP probe can't tell apart from an app that bound
    the port and then crashed) and that the weight files this process
    already validated at import time are still there. It does NOT confirm
    those files are valid, loadable models -- see /ready for that."""
    if not BACKGROUND_MODEL_PATH.exists() or not STAFFLINES_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="staffline classifier weights missing")
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness signal: the weight files must exist AND have actually
    loaded as valid Keras models at startup (see _validate_classifier_models()
    above). Point Compose's healthcheck and k8s's readinessProbe at this
    endpoint, not /health -- a corrupt or version-incompatible .h5 passes
    /health (files merely exist) but fails every real /classify call,
    silently falling back to raw-page stave detection on the Mothra side
    (tasks_predict.py) with nothing surfaced here. This is a cached,
    startup-only result, not a fresh model load per poll -- see that
    function's docstring for why."""
    if _classifier_ready_error is not None:
        raise HTTPException(
            status_code=503,
            detail=f"staffline classifier models failed to load: {_classifier_ready_error}",
        )
    return {"status": "ready"}


@app.post("/classify")
def classify(image: Annotated[UploadFile, File()]):
    # Plain `def`, not `async def`: FastAPI runs a sync endpoint in its
    # threadpool automatically, so the blocking TensorFlow/OpenCV/PNG work
    # below (process_image_msae, _layer_to_rgba_png) doesn't stall the event
    # loop the way it would inside an `async def` with no `await` yield
    # points. image.file is a SpooledTemporaryFile -- .read() is sync here.
    #
    # Read one byte past the cap so an exactly-at-the-limit upload doesn't
    # get misreported, without ever buffering more than MAX_UPLOAD_BYTES+1
    # bytes for an oversized one.
    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"uploaded image exceeds the {MAX_UPLOAD_BYTES}-byte limit",
        )
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode uploaded image")

    height, width = img.shape[:2]
    if height * width > MAX_DECODED_PIXELS:
        raise HTTPException(
            status_code=413,
            detail=f"decoded image {width}x{height} exceeds the {MAX_DECODED_PIXELS}-pixel limit",
        )

    label_map = recognition_engine.process_image_msae(
        img,
        [str(BACKGROUND_MODEL_PATH), str(STAFFLINES_MODEL_PATH)],
        PATCH_HEIGHT, PATCH_WIDTH,
        mode="logical",
    )
    # process_image_msae restores full input resolution internally when
    # resize_ratio/max_dimension are used (neither is passed here, so this
    # should always hold) — re-checked because the Mothra side needs the
    # two models' box coordinates to share one frame.
    if label_map.shape[:2] != img.shape[:2]:
        raise HTTPException(
            status_code=500,
            detail=f"classifier output shape {label_map.shape[:2]} != input {img.shape[:2]}",
        )

    background_png = _layer_to_rgba_png(img, label_map, BACKGROUND_LABEL)
    stafflines_png = _layer_to_rgba_png(img, label_map, STAFFLINES_LABEL)
    return {
        "background_png_base64": base64.b64encode(background_png).decode(),
        "stafflines_png_base64": base64.b64encode(stafflines_png).decode(),
    }