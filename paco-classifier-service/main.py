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

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

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

if not BACKGROUND_MODEL_PATH.exists() or not STAFFLINES_MODEL_PATH.exists():
    # Fail fast at startup (mirrors resolve_medieval_model_paths()'s own
    # RuntimeError-on-missing-weights philosophy) rather than on first request.
    raise RuntimeError(
        f"staffline classifier weights not found in {MODELS_DIR} "
        f"(expected model_0.h5 and model_1.h5) -- set STAFFLINE_MODELS_DIR, "
        f"or check that the paco-classifier submodule is checked out "
        f"(git submodule update --init --recursive)"
    )

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
    """Liveness/readiness signal for Compose's healthcheck and k8s's probes
    (see docker-compose.yml / k8s/paco-classifier-service.yaml). Deliberately
    does NOT exercise recognition_engine.process_image_msae() -- it calls
    tensorflow.keras.models.load_model() fresh on every single request (no
    caching in the vendored submodule), so there is no persistent "model
    warmed" state to probe for; the meaningful, cheap thing to confirm
    instead is that this process is actually serving HTTP (not just that the
    OS has a listener on the port, which a bare TCP probe can't tell apart
    from an app that bound the port and then crashed) and that the weight
    files this process already validated at import time are still there."""
    if not BACKGROUND_MODEL_PATH.exists() or not STAFFLINES_MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="staffline classifier weights missing")
    return {"status": "ok"}


@app.post("/classify")
def classify(image: UploadFile = File(...)):
    # Plain `def`, not `async def`: FastAPI runs a sync endpoint in its
    # threadpool automatically, so the blocking TensorFlow/OpenCV/PNG work
    # below (process_image_msae, _layer_to_rgba_png) doesn't stall the event
    # loop the way it would inside an `async def` with no `await` yield
    # points. image.file is a SpooledTemporaryFile -- .read() is sync here.
    data = image.file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="could not decode uploaded image")

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