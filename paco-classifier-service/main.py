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

The response streams as SSE `data: {...}\n\n` lines (mothra#247 follow-up)
rather than one blocking JSON body: zero or more `{"type": "progress", ...}`
events as the sliding-window TF pass advances (recognition_engine's own
"row N / total" print, now also handed to a progress_callback), then
exactly one terminal `{"type": "result", ...}` or `{"type": "error", ...}`
event. This only applies once the request has actually started streaming
(HTTP 200) — a request that fails validation before classification even
starts (bad upload, oversized image) still gets a normal HTTP 4xx JSON
error body, since nothing has been streamed yet at that point.
"""
import asyncio
import base64
import json
import os
import sys
import threading
from pathlib import Path
from typing import Annotated, AsyncGenerator

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse
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

# How often /classify polls the ASGI connection for a client disconnect
# while the background classification thread is running -- see classify()'s
# own comment for why this matters at all.
_DISCONNECT_POLL_INTERVAL_S = 0.25

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
async def classify(request: Request, image: Annotated[UploadFile, File()]):
    # `async def`, not a plain `def`, specifically so this can poll
    # request.is_disconnected() concurrently with the blocking TensorFlow
    # work below -- see the cancellation comment further down for why.
    # UploadFile.read() is itself async-safe (Starlette runs the
    # underlying SpooledTemporaryFile.read in its threadpool for us), so
    # this doesn't block the event loop the way a bare image.file.read()
    # would from inside an async def.
    #
    # Read one byte past the cap so an exactly-at-the-limit upload doesn't
    # get misreported, without ever buffering more than MAX_UPLOAD_BYTES+1
    # bytes for an oversized one.
    data = await image.read(MAX_UPLOAD_BYTES + 1)
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

    # Run the actual classification on a background thread so this
    # coroutine is free to poll for a client disconnect concurrently (see
    # below) -- a plain `await run_in_threadpool(...)` would block this
    # coroutine until the thread finishes, defeating the whole point.
    #
    # Mothra's caller (tasks_predict.py) aborts its HTTP connection the
    # instant a predict job is cancelled (paco_api.py's
    # abort_classify_request()), but until now that only freed up
    # Mothra's OWN worker thread -- this process kept running the
    # abandoned TensorFlow inference to completion regardless, since
    # nothing here ever checked whether the client was still listening.
    # cancel_event is polled by process_image_msae() once per patch (see
    # its should_cancel param), so setting it here now actually stops the
    # inference within about one patch-predict's worth of time instead of
    # letting the whole page finish for a result nobody will ever read.
    cancel_event = threading.Event()
    outcome: dict = {}
    # mothra#247 follow-up: process_image_msae() calls this once per
    # sliding-window row from the BACKGROUND thread -- it only ever writes
    # two ints, and simple dict-item assignment is already atomic under the
    # GIL, but the lock keeps this correct even if the shape of `progress`
    # ever grows beyond that.
    progress_lock = threading.Lock()
    progress: dict = {"row": 0, "total": 0}

    def _on_progress(row: int, total: int) -> None:
        with progress_lock:
            progress["row"] = row
            progress["total"] = total

    def _run():
        try:
            outcome["label_map"] = recognition_engine.process_image_msae(
                img,
                [str(BACKGROUND_MODEL_PATH), str(STAFFLINES_MODEL_PATH)],
                PATCH_HEIGHT, PATCH_WIDTH,
                mode="logical",
                should_cancel=cancel_event.is_set,
                progress_callback=_on_progress,
            )
        except recognition_engine.ClassificationCancelled:
            outcome["cancelled"] = True
        except Exception as exc:  # noqa: BLE001 - reported back to the request handler, not raised in this thread
            outcome["error"] = exc

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()

    async def _stream() -> AsyncGenerator[str, None]:
        # mothra#247 follow-up: streams progress as it happens instead of
        # only returning a final JSON body once the whole page is done --
        # see this module's docstring for the event shapes. Cancellation
        # behavior (poll request.is_disconnected(), set cancel_event) is
        # otherwise unchanged from before this endpoint streamed anything.
        last_sent = None
        try:
            while thread.is_alive():
                if await request.is_disconnected():
                    cancel_event.set()
                    break
                with progress_lock:
                    row, total = progress["row"], progress["total"]
                if total and (row, total) != last_sent:
                    last_sent = (row, total)
                    yield f"data: {json.dumps({'type': 'progress', 'row': row, 'total': total})}\n\n"
                await asyncio.sleep(_DISCONNECT_POLL_INTERVAL_S)
        finally:
            # CodeRabbit: Starlette can cancel THIS generator directly at the
            # ASGI level (its own disconnect handling), independent of --
            # and possibly before -- our own is_disconnected() poll above
            # ever runs again. If that happens while suspended at the
            # `asyncio.sleep()` above, execution never returns to set
            # cancel_event itself, and the background TensorFlow thread
            # would otherwise keep running the full page to completion for
            # a request nobody is listening to anymore (exactly the kind of
            # wasted, compounding memory pressure implicated in mothra#212's
            # "Error in input stream" OOM investigation). This runs no
            # matter how/why the generator is exiting, so cancel_event ends
            # up set either way -- a harmless no-op on the normal
            # thread-already-finished path, since process_image_msae's
            # should_cancel is only ever polled from INSIDE that thread.
            cancel_event.set()
        # Whether the loop above exited because the thread finished on its
        # own or because we just set cancel_event, the thread may still be
        # running for up to one more patch -- wait for it (off the event
        # loop) before trusting `outcome`. Not reached if the generator
        # itself was torn down by cancellation (see the finally above) --
        # nothing downstream would consume it in that case anyway, and
        # `thread` is a daemon so it can't block shutdown regardless.
        await asyncio.to_thread(thread.join)

        if outcome.get("cancelled"):
            # The client that would have read this response is already gone
            # (that's what set cancel_event above) -- nothing reads this
            # event, but emit it anyway rather than leaving the stream to
            # end on nothing.
            yield f"data: {json.dumps({'type': 'error', 'detail': 'client disconnected; classification cancelled'})}\n\n"
            return
        if "error" in outcome:
            yield f"data: {json.dumps({'type': 'error', 'detail': str(outcome['error'])})}\n\n"
            return

        label_map = outcome["label_map"]
        # process_image_msae restores full input resolution internally when
        # resize_ratio/max_dimension are used (neither is passed here, so
        # this should always hold) — re-checked because the Mothra side
        # needs the two models' box coordinates to share one frame.
        if label_map.shape[:2] != img.shape[:2]:
            yield (
                f"data: {json.dumps({'type': 'error', 'detail': f'classifier output shape {label_map.shape[:2]} != input {img.shape[:2]}'})}"
                "\n\n"
            )
            return

        background_png = _layer_to_rgba_png(img, label_map, BACKGROUND_LABEL)
        stafflines_png = _layer_to_rgba_png(img, label_map, STAFFLINES_LABEL)
        yield "data: " + json.dumps({
            "type": "result",
            "background_png_base64": base64.b64encode(background_png).decode(),
            "stafflines_png_base64": base64.b64encode(stafflines_png).decode(),
        }) + "\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")