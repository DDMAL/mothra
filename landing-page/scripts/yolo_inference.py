import json
import uuid as _uuid
from typing import Optional

from medieval_models import resolve_medieval_model_paths, TEXT_MUSIC_CLASS_MAP, STAVE_CLASS_MAP
from models_api import get_model_file_path

CATEGORY_TO_SLOT = {"text": 0, "music": 1, "staves": 2}


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def resolve_device(requested: Optional[str]) -> str:
    """Pick the effective inference device: use the GPU whenever one exists,
    otherwise CPU (per the deployment's 'use GPU if present, else CPU' rule).

    An explicit CUDA index (e.g. "cuda:1" / "1") is honored when a GPU is
    available; anything else (including "auto"/"cpu"/None) resolves to "cuda"
    when available. With no GPU (CPU nodes, local compose) it always returns
    "cpu", so CPU deployments keep working unchanged.
    """
    if _cuda_available():
        req = (requested or "").strip().lower()
        if req.startswith("cuda") or req.isdigit():
            return requested
        return "cuda"
    return "cpu"

def _append_boxes(lines, inference, cls_map, threshold):
    if inference.boxes is None or not len(inference.boxes):
        return
    for box in inference.boxes:
        if float(box.conf[0]) < threshold:
            continue
        raw_cls = int(box.cls[0])
        cls = cls_map.get(raw_cls) if cls_map is not None else raw_cls
        if cls is None:
            continue
        x, y, w, h = box.xywhn[0].tolist()
        lines.append(f"{cls} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

class YoloModelSet:
    """A resolved YOLO model (or medieval pair) ready to run inference on
    one image at a time. Built by resolve_yolo_models(); shared by
    run_predict (single/grid) and batch_api.py's run_text_batch."""

    def __init__(self, medieval_models, class_maps, single_model, custom_cls_map,
                 tm_threshold, tm_device, st_threshold, st_device,
                 confidence_threshold, device,
                 stored_model_id, model_label, model_hash):
        self.medieval_models = medieval_models
        self.class_maps = class_maps
        self.single_model = single_model
        self.custom_cls_map = custom_cls_map
        self.tm_threshold = tm_threshold
        self.tm_device = tm_device
        self.st_threshold = st_threshold
        self.st_device = st_device
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.stored_model_id = stored_model_id
        self.model_label = model_label
        self.model_hash = model_hash

    def infer(self, img_arr) -> str:
        lines = []
        if self.medieval_models is not None:
            tm_model, st_model = self.medieval_models
            tm_map, st_map = self.class_maps
            _append_boxes(lines, tm_model(img_arr, device=self.tm_device, verbose=False)[0], tm_map, self.tm_threshold)
            _append_boxes(lines, st_model(img_arr, device=self.st_device, verbose=False)[0], st_map, self.st_threshold)
        else:
            _append_boxes(lines, self.single_model(img_arr, device=self.device, verbose=False)[0], self.custom_cls_map, self.confidence_threshold)
        return "\n".join(lines)

def resolve_yolo_models(
        cur, project_id: int, model_preset: str, model_id: Optional[str],
        confidence_threshold: float, device: str,
        text_music_confidence_threshold: Optional[float], text_music_device: Optional[str],
        stave_confidence_threshold: Optional[float], stave_device: Optional[str],
) -> tuple["YoloModelSet", list[str]]:
    """Loads the requested YOLO model(s), returns a ready-to-use
    YoloModelSet plus human-readable log lines describing what loaded.
    Raises RuntimeError (medieval preset unavailable) or ValueError (custom
    model not found) - callers decide how to surface that as an SSE error.
    """
    # Raised as RuntimeError, not left as the bare ModuleNotFoundError: callers
    # (tasks_predict, batch_api) catch RuntimeError/ValueError and surface the
    # text as an SSE error, whereas an ImportError escapes those handlers and
    # kills the job with an opaque traceback instead of an actionable message.
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "YOLO inference needs the 'ultralytics' package, which isn't installed "
            "in this environment. Either install it "
            "(pip install -r landing-page/scripts/requirements.txt), or skip the "
            "predict step: set MOTHRA_SKIP_YOLO=1 in landing-page/scripts/.env and "
            "VITE_SKIP_PREDICT=1 in landing-page/.env.local, then restart the "
            "backend, the Celery worker, and the Vite dev server."
        ) from exc

    tm_threshold = text_music_confidence_threshold if text_music_confidence_threshold is not None else confidence_threshold
    st_threshold = stave_confidence_threshold if stave_confidence_threshold is not None else confidence_threshold
    # Resolve to the GPU when one is present, else CPU (guards an explicit
    # cuda request on a CPU-only node from crashing ultralytics/torch).
    device = resolve_device(device)
    tm_device = resolve_device(text_music_device or device)
    st_device = resolve_device(stave_device or device)

    if model_preset == "medieval":
        tm_path, st_path = resolve_medieval_model_paths()
        model_set = YoloModelSet(
            medieval_models=(YOLO(tm_path), YOLO(st_path)),
            class_maps=(TEXT_MUSIC_CLASS_MAP, STAVE_CLASS_MAP),
            single_model=None, custom_cls_map=None,
            tm_threshold=tm_threshold, tm_device=tm_device,
            st_threshold=st_threshold, st_device=st_device,
            confidence_threshold=confidence_threshold, device=device,
            stored_model_id=model_preset,
            model_label="medieval manuscripts (text_music_detector_fulldata.pt + stave_detector_fulldata.pt)",
            model_hash=None,
        )
        return model_set, ["medieval manuscripts preset: loaded text/music + stave detectors"]
    
    model_row = get_model_file_path(cur, project_id, model_id, "yolo")
    if not model_row:
        raise ValueError("Model file not found")
    file_path, model_name, class_map_json, file_hash = model_row
    custom_cls_map = None
    if class_map_json:
        raw_map = json.loads(class_map_json)
        custom_cls_map = {int(k): CATEGORY_TO_SLOT[v] for k, v in raw_map.items()}
    model_set = YoloModelSet(
        medieval_models=None, class_maps=None,
        single_model=YOLO(file_path), custom_cls_map=custom_cls_map,
        tm_threshold=tm_threshold, tm_device=tm_device,
        st_threshold=st_threshold, st_device=st_device,
        confidence_threshold=confidence_threshold, device=device,
        stored_model_id=model_id,
        model_label=f"custom: {model_name}", model_hash=file_hash,
    )
    return model_set, [f"Model loaded: {model_name}"]

def write_annotation(cur, con, project_id: int, image_id: str, image_name: str,
                     yolo_txt: str, stored_model_id: str, model_label: str,
                     model_hash: Optional[str]) -> str:
    """Replaces this image's annotations row. Returns the new row's id."""
    cur.execute("DELETE FROM annotations WHERE project_id=%s AND image_id=%s", (project_id, image_id))
    ann_id = _uuid.uuid4().hex
    cur.execute(
        "INSERT INTO annotations (id, project_id, image_id, image_name, yolo_txt, model_id, model_label, model_hash)"
        " VALUES (%s,%s,%s,%s,%s,%s,%s,%s)",
        (ann_id, project_id, image_id, image_name, yolo_txt, stored_model_id, model_label, model_hash),
    )
    con.commit()
    return ann_id