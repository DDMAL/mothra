"""Validate uploaded YOLO checkpoints and derive their text/music/staves
class mapping, so inference_api.py never has to assume a custom model's raw
class indices happen to already be 0=text/1=music/2=staves (the assumption
that caused the stave-clustering bug for the bundled medieval models)."""
import json
from typing import Optional

_NAME_SYNONYMS = {
    "text": "text",
    "music": "music", "neume": "music", "neumes": "music",
    "stave": "staves", "staves": "staves", "staff": "staves",
    "staffline": "staves", "stafflines": "staves",
}

def inspect_yolo_checkpoint(file_path: str) -> dict:
    """Load the checkpoint and return {"names": {idx: raw_name}, "class_map": {...} | None}.
    Raises ValueError if the file isn't a loadable YOLO *detection* checkpoint —
    callers should reject the upload on that, rather than deferring the failure
    to a confusing mid-/predict/ ultralytics stack trace."""
    from ultralytics import YOLO
    try:
        model = YOLO(file_path)
    except Exception as e:
        raise ValueError(f"not a loadable YOLO checkpoint: {e}") from e
    if getattr(model, "task", None) != "detect":
        raise ValueError(f"expected a detection model, got task={model.task!r}")
    names = {int(k): str(v) for k, v in model.names.items()}
    return {"names": names, "class_map": _auto_class_map(names)}

def _auto_class_map(names: dict[int, str]) -> Optional[dict[str, str]]:
    resolved: dict[str, str] = {}
    seen_categories: set[str] = set()
    for idx, name in names.items():
        category = _NAME_SYNONYMS.get(name.strip().lower())
        if not category or category in seen_categories:
            return None # unrecognized name, or two classes mapping to the same category
        resolved[str(idx)] = category
        seen_categories.add(category)
    return resolved

