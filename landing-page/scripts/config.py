import os
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).parent
with open(_SCRIPTS_DIR / "config.yaml") as f:
    _cfg = yaml.safe_load(f)

def _path(key: str) -> Path:
    p = Path(_cfg["paths"][key])
    return p if p.is_absolute() else (_SCRIPTS_DIR / p).resolve()

def _path_env(key: str, env_var: str) -> Path:
    """Same as _path, but an env var overrides the YAML outright.

    Needed for paths whose in-container location is not a fixed offset from
    scripts/ the way assets/ and stored_models are -- the pitch-finding
    submodule is COPYd in from an additional build context, so only the
    Dockerfile knows where it landed. Mirrors main.py's own
    STAFFLINE_MODELS_DIR override.
    """
    override = os.environ.get(env_var, "").strip()
    if override:
        return Path(override)
    return _path(key)

def _url(env_var: str, cfg_key: str) -> str:
    return os.environ.get(env_var, _cfg["services"][cfg_key]).rstrip("/")

MODELS_DIR = _path("models_dir")
MEDIEVAL_MODELS_DIR = _path("medieval_models_dir")
MEI_ENCODING_DIR = _path("mei_encoding_dir")
PITCH_FINDING_DIR = _path_env("pitch_finding_dir", "PITCH_FINDING_DIR")

IC_API_URL = _url("IC_API_URL", "ic_api_url")
IC_PUBLIC_URL = _url("IC_PUBLIC_URL", "ic_public_url")
TEXT_API_URL = _url("TEXT_API_URL", "text_api_url")
PACO_API_URL = _url("PACO_API_URL", "paco_api_url")

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", _cfg["celery"]["broker_url"])

# Dev-only escape hatch for machines that can't run ultralytics (e.g. no
# compatible torch build). When set, uploaded YOLO checkpoints are stored
# without inspection (class mapping must be set by hand) and the predict step
# is expected to be skipped from the frontend (VITE_SKIP_PREDICT). Leave unset
# for real deployments — inspection catches bad checkpoints up front.
SKIP_YOLO = os.environ.get("MOTHRA_SKIP_YOLO", "").strip().lower() in ("1", "true", "yes")

# Off switch for the real pitch-finding stage (pitch_stage.py, algorithm #1
# of the pitch-finding/ submodule). Set MOTHRA_PITCH_FINDING=0 to encode with
# encode_to_mei.py's older geometric placeholder pitch instead -- the same
# path every glyph the algorithm cannot resolve already falls back to. On by
# default; this exists to isolate a suspected pitch regression without a
# redeploy, not as a supported long-term configuration.
PITCH_FINDING_ENABLED = os.environ.get("MOTHRA_PITCH_FINDING", "").strip().lower() not in ("0", "false", "no")
