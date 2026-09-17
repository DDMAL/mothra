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
    """The service URL, with an env var overriding config.yaml's default.

    An EMPTY env var counts as unset, matching _path_env above. This is not
    hypothetical tidiness: docker-compose.yml passes optional settings as
    `VAR: ${VAR:-}`, which hands the container an empty string rather than
    omitting the variable. A bare os.environ.get would treat that as a real
    override and return "", producing requests against no host at all --
    the yaml default would be silently unreachable precisely when nobody had
    configured anything.
    """
    override = os.environ.get(env_var, "").strip()
    return (override or _cfg["services"][cfg_key]).rstrip("/")

MODELS_DIR = _path("models_dir")
NEON_MANIFESTS_DIR = _path("neon_manifests_dir")
MEDIEVAL_MODELS_DIR = _path("medieval_models_dir")
MEI_ENCODING_DIR = _path("mei_encoding_dir")
PITCH_FINDING_DIR = _path_env("pitch_finding_dir", "PITCH_FINDING_DIR")

IC_API_URL = _url("IC_API_URL", "ic_api_url")
IC_PUBLIC_URL = _url("IC_PUBLIC_URL", "ic_public_url")
TEXT_API_URL = _url("TEXT_API_URL", "text_api_url")
PACO_API_URL = _url("PACO_API_URL", "paco_api_url")
CU_API_URL = _url("CU_API_URL", "cu_api_url")

# Cantus Ultimus deposit credential (cu_api.py) -- a DRF token belonging to
# CU's `mothra` service account, sent as `Authorization: Token <key>`. This is
# a SECRET, so unlike CU_API_URL above it lives only in .env / the process
# environment, never config.yaml.
#
# Deliberately NOT required at import, unlike MOTHRA_SECRET: a deployment that
# never submits to CU is a perfectly valid one, and making this fatal would
# CrashLoopBackOff every such backend. cu_api.py raises its own
# "not_configured" error at call time instead, so the failure reaches the one
# user who tried to submit rather than taking the whole app down.
CU_DEPOSIT_TOKEN = os.environ.get("CU_DEPOSIT_TOKEN", "").strip()

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
