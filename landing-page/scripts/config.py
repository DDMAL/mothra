import os
from pathlib import Path

import yaml

_SCRIPTS_DIR = Path(__file__).parent
with open(_SCRIPTS_DIR / "config.yaml") as f:
    _cfg = yaml.safe_load(f)

def _path(key: str) -> Path:
    p = Path(_cfg["paths"][key])
    return p if p.is_absolute() else (_SCRIPTS_DIR / p).resolve()

def _url(env_var: str, cfg_key: str) -> str:
    return os.environ.get(env_var, _cfg["services"][cfg_key]).rstrip("/")

MODELS_DIR = _path("models_dir")
NEON_MANIFESTS_DIR = _path("neon_manifests_dir")
MOCK_DATA_DIR = _path("mock_data_dir")
MEDIEVAL_MODELS_DIR = _path("medieval_models_dir")

IC_API_URL = _url("IC_API_URL", "ic_api_url")
IC_PUBLIC_URL = _url("IC_PUBLIC_URL", "ic_public_url")
TEXT_API_URL = _url("TEXT_API_URL", "text_api_url")

CELERY_BROKER_URL = os.environ.get("CELERY_BROKER_URL", _cfg["celery"]["broker_url"])