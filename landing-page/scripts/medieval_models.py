"""Resolution of the bundled 'medieval manuscripts' layer-separation checkpoints.

Two YOLO .pt files ship in-repo via Git LFS (see .gitattributes) so the
default preset works out of the box, offline, with no HuggingFace auth.
MOTHRA_MEDIEVAL_MODELS_DIR lets operators swap in updated weights without
a repo commit; MOTHRA_HF_TOKEN is a last-resort runtime-download fallback
for operators who don't want to commit weights to the repo at all (the
upstream HF repo is gated and returns 401 unauthenticated).
"""
import os
from functools import lru_cache
from pathlib import Path
from config import MEDIEVAL_MODELS_DIR as BUNDLED_DIR

TEXT_MUSIC_FILENAME = "text_music_detector_fulldata.pt"
STAVE_FILENAME = "stave_detector_fulldata.pt"
HF_REPO = "DDMAL-lab/mothra-yolov11-checkpoints"

# Merged 0-indexed class space written into the YOLO .txt annotation lines:
# 0=text, 1=music, 2=staves. This matches configs/mothra_base.yaml's class
# order and, critically, what ic_core.ingest_page_yolo expects — it adds +1
# to each .txt class id to reach MOTHRA's 1-indexed classId space (1=Text,
# 2=Neumes, 3=Staves). Do NOT copy the 1/2/3 maps from
# mothra-text/scripts/run_mothra_inference.py here — those target MOTHRA
# JSON's classId directly (no downstream shift), a different consumer than
# this .txt-based pipeline.
TEXT_MUSIC_CLASS_MAP = {0: 0, 1: 1}  # text_music_detector already trained 0=text, 1=music
STAVE_CLASS_MAP = {0: 2}  # stave_detector's only class -> merged slot 2

ENV_MODELS_DIR = "MOTHRA_MEDIEVAL_MODELS_DIR"
ENV_HF_TOKEN = "MOTHRA_HF_TOKEN"

def _dir_has_both(d: Path) -> bool:
    return (d / TEXT_MUSIC_FILENAME).is_file() and (d / STAVE_FILENAME).is_file()

def _download_from_hf(token: str) -> tuple[str, str]:
    from huggingface_hub import hf_hub_download
    tm_path = hf_hub_download(HF_REPO, TEXT_MUSIC_FILENAME, token=token)
    st_path = hf_hub_download(HF_REPO, STAVE_FILENAME, token=token)
    return tm_path, st_path

@lru_cache(maxsize=1)
def resolve_medieval_model_paths() -> tuple[str, str]:
    """Return (text_music_model_path, stave_model_path), in priority order:
    1. MOTHRA_MEDIEVAL_MODELS_DIR override directory.
    2. Bundled in-repo copies (Git LFS) — the default, offline, no-auth path.
    3. HuggingFace download, only if MOTHRA_HF_TOKEN is set.
    """
    override = os.environ.get(ENV_MODELS_DIR)
    if override: 
        d = Path(override).expanduser()
        if _dir_has_both(d):
            return str(d / TEXT_MUSIC_FILENAME), str(d / STAVE_FILENAME)
        raise RuntimeError(
            f"{ENV_MODELS_DIR}={override} does not contain both "
            f"'{TEXT_MUSIC_FILENAME}' and '{STAVE_FILENAME}'."
        )
    
    if _dir_has_both(BUNDLED_DIR):
        return str(BUNDLED_DIR / TEXT_MUSIC_FILENAME), str(BUNDLED_DIR / STAVE_FILENAME)
    
    hf_token = os.environ.get(ENV_HF_TOKEN)
    if hf_token:
        try:
            return _download_from_hf(hf_token)
        except Exception as e:
            raise RuntimeError(
                f"Bundled medieval model files are missing and the HuggingFace "
                f"download fallback failed: {e}"
            ) from e
    
    raise RuntimeError(
        f"Medieval layer-separation model files not found at {BUNDLED_DIR}. "
        "This usually means the repo was cloned without Git LFS — run "
        "`git lfs pull`. Alternatively set MOTHRA_MEDIEVAL_MODELS_DIR to a "
        f"local directory containing {TEXT_MUSIC_FILENAME} and "
        f"{STAVE_FILENAME}, or set {ENV_HF_TOKEN} to fetch them at runtime."
    )