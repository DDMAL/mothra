"""
BGR (background removal) adapter.

Wraps the inference helpers from the external manuscript-ink-separation script
(`inference_simple.py`) for in-memory use. The external script's public entry
point writes to disk; this module provides the same processing pipeline but
returns the ink-on-white image directly, without disk I/O.

If the path to the external script changes, update _CANDIDATE_DIRS below (or
set MUSCRAT_LAYER_SEP_DIR). This module is the single place that needs to
know where it lives; downstream code imports from here.
"""

import os
import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Location of the external inference script
# ---------------------------------------------------------------------------

# Known locations of the external muscrat/layer_sep repo across the machines
# this project has been developed on; first one that actually contains
# inference_simple.py wins. Override with MUSCRAT_LAYER_SEP_DIR for a machine
# not listed here.
_CANDIDATE_DIRS = [
    os.environ.get("MUSCRAT_LAYER_SEP_DIR"),
    "/Users/kyriebouressa/Documents/muscrat/layer_sep/scripts",  # mini
    "/Users/ekaterina/Documents/muscrat/layer_sep/scripts",  # macbook
    "/Users/ekaterina/Documents/Documents_angantyr/GitHub/muscrat/layer_sep/scripts",
]
INFERENCE_SCRIPT_DIR = next(
    (
        d
        for d in _CANDIDATE_DIRS
        if d and os.path.isfile(os.path.join(d, "inference_simple.py"))
    ),
    None,
)
if INFERENCE_SCRIPT_DIR is None:
    raise ModuleNotFoundError(
        "Could not find inference_simple.py (external muscrat/layer_sep repo, "
        "needed for BGR preprocessing) in any known location:\n  "
        + "\n  ".join(d for d in _CANDIDATE_DIRS if d)
        + "\nSet MUSCRAT_LAYER_SEP_DIR to override, or pass --no-bgr to skip "
        "BGR preprocessing entirely (run_page.py only)."
    )

sys.path.insert(0, INFERENCE_SCRIPT_DIR)
from inference_simple import (  # noqa: E402  (sys.path insertion above)
    load_model as _load_model,
    sliding_window_inference,
    post_process_ink,
    separate_layers,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BGR_WINDOW_SIZE = 512
DEFAULT_BGR_STRIDE = 256
DEFAULT_BGR_CONFIDENCE = 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_bgr_model(checkpoint_path: str, device: str):
    """Load the BGR model from a checkpoint. Thin wrapper for naming clarity."""
    return _load_model(checkpoint_path, device)


def run_bgr_inference(
    model,
    image_rgb: np.ndarray,
    window_size: int = DEFAULT_BGR_WINDOW_SIZE,
    stride: int = DEFAULT_BGR_STRIDE,
    confidence: float = DEFAULT_BGR_CONFIDENCE,
    device: str = "cpu",
) -> np.ndarray:
    """Run the BGR model on a full RGB page and return the ink-on-white layer.

    This is the in-memory equivalent of inference_simple.process_image without
    the disk writes or the parchment/comparison outputs. We return just the
    ink layer (RGB on white background), which is what downstream stage 1
    consumers expect.

    Args:
        model: BGR model returned by load_bgr_model.
        image_rgb: Full page as an RGB numpy array.
        window_size: Sliding-window size in pixels.
        stride: Sliding-window stride in pixels.
        confidence: Threshold on the per-pixel ink probability.
        device: 'cuda' or 'cpu'.

    Returns:
        RGB ink layer (same dtype and spatial size as image_rgb).
    """
    probability_map = sliding_window_inference(
        model,
        image_rgb,
        window_size=window_size,
        stride=stride,
        device=device,
    )
    ink_mask = post_process_ink(probability_map, confidence_threshold=confidence)
    ink_layer, _parchment_layer = separate_layers(image_rgb, ink_mask)
    return ink_layer
