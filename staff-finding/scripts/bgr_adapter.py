"""
BGR (background removal) adapter.

Wraps the inference helpers from the external manuscript-ink-separation script
(`inference_simple.py`) for in-memory use. The external script's public entry
point writes to disk; this module provides the same processing pipeline but
returns the ink-on-white image directly, without disk I/O.

If the path to the external script changes, update INFERENCE_SCRIPT_DIR below.
This module is the single place that needs to know where it lives; downstream
code imports from here.
"""

import sys
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Location of the external inference script
# ---------------------------------------------------------------------------

# Adjust to wherever inference_simple.py lives in your environment, or
# pip-install it as a package and import normally.
INFERENCE_SCRIPT_DIR = "/Users/kyriebouressa/Documents/muscrat/layer_sep/scripts"
# mini: /Users/kyriebouressa/Documents/muscrat/layer_sep/scripts
# macbook: "/Users/ekaterina/Documents/Documents - angantyr/muscrat/layer_sep/scripts/"

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
