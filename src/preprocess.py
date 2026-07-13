from __future__ import annotations

import numpy as np

from .utils import normalize_audio


def preprocess_audio(y: np.ndarray) -> np.ndarray:
    """Normalize an audio signal after resampling/mono conversion."""
    return normalize_audio(y)

