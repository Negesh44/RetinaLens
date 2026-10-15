"""DR grade probability helpers shared by training and the API server (NumPy only)."""

from typing import Union

import numpy as np


def continuous_to_probabilities(
    continuous_pred: Union[float, np.ndarray],
    sigma: float = 0.65,
) -> np.ndarray:
    """Converts a continuous severity prediction into calibrated 5-class probabilities

    using a Gaussian distance kernel.
    """
    is_scalar = np.isscalar(continuous_pred)
    preds = np.atleast_1d(continuous_pred)

    class_indices = np.arange(5, dtype=float)  # [0, 1, 2, 3, 4]
    # Distances shape: [N, 5]
    diffs = preds[:, None] - class_indices[None, :]
    logits = -(diffs ** 2) / (2 * (sigma ** 2))
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    if is_scalar:
        return probs[0]
    return probs
