"""Common backend utilities built on top of the Keras multi-backend API."""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
from keras import backend as K
from keras import ops


def to_tensor(value, dtype: Optional[str] = None):
    """Convert ``value`` to a Keras tensor with the requested dtype."""

    return ops.convert_to_tensor(value, dtype=dtype) if dtype is not None else ops.convert_to_tensor(value)


def epsilon_like(reference):
    """Return a small epsilon with the same dtype as ``reference``."""

    dtype = getattr(reference, "dtype", None)
    eps = ops.convert_to_tensor(K.epsilon(), dtype="float32")
    return ops.cast(eps, dtype or "float32")


def safe_divide(numerator, denominator, epsilon: Optional[float] = None):
    """Safely divide two tensors, guarding against division by zero."""

    denom = denominator
    if epsilon is not None:
        denom = ops.maximum(denom, ops.convert_to_tensor(epsilon, dtype=denominator.dtype))
    else:
        denom = ops.maximum(denom, epsilon_like(denominator))
    return numerator / denom


def flatten_batch_channels(x):
    """Flatten all spatial dimensions while keeping batch and channel axes."""

    shape = ops.shape(x)
    batch = shape[0]
    channels = shape[-1]
    return ops.reshape(x, (batch, -1, channels))


def soft_quantize(
    x,
    bin_centers: Optional[Iterable[float]] = None,
    nb_bins: Optional[int] = 16,
    alpha: float = 1.0,
    min_clip: float = -np.inf,
    max_clip: float = np.inf,
    return_log: bool = False,
):
    """Softly quantize values in ``x`` using radial basis functions."""

    x = to_tensor(x)
    dtype = x.dtype

    if bin_centers is not None:
        if nb_bins is not None:
            raise ValueError("Provide either bin_centers or nb_bins, not both.")
        bin_centers = to_tensor(bin_centers, dtype="float32")
        nb_bins = bin_centers.shape[0]
    else:
        if nb_bins is None:
            nb_bins = 16
        minval = ops.min(x)
        maxval = ops.max(x)
        bin_centers = ops.linspace(minval, maxval, nb_bins)

    x = ops.clip(x, min_clip, max_clip)
    x = ops.expand_dims(x, axis=-1)

    rank = len(x.shape)
    target_shape = (1,) * (rank - 1) + (nb_bins,)
    bin_centers = ops.reshape(bin_centers, target_shape)

    alpha_tensor = ops.cast(alpha, dtype)
    bin_diff = ops.square(x - bin_centers)
    log_values = -alpha_tensor * bin_diff

    if return_log:
        return log_values

    return ops.exp(log_values)


__all__ = [
    "to_tensor",
    "epsilon_like",
    "safe_divide",
    "flatten_batch_channels",
    "soft_quantize",
]
