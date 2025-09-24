"""
Metrics for the neurite project implemented with the multi-backend Keras API.

If you use this code, please cite the following, and read function docs for further info/citations
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,
CVPR 2018. https://arxiv.org/abs/1903.03148


Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in
compliance with the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
implied. See the License for the specific language governing permissions and limitations under
the License.
"""

from __future__ import annotations

import warnings
from typing import Iterable, Optional, Sequence

import numpy as np
from keras import ops

from ._ops import epsilon_like, safe_divide, soft_quantize
from .losses import (
    CategoricalCrossentropy as _LossCategoricalCrossentropy,
    Dice as _LossDice,
    HardDice as _LossHardDice,
    MeanSquaredErrorProb as _LossMeanSquaredErrorProb,
    SoftDice as _LossSoftDice,
)


class MutualInformation:
    """Soft mutual information approximation for intensity and probabilistic volumes."""

    def __init__(
        self,
        bin_centers: Optional[Iterable[float]] = None,
        nb_bins: Optional[int] = None,
        soft_bin_alpha: Optional[float] = None,
        min_clip: Optional[float] = None,
        max_clip: Optional[float] = None,
    ) -> None:
        if bin_centers is not None:
            bin_centers = np.asarray(bin_centers, dtype=np.float32)
            if nb_bins is not None:
                raise ValueError('Provide either bin_centers or nb_bins, not both.')
            nb_bins = int(bin_centers.shape[0])

        if nb_bins is None:
            nb_bins = 16

        self.bin_centers = bin_centers
        self.nb_bins = nb_bins
        self.min_clip = -np.inf if min_clip is None else min_clip
        self.max_clip = np.inf if max_clip is None else max_clip

        if soft_bin_alpha is None:
            sigma_ratio = 0.5
            if self.bin_centers is None:
                sigma = sigma_ratio / max(self.nb_bins - 1, 1)
            else:
                diffs = np.diff(self.bin_centers)
                sigma = sigma_ratio * float(np.mean(diffs))
            soft_bin_alpha = 1.0 / (2.0 * sigma ** 2)

        self.soft_bin_alpha = float(soft_bin_alpha)

    def volumes(self, x, y):
        """Mutual information for batches of single-channel volumes."""

        x = ops.convert_to_tensor(x)
        y = ops.convert_to_tensor(y)

        self._assert_single_channel(x, 'volume_mi requires two single-channel volumes.')
        self._assert_single_channel(y, 'volume_mi requires two single-channel volumes.')
        return ops.reshape(self.channelwise(x, y), (-1,))

    def segs(self, x, y):
        """Mutual information between probabilistic segmentation maps."""

        x = ops.convert_to_tensor(x)
        y = ops.convert_to_tensor(y)
        return self.maps(x, y)

    def volume_seg(self, x, y):
        """Mutual information between a volume and a probabilistic segmentation map."""

        x = ops.convert_to_tensor(x)
        y = ops.convert_to_tensor(y)

        x_channels = _static_channels(x)
        y_channels = _static_channels(y)

        if x_channels is not None and y_channels is not None:
            if min(x_channels, y_channels) != 1:
                raise ValueError('volume_seg_mi requires one single-channel volume.')
            if max(x_channels, y_channels) <= 1:
                raise ValueError('volume_seg_mi requires one multi-channel segmentation.')

        if x_channels == 1:
            x = self._soft_sim_map(ops.squeeze(x, axis=-1))
        else:
            y = self._soft_sim_map(ops.squeeze(y, axis=-1))

        return self.maps(x, y)

    def channelwise(self, x, y):
        """Channel-wise mutual information for batches of multi-channel volumes."""

        x = ops.convert_to_tensor(x)
        y = ops.convert_to_tensor(y)

        self._assert_same_shape(x, y)
        x, y = self._reshape_to_batch_voxels_channels(x, y)

        batch = ops.shape(x)[0]
        voxels = ops.shape(x)[1]

        x_perm = ops.transpose(x, (0, 2, 1))
        y_perm = ops.transpose(y, (0, 2, 1))

        x_flat = ops.reshape(x_perm, (-1, voxels))
        y_flat = ops.reshape(y_perm, (-1, voxels))

        x_quant = self._soft_sim_map(x_flat)
        y_quant = self._soft_sim_map(y_flat)

        mi = self.maps(x_quant, y_quant)
        return ops.reshape(mi, (batch, -1))

    def maps(self, x, y):
        """Mutual information for probability or similarity maps."""

        x = ops.convert_to_tensor(x)
        y = ops.convert_to_tensor(y)

        self._assert_same_shape(x, y)

        if len(x.shape) != 3:
            batch = ops.shape(x)[0]
            bins = ops.shape(x)[-1]
            x = ops.reshape(x, (batch, -1, bins))
            y = ops.reshape(y, (batch, -1, bins))

        eps = epsilon_like(x)

        x_trans = ops.transpose(x, (0, 2, 1))
        pxy = ops.matmul(x_trans, y)
        total = ops.sum(pxy, axis=(1, 2), keepdims=True)
        pxy = safe_divide(pxy, total)

        px = ops.sum(x, axis=1, keepdims=True)
        px = safe_divide(px, ops.sum(px, axis=2, keepdims=True))

        py = ops.sum(y, axis=1, keepdims=True)
        py = safe_divide(py, ops.sum(py, axis=2, keepdims=True))

        px_trans = ops.transpose(px, (0, 2, 1))
        pxpy = ops.matmul(px_trans, py)
        ratio = safe_divide(pxy, pxpy)
        log_term = ops.log(ratio + eps)
        return ops.sum(pxy * log_term, axis=(1, 2))

    def _soft_log_sim_map(self, x):
        return soft_quantize(
            x,
            alpha=self.soft_bin_alpha,
            bin_centers=self.bin_centers,
            nb_bins=self.nb_bins,
            min_clip=self.min_clip,
            max_clip=self.max_clip,
            return_log=True,
        )

    def _soft_sim_map(self, x):
        return soft_quantize(
            x,
            alpha=self.soft_bin_alpha,
            bin_centers=self.bin_centers,
            nb_bins=self.nb_bins,
            min_clip=self.min_clip,
            max_clip=self.max_clip,
            return_log=False,
        )

    def _soft_prob_map(self, x):
        hist = self._soft_sim_map(x)
        hist_sum = ops.sum(hist, axis=-1, keepdims=True)
        return safe_divide(hist, hist_sum)

    @staticmethod
    def _reshape_to_batch_voxels_channels(x, y):
        if len(x.shape) != 3:
            batch = ops.shape(x)[0]
            channels = ops.shape(x)[-1]
            x = ops.reshape(x, (batch, -1, channels))
            y = ops.reshape(y, (batch, -1, channels))
        return x, y

    @staticmethod
    def _assert_same_shape(x, y):
        x_shape = tuple(x.shape)
        y_shape = tuple(y.shape)
        if None not in x_shape + y_shape and x_shape != y_shape:
            raise ValueError('Input tensors must have the same shape.')

    @staticmethod
    def _assert_single_channel(x, message: str):
        channels = _static_channels(x)
        if channels is not None and channels != 1:
            raise ValueError(message)


class Dice(_LossDice):
    """Dice score metric with optional loss helpers."""

    def loss(self, y_true, y_pred):
        warnings.warn(
            'ne.metrics.*.loss functions are deprecated. Please use the ne.losses equivalents.',
            RuntimeWarning,
        )
        return super().loss(y_true, y_pred)

    def mean_loss(self, y_true, y_pred):
        warnings.warn(
            'ne.metrics.*.loss functions are deprecated. Please use the ne.losses equivalents.',
            RuntimeWarning,
        )
        return super().mean_loss(y_true, y_pred)


class SoftDice(_LossSoftDice):
    """Soft Dice score metric."""


class HardDice(_LossHardDice):
    """Hard Dice score metric."""


class CategoricalCrossentropy(_LossCategoricalCrossentropy):
    """Categorical crossentropy metric with optional per-label weights."""


class MeanSquaredErrorProb(_LossMeanSquaredErrorProb):
    """Mean squared error metric tailored for probabilistic label logits."""


def multiple_metrics_decorator(metrics: Sequence, weights: Optional[Sequence[float]] = None):
    """Combine multiple metrics with optional weights."""

    if weights is None:
        weights = np.ones(len(metrics))

    def metric(y_true, y_pred):
        total_val = 0.0
        for idx, met in enumerate(metrics):
            total_val = total_val + weights[idx] * met(y_true, y_pred)
        return total_val

    return metric


def _static_channels(tensor) -> Optional[int]:
    shape = getattr(tensor, 'shape', None)
    if shape is None or len(shape) == 0:
        return None
    return shape[-1]


__all__ = [
    'MutualInformation',
    'Dice',
    'SoftDice',
    'HardDice',
    'CategoricalCrossentropy',
    'MeanSquaredErrorProb',
    'multiple_metrics_decorator',
]
