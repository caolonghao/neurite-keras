"""
Loss functions for neurite using the multi-backend Keras API.

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
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
from keras import backend as K
from keras import ops
from keras.losses import CategoricalCrossentropy as _BaseCategoricalCrossentropy
from keras.losses import MeanSquaredError as _BaseMeanSquaredError

from ._ops import flatten_batch_channels, safe_divide


class Dice:
    """Dice metric with utilities to compute losses."""

    def __init__(
        self,
        dice_type: str = "soft",
        input_type: str = "prob",
        nb_labels: Optional[int] = None,
        weights: Optional[Sequence[float]] = None,
        check_input_limits: bool = True,
        laplace_smoothing: float = 0.0,
        normalize: bool = False,
    ) -> None:
        self.dice_type = dice_type
        self.input_type = input_type
        self.nb_labels = nb_labels
        self.weights = weights
        self.normalize = normalize
        self.check_input_limits = check_input_limits
        self.laplace_smoothing = laplace_smoothing

        if self.input_type not in {"prob", "one_hot", "max_label"}:
            raise ValueError("input_type must be 'prob', 'one_hot', or 'max_label'.")
        if self.dice_type not in {"soft", "hard"}:
            raise ValueError("dice_type must be 'soft' or 'hard'.")
        if self.dice_type == "hard" and self.input_type == "max_label" and self.nb_labels is None:
            raise ValueError("nb_labels must be provided when using hard Dice with max labels.")
        if self.dice_type == "soft" and self.input_type == "max_label":
            raise ValueError("Soft Dice requires probabilistic inputs.")

    def _maybe_normalize(self, tensor):
        axis = -1
        denom = ops.sum(tensor, axis=axis, keepdims=True)
        return safe_divide(tensor, denom)

    def _validate_range(self, tensor, name: str) -> None:
        if not self.check_input_limits:
            return

        # Best-effort validation in eager mode; silently skip for symbolic tensors.
        try:
            min_val = float(ops.min(tensor))
            max_val = float(ops.max(tensor))
        except TypeError:
            return
        if min_val < -1e-6 or max_val > 1.0 + 1e-6:
            raise ValueError(f"{name} expected to be within [0, 1], but received values in [{min_val}, {max_val}].")

    def dice(self, y_true, y_pred):
        if self.input_type in {"prob", "one_hot"}:
            if self.normalize:
                y_true = self._maybe_normalize(y_true)
                y_pred = self._maybe_normalize(y_pred)
            self._validate_range(y_true, "y_true")
            self._validate_range(y_pred, "y_pred")

        if self.dice_type == "hard":
            if self.input_type == "prob":
                warnings.warn(
                    "Computing hard Dice from probabilistic inputs removes gradients. "
                    "Only use this configuration in evaluation settings.",
                    RuntimeWarning,
                )
                if self.nb_labels is None:
                    shape = y_pred.shape
                    if shape[-1] is None:
                        raise ValueError("nb_labels must be provided when tensor channel dimension is dynamic.")
                    self.nb_labels = int(shape[-1])
                y_pred = ops.argmax(y_pred, axis=-1)
                y_true = ops.argmax(y_true, axis=-1)

            if self.nb_labels is None:
                raise ValueError("nb_labels must be provided for hard Dice computations.")

            dtype = "float32"
            y_pred = ops.one_hot(y_pred, self.nb_labels, dtype=dtype)
            y_true = ops.one_hot(y_true, self.nb_labels, dtype=dtype)

        y_true_flat = flatten_batch_channels(y_true)
        y_pred_flat = flatten_batch_channels(y_pred)

        intersection = ops.sum(y_true_flat * y_pred_flat, axis=1)
        numerator = 2.0 * intersection
        denominator = ops.sum(ops.square(y_true_flat), axis=1) + ops.sum(ops.square(y_pred_flat), axis=1)

        if self.laplace_smoothing > 0:
            smooth = ops.convert_to_tensor(self.laplace_smoothing, dtype=numerator.dtype)
            numerator = numerator + smooth
            denominator = denominator + smooth
            return numerator / denominator

        return safe_divide(numerator, denominator)

    def mean_dice(self, y_true, y_pred):
        dice_scores = self.dice(y_true, y_pred)
        if self.weights is not None:
            weight_tensor = ops.convert_to_tensor(self.weights, dtype=dice_scores.dtype)
            dice_scores = dice_scores * weight_tensor
        return ops.mean(dice_scores)

    def loss(self, y_true, y_pred):
        return -self.dice(y_true, y_pred)

    def mean_loss(self, y_true, y_pred):
        return -self.mean_dice(y_true, y_pred)


class SoftDice(Dice):
    def __init__(
        self,
        weights: Optional[Sequence[float]] = None,
        check_input_limits: bool = True,
        laplace_smoothing: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            dice_type="soft",
            input_type="prob",
            weights=weights,
            check_input_limits=check_input_limits,
            laplace_smoothing=laplace_smoothing,
            normalize=normalize,
        )


class HardDice(Dice):
    def __init__(
        self,
        nb_labels: int,
        input_type: str = "max_label",
        weights: Optional[Sequence[float]] = None,
        check_input_limits: bool = True,
        laplace_smoothing: float = 0.0,
        normalize: bool = False,
    ) -> None:
        super().__init__(
            dice_type="hard",
            input_type=input_type,
            nb_labels=nb_labels,
            weights=weights,
            check_input_limits=check_input_limits,
            laplace_smoothing=laplace_smoothing,
            normalize=normalize,
        )


class CategoricalCrossentropy(_BaseCategoricalCrossentropy):
    """Categorical crossentropy with optional per-label weights."""

    def __init__(self, label_weights: Optional[Iterable[float]] = None, **kwargs) -> None:
        self.label_weights = None
        if label_weights is not None:
            self.label_weights = ops.convert_to_tensor(label_weights, dtype="float32")
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        if self.label_weights is not None:
            if y_pred.shape[-1] is not None and self.label_weights.shape[-1] != y_pred.shape[-1]:
                raise ValueError(
                    f"Label weights must have length {y_pred.shape[-1]}, "
                    f"but received {self.label_weights.shape[-1]}."
                )
            weights = ops.cast(self.label_weights, y_true.dtype)
            y_true = y_true * weights
        return super().call(y_true, y_pred)

    def cce(self, y_true, y_pred):
        return self.call(y_true, y_pred)


class MeanSquaredErrorProb(_BaseMeanSquaredError):
    """Mean squared error tailored for probabilistic label logits."""

    def __init__(self, label_weights: Optional[Iterable[float]] = None, **kwargs) -> None:
        self.label_weights = None
        if label_weights is not None:
            self.label_weights = ops.convert_to_tensor(label_weights, dtype="float32")
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        sample_weight = None
        if self.label_weights is not None:
            if y_pred.shape[-1] is not None and self.label_weights.shape[0] != y_pred.shape[-1]:
                raise ValueError(
                    f"Label weights must have length {y_pred.shape[-1]}, "
                    f"but received {self.label_weights.shape[0]}."
                )
            y_true = ops.expand_dims(y_true, axis=-1)
            y_pred = ops.expand_dims(y_pred, axis=-1)
            sample_weight = ops.cast(self.label_weights, y_true.dtype)
        return super().call(y_true, y_pred, sample_weight=sample_weight)

    def mse(self, y_true, y_pred):
        return self.call(y_true, y_pred)


def multiple_losses_decorator(losses: Sequence[Callable], weights: Optional[Sequence[float]] = None):
    """Combine multiple loss functions with optional weights."""

    if weights is None:
        weights = np.ones(len(losses), dtype=float)

    def loss(y_true, y_pred):
        total_val = 0.0
        for idx, los in enumerate(losses):
            total_val = total_val + weights[idx] * los(y_true, y_pred)
        return total_val

    return loss


# TODO: Provide a PyTorch-backend MutualInformation implementation when utilities have been ported.
