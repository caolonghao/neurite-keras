"""Simplified torch-friendly model builders for neurite.

The original TensorFlow-centric implementations exposed a very broad API surface that powered
numerous data-generation pipelines. To keep the migration manageable, we currently provide a
minimal subset that covers the most common entry points while avoiding any TensorFlow
dependencies. The exported helpers build lightweight Keras (multi-backend) models that operate on
PyTorch tensors when the torch backend is selected.

The long-term goal is to re-introduce the richer feature set incrementally. Functions that have not
yet been ported raise ``NotImplementedError`` to make the work-in-progress status explicit.
"""

from __future__ import annotations

from typing import Iterable, Optional, Sequence

import numpy as np
from keras import Model
from keras import layers as KL
from keras import ops

from . import layers as nlayers
from ._ops import safe_divide as _safe_divide


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _get_ndims(input_shape: Sequence[int]) -> int:
    ndims = len(input_shape) - 1
    if ndims not in (2, 3):
        raise ValueError('Only 2D and 3D inputs are currently supported.')
    return ndims


def _conv_layer(ndims: int):
    return {2: KL.Conv2D, 3: KL.Conv3D}[ndims]


def _pool_layer(ndims: int):
    return {2: KL.MaxPooling2D, 3: KL.MaxPooling3D}[ndims]


def _upsample_layer(ndims: int):
    return {2: KL.UpSampling2D, 3: KL.UpSampling3D}[ndims]


def _activation(name: Optional[str]):
    return KL.Activation(name) if name else KL.Activation('relu')


def _conv_block(x, filters: int, kernel_size: int, activation: str = 'relu', dropout: float = 0.0, batch_norm: bool = False):
    ndims = len(x.shape) - 1
    Conv = _conv_layer(ndims - 1)
    x = Conv(filters, kernel_size, padding='same')(x)
    if batch_norm:
        x = KL.BatchNormalization()(x)
    x = KL.Activation(activation)(x)
    if dropout > 0:
        x = KL.Dropout(dropout)(x)
    return x


# -----------------------------------------------------------------------------
# Encoder / decoder scaffolding
# -----------------------------------------------------------------------------


def conv_enc(
    nb_features: int,
    input_shape: Sequence[int],
    nb_levels: int,
    conv_size: int,
    name: str = 'conv_enc',
    feat_mult: int = 1,
    pool_size: int = 2,
    activation: str = 'relu',
    conv_dropout: float = 0.0,
    batch_norm: bool = False,
) -> Model:
    """Build a simple convolutional encoder."""

    inputs = KL.Input(shape=input_shape, name=f'{name}_input')
    ndims = _get_ndims(input_shape)
    x = inputs
    skips = []
    for level in range(nb_levels):
        filters = int(nb_features * (feat_mult ** level))
        x = _conv_block(x, filters, conv_size, activation=activation, dropout=conv_dropout, batch_norm=batch_norm)
        skips.append(x)
        x = _pool_layer(ndims)(pool_size=pool_size)(x)

    return Model(inputs, [x, *skips], name=name)


def conv_dec(
    nb_features: int,
    input_shape: Sequence[int],
    nb_levels: int,
    conv_size: int,
    name: str = 'conv_dec',
    feat_mult: int = 1,
    activation: str = 'relu',
    conv_dropout: float = 0.0,
    batch_norm: bool = False,
) -> Model:
    """Build a simple convolutional decoder mirroring ``conv_enc``."""

    inputs = [KL.Input(shape=s, name=f'{name}_input_{i}') for i, s in enumerate(input_shape)]
    x = inputs[0]
    ndims = len(x.shape) - 1
    skips = inputs[1:]

    for level, skip in enumerate(skips[::-1]):
        filters = int(nb_features * (feat_mult ** (nb_levels - level - 1)))
        x = _upsample_layer(ndims - 1)(size=2)(x)
        x = KL.Concatenate()([x, skip])
        x = _conv_block(x, filters, conv_size, activation=activation, dropout=conv_dropout, batch_norm=batch_norm)

    outputs = KL.Conv3D(nb_features, 1, padding='same', activation=activation)(x) if ndims == 4 else KL.Conv2D(nb_features, 1, padding='same', activation=activation)(x)
    return Model(inputs, outputs, name=name)


def unet(
    nb_features: int,
    input_shape: Sequence[int],
    nb_levels: int,
    conv_size: int,
    nb_labels: int,
    name: str = 'unet',
    feat_mult: int = 1,
    pool_size: int = 2,
    activation: str = 'relu',
    final_activation: str = 'softmax',
) -> Model:
    """Construct a compact U-Net style network."""

    inputs = KL.Input(shape=input_shape, name=f'{name}_input')
    ndims = _get_ndims(input_shape)
    x = inputs
    skips = []

    # Encoder
    for level in range(nb_levels):
        filters = int(nb_features * (feat_mult ** level))
        x = _conv_block(x, filters, conv_size, activation=activation)
        skips.append(x)
        x = _pool_layer(ndims)(pool_size=pool_size)(x)

    # Bottleneck
    x = _conv_block(x, filters * feat_mult, conv_size, activation=activation)

    # Decoder
    for level, skip in enumerate(skips[::-1]):
        filters = int(nb_features * (feat_mult ** (nb_levels - level - 1)))
        x = _upsample_layer(ndims)(size=pool_size)(x)
        x = KL.Concatenate()([x, skip])
        x = _conv_block(x, filters, conv_size, activation=activation)

    Conv = _conv_layer(ndims)
    outputs = Conv(nb_labels, 1, padding='same', activation=final_activation, name=f'{name}_output')(x)
    return Model(inputs, outputs, name=name)


def dilation_net(*args, **kwargs) -> Model:
    """Alias to ``unet`` for backwards compatibility."""

    return unet(*args, **kwargs)


# -----------------------------------------------------------------------------
# Label-to-image synthesis (simplified)
# -----------------------------------------------------------------------------


def labels_to_image_old(*args, **kwargs) -> Model:
    """Legacy alias for ``labels_to_image``."""

    return labels_to_image(*args, **kwargs)


def labels_to_image(
    labels_in: Iterable[int],
    in_shape: Optional[Sequence[int]] = None,
    num_chan: int = 1,
    input_model: Optional[Model] = None,
    normalize: bool = True,
    one_hot: bool = False,
    return_im: bool = True,
    return_map: bool = False,
    **_,
) -> Model:
    """Build a lightweight label-to-image generator.

    Only a subset of the original arguments is honoured at this stage of the migration. Unsupported
    options are silently ignored for now and will be reintroduced incrementally.
    """

    if input_model is None and in_shape is None:
        raise ValueError('Either `input_model` or `in_shape` must be provided.')

    if isinstance(labels_in, dict):
        unique_labels = sorted(set(labels_in.values()))
    else:
        unique_labels = sorted(set(labels_in))
    max_label = max(unique_labels)

    if input_model is None:
        inputs = KL.Input(shape=(*in_shape, 1), name='labels_input')
        model_inputs = inputs
    else:
        model_inputs = input_model.inputs
        inputs = input_model.outputs[0]

    image = nlayers.DrawImage(max_label=max_label, channels=num_chan, name='draw_image')(inputs)

    if normalize:
        def _minmax(x):
            xmin = ops.min(x, axis=None, keepdims=True)
            xmax = ops.max(x, axis=None, keepdims=True)
            return _safe_divide(x - xmin, xmax - xmin)
        image = KL.Lambda(_minmax, name='normalize')(image)

    outputs = []
    if return_im:
        outputs.append(image)

    if return_map:
        if one_hot:
            outputs.append(ops.one_hot(ops.squeeze(inputs, axis=-1), max_label + 1))
        else:
            outputs.append(inputs)

    if not outputs:
        outputs.append(image)

    return Model(model_inputs, outputs, name='labels_to_image')


# -----------------------------------------------------------------------------
# Placeholders for unported functionality
# -----------------------------------------------------------------------------


def labels_to_image_full(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError('The full label-to-image pipeline has not yet been ported to the torch backend.')


def conditional_template(*args, **kwargs):  # pragma: no cover - placeholder
    raise NotImplementedError('Template generation is not yet available on the torch backend.')


__all__ = [
    'conv_enc',
    'conv_dec',
    'unet',
    'dilation_net',
    'labels_to_image',
    'labels_to_image_old',
]
