"""PyTorch-backed layer implementations used across neurite models.

This module provides a progressively expanding subset of the original TensorFlow
implementations. It focuses first on the layers that registration and
segmentation models depend on directly. Data-augmentation layers that have not
yet been ported are represented by placeholders that raise a clear
``NotImplementedError`` when instantiated.

Stages:
    1. Core geometric layers (`Resize`, `SpatialTransformer`, integration, ...)
       and lightweight arithmetic layers ported to the torch backend.
    2. Remaining augmentation utilities (e.g. Perlin noise, random clipping),
       to be migrated in follow-up iterations.
"""

from __future__ import annotations

import math
from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from keras import activations, initializers, ops
from keras.layers import Layer

from ._ops import safe_divide
from .py import utils as py_utils
from .utils import augment as augment_utils


TensorLike = Union[torch.Tensor, np.ndarray]


def _ensure_tensor(x: TensorLike, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x if dtype is None else x.to(dtype)
    return torch.as_tensor(x, dtype=dtype)


def _resolve_torch_dtype(dtype: Optional[Union[str, torch.dtype]]) -> torch.dtype:
    if dtype is None:
        return torch.float32
    if isinstance(dtype, torch.dtype):
        return dtype
    if isinstance(dtype, str):
        cand = getattr(torch, dtype, None)
        if isinstance(cand, torch.dtype):
            return cand
        cand = getattr(torch, dtype.lower(), None)
        if isinstance(cand, torch.dtype):
            return cand
    raise ValueError(f'Unsupported dtype specification: {dtype!r}')


def _channels_last_to_first(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError('expected at least 3 dimensions (batch, spatial..., channels)')
    return x.movedim(-1, 1)


def _channels_first_to_last(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError('expected at least 3 dimensions (batch, channels, spatial...)')
    return x.movedim(1, -1)


def _spatial_shape(x: torch.Tensor) -> Tuple[int, ...]:
    return tuple(int(d) for d in x.shape[1:-1])


def _compute_zoomed_shape(shape: Sequence[int], zoom: Sequence[float]) -> Tuple[int, ...]:
    return tuple(int(round(s * z)) for s, z in zip(shape, zoom))


def _default_interp_mode(ndims: int, method: str) -> str:
    if method == 'nearest':
        return 'nearest'
    if ndims == 1:
        return 'linear'
    if ndims == 2:
        return 'bilinear'
    if ndims == 3:
        return 'trilinear'
    raise ValueError(f'unsupported ndims {ndims} for interpolation')


class Negate(Layer):
    def call(self, inputs):
        return -inputs


class RescaleValues(Layer):
    def __init__(self, resize: float, **kwargs):
        self.resize = float(resize)
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs * self.resize


class Resize(Layer):
    """Resize an N-D volume using ``torch.nn.functional.interpolate``."""

    def __init__(self, zoom_factor: Union[float, Sequence[float]], interp_method: str = 'linear', **kwargs):
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self._zoom: Optional[Tuple[float, ...]] = None
        self._ndims: Optional[int] = None
        super().__init__(**kwargs)

    def build(self, input_shape):
        if isinstance(input_shape, (list, tuple)) and len(input_shape) == 1:
            input_shape = input_shape[0]
        if len(input_shape) < 3:
            raise ValueError('resize expects inputs of shape (batch, spatial..., channels)')

        self._ndims = len(input_shape) - 2
        if isinstance(self.zoom_factor, (int, float)):
            self._zoom = tuple([float(self.zoom_factor)] * self._ndims)
        else:
            zoom = tuple(float(z) for z in self.zoom_factor)
            if len(zoom) != self._ndims:
                raise ValueError('zoom factor length does not match spatial rank')
            self._zoom = zoom
        super().build(input_shape)

    def call(self, inputs):
        x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
        if self._ndims is None or self._zoom is None:
            raise RuntimeError('Resize layer must be built before call')

        x_t = _ensure_tensor(x)
        spatial_in = _spatial_shape(x_t)
        spatial_out = _compute_zoomed_shape(spatial_in, self._zoom)

        mode = _default_interp_mode(self._ndims, self.interp_method)
        align_corners = mode != 'nearest'

        x_t = _channels_last_to_first(x_t)
        out = F.interpolate(x_t, size=spatial_out, mode=mode, align_corners=align_corners)
        return _channels_first_to_last(out)


class LocalBias(Layer):
    def __init__(self, initializer: str = 'zeros', biasmult: float = 1.0, **kwargs):
        self.initializer = initializer
        self.biasmult = float(biasmult)
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel', shape=input_shape[1:], initializer=self.initializer, trainable=True
        )
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.kernel * self.biasmult


class LocalParamWithInput(Layer):
    """Layer holding a learnable tensor broadcast across the batch dimension."""

    def __init__(self, shape, initializer='random_normal', mult: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.param_shape = tuple(int(s) for s in shape)
        self.initializer = initializer
        self.mult = float(mult)

    def build(self, input_shape):
        init = initializers.get(self.initializer)
        self.kernel = self.add_weight(
            name='kernel',
            shape=self.param_shape,
            initializer=init,
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        inputs = inputs if not isinstance(inputs, (list, tuple)) else inputs[0]
        batch = ops.shape(inputs)[0]
        params = ops.reshape(self.kernel * self.mult, (1, -1))
        ones = ops.ones((batch, 1), dtype=params.dtype)
        tiled = ops.matmul(ones, params)
        output_shape = (batch, *self.param_shape)
        return ops.reshape(tiled, output_shape)


class Constant(Layer):
    """Layer returning a constant value with a dynamic batch dimension."""

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self._value = np.array(value)
        if self._value.ndim == 0:
            self._value = np.reshape(self._value, (1,))
        self._value_shape = tuple(int(s) for s in self._value.shape)

    def build(self, _):
        initializer = initializers.Constant(self._value)
        self.const = self.add_weight(
            name='constant',
            shape=self._value_shape,
            initializer=initializer,
            trainable=False,
        )
        super().build(None)

    def call(self, inputs=None):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0] if inputs else None
        batch = ops.shape(inputs)[0] if inputs is not None else ops.convert_to_tensor(1, dtype='int32')
        batch = ops.cast(batch, 'int32')
        const = ops.expand_dims(self.const, axis=0)
        const_shape = ops.shape(self.const)
        target_shape = ops.concatenate([ops.expand_dims(batch, axis=0), const_shape], axis=0)
        return ops.broadcast_to(const, target_shape)


class MeanStream(Layer):
    """Maintain a streaming estimate of the input mean up to a capped sample size."""

    def __init__(self, cap: float = 100.0, **kwargs):
        super().__init__(**kwargs)
        self.cap = float(cap)

    def build(self, input_shape):
        self.mean = self.add_weight(
            name='stream_mean',
            shape=input_shape[1:],
            initializer='zeros',
            trainable=False,
        )
        self.count = self.add_weight(
            name='stream_count',
            shape=(),
            initializer='zeros',
            trainable=False,
        )
        super().build(input_shape)

    def call(self, inputs, training=None):
        batch = ops.cast(ops.shape(inputs)[0], self.count.dtype)
        cap = ops.convert_to_tensor(self.cap, dtype=self.count.dtype)

        scalar_one = ops.convert_to_tensor(1.0, dtype=self.count.dtype)

        if training is False:
            weight = ops.minimum(scalar_one, self.count / cap)
            weight = ops.cast(weight, self.mean.dtype)
            return ops.expand_dims(self.mean, axis=0) * weight

        batch_mean = ops.mean(inputs, axis=0)
        new_count = ops.minimum(self.count + batch, cap)
        total = ops.maximum(new_count, scalar_one)
        old_weight = ops.minimum(self.count, cap)
        new_mean = safe_divide(self.mean * old_weight + batch_mean * batch, total)

        self.mean.assign(new_mean)
        self.count.assign(new_count)

        weight = ops.cast(ops.minimum(scalar_one, new_count / cap), new_mean.dtype)
        return ops.expand_dims(new_mean, axis=0) * weight


class SampleNormalLogVar(Layer):
    """Reparameterization trick sampling from N(mu, exp(log_var))."""

    def call(self, inputs):
        mu, log_var = inputs
        mu_t = _ensure_tensor(mu)
        log_var_t = _ensure_tensor(log_var, dtype=mu_t.dtype)
        noise = torch.randn_like(mu_t)
        return mu_t + torch.exp(0.5 * log_var_t) * noise


class GaussianNoise(Layer):
    def __init__(self, noise_min: float = 0.01, noise_max: float = 0.1, noise_only: bool = False, **kwargs):
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.noise_only = noise_only
        super().__init__(**kwargs)

    def call(self, inputs, training: Optional[bool] = None):
        if training is False or (training is None and not self.trainable):
            return inputs

        x = _ensure_tensor(inputs)
        if self.noise_max == 0.0 and not self.noise_only:
            return inputs

        std = torch.empty(x.shape[0], *(1 for _ in range(x.ndim - 1)), device=x.device, dtype=x.dtype)
        std.uniform_(self.noise_min, self.noise_max)
        noise = torch.randn_like(x) * std
        return noise if self.noise_only else x + noise


class GaussianBlur(Layer):
    """Apply isotropic Gaussian blur via separable convolutions."""

    def __init__(self, sigma: float, kernel_size: Optional[int] = None, **kwargs):
        if sigma <= 0:
            raise ValueError('sigma must be positive')
        self.sigma = float(sigma)
        self.kernel_size = kernel_size
        super().__init__(**kwargs)

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        if ndims not in (1, 2, 3):
            raise ValueError('GaussianBlur supports 1D, 2D, or 3D inputs')

        size = self.kernel_size
        if size is None:
            radius = max(int(math.ceil(self.sigma * 3)), 1)
            size = 2 * radius + 1

        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2
        kernel_1d = torch.exp(-0.5 * (coords / self.sigma) ** 2)
        kernel_1d /= kernel_1d.sum()

        kernels = []
        eye = torch.zeros((ndims, size, *([1] * (ndims - 1))), dtype=torch.float32)
        for i in range(ndims):
            view_shape = [1] * ndims
            view_shape[i] = size
            kernels.append(kernel_1d.view(*view_shape))

        kernel = kernels[0]
        for extra in kernels[1:]:
            kernel = kernel * extra

        kernel = kernel.reshape(1, 1, *kernel.shape)
        self.register_buffer('kernel', kernel)
        self._ndims = ndims
        super().build(input_shape)

    def call(self, inputs):
        x = _ensure_tensor(inputs)
        x_c = _channels_last_to_first(x)
        channels = x_c.shape[1]
        kernel = self.kernel.repeat(channels, 1, *[1] * self._ndims)

        if self._ndims == 1:
            out = F.conv1d(x_c, kernel, padding='same', groups=channels)
        elif self._ndims == 2:
            out = F.conv2d(x_c, kernel, padding='same', groups=channels)
        else:
            out = F.conv3d(x_c, kernel, padding='same', groups=channels)

        return _channels_first_to_last(out)


class RandomClip(Layer):
    """Clip tensor values to randomly drawn bounds (simplified implementation)."""

    def __init__(self, low: float = 0.0, high: float = 1.0, **kwargs):
        self.low = float(low)
        self.high = float(high)
        super().__init__(**kwargs)

    def call(self, inputs):
        x = _ensure_tensor(inputs)
        lower = torch.empty(x.shape[0], *(1 for _ in range(x.ndim - 1)), device=x.device, dtype=x.dtype)
        upper = torch.empty_like(lower)
        lower.uniform_(self.low, self.high * 0.5)
        upper.uniform_(self.high * 0.5, self.high)
        return torch.clamp(x, min=lower, max=upper)


class RandomGamma(Layer):
    def __init__(self, low: float = 0.9, high: float = 1.1, **kwargs):
        self.low = float(low)
        self.high = float(high)
        super().__init__(**kwargs)

    def call(self, inputs):
        x = _ensure_tensor(inputs)
        gamma = torch.empty(x.shape[0], *(1 for _ in range(x.ndim - 1)), device=x.device, dtype=x.dtype)
        gamma.uniform_(self.low, self.high)
        gamma = torch.clamp(gamma, min=1e-6)
        return torch.pow(torch.clamp(x, min=1e-6), gamma)


class RandomCrop(Layer):
    def __init__(self,
                 crop_min: float = 0.0,
                 crop_max: float = 0.5,
                 axis: Optional[Union[int, Sequence[int]]] = None,
                 prob: float = 1.0,
                 bilateral: bool = False,
                 seed: Optional[int] = None,
                 **kwargs):
        self.crop_min = float(crop_min)
        self.crop_max = float(crop_max)
        self.axis = axis
        self.prob = float(prob)
        self.bilateral = bilateral
        self.seed = seed
        super().__init__(**kwargs)

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        axes = self.axis
        if axes is None:
            axes = list(range(1, ndims + 1))
        elif isinstance(axes, int):
            axes = [axes]
        self._axes = [ax if ax >= 0 else ndims + ax + 1 for ax in axes]
        if self.seed is not None:
            torch.manual_seed(self.seed)
        super().build(input_shape)

    def call(self, inputs):
        x = _ensure_tensor(inputs)
        if self.prob <= 0 or self.crop_max <= 0:
            return x

        if torch.rand(1, device=x.device) > self.prob:
            return x

        axis = self._axes[torch.randint(len(self._axes), (1,), device=x.device).item()]
        spatial_axis = axis
        size = x.shape[spatial_axis]

        if size <= 1:
            return x

        crop_prop = torch.rand(1, device=x.device) * (self.crop_max - self.crop_min) + self.crop_min
        crop_vox = int(round(float(crop_prop.item()) * size))
        crop_vox = max(0, min(crop_vox, size - 1))
        if crop_vox == 0:
            return x

        mask = torch.ones_like(x)
        if self.bilateral:
            left = torch.randint(0, crop_vox + 1, (1,), device=x.device).item()
            right = crop_vox - left
            if left > 0:
                slices = [slice(None)] * x.ndim
                slices[spatial_axis] = slice(0, left)
                mask[tuple(slices)] = 0
            if right > 0:
                slices = [slice(None)] * x.ndim
                slices[spatial_axis] = slice(size - right, size)
                mask[tuple(slices)] = 0
        else:
            start = torch.randint(0, size - crop_vox + 1, (1,), device=x.device).item()
            slices = [slice(None)] * x.ndim
            slices[spatial_axis] = slice(start, start + crop_vox)
            mask[tuple(slices)] = 0

        return x * mask


class RandomClearLabel(Layer):
    def __init__(self, prob: float = 0.5, clear: Union[int, Sequence[int]] = 0, **kwargs):
        self.prob = float(prob)
        self.clear = clear
        super().__init__(**kwargs)

    def call(self, inputs):
        image, labels = inputs
        if self.prob <= 0:
            return image

        image_t = _ensure_tensor(image)
        labels_t = _ensure_tensor(labels)
        bg = torch.tensor([self.clear] if np.isscalar(self.clear) else np.unique(self.clear), device=labels_t.device)

        rand = torch.rand((image_t.shape[0], *(1 for _ in range(image_t.ndim - 1))), device=image_t.device)
        mask = rand < self.prob
        bg_mask = (labels_t[..., None] == bg).any(dim=-1)
        mask = mask & bg_mask
        mask = mask.unsqueeze(-1)
        return image_t * (~mask).to(image_t.dtype)


class PerlinNoise(Layer):
    """Generate Perlin-like noise volumes using the torch-backed utilities."""

    def __init__(
        self,
        shape: Optional[Sequence[int]] = None,
        noise_min: float = 0.01,
        noise_max: float = 1.0,
        fwhm_min: Union[float, Sequence[float]] = 4.0,
        fwhm_max: Union[float, Sequence[float]] = 32.0,
        isotropic: bool = False,
        reduce: Callable[[torch.Tensor], torch.Tensor] = torch.std,
        out_dtype: Optional[Union[str, torch.dtype]] = torch.float32,
        axes: Optional[Sequence[int]] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        self.shape = tuple(int(s) for s in shape) if shape is not None else None
        self.noise_min = float(noise_min)
        self.noise_max = float(noise_max)
        self.fwhm_min = fwhm_min
        self.fwhm_max = fwhm_max
        self.isotropic = bool(isotropic)
        self.reduce = reduce
        self.out_dtype = _resolve_torch_dtype(out_dtype)
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        dtype_name = str(self.out_dtype).split('.')[-1]
        config.update(
            {
                'shape': self.shape,
                'noise_min': self.noise_min,
                'noise_max': self.noise_max,
                'fwhm_min': self.fwhm_min,
                'fwhm_max': self.fwhm_max,
                'isotropic': self.isotropic,
                'reduce': self.reduce,
                'out_dtype': dtype_name,
                'axes': self.axes,
                'seed': self.seed,
            }
        )
        return config

    def build(self, input_shape):
        if len(input_shape) < 2:
            raise ValueError('PerlinNoise expects at least a batch and feature dimension.')
        allowed = range(1, len(input_shape))
        self.axes = tuple(py_utils.normalize_axes(self.axes, input_shape, allowed=allowed, none_means_all=False))
        self._rng = np.random.default_rng(self.seed)
        super().build(input_shape)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        batch = inputs.shape[0]
        if batch is None:
            batch = int(ops.shape(inputs)[0])
        batch = int(batch)

        if self.shape is None:
            inferred = tuple(inputs.shape[1:])
        else:
            inferred = self.shape

        if any(dim is None for dim in inferred):
            raise ValueError('PerlinNoise requires concrete spatial dimensions when using the torch backend.')
        target_shape = tuple(int(dim) for dim in inferred)

        axes = [ax - 1 for ax in self.axes]

        samples = []
        for _ in range(batch):
            sample = augment_utils.draw_perlin_full(
                target_shape,
                noise_min=self.noise_min,
                noise_max=self.noise_max,
                fwhm_min=self.fwhm_min,
                fwhm_max=self.fwhm_max,
                isotropic=self.isotropic,
                batched=False,
                featured=True,
                reduce=self.reduce,
                dtype=self.out_dtype,
                axes=axes,
                seed=int(self._rng.integers(np.iinfo(np.int64).max)),
            )
            samples.append(sample)

        noise = torch.stack(samples, dim=0)
        return noise.to(device=_ensure_tensor(inputs).device, dtype=self.out_dtype)


class DrawImage(Layer):
    def __init__(self, max_label: int, low: float = 0.0, high: float = 1.0, channels: int = 1, **kwargs):
        self.max_label = int(max_label)
        self.low = float(low)
        self.high = float(high)
        self.channels = int(channels)
        super().__init__(**kwargs)

    def call(self, inputs):
        labels = _ensure_tensor(inputs, dtype=torch.long)
        batch = labels.shape[0]
        num_labels = self.max_label + 1
        intensities = torch.empty((batch, self.channels, num_labels), device=labels.device, dtype=torch.float32)
        intensities.uniform_(self.low, self.high)

        flat = labels.view(batch, -1)
        gathered = torch.gather(intensities, 2, flat.unsqueeze(1))
        gathered = gathered.view(batch, self.channels, *labels.shape[1:])
        return gathered.movedim(1, -1)


class DrawAffineParams(Layer):
    """Sample affine parameters (shift, rotation, scale, shear) for augmentation."""

    def __init__(
        self,
        shift: Optional[Union[float, Sequence[float]]] = None,
        rot: Optional[Union[float, Sequence[float]]] = None,
        scale: Optional[Union[float, Sequence[float]]] = None,
        shear: Optional[Union[float, Sequence[float]]] = None,
        normal_shift: bool = False,
        normal_rot: bool = False,
        normal_scale: bool = False,
        normal_shear: bool = False,
        shift_scale: bool = False,
        ndims: int = 3,
        concat: bool = True,
        out_dtype: Optional[Union[str, torch.dtype]] = torch.float32,
        seeds: Optional[dict] = None,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if ndims not in (2, 3):
            raise ValueError('DrawAffineParams currently supports 2D or 3D transforms.')
        self.shift = shift
        self.rot = rot
        self.scale = scale
        self.shear = shear
        self.normal_shift = normal_shift
        self.normal_rot = normal_rot
        self.normal_scale = normal_scale
        self.normal_shear = normal_shear
        self.shift_scale = shift_scale
        self.ndims = int(ndims)
        self.concat = concat
        self.out_dtype = _resolve_torch_dtype(out_dtype)
        self.seeds = seeds.copy() if seeds is not None else {}
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        dtype_name = str(self.out_dtype).split('.')[-1]
        config.update(
            {
                'shift': self.shift,
                'rot': self.rot,
                'scale': self.scale,
                'shear': self.shear,
                'normal_shift': self.normal_shift,
                'normal_rot': self.normal_rot,
                'normal_scale': self.normal_scale,
                'normal_shear': self.normal_shear,
                'shift_scale': self.shift_scale,
                'ndims': self.ndims,
                'concat': self.concat,
                'out_dtype': dtype_name,
                'seeds': self.seeds,
                'seed': self.seed,
            }
        )
        return config

    def build(self, _):
        self._base_rng = np.random.default_rng(self.seed)
        super().build(None)

    def call(self, inputs):
        if isinstance(inputs, (list, tuple)):
            inputs = inputs[0]

        device = _ensure_tensor(inputs).device
        batch = inputs.shape[0]
        if batch is None:
            batch = int(ops.shape(inputs)[0])
        batch = int(batch)

        group_dims = dict(shift=self.ndims, rot=(1 if self.ndims == 2 else 3), scale=self.ndims, shear=(1 if self.ndims == 2 else 3))
        ranges = {
            'shift': self._normalize_range(self.shift, group_dims['shift'], 'shift'),
            'rot': self._normalize_range(self.rot, group_dims['rot'], 'rot'),
            'scale': self._normalize_range(self.scale, group_dims['scale'], 'scale'),
            'shear': self._normalize_range(self.shear, group_dims['shear'], 'shear'),
        }
        normals = {
            'shift': self.normal_shift,
            'rot': self.normal_rot,
            'scale': self.normal_scale,
            'shear': self.normal_shear,
        }
        trunc = {'shift': False, 'rot': False, 'scale': True, 'shear': False}

        outputs = {}
        for key, lims in ranges.items():
            shape = (batch, lims.size)
            rng = self._rng_for(key)
            outputs[key] = self._sample(lims, shape, normals[key], trunc[key], rng)

        if self.shift_scale:
            outputs['scale'] = outputs['scale'] + 1.0

        tensors = {k: torch.as_tensor(v, device=device, dtype=self.out_dtype) for k, v in outputs.items()}
        if self.concat:
            ordered = [tensors['shift'], tensors['rot'], tensors['scale'], tensors['shear']]
            return torch.cat(ordered, dim=-1)
        return tensors['shift'], tensors['rot'], tensors['scale'], tensors['shear']

    def _rng_for(self, key: str) -> np.random.Generator:
        seed = self.seeds.get(key)
        if seed is not None:
            return np.random.default_rng(seed)
        return self._base_rng

    @staticmethod
    def _normalize_range(value, expected: int, name: str) -> np.ndarray:
        if value is None:
            arr = np.zeros(expected, dtype=np.float32)
        else:
            arr = np.asarray(value, dtype=np.float32).ravel()
            if arr.size == 1:
                arr = np.repeat(arr, expected)
            if arr.size != expected:
                raise ValueError(f'{name} expects {expected} values, got {arr.size}')
        return arr

    @staticmethod
    def _sample(lims: np.ndarray, shape: Tuple[int, int], normal: bool, truncate: bool, rng: np.random.Generator) -> np.ndarray:
        if shape[0] == 0:
            return np.zeros(shape, dtype=np.float32)
        lims = lims.astype(np.float32)
        expand = (1,) * (len(shape) - 1) + (lims.size,)
        lims_view = lims.reshape(expand)
        if normal:
            samples = rng.normal(loc=0.0, scale=1.0, size=shape).astype(np.float32) * lims_view
            if truncate:
                limit = 2.0 * lims_view
                samples = np.clip(samples, -limit, limit)
        else:
            samples = (rng.uniform(-1.0, 1.0, size=shape).astype(np.float32)) * lims_view
        return samples


class DownUpSample(Layer):
    """Perform average pooling followed by interpolation back to the original size."""

    def __init__(self, downsample: int = 2, interp_method: str = 'linear', **kwargs):
        self.downsample = int(downsample)
        self.interp_method = interp_method
        super().__init__(**kwargs)

    def call(self, inputs):
        x = _ensure_tensor(inputs)
        x_c = _channels_last_to_first(x)
        ndims = x_c.ndim - 2
        if ndims == 2:
            pooled = F.avg_pool2d(x_c, kernel_size=self.downsample, stride=self.downsample, padding=0)
        elif ndims == 3:
            pooled = F.avg_pool3d(x_c, kernel_size=self.downsample, stride=self.downsample, padding=0)
        else:
            raise ValueError('DownUpSample currently supports 2D or 3D tensors')

        size = x_c.shape[2:]
        mode = _default_interp_mode(ndims, self.interp_method)
        out = F.interpolate(pooled, size=size, mode=mode, align_corners=mode != 'nearest')
        return _channels_first_to_last(out)


class RescaleTransform(Layer):
    def __init__(self, zoom_factor: float, **kwargs):
        self.zoom_factor = float(zoom_factor)
        super().__init__(**kwargs)

    def call(self, inputs):
        return inputs * self.zoom_factor


def _create_meshgrid(shape: Sequence[int], device: torch.device, dtype: torch.dtype, indexing: str = 'ij') -> torch.Tensor:
    ranges = [torch.linspace(-1.0, 1.0, steps=s, device=device, dtype=dtype) for s in shape]
    grid = torch.meshgrid(*ranges, indexing=indexing)
    return torch.stack(grid, dim=-1)


def _create_voxel_meshgrid(shape: Sequence[int], device: torch.device, dtype: torch.dtype, indexing: str = 'ij') -> torch.Tensor:
    ranges = [torch.linspace(0.0, float(s - 1), steps=s, device=device, dtype=dtype) for s in shape]
    grid = torch.meshgrid(*ranges, indexing=indexing)
    return torch.stack(grid, dim=-1)


class SpatialTransformer(Layer):
    """Warp a moving image/tensor with a dense displacement field."""

    def __init__(self, interp_method: str = 'linear', indexing: str = 'ij', **kwargs):
        self.interp_method = interp_method
        self.indexing = indexing
        super().__init__(**kwargs)

    def call(self, inputs):
        if not isinstance(inputs, (list, tuple)) or len(inputs) != 2:
            raise ValueError('SpatialTransformer expects [moving, flow] inputs')
        moving, flow = inputs

        moving_t = _ensure_tensor(moving)
        flow_t = _ensure_tensor(flow, dtype=moving_t.dtype)

        spatial_shape = flow_t.shape[1:-1]
        ndims = len(spatial_shape)
        if flow_t.shape[-1] != ndims:
            raise ValueError('flow last dimension must match spatial rank')

        # Normalized base grid in [-1, 1].
        grid = _create_meshgrid(spatial_shape, device=flow_t.device, dtype=flow_t.dtype, indexing=self.indexing)
        grid = grid.unsqueeze(0)

        # Convert displacement from voxel units to normalized coordinates.
        scale = []
        for dim in spatial_shape:
            if dim <= 1:
                scale.append(0.0)
            else:
                scale.append(2.0 / (dim - 1))
        scale = torch.tensor(scale, device=flow_t.device, dtype=flow_t.dtype)
        flow_norm = flow_t * scale

        sampling_grid = grid + flow_norm
        mode = _default_interp_mode(ndims, self.interp_method)
        align_corners = mode != 'nearest'

        moving_cf = _channels_last_to_first(moving_t)
        warped = F.grid_sample(
            moving_cf,
            sampling_grid,
            mode=mode,
            padding_mode='border',
            align_corners=align_corners,
        )
        return _channels_first_to_last(warped)


class ComposeTransform(Layer):
    """Compose multiple displacement fields via successive warps."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._transformer = SpatialTransformer()

    def call(self, inputs):
        if len(inputs) < 2:
            raise ValueError('ComposeTransform needs at least two fields to compose')

        composed = _ensure_tensor(inputs[0])
        for flow in inputs[1:]:
            flow_t = _ensure_tensor(flow, dtype=composed.dtype)
            composed = composed + self._transformer([flow_t, composed])
        return composed


class ParamsToAffineMatrix(Layer):
    """Convert flattened affine parameters to dense matrices.

    Expects parameter tensors of shape ``(batch, D * (D + 1))`` representing
    row-major affine matrices ``[A | b]`` where ``A`` is ``D x D`` and ``b`` is
    a translation vector.
    """

    def __init__(self, ndims: int, **kwargs):
        self.ndims = ndims
        super().__init__(**kwargs)

    def call(self, inputs):
        params = _ensure_tensor(inputs)
        expected = self.ndims * (self.ndims + 1)
        if params.shape[-1] != expected:
            raise ValueError(f'expected last dimension {expected}, got {params.shape[-1]}')

        affine = params.view(params.shape[0], self.ndims, self.ndims + 1)
        return affine


class AffineToDenseShift(Layer):
    """Convert affine matrices to dense displacement fields."""

    def __init__(self, output_shape: Sequence[int], shift_center: bool = True, **kwargs):
        self.output_shape = tuple(int(s) for s in output_shape)
        self.shift_center = shift_center
        super().__init__(**kwargs)

    def call(self, inputs):
        affine = _ensure_tensor(inputs)
        batch = affine.shape[0]
        ndims = affine.shape[1]
        spatial = self.output_shape

        grid = _create_voxel_meshgrid(spatial, affine.device, affine.dtype, indexing='ij')
        if self.shift_center:
            shifts = [(dim - 1) / 2.0 for dim in spatial]
            shift = torch.tensor(shifts, device=affine.device, dtype=affine.dtype)
            grid = grid - shift

        grid = grid.reshape(1, *spatial, ndims)
        grid = grid.repeat(batch, *(1 for _ in spatial))

        A = affine[..., :ndims]
        b = affine[..., ndims:]
        grid_flat = grid.reshape(batch, -1, ndims)
        transformed = torch.matmul(grid_flat, A.transpose(-1, -2)) + b.transpose(-1, -2)
        flow = transformed - grid_flat
        return flow.view(batch, *spatial, ndims)


class RescaleValuesTransform(Layer):  # pragma: no cover - compatibility alias
    def __init__(self, scale: float, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale

    def call(self, inputs):
        return inputs * self.scale


class VecInt(Layer):
    """Scaling and squaring integration of a velocity field."""

    def __init__(self, int_steps: int, **kwargs):
        self.int_steps = int(int_steps)
        self.transformer = SpatialTransformer()
        super().__init__(**kwargs)

    def call(self, inputs):
        flow = _ensure_tensor(inputs)
        if self.int_steps <= 0:
            return flow

        flow = flow / (2 ** self.int_steps)
        disp = flow
        for _ in range(self.int_steps):
            disp = disp + self.transformer([disp, disp])
        return disp


class HyperConvFromDense(Layer):
    """Lightweight hyper-convolution layer driven by dense projections."""

    def __init__(
        self,
        rank: int,
        filters: int,
        kernel_size,
        strides=1,
        padding='same',
        use_bias=True,
        hyperkernel_use_bias=True,
        hyperbias_use_bias=True,
        hyperkernel_activation=None,
        hyperbias_activation=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rank = int(rank)
        self.filters = int(filters)
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * self.rank
        self.kernel_size = tuple(int(k) for k in kernel_size)
        if isinstance(strides, int):
            strides = (strides,) * self.rank
        self.strides = tuple(int(s) for s in strides)
        self.padding = padding.upper()
        self.padding_mode = padding.lower()
        self.use_bias = use_bias
        self.hyperkernel_use_bias = hyperkernel_use_bias
        self.hyperbias_use_bias = hyperbias_use_bias
        self.hyperkernel_activation = activations.get(hyperkernel_activation)
        self.hyperbias_activation = activations.get(hyperbias_activation)

    def build(self, input_shape):
        feature_shape, latent_shape = input_shape
        self.in_channels = int(feature_shape[-1])
        latent_dim = int(latent_shape[-1])
        kernel_elems = np.prod(self.kernel_size) * self.in_channels * self.filters

        self.kernel_w = self.add_weight(
            name='hyperkernel_kernel',
            shape=(latent_dim, kernel_elems),
            initializer='glorot_uniform',
            trainable=True,
        )
        if self.hyperkernel_use_bias:
            self.kernel_b = self.add_weight(
                name='hyperkernel_bias',
                shape=(kernel_elems,),
                initializer='zeros',
                trainable=True,
            )
        else:
            self.kernel_b = None

        if self.use_bias:
            self.bias_w = self.add_weight(
                name='hyperbias_kernel',
                shape=(latent_dim, self.filters),
                initializer='glorot_uniform',
                trainable=True,
            )
            if self.hyperbias_use_bias:
                self.bias_b = self.add_weight(
                    name='hyperbias_bias',
                    shape=(self.filters,),
                    initializer='zeros',
                    trainable=True,
                )
            else:
                self.bias_b = None
        else:
            self.bias_w = None
            self.bias_b = None

        super().build(input_shape)

    def call(self, inputs):
        features, latent = inputs
        batch = ops.shape(features)[0]
        latent = ops.reshape(latent, (batch, -1))

        kernel_flat = ops.matmul(latent, self.kernel_w)
        if self.kernel_b is not None:
            kernel_flat = kernel_flat + self.kernel_b
        if self.hyperkernel_activation is not None:
            kernel_flat = self.hyperkernel_activation(kernel_flat)

        kernels = ops.reshape(kernel_flat, (-1, *self.kernel_size, self.in_channels, self.filters))

        if self.use_bias:
            bias = ops.matmul(latent, self.bias_w)
            if self.bias_b is not None:
                bias = bias + self.bias_b
            if self.hyperbias_activation is not None:
                bias = self.hyperbias_activation(bias)
        else:
            bias = None

        kernel = kernels[0]
        result = ops.conv(features, kernel, strides=self.strides, padding=self.padding_mode, data_format='channels_last')
        if bias is not None:
            bias_term = ops.reshape(bias[0], (1,) + (1,) * self.rank + (self.filters,))
            result = result + bias_term
        return result


__all__ = [
    'Negate',
    'RescaleValues',
    'Resize',
    'LocalBias',
    'LocalParamWithInput',
    'Constant',
    'MeanStream',
    'SampleNormalLogVar',
    'GaussianNoise',
    'GaussianBlur',
    'RandomClip',
    'RandomGamma',
    'RandomCrop',
    'RandomClearLabel',
    'PerlinNoise',
    'DrawImage',
    'DrawAffineParams',
    'DownUpSample',
    'RescaleTransform',
    'SpatialTransformer',
    'ComposeTransform',
    'ParamsToAffineMatrix',
    'AffineToDenseShift',
    'VecInt',
    'RescaleValuesTransform',
    'HyperConvFromDense',
]
