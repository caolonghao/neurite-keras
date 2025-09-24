"""Torch-first utility helpers for neurite.

This module replaces the TensorFlow-heavy implementation that historically lived here with
lightweight PyTorch / NumPy equivalents. Only the subset of functions that are used by the
current migration path are implemented; the remaining utilities will be reintroduced as the
surrounding code is ported. Functions that were previously backed by TensorFlow have been
reimplemented to operate on PyTorch tensors while still accepting NumPy arrays for
convenience.

The goal is to supply numerically stable, differentiable primitives that mirror the public
API surface expected by the rest of neurite. As we continue the migration, additional helpers
will either be added here or replaced by specialised modules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from .._ops import (
    flatten_batch_channels as _flatten_batch_channels,
    safe_divide as _safe_divide,
    soft_quantize as _soft_quantize,
)

TensorLike = Union[torch.Tensor, np.ndarray]


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------


def _to_tensor(x: TensorLike, dtype: Optional[torch.dtype] = None, device: Optional[torch.device] = None) -> torch.Tensor:
    """Convert ``x`` to a torch tensor."""

    if isinstance(x, torch.Tensor):
        tensor = x
    else:
        tensor = torch.as_tensor(x)
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.to(dtype)
    if device is not None and tensor.device != device:
        tensor = tensor.to(device)
    return tensor


def _channels_last_to_first(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError('expected at least 3 dimensions (batch?, spatial..., channels)')
    return x.movedim(-1, 1)


def _channels_first_to_last(x: torch.Tensor) -> torch.Tensor:
    if x.ndim < 3:
        raise ValueError('expected at least 3 dimensions (batch, channels, spatial...)')
    return x.movedim(1, -1)


def _normalize_coords(coords: torch.Tensor, spatial_shape: Sequence[int]) -> torch.Tensor:
    """Map voxel coordinates to the range ``[-1, 1]`` as expected by ``grid_sample``."""

    norm = coords.clone()
    for dim, size in enumerate(spatial_shape):
        if size <= 1:
            norm[..., dim] = 0.0
        else:
            norm[..., dim] = 2.0 * norm[..., dim] / (size - 1) - 1.0
    return norm


def _resolve_padding(kernel_length: int, mode: str) -> int:
    if mode.lower() == 'same':
        return kernel_length // 2
    if mode.lower() == 'valid':
        return 0
    raise ValueError(f'Unsupported padding mode: {mode}')


# -----------------------------------------------------------------------------
# Public utilities
# -----------------------------------------------------------------------------


def setup_device(device: Optional[Union[str, int]] = None) -> Tuple[str, int]:
    """Return a device string and number of visible devices for torch."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if isinstance(device, int):
        device = f'cuda:{device}' if torch.cuda.is_available() else 'cpu'
    if device.startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    nb_devices = torch.cuda.device_count() if device.startswith('cuda') else 1
    return device, max(nb_devices, 1)


def volshape_to_meshgrid(vol_shape: Sequence[int], indexing: str = 'ij') -> np.ndarray:
    grids = [np.arange(d) for d in vol_shape]
    mesh = np.meshgrid(*grids, indexing=indexing)
    return np.stack(mesh, axis=-1)


def resize(vol: TensorLike, zoom_factor: Union[float, Sequence[float]], interp_method: str = 'linear') -> torch.Tensor:
    """Resize an N-D volume using torch interpolation."""

    tensor = _to_tensor(vol, dtype=torch.float32)
    spatial_rank = tensor.ndim - 1
    if spatial_rank <= 0:
        raise ValueError('Expected volume with a trailing channel dimension.')

    if isinstance(zoom_factor, (int, float)):
        zoom = [float(zoom_factor)] * spatial_rank
    else:
        zoom = [float(z) for z in zoom_factor]
    if len(zoom) != spatial_rank:
        raise ValueError('zoom_factor length does not match volume rank')

    spatial_shape = tensor.shape[:-1]
    new_shape = [max(1, int(round(s * z))) for s, z in zip(spatial_shape, zoom)]

    coords = [torch.linspace(0, s - 1, n, device=tensor.device, dtype=tensor.dtype)
              for s, n in zip(spatial_shape, new_shape)]
    grid = torch.meshgrid(*coords, indexing='ij')
    grid = torch.stack(grid, dim=-1)
    return interpn(tensor, grid, interp_method=interp_method)


def interpn(vol: TensorLike, loc: Union[TensorLike, Sequence[TensorLike]], interp_method: str = 'linear', fill_value: Optional[float] = None) -> torch.Tensor:
    """Multi-dimensional interpolation using ``torch.grid_sample``."""

    vol_t = _to_tensor(vol, dtype=torch.float32)

    if isinstance(loc, (list, tuple)):
        loc_t = torch.stack([_to_tensor(l, dtype=vol_t.dtype, device=vol_t.device) for l in loc], dim=-1)
    else:
        loc_t = _to_tensor(loc, dtype=vol_t.dtype, device=vol_t.device)

    if vol_t.ndim == loc_t.shape[-1]:
        vol_t = vol_t.unsqueeze(-1)
    if vol_t.ndim == loc_t.shape[-1] + 1:
        vol_t = vol_t.unsqueeze(0)
    if vol_t.ndim != loc_t.shape[-1] + 2:
        raise ValueError('Unsupported volume rank for interpolation')

    spatial_shape = vol_t.shape[1:-1]
    channels = vol_t.shape[-1]

    if loc_t.ndim == len(spatial_shape):
        loc_t = loc_t.unsqueeze(0)
    if loc_t.ndim == len(spatial_shape) + 1:
        loc_t = loc_t.unsqueeze(0)
    if loc_t.ndim != len(spatial_shape) + 2:
        raise ValueError('Unsupported location shape for interpolation')

    if loc_t.shape[0] != vol_t.shape[0]:
        if loc_t.shape[0] == 1:
            loc_t = loc_t.expand(vol_t.shape[0], *loc_t.shape[1:])
        else:
            raise ValueError('Batch dimension mismatch between volume and coordinates')

    grid = _normalize_coords(loc_t, spatial_shape)
    vol_cf = _channels_last_to_first(vol_t)

    mode_map = {'linear': 'bilinear', 'nearest': 'nearest'}
    mode = mode_map.get(interp_method, 'bilinear')
    align_corners = mode != 'nearest'
    padding_mode = 'zeros' if fill_value is not None else 'border'

    sampled = F.grid_sample(vol_cf, grid, mode=mode, align_corners=align_corners, padding_mode=padding_mode)
    if fill_value is not None:
        outside = ((grid < -1) | (grid > 1)).any(dim=-1, keepdim=True)
        sampled = sampled.masked_fill(outside.movedim(-1, 1), fill_value)

    out = _channels_first_to_last(sampled)
    if out.shape[0] == 1:
        out = out[0]
    if channels == 1:
        out = out[..., 0]
    return out


def barycenter(x: TensorLike, axes: Optional[Sequence[int]] = None, normalize: bool = False) -> torch.Tensor:
    tensor = _to_tensor(x, dtype=torch.float32)
    spatial_dims = tensor.ndim - 1
    if axes is None:
        axes = list(range(spatial_dims))

    coords = [torch.arange(tensor.shape[i], device=tensor.device, dtype=tensor.dtype) for i in range(spatial_dims)]
    grid = torch.meshgrid(*coords, indexing='ij')
    grid = torch.stack(grid, dim=-1)

    if normalize:
        grid = grid / torch.tensor([tensor.shape[i] for i in range(spatial_dims)], device=tensor.device, dtype=tensor.dtype)

    weights = tensor
    while weights.ndim < grid.ndim:
        weights = weights.unsqueeze(-1)
    numerator = torch.sum(grid * weights, dim=tuple(axes))
    denominator = torch.sum(weights, dim=tuple(axes))
    return _safe_divide(numerator, denominator)


def gaussian_kernel(
    sigma: Union[float, Sequence[float]],
    windowsize: Optional[Union[int, Sequence[int]]] = None,
    indexing: str = 'ij',
    separate: bool = False,
    random: bool = False,
    min_sigma: Union[float, Sequence[float]] = 0.0,
    dtype: torch.dtype = torch.float32,
    seed: Optional[int] = None,
):
    """Construct an N-D Gaussian kernel."""

    rng = np.random.default_rng(seed)
    sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
    min_sigma_arr = np.atleast_1d(np.asarray(min_sigma, dtype=np.float64))
    if min_sigma_arr.size == 1:
        min_sigma_arr = np.repeat(min_sigma_arr, sigma_arr.size)
    if sigma_arr.size != min_sigma_arr.size:
        raise ValueError('sigma and min_sigma must have the same length')

    if random:
        sigma_arr = np.array([rng.uniform(a, b) for a, b in zip(min_sigma_arr, sigma_arr)])
    sigma_arr = np.clip(sigma_arr, np.finfo(np.float64).eps, None)

    if windowsize is None:
        win = [int(round(s * 3) * 2 + 1) for s in sigma_arr]
    else:
        win = np.atleast_1d(np.asarray(windowsize, dtype=np.int32)).tolist()
    if len(win) == 1 and sigma_arr.size > 1:
        win = win * sigma_arr.size
    if len(win) != sigma_arr.size:
        raise ValueError('windowsize length mismatch')

    grids = []
    for size, sig in zip(win, sigma_arr):
        coords = torch.arange(size, dtype=dtype)
        coords = coords - (size - 1) / 2
        kernel = torch.exp(-0.5 * (coords / sig) ** 2)
        kernel = kernel / kernel.sum()
        grids.append(kernel)

    if separate:
        return grids

    meshes = torch.meshgrid(*grids, indexing=indexing)
    kernel_nd = torch.ones((), dtype=dtype)
    for mesh in meshes:
        kernel_nd = kernel_nd * mesh
    kernel_nd = kernel_nd / kernel_nd.sum()
    return kernel_nd


def separable_conv(
    x: TensorLike,
    kernels: Union[TensorLike, Sequence[TensorLike]],
    axis: Optional[Sequence[int]] = None,
    batched: bool = False,
    padding: str = 'SAME',
):
    """Apply 1-D separable convolution along given axes."""

    tensor = _to_tensor(x, dtype=torch.float32)
    if not batched:
        tensor = tensor.unsqueeze(0)

    spatial_rank = tensor.ndim - 2
    if spatial_rank not in (1, 2, 3):
        raise ValueError('Only supports 1D/2D/3D tensors')

    if axis is None:
        axes = list(range(spatial_rank))
    else:
        axes = [ax if ax >= 0 else spatial_rank + ax for ax in axis]

    if not isinstance(kernels, (list, tuple)):
        kernels = [kernels] * len(axes)
    if len(kernels) != len(axes):
        raise ValueError('Number of kernels must match number of axes')

    tensor_cf = _channels_last_to_first(tensor)
    batch, channels = tensor_cf.shape[:2]

    for ax, ker in zip(axes, kernels):
        kernel = _to_tensor(ker, dtype=tensor_cf.dtype, device=tensor_cf.device).reshape(-1)
        pad = _resolve_padding(int(kernel.shape[0]), padding)

        perm_order = [0, 1] + [i for i in range(2, tensor_cf.ndim) if i != ax + 2] + [ax + 2]
        tensor_perm = tensor_cf.permute(perm_order)
        flat = tensor_perm.reshape(batch * channels, -1, tensor_perm.shape[-1])
        kernel_view = kernel.view(1, 1, -1)
        conv = F.conv1d(flat, kernel_view, padding=pad)
        tensor_perm = conv.reshape(*tensor_perm.shape[:-1], conv.shape[-1])
        inv_perm = torch.argsort(torch.tensor(perm_order))
        tensor_cf = tensor_perm.permute(tuple(inv_perm.tolist()))

    out = _channels_first_to_last(tensor_cf)
    return out if batched else out[0]


def minmax_norm(x: TensorLike, axis: Optional[Sequence[int]] = None) -> torch.Tensor:
    tensor = _to_tensor(x, dtype=torch.float32)
    if axis is None:
        axis = list(range(tensor.ndim))
    tensor_min = tensor.amin(dim=axis, keepdim=True)
    tensor_max = tensor.amax(dim=axis, keepdim=True)
    return _safe_divide(tensor - tensor_min, tensor_max - tensor_min)


def perlin_vol(
    vol_shape: Sequence[int],
    min_scale: int = 0,
    max_scale: Optional[int] = None,
    wt_type: str = 'monotonic',
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate Perlin-like noise by summing random volumes at multiple scales."""

    if wt_type not in {'monotonic', 'random'}:
        raise ValueError("wt_type must be 'monotonic' or 'random'")

    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    vol_shape = tuple(int(s) for s in vol_shape)
    if max_scale is None:
        max_scale = int(np.ceil(np.log2(max(vol_shape))))

    weights = []
    noise_fields = []
    for scale in range(min_scale, max_scale + 1):
        down_shape = [max(1, int(math.ceil(s / (2 ** scale)))) for s in vol_shape]
        noise = torch.rand(*down_shape, device=device, dtype=dtype)
        if any(d != s for d, s in zip(down_shape, vol_shape)):
            zoom = [s / d for s, d in zip(vol_shape, down_shape)]
            noise = resize(noise[..., None], zoom_factor=zoom)[..., 0]
        noise_fields.append(noise)
        if wt_type == 'monotonic':
            weights.append(scale + 1)
        else:
            weights.append(np.random.rand())

    weights = torch.tensor(weights, device=device, dtype=dtype)
    weights = weights / weights.sum()
    stacked = torch.stack(noise_fields, dim=0)
    return torch.tensordot(weights, stacked, dims=([0], [0]))


# -----------------------------------------------------------------------------
# Keras op re-exports
# -----------------------------------------------------------------------------


def soft_quantize(*args, **kwargs):
    return _soft_quantize(*args, **kwargs)


def batch_channel_flatten(x):
    return _flatten_batch_channels(x)


def flatten_axes(x, axes):
    tensor = _to_tensor(x)
    axes = list(axes)
    if not axes:
        return tensor
    start, end = axes[0], axes[-1] + 1
    prefix = tensor.shape[:start]
    suffix = tensor.shape[end:]
    flattened = tensor.reshape(*prefix, -1, *suffix)
    return flattened


def fftn(x: TensorLike, axes: Optional[Sequence[int]] = None, inverse: bool = False) -> torch.Tensor:
    tensor = _to_tensor(x)
    if axes is None:
        axes = list(range(tensor.ndim))
    if inverse:
        return torch.fft.ifftn(tensor, dim=axes)
    return torch.fft.fftn(tensor, dim=axes)


def ifftn(x: TensorLike, axes: Optional[Sequence[int]] = None) -> torch.Tensor:
    return fftn(x, axes=axes, inverse=True)


def complex_to_channels(x: TensorLike) -> torch.Tensor:
    tensor = _to_tensor(x)
    return torch.stack([tensor.real, tensor.imag], dim=-1)


def channels_to_complex(x: TensorLike) -> torch.Tensor:
    tensor = _to_tensor(x)
    if tensor.shape[-1] != 2:
        raise ValueError('Expected last dimension to encode real/imag parts')
    return torch.complex(tensor[..., 0], tensor[..., 1])


def soft_delta(x: TensorLike, x0: float = 0.0, alpha: float = 100.0, reg: str = 'l1') -> torch.Tensor:
    tensor = _to_tensor(x, dtype=torch.float32)
    diff = torch.abs(tensor - x0) if reg == 'l1' else (tensor - x0) ** 2
    return (1.0 - torch.sigmoid(alpha * diff)) * 2.0


__all__ = [
    'setup_device',
    'interpn',
    'resize',
    'volshape_to_meshgrid',
    'barycenter',
    'gaussian_kernel',
    'separable_conv',
    'minmax_norm',
    'perlin_vol',
    'soft_quantize',
    'batch_channel_flatten',
    'flatten_axes',
    'fftn',
    'ifftn',
    'complex_to_channels',
    'channels_to_complex',
    'soft_delta',
]

