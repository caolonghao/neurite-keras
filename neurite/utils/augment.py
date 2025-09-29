"""Torch-based augmentation utilities for neurite."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch

import neurite as ne


def _next_generator(rng: np.random.Generator, device: torch.device) -> torch.Generator:
    """Return a ``torch.Generator`` seeded from the provided NumPy RNG."""

    gen = torch.Generator(device=device)
    gen.manual_seed(int(rng.integers(np.iinfo(np.int64).max)))
    return gen


def draw_perlin(out_shape,
                scales,
                min_std=0,
                max_std=1,
                dtype=torch.float32,
                seed=None):
    out_shape = np.asarray(out_shape, dtype=np.int64)
    if np.isscalar(scales):
        scales = [scales]

    rng = np.random.default_rng(seed)
    device = torch.device('cpu')
    out = torch.zeros(tuple(out_shape.tolist()), dtype=dtype, device=device)
    for scale in scales:
        sample_shape = np.ceil(out_shape[:-1] / float(scale)).astype(np.int64)
        sample_shape = tuple(sample_shape.tolist()) + (int(out_shape[-1]),)

        std_gen = _next_generator(rng, device)
        std = torch.rand((), generator=std_gen, dtype=dtype, device=device)
        std = std * (max_std - min_std) + min_std

        noise_gen = _next_generator(rng, device)
        gauss = torch.randn(sample_shape, generator=noise_gen, dtype=dtype, device=device) * std

        zoom = [float(o) / float(s) for o, s in zip(out_shape.tolist(), sample_shape)]
        if math.isclose(scale, 1.0):
            out = out + gauss
        else:
            upsampled = ne.utils.resize(gauss, zoom[:-1])
            out = out + upsampled

    return out


def random_blur_rescale(x,
                        std_min=8 / 2.355,
                        std_max=32 / 2.355,
                        isotropic=False,
                        seed=None,
                        reduce=torch.std,
                        batched=False):
    n_dim = len(x.shape[int(batched):-1])
    prop = dict(sigma=std_max, separate=True, random=True, min_sigma=std_min, dtype=x.dtype)

    rand = np.random.default_rng(seed)
    seeds = rand.integers(np.iinfo(int).max, size=n_dim)
    kernel = [ne.utils.gaussian_kernel(**prop, seed=int(s)) for s in seeds]
    if isotropic:
        kernel = kernel[:1] * n_dim

    before = reduce(x)
    x = ne.utils.separable_conv(x, kernel, batched=batched)
    after = reduce(x)
    after = after if torch.is_tensor(after) else torch.as_tensor(after, dtype=x.dtype)
    before = before if torch.is_tensor(before) else torch.as_tensor(before, dtype=x.dtype)
    return x * (before / (after + 1e-8))


def draw_perlin_full(shape,
                     noise_min=0.01,
                     noise_max=1,
                     fwhm_min=4,
                     fwhm_max=32,
                     isotropic=False,
                     batched=False,
                     featured=False,
                     reduce=torch.std,
                     dtype=torch.float32,
                     axes=None,
                     seed=None):
    assert 0 < noise_min <= noise_max
    rng = np.random.default_rng(seed)
    device = torch.device('cpu')

    if isinstance(shape, torch.Tensor):
        shape_vals = [int(v) for v in shape.detach().cpu().tolist()]
    elif isinstance(shape, np.ndarray):
        shape_vals = [int(v) for v in shape.tolist()]
    else:
        shape_vals = [int(v) for v in shape]

    axes = ne.py.utils.normalize_axes(axes, shape_vals, none_means_all=False)
    if not batched:
        shape_vals = [1, *shape_vals]
        axes = tuple(ax + 1 for ax in axes)
    if not featured:
        shape_vals = [*shape_vals, 1]

    shape_np = np.asarray(shape_vals, dtype=np.int64)
    mask = np.zeros_like(shape_np, dtype=bool)
    for ax in axes:
        mask[ax] = True
    shape_sd = shape_np.copy()
    shape_sd[~mask] = 1

    if not hasattr(fwhm_min, '__iter__'):
        fwhm_min = [fwhm_min]
    if not hasattr(fwhm_max, '__iter__'):
        fwhm_max = [fwhm_max]
    if len(fwhm_min) != len(fwhm_max):
        raise ValueError('fwhm_min and fwhm_max must have the same length')

    out = []
    sigma_shape = tuple(int(s) for s in shape_sd.tolist())
    noise_shape = tuple(int(s) for s in shape_np.tolist())
    for low, upp in zip(fwhm_min, fwhm_max):
        sigma_gen = _next_generator(rng, device)
        noise_sigma = torch.rand(sigma_shape, generator=sigma_gen, dtype=dtype, device=device)
        noise_sigma = noise_sigma * (noise_max - noise_min) + noise_min

        noise_gen = _next_generator(rng, device)
        noise = torch.randn(noise_shape, generator=noise_gen, dtype=dtype, device=device)
        noise = noise * noise_sigma.reshape(sigma_shape)

        blurred = random_blur_rescale(
            noise,
            std_min=low / 2.355,
            std_max=upp / 2.355,
            batched=True,
            isotropic=isotropic,
            seed=int(rng.integers(np.iinfo(np.int64).max)),
            reduce=reduce,
        )
        out.append(blurred)

    result = torch.stack(out, dim=0).mean(dim=0)
    if not batched:
        result = result[0]
    if not featured:
        result = result[..., 0]
    return result


def draw_crop_mask(x,
                   crop_min=0,
                   crop_max=0.5,
                   axis: Optional[Sequence[int]] = None,
                   prob=1,
                   bilateral=False,
                   seed=None):
    is_tensor = isinstance(x, torch.Tensor)
    tensor = x if is_tensor else torch.as_tensor(x)
    device = tensor.device if is_tensor else torch.device('cpu')
    dtype = tensor.dtype

    if not (0 <= crop_min <= crop_max <= 1):
        raise ValueError(f'invalid crop proportions: {crop_min}, {crop_max}')
    if not (0 <= prob <= 1):
        raise ValueError(f'prob must be within [0, 1], got {prob}')

    rng = np.random.default_rng(seed)
    axes = ne.py.utils.normalize_axes(axis, tensor.shape, none_means_all=True)
    if not axes:
        return torch.ones_like(tensor)

    prop_cut = float(crop_max)
    if crop_min < crop_max:
        prop_cut = float(rng.uniform(crop_min, crop_max))

    if prob < 1 and rng.uniform() >= prob:
        prop_cut = 0.0

    rand_prop = rng.uniform()
    if not bilateral:
        rand_prop = 0.0 if rand_prop < 0.5 else 1.0
    prop_low = prop_cut * rand_prop
    prop_cen = 1.0 - prop_cut

    crop_axis = axes[rng.integers(len(axes))]

    masks = []
    for ax in axes:
        width = tensor.shape[ax]
        mask = torch.ones(int(width), dtype=dtype, device=device)
        if prop_cut > 0 and ax == crop_axis:
            start = int(np.floor(prop_low * width))
            end = int(np.ceil((prop_low + prop_cen) * width))
            start = max(0, min(start, width))
            end = max(start, min(end, width))
            mask = torch.zeros(int(width), dtype=dtype, device=device)
            mask[start:end] = 1

        view_shape = [1] * tensor.ndim
        view_shape[ax] = width
        mask = mask.view(view_shape)
        masks.append(mask)

    out = masks[0]
    for mask in masks[1:]:
        out = out * mask

    target_shape = tuple(int(s) for s in tensor.shape)
    return out.expand(target_shape).to(dtype)
