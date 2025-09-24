"""Torch-based augmentation utilities for neurite."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch

import neurite as ne
import warnings

def draw_perlin(out_shape,
                scales,
                min_std=0,
                max_std=1,
                dtype=torch.float32,
                seed=None):
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    rand = np.random.default_rng(seed)
    out = torch.zeros(tuple(out_shape), dtype=dtype)
    for scale in scales:
        sample_shape = np.ceil(out_shape[:-1] / scale).astype(np.int32)
        sample_shape = (*sample_shape, out_shape[-1])

        std = torch.empty((), dtype=dtype).uniform_(min_std, max_std)
        gauss = torch.randn(sample_shape, dtype=dtype) * std

        zoom = [o / s for o, s in zip(out_shape, sample_shape)]
        if math.isclose(scale, 1):
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
    rand = np.random.default_rng(seed)

    axes = ne.py.utils.normalize_axes(axes, shape, none_means_all=False)
    if not batched:
        shape = [1, *shape]
        axes = [ax + 1 for ax in axes]
    if not featured:
        shape = [*shape, 1]

    shape = torch.tensor(shape, dtype=torch.int64)
    sized_axes = torch.tensor([1 if i not in axes else shape[i] for i in range(len(shape))])

    if not hasattr(fwhm_min, '__iter__'):
        fwhm_min = [fwhm_min]
    if not hasattr(fwhm_max, '__iter__'):
        fwhm_max = [fwhm_max]

    out = []
    for low, upp in zip(fwhm_min, fwhm_max):
        noise = torch.empty(tuple(sized_axes.tolist()), dtype=dtype).uniform_(noise_min, noise_max)
        blurred = torch.randn(tuple(shape.tolist()), dtype=dtype)
        blurred = ne.utils.separable_conv(blurred, ne.utils.gaussian_kernel(low, separate=True), batched=True)
        stat = reduce(blurred)
        stat = stat if torch.is_tensor(stat) else torch.as_tensor(stat, dtype=dtype)
        blurred = blurred * (reduce(noise) / (stat + 1e-8))
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
    warnings.warn('`utils.subsample_axis` and `layers.Subsample` are deprecated; use DownUpSample instead.', RuntimeWarning)
    return torch.ones_like(x)
