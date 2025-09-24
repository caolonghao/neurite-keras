"""Torch-friendly segmentation helpers for neurite."""

from __future__ import annotations

import itertools
from typing import Iterable, Sequence, Tuple

import numpy as np
from tqdm import tqdm

import neurite as ne
import neurite.py.utils
import pystrum.pynd.patchlib as pl
import pystrum.pytools.timer as timer


def predict_volumes(models,
                    data_generator,
                    batch_size,
                    patch_size,
                    patch_stride,
                    grid_size,
                    nan_func=np.nanmedian,
                    do_extra_vol=False,
                    do_prob_of_true=False,
                    verbose=False):
    if not isinstance(models, (list, tuple)):
        models = (models,)

    with timer.Timer('predict_volume_stack', verbose):
        stacks = predict_volume_stack(models,
                                      data_generator,
                                      batch_size,
                                      grid_size,
                                      verbose)

    results = []
    for midx, model in enumerate(models):
        stack = stacks if len(models) == 1 else stacks[midx]
        all_true, all_pred, all_vol = stack
        true_labels, pred_labels = pred_to_label(all_true, all_pred)

        args = [patch_size, grid_size, patch_stride]
        label_kwargs = {'nan_func_layers': nan_func, 'nan_func_K': nan_func, 'verbose': verbose}
        vol_true_label = _quilt(true_labels, *args, **label_kwargs).astype('int')
        vol_pred_label = _quilt(pred_labels, *args, **label_kwargs).astype('int')

        ret = (vol_true_label, vol_pred_label)

        if do_extra_vol:
            vol_input = _quilt(all_vol, *args)
            ret += (vol_input,)

        if do_extra_vol and do_prob_of_true:
            prob = prob_of_label(all_pred, true_labels)
            pred_prob = _quilt(prob, *args, **label_kwargs)
            ret += (pred_prob,)

        results.append(ret)

    return results[0] if len(results) == 1 else tuple(results)


def predict_volume_stack(models,
                         data_generator,
                         batch_size,
                         grid_size,
                         verbose=False):
    if not isinstance(models, (list, tuple)):
        models = (models,)
    num_patches = int(np.prod(grid_size))
    outputs = []
    for model in models:
        all_true = []
        all_pred = []
        all_inputs = []
        loop = range(0, num_patches, batch_size)
        loop_iter = tqdm(loop, desc='predict', disable=not verbose)
        for _ in loop_iter:
            batch = next(data_generator)
            inputs, labels = batch
            logits = model.predict(inputs, verbose=0)
            batch_inputs = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
            all_true.append(np.asarray(labels))
            all_pred.append(np.asarray(logits))
            all_inputs.append(np.asarray(batch_inputs))
        outputs.append((np.concatenate(all_true, axis=0),
                        np.concatenate(all_pred, axis=0),
                        np.concatenate(all_inputs, axis=0)))
    return outputs[0] if len(outputs) == 1 else tuple(outputs)


def pred_to_label(*arrays: np.ndarray) -> Tuple[np.ndarray, ...]:
    return tuple(np.argmax(arr, axis=-1).astype(int) for arr in arrays)


def prob_of_label(pred_vol, true_label_vol):
    flat_pred = pred_vol.reshape(pred_vol.shape[0], -1, pred_vol.shape[-1])
    flat_true = true_label_vol.reshape(true_label_vol.shape[0], -1)
    idx = np.arange(flat_true.shape[1])
    probs = np.take_along_axis(flat_pred, flat_true[..., None], axis=-1)
    return probs.reshape(true_label_vol.shape)


def next_pred_label(model, data_generator, verbose=False):
    sample = next(data_generator)
    with timer.Timer('prediction', verbose):
        pred = model.predict(sample[0], verbose=0)
    sample_input = sample[0] if not isinstance(sample[0], (list, tuple)) else sample[0][0]
    max_labels = pred_to_label(sample_input, pred)
    return (sample, pred) + max_labels


def next_label(model, data_generator):
    sample, pred, true_label, pred_label = next_pred_label(model, data_generator)
    return true_label, pred_label


def sample_to_label(model, sample):
    pred = model.predict(sample[0], verbose=0)
    return pred_to_label(sample[1], pred)


def next_vol_pred(model, data_generator, verbose=False):
    sample = next(data_generator)
    with timer.Timer('prediction', verbose):
        pred = model.predict(sample[0], verbose=0)
    inputs = sample[0][0] if isinstance(sample[0], (list, tuple)) else sample[0]
    data = (inputs, sample[1], pred)
    if isinstance(sample[0], (list, tuple)) and len(sample[0]) > 1:
        data = (*data, sample[0][1])
    return data


def recode(seg, mapping, max_label=None):
    if isinstance(mapping, (list, tuple, np.ndarray)):
        mapping = {l: i + 1 for i, l in enumerate(mapping)}
    elif hasattr(mapping, 'mapping'):
        mapping = mapping.mapping
    if not isinstance(mapping, dict):
        raise ValueError('Invalid mapping type %s.' % type(mapping).__name__)

    in_labels = np.int32(np.unique(list(mapping.keys())))
    max_label = int(np.max(in_labels) if max_label is None else max_label)
    lookup = np.zeros(max_label + 1, dtype=np.float32)
    for src, trg in mapping.items():
        lookup[int(src)] = trg
    seg = np.asarray(seg, dtype=np.int32)
    seg = np.clip(seg, 0, max_label)
    return lookup[seg]


def _quilt(patches, patch_size, grid_size, patch_stride, verbose=False, **kwargs):
    patches = np.reshape(patches, (patches.shape[0], -1, 1))
    quilted = pl.quilt(patches, patch_size, grid_size, patch_stride=patch_stride, **kwargs)
    return quilted

