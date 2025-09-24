"""Keras model utilities that avoid TensorFlow-specific APIs."""

from __future__ import annotations

import warnings
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Sequence

import numpy as np
from keras import Model
from keras.utils import plot_model


def _ensure_list(value):
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def stack_models(models: Sequence[Model], connecting_node_ids: Sequence[Sequence[int]] | None = None) -> Model:
    """Sequentially connect multiple Keras models without nesting.

    This simplified implementation supports the common case where the outputs of model ``i`` feed
    particular input indices of model ``i+1``. Additional inputs of the downstream model are
    promoted to inputs of the stacked model. Layers are reused, so weight updates reflect in all
    models.
    """

    models = list(models)
    if not models:
        raise ValueError('No models provided to stack.')

    base_inputs: List = list(models[0].inputs)
    outputs = list(_ensure_list(models[0].outputs))

    if connecting_node_ids is not None and len(connecting_node_ids) != len(models) - 1:
        raise ValueError('connecting_node_ids must have len(models) - 1 entries.')

    for idx, model in enumerate(models[1:], start=1):
        conn = connecting_node_ids[idx - 1] if connecting_node_ids is not None else list(range(len(outputs)))
        conn = list(conn)
        if len(conn) != len(outputs):
            raise ValueError('Mismatch between number of outputs and connecting indices.')
        if max(conn, default=-1) >= len(model.inputs):
            raise ValueError('Connecting index exceeds model input count.')

        mapping = {target_idx: outputs[i] for i, target_idx in enumerate(conn)}
        feed_inputs = []
        external_inputs = []
        for i, orig_input in enumerate(model.inputs):
            if i in mapping:
                feed_inputs.append(mapping[i])
            else:
                feed_inputs.append(orig_input)
                external_inputs.append(orig_input)

        model_outputs = _ensure_list(model(feed_inputs))
        outputs = model_outputs
        for extra in external_inputs:
            if extra not in base_inputs:
                base_inputs.append(extra)

    final_outputs = outputs[0] if len(outputs) == 1 else outputs
    return Model(inputs=base_inputs, outputs=final_outputs)


def mod_submodel(orig_model: Model, new_input_nodes=None, input_layers=None):
    """Minimal re-routing helper compatible with the legacy signature."""

    if input_layers is not None:
        raise NotImplementedError('input_layers rewrite is not supported in this torch port.')

    if new_input_nodes is None:
        return orig_model.outputs

    new_inputs = _ensure_list(new_input_nodes)
    if len(new_inputs) != len(orig_model.inputs):
        raise ValueError('Number of replacement inputs must match original inputs.')
    return _ensure_list(orig_model(new_inputs))


def reset_weights(model: Model):  # pragma: no cover - best-effort helper
    """Reset model weights using the layers' initializers."""

    for layer in model.layers:
        for weight in layer.weights:
            initializer = getattr(weight, 'initializer', None)
            if initializer is not None:
                weight.assign(initializer(weight.shape, weight.dtype))


def copy_weights(dest_model: Model, src_model: Model, skip_mismatch: bool = True):  # pragma: no cover
    """Copy weights between models based on layer names."""

    src_layers = {layer.name: layer for layer in src_model.layers}
    for layer in dest_model.layers:
        if layer.name not in src_layers:
            if not skip_mismatch:
                raise ValueError(f'Missing layer {layer.name} in source model.')
            continue
        try:
            layer.set_weights(src_layers[layer.name].get_weights())
        except Exception as err:  # pylint: disable=broad-except
            if skip_mismatch:
                warnings.warn(f'Could not copy weights for layer {layer.name}: {err}', RuntimeWarning)
            else:
                raise


def robust_multi_gpu(model: Model, gpus, verbose: bool = True) -> Model:
    """Return the model unchanged while warning about unsupported multi-GPU mode."""

    if (isinstance(gpus, int) and gpus > 1) or (isinstance(gpus, (list, tuple)) and len(gpus) > 1):
        if verbose:
            warnings.warn('Multi-GPU replication is not available in the torch-backed port; returning original model.', RuntimeWarning)
    return model


def diagram(model: Model):  # pragma: no cover - visualization helper
    outfile = NamedTemporaryFile(suffix='.png', delete=False).name
    plot_model(model, to_file=outfile, show_shapes=True)
    try:
        from IPython.display import Image  # type: ignore
        return Image(outfile, width=100)
    except Exception:  # pylint: disable=broad-except
        warnings.warn(f'Model diagram saved to {outfile}', RuntimeWarning)
        return outfile


__all__ = [
    'stack_models',
    'mod_submodel',
    'reset_weights',
    'copy_weights',
    'robust_multi_gpu',
    'diagram',
]
