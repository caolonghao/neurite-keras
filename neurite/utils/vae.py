"""Utilities for inspecting variational autoencoders without TensorFlow dependencies."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras import layers as KL
from keras.utils import plot_model
from tqdm.auto import tqdm

try:  # PCA is optional; warn if unavailable.
    from sklearn.decomposition import PCA
except Exception as exc:  # pragma: no cover - best effort warning
    PCA = None
    warnings.warn(str(exc))

import neurite as ne
from .model import mod_submodel


# -----------------------------------------------------------------------------
# Decoder extraction / sampling
# -----------------------------------------------------------------------------


def extract_z_dec(model: Model, sample_layer_name: str, vis: bool = False, wt_chk: bool = False) -> Model:
    """Create a decoder-only model that takes latent ``z`` as input."""

    tmp_model = Model(model.inputs, model.outputs[0])
    sample_layer = model.get_layer(sample_layer_name)
    sample_output = sample_layer.output
    if isinstance(sample_output, (list, tuple)):
        sample_output = sample_output[0]
    latent_shape = tuple(int(s) for s in sample_output.shape[1:])

    new_z_input = KL.Input(shape=latent_shape, name='z_input')
    new_inputs = [new_z_input, *model.inputs[1:]]

    dec_outputs = mod_submodel(tmp_model, new_input_nodes=new_inputs)
    decoder = Model(new_inputs, dec_outputs, name=f'{model.name}_decoder')

    if vis:
        outfile = NamedTemporaryFile(suffix='.png', delete=False).name
        plot_model(decoder, to_file=outfile, show_shapes=True)
        try:
            from IPython.display import Image  # type: ignore
            Image(outfile, width=100)
        except Exception:  # pylint: disable=broad-except
            warnings.warn(f'Decoder diagram saved to {outfile}', RuntimeWarning)

    if wt_chk:
        for layer in decoder.layers:
            if layer.name in {l.name for l in model.layers}:
                original = model.get_layer(layer.name).get_weights()
                copied = layer.get_weights()
                if len(original) != len(copied) or any(np.any(np.abs(o - c) > 1e-8) for o, c in zip(original, copied)):
                    raise RuntimeError(f'Weights mismatch detected for layer {layer.name}.')

    return decoder


def z_effect(*_, **__):  # pragma: no cover - advanced feature
    raise NotImplementedError('Gradient-based latent analysis is not available in the torch-backed port.')


def sample_dec(z_dec_model: Model,
               z_mu: Optional[np.ndarray] = None,
               z_logvar: Optional[np.ndarray] = None,
               nb_samples: int = 5,
               tqdm=tqdm,
               z_id: Optional[int] = None,
               do_sweep: bool = False,
               nb_sweep_stds: float = 3,
               extra_inputs: Iterable[np.ndarray] | None = None,
               nargout: int = 1):
    """Sample the decoder given Gaussian latent statistics."""

    extra_inputs = list(extra_inputs or [])
    latent_shape = tuple(int(s) for s in z_dec_model.inputs[0].shape[1:])
    z_mu = np.zeros((1, *latent_shape)) if z_mu is None else np.reshape(z_mu, (1, *latent_shape))
    z_logvar = np.zeros((1, *latent_shape)) if z_logvar is None else np.reshape(z_logvar, (1, *latent_shape))

    z_std = np.exp(z_logvar / 2.0)

    if do_sweep:
        if z_id is not None:
            low = z_mu.copy()
            high = z_mu.copy()
            low[0, z_id] = z_mu[0, z_id] - nb_sweep_stds * z_std[0, z_id]
            high[0, z_id] = z_mu[0, z_id] + nb_sweep_stds * z_std[0, z_id]
        else:
            low = z_mu - nb_sweep_stds * z_std
            high = z_mu + nb_sweep_stds * z_std
        sweep = np.linspace(0, 1, nb_samples)
        z_samples = [alpha * high + (1 - alpha) * low for alpha in sweep]
    else:
        if z_id is not None:
            mask = np.zeros_like(z_std)
            mask[..., z_id] = z_std[..., z_id]
            z_samples = [np.random.normal(loc=z_mu, scale=mask)]
        else:
            z_samples = [np.random.normal(loc=z_mu, scale=z_std) for _ in range(nb_samples)]

    outputs = []
    for z_sample in tqdm(z_samples):
        inputs = [z_sample, *extra_inputs]
        outputs.append(z_dec_model.predict(inputs, verbose=0))

    if nargout == 1:
        return outputs
    return outputs, z_samples


def sweep_dec_given_x(*_, **__):  # pragma: no cover - advanced feature
    raise NotImplementedError('Decoder sweep requires backend-specific graph access and is not ported.')


def pca_init_dense(*_, **__):  # pragma: no cover - advanced feature
    raise NotImplementedError('PCA-based dense initialisation is not supported in the torch-backed port.')


def model_output_pca(pre_mu_model: Model,
                     generator,
                     nb_samples: int,
                     nb_components: int,
                     vis: bool = False,
                     tqdm=tqdm):
    if PCA is None:
        raise RuntimeError('scikit-learn is required for PCA analysis.')

    sample = next(generator)
    batch_size = _sample_batch_size(sample)
    outputs: List[np.ndarray] = []

    if batch_size == 1:
        outputs.append(pre_mu_model.predict(sample[0], verbose=0))
        for _ in tqdm(range(1, nb_samples)):
            sample = next(generator)
            outputs.append(pre_mu_model.predict(sample[0], verbose=0))
        y = np.vstack(outputs)
    else:
        if batch_size != nb_samples:
            raise ValueError('Generator must yield either single samples or batches equal to nb_samples.')
        y = pre_mu_model.predict(sample[0], verbose=0)

    pca = PCA(n_components=nb_components)
    x = pca.fit_transform(y)
    if vis:
        ne.plt.pca(pca, x, y)
    return pca, x, y


@dataclass
class LatentStats:
    mu: np.ndarray
    logvar: np.ndarray


def latent_stats(model: Model, gen, nb_reps: int = 100, tqdm=tqdm) -> LatentStats:
    mu_records: List[np.ndarray] = []
    logvar_records: List[np.ndarray] = []

    for _ in tqdm(range(nb_reps)):
        sample = next(gen)
        outputs = model.predict(sample[0], verbose=0)
        if len(outputs) < 3:
            raise ValueError('Expected model to output at least mu and logvar components.')
        mu_records.append(outputs[1])
        logvar_records.append(outputs[2])

    mu_data = np.reshape(np.vstack(mu_records), (len(mu_records), -1))
    logvar_data = np.reshape(np.vstack(logvar_records), (len(logvar_records), -1))
    return LatentStats(mu=mu_data, logvar=logvar_data)


def latent_stats_plots(model: Model,
                       gen,
                       nb_reps: int = 100,
                       dim_1: int = 0,
                       dim_2: int = 1,
                       figsize: Tuple[int, int] = (15, 7),
                       colors: Optional[np.ndarray] = None,
                       tqdm=tqdm):
    stats = latent_stats(model, gen, nb_reps=nb_reps, tqdm=tqdm)
    mu_data = stats.mu
    logvar_data = stats.logvar

    z = mu_data.shape[0]
    if colors is None:
        colors = np.linspace(0, 1, z)
    datapoints = np.zeros_like(mu_data)
    for idx, mu in enumerate(mu_data):
        logvar = logvar_data[idx]
        eps = np.random.normal(size=mu.shape)
        datapoints[idx] = mu + np.exp(logvar / 2.0) * eps

    x = np.arange(mu_data.shape[1])

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.scatter(datapoints[:, dim_1], datapoints[:, dim_2], c=colors)
    plt.title(f'sample distribution (nb_reps={nb_reps})')
    plt.xlabel(f'dim {dim_1}')
    plt.ylabel(f'dim {dim_2}')

    plt.subplot(1, 2, 2)
    d_mean = np.mean(datapoints, axis=0)
    order = np.argsort(d_mean)
    d_std = np.std(datapoints, axis=0)
    plt.scatter(x, d_mean[order], c=colors[order])
    plt.plot(x, d_mean[order] + d_std[order], 'k')
    plt.plot(x, d_mean[order] - d_std[order], 'k')
    plt.xlabel('sorted dims')
    plt.ylabel('mean sample z')
    plt.title('mean +/- std of sampled z')

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    plt.scatter(mu_data[:, dim_1], mu_data[:, dim_2], c=colors)
    plt.title('mu distribution')
    plt.xlabel(f'dim {dim_1}')
    plt.ylabel(f'dim {dim_2}')

    plt.subplot(1, 2, 2)
    plt.scatter(logvar_data[:, dim_1], logvar_data[:, dim_2], c=colors)
    plt.title('logvar distribution')
    plt.xlabel(f'dim {dim_1}')
    plt.ylabel(f'dim {dim_2}')

    plt.figure(figsize=figsize)
    plt.subplot(1, 2, 1)
    mu_mean = np.mean(mu_data, axis=0)
    order = np.argsort(mu_mean)
    mu_std = np.std(mu_data, axis=0)
    plt.scatter(x, mu_mean[order], c=colors[order])
    plt.plot(x, mu_mean[order] + mu_std[order], 'k')
    plt.plot(x, mu_mean[order] - mu_std[order], 'k')
    plt.xlabel('sorted dims')
    plt.ylabel('mean mu')
    plt.title('Mean of mu components')

    plt.subplot(1, 2, 2)
    logvar_mean = np.mean(logvar_data, axis=0)
    logvar_std = np.std(logvar_data, axis=0)
    plt.scatter(x, logvar_mean[order], c=colors[order])
    plt.plot(x, logvar_mean[order] + logvar_std[order], 'k')
    plt.plot(x, logvar_mean[order] - logvar_std[order], 'k')
    plt.xlabel('sorted dims')
    plt.ylabel('mean logvar')
    plt.title('Mean of logvar components')

    plt.show()
    return stats


# -----------------------------------------------------------------------------
# Helper utilities
# -----------------------------------------------------------------------------

def _sample_batch_size(sample) -> int:
    if isinstance(sample[0], (list, tuple)):
        return _sample_batch_size(sample[0])
    return sample[0].shape[0]


__all__ = [
    'extract_z_dec',
    'z_effect',
    'sample_dec',
    'sweep_dec_given_x',
    'pca_init_dense',
    'model_output_pca',
    'latent_stats',
    'latent_stats_plots',
]
