"""
layers for the neuron project

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

# internal python imports
import sys
import itertools
import warnings

# third party
import numpy as np
import keras
from keras import backend as K
from keras import activations as KActivations
from keras import constraints as KConstraints
from keras import initializers as KInitializers
from keras import regularizers as KRegularizers
from keras.layers import Layer, InputLayer, Input, InputSpec

from . import keras_backend as tf

# local imports
from . import utils
from . import py


class Negate(Layer):
    """ 
    Keras Layer: negative of the input.
    """

    def __init__(self, **kwargs):
        super(Negate, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Negate, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -x

    def compute_output_shape(self, input_shape):
        return input_shape


class RescaleValues(Layer):
    """ 
    Very simple Keras layer to rescale data values (e.g. intensities) by fixed factor
    """

    def __init__(self, resize, **kwargs):
        self.resize = resize
        super(RescaleValues, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'resize': self.resize})
        return config

    def build(self, input_shape):
        super(RescaleValues, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.resize

    def compute_output_shape(self, input_shape):
        return input_shape


class Resize(Layer):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
    """

    def __init__(self,
                 zoom_factor,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters: 
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow 
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'zoom_factor': self.zoom_factor,
            'interp_method': self.interp_method,
        })
        return config

    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume
                should be a *vol_shape x N
        """

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape
        if not isinstance(self.zoom_factor, (list, tuple)):
            self.zoom_factor = [self.zoom_factor] * self.ndims
        else:
            assert len(self.zoom_factor) == self.ndims, \
                'zoom factor length {} does not match number of dimensions {}'\
                .format(len(self.zoom_factor), self.ndims)

        # confirm built
        self.built = True

        super(Resize, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = K.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        return tf.map_fn(self._single_resize, vol)

    def compute_output_shape(self, input_shape):

        output_shape = [input_shape[0]]
        output_shape += [int(input_shape[1:-1][f] * self.zoom_factor[f]) for f in range(self.ndims)]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return utils.resize(inputs, self.zoom_factor, interp_method=self.interp_method)


# Zoom naming of resize, to match scipy's naming
Zoom = Resize


class SoftQuantize(Layer):
    """ 
    Keras Layer: soft quantization of intentity input

    If you find this class useful, please consider citing:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self,
                 alpha=1,
                 bin_centers=None,
                 nb_bins=16,
                 min_clip=-np.inf,
                 max_clip=np.inf,
                 return_log=False,
                 **kwargs):

        self.alpha = alpha
        self.bin_centers = bin_centers
        self.nb_bins = nb_bins
        self.min_clip = min_clip
        self.max_clip = max_clip
        self.return_log = return_log
        super(SoftQuantize, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SoftQuantize, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return -utils.soft_quantize(x,
                                    alpha=self.alpha,
                                    bin_centers=self.bin_centers,
                                    nb_bins=self.nb_bins,
                                    min_clip=self.min_clip,
                                    max_clip=self.max_clip,
                                    return_log=False)              # [bs, ..., B]

    def compute_output_shape(self, input_shape):
        output_shape_lst = list(input_shape) + [self.nb_bins]
        return tuple(output_shape_lst)


class MSE(Layer):
    """ 
    Keras Layer: mean squared error
    """

    def __init__(self, **kwargs):
        super(MSE, self).__init__(**kwargs)

    def build(self, input_shape):
        super(MSE, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return K.mean(K.batch_flatten(K.square(x[0] - x[1])), -1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], )


class GaussianBlur(Layer):
    """
    Blur a tensor by convolving it with a Gaussian kernel. The layer supports isotropic and
    anisotropic blurring, randomized or not.

    If you find this layer useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self,
                 sigma=None,
                 level=None,
                 random=False,
                 min_sigma=0,
                 isotropic=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            sigma: SD of the blurring kernel, as a scalar or length-N iterable. While specifying
                a single SD will result in isotropic blurring, you can pass a separate SD for each
                axis of space. For `random=True`, this argument defines the upper bounds on the
                blurring SDs, and the layer will sample the SD for axis i between the i-th
                elements of `min_sigma` and `sigma`, respectively. Careful: random blur is
                anisotropic unless you pass `isotropic=True`!
            level: Deprecated in favor of `sigma`.
            random: Sample the blurring SDs uniformly from the interval [`min_sigma`, `sigma`).
            min_sigma: Lower bound on the blurring SDs, only used for random sampling. See `sigma`.
            isotropic: For non-random blur, isotropy is implicitly controlled by the length of
                `sigma`. For random isotropic blur, set `isotropic=True` and pass a single SD.
            seed: Integer for reproducible randomization. Only considered for random sampling.
        """
        assert sigma is not None or level is not None, 'sigma or level must be provided'
        assert not (sigma is not None and level is not None), 'only sigma or level must be provided'

        if level is not None:
            warnings.warn('The `level` argument to ne.layers.GaussianBlur is deprecated and will '
                          'be removed in a future version. Please use `sigma` instead.')
            if level < 1:
                raise ValueError('Gaussian blur level must not be less than 1')
            if random:
                raise ValueError('level argument incompatible with random blurring')

            self.sigma = (level - 1) ** 2

        if isotropic and not random:
            raise ValueError('For non-random blurring, isotropy is implicitly controlled by the '
                             'number of sigmas provided. Set `isotropic` only for random blur.')

        self.sigma = sigma
        self.random = random
        self.min_sigma = min_sigma
        self.isotropic = isotropic
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'sigma': self.sigma,
            'random': self.random,
            'min_sigma': self.min_sigma,
            'isotropic': self.isotropic,
            'seed': self.seed,
        })
        return config

    def _normalize_sigma(self, sigma, ndims):
        sigma = np.ravel(sigma)
        sigma = sigma.tolist()
        if len(sigma) not in (1, ndims):
            raise ValueError(f'1 or {ndims} sigmas expected in {ndims}D space, got {len(sigma)}')

        if any(s < 0 for s in sigma):
            raise ValueError('Gaussian blur sigma must not be less than 0')

        if len(sigma) > 1 and self.isotropic:
            raise ValueError(f'random isotropic blur requires a single sigma, got {len(sigma)}')

        if len(sigma) == 1:
            sigma = sigma * ndims
        return sigma

    def build(self, input_shape):
        # Convert to N-element lists.
        ndims = len(input_shape) - 2
        self.sigma = self._normalize_sigma(self.sigma, ndims)
        self.min_sigma = self._normalize_sigma(self.min_sigma, ndims)

        # Have `separable_conv` apply the same random kernel along all axes.
        if self.isotropic and self.random:
            self.sigma = self.sigma[:1]
            self.min_sigma = self.min_sigma[:1]

        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Tensor that will be blurred.
        """
        if not any(s > 0 for s in self.sigma):
            return x

        kernel = utils.gaussian_kernel(sigma=self.sigma,
                                       random=self.random,
                                       min_sigma=self.min_sigma,
                                       separate=True,
                                       dtype=x.dtype)

        return utils.separable_conv(x, kernel, batched=True)


class Subsample(Layer):
    """
    Symmetrically subsample a tensor by a factor f (stride) along a single spatial axis using
    nearest-neighbor interpolation and optionally upsample again, to reduce its resolution. Both
    f and the subsampling axis can be randomly drawn.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 stride_min=1,
                 stride_max=8,
                 axes=None,
                 prob=1,
                 upsample=True,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            stride_min: Lower bound on the subsampling factor.
            stride_max: Upper bound on the subsampling factor.
            axes: Spatial axes to draw the subsampling axis from. None means all.
            prob: Subsampling probability. A value of 1 means always, 0 never.
            upsample: Upsample the tensor to restore its original shape.
            seed: Integer for reproducible randomization.
        """
        self.stride_min = stride_min
        self.stride_max = stride_max
        self.axes = axes
        self.prob = prob
        self.upsample = upsample
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'stride_min': self.stride_min,
            'stride_max': self.stride_max,
            'axes': self.axes,
            'prob': self.prob,
            'upsample': self.upsample,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'

        allowed = range(1, ndims + 1)
        self.axes = py.utils.normalize_axes(self.axes, input_shape, allowed, none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor to subsample.
        """
        if self.prob == 0 or self.stride_max == 1:
            return x

        shape = tf.shape(x)
        x = utils.subsample_axis(x,
                                 stride_min=self.stride_min,
                                 stride_max=self.stride_max,
                                 axes=self.axes,
                                 prob=self.prob,
                                 upsample=self.upsample,
                                 seed=self.seed)

        # Avoid unnecessary dynamic shapes (showing up as `None`).
        return tf.reshape(x, shape) if self.upsample else x


class RandomCrop(Layer):
    """
    Randomly crop the content of a tensor by multiplying with a binary mask.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 crop_min=0,
                 crop_max=0.5,
                 axis=None,
                 prob=1,
                 bilateral=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            crop_min: Minimum proportion of voxels to remove, in [0, `crop_max`].
            crop_max: Maximum proportion of voxels to remove, in [`crop_min`, 1].
            axis: Spatial axis along which to crop, where None means any spatial axis. With more
                than one axis specified, a single axis will be drawn at each execution.
            prob: Cropping probability, where 1 means always and 0 never.
            bilateral: Randomly distribute the cropping proportion between top/bottom end.
            seed: Integer for reproducible randomization.
        """
        self.crop_min = crop_min
        self.crop_max = crop_max
        self.axis = axis
        self.prob = prob
        self.bilateral = bilateral
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'crop_min': self.crop_min,
            'crop_max': self.crop_max,
            'axis': self.axis,
            'prob': self.prob,
            'bilateral': self.bilateral,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        allowed = range(1, ndims + 1)
        self.axis = py.utils.normalize_axes(self.axis, input_shape, allowed, none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor whose content to crop.
        """
        if self.prob == 0:
            return x

        return x * utils.augment.draw_crop_mask(x,
                                                crop_min=self.crop_min,
                                                crop_max=self.crop_max,
                                                axis=self.axis,
                                                prob=self.prob,
                                                bilateral=self.bilateral,
                                                seed=self.seed)


class RandomClip(Layer):
    """
    Clip the contents of a tensor in a randomized fashion.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 clip_min=None,
                 clip_max=None,
                 prob_min=1,
                 prob_max=1,
                 axes=0,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            clip_min: Minimum value to clip to, as a scalar. Any value below this value will be
                set to `clip_min` when clipping is performed. Pass a two-element iterable to
                uniformly sample this threshold from the interval [`clip_min[0], `clip_min[1]`).
                None means no clipping at the lower end.
            clip_max: Maximum value to clip to. See `clip_min`.
            prob_min: Clipping probability at the lower end, where 1 means always and 0 never.
            prob_max: Clipping probability at the upper end. See `prob_min`.
            axes: Axes along which clipping will vary independently. None means the the input
                tensor will be clipped as a whole. Pass `axes=(0, -1)` to randomize clipping
                independently for each channel of each batch.
            seed: Integer for reproducible randomization.
        """
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.prob_min = prob_min
        self.prob_max = prob_max
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'clip_min': self.clip_min,
            'clip_max': self.clip_max,
            'prob_min': self.prob_min,
            'prob_max': self.prob_max,
            'axes': self.axes,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        self.ndims = len(input_shape)
        self.axes = py.utils.normalize_axes(self.axes, input_shape, none_means_all=False)
        self.rand = np.random.default_rng(self.seed)
        super().build(input_shape)

    def draw_thresh(self, bounds, no_clip_tensor, prob, shape):
        assert 0 <= prob <= 1, f'{prob} is not a probability'
        assert tf.is_tensor(no_clip_tensor), 'not a tensor'

        def seed():
            return self.rand.integers(np.iinfo(int).max)

        # Threshold.
        if bounds is None or prob == 0:
            return no_clip_tensor

        if np.isscalar(bounds):
            clip_at = tf.constant(bounds, no_clip_tensor.dtype)

        else:
            clip_at = tf.random.uniform(shape, minval=bounds[0], maxval=bounds[1], seed=seed())
            if clip_at.dtype != no_clip_tensor.dtype:
                clip_at = tf.cast(clip_at, no_clip_tensor.dtype)

        # Randomization.
        if prob < 1:
            rand_bit = tf.less(tf.random.uniform(shape, seed=seed()), prob)
            rand_bit = tf.cast(rand_bit, no_clip_tensor.dtype)
            clip_at = rand_bit * clip_at + (1 - rand_bit) * no_clip_tensor

        return clip_at

    def call(self, x):
        """
        Parameters:
            x: Input tensor containing values to clip.
        """
        if self.prob_min == self.prob_max == 0:
            return x

        # No-clip values.
        x_min = tf.reduce_min(x)
        x_max = tf.reduce_max(x)

        # Shape. Defines along which axes clipping varies independently.
        shape = [(tf.shape(x)[i] if i in self.axes else 1,) for i in range(self.ndims)]
        shape = tf.concat(shape, axis=0)

        # Randomized thresholds.
        low = self.draw_thresh(self.clip_min, x_min, self.prob_min, shape)
        upp = self.draw_thresh(self.clip_max, x_max, self.prob_max, shape)

        return tf.clip_by_value(x, clip_value_min=low, clip_value_max=upp)


class RandomGamma(Layer):
    """Exponentiate the voxel intensities of a tensor by a random parameter.

    The layer draws gamma parameters from a uniform distribution.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 low=0.5,
                 high=2,
                 shared=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            low: Lower bound on the sampled exponents, as a scalar.
            high: Upper bound on the sampled exponents, as a scalar.
            shared: Apply the same gamma exponentiation across all channels.
            seed: Integer for reproducible randomization.
        """
        self.low = low
        self.high = high
        self.shared = shared
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'low': self.low,
            'high': self.high,
            'shared': self.shared,
            'seed': self.seed,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input tensor.
        """
        # Dimensions.
        num_dim = len(x.shape) - 2
        num_batch = tf.shape(x)[0]
        num_chan = 1 if self.shared else tf.shape(x)[-1]
        shape = (num_batch, *[1] * num_dim, num_chan)

        gamma = tf.random.uniform(
            shape,
            minval=self.low,
            maxval=self.high,
            dtype=x.dtype,
            seed=self.seed,
        )
        return tf.pow(x, gamma)


class RandomIntensityLookup(Layer):
    """Augment the contrast of a grayscale image using random intensity lookup tables.

    At each invocation, this layer will synthesize a smoothly varying lookup table (LUT),
    associating a new output intensity to each intensity value in the input image. We will apply
    the LUT to the input image to synthesize a new image contrast. Each batch will undergo an
    independent lookup.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 levels=256,
                 blur_min=32,
                 blur_max=64,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            levels: Number of grayscale levels to look up and re-assign.
            blur_min: Lower bound on the smoothing SD for random-contrast lookup.
            blur_max: Upper bound on the smoothing SD for random-contrast lookup.
            seed: Integer for reproducible randomization.
        """
        self.levels = levels
        self.blur_min = blur_min
        self.blur_max = blur_max
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'levels': self.levels,
            'blur_min': self.blur_min,
            'blur_max': self.blur_max,
            'seed': self.seed,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input image.
        """
        max_val = self.levels - 1
        if not x.dtype.is_floating:
            x = tf.cast(x, self.dtype)

        # Oversample LUT. Convolution requires trailing singleton dimension.
        num_draw = 5 * self.levels
        lut = tf.random.uniform(
            shape=(tf.shape(x)[0], num_draw, 1),
            minval=0,
            maxval=max_val,
            dtype=x.dtype,
            seed=self.seed,
        )

        # Smoothing. Filter shape: space, in, out.
        kernel = utils.gaussian_kernel(
            sigma=self.blur_max,
            min_sigma=self.blur_min,
            random=self.blur_min != self.blur_max,
            dtype=x.dtype,
            seed=self.seed,
        )
        kernel = tf.reshape(kernel, shape=(-1, 1, 1))
        lut = tf.nn.convolution(lut, kernel, padding='SAME')[..., 0]

        # Remove tapered edges from zero-padding.
        keep = np.arange(self.levels) + (num_draw - self.levels) // 2
        lut = tf.gather(lut, indices=keep, axis=1)

        # Normalize batches independently.
        space = range(1, len(x.shape) - 1)
        x = max_val * utils.minmax_norm(x, axis=space)
        lut = max_val * utils.minmax_norm(lut, axis=1)

        # Lookup.
        indices = tf.cast(x, tf.int32)
        return tf.map_fn(
            fn=lambda x: tf.gather(*x, axis=0),
            elems=(lut, indices),
            fn_output_signature=lut.dtype,
        )


class RandomClearLabel(Layer):
    """Randomly clear image regions corresponding to specific labels.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 prob,
                 clear=0,
                 shared=False,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            prob: Probability that we clear image regions corresponding to the label.
            shared: Synchronize the random clearing across all channels.
            clear: Integer labels to clear, as a scalar or iterable. When passing several values,
                the layer will combine and treat them as a single structure.
            seed: Integer for reproducible randomization.
        """
        self.prob = prob
        self.clear = clear
        self.shared = shared
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'prob': self.prob,
            'clear': self.clear,
            'shared': self.shared,
            'seed': self.seed,
        })
        return config

    def call(self, inputs):
        """
        Parameters:
            inputs: Input image and corresponding label map as an iterable.
        """
        image, labels = inputs
        if self.prob == 0:
            return image

        # Labels to clear.
        bg = [self.clear] if np.isscalar(self.clear) else np.unique(self.clear)
        bg = tf.convert_to_tensor(bg, labels.dtype)

        # Dimensions.
        num_dim = len(image.shape) - 2
        num_batch = tf.shape(image)[0]
        num_chan = 1 if self.shared else tf.shape(image)[-1]
        shape = (num_batch, *[1] * num_dim, num_chan)

        # Randomization.
        rand = tf.random.uniform(shape, dtype=self.dtype, seed=self.seed)
        rand = tf.less(rand, self.prob)

        # Mask.
        mask = tf.reduce_any(tf.equal(labels[..., None], bg), axis=-1)
        mask = tf.math.logical_and(mask, rand)
        mask = tf.math.logical_xor(True, mask)

        return image * tf.cast(mask, image.dtype)


class DrawImage(Layer):
    """ Generate an image from a label map by uniformly sampling a random intensity for each label.

    This layer assumes that input label maps have a single channel and integer values in the
    interval [0, N), where N corresponds to `max_label`. If you pass a list defining intensity
    bounds, the (zero-based) i-th element of the list applies to label value i.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 max_label,
                 low=0,
                 high=1,
                 channels=1,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            max_label: Highest integer value the input index label maps will ever take.
            low: Lower bounds on the intensities drawn for each label, as a scalar or list.
            high: Upper bounds on the intensities drawn for each label, as a scalar or list.
            channels: Number of generated output channels.
            seed: Integer for reproducible randomization.
        """
        self.max_label = max_label
        self.channels = channels
        self.low = low
        self.high = high
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'max_label': self.max_label,
            'channels': self.channels,
            'low': self.low,
            'high': self.high,
            'seed': self.seed,
        })
        return config

    def call(self, x):
        """
        Parameters:
            x: Input label map with non-negative, zero-based index label values.
        """
        if x.dtype not in (tf.int32, tf.int64):
            x = tf.cast(x, tf.int32)

        num_dim = len(x.shape) - 2
        num_batch = tf.shape(x)[0]
        num_label = self.max_label + 1

        # Random means.
        means = tf.random.uniform(
            shape=(num_batch, self.channels, num_label),
            minval=self.low,
            maxval=self.high,
            dtype=self.dtype,
            seed=self.seed,
        )
        means = tf.reshape(means, shape=(-1,))

        # Label intensities.
        offset_chan = tf.range(self.channels) * num_label
        offset_batch = tf.range(num_batch) * self.channels * num_label
        offset_batch = tf.reshape(offset_batch, shape=(-1, *[1] * num_dim, 1))
        x += offset_batch + offset_chan
        return tf.gather(means, indices=x)


#########################################################
# Sparse layers
#########################################################

class SpatiallySparse_Dense(Layer):
    """ 
    Spatially-Sparse Dense Layer (great name, huh?)
    This is a Densely connected (Fully connected) layer with sparse observations.

    # layer can (and should) be used when going from vol to embedding *and* going back.
    # it will account for the observed variance and maintain the same weights

    # if going vol --> enc:
    # tensor inputs should be [vol, mask], and output will be a encoding tensor enc
    # if going enc --> vol:
    # tensor inputs should be [enc], and output will be vol
    """

    def __init__(self, input_shape, output_len, use_bias=False,
                 kernel_initializer='RandomNormal',
                 bias_initializer='RandomNormal', **kwargs):
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.output_len = output_len
        self.cargs = 0
        self.use_bias = use_bias
        self.orig_input_shape = input_shape  # just the image size
        super(SpatiallySparse_Dense, self).__init__(**kwargs)

    def build(self, input_shape):

        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='mult-kernel',
                                      shape=(np.prod(self.orig_input_shape),
                                             self.output_len),
                                      initializer=self.kernel_initializer,
                                      trainable=True)

        M = K.reshape(self.kernel, [-1, self.output_len])  # D x d
        mt = K.transpose(M)  # d x D
        mtm_inv = tf.matrix_inverse(K.dot(mt, M))  # d x d
        self.W = K.dot(mtm_inv, mt)  # d x D

        if self.use_bias:
            self.bias = self.add_weight(name='bias-kernel',
                                        shape=(self.output_len, ),
                                        initializer=self.bias_initializer,
                                        trainable=True)

        # self.sigma_sq = self.add_weight(name='bias-kernel',
        #                                 shape=(1, ),
        #                                 initializer=self.initializer,
        #                                 trainable=True)

        super(SpatiallySparse_Dense, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, args):

        if not isinstance(args, (list, tuple)):
            args = [args]
        self.cargs = len(args)

        # flatten
        if len(args) == 2:  # input y, m
            # get inputs
            y, y_mask = args
            a_fact = int(y.get_shape().as_list()[-1] / y_mask.get_shape().as_list()[-1])
            y_mask = K.repeat_elements(y_mask, a_fact, -1)
            y_flat = K.batch_flatten(y)  # N x D
            y_mask_flat = K.batch_flatten(y_mask)  # N x D

            # prepare switching matrix
            W = self.W  # d x D

            w_tmp = K.expand_dims(W, 0)  # 1 x d x D
            Wo = K.permute_dimensions(w_tmp, [0, 2, 1]) * \
                K.expand_dims(y_mask_flat, -1)  # N x D x d
            WoT = K.permute_dimensions(Wo, [0, 2, 1])    # N x d x D
            WotWo_inv = tf.matrix_inverse(K.batch_dot(WoT, Wo))  # N x d x d
            pre = K.batch_dot(WotWo_inv, WoT)  # N x d x D
            res = K.batch_dot(pre, y_flat)  # N x d

            if self.use_bias:
                res += K.expand_dims(self.bias, 0)

        else:
            x_data = args[0]
            shape = K.shape(x_data)

            x_data = K.batch_flatten(x_data)  # N x d

            if self.use_bias:
                x_data -= self.bias

            res = K.dot(x_data, self.W)

            # reshape
            # Here you can mix integers and symbolic elements of `shape`
            pool_shape = tf.stack([shape[0], *self.orig_input_shape])
            res = K.reshape(res, pool_shape)

        return res

    def compute_output_shape(self, input_shape):
        # print(self.cargs, input_shape, self.output_len, self.orig_input_shape)
        if self.cargs == 2:
            return (input_shape[0][0], self.output_len)
        else:
            return (input_shape[0], *self.orig_input_shape)


#########################################################
# "Local" layers -- layers with parameters at each voxel
#########################################################

class LocalBias(Layer):
    """ 
    Local bias layer: each pixel/voxel has its own bias operation (one parameter)
    out[v] = in[v] + b

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
    """

    def __init__(self, my_initializer='RandomNormal', biasmult=1.0, **kwargs):
        self.initializer = my_initializer
        self.biasmult = biasmult
        super(LocalBias, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=input_shape[1:],
                                      initializer=self.initializer,
                                      trainable=True)
        super(LocalBias, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x + self.kernel * self.biasmult  # weights are difference from input

    def compute_output_shape(self, input_shape):
        return input_shape


class LocalLinear(Layer):
    """ 
    Local linear layer: each pixel/voxel has its own linear operation (two parameters)
    out[v] = a * in[v] + b

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148
    """

    def __init__(self, initializer='RandomNormal', **kwargs):
        self.initializer = initializer
        super(LocalLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.mult = self.add_weight(name='mult-kernel',
                                    shape=input_shape[1:],
                                    initializer=self.initializer,
                                    trainable=True)
        self.bias = self.add_weight(name='bias-kernel',
                                    shape=input_shape[1:],
                                    initializer=self.initializer,
                                    trainable=True)
        super(LocalLinear, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.mult + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape


class LocallyConnected3D(Layer):
    """Placeholder for the TensorFlow-specific implementation.

    This class previously depended on TensorFlow private kernels for locally-
    connected convolutions. Re-implementing it purely with Keras ops requires a
    bespoke sparse kernel formulation, so the functionality is temporarily
    unavailable on the Keras 3 backend.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'LocallyConnected3D is not yet ported to the Keras 3 backend. '
            'Please reach out if you rely on it so we can prioritize the rewrite.'
        )


class LocalCrossLinear(Layer):
    """ 
    Local cross mult layer

    input: [batch_size, *vol_size, nb_feats_1]
    output: [batch_size, *vol_size, nb_feats_2]

    at each spatial voxel, there is a different linear relation learned.
    """

    def __init__(self, output_features,
                 mult_initializer=None,
                 bias_initializer=None,
                 mult_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 **kwargs):

        self.output_features = output_features
        self.mult_initializer = mult_initializer
        self.bias_initializer = bias_initializer
        self.mult_regularizer = mult_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_bias = use_bias

        super(LocalCrossLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mult_shape = [1] + list(input_shape)[1:] + [self.output_features]

        # verify initializer
        if self.mult_initializer is None:
            mean = 1 / input_shape[-1]
            stddev = 0.01
            self.mult_initializer = KInitializers.RandomNormal(mean=mean, stddev=stddev)

        self.mult = self.add_weight(name='mult-kernel',
                                    shape=mult_shape,
                                    initializer=self.mult_initializer,
                                    regularizer=self.mult_regularizer,
                                    trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1 / input_shape[-1]
                stddev = 0.01
                self.bias_initializer = KInitializers.RandomNormal(mean=mean, stddev=stddev)

            bias_shape = [1] + list(input_shape)[1:-1] + [self.output_features]
            self.bias = self.add_weight(name='bias-kernel',
                                        shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        trainable=True)
        super(LocalCrossLinear, self).build(input_shape)

    def call(self, x):
        map_fn = lambda z: self._single_matmul(z, self.mult[0, ...])
        y = tf.stack(tf.map_fn(map_fn, x, dtype=tf.float32), 0)

        if self.use_bias:
            y = y + self.bias

        return y

    def _single_matmul(self, x, mult):
        x = K.expand_dims(x, -2)
        y = tf.matmul(x, mult)[..., 0, :]
        return y

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.output_features])


class LocalCrossLinearTrf(Layer):
    """ 
    Local cross mult layer with transform

    input: [batch_size, *vol_size, nb_feats_1]
    output: [batch_size, *vol_size, nb_feats_2]

    at each spatial voxel, there is a different linear relation learned.
    """

    def __init__(self, output_features,
                 mult_initializer=None,
                 bias_initializer=None,
                 mult_regularizer=None,
                 bias_regularizer=None,
                 use_bias=True,
                 trf_mult=1,
                 **kwargs):

        self.output_features = output_features
        self.mult_initializer = mult_initializer
        self.bias_initializer = bias_initializer
        self.mult_regularizer = mult_regularizer
        self.bias_regularizer = bias_regularizer
        self.use_bias = use_bias
        self.trf_mult = trf_mult
        self.interp_method = 'linear'

        super(LocalCrossLinearTrf, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        mult_shape = list(input_shape)[1:] + [self.output_features]
        ndims = len(list(input_shape)[1:-1])

        # verify initializer
        if self.mult_initializer is None:
            mean = 1 / input_shape[-1]
            stddev = 0.01
            self.mult_initializer = KInitializers.RandomNormal(mean=mean, stddev=stddev)

        self.mult = self.add_weight(name='mult-kernel',
                                    shape=mult_shape,
                                    initializer=self.mult_initializer,
                                    regularizer=self.mult_regularizer,
                                    trainable=True)

        self.trf = self.add_weight(name='def-kernel',
                                   shape=mult_shape + [ndims],
                                   initializer=KInitializers.RandomNormal(
                                       mean=0, stddev=0.001),
                                   trainable=True)

        if self.use_bias:
            if self.bias_initializer is None:
                mean = 1 / input_shape[-1]
                stddev = 0.01
                self.bias_initializer = KInitializers.RandomNormal(mean=mean, stddev=stddev)

            bias_shape = list(input_shape)[1:-1] + [self.output_features]
            self.bias = self.add_weight(name='bias-kernel',
                                        shape=bias_shape,
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        trainable=True)

        super(LocalCrossLinearTrf, self).build(input_shape)

    def call(self, x):

        # for each element in the batch
        y = tf.map_fn(self._single_batch_trf, x, dtype=tf.float32)

        return y

    def _single_batch_trf(self, vol):
        # vol should be vol_shape + [nb_features]
        # self.trf should be vol_shape + [nb_features] + [ndims]

        vol_shape = vol.shape.as_list()
        nb_input_dims = vol_shape[-1]

        # this is inefficient...
        new_vols = [None] * self.output_features
        for j in range(self.output_features):
            new_vols[j] = tf.zeros(vol_shape[:-1], dtype=tf.float32)
            for i in range(nb_input_dims):
                trf_vol = transform(vol[..., i], self.trf[..., i, j, :] *
                                    self.trf_mult, interp_method=self.interp_method)
                trf_vol = tf.reshape(trf_vol, vol_shape[:-1])
                new_vols[j] += trf_vol * self.mult[..., i, j]

                if self.use_bias:
                    new_vols[j] += self.bias[..., j]

        return tf.stack(new_vols, -1)

    def compute_output_shape(self, input_shape):
        return tuple(list(input_shape)[:-1] + [self.output_features])


class LocalParamLayer(Layer):
    """Placeholder for TensorFlow-dependent LocalParamLayer."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'LocalParamLayer depended on TensorFlow internals and is not yet ported.'
        )


class LocalParamWithInput(Layer):
    """Placeholder for TensorFlow-dependent LocalParamWithInput."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'LocalParamWithInput depended on TensorFlow internals and is not yet ported.'
        )


def LocalParam(*_args, **_kwargs):  # pylint: disable=invalid-name
    raise NotImplementedError(
        'LocalParam depended on TensorFlow internals and is not yet ported.'
    )


##########################################
# Stream layers
##########################################


class MeanStream(Layer):
    """ 
    Maintain stream of data mean. 

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.

    If you find this class useful, please cite the original paper this was written for:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks 
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019. 
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = float(cap)
        super(MeanStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean and count
        # These are weights because just maintaining variables don't get saved with the model,
        # and we'd like to have these numbers saved when we save the model.
        # But we need to make sure that the weights are untrainable.
        self.mean = self.add_weight(name='mean',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=False)
        self.count = self.add_weight(name='count',
                                     shape=[1],
                                     initializer='zeros',
                                     trainable=False)

        # self.mean = K.zeros(input_shape[1:], name='mean')
        # self.count = K.variable(0.0, name='count')
        super(MeanStream, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training=None):
        training = _get_training_value(training, self.trainable)

        # get batch shape:
        this_bs_int = K.shape(x)[0]

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.mean)), 0)
        z = tf.ones(p)

        # If calling in inference mode, use moving stats
        if training is False:
            return K.minimum(1., self.count / self.cap) * (z * K.expand_dims(self.mean, 0))

        # get new mean and count
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # update op
        self.count.assign(new_count)
        self.mean.assign(new_mean)

        # the first few 1000 should not matter that much towards this cost
        return K.minimum(1., new_count / self.cap) * (z * K.expand_dims(new_mean, 0))

    def compute_output_shape(self, input_shape):
        return input_shape


class CovStream(Layer):
    """ 
    Maintain stream of data covariance. 

    cap refers to mainting an approximation of up to that number of subjects -- that is,
    any incoming datapoint will have at least 1/cap weight.

    If you find this class useful, please cite the original paper this was written for:
        A.V. Dalca, M. Rakic, J. Guttag, M.R. Sabuncu.
        Learning Conditional Deformable Templates with Convolutional Networks 
        NeurIPS: Advances in Neural Information Processing Systems. pp 804-816, 2019. 
    """

    def __init__(self, cap=100, **kwargs):
        self.cap = float(cap)
        super(CovStream, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create mean, cov and and count
        # See note in MeanStream.build()
        self.mean = self.add_weight(name='mean',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=False)
        v = np.prod(input_shape[1:])
        self.cov = self.add_weight(name='cov',
                                   shape=[v, v],
                                   initializer='zeros',
                                   trainable=False)
        self.count = self.add_weight(name='count',
                                     shape=[1],
                                     initializer='zeros',
                                     trainable=False)

        super(CovStream, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, training=None):
        training = _get_training_value(training, self.trainable)

        # get batch shape:
        this_bs_int = K.shape(x)[0]

        # prep for broadcasting :(
        p = tf.concat((K.reshape(this_bs_int, (1,)), K.shape(self.cov)), 0)
        z = tf.ones(p)

        # If calling in inference mode, use moving stats
        if training is False:
            return K.minimum(1., self.count / self.cap) * (z * K.expand_dims(self.cov, 0))

        x_orig = x

        # update mean
        new_mean, new_count = _mean_update(self.mean, self.count, x, self.cap)

        # x reshape
        this_bs = tf.cast(this_bs_int, 'float32')  # this batch size
        prev_count = self.count
        x = K.batch_flatten(x)  # B x N

        # new C update. Should be B x N x N
        x = K.expand_dims(x, -1)
        C_delta = K.batch_dot(x, K.permute_dimensions(x, [0, 2, 1]))

        # update cov
        prev_cap = K.minimum(prev_count, self.cap)
        C = self.cov * (prev_cap - 1) + K.sum(C_delta, 0)
        new_cov = C / (prev_cap + this_bs - 1)

        # updates
        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.cov.assign(new_cov)

        return K.minimum(1., new_count / self.cap) * (z * K.expand_dims(new_cov, 0))

    def compute_output_shape(self, input_shape):
        v = np.prod(input_shape[1:])
        return (input_shape[0], v, v)


def _mean_update(pre_mean, pre_count, x, pre_cap=None):

    # compute this batch stats
    this_sum = tf.reduce_sum(x, 0)
    this_bs = tf.cast(K.shape(x)[0], 'float32')  # this batch size

    # increase count and compute weights
    new_count = pre_count + this_bs
    alpha = this_bs / K.minimum(new_count, pre_cap)

    # compute new mean. Note that once we reach self.cap (e.g. 1000),
    # the 'previous mean' matters less
    new_mean = pre_mean * (1 - alpha) + (this_sum / this_bs) * alpha

    return (new_mean, new_count)


def _get_training_value(training, trainable_flag):
    """
    Return a flag indicating whether a layer should be called in training
    or inference mode.

    Modified from https://git.io/JUGHX

    training: the setting used when layer is called for inference.
    trainable: flag indicating whether the layer is trainable.
    """
    if training is None:
        training = K.learning_phase()

    if isinstance(training, int):
        training = bool(training)

    # If layer not trainable, override value passed from model.
    if trainable_flag is False:
        training = False

    return training


##########################################
# FFT Layers
##########################################

class FFT(Layer):
    """
    Apply the fast Fourier transform (FFT) to a tensor. Supports forward and backward
    (inverse) transforms, and the transformed axes can be specified. The first and last
    dimensions of the input tensor are supposed to indicate batches and features,
    respectively. The output tensor will be complex.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def __init__(self, axes=None, inverse=False, **kwargs):
        """
        Parameters:
            axes: Spatial axes along which to take the FFT. None means all axes.
            inverse: Whether to perform a backward (inverse) transform.
        """
        self.axes = axes
        self.inverse = inverse
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axes': self.axes,
            'inverse': self.inverse,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'
        self.axes = py.utils.normalize_axes(self.axes,
                                            input_shape,
                                            allowed=range(1, ndims + 1),
                                            none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        return utils.fftn(x, axes=self.axes, inverse=self.inverse)


class IFFT(FFT):
    """
    Apply the inverse fast Fourier transform (iFFT) to a tensor. The transformed axes can be
    specified. The first and last dimensions of the input tensor are supposed to indicate
    batches and features, respectively. The output tensor will be complex. For more information
    see ne.layers.FFT.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, inverse=True, **kwargs)


class FFTShift(Layer):
    """
    Shift the zero-frequency component to the center of the tensor.
    """

    def __init__(self, axes=None, inverse=False, **kwargs):
        """
        Parameters:
            axes: Spatial axes along which to shift the spectrum. None means all axes.
            inverse: Undo the shift operation.
        """
        self.axes = axes
        self.inverse = inverse
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'axes': self.axes,
            'inverse': self.inverse,
        })
        return config

    def build(self, input_shape):
        ndims = len(input_shape) - 2
        assert ndims in (1, 2, 3), 'only 1D, 2D, or 3D supported'
        self.axes = py.utils.normalize_axes(self.axes,
                                            input_shape,
                                            allowed=range(1, ndims + 1),
                                            none_means_all=True)
        super().build(input_shape)

    def call(self, x):
        f = tf.signal.ifftshift if self.inverse else tf.signal.fftshift
        return f(x, axes=self.axes)


class IFFTShift(FFTShift):
    """
    Undo the effect of applying FFTShift. While FFTShift and IFFTShift are identical for
    even-size tensor dimensions, their effect differs by one voxel for dimensions of odd
    size. For more information, see ne.layers.FFTShift.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, inverse=True, **kwargs)


class ComplexToChannels(Layer):
    """
    Split a complex tensor into a real tensor with features corresponding to the
    real and imaginary components.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI 
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def call(self, x):
        return utils.complex_to_channels(x)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] *= 2
        return tuple(shape)


class ChannelsToComplex(Layer):
    """
    Convert a real tensor with an even number N of features into a complex N/2-feature tensor.
    The first N/2 features will be taken as real, the last N/2 features as imaginary components.

    If you find this class useful, please cite the original paper this was written for:
        Deep-learning-based Optimization of the Under-sampling Pattern in MRI 
        C. Bahadir, A.Q. Wang, A.V. Dalca, M.R. Sabuncu.
        IEEE TCP: Transactions on Computational Imaging. 6. pp. 1139-1152. 2020.
    """

    def call(self, x):
        return utils.channels_to_complex(x)

    def compute_output_shape(self, input_shape):
        shape = list(input_shape)
        shape[-1] = shape[-1] // 2
        return tuple(shape)


##########################################
# Stochastic Sampling layers
##########################################

class SampleNormalLogVar(Layer):
    """ 
    Keras Layer: Gaussian sample given mean and log_variance

    If you find this class useful, please cite the original paper this was written for:
        Dalca AV, Guttag J, Sabuncu MR
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
        CVPR 2018. https://arxiv.org/abs/1903.03148

    inputs: list of Tensors [mu, log_var]
    outputs: Tensor sample from N(mu, sigma^2)
    """

    def __init__(self, **kwargs):
        super(SampleNormalLogVar, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SampleNormalLogVar, self).build(input_shape)

    def call(self, x):
        return self._sample(x)

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def _sample(self, args):
        """
        sample from a normal distribution

        args should be [mu, log_var], where log_var is the log of the squared sigma

        This is probably equivalent to 
            K.random_normal(shape, args[0], exp(args[1]/2.0))
        """
        mu, log_var = args

        # sample from N(0, 1)
        noise = tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)

        # make it a sample from N(mu, sigma^2)
        z = mu + tf.exp(log_var / 2.0) * noise
        return z


class GaussianNoise(Layer):
    """
    Sample and add or return Gaussian noise.

    The layer first draws a standard deviation (SD) from a uniform distribution, then samples
    noise from a normal distribution using the drawn SD. It adds the noise to the input tensor
    or returns only the noise tensor, which will have the same shape as the input tensor.

    If you find this layer useful, please cite:
        M Hoffmann, B Billot, DN Greve, JE Iglesias, B Fischl, AV Dalca
        SynthMorph: learning contrast-invariant registration without acquired images
        IEEE Transactions on Medical Imaging (TMI), 41 (3), 543-558, 2022
        https://doi.org/10.1109/TMI.2021.3116879
    """

    def __init__(self,
                 noise_min=0.01,
                 noise_max=0.10,
                 noise_only=False,
                 absolute=False,
                 axes=(0, -1),
                 seed=None,
                 **kwargs):
        """
        Parameters:
            noise_min: Minimum noise SD. Shape must be broadcastable with the output shape.
            noise_max: Maximum noise SD. Shape must be broadcastable with the output shape.
            noise_only: Return the noise tensor rather than adding it to the input.
            absolute: Instead of interpreting the SD bounds relative to the absolute maximum
                value of the input tensor, treat them as absolute values.
            axes: Input axes along which noise will be sampled with a separate SD.
            seed: Optional seed for initializing the random number generation.
        """
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.noise_only = noise_only
        self.absolute = absolute
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'noise_min': self.noise_min,
            'noise_max': self.noise_max,
            'noise_only': self.noise_only,
            'absolute': self.absolute,
            'axes': self.axes,
            'seed': self.seed,
        })
        return config

    def build(self, in_shape):
        num_dim = len(in_shape)
        self.axes = np.ravel(self.axes)
        self.axes = [ax + num_dim if ax < 0 else ax for ax in self.axes]
        assert all(0 <= ax < num_dim for ax in self.axes), 'invalid axes'

        self.rand = tf.random.Generator.from_non_deterministic_state()
        if self.seed is not None:
            self.rand.reset_from_seed(self.seed)

    def call(self, x):
        """
        Parameters:
            x: Input tensor to add noise to.
        """
        if self.noise_max == 0 and not self.noise_only:
            return x

        # Types.
        assert x.dtype.is_floating or x.dtype.is_complex, 'non-FP output type'
        real_type = x.dtype.real_dtype

        # Shapes.
        shape_out = tf.shape(x)
        shape_sd = []
        for i, _ in enumerate(x.shape):
            shape_sd.append(shape_out[i] if i in self.axes else 1)

        # Standard deviation.
        sd = self.rand.uniform(
            shape_sd, minval=self.noise_min, maxval=self.noise_max, dtype=real_type,
        )
        if not self.absolute:
            sd *= tf.reduce_max(tf.abs(x))

        # Direct sampling of complex numbers not supported.
        if x.dtype.is_complex:
            noise = tf.complex(
                self.rand.normal(shape_out, stddev=sd, dtype=real_type),
                self.rand.normal(shape_out, stddev=sd, dtype=real_type),
            )

        else:
            noise = self.rand.normal(shape_out, stddev=sd, dtype=real_type)

        return noise if self.noise_only else x + noise


class PerlinNoise(Layer):
    """
    Sample Perlin Noise.

    The function combines noise tensors at different scales by drawing them at full resolution and
    smoothing randomly. Users control the number of levels via the length of the smoothing bounds.

    At each scale, we uniformly draw a standard deviation (SD). Second, we sample noise from a
    normal distribution with that SD. Before averaging over scales, we blur each tensor over its
    spatial dimensions, keeping constant a global statistic of choice, such as the SD.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self,
                 shape=None,
                 noise_min=0.01,
                 noise_max=1,
                 fwhm_min=4,
                 fwhm_max=32,
                 isotropic=False,
                 reduce=tf.math.reduce_std,
                 out_type=tf.float32,
                 axes=None,
                 seed=None,
                 **kwargs):
        """
        Parameters:
            shape: Output shape including feature but excluding batch dimension. None means the
                shape of the input tensor (from which the layer will infer the batch size).
            noise_min: Lower bound on the sampled noise SDs, as a scalar.
            noise_max: Upper bound on the sampled noise SDs, as a scalar.
            fwhm_min: Lower bounds on the smoothing FWHMs. Scalar or iterable. One element/level.
            fwhm_max: Upper bounds on the smoothing FWHMs. Same length as `fwhm_min`.
            isotropic: Whether smoothing should be isotropic.
            reduce: TensorFlow function returning a global statistic to keep constant.
            out_type: Floating-point data type of the output tensor.
            axes: Axes along which the noise has a separate SD. As the layer always varies the SD
                along the batch dimension, passing `0` is not allowed. With `axes=-1`, each channel
                will use a separate sampling SD. With `None`, the same SD will be used for all axes.
            seed: Integer for reproducible randomization.
        """
        self.shape = shape
        self.noise_min = noise_min
        self.noise_max = noise_max
        self.fwhm_min = fwhm_min
        self.fwhm_max = fwhm_max
        self.isotropic = isotropic
        self.reduce = reduce
        self.out_type = tf.dtypes.as_dtype(out_type)
        self.axes = axes
        self.seed = seed
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'shape': self.shape,
            'noise_min': self.noise_min,
            'noise_max': self.noise_max,
            'fwhm_min': self.fwhm_min,
            'fwhm_max': self.fwhm_max,
            'isotropic': self.isotropic,
            'reduce': self.reduce,
            'out_type': self.out_type,
            'axes': self.axes,
            'seed': self.seed,
        })
        return config

    def build(self, input_shape):
        self.rand = np.random.default_rng(self.seed)

        allowed = range(1, len(input_shape))
        self.axes = py.utils.normalize_axes(self.axes, input_shape, allowed, none_means_all=False)
        super().build(input_shape)

    def call(self, x):
        """
        Parameters:
            x: Input tensor defining the batch size (and potentially shape).
        """
        shape = x.shape[1:] if self.shape is None else self.shape
        dtype = self.out_type
        return tf.map_fn(lambda x: self._single_batch(x, shape), x, fn_output_signature=dtype)

    def _single_batch(self, _, shape):
        return utils.augment.draw_perlin_full(shape,
                                              noise_min=self.noise_min,
                                              noise_max=self.noise_max,
                                              isotropic=self.isotropic,
                                              fwhm_min=self.fwhm_min,
                                              fwhm_max=self.fwhm_max,
                                              batched=False,
                                              featured=True,
                                              dtype=self.out_type,
                                              seed=self.rand.integers(np.iinfo(int).max),
                                              axes=[ax - 1 for ax in self.axes],
                                              reduce=self.reduce)


class Constant(Layer):
    """Convert an input value to a constant Keras layer with a batch dimension.

    The layer will derive the batch size from an input tensor. Without an input tensor or if you
    pass an empty list, the batch size will be one.

    If you find this layer useful, please cite:
        Anatomy-specific acquisition-agnostic affine registration learned from fictitious images
        M Hoffmann, A Hoopes, B Fischl*, AV Dalca* (*equal contribution)
        SPIE Medical Imaging: Image Processing, 12464, p 1246402, 2023
        https://doi.org/10.1117/12.2653251
    """

    def __init__(self, value, **kwargs):
        """
        Parameters:
            value: Constant value. Must be convertible to a tensor and exclude the batch dimension.
            out_type: Data type of the constant. Has to be convertible to a TensorFlow type.
                Defaults to None, meaning the type will be inferred from the passed value.
        """
        self.value = value
        super().__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update(value=self.value)
        return config

    def build(self, _):
        self.const = tf.cast(self.value, self.dtype)
        if tf.rank(self.const) == 0:
            self.const = tf.reshape(self.const, shape=(1,))

    def call(self, x=[]):
        """
        Parameters:
            x: Any input tensor that we can derive a batch size from. Without an input tensor or an
                empty list, the batch size will be one. Some Keras versions may be incompatible
                with the default argument and require that you explicitly pass an empty list.
        """
        batch = tf.maximum(tf.shape(x)[0], 1)
        shape = tf.concat(([batch], tf.shape(self.const)), axis=0)
        return tf.broadcast_to(self.const, shape)


##########################################
# HyperMorph Layers
##########################################

class HyperConv(Layer):
    """Placeholder for the historical TensorFlow hyper-convolution layer."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'HyperConv and related layers relied on TensorFlow private ops. '
            'They are not yet ported to the pure Keras backend.'
        )


class HyperConv2D(HyperConv):
    pass


class HyperConv3D(HyperConv):
    pass


class HyperConvFromDense(HyperConv):
    pass


class HyperDense(Layer):
    """Placeholder for the historical TensorFlow hyper-dense layer."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            'HyperDense and related layers relied on TensorFlow private ops. '
            'They are not yet ported to the pure Keras backend.'
        )


class HyperDenseFromDense(HyperDense):
    pass
