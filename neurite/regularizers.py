"""
tensorflow/keras regularizers for the neuron project

If you use this code, please cite 
Dalca AV, Guttag J, Sabuncu MR
Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation, 
CVPR 2018

or for the transformation/interpolation related functions:

Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
MICCAI 2018.

Contact: adalca [at] csail [dot] mit [dot] edu

Copyright 2020 Adrian V. Dalca

Licensed under the Apache License, Version 2.0 (the "License"); you may not use
this file except in compliance with the License. You may obtain a copy of the
License at http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
"""

from keras import ops

from .utils import soft_delta


def soft_l0_wrap(wt=1.):

    def soft_l0(x):
        """
        maximize the number of 0 weights
        """
        flat = ops.reshape(x, (-1,))
        nb_weights = ops.sum(ops.ones_like(flat))
        nb_zero_wts = ops.sum(soft_delta(flat))
        nb_weights = ops.cast(nb_weights, nb_zero_wts.dtype)
        return wt * (nb_weights - nb_zero_wts) / nb_weights

    return soft_l0
