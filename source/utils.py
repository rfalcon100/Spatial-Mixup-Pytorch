'''
The file containd utility functions.

Author: Ricardo Falcon
2021
'''

import torch
import torch.nn as nn
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
import scipy
import math
from typing import Iterable, Tuple, TypeVar, Callable, Any, List, Union
from matplotlib.colors import to_rgb


import spaudiopy as spa
from sphere_fibonacci_grid_points import sphere_fibonacci_grid_points


def colat2ele(colat: Union[float, torch.Tensor]) -> torch.Tensor:
    """Transforms colatitude to elevation (latitude). In radians.

    The polar angle on a Sphere measured from the North Pole instead of the equator.
    The angle $\phi$ in Spherical Coordinates is the Colatitude.
    It is related to the Latitude $\delta$ by $\phi=90^\circ-\delta$.
    """
    ele = math.pi/2 - colat
    return ele


def ele2colat(ele: Union[float, torch.Tensor]) -> torch.Tensor:
    """Transforms colatitude to elevation (latitude). In radians.

    The polar angle on a Sphere measured from the North Pole instead of the equator.
    The angle $\phi$ in Spherical Coordinates is the Colatitude.
    It is related to the Latitude $\delta$ by $\phi=90^\circ-\delta$.
    """
    colat = math.pi/2 - ele
    return colat


def vecs2dirs(vecs, positive_azi=True, include_r=False, use_elevation=False):
    """Helper to convert [x, y, z] to [azi, colat].
    From Spaudiopyy, but with safe case when r=0"""
    azi, colat, r = spa.utils.cart2sph(vecs[:, 0], vecs[:, 1], vecs[:, 2], steady_colat=True)
    if positive_azi:
        azi = azi % (2 * np.pi)  # [-pi, pi] -> [0, 2pi)
    if use_elevation:
        colat = colat2ele(colat)
    if include_r:
        output = np.c_[azi, colat, r]
    else:
        output = np.c_[azi, colat]
    return output


def sph2unit_vec(azimuth: Union[float, torch.Tensor], elevation: Union[float, torch.Tensor]) -> torch.Tensor:
    """ Transforms spherical coordinates into a unit vector .
    Equaiton 2.1 of
    [1]M. Kronlachner, “Spatial transformations for the alteration of ambisonic recordings”.
    """
    assert torch.all(azimuth >= 0) and torch.all(azimuth <= 2*np.pi), 'Azimuth should be in radians, between 0 and 2*pi'

    x = torch.cos(azimuth) * torch.cos(elevation)
    y = torch.sin(azimuth) * torch.cos(elevation)
    z = torch.sin(elevation)

    return torch.stack([x,y,z], dim=-1)


def unit_vec2sph(angle_x: Union[torch.Tensor, float],
                angle_y: Union[torch.Tensor, float],
                angle_z: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Transforms angles over each axis (unit vector notation) to spherical angles.
    Equation 2.2 of
    [1]M. Kronlachner, “Spatial transformations for the alteration of ambisonic recordings”.
    """

    azimuth = torch.arctan(angle_y / angle_x)
    elevation = torch.arctan(angle_z / (torch.sqrt(angle_x**2 + angle_y**2)))

    return azimuth, elevation


def rms(x: torch.Tensor):
    """Computes th RMS (root-mean-squared) for each channel in the signal tensor, in dB.

    Parameters
    ----------
    x : Tensor
        Input signal in format [..., channels, timesteps]

    """
    tmp = torch.sqrt(torch.mean(x ** 2, dim=-1))
    t = torchaudio.transforms.AmplitudeToDB(stype='amplitude', top_db=80)
    #tmp = torchaudio.functional.amplitude_to_DB(tmp, multiplier=10, amin=80, db_multiplier=1)
    tmp = t(tmp)
    return tmp


def sample_beta(alpha: float = 0.0, beta: float = None, shape=[1]):
    """ Draws a sample from the beta distribution. """
    if beta is None:
        beta = alpha
    if alpha > 0 and beta > 0:
        try:
            dist = torch.distributions.beta.Beta(alpha, beta)
            lambda_var = dist.sample(sample_shape=shape)
        except:
            print(f'{alpha}     {beta}')
            raise
    else:
        if beta == 0:
            prob = alpha if alpha <= 1 else 1
        else:
            prob = (1 - beta) if beta <= 1 else 0
        lambda_var = torch.distributions.bernoulli.Bernoulli(probs=prob).sample(sample_shape=shape)
    return lambda_var


def sample_geometric(low=0, high=1, shape=[1]):
    """ Draw a random sample from a geometric distribution between low and high.
    This returns integers.

    From Bergstra and Bengio:
        "We will use the phrase drawn gemietrically from A to B for 0 < A < B to mean drawing
        uniformly in the log domain between log(A) and log(B), exponentiating to get a number
        between A dn B, and then rounding to the nearest integer."
    """

    tmp_low, tmp_high = torch.log(tmp * low), torch.log(tmp * high)
    tmp = torch.rand(size=shape)
    sample = torch.round(torch.exp(tmp_low + (tmp_high - tmp_low) * tmp))
    return sample


def sample_exponential(low=0, high=1, shape=[1], device='cpu'):
    """ Draw a random sample from an exponential distribution between low and high.
    This returns floats.

    From Bergstra and Bengio:
        This is like geometricSample, but without rounding to the nearest integer.

    """
    eps = 1e-10
    tmp = torch.ones(shape, device=device)
    tmp_low, tmp_high = torch.log(tmp * low + 1e-15), torch.log(tmp * high)
    tmp = torch.rand(size=shape, device=device)
    sample = torch.exp(tmp_low + (tmp_high - tmp_low) * tmp)
    sample[sample<= eps] = 0
    return sample


def load_SOFA(order, filename='data/HRIR_L2354.sofa', url='http://sofacoustics.org/data/database/thk/HRIR_L2354.sofa'):
    """ Returns the selected HRTF sofa file encoded in spherical harmonics domain 
    This downloads the sofa file is needed. 
    """
    from os.path import exists
    import wget
    if not exists(filename):
        response = wget.download(url, filename)
    try:
        irs, fs = spa.IO.sofa_to_sh(filename, order, 'real')
    except:
        raise ValueError('Sofa file not found')
    return irs, fs


def sh_sig_to_binaural(sig: torch.Tensor, device, order: int = 1, sofa='data/HRIR_L2354.sofa', do_normalize=True):
    """ Binauralizes the input sh signal by doing a 1d convolution with an HRTF in sh domain """
    brir, fs = load_SOFA(order=order, filename=sofa)
    brir = torch.from_numpy(brir).float() # [2, (N+1)**2, brir_length]  
    assert isinstance(sig, torch.Tensor), 'ERROR: Signal to binaualize should be a torch.Tensor'
    assert sig.shape[-2] == brir.shape[-2], 'ERROR: Signal and HRTF should have the same number of channels, i.e. order'
    
    binaural_convolver = nn.Conv1d(in_channels=sig.shape[-2], out_channels=brir.shape[-3],
                   kernel_size=(brir.shape[-1]), stride=1, bias=False,
                   padding='same', groups=1, device=device)
    binaural_convolver.weight.data[..., :] = torch.flip(brir, dims=[-1])  # Flip along time axis because nn.Conv1d is correlation, not conv

    with torch.no_grad():
        sig_binaural = binaural_convolver(sig[None,...].float())
        if do_normalize:
            sig_binaural = sig_binaural / sig_binaural.abs().max()
    
    return sig_binaural


def get_noise_signal(t_seconds=3, fs=48000, order=1, sources_gain=[1, 0.6], sources_direction=[(0, np.pi/2), (np.pi/2, np.pi/4)]):
    """ Generates a white noise signal in spherical harmonic domain with the selcted number of noise sources.
    This is useful to visualize the rms , but a bit difficult to listen to
    """
    from numpy.random import default_rng
    rng = default_rng()
    grid = spa.grids.load_t_design(degree=4)
    tmp_directions = vecs2dirs(grid) 

    # Add noise_floor
    sig = 1 * rng.standard_normal((t_seconds*fs, (order+1)**2))
    sig = 0
    # Add sources
    for ii, gain in enumerate(sources_gain):
        #tmp_source = 1 * rng.standard_normal((t_seconds*fs, 1)) * spa.sph.sh_matrix(params['order_input'], tmp_directions[ii + 1,0], tmp_directions[ii + 1,1], 'real').conj()
        tmp_source = gain * rng.standard_normal((t_seconds*fs, 1)) * spa.sph.sh_matrix(order, sources_direction[ii][0], sources_direction[ii][1], 'real').conj()
        sig += tmp_source
    sig = sig.T 
    sig = sig / np.max(np.abs(sig)) # normalize
    
    return sig


def get_fake_sound_scene(t_seconds=3, fs=48000, order=1, 
                         sources_gain=[1, 0.6], 
                         sources_freq=[200, 640],
                         sources_direction=[(0, np.pi/2), (np.pi/2, np.pi/4)]):
    """ Generates a signal in spherical harmonic domain, with the selected number of sinusaidal sources """ 
    from numpy.random import default_rng
    rng = default_rng()
    grid = spa.grids.load_t_design(degree=4)
    tmp_directions = vecs2dirs(grid) 
    time_vec = np.arange(0, t_seconds, 1/fs)[..., None]

    # Add noise_floor
    #sig = 0.01 * rng.standard_normal((t_seconds*fs, (order+1)**2))
    sig = 0
    # Add sources
    for ii, (gain, freq, direction) in enumerate(zip(sources_gain, sources_freq, sources_direction)):
        tmp_source = np.sin(2 * np.pi * freq *time_vec) + 0.5 * np.sin(2 * np.pi * freq * time_vec * 1.1) + 0.2 * np.sin(2 * np.pi * freq * time_vec * 3) 
        tmp_source = gain * tmp_source * spa.sph.sh_matrix(order, direction[0], direction[1], 'real').conj()
        sig += tmp_source
    sig = sig.T 
    sig = sig / np.max(np.abs(sig)) # normalize
    
    return sig


def test_beta_distributions(sampler='pytorch'):
    if sampler=='pytorch':
        sampler_f = sample_beta
    else:
        sampler_f = mixup_data

    alphas = [0, 0.25, 0.5, 0.75, 1, 2, 10]
    betas = alphas
    trials = 5000
    results = np.zeros((trials, len(alphas), len(betas)))
    bins = 25
    for counter_alpha, this_alpha in enumerate(alphas):
        for counter_beta, this_beta in enumerate(betas):
            print(f'Drawing samples: alpha = {this_alpha}, beta = {this_beta}')
            #for i in range(trials):
            #    results[i, counter_alpha, counter_beta] = sampler_f(this_alpha, this_beta)
            results[:, counter_alpha, counter_beta] = sampler_f(this_alpha, this_beta, shape=[trials])

    fig, axes = plt.subplots(len(alphas), len(alphas), figsize=(10,10), sharex=True, sharey=True)
    for ii, this_alpha in enumerate(alphas):
        for jj, this_beta in enumerate(betas):
            dat = results[:, ii, jj]
            ax = axes[ii, jj]
            ax.hist(dat, bins=bins, density=True, log=True)
            ax.set_title(f'a={this_alpha}, b={this_beta}')
    fig.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_beta_distributions()
    print('Finished test')


