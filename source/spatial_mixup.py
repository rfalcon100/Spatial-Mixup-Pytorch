'''
The file contains spatial audio data augmentation methods, including:
    - Directional Loudness modifications
    
This has been updated with the a new option for the backend, using the spatial filterbank.

Author: Ricardo Falcon
2022
'''


import os
import torch
import numpy as np
import math
from typing import Union, Optional, Tuple
import torchaudio
import fnmatch

import utils
import plots
import spaudiopy as spa


class DirectionalLoudness(object):
    """ Class to modify the directional loudness of an ambisonics recording.

    TODO: Add formula here

    Backend:
        The backend defines how the matrices Y and W are computed. In the original SpatialMixup paper
        we use a basic form , where Y is a matrix of spherical harmonics pointing to the directions of
        the grid; and W is Y transposed. This is equivalent to a hypercardiods pointing to the grid. 
        We then apply and scaling factor to the resulting matrix. 
        
        We also include a spatial filtebank backend, where Y and W and specially constructed matrices to
        have perfect reconstruction, and can include other patterns for the beamforming. This allows for
        more extreme transformations in G.

    References:
     F. Zotter and M. Frank, Ambisonics: A Practical 3D Audio Theory for Recording, Studio Production,
     Sound Reinforcement, and Virtual Reality  Franz Zotter Matthias Frank. Springer, 2019.
     
     Hold, C., Politis, A., Mc Cormack, L., & Pulkki, V. (2021). Spatial Filter Bank Design in the 
     Spherical Harmonic Domain. Proceedings of European Signal Processing Conference, August, 106–110.

     Hold, C., Schlecht, S. J., Politis, A., & Pulkki, V. (2021). SPATIAL FILTER BANK IN THE SPHERICAL 
     HARMONIC DOMAIN : RECONSTRUCTION AND APPLICATION. IEEE Workshop on Applications of Signal Processing 
     to Audio and Acoustics (WASPAA), 1(1).

"""
    def __init__(self, order_input: int = 1, t_design_degree: int = 3, order_output: int = 1,
                 G_type: str = 'random_diag', G_values : Union[np.ndarray, torch.Tensor] = None,
                 device: str = 'cpu', T_pseudo_floor=1e-8, backend='basic', w_pattern='hypercardioid'):
        assert t_design_degree > 2 * order_output, 'The t-design degree should be > 2 * N_{tilde} of the output order '

        self._Y = None
        self._G = None
        self._W = None
        self._T_mat = None

        self.grid = DirectionalLoudness.get_grid(degree=t_design_degree)
        self.n_directions = self.grid.shape[0]
        self.order_input = order_input
        self.order_output = order_output
        self.G_type = G_type
        self.device = device
        self.backend = backend  # {'spat_filterbank', 'basic'}
        self.w_pattern = w_pattern
        
        self.G_cap_center = None  # For spherical caps
        self.G_cap_width = None
        self.G_g1 = None
        self.G_g2 = None
    

        # Initialize the matrices
        self.Y, self.W = self.compute_Y_and_W(order=self.order_output)
        self.G = self.compute_G(G_type=G_type, G_values=G_values)
        self.T_mat = self.compute_T_mat()

    def __repr__(self):
        rep = "DirectionalLoudness with: \n"
        rep += f'Device = {self.device} \n'
        rep += f'order_input = {self.order_input} \n'
        rep += f'order_output = {self.order_output} \n'
        rep += f'backend = {self.backend} \n'
        rep += f'w_pattern = {self.w_pattern} \n'
        rep += f'n_directions = {self.n_directions} \n'
        rep += f'G_type = {self.G_type} \n'
        rep += f'Spherical_cap params: \n\t{self.G_cap_center}\n, \t{self.G_cap_width}\n, \t{self.G_g1}\n, \t{self.G_g2}'

        return rep

    @staticmethod
    def get_grid(degree: int):
        """Returns the cartesian coordinates for a t_design.
        This represents a grid of directions, that uniformly samples the unit sphere."""
        t_design = spa.grids.load_t_design(degree=degree)
        return t_design

    def reset_G(self, G_type: str = 'identity',
                  G_values : Union[np.ndarray, torch.Tensor] = None,
                  capsule_center: Optional[torch.Tensor] = None,
                  capsule_width: Optional[float] = np.pi/2,
                  g1_db: Optional[float] = 0,
                  g2_db: Optional[float] = -10):

        tmp = self.compute_G(G_type=G_type,
                             G_values=G_values,
                             capsule_center=capsule_center,
                             capsule_width=capsule_width,
                             g1_db=g1_db,
                             g2_db=g2_db)
        self.G = tmp
        self.T_mat = self.compute_T_mat()
        self.G_type = G_type

    def compute_Y_and_W(self, order: int = 1) -> torch.Tensor:
        """ Computes the reconstruction matrix Y, and beamforming matrix W """

        tmp_directions = utils.vecs2dirs(self.grid)
        if self.backend == 'basic':
            Y = spa.sph.sh_matrix(order, tmp_directions[:, 0], tmp_directions[:, 1], SH_type='real', weights=None)
            W = np.copy(Y)
            # W = W * (4 * np.pi / (self.order_input +1 )**2)
        elif self.backend == 'spatial_filterbank':
            assert self.order_input == self.order_output, 'When using spatial filterbank, the input and output orders should be the same'
            
            # Weights for polar patterns
            if self.w_pattern.lower() == "cardioid":
                c_n = spa.sph.cardioid_modal_weights(self.order_output)
            elif self.w_pattern.lower() == "hypercardioid":
                c_n = spa.sph.hypercardioid_modal_weights(self.order_output)
            elif self.w_pattern.lower() == "maxre":
                c_n = spa.sph.maxre_modal_weights(self.order_output, True)  # works with amplitude compensation and without!
            else:
                raise ValueError(f'ERROR: Unknown w_pattern type: {self.w_pattern} . Check spelling? ')
            [W, Y] = spa.sph.design_spat_filterbank(self.order_output, tmp_directions[:, 0], tmp_directions[:, 1], c_n, 'real', 'perfect')

        Y = Y.astype(np.double)
        W = W.astype(np.double)
        W = torch.from_numpy(W)
        Y = torch.from_numpy(Y)

        return Y.to(self.device), W.to(self.device)

    def compute_G(self, G_type: str = 'identity',
                  G_values : Union[np.ndarray, torch.Tensor] = None,
                  capsule_center: Optional[torch.Tensor] = None,
                  capsule_width: Optional[float] = np.pi/2,
                  g1_db: Optional[float] = 0,
                  g2_db: Optional[float] = -10) -> torch.Tensor:
        """
        Returns a matrix G with the gains for each direction.
        Currently supports only this types:
            -- Identity matrix.
            -- Random diagonal.
            -- Fixed values (set diagonal to a vector of values)
            -- Random matrix (including values outside the diagonal)
            -- spherical_cap - Equation 3.18 of the Ambisonics book.
            """
        G = np.eye(self.n_directions)

        if G_type == 'identity':
            pass

        elif G_type == 'random_diag':
            values = np.random.rand(self.n_directions)
            np.fill_diagonal(G, values)

        elif G_type == 'fixed':
            values = G_values
            np.fill_diagonal(G, values)

        elif G_type == 'random':
            G = np.random.rand(self.n_directions, self.n_directions)

        elif fnmatch.fnmatch(G_type, 'spherical_cap*'):
            G = torch.eye(self.n_directions, device=self.device)

            if capsule_center is None:
                if fnmatch.fnmatch(G_type, '*_soft'):
                    cap_type = 'soft'
                else:
                    cap_type = 'hard'
                capsule_center, capsule_width, g1_db, g2_db = DirectionalLoudness.draw_random_spherical_cap(spherical_cap_type=cap_type, device=self.device)
                    
            assert capsule_center.shape[-1] == 2, 'The capsule center should be [1, 2] vector of azimuth and elevation.'
            assert capsule_width > 0 and capsule_width < 1 * np.pi, 'The capsule width should be within 0 and 1*pi radians'

            tmpA = utils.sph2unit_vec(capsule_center[:,0], capsule_center[:,1]).to(self.device).double()
            tmpB = torch.from_numpy(self.grid).to(self.device).double()
            tmpA = tmpA.repeat(tmpB.shape[0], 1)
            assert tmpA.shape == tmpB.shape, 'Wrong shape for the capsule center or the grid coordinates'

            # Dot product batch wise, between the angles of the capsule and the grid points
            tmp = torch.bmm(tmpA.unsqueeze(dim=1), tmpB.unsqueeze(dim=2)).squeeze(dim=-1).to(self.device)
            g1, g2 = spa.utils.from_db(g1_db), spa.utils.from_db(g2_db)

            values = g1 * torch.heaviside(tmp - torch.cos(capsule_width/2), torch.ones_like(tmp) * g1) + \
                     g2 * torch.heaviside(torch.cos(capsule_width/2) -tmp, torch.ones_like(tmp) * g2)
            G.diagonal().copy_(values.squeeze()) # replace diagonal inline

            self.G_cap_center = capsule_center
            self.G_cap_width = capsule_width
            self.G_g1 = g1_db
            self.G_g2 = g2_db

        if isinstance(G, np.ndarray):
            G = torch.from_numpy(G)
            G = G.to(self.device)

        return G.double()

    @staticmethod
    def draw_random_spherical_cap(spherical_cap_type='soft', device='cpu') -> Tuple[torch.Tensor, torch.Tensor, float, float]:
        """ Draws the parameters for a random spherical cap using:
        spherical_cap_type = 'hard':
            - cap_center = Uniform between [0, 2*pi] for azimuth, and [-pi/2, pi/2]
            - cap_width = Uniform between [pi/4 and pi]
            - g1 = 0
            ######- g1 = Exponential with high = 0, low = -6
            - g2 = Uniform [-20, -6]

        spherical_cap_type = 'soft':
            - cap_center = Uniform between [0, 2*pi] for azimuth, and [-pi/2, pi/2]
            - cap_width = Uniform between [pi/4 and pi]
            - g1 = 0   
            ####- g1 = Exponential with high = 0, low = -6   
            - g2 = Exponential with high = -3, low = -6
            """

        cap_center = torch.stack([torch.rand(1, device=device) * 2 * np.pi ,
                                  torch.rand(1, device=device) * np.pi - np.pi/2], dim=-1)
        cap_width = torch.rand(1, device=device) * (np.pi - np.pi/4) + np.pi/4
        #g1 = - utils.sample_exponential(0, 6, shape=[1], device=device)
        g1 = 0

        if spherical_cap_type == 'hard':
            g2 = torch.rand(1, device=device) * -(20 - 6) - 6
        elif spherical_cap_type == 'soft':
            g2 = - utils.sample_exponential(3, 6, shape=[1], device=device)
        else:
            raise ValueError(f'Unsupported spherical cap type: {spherical_cap_type} ')

        return cap_center, cap_width, g1, g2

    def compute_T_mat(self) -> torch.Tensor:
        """ Computes the full transformation matrix T_mat, and applies the scaling if selected."""
        if False:
            print('Debugging')
            print(self.Y.shape)
            print(self.G.shape)
            print(self.W.shape)
            print(self.Y)
            print(self.G)
            print(self.W)
        
        tmp = torch.matmul(self.Y.transpose(1,0), self.G)
        T_mat = torch.matmul(tmp, self.W)
        if self.backend == 'basic':
            scale = 4 * np.pi / self.n_directions  # TODO August 05, this works ok , except for input_order > 1
            T_mat = scale * T_mat

        return T_mat.double()

    def process(self, X: Union[torch.Tensor, np.ndarray],
                mix_alpha: float = 0, do_scaling=False) -> np.ndarray:
        """ Applies the transformation matrix T to an input signal matrix X"""

        # This methods is wrong
        # USe the forward now
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X)
        if X.shape[0] > X.shape[1]:  # Channels first format
            X = X.permute([1,0])
        if self.W is not None and self.Y is not None:
            assert X.shape[0] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'

        # Matrix multiplication with the T_mat
        X = X.to(self.device).double()
        X_hat = torch.matmul(self.T_mat, X)

        # mix_alpha = 0 --> only augmented data, 1 --> only original data
        if mix_alpha > 0:
            # Add zeros to input data if output has higher order
            channels_out, channels_in = X_hat.shape[-2], X.shape[-2]
            if channels_out != channels_in:
                X = torch.cat([X, torch.zeros(channels_out - channels_in, X.shape[-1])], dim=-2)
            X_hat = mix_alpha * X + (1-mix_alpha) * X_hat

        if X_hat.shape[0] < X_hat.shape[1]:  # Back to timesteps first format
            X_hat = X_hat.permute([1,0])
        return X_hat

    def forward(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).double()
        if X.shape[-2] > X.shape[-1]:  # Channels first format
            warnings.warn('WARNING: It seems that the input tensor X is NOT in channels-first format')
        if self.W is not None and self.Y is not None:
            assert X.shape[-2] == self.W.shape[-1], 'Wrong shape for input signal or matrix W.'
        assert self.T_mat.shape[-1] == X.shape[-2], 'Wrong shape for input signal or matrix T.'
        
        out = torch.matmul(self.T_mat, X)
        
        return out
        
    def plot_response(self, plot_channel=0, title=None, plot3d=True, plot2d=True, plot_matrix=False, show_plot=True, do_scaling=True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Plots the polar response of the transformation matrix.

            Method can be {'cross', 'linear'} for either a crosssection fo azimuth and elevaiton, or a 2d
            linear projection.

        Returns:
            Tuple of tensors of [azi, azi_response, ele, ele_response] for azimuth and elevation responses.
            """
        if plot_matrix:
            plots.plot_transform_matrix(self.T_mat, xlabel='Input ACN', ylabel='Output ACN')
            
        if plot3d:
            responses_3d = plots.plot_sphere_points(self.T_mat, self.order_input, plot_channel=plot_channel, show_plot=show_plot, do_scaling=do_scaling)

        if plot2d:
            azis, response_azi, eles, response_ele = plots.plot_matrix2polar(self.T_mat, self.order_input,
                                                           plot_channel=plot_channel,  title=title, show_plot=show_plot, do_scaling=do_scaling)
            return azis, response_azi, eles, response_ele

    def plot_W(self):
        import matplotlib.pyplot as plt
        spa.plots.sh_coeffs_subplot(self.W.detach().cpu().numpy())
        plt.show()

    def plot_Y(self):
        import matplotlib.pyplot as plt
        spa.plots.sh_coeffs_subplot(self.Y.detach().cpu().numpy())
        plt.show()

    def plot_T_mat(self):
        import matplotlib.pyplot as plt
        spa.plots.sh_coeffs_subplot(self.T_mat.detach().cpu().numpy())
        plt.show()
        
    def plot_G(self):
        plots.plot_transform_matrix(self.G, xlabel='Input direction', ylabel='Output direction', title='G_matrix')
        
        
def test_directional_loudness():
    """
    This is a test for the Directional_Loudness class. It mostly tests:
    - When G = identity, the transformation matrix T is close to identity.
    - When G = identity, a processed signal has low reconstruction error
    - Wehn 
    """
    params = {'t_design_degree': 4,
         'G_type': 'identity',
         'order_output': 1,
         'order_input': 1,
         'w_pattern': 'cardioid' }
    transform = DirectionalLoudness(t_design_degree=params['t_design_degree'], 
                                    G_type=params['G_type'],
                                    order_output=params['order_output'], 
                                    order_input=params['order_input'])
    
    assert torch.all(torch.isclose(transform.T_mat, torch.eye(transform.T_mat.shape))), 'T_mat should be close to identity'
    
    
    
if __name__ == '__main__':
    test_directional_loudness()
    
    