import torch
import math
from typing import Optional

"""
t: continuous float value in [0,1], batch shape (B,)
step: long index of the current timestep, [0, T]
alpha, sigma: values corresponding to t, batch shape (B, 1, 1, 1)
"""

def log_clip_x(x: torch.Tensor, multiplier=1.0):
    x = torch.where(x < 1, x, x.log() * multiplier + 1)
    x = torch.where(x > -1, x, -(-x).log() * multiplier - 1)
    return x

def noisy_correction_out_of_bounds_upper_(x: torch.Tensor):
    # The idea is that the model is always in error when it predicts values outside of -1, 1 for x,
    # and we can fix that by randomly pushing back those out-of-bounds values with noise.
    # Notice that this is in-place!
    
    # iterate along batch dimension
    for i, sample in enumerate(x):
        oob_std = x[i][x[i] > 1.0].std()
        comp_noise = torch.randn_like(sample) * oob_std

        # Make all noise negative
        comp_noise = torch.where(comp_noise < 0, comp_noise, -comp_noise)
        x[i][x[i] > 1.0] += comp_noise[x[i] > 1.0]

def noisy_correction_out_of_bounds_lower_(x: torch.Tensor):
    # The idea is that the model is always in error when it predicts values outside of -1, 1 for x,
    # and we can fix that by randomly pushing back those out-of-bounds values with noise.
    # Notice that this is in-place!
    
    # iterate along batch dimension
    for i, sample in enumerate(x):
        oob_std = x[i][x[i] < -1.0].std()
        comp_noise = torch.randn_like(sample) * oob_std

        # Make all noise positive
        comp_noise = torch.where(comp_noise > 0, comp_noise, -comp_noise)
        x[i][x[i] < -1.0] += comp_noise[x[i] < -1.0]


def t_to_alpha_sigma(t):
    if not isinstance(t, torch.Tensor):
        t = torch.Tensor(t)
    if t.dim() == 0:
        t = t.view(1, 1, 1, 1)
    elif t.dim() == 1:
        t = t[:, None, None, None]
    else:
        raise ValueError('phi should be either a 0-dimensional float or 1-dimensional float array')
    
    clip_min = 1e-9
    
    alpha = torch.clip(torch.cos(t * math.pi / 2), clip_min, 1.)
    sigma = torch.clip(torch.sin(t * math.pi / 2), clip_min, 1.)

    return alpha, sigma


def sigma_dynamic_clip(x, sigma):
    """
    Intended as an option for x0 clipping.
    sigma = 0 at t = 0, clip to [-1, 1]
    """
    minmax = sigma + 1.
    return torch.clip(x, -minmax, minmax)

def replace_eps_noise_and_normalize(eps: torch.Tensor, alpha: float = 0.0) -> torch.Tensor:
    if alpha > 0.0:
        eps = torch.randn_like(eps) * alpha + eps * (1 - alpha)
    return eps / eps.std(dim=(1, 2, 3), keepdim=True)

def scale_by_minmax(x: torch.Tensor, a=-1, b=1):
    """
    Compress an image into the range -1, 1
    """
    x_flat = x.view(x.size(0), -1)
    min_x = x_flat.min(dim=1).values.view(-1, 1, 1, 1)
    max_x = x_flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (b - a) * (x - min_x) / (max_x - min_x) + a