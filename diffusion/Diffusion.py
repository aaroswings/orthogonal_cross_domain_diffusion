import torch
from torch import Tensor
from torch.nn import Module

from tqdm import tqdm

from diffusion.UtilFunctions import *

class ConditionalDiffusion(Module):
    """
    Algorithm 1 and 2, Method 1.
    """
    def __init__(self,
        sampling_num_timesteps: int = 1000,
        sampling_num_latents_to_return: int = 1,
        sampling_clip_latent: str = None,
        normalize_x0_A: bool = False,
        center_x0_A: bool = False,
        continuous_partial_sample_normalization_x: str = 'none',
        sampling_use_cosine_noise_schedule: bool = False,
        sampling_noise_churn_schedule: str = 'none',
        sampling_noisy_correction_out_of_bounds: bool = False,
        sampling_scale_x_by_minmax: bool = False
    ):
        super().__init__()
        self.sampling_num_timesteps = sampling_num_timesteps
        self.sampling_num_latents_to_return = sampling_num_latents_to_return
        self.sampling_clip_latent = sampling_clip_latent
        self.normalize_x0_A = normalize_x0_A
        self.center_x0_A = center_x0_A
        self.continuous_partial_sample_normalization = continuous_partial_sample_normalization_x
        self.sampling_use_cosine_noise_schedule = sampling_use_cosine_noise_schedule
        self.sampling_noise_churn_schedule = sampling_noise_churn_schedule
        self.sampling_noisy_correction_out_of_bounds = sampling_noisy_correction_out_of_bounds
        self.sampling_scale_x_by_minmax = sampling_scale_x_by_minmax

    def loss(self, net, x0_A, x0_B):
        """
        Args:
            net: The neural network. Input to forward pass of shape (B, C_A + C_B, H, W), output of shape (B, C_B, H, W)
            x0_A: Samples from domain A in a tensor of shape (B, C_A, H, W)
            x0_B: Samples from domain B in a tensor of shape (B, C_B, H, W)
        Returns:
            MSE loss.
        """
        with torch.no_grad():
            if self.normalize_x0_A:
                x0_A = x0_A / x0_A.std(dim=(1, 2, 3), keepdim=True)
            if self.center_x0_A:
                x0_A = x0_A - x0_A.mean(dim=(1, 2, 3), keepdim=True)
            eps = torch.randn_like(x0_B)
            t = torch.rand_like(x0_B[:, 0, 0, 0]) # (B, 1, 1, 1) tensor of values on the interval [0,1]
            t = t.clamp(1e-2, 1.-1e-2)
            alpha, sigma = t_to_alpha_sigma(t)

            if self.continuous_partial_sample_normalization == 'x0_B':
                x0_B = x0_B / (x0_B.std(dim=(1, 2, 3), keepdim=True) * sigma + (1. - sigma))

            z = alpha * x0_B + sigma * eps

            # Network is conditioned on clean samples from domain A and corrupted samples from domain B, concatenated into one tensor on the channel dimension.
            z_cond = torch.concat([x0_A, z], dim=1)

            v = alpha * eps - sigma * x0_B # Network being trained to predict v.

        with torch.cuda.amp.autocast():
            v_pred = net(z_cond, t)
            loss: Tensor = torch.nn.functional.mse_loss(v_pred, v)

        return loss

    @torch.no_grad()
    def sample(self, net, x0_A):
        if self.sampling_num_latents_to_return > 0:
            save_intermediate_every = self.sampling_num_timesteps // self.sampling_num_latents_to_return
        steps_since_last_intermediate_saved = 0

        if self.normalize_x0_A:
                x0_A = x0_A / x0_A.std(dim=(1, 2, 3), keepdim=True)
        if self.center_x0_A:
            x0_A = x0_A - x0_A.mean(dim=(1, 2, 3), keepdim=True)
        
        eps = torch.randn_like(x0_A)
        x0_B_pred = torch.randn_like(x0_A)

        timesteps = torch.linspace(1, 0, self.sampling_num_timesteps + 1)[:-1].to(x0_A)

        # reversing time
        ret = []

        for step, t in tqdm(enumerate(timesteps)):
            if self.sampling_use_cosine_noise_schedule:
                # "(1-t)" would be replaced with "t" if we hadn't flipped the order of the steps for convenience,
                # meaning if we used timesteps = torch.linspace(0, 1...) instead of timesteps = torch.linspace(1, 0...)
                t = 1 - torch.cos((1 - t) * math.pi / 2)

            # Establish latent diffusion variable states for timestep t
            alpha, sigma = t_to_alpha_sigma(t)
            if self.continuous_partial_sample_normalization == 'x0_B':
                x0_B_pred = x0_B_pred / (x0_B_pred.std(dim=(1, 2, 3), keepdim=True) * sigma + (1. - sigma))

            if self.sampling_noise_churn_schedule == 'cosine':
                # Replace eps according to a schedule
                rho = 1 - torch.cos((1 - t) * math.pi / 2)
                eps = torch.randn_like(eps) * rho.sqrt() + eps * (1. - rho.sqrt())

            z = alpha * x0_B_pred + sigma * eps
            z_cond = torch.concat([x0_A, z], dim=1)

            # Save intermediate latent state?
            if self.sampling_num_latents_to_return > 0 and steps_since_last_intermediate_saved == save_intermediate_every:
                ret.append(x0_B_pred)
                steps_since_last_intermediate_saved = 0
            else:
                steps_since_last_intermediate_saved += 1
            
            # Get prediction from network
            with torch.cuda.amp.autocast():
                v_pred = net(z_cond, t)
                x0_B_pred = alpha * z - sigma * v_pred
                # eps_pred = sigma * z + alpha * v_pred

            if self.sampling_clip_latent == 'absolute':
                x0_B_pred = torch.clip(x0_B_pred, -1., 1.)
            elif self.sampling_clip_latent == 'log_clip_x_static':
                x0_B_pred = log_clip_x(x0_B_pred, multiplier=1.0)
            elif self.sampling_clip_latent == 'log_clip_x_scheduled':
                x0_B_pred = log_clip_x(x0_B_pred, multiplier=sigma)

            if self.sampling_scale_x_by_minmax:
                x0_B_pred = scale_by_minmax(x0_B_pred)

            if self.sampling_noisy_correction_out_of_bounds:
                noisy_correction_out_of_bounds_lower_(x0_B_pred)
                noisy_correction_out_of_bounds_upper_(x0_B_pred)
        ret.append(x0_B_pred)
        return ret
            

            