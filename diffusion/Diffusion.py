import torch
from torch import Tensor
from torch.nn import Module

import math
from tqdm import trange

from diffusion.UtilFunctions import *

def _noise_schedule_t(t, noise_schedule):
    t = t.clamp(1e-6, 1-1e-6)
    if noise_schedule == 'cosine-alpha':
        return 1. - torch.cos((t) * math.pi / 2)
    elif noise_schedule == 'linear':
        return t
    else:
        raise ValueError
    
def _eta_schedule_t(t, schedule_name, eta):
    if schedule_name == 'cosine':
        return 1 - torch.cos((1 - t) * math.pi / 2)
    elif schedule_name == 'constant':
        return torch.ones_like(t) * eta
    elif schedule_name== 'none':
        return torch.zeros_like(t)
    else:
        raise ValueError
    
class ConditionalDiffusion(Module):
    """
    Algorithm 1 and 2, Method 1.
    """
    def __init__(self,
        sampling_num_timesteps: int = 1000,
        sampling_num_latents_to_return: int = 1,
        noise_schedule: str = 'linear',
        noise_replacement_schedule: str = 'none',
        noise_replacement_eta: float = 0.3,
        normalize_eps: bool = True
    ):
        super().__init__()
        self.sampling_num_timesteps = sampling_num_timesteps
        self.sampling_num_latents_to_return = sampling_num_latents_to_return
        self.noise_schedule = noise_schedule
        self.noise_replacement_schedule = noise_replacement_schedule
        self.noise_replacement_eta = noise_replacement_eta
        self.normalize_eps = normalize_eps

    def loss(self, net, x0_A, x0_B):
        """
        Args:
            net: The neural network. Forward pass input "z" of shape (B, C_A + C_B, H, W), output of shape (B, C_B, H, W)
            x0_A: Samples from domain A in a tensor of shape (B, C_A, H, W)
            x0_B: Samples from domain B in a tensor of shape (B, C_B, H, W)
        Returns:
            MSE loss.
        """
        with torch.no_grad():
            # Setup noise schedule
            t = torch.rand_like(x0_B[:, 0, 0, 0]) # (B, 1, 1, 1) tensor of values on the interval [0,1]

            # Diffuse
            eps = quantile_clip(torch.randn_like(x0_B), 0.999) # quantile clip for stability
            alpha, sigma = t_to_alpha_sigma(t)

            z = alpha * x0_B + sigma * eps
            z_cond = torch.concat([x0_A, z], dim=1)
            
            v = alpha * eps - sigma * x0_B

        # Loss on current velocity prediction
        with torch.cuda.amp.autocast():
            v_pred = net(z_cond, t)
            loss = torch.nn.functional.mse_loss(v_pred, v)

        return loss

    @torch.no_grad()
    def sample(self, net, x0_A):
        # Return values
        ret = []
        if self.sampling_num_latents_to_return > 0:
            save_intermediate_every = self.sampling_num_timesteps // self.sampling_num_latents_to_return
        steps_since_last_intermediate_saved = 0

        # Time schedule
        ts_linear = torch.linspace(0, 1, self.sampling_num_timesteps + 1).to(x0_A)
        etas = _eta_schedule_t(ts_linear, schedule_name=self.noise_replacement_schedule, eta=self.noise_replacement_eta)
        ts_scheduled = _noise_schedule_t(ts_linear, self.noise_schedule)

        # reverse schedules
        ts_reverse = ts_scheduled.flip(0) 
        etas = etas.flip(0)

        alphas, sigmas = t_to_alpha_sigma(ts_reverse)

        # eps and x
        eps = quantile_clip(torch.randn_like(x0_A), 0.995)
        z = torch.rand_like(x0_A) * 2. - 1.
        pred = torch.rand_like(x0_A)

        for i in trange(self.sampling_num_timesteps):
            z = pred * alphas[i] + eps * sigmas[i]
            z_cond = torch.concat([x0_A, z], dim=1)
            
            # Assign x0_B, eps from network's prediction of current velocity
            with torch.cuda.amp.autocast():
                v_pred = net(z_cond, ts_reverse[i])
            
            pred = alphas[i] * z - sigmas[i] * v_pred
            eps = sigmas[i] * z + alphas[i] * v_pred

            if i < self.sampling_num_timesteps - 1 and etas[i]:
                eps = eps * (1 - etas[i]).sqrt() + torch.randn_like(eps) * etas[i].sqrt()

            # Save intermediate latent state?
            if self.sampling_num_latents_to_return > 0 and steps_since_last_intermediate_saved == save_intermediate_every:
                ret.append(torch.clip(pred, -1., 1))
                steps_since_last_intermediate_saved = 0
            else:
                steps_since_last_intermediate_saved += 1


        ret.append(torch.clip(pred, -1., 1.))
        return ret
            

class CrossDiffusion(Module):
    """
    Model of the diffusion process that represents the latent variable as being composed of three basis vectors: 
        eps, A, & B (Gaussian noise, domain A sample, domain B sample).

    Spherical coordinate math gives the following variables:
        r(theta, phi):  Combination of eps, A, and B vectors. 
                        For diffusion model purposes, the latent variable - same role as z(t) in ConditionalDiffusion.
        v_ab:   Vector perpendicular to r. A step in the v_ab direction (increasing phi) moves r toward the B pole, a step in the -v_ab direction moves r toward the A pole.
        v_eps:  Vector perpendicular to r. A step in the v_eps direction (increasing theta) moves r toward the A/B plane, in other words denoising r.
                Same role as t in ConditionalDiffusion. theta = 0 = complete noise, theta = pi/2 = some blending of A and B, no noise.
        theta:  Increasing theta from 0 to pi/2 denoises r. theta is meant for use in range [0, pi/2].
        phi: Increasing phi from 0 to pi/2 moves the AB component of r from domain A to domain B. phi is meant for use in range [0, pi/2]

        About the network in cross-diffusion:
        In a conditional diffusion model, the conditioning tensor (x0_A for cross-domain translation) is concatenated into the input.
        In effect, the network has two z's as input, z_1 and z_2, but z_1 is fixed at time t=1. So we ignore the eps/domain A plane.
        For Method 2 to have similar conditioning, it can be conditioned on two r vectors, r_1 and r_2. But we can free both of them from being fixed at a pole.
        This means we have to account for two theta parameters and two phi parameters, one of each for each r vector.
        The network then predicts two velocity vectors for each r: v_ab_1, v_eps_1 for r_1, and v_ab_2, v_eps_2 for r_2.
        For RGB images, the network should be customized to have 6 input channels (two r's) and 12 output channels (four v's).

    """
    def __init__(
        self,
        sampling_num_timesteps: int = 1000,
        sampling_num_latents_to_return: int = 1,
        noise_schedule: str = 'linear',
        sampling_method: str = 'none',
        noise_replacement_schedule: str = 'none',
        noise_replacement_eta: float = 0.0,
        normalize_eps: bool = True
    ):
        super().__init__()
        self.sampling_num_timesteps = sampling_num_timesteps
        self.sampling_num_latents_to_return = sampling_num_latents_to_return
        self.noise_schedule = noise_schedule
        self.sampling_method = sampling_method
        self.noise_replacement_schedule = noise_replacement_schedule
        if self.noise_replacement_schedule != 'none':
            raise NotImplementedError
        self.noise_replacement_eta = noise_replacement_eta
        self.normalize_eps = normalize_eps

    def _basis_to_local(self, x0_A, x0_B, eps, theta, phi):
        r = torch.sin(theta) * torch.cos(phi) * x0_A + \
            torch.sin(theta) * torch.sin(phi) * x0_B + \
            torch.cos(theta) * eps
        
        # theta_hat
        v_eps = torch.cos(theta) * torch.cos(phi) * x0_A + \
                torch.cos(theta) * torch.sin(phi) * x0_B - \
                torch.sin(theta) * eps
        
        # phi_hat
        v_ab =  -torch.sin(phi) * x0_A + torch.cos(phi) * x0_B

        return r, v_eps, v_ab

    def _local_to_basis(self, r, v_eps, v_ab, theta, phi):
        A = torch.sin(theta) * torch.cos(phi) * r + \
            torch.cos(theta) * torch.cos(phi) * v_eps - \
            torch.sin(phi) * v_ab
        
        B = torch.sin(theta) * torch.sin(phi) * r + \
            torch.cos(theta) * torch.sin(phi) * v_eps + \
            torch.cos(phi) * v_ab
        
        eps = torch.cos(theta) * r - torch.sin(theta) * v_eps
        return A, B, eps
    
    # def _select_samples_by_mask(self, x, x_mask):
    #     """
    #     Args:
    #         x: (B, C, H, W) tensor of samples. Some might be missing.
    #         x_mask: (B, 1, 1, 1) tensor of masks where x_mask == 0 means the sample is missing, x_mask == 1 means the sample is not missing.
    #     Returns:
    #         x_choose: Only the samples from x at indices where x_mask == 1.
    #     """
    #     out_channels = x_mask.sum().item()
    #     # Flatten x's channel/spatial dimensions to one dimension
    #     x_flat = x.view(x.size(0), -1)
    #     # Same shape as x_flat
    #     mask_view_as_x_flat = (x_mask == 1).view(x_mask.size(0), 1).broadcast_to(x_mask.size(0), x_flat.size(1))

    #     x_choose = x_flat[mask_view_as_x_flat]
    #     x_choose = x_choose.view((-1, x.size(1), x.size(2), x.size(3)))
    #     return x_choose

    # def _insert_fake_noise_for_missing_samples(self, x, x_mask):
    #     """
    #     Args:
    #         x: (B, C, H, W) tensor of samples. Some might be missing.
    #         x_mask: (B, 1, 1, 1) tensor of masks where x_mask == 0 means the sample is missing, x_mask == 1 means the sample is not missing.
    #     Returns:
    #         x_replaced: x, but possibly modified such that the values of the missing samples are replaced with uniform noise with the variance of the not-missing samples.
        
    #     Using this to fix x0_A and x0_B should prevent bias in the network's normalization layers.
    #     """
    #     x_flat = x.view(x.size(0), -1)
    #     # Standard deviation of only the samples indicated as not-missing by x_mask == 1
    #     real_x_std = self._select_samples_by_mask(x, x_mask).std()
    #     rand = torch.rand_like(x) * 2.0 - 1.0
    #     rand = rand / rand.std() * real_x_std
    #     x_replaced = torch.where(x_mask == 0, rand, x)
    #     return x_replaced
    
    def _get_random_thetas_phis(self, x, A_masks, B_masks):
        pi_half = torch.tensor(torch.pi / 2).to(x.device)
        theta = torch.rand_like(A_masks) * pi_half # "time"
        phi = torch.rand_like(A_masks) * pi_half # blend of A and B

        phi = torch.where(A_masks == 0, pi_half, phi) # Train only on x0_B for that sample in batch.
        phi == torch.where(B_masks == 0,  0.0, phi) # Train only on x0_A for that sample in batch.

        return theta, phi
    
    def loss(self, net, x0_A, x0_B, A_masks, B_masks, p_binarize_phi = 0.5, t_clip_min=1e-8):
        """
        Args:
            net: The neural network. Forward pass input "r" of shape (B, C_A + C_B, H, W), output of shape (B, C_B, H, W) Set network constructor's t_dim_in=2 for the U-net to accept a 2d "time" parameter.
            x0_A: Samples from domain A in a tensor of shape (B, C_A, H, W)
            x0_B: Samples from domain B in a tensor of shape (B, C_B, H, W)
            A_masks: Tensor of shape (B, 1, 1, 1) of 0 or 1's, where 0=x0_A is missing for that sample, 1=x0_A is not missing for that sample.
            B_masks: Same as A_masks, but for x0_B.
            p_binarize_phi: Probability of binarizing phi. Prioritize training so that r1 and r2 are not blended on the AB dimension.
        In partially paired training, x0_A or x0_B may be missing. This gives three cases:
            x0_A missing: Set phi equal to pi/2 and replace x0_A with uniform noise. Train only on x0_B. Loss is zeroed on v_ab prediction.
            x0_B missing: Set phi equal to 0 and replace x0_B with noise. Train only on x0_A. Loss is zeroed on v_ab prediction.
            Neither missing: theta and phi are randomly, uniformly sampled on [0, pi/2]. Loss is backpropogated on v_eps and v_ab derived from network's prediction of r.
        """
        # Masks
        assert (A_masks + B_masks).all() # Ensures no samples where both A and B are missing. Only A or B.
        is_paired = torch.where(A_masks == B_masks, True, False) # (B, 1, 1, 1) tensor with 0's at indices of samples that are unpaired, 1's otherwise
        # and operation for A_masks and B_masks - will be 0 at the index of a sample where A_masks == 0 or B_masks == 0
        pi_half = torch.tensor(torch.pi / 2).to(x0_A)
        with torch.no_grad():
            # Get random theta and phi
            theta1 = torch.rand_like(A_masks) * pi_half # "time" of guidance datapoint r1
            theta2 = torch.rand_like(A_masks) * pi_half # "time" of target datapoint r2
            phi1 = torch.rand_like(A_masks) * pi_half # blend of A and B
            phi2 = torch.rand_like(A_masks) * pi_half # blend of A and B
            
            # Maybe threshold phi to 0.0 or pi/2
            if p_binarize_phi > 0.0:
                binarize_mask = torch.rand_like(phi1) < p_binarize_phi
                phi1[binarize_mask] = torch.where(phi1[binarize_mask] < pi_half / 2, 0.0, pi_half)
                phi2[binarize_mask] = pi_half - phi1[binarize_mask]
                
            # Train only on x0_B for batches where x0_A is missing. 
            phi1 = torch.where(A_masks == 0, pi_half, phi1)
            phi2 = torch.where(A_masks == 0, pi_half, phi2)
            # Train only on x0_A for batches where x0_B is missing. 
            phi1 = torch.where(B_masks == 0, 0.0, phi1)
            phi2 = torch.where(B_masks == 0, 0.0, phi2)

            eps = quantile_clip(torch.randn_like(x0_A), 0.999)

            r_1, v_eps_1, v_ab_1 = self._basis_to_local(x0_A, x0_B, eps, theta1, phi1)
            r_2, v_eps_2, v_ab_2 = self._basis_to_local(x0_A, x0_B, eps, theta2, phi2)

            # "time" is a tensor of shape (B, k) as expected by a network's FourierFeatures module
            angles = torch.cat([theta1, phi1, theta2, phi2], dim=1)
            angles = angles / (pi_half) # range [0, 1]
            angles = torch.clamp(angles, t_clip_min)

            rs = torch.cat([r_1, r_2], dim=1)

        with torch.cuda.amp.autocast():
            v_eps_2_pred, v_ab_2_pred = net(rs, angles).tensor_split(2, dim=1)

            x0_A_pred, x0_B_pred, eps_pred = self._local_to_basis(r_2, v_eps_2_pred, v_ab_2_pred, theta2, phi2)
            _, v_eps_1_pred,  v_ab_1_pred = self._basis_to_local(x0_A_pred, x0_B_pred, eps_pred, theta1, phi1)
            loss_v_eps_1 = torch.nn.functional.mse_loss(v_eps_1_pred, v_eps_1)
            loss_v_ab_1 = torch.nn.functional.mse_loss(v_ab_1_pred, v_ab_1)
            loss_v_eps_2 = torch.nn.functional.mse_loss(v_eps_2_pred, v_eps_2)
            loss_v_ab_2 = torch.nn.functional.mse_loss(v_ab_2_pred, v_ab_2)
            loss = (loss_v_eps_1 + loss_v_ab_1 + loss_v_eps_2 + loss_v_ab_2) / 4.0

        return loss
    
    @torch.no_grad()
    def sample(self, net, x0_A=None, x0_B=None, r_1_t = 1., t_clip_min=1e-8):
        """
        Args:
            net: the neural network.
            x0_A: optional sample from domain A. If provided, it will be used in r_1, the guidance datapoint for the model's sampling.
            x0_B: optional sample from domain B. If x0_A is not provided and x0_B is provided, x0_B will be used in r_1.
            r_1_t: How clean to keep r_1 from noise. r_1_t means r_1 is only the clean guidance image.
        Generate a domain A sample from B, or a domain B sample from A.
        Net should accept input tensor [r_1, r_2].
        r_1 is the conditioning tensor. Then theta_1 = pi/2 at every sampling step.
        x0_A is present but x0_B is missing, we are trying to infer x0_B. 
        At every sampling step, phi_1 should be 0 so that r_1 is a blending of noise and sample A, and phi_2 should be pi/2 so that r_2 is a blending of noise and sample B.
        Conversely, if x0_B is present but x0_A is missing, we are trying to infer x0_A from x0_B.
        then, at every sampling step, phi_1 should be pi/2 so that r_2 is a blending of noise and sample B, and phi_2 should be 0 so that r_2 is a blending of noise and sample A.
        theta_2 takes the role of "time" t from the conditioning sampler and increases from 0 to pi/2 during sampling. 
        """
        pi_half = torch.tensor([torch.pi / 2]).to(x0_A.device).view(1, 1, 1, 1)
        zero = pi_half * 0.0

        theta1 = pi_half * r_1_t
        if x0_A is None:
            sample_A_from_B = True
            x0_A = torch.zeros_like(x0_B)
            r_1 = x0_B
            phi1 = pi_half 
            phi2 = zero
        else:
            sample_A_from_B = False # sampling B from A 
            x0_B = torch.zeros_like(x0_A)
            r_1 = x0_A
            phi1 = zero # guidance input r1 is blend of x0_A and noise
            phi2 = pi_half

        eps = quantile_clip(torch.randn_like(x0_A), 0.995)
        ret = []
        if self.sampling_num_latents_to_return > 0:
            save_intermediate_every = self.sampling_num_timesteps // self.sampling_num_latents_to_return
        steps_since_last_intermediate_saved = 0

        # Time schedule
        ts_linear = torch.linspace(0, 1, self.sampling_num_timesteps).to(x0_A.device)
        ts_scheduled = _noise_schedule_t(ts_linear, self.noise_schedule)
        # Note about theta: Theta increasing to pi/2 denoises the image. In conditional diffusion, *decreasing* t from 1 to 0 denoises the image.
        # Careful little detail.
        theta2s = ts_scheduled * torch.pi / 2

        etas = _eta_schedule_t(ts_linear, schedule_name=self.noise_replacement_schedule, eta=self.noise_replacement_eta)
        etas = etas.flip(0)

        for i in trange(self.sampling_num_timesteps):
            # At theta2 = 0, v_ab_2_pred is roughly -x0_A if the network is well trained, when predicting x0_B.
            # At theta2 = 0, v_eps_2_pred is roughly -x0_B, if the network is well trained, when predicting x0_B.
            
            if i == self.sampling_num_timesteps // 2:
                pass # hey I'm a debug breakpoint
            if i == self.sampling_num_timesteps - 1:
                last_step = True # maybe delete me later

            theta2 = theta2s[i].view(1, 1, 1, 1)
            
            r_2, _, _ = self._basis_to_local(x0_A, x0_B, eps, theta2, phi2)

            angles = torch.cat([theta1, phi1, theta2, phi2], dim=1)
            angles = angles / pi_half # range [0, 1]
            rs = torch.cat([r_1, r_2], dim=1)

            with torch.cuda.amp.autocast():
                v_eps_2_pred, v_ab_2_pred = net(rs, angles).tensor_split(2, dim=1)

            x0_A, x0_B, eps = self._local_to_basis(r_2, v_eps_2_pred, v_ab_2_pred, theta2, phi2)
            
            if self.normalize_eps:
                eps / eps.std(dim=(1, 2, 3), keepdim=True)
                
            # Replace some noise?
            if i < self.sampling_num_timesteps - 1 and etas[i]:
                eps = eps * (1 - etas[i]).sqrt() + torch.randn_like(eps) * etas[i].sqrt()

            # Save intermediate latent state?
            if self.sampling_num_latents_to_return > 0 and steps_since_last_intermediate_saved == save_intermediate_every:
                ret.append(torch.clip((x0_A if sample_A_from_B else x0_B), -1., 1))
                steps_since_last_intermediate_saved = 0
            else:
                steps_since_last_intermediate_saved += 1

        ret.append(torch.clip((x0_A if sample_A_from_B else x0_B), -1., 1.))
        return ret


        

        
