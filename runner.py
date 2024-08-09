import argparse
import json
from types import SimpleNamespace

from diffusion import Diffusion
from net import Net
from trainer import Trainer

parser = argparse.ArgumentParser('DiffusionModel')
parser.add_argument('--profile') # name of a json file in ./profiles without extension
parser.add_argument('--program', choices=['train','sample'])
args = parser.parse_args()

if __name__ == '__main__':
    with open(f'profiles/{args.profile}.json') as f:
        config = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    # Todo load trainer state from checkpoint here - add key to profiles for this

    net = Net.UNet(
        in_channels=config.net.in_channels,
        out_channels=config.net.out_channels,
        emb_dim=config.net.emb_dim,
        dims=config.net.dims,
        dropouts=config.net.dropouts,
        num_resblocks=config.net.num_resblocks,
        use_attn=config.net.use_attn,
        in_conv_kernel_size=config.net.in_conv_kernel_size
    )

    diffusion = Diffusion.ConditionalDiffusion(
        sampling_num_timesteps=config.diffusion.sampling_num_timesteps,
        sampling_num_latents_to_return=config.diffusion.sampling_num_latents_to_return,
        sampling_clip_latent=config.diffusion.sampling_clip_latent,
        normalize_x0_A=config.diffusion.normalize_x0_A,
        center_x0_A=config.diffusion.center_x0_A,
        continuous_partial_sample_normalization_x=config.diffusion.continuous_partial_sample_normalization_x
    )

    if args.program == 'train':
        # Todo code to load trainer state from checkpoint here
        trainer = Trainer.SimpleTrainer(
            net=net,
            artifact_dir=f'out/{args.profile}',
            ema_config=config.trainer.ema,
            optim_config=config.trainer.optim,
            dataset_config=config.trainer.dataset,
            data_loader_config=config.trainer.data_loader,
            seed=config.trainer.seed,
            save_checkpoint_every=config.trainer.save_checkpoint_every,
            draw_val_sample_every=config.trainer.draw_val_sample_every
        )
        if config.load_checkpoint_step > 0:
            trainer.load_checkpoint(config.load_checkpoint_step)
        print("Fitting model...")
        trainer.fit(num_steps=config.final_train_step, diffusion=diffusion)