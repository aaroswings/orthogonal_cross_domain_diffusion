import argparse
import json
from types import SimpleNamespace

from diffusion import Diffusion
from net import Net
from trainer import Trainer

parser = argparse.ArgumentParser('DiffusionModel')
parser.add_argument('--profile') # name of a json file in ./profiles without extension
parser.add_argument('--program', choices=['train','sample'])
parser.add_argument('--num_sample_batches', type=int)
parser.add_argument('--diffusion_type', choices=['conditional_diffusion', 'cross_diffusion'])
parser.set_defaults(diffusion_type='conditional_diffusion')
parser.add_argument('--sample_save_intermediates', dest='sample_save_intermediates', action='store_true')
parser.add_argument('--no-sample_save_intermediates', dest='sample_save_intermediates', action='store_false')
parser.set_defaults(sample_save_intermediates=True)
args = parser.parse_args()

if __name__ == '__main__':
    with open(f'profiles/{args.profile}.json') as f:
        config = json.loads(f.read(), object_hook=lambda d: SimpleNamespace(**d))

    artifact_dir = f'out/{args.profile}/train' if config.artifact_dir is None else config.artifact_dir

    net = Net.UNet(
        in_channels=config.net.in_channels,
        out_channels=config.net.out_channels,
        emb_dim=config.net.emb_dim,
        dims=config.net.dims,
        dropouts=config.net.dropouts,
        num_resblocks=config.net.num_resblocks,
        use_attn=config.net.use_attn,
        in_conv_kernel_size=config.net.in_conv_kernel_size,
        t_dim_in=config.net.t_dim_in
    )

    if args.diffusion_type == 'conditional_diffusion':
        diffusion = Diffusion.ConditionalDiffusion(
            sampling_num_timesteps=config.diffusion.sampling_num_timesteps,
            sampling_num_latents_to_return=config.diffusion.sampling_num_latents_to_return,
            noise_schedule=config.diffusion.noise_schedule,
            noise_replacement_schedule=config.diffusion.noise_replacement_schedule,
            noise_replacement_eta=config.diffusion.noise_replacement_eta
        )
    if args.diffusion_type == 'cross_diffusion':
        diffusion = Diffusion.CrossDiffusion(
            sampling_num_timesteps=config.diffusion.sampling_num_timesteps,
            sampling_num_latents_to_return=config.diffusion.sampling_num_latents_to_return,
            noise_schedule=config.diffusion.noise_schedule,
            noise_replacement_schedule=config.diffusion.noise_replacement_schedule,
            noise_replacement_eta=config.diffusion.noise_replacement_eta
        )

    trainer = Trainer.SimpleTrainer(
        net=net,
        artifact_dir=artifact_dir,
        ema_config=config.trainer.ema,
        optim_config=config.trainer.optim,
        dataset_config=config.trainer.dataset,
        diffusion_type=type(diffusion),
        data_loader_config=config.trainer.data_loader,
        seed=config.trainer.seed,
        save_checkpoint_every=config.trainer.save_checkpoint_every,
        draw_val_sample_every=config.trainer.draw_val_sample_every
    )

    if config.load_from_checkpoint_path is not None:
        trainer.load_from_path(config.load_from_checkpoint_path)
        trainer.artifact_dir = artifact_dir
    elif config.load_checkpoint_step is not None and config.load_checkpoint_step> 0:
        trainer.load_from_step_checkpoint(config.load_checkpoint_step)

    if args.program == 'train':
        print("Fitting model...")
        trainer.fit(num_steps=config.final_train_step, diffusion=diffusion)

    elif args.program == 'sample':
        trainer.save_samples(diffusion, save_intermediates=args.sample_save_intermediates, num_samples=args.num_sample_batches)
