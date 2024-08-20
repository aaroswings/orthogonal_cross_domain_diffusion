import torch
from torch.nn import Module
from torch.utils import tensorboard

import numpy as np
from pathlib import Path
import random
from types import SimpleNamespace
from tqdm import tqdm
import shutil

from diffusion import Diffusion
from trainer import EMA
from trainer import LinearWarmupLR
from trainer import Dataset
from trainer import UtilFunctions

class SimpleTrainer(Module):
    def __init__(
        self, 
        net: Module, 
        artifact_dir: str,
        ema_config: SimpleNamespace,
        optim_config: SimpleNamespace,
        dataset_config: SimpleNamespace,
        data_loader_config: SimpleNamespace,
        diffusion_type: type,
        seed: int = 0,
        save_checkpoint_every: int = 1,
        draw_val_sample_every: int = 1,
        val_loss_every: int = 1000
    ):
        super().__init__()
        self.device = torch.device('cuda')
        self.net: Module = net.to(self.device)
        self.artifact_dir = artifact_dir
        self.save_checkpoint_every = save_checkpoint_every
        self.diffusion_type = diffusion_type
        self.start_step = 0
        
        self.register_buffer('step',torch.tensor([0]))
        self.val_loss_every = val_loss_every
        self.draw_val_sample_every = draw_val_sample_every

        # Set up modules that Trainer depends on
        self.writer = tensorboard.SummaryWriter(self.artifact_dir)
        self.scaler = torch.cuda.amp.GradScaler()

        # Optimization #########################
        self.optimizer = torch.optim.Adam([*self.net.parameters()], lr=0.0, betas=optim_config.adam_betas, eps=1e-7)
        
        self.lr_scheduler = LinearWarmupLR.LinearWarmupLR(
            optimizer=self.optimizer,
            lr_after_warmup=optim_config.lr_scheduler.lr_after_warmup,
            final_lr=optim_config.lr_scheduler.final_lr,
            warmup_steps=optim_config.lr_scheduler.warmup_steps,
            total_scheduled_steps=optim_config.lr_scheduler.total_scheduled_steps,
        )

        self.ema = EMA.EMA(
            self.net,
            beta=ema_config.beta,
            update_after_step=ema_config.update_after_step,
            update_every=ema_config.update_every
        )

        # Data ###############################
        if self.diffusion_type == Diffusion.ConditionalDiffusion:
            print('Initializing Trainer for paired training with ConditionalDiffusion.')
            train_dataset = Dataset.PairedRGBImageDataset(
                file_dirs=dataset_config.train_file_dirs,
                image_size=dataset_config.size,
                contrast_augment_std=dataset_config.contrast_augment_std,
                brightness_augment_std=dataset_config.brightness_augment_std
            )

            val_dataset = Dataset.PairedRGBImageDataset(
                file_dirs=dataset_config.val_file_dirs,
                image_size=dataset_config.size,
                hflip_augmentation_probability=0.0
            )
        elif self.diffusion_type == Diffusion.CrossDiffusion:
            print('Initializing Trainer for partially-paired training with CrossDiffusion.')
            dir_a_paired, dir_b_paired, dir_a_unpaired, dir_b_unpaired = dataset_config.train_file_dirs
            train_dataset = Dataset.PartiallyPairedRGBImageDataset(
                paired_file_dirs=(dir_a_paired, dir_b_paired),
                unpaired_file_dirs=(dir_a_unpaired, dir_b_unpaired),
                image_size=dataset_config.size,
                contrast_augment_std=dataset_config.contrast_augment_std,
                brightness_augment_std=dataset_config.brightness_augment_std
            )

            val_dataset = Dataset.PartiallyPairedRGBImageDataset(
                paired_file_dirs=(dir_a_paired, dir_b_paired),
                unpaired_file_dirs=(dir_a_unpaired, dir_b_unpaired),
                image_size=dataset_config.size,
                contrast_augment_std=dataset_config.contrast_augment_std,
                brightness_augment_std=dataset_config.brightness_augment_std
            )

        self.loaders = {
            'train': torch.utils.data.DataLoader(train_dataset, data_loader_config.batch_size, shuffle=True, num_workers=data_loader_config.num_workers),
            'val': torch.utils.data.DataLoader(val_dataset, data_loader_config.batch_size, shuffle=True, num_workers=data_loader_config.num_workers)
        }

        self.loader_iters = {
            'train': iter(self.loaders['train']),
            'val': iter(self.loaders['val'])
        }

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

    def _next_batch(self, loader='train'):
        # fancy way of ignoring dataset size and epochs
        try:
             data = next(self.loader_iters[loader])
        except StopIteration:
            self.loader_iters[loader] = iter(self.loaders[loader])
            data = next(self.loader_iters[loader])
        return data

    def fit(self, diffusion, num_steps=1000):
        # Decouple training steps from length of dataset
        last_step_i = num_steps - 1
        print(f'Training from step {self.step.item()} to step {num_steps}...')
        for step in tqdm(range(self.step, num_steps + 1)):
            if step <= last_step_i:
                train_loss = self._forward_backward(diffusion)
            self._update_train_artifacts(train_loss, diffusion)
            self.step += 1
        print('Done.')

    def _update_train_artifacts(self, train_loss, diffusion):
        self.writer.add_scalar('Loss/train', train_loss, self.step)
        if self.step % self.val_loss_every == 0:
            self.writer.add_scalar('Loss/val', self._get_val_batch_loss(diffusion=diffusion), self.step)

        if self.step > self.start_step and self.step % self.save_checkpoint_every == 0:
            self.save_checkpoint()

        # Should save a sample at the start of training as a smoke test
        if self.step % self.draw_val_sample_every == 0:
            self._save_val_sample(diffusion)

    def _checkpoint_paths_from_step(self, step):
        dir = self._checkpoint_dir_from_step(step)
        file_path = dir / 'train_checkpoint.ckpt'
        return dir, file_path

    def _checkpoint_dir_from_step(self, step):
        return Path(self.artifact_dir) / f'train_step_{step}_checkpoint'
    
    def _sample_dir_from_step(self, step):
         return Path(self.artifact_dir) / f'train_step_{step}_samples'

    def load_from_path(self, path):
        file_path = Path(path)
        print(f'Loading checkpoint from {file_path}...')
        checkpoint = torch.load(file_path)
        self.load_state_dict(checkpoint['trainer_state_dict'])
        self.start_step = self.step.item()

    def load_from_step_checkpoint(self, step):
        dir, file_path = self._checkpoint_paths_from_step(step)
        print(f'Loading checkpoint from {file_path}...')
        self.load_from_path(file_path)
        print('Done.')

    def save_checkpoint(self):
        dir, file_path = self._checkpoint_paths_from_step(self.step.item())
        print(f'Creating direcotry and saving checkpoint to {file_path}...')
        dir.mkdir()
        torch.save({'trainer_state_dict': self.state_dict()}, file_path)
        print('Done.')

    def _draw_val_sample(self, diffusion):
        assert type(diffusion) == self.diffusion_type
        self.net.eval()
        data = self._next_batch('val')
        if isinstance(diffusion, Diffusion.ConditionalDiffusion):
            sample_sequence = self._conditional_diffusion_sample(data, diffusion)
        elif isinstance(diffusion, Diffusion.CrossDiffusion):
            sample_sequence = self._cross_diffusion_sample(data, diffusion)
        else:
            raise NotImplementedError
        return sample_sequence

    def _conditional_diffusion_sample(self, data, diffusion: Diffusion.ConditionalDiffusion):
        a, b = self._batch_to_device(data, Diffusion.ConditionalDiffusion)
        sample_sequence = diffusion.sample(self.ema, a)
        return sample_sequence
    
    def _cross_diffusion_sample(self, data, diffusion: Diffusion.CrossDiffusion, do_sample_A_from_B=False):
        a, b, a_mask, b_mask = self._batch_to_device(data, Diffusion.CrossDiffusion)

        if do_sample_A_from_B:
            sample_sequence = diffusion.sample(self.ema, x0_A=None, x0_B=b)
        else:
            sample_sequence = diffusion.sample(self.ema, x0_A=a, x0_B=None)
        return sample_sequence

    def _save_val_sample(self, diffusion, save_intermediates=True):
        out_dir = self._sample_dir_from_step(self.step.item())
        
        if out_dir.exists():
            # Count up the batch index of existing files as a starting point to
            # name new files - don't overwrite samples that already exist.
            # Existing files have to follow this function's naming convention.
            sample_count_offset = 0

            for f in out_dir.iterdir():
                if f.is_file():
                    file_sample_count = [int(s) for s in f.name.split('_') if s.isnumeric()][0]
                    if file_sample_count > sample_count_offset:
                        sample_count_offset = file_sample_count
            sample_count_offset += 1
            print(f'Saving new samples in existing directory {out_dir} from batch index {sample_count_offset}')
        else:
            sample_count_offset = 0
            print(f'Creating directory {out_dir} to save samples...')
            out_dir.mkdir()

        print('Creating sample with network...')
        sample_sequence = self._draw_val_sample(diffusion)

        print('Saving samples...')

        saved_paths = []
        
        for i, batch_of_outputs in enumerate(sample_sequence):
            batch_of_outputs = UtilFunctions.format_bchw_network_output_to_images(batch_of_outputs)
            if save_intermediates or i == len(sample_sequence) - 1:
                for j, chw_sample in enumerate(batch_of_outputs):
                    im = UtilFunctions.chw_tensor_to_pil_image(chw_sample)
                    save_path = out_dir / f'{j + sample_count_offset}_{i}.png'
                    im.save(save_path, format='PNG')
                    saved_paths.append(save_path)
        print(f'Saved files: {saved_paths}')
        print('Done.')

    def save_samples(self, diffusion, save_intermediates=False, num_samples=1):
        for i in range(num_samples):
            self._save_val_sample(diffusion, save_intermediates)

    @torch.no_grad()
    def _get_val_batch_loss(self, diffusion):
        assert type(diffusion) == self.diffusion_type
        self.net.eval()
        return self._get_loss(data=self._next_batch('val'), diffusion=diffusion).detach().cpu().numpy()

    def _get_loss(self, data, diffusion):
        assert type(diffusion) == self.diffusion_type
        self.net.train()
        if isinstance(diffusion, Diffusion.ConditionalDiffusion):
            loss = self._conditional_diffusion_loss(data, diffusion) 
        elif isinstance(diffusion, Diffusion.CrossDiffusion):
            loss = self._cross_diffusion_loss(data, diffusion)
        else:
            raise NotImplementedError
        return loss

    def _conditional_diffusion_loss(self, data, diffusion: Diffusion.ConditionalDiffusion):
        a, b = self._batch_to_device(data, Diffusion.ConditionalDiffusion)
        return diffusion.loss(self.net, a, b)
    
    def _cross_diffusion_loss(self, data, diffusion: Diffusion.CrossDiffusion):
        a, b, a_mask, b_mask = self._batch_to_device(data, Diffusion.CrossDiffusion)
        return diffusion.loss(self.net, a, b, a_mask, b_mask)

    def _batch_to_device(self, data, diffusion_type: type):
        if diffusion_type == Diffusion.ConditionalDiffusion:
            a, b = data
            a, b = a.to(self.device), b.to(self.device)
            return a, b
        elif diffusion_type == Diffusion.CrossDiffusion:
            a, b, a_mask, b_mask = data
            a, b = a.to(self.device), b.to(self.device)
            a_mask, b_mask = a_mask.to(self.device), b_mask.to(self.device)
            return a, b, a_mask, b_mask
        else:
            raise NotImplementedError

    def _forward_backward(self, diffusion):
        self.optimizer.zero_grad()
        data = self._next_batch('train')
        loss = self._get_loss(data, diffusion)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_value_(self.net.parameters(), 1.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.update()
        self.lr_scheduler.update()
        return loss.detach().cpu().numpy()
