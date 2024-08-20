import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import numpy as np
import os
from PIL import Image, ImageEnhance
import random
from typing import Tuple

from trainer import UtilFunctions

class BaseDataset(Dataset):
    def __init__(
        self, 
        image_size = 256, 
        hflip_augmentation_probability = 0.5,
        contrast_augment_std = 0.0,
        brightness_augment_std = 0.0
    ):
        super().__init__()
        self.image_size = image_size
        self.contrast_augment_std = contrast_augment_std
        self.brightness_augment_std = brightness_augment_std
        self.hflip_augmentation_probability = hflip_augmentation_probability

    def load_image(self, path) -> Image:
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.image_size)
        return img
    
    def augment_image(self, img, color = False, flip = False):
        if flip:
            img = TF.hflip(img)
        if self.contrast_augment_std > 0.0:
            enhancer = ImageEnhance.Contrast(img)
            contrast = np.random.normal(loc=1.0, scale=self.contrast_augment_std)
            img = enhancer.enhance(contrast)
        if self.brightness_augment_std > 0.0:
            enhancer = ImageEnhance.Brightness(img)
            brightness = np.random.normal(loc=1.0, scale=self.brightness_augment_std)
            img = enhancer.enhance(1.0 + self.brightness_augment_std)
        return img

class PairedRGBImageDataset(BaseDataset):
    def __init__(
        self, 
        file_dirs: Tuple[str, str], 
        image_size = 256, 
        hflip_augmentation_probability = 0.5,
        contrast_augment_std = 0.0,
        brightness_augment_std = 0.0
    ):
        super().__init__(image_size, hflip_augmentation_probability, contrast_augment_std, brightness_augment_std)
        self.a_images_directory, self.b_images_directory = file_dirs
        self.a_names = UtilFunctions.get_image_file_names(self.a_images_directory)
        self.b_names = UtilFunctions.get_image_file_names(self.b_images_directory)
        for a_name, b_name in zip(self.a_names, self.b_names):
            if a_name != b_name:
                raise ValueError("Files in A directory and B directory of paired dataset should have one-to-one correspondence of file names.") 

    def __len__(self): 
        return len(self.a_names)

    @torch.no_grad()
    def __getitem__(self, i):
        name = self.a_names[i]
        img_a = self.load_image(os.path.join(self.a_images_directory, name))
        img_b = self.load_image(os.path.join(self.b_images_directory, name))

        # Augmentation
        do_flip = random.random() > self.hflip_augmentation_probability
        
        img_a = self.augment_image(img_a, color=False, flip=do_flip)
        img_b = self.augment_image(img_b, color=True, flip=do_flip)

        # Scale to interval [-1, 1]
        a, b = TF.to_tensor(img_a) * 2.0 - 1.0, TF.to_tensor(img_b) * 2.0 - 1.0

        return a, b
    
class PartiallyPairedRGBImageDataset(BaseDataset):
    def __init__(
        self,
        paired_file_dirs: Tuple[str, str],
        unpaired_file_dirs: Tuple[str, str],
        image_size = 256, 
        hflip_augmentation_probability = 0.5,
        contrast_augment_std = 0.0,
        brightness_augment_std = 0.0,
        probability_load_unpaired = 0.5
    ):
        super().__init__(image_size, hflip_augmentation_probability, contrast_augment_std, brightness_augment_std)
        self.contrast_augment_std = contrast_augment_std
        self.brightness_augment_std = brightness_augment_std
        self.paired_data = PairedRGBImageDataset(paired_file_dirs, image_size, hflip_augmentation_probability, contrast_augment_std, brightness_augment_std)
        self.unpaired_a_images_directory, self.unpaired_b_images_directory = unpaired_file_dirs
        if self.unpaired_a_images_directory is not None:
            self.a_names = UtilFunctions.get_image_file_names(self.unpaired_a_images_directory)
        else:
            self.a_names = []
        if self.unpaired_b_images_directory is not None:
            self.b_names = UtilFunctions.get_image_file_names(self.unpaired_b_images_directory)
        else:
            self.b_names = []
        self.probability_load_unpaired = probability_load_unpaired

    def __len__(self):
        return len(self.paired_data)
    
    @torch.no_grad()
    def __getitem__(self, i) -> torch.Tensor:
        """
        Default behavior is to load paired samples, with a random chance of loading unpaired samples
        """
        mask_a = torch.ones((1, 1, 1))
        mask_b = torch.ones((1, 1, 1))
        if random.random() > self.probability_load_unpaired and len(self.a_names) + len(self.b_names) > 0:
            i = random.randint(0, len(self.a_names) + len(self.b_names) - 1)
            if i < len(self.a_names):
                # Not loading an image from domain b this time
                mask_b = torch.zeros_like(mask_b)
                name = self.a_names[i]
                img = self.load_image(os.path.join(self.unpaired_a_images_directory, name))
                do_flip = random.random() > self.hflip_augmentation_probability
                img = self.augment_image(img, color=False, flip=do_flip)
                a = TF.to_tensor(img) * 2.0 - 1.0
                b = torch.zeros_like(a)
            else:
                mask_a = torch.zeros_like(mask_a)
                name = self.b_names[i - len(self.a_names)]
                img = self.load_image(os.path.join(self.unpaired_b_images_directory, name))
                do_flip = random.random() > self.hflip_augmentation_probability
                img = self.augment_image(img, color=True, flip=do_flip)
                b = TF.to_tensor(img) * 2.0 - 1.0
                a  = torch.zeros_like(b)
        else:
            a, b = self.paired_data[i]

        return a, b, mask_a, mask_b

