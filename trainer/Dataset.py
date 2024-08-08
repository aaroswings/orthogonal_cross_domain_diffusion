import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

import os
from PIL import Image
import random
from typing import Tuple

from trainer import UtilFunctions

class PairedRGBImageDataset(Dataset):
    def __init__(
        self, 
        file_dirs: Tuple[str, str], 
        image_size = 256, 
        hflip_augmentation_probability = 0.5
    ):
        super().__init__()
        self.a_images_directory, self.b_images_directory = file_dirs
        self.a_names = UtilFunctions.get_image_file_names(self.a_images_directory)
        self.b_names = UtilFunctions.get_image_file_names(self.b_images_directory)
        for a_name, b_name in zip(self.a_names, self.b_names):
            if a_name != b_name:
                raise ValueError("Files in A directory and B directory of paired dataset should have one-to-one correspondence of file names.") 
        self.image_size = image_size
        self.hflip_augmentation_probability = hflip_augmentation_probability

    def __len__(self): 
        return len(self.a_names)

    def load_image(self, path) -> Image:
        img = Image.open(path).convert('RGB')
        img = TF.resize(img, self.image_size)
        return img

    @torch.no_grad()
    def __getitem__(self, i) -> torch.Tensor:
        name = self.a_names[i]
        img_a = self.load_image(os.path.join(self.a_images_directory, name))
        img_b = self.load_image(os.path.join(self.b_images_directory, name))

        # Augmentation
        if random.random() > self.hflip_augmentation_probability:
            img_a, img_b = TF.hflip(img_a), TF.hflip(img_b)

        # Scale to interval [-1, 1]
        a, b = TF.to_tensor(img_a) * 2.0 - 1.0, TF.to_tensor(img_b) * 2.0 - 1.0

        return a, b