import torch
from pathlib import Path
import os
import math
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from typing import List

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def resize_reformat_files(root_in: str, root_out: str, size: int,
                          do_crop_by_ratio: bool = False) -> None:
    # Files to convert to resized JPEGs
    image_names = get_image_file_names(root_in)

    os.makedirs(root_out, exist_ok=True)
    for name in image_names:
        path_in = os.path.join(root_in, name)
        img = Image.open(path_in).convert('RGB')
        if crop_by_ratio:
            crops = [x for x in crop_by_ratio(img, max_aspect_ratio=1.0)] + [img]
            img = crops[0]
        img = TF.resize(img, size)
        name_out = name.replace('.png', '.jpeg')
        path_out = os.path.join(root_out, name_out)
        img.save(path_out)

def crop_by_ratio(img, max_aspect_ratio=1.25):
    '''
    Split the image up into multiple parts along the long edge if it exceeds a certain aspect ratio.
    '''
    long_edge = max(img.size[0], img.size[1])
    short_edge = min(img.size[0], img.size[1])

    transposed = False

    if long_edge / short_edge > max_aspect_ratio:
        '''
        Rotate all images so that the long edge is the horizontal edge before slicing, 
        then unrotate the slices individually..
        '''
        if long_edge == img.size[1]:
            img = img.transpose(method=Image.ROTATE_90)
            transposed = True

        n_crops = math.ceil(long_edge / short_edge)

        for i in range(n_crops - 1):
            box = (i * img.size[1],  0, (i+1) * img.size[1], img.size[1])
            cropped = img.crop(box)

            if transposed:
                yield cropped.transpose(method=Image.ROTATE_270)
            else:
                yield cropped

        cropped = img.crop((img.size[0] - img.size[1], 0, img.size[0], img.size[1]))

        if transposed:
            yield cropped.transpose(method=Image.ROTATE_270)
        else:
            yield cropped

def is_image_file(filename: str) -> bool:
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def get_image_file_names(dir) -> List[str]:
    images = []
    for _, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                images.append(fname)
    return images

@torch.no_grad()
def format_bchw_network_output_to_images(y: torch.Tensor):
    y = torch.clamp(y.float(), -1, 1)
    y = (y / 2.0 + 0.5) * 255.
    return y

def chw_tensor_to_pil_image(y: torch.Tensor):
    arr = y.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    return Image.fromarray(arr)

@torch.no_grad()
def cat_images(sequence_of_images: List[object]) -> Image:
    # https://stackoverflow.com/questions/30227466/combine-several-images-horizontally-with-python
    widths, heights = zip(*(i.size for i in sequence_of_images))
    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in sequence_of_images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    return new_im

