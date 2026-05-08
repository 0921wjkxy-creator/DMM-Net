import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from datasets import register
import cv2
from math import pi
from torchvision.transforms import InterpolationMode

import torch.multiprocessing

torch.multiprocessing.set_sharing_strategy('file_system')

import torch.nn.functional as F


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    mask = mask.permute(1, 2, 0) * 255
    mask.int()
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = torch.all(equality, dim=-1)
        semantic_map.append(class_map)

    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    semantic_map = torch.as_tensor(semantic_map)
    semantic_map = semantic_map.permute(2, 0, 1)
    map = torch.sum(semantic_map, dim=0)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    mask = mask.permute(1, 2, 0)
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    x = x.permute(2, 0, 1)
    return x


def to_mask(mask):
    return transforms.ToTensor()(
        transforms.Grayscale(num_output_channels=1)(
            transforms.ToPILImage()(mask)))

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size)(
            transforms.ToPILImage()(img)))


@register('val')
class ValDataset(Dataset):
    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

        self.mask_transform = transforms.Compose([
            transforms.Resize((inp_size, inp_size), interpolation=Image.NEAREST),
            transforms.ToTensor(),
        ])

    def transform_gray(self, img):
        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        img = transforms.ToTensor()(img)
        img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
        img = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(img)
        return img

    def transform_rgb(self, img):
        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(img)
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        vv, vh, rgb, mask, filename = self.dataset[idx]
        vv = self.transform_gray(vv)
        vh = self.transform_gray(vh)
        rgb = self.transform_rgb(rgb)
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask, self.dataset.palette)
        return {
            'vv': vv,
            'vh': vh,
            'rgb': rgb,
            'gt': mask,
            'filename': filename
        }


@register('train')
class TrainDataset(Dataset):
    def __init__(self, dataset, size_min=None, size_max=None, inp_size=None,
                 augment=False, gt_resize=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize

        self.inp_size = inp_size
        self.mask_transform = transforms.Compose([
            transforms.Resize((self.inp_size, self.inp_size)),
            transforms.ToTensor(),
        ])

    def transform_gray(self, img):
        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        img = transforms.ToTensor()(img)
        img = img.repeat(3, 1, 1) if img.shape[0] == 1 else img
        img = transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])(img)
        return img

    def transform_rgb(self, img):
        img = transforms.Resize((self.inp_size, self.inp_size))(img)
        img = transforms.ToTensor()(img)
        img = transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])(img)
        return img

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        vv, vh, rgb, mask, filename = self.dataset[idx]
        vv = self.transform_gray(vv)
        vh = self.transform_gray(vh)
        rgb = self.transform_rgb(rgb)
        mask = self.mask_transform(mask)
        mask = mask_to_onehot(mask, self.dataset.palette)
        return {
            'vv': vv,
            'vh': vh,
            'rgb': rgb,
            'gt': mask,
            'filename': filename
        }