import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from datasets import register
from datasets.image_folder import ImageFolder

@register('multimodal-image-folders')
class MultiModalImageFolders(Dataset):
    def __init__(self, vv_root_path, vh_root_path, rgb_root_path, lab_root_path, classes, palette, size=1024, cache='none', split_key=None, **kwargs):
        self.vv_folder = ImageFolder(vv_root_path, size=size, cache=cache)
        self.vh_folder = ImageFolder(vh_root_path, size=size, cache=cache)
        self.rgb_folder = ImageFolder(rgb_root_path, size=size, cache=cache)
        self.lab_folder = ImageFolder(lab_root_path, size=size, cache=cache, mask=True)
        self.classes = classes
        self.palette = palette

        assert len(self.vv_folder) == len(self.vh_folder) == len(self.rgb_folder) == len(self.lab_folder), "模态文件数量不一致！"

    def __len__(self):
        return len(self.vv_folder)

    def __getitem__(self, idx):
        vv, _ = self.vv_folder[idx]   # [3, H, W] tensor
        vh, _ = self.vh_folder[idx]   # [3, H, W] tensor
        rgb, _ = self.rgb_folder[idx] # [3, H, W] tensor
        mask, filename = self.lab_folder[idx] # mask: [3, H, W] or [1, H, W] tensor
        return vv, vh, rgb, mask, filename