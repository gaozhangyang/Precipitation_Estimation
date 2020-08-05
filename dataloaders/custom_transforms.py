import torch
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = np.array(mean)
        self.std = np.array(std)

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']
        img = np.array(img).astype(np.float32)
        label = np.array(label).astype(np.float32)
        mean = self.mean.reshape(-1, 1, 1)
        std = self.std.reshape(-1, 1, 1)
        img -= mean
        img /= std

        return {'image': img,
                'label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}
