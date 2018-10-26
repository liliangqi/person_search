# -----------------------------------------------------
# Data transformation for Person Search
#
# Author: Liangqi Li
# Creating Date: Aug 8, 2018
# Latest rectifying: Oct 26, 2018
# -----------------------------------------------------
import random

import numpy as np
import torch
import cv2


class Compose:
    """Composes several transforms together."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, im):
        im_scale = 1
        flip = False
        for t in self.transforms:
            if isinstance(t, Scale):
                im, im_scale = t(im)
            elif isinstance(t, RandomHorizontalFlip):
                im, flip = t(im)
            else:
                im = t(im)
        return im, im_scale, flip


class ToTensor:
    """Convert a np.ndarray of shape (H x W x C) to torch.FloatTensor
    of shape (C x H x W)."""

    def __init__(self):
        # TODO: add normalization value
        pass

    def __call__(self, im):
        return torch.Tensor(im.transpose((2, 0, 1))).float()


class Normalize:
    """Normalize an tensor image with mean and standard deviation."""

    def __init__(self, mean):
        """
        Initialize the transformation.
        ---
        :param mean: Sequence of means
        """
        # TODO: add std
        self.mean = mean

    def __call__(self, im_tensor):
        for t, m in zip(im_tensor, self.mean):
            t.sub_(m)
        return im_tensor


class Scale:

    def __init__(self, target_size, max_size):
        self.target_size = target_size
        self.max_size = max_size

    def __call__(self, im):
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(self.target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.max_size:
            im_scale = float(self.max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        return im, im_scale


class RandomHorizontalFlip:
    """Horizontally flip the given image randomly with a probability of 0.5."""

    def __call__(self, im):
        flip = random.random() < 0.5
        if flip:
            im = im[:, ::-1, :]
        return im, flip
