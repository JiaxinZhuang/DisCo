# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random


class SWAVTwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, small_transform):
        self.base_transform = base_transform
        self.small_transform = small_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.small_transform(x)
        return [q, k]


class FourCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q1 = self.base_transform(x)
        k1 = self.base_transform(x)
        q2 = self.base_transform(x)
        k2 = self.base_transform(x)
        return [q1, k1, q2, k2]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x