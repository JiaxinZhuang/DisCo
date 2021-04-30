
from __future__ import print_function
import io
import csv
import tqdm
import torch
import pickle
import base64
import random
import numpy as np
from PIL import Image
#from moco.tsv_io import TSVFile
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class TSVDataset(Dataset):
    """ TSV dataset for ImageNet 1K training
    """
    def __init__(self, tsv_file, transform=None, target_transform=None):
        self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        row = self.tsv.seek(index)
        image_data = base64.b64decode(row[-1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')
        target = int(row[1])

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return round(self.tsv.num_rows())


class KM_TSVDataset(Dataset):
    """ TSV dataset for ImageNet 1K training
    """
    def __init__(self, tsv_file, km_tsv_file, transform=None, target_transform=None):
        self.tsv = TSVFile(tsv_file)

        with open(km_tsv_file) as tsvfile:
            self.km_tsv_file = list(csv.reader(tsvfile, delimiter='\t'))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        row = self.tsv.seek(index)
        image_data = base64.b64decode(row[-1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')

        # print(self.km_tsv_file[index])
        target = int(self.km_tsv_file[index][1])

        if self.transform is not None:
            img = self.transform(image)
        else:
            img = image
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return round(self.tsv.num_rows())


class Jigsaw_TSVDataset(Dataset):
    """
        TSV dataset for ImageNet 1K training with jigsaw random crop.
    """
    def __init__(self, tsv_file, transform=None, jigsaw_transform=None):
        self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.jigsaw_transform = jigsaw_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        row = self.tsv.seek(index)
        image_data = base64.b64decode(row[-1])
        image = Image.open(io.BytesIO(image_data))
        image = image.convert('RGB')

        img = self.transform(image)
        jigsaw_image = self.jigsaw_transform(image)

        return img, jigsaw_image

    def __len__(self):
        return round(self.tsv.num_rows())


class Small_Patch_TSVDataset(ImageFolder):
    """
        TSV dataset for ImageNet 1K training with jigsaw random crop.
    """
    def __init__(self, traindir, transform=None, jigsaw_transform=None, num_patches=6):
        super(Small_Patch_TSVDataset, self).__init__(traindir)
        #self.tsv = TSVFile(tsv_file)
        self.transform = transform
        self.jigsaw_transform = jigsaw_transform
        self.num_patches = num_patches

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img_path, _ = self.imgs[index]
        image = Image.open(img_path)
        image = image.convert('RGB')
        img_1 = self.transform(image)
        img_2 = self.transform(image)

        # Stack small images
        small_patches = []
        for img_index in range(self.num_patches):
            small_patches.extend(
                self.jigsaw_transform(image).unsqueeze(0))

        return img_1, img_2, torch.stack(small_patches)

    def __len__(self):
        return len(self.imgs)
