import torch.utils.data as data
import os
import torch
import pandas as pd
import numpy as np
import cv2
from osgeo import gdal
import tifffile as tif
import albumentations as A
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from PIL import Image
import shutil
import torch.nn.functional as F
import rasterio

image_transform = A.Compose([
    # A.Flip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.HorizontalFlip(p=0.5),
    A.RandomGridShuffle(grid=(2, 2), p=0.5),
    A.Rotate(p=0.5),
], additional_targets={
    'height': 'image',
    'build': 'mask'
})


class sentinelData(data.Dataset):
    def __init__(self, datalist="./merged_data.csv", datarange=(-1, 1), aug=False, transform=None):
        self.datalist = pd.read_csv(datalist, sep=',')
        self.datarange = datarange
        self.aug = aug
        self.transform = transform

    def __getitem__(self, index):
        basename = self.datalist.iloc[index, 0]
        s1dir = self.datalist.iloc[index, 1]
        s2dir = self.datalist.iloc[index, 2]
        bhdir = self.datalist.iloc[index, 3]
        builddir = self.datalist.iloc[index, 4]

        s2 = tif.imread(os.path.join(s2dir))
        s1 = tif.imread(os.path.join(s1dir))
        img = np.concatenate((s1, s2), axis=-1)

        height_path = os.path.join(bhdir)
        height = tif.imread(height_path)
        height[height == 65535] = 0
        height = height.astype('uint8')

        # build footprint
        build_path = os.path.join(builddir)
        build = tif.imread(build_path)
        build = (build == 255).astype('uint8')

        non_zero_height = height != 0
        non_zero_build = build != 0
        if np.array_equal(non_zero_build, non_zero_height) == False:
            print("problem")

        # Augmentation
        if self.aug:
            img_before = img.copy()
            height_before = height.copy()
            build_before = build.copy()

            transformed = image_transform(image=img, mask1=height, mask2=build, mask3=png)
            img = transformed["image"]
            height = transformed["mask1"]
            build = transformed["mask2"]
            png = transformed["mask3"]

        # Normalization
        img = torch.from_numpy(img.astype('float32')).float()
        img = img.permute(2, 0, 1)

        # if isinstance(self.datarange, tuple):
        #     img[img < self.datarange[0]] = self.datarange[0]
        #     img[img > self.datarange[1]] = self.datarange[1]

        img = torch.nan_to_num(img, nan=0.0)
        build = torch.from_numpy(build).long()
        build = F.one_hot(build, num_classes=2).permute(2, 0, 1)
        height = torch.from_numpy(height).float()

        if torch.max(height) == 0:
            print(basename)

        return img, height, build, basename

    def __len__(self):
        return len(self.datalist)
