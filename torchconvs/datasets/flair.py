#!/usr/bin/env python

import os.path as osp

import numpy as np
from PIL import Image
import torch

from collections import defaultdict
from torch.utils import data
import tifffile as tiff
import albumentations as A
import scripts

class FLAIRSegBase(data.Dataset):
    """
    Base FLAIR dataset class

    :param root: Dataset root directory.
    :param split: Dataset split (train/val/test).
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    :param test: Whether this is a test dataset.
    """

    class_names = np.array([
        'Soil, Snow, clear - cuts, herbaceous vegetation, brushes, low-vegetation',
        'Pervious, Impervious and transportation surfaces and sports fields',
        'Buildings, swimming pools, Green houses',
        'Trees',
        'Agricultural surfaces',
        'Water bodies'
    ])

    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)
    transforms = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                            A.GaussNoise()])
    def __init__(self, root: str, split: str, transform: bool=False, patch_size=256, tile: bool=False):
        self.root = root
        self._transform = transform
        self.patch_size = patch_size
        self.tile = tile
        self.files = []
        imgsets_file = osp.join(self.root, '%s.txt' % split)
        for did in open(imgsets_file):
            img_file, lbl_file = did.strip().split(' ')
            img_file = osp.join(self.root, img_file)
            lbl_file = osp.join(self.root, lbl_file)
            self.files.append({
                'img': img_file,
                'msk': lbl_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = tiff.imread(self.files[idx]['img'])
        img = img[..., :3]
        mask = Image.open(self.files[idx]['msk'])
        mask = np.asarray(mask, dtype=np.uint8)
        mask = self.mask_encode(mask)

        if self._transform:
            img, mask = self.transform(img, mask)
        else:
            img = img.astype(np.float32)
            img = self._normalize(img)
            img = np.transpose(img, (2, 0, 1))

        if self.patch_size:
            if not self.tile:
                img, mask = scripts.utils.patch_sample(img=img, mask=mask, patch_size=self.patch_size)
            else:
                img, mask = scripts.utils.patch_divide(img=img, mask=mask, patch_size=self.patch_size)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
    def transform(self, img, mask=None):
        aug = self.transforms(image=img, mask=mask)
        img = aug['image'].astype(np.float32)
        img = self._normalize(img)
        img = np.transpose(img, (2, 0, 1))
        mask = aug['mask']
        return img, mask
    def untransform(self, img, mask=None):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img *= self.std_rgb
        img += self.mean_rgb
        img = img.astype(np.uint8)
        mask = mask.numpy()
        return img, mask

    def _normalize(self, img):
        img -= self.mean_rgb
        img /= self.std_rgb
        return img
    def mask_encode(self, mask):
        """
        Encode mask pixel distinct values to discrete class numbers
        :param mask:
        mask with ([0, 1, 2, .., num_classes]), original pixels values
        :return:
        new_mask with ([0, 1, 2, .., num_classes]), new pixels values
        """

        class_mapping = {
            4: 0, 10: 0, 14: 0, 15: 0, 8: 0, 19:0,   # Soil, Snow, clear-cuts, herbaceous vegetation, bushes
            2: 1, 3: 1,                              # Pervious, Impervious and transportation surfaces and sports fields
            1: 2, 13: 2, 18: 2,                      # Buildings, swimming pools, Green houses
            6: 3, 7: 3, 16: 3, 17: 3,                # Trees
            9: 4, 11: 4, 12: 4,                      # Agricultural surfaces
            5: 5,                                    # Water bodies
        }
        # Initialize a new mask with the same shape
        new_mask = np.zeros_like(mask)

        # Reassign classes according to the mapping
        for old_class, new_class in class_mapping.items():
            new_mask[mask == old_class] = new_class
        return new_mask

    def class_distribution(self):
        total_pixels_per_class = defaultdict(int)
        total_pixels = 0

        # Iterate over all masks
        for idx, file_dict in enumerate(self.files):
            if idx % 100 == 0:
                print(f'{idx}/{len(self.files)}')
            mask = Image.open(file_dict['msk'])
            mask = np.asarray(mask)
            mask = self.mask_encode(mask)  # Ensure it's encoded into semantic classes

            unique_classes, counts = np.unique(mask, return_counts=True)

            # Sum the pixels for each class
            for cls, cnt in zip(unique_classes, counts):
                total_pixels_per_class[cls] += cnt

            # Keep track of total pixels
            total_pixels += mask.size

        # Print out the percentage-wise class distribution
        print("Class Distribution (%):")
        for cls in range(len(self.class_names)):
            percentage = (total_pixels_per_class[cls] / total_pixels) * 100
            print(f"{self.class_names[cls]}: {percentage:.2f}%")

class FLAIRSegMeta(FLAIRSegBase):
    """
    Inherits from FLAIRSegBase and adds functionality to track image metadata such as
    camera type and capture month.

    :param root: Dataset root directory.
    :param split: Dataset split (train/val/test).
    :param metadata: A dictionary containing metadata for each image (camera type, date, etc.).
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    :param test: Whether this is a test dataset.
    """
    def __init__(self, root, split, metadata, transform=False, patch_size=256, test=False):
        super().__init__(root, split, transform, patch_size, test)
        self.metadata = metadata

    def __getitem__(self, idx):
        img, mask = super().__getitem__(idx)
        img_file = self.files[idx]['img']
        img_name = osp.basename(img_file)
        img_key = osp.splitext(img_name)[0]

        if img_key in self.metadata:
            camera = self.metadata[img_key].get('camera', 'Unknown')
            date = self.metadata[img_key].get('date', 'Unknown')
            time = self.metadata[img_key].get('time', 'Unknown')
            time = time.split('h')[0] if time != 'Unknown' else 'Unknown'
            month = date.split('-')[1] if date != 'Unknown' else 'Unknown'

        else:
            raise Exception(f'{img_key} not found in metadata')
        return img, mask, int(month), camera, time


if __name__ == '__main__':
    train_root = osp.expanduser('~/datasets/FLAIR/flair_dataset_train_val')
    test_root = osp.expanduser('~/datasets/FLAIR/flair_dataset_test')

    flair_base =  FLAIRSegBase(root = train_root, split = 'train', transform=False, patch_size=None)
    print(flair_base[0])
    # for idx, (img, msk) in enumerate(flair_base):
    #     img, msk = flair_base.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=flair_base.class_names)
    #     if idx == 500:
    #         break
    # # flair_base[0]
    # print(flair_base.__len__())

    # for idx, (img, msk) in enumerate(flair_base):
    #     img, msk = flair_base.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=flair_base.class_names)
    #     if idx == 4:
    #         break
    # file_path = osp.join(root, 'flair-1_metadata_aerial.json')
    # images_root = osp.join(root, 'flair_dataset_train_val')
    # import json
    # with open(file_path, 'r') as file:
    #     metadata = json.load(file)
    #
    # test_meta = FLAIRSegMeta(root=images_root, split='val', metadata=metadata, transform=False)
    # print(test_meta[0])

    # Uncomment respectively to calculate the class distrubitions in stratified subsets and plot examples

    # flair_urban = FLAIRSegBase(root = test_root, split = 'urban', transform=False, patch_size=None)
    # # flair_urban.class_distribution()
    # for idx, (img, msk) in enumerate(flair_urban):
    #     img, msk = flair_urban.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=flair_urban.class_names)
    #     if idx == 4:
    #         break

    # flair_agric = FLAIRSegBase(root = test_root, split = 'agri', transform=False, patch_size=None)
    ## flair_agric.class_distribution()

    # for idx, (img, msk) in enumerate(flair_agric):
    #     img, msk = flair_agric.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=flair_agric.class_names)
    #     if idx == 4:
    #         break


    # flair_mount = FLAIRSegBase(root = test_root, split = 'mountain', transform=False, patch_size=None)
    # # flair_mount.class_distribution()
    # for idx, (img, msk) in enumerate(flair_mount):
    #     img, msk = flair_mount.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=flair_mount.class_names)
    #     if idx == 4:
    #         break

    # flair_coast = FLAIRSegBase(root = test_root, split = 'coastal', transform=False, patch_size=None)
    # # flair_coast.class_distribution()
    # for idx, (img, msk) in enumerate(flair_coast):
    #     img, msk = flair_coast.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=flair_coast.class_names)
    #     if idx == 4:
    #         break

    # root_viz = '/home/general992/datasets/vizualize'
    # dataset = FLAIRSegBase(
    #     root_viz, split='viz', transform=False, patch_size=None)
    #
    # for idx, (img, msk) in enumerate(dataset):
    #     img, msk = dataset.untransform(img, msk)
    #     scripts.plot_image_label_classes(img, msk, class_names=dataset.class_names)
    #     if idx == 4:
    #         break