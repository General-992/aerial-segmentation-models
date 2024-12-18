import os.path
import os.path as osp

import numpy as np
from PIL import Image
import torch

from torch.utils import data
import tifffile as tiff
import albumentations as A
import scripts
from skimage.transform import resize
from torchconvs.datasets.flair import FLAIRSegBase
class ISPRSBase(data.Dataset):
    """
    Base ISPRS dataset class

    :param root: Dataset root directory.
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    :param test: Whether this is a test dataset.
    """
    class_names = np.array([
        'Soil, Snow, clear - cuts, herbaceous vegetation',
        'Pervious, Impervious and transportation surfaces and sports fields',
        'Buildings, swimming pools, Green houses',
        'Trees',
        'Agricultural surfaces',
        'Water bodies'
    ])
    ISPRS_labels = ['Impervious surfaces', 'Buildings',
                    'Low vegetation', 'Tree', 'Car', 'Background']
    isprs_to_flair_mapping = {
        (255, 255, 255): 1, (0, 0, 255): 2,
        (0, 255, 255): 0, (0, 255, 0): 3,
        (255, 255, 0): 1, (255, 0, 0): 0
    }

    # These are for Potsdam
    mean_rgb = np.array([86.551, 92.545 , 85.9151], dtype=np.float32)
    std_rgb = np.array([35.818, 35.399, 36.802], dtype=np.float32)
    transforms = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                            A.GaussNoise()])
    def __init__(self, root: str, transform: bool=False, patch_size = None, tile: bool=False):
        self.root = root
        self._transform = transform
        self.patch_size = patch_size
        self.tile = tile
        self.files = []
        imgsets_file = osp.join(self.root, 'test.txt')
        for did in open(imgsets_file):
            img_file, lbl_file = did.strip().split(' ')
            img_file = osp.join(self.root, 'img/patches',img_file)
            lbl_file = osp.join(self.root, 'msk/patches',lbl_file)
            self.files.append({
                'img': img_file,
                'msk': lbl_file,
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = tiff.imread(self.files[idx]['img'])
        mask = Image.open(self.files[idx]['msk'])
        mask = np.asarray(mask)
        mask = self.mask_encode(mask)


        target_size = (512, 512)  # Adjust to your desired size

        if img.shape[:2] == (500, 500):
            # If ISPRS Potsdam then resolutions are 500x500
            # then interpolate to 512x512, not a big deal
            img = resize(img, target_size, preserve_range=True, anti_aliasing=True)  # For the image (use bilinear)
            mask = resize(mask, target_size, order=0, preserve_range=True, anti_aliasing=False).astype(
                np.int64)  # For the mask (use nearest-neighbor)

        if self._transform:
            img, mask = self.transform(img, mask)
        else:
            img = img.astype(np.float32)
            img = self._normalize(img)
            img = np.transpose(img, (2, 0, 1))

        if img.shape[1:] == (450, 450):
            # If ISPRS Vaihingen then their resolutions are 450x450
            # draw patches of 256x256, it will relieve from the distortion of interpolation to 512x512
            img, mask = scripts.utils.patch_sample(img=img, mask=mask, patch_size=256)

        if self.patch_size:
            if not self.tile:
                img, mask = scripts.utils.patch_sample(img=img, mask=mask, patch_size=self.patch_size)
            else:
                ## NOT IMPLEMETED
                img, mask = scripts.utils.patch_divide(img=img, mask=mask, patch_size=self.patch_size)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).long()

        return img, mask
    def transform(self, img, mask):
        aug = self.transforms(image=img, mask=mask)
        img = aug['image'].astype(np.float32)
        img = self._normalize(img)
        img = np.transpose(img, (2, 0, 1))
        mask = aug['mask']
        return img, mask
    def untransform(self, img, mask):
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
    def mask_encode(self, isprs_mask):

        """
        Encode mask pixel distinct values to discrete class numbers
        :param mask:
        mask with ([0, 1, 2, .., num_classes]), original pixels values
        :return:
        new_mask with ([0, 1, 2, .., num_classes]), new pixels values
        """
        flair_mask = np.zeros((isprs_mask.shape[0], isprs_mask.shape[1]), dtype=np.uint8)

        for rgb, class_label in self.isprs_to_flair_mapping.items():
            mask = np.all(isprs_mask == rgb, axis=-1)
            flair_mask[mask] = class_label
        return flair_mask


class CombinedFLAIR_ISPRS(data.Dataset):
    def __init__(self, flair_dataset: FLAIRSegBase, isprs_dataset: ISPRSBase):
        """
        Combines FLAIR and ISPRS datasets into one.

        :param flair_dataset: Instance of FLAIRSegBase dataset.
        :param isprs_dataset: Instance of ISPRSBase dataset.
        """
        self.flair_dataset = flair_dataset
        self.isprs_dataset = isprs_dataset

        # The length is the total length of both datasets
        self.flair_len = len(flair_dataset)
        self.isprs_len = len(isprs_dataset)
        self.class_names = flair_dataset.class_names

    def __len__(self):
        return self.flair_len + self.isprs_len

    def __getitem__(self, idx):
        # If the index is within the range of FLAIR dataset
        if idx < self.flair_len:
            return self.flair_dataset[idx]
        # Otherwise, use the ISPRS dataset, but adjust the index
        else:
            return self.isprs_dataset[idx - self.flair_len]


if __name__ == '__main__':
    root_isprs = os.path.expanduser('~/datasets/ISPRS/Potsdam')
    isprs = ISPRSBase(root=root_isprs, patch_size=512)
    # root_flair = os.path.expanduser('~/datasets/FLAIR/flair_dataset_train_val')
    # flair = FLAIRSegBase(
    #     root_flair, split='train', transform=False, patch_size=256)
    # isprs_flair = CombinedFLAIR_ISPRS(flair_dataset=flair, isprs_dataset=isprs)
    # print(isprs.__len__())
    # print(flair.__len__())
    # print(isprs_flair.__len__())
    class_names = isprs.class_names
    image, label = isprs[1444]
    image, label = isprs.untransform(image, label)
    scripts.utils.plot_image_label_classes(image, label, class_names)


