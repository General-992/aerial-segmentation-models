import os.path as osp

import numpy as np
from PIL import Image
import torch

from collections import defaultdict
from torch.utils import data
import tifffile as tiff
import albumentations as A
from torch.utils.data import Dataset

import os


class DatasetBase(data.Dataset):
    """
    Base dataset class

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
    class_mapping = {
        4: 0, 10: 0, 14: 0, 15: 0, 8: 0, 19: 0,  # Soil, Snow, clear-cuts, herbaceous vegetation, bushes
        2: 1, 3: 1,                              # Pervious, Impervious and transportation surfaces and sports fields
        1: 2, 13: 2, 18: 2,                      # Buildings, swimming pools, Green houses
        6: 3, 7: 3, 16: 3, 17: 3,                # Trees
        9: 4, 11: 4, 12: 4,                      # Agricultural surfaces
        5: 5,                                    # Water bodies
    }
    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)
    transforms = A.Compose([A.HorizontalFlip(), A.VerticalFlip(),
                            A.GridDistortion(p=0.2), A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                            A.GaussNoise()])
    n_classes = len(class_names)

    def __init__(self, root: str, transform: bool = False, patch_size=None):
        self.root = root
        self._transform = transform
        self.patch_size = patch_size
        self.files = []

        img_folder = os.path.join(self.root, 'img')
        msk_folder = os.path.join(self.root, 'msk')

        if not os.path.exists(img_folder):
            raise FileNotFoundError(f"Image folder '{img_folder}' not found.")

        img_files = sorted(os.listdir(img_folder))
        if os.path.exists(msk_folder):
            self.has_labels = True
            msk_files = sorted(os.listdir(msk_folder))
            # Assuming images and masks have the same filenames but different extensions
            self.files = [
                {'img': os.path.join(img_folder, img_file), 'msk': os.path.join(msk_folder, 'MSK_' + img_file.split('_')[1])}
                for img_file in img_files if 'MSK_' + img_file.split('_')[1] in msk_files
            ]
        else:
            # Only images, no masks
            print('No labels found')
            self.has_labels = False
            self.files = [{'img': os.path.join(img_folder, img_file)} for img_file in img_files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]['img']
        img_name = os.path.basename(img_path)
        # Load the image based on its extension
        if img_path.lower().endswith(('.jpeg', '.jpg', '.png')):
            img = Image.open(img_path).convert('RGB')
            img = np.array(img)
        elif img_path.lower().endswith(('.tiff', '.tif')):
            img = tiff.imread(img_path)[..., :3]  # Read only the first three channels if it's a tiff file
        else:
            raise Exception('Image format must be jpeg, png, or tiff.')

        # Check if the dataset has labels
        if self.has_labels and 'msk' in self.files[idx]:
            mask_path = self.files[idx]['msk']
            mask = Image.open(mask_path)
            mask = np.asarray(mask)
            mask = self.mask_encode(mask)
            mask = torch.from_numpy(mask).long()  # Convert to tensor
        else:
            # Return a placeholder mask (e.g., a tensor of zeros with the same spatial dimensions)
            mask = torch.zeros((img.shape[0], img.shape[1]), dtype=torch.long)

        # Normalize the image and convert to tensor
        img = img.astype(np.float32)
        img = self._normalize(img)
        img = np.transpose(img, (2, 0, 1))  # Convert to (C, H, W) format
        img = torch.from_numpy(img).float()

        return img, mask, img_name

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
    def mask_encode(self, mask):
        """
        Encode mask pixel distinct values to discrete class numbers
        :param mask:
        mask with ([0, 1, 2, .., num_classes]), original pixels values
        :return:
        new_mask with ([0, 1, 2, .., num_classes]), new pixels values
        """

        # Initialize a new mask with the same shape
        new_mask = np.zeros_like(mask)
        # Reassign classes according to the mapping
        for old_class, new_class in self.class_mapping.items():
            new_mask[mask == old_class] = new_class
        return new_mask


if __name__ == '__main__':
    import os.path as osp

    root = osp.expanduser('~/datasets/vizualize')

    dataset = DatasetBase(root=root)

    try:
        print(dataset.__getitem__(0))
    except:
        print('Error with item getter')
