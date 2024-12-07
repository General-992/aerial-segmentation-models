import os.path as osp
import os
import csv

import numpy as np
from PIL import Image

import pandas as pd

root = osp.expanduser('~/datasets/FLAIR')
print(root)
meta_path = osp.expanduser('~/datasets/FLAIR/flair-1_metadata_aerial.json')
images_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test/img')
labels_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test/msk')


image_files = [f for f in os.listdir(images_path) if f.startswith('IMG') and f.endswith('.tif')]
mask_files = [f for f in os.listdir(labels_path) if f.startswith('MSK') and f.endswith('.tif')]

X = []
for image_file in image_files:
    corresponding_mask = image_file.replace('IMG', 'MSK')
    if corresponding_mask in mask_files:
        image_path = os.path.join('img', image_file)
        mask_path = os.path.join('msk', corresponding_mask)
        X.append((image_path, mask_path))
    else:
        raise FileNotFoundError

urban_subfolders = ['UU', 'UA', 'UF', 'UN']
agricultural_subfolders = ['AA', 'AU', 'AN', 'AF']
labels_base_path = osp.expanduser('~/datasets/flair_1_labels_test')


def stratify(mask_path, stratify_subfolder, strat_classes, min_strat_percentage=0.15):
    """
    Stratify based on the presence of certain classes in the mask with a minimum percentage threshold.

    :param mask_path: Path to the mask file.
    :param stratify_subfolder: List of subfolder names to search for.
    :param strat_classes: List of class IDs to include in the stratification (e.g., [9, 11, 12] for agricultural land).
    :param min_strat_percentage: Minimum percentage of the mask that must belong to the specified classes.
    :return: True if the mask meets the stratification criteria, False otherwise.
    """

    # Extract the mask filename
    mask_filename = os.path.basename(mask_path)

    # Go through the label directories
    for domain_folder in os.listdir(labels_base_path):
        domain_path = os.path.join(labels_base_path, domain_folder)
        if os.path.isdir(domain_path):
            for zone_folder in os.listdir(domain_path):
                zone_path = os.path.join(domain_path, zone_folder)
                if any(subfolder in zone_folder for subfolder in stratify_subfolder):
                    # Check if the mask exists in this folder
                    mask_folder_path = os.path.join(zone_path, 'msk')
                    if mask_filename in os.listdir(mask_folder_path):
                        # Load the mask
                        mask_full_path = os.path.join(mask_folder_path, mask_filename)
                        mask = Image.open(mask_full_path)
                        mask = np.array(mask)

                        # Calculate the percentage of pixels that belong to the stratification classes
                        total_pixels = mask.size
                        strat_pixels = np.sum(
                            np.isin(mask, strat_classes))  # Check if any pixel belongs to strat_classes
                        strat_percentage = strat_pixels / total_pixels

                        # Ensure that at least a certain percentage of the mask belongs to the specified classes
                        if strat_percentage >= min_strat_percentage:
                            return True
    return False

urban_images = [(img, msk) for img, msk in X if stratify(msk, urban_subfolders, strat_class=1)]
print('Urban images len:', len(urban_images))
output_txt_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test/urban.txt')

with open(output_txt_path, mode='w') as file:
    for img, msk in urban_images:
        file.write(f"{img} {msk}\n")

print(f"Urban images and masks saved to {output_txt_path}.")

agricultural_images = [(img, msk) for img, msk in X if stratify(msk, agricultural_subfolders, strat_classes=[9, 11, 12])]

agric_txt_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test/agri.txt')

with open(agric_txt_path, mode='w') as file:
    for img, msk in agricultural_images:
        file.write(f"{img} {msk}\n")

print(f"Agricultural images and masks saved to {agric_txt_path}.")

df_metadata = pd.read_json(osp.expanduser('~/datasets/FLAIR/flair-1_metadata_aerial.json')).T

mountainous_threshold = 1500

df_mountainous = df_metadata[df_metadata['patch_centroid_z'] > mountainous_threshold]
mountainous_images = df_mountainous.index.tolist()
mountainous_images = [(img, msk) for img, msk in X if img.split('/')[1].replace('.tif','') in mountainous_images]

mountain_txt_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test/mountain.txt')
with open(mountain_txt_path, mode='w') as file:
    for img, msk in mountainous_images:
        file.write(f"{img} {msk}\n")
print(f"Mountainous images and masks saved to {mountain_txt_path}.")



# COASTAL
labels_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test')
# Thresholds
water_class = 5
min_water_percentage = 0.3
max_water_percentage = 1.0
coastal_images = []

# Iterate through image-mask pairs
for img, msk in X:
    # Open and process mask
    msk_path = osp.join(labels_path, msk)
    label = Image.open(msk_path)
    label = np.array(label)

    # Calculate total pixels and water pixels
    total_pixels = label.size
    water_pixels = np.sum(label == water_class)

    # Calculate water percentage
    water_percentage = water_pixels / total_pixels

    # Check if the image meets the criteria for coastal
    if min_water_percentage <= water_percentage < max_water_percentage:
        coastal_images.append((img, msk))

# Write the coastal images to a text file
coastal_txt_path = osp.expanduser('~/datasets/FLAIR/flair_dataset_test/coastal.txt')
with open(coastal_txt_path, mode='w') as file:
    for img, msk in coastal_images:
        file.write(f"{img} {msk}\n")

print(f"Coastal images and masks saved to {coastal_txt_path}.")
