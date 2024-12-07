from random import randint
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from torchconvs.models import UnetPlusPlus, Deeplabv3plus_resnet, SegNet, HRNet
import seaborn as sns
from PIL import Image
import os

import matplotlib
# matplotlib.use('Agg')
def patch_divide(img, mask, patch_size):
    """
    If the image is larger than 256x256 and it is a testing stage
    it divides the image into patches of 256x256 pixels.
    input shape: ([3, l, w])
    output shape: ([3, num_patches^2, l/num_patches, w/num_patches])
    """
    if img.shape[-1] == img.shape[-2]:
        length = img.shape[-1] / patch_size
        number_of_patches = int((length) ** 2)
        img = img.view(number_of_patches, 3, int(img.shape[-1] / length), int(img.shape[-1] / length))
        mask = mask.view(number_of_patches, int(mask.shape[-1] / length), int(mask.shape[-1] / length))
    else:
        raise NotImplementedError(
            'Not implemented the division of the test images of non-rectangular shape')
    return img, mask

def patch_sample(img, mask, patch_size):
    """
    Randomly samples patches from images, useful
    when dealing with large resolution map images.
    Each patch has a fixed size of patch_size x patch_size pixels

    Input:
    :img: ndarray[num_channels, height, width]
    :mask: ndarray[height, width]
    :patch_size: int

    Output:
    :image_patch: ndarray[num_channels, patch_size, patch_size]
    :mask_patch: ndarray[patch_size, patch_size]
    """
    _, h, w = img.shape
    top = randint(0, h - patch_size)
    left = randint(0, w - patch_size)

    image_patch = img[:, top:top + patch_size, left:left + patch_size]
    mask_patch = mask[top:top + patch_size, left:left + patch_size]
    return image_patch, mask_patch

def plot_image_label(img, mask, num_classes: int=6, output_dir: str = None):
    """
    Plots the image and the mask with semantic classes. Does not include a legend.
    If `output_dir` is provided, saves only the label image instead of displaying the plots.

    :param img: numpy array of shape [512, 512, 3], the image.
    :param mask: numpy array of shape [512, 512], the label mask.
    :param num_classes: int, the number of unique classes in the label mask.
    :param output_dir: if provided, the directory to save only the label image.
    """
    # Define the colormap for the mask (label)
    cmap = plt.get_cmap('tab10')
    colors = cmap(np.linspace(0, 1, num_classes))

    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    unique_classes = np.unique(mask)
    for cls in unique_classes:
        colored_mask[mask == cls] = (colors[cls][:3] * 255).astype(np.uint8)

    if output_dir:
        # Save only the label image
        dir, _ = os.path.split(output_dir)
        os.makedirs(dir, exist_ok=True)
        label_image = Image.fromarray(colored_mask)
        label_image.save(output_dir)
        print(f"Saved label image to {output_dir}")
    else:
        # Create a figure and axes for plotting
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        # Plot the image
        axes[0].imshow(img.astype(np.uint8))
        axes[0].axis('off')
        axes[0].set_title("Image")

        # Plot the mask (label)
        axes[1].imshow(colored_mask)
        axes[1].axis('off')
        axes[1].set_title("Label")

        # Adjust layout to avoid overlapping
        plt.tight_layout()
        plt.show()

def plot_image_label_classes(img, mask, class_names: list, output_dir: str=None):
    """
    Plots the image, the mask, and the semantic classes on the same level.
    If `output_dir` is provided, saves only the label image instead of displaying all plots.

    :param img: numpy array of shape [512, 512, 3], the image.
    :param mask: numpy array of shape [512, 512], the label mask.
    :param class_names: list or array of class names.
    :param output_dir: if provided, the directory to save only the label image.
    """
    # Define the colormap for the mask (label) to make it more visually distinct
    cmap = plt.get_cmap('tab10')
    unique_classes = np.unique(mask)
    if class_names is int:
        colors = cmap(np.linspace(0, 1, class_names))

    # Create a colored mask
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for i, cls in enumerate(unique_classes):
        colored_mask[mask == cls] = (colors[cls][:3] * 255).astype(np.uint8)



    if output_dir:
        # Save only the label image
        dir, image = os.path.split(output_dir)
        os.makedirs(dir, exist_ok=True)

        # label_path = os.path.join(output_dir)
        label_image = Image.fromarray(colored_mask)
        label_image.save(output_dir)
        print(f"Saved label image to {output_dir}")
    else:
        # Create a figure and axes for plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Plot the image
        axes[0].imshow(img.astype(np.uint8))
        axes[0].axis('off')
        axes[0].set_title("Image")

        # Plot the mask (label)
        axes[1].imshow(colored_mask)
        axes[1].axis('off')
        axes[1].set_title("Label")

        # Plot the semantic classes
        legend_patches = [mpatches.Patch(color=colors[i], label=class_names[i]) for i in range(len(class_names))]
        axes[2].legend(handles=legend_patches, loc='center', fontsize='small')
        axes[2].axis('off')
        axes[2].set_title("Semantic Classes")

        # Adjust layout to avoid overlapping
        plt.tight_layout()
        plt.show()

def plot_classes_IoU(IoUs: np.ndarray, class_names: list):
    """
    Plots the IoU scores for each class with the bar plot.
    """

    plt.figure(figsize=(8, 5))
    plt.bar(class_names, IoUs, color='skyblue')
    plt.ylim(0, 100)
    plt.title("IoU per Class for Testing Dataset")
    plt.xlabel("Classes")
    plt.ylabel("IoU %")
    plt.tight_layout()
    plt.savefig('IoU_per_class.png')


def plot_metrics_per_meta(meta_metrics: dict, model_name: str, meta_type: str):
    """
    Plots bar charts showing the segmentation metrics (accuracy, class accuracy, mean IU, FWAV accuracy)
    for each meta type.

    Parameters:
    meta_metrics (dict): A dictionary where keys are months ('1', '2', ..., '12')
                          and values are lists of metric tuples (accuracy, accuracy_class, mean_iu, fwavacc).
    """
    metas = []
    accuracies = []
    accuracy_classes = []
    mean_ius = []
    fwav_accs = []

    for meta, metrics in meta_metrics.items():
        if metrics:
            avg_metrics = np.mean(metrics, axis=0)
            accuracy, acc_cls, mean_iu, fwavacc = avg_metrics * 100
            metas.append(int(meta))
            accuracies.append(accuracy)
            accuracy_classes.append(acc_cls)
            mean_ius.append(mean_iu)
            fwav_accs.append(fwavacc)

    metas_, accuracies, accuracy_classes, mean_ius, fwav_accs = zip(
        *sorted(zip(metas, accuracies, accuracy_classes, mean_ius, fwav_accs)))

    metrics_names = ['Accuracy (%)', 'Class Accuracy (%)', 'Mean IU (%)', 'FWAV Accuracy (%)']
    metrics_data = [accuracies, accuracy_classes, mean_ius, fwav_accs]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        ax.bar(metas_, metrics_data[idx], color='skyblue')
        ax.set_xlabel(meta_type, fontsize=12)
        ax.set_ylabel(metrics_names[idx], fontsize=12)
        ax.set_title(f'{metrics_names[idx]} per {meta_type}', fontsize=14)
        ax.set_ylim(0, 100)
        ax.set_xticks(metas_)
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    fig.suptitle(f'Metrics per {meta_type} for {model_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig('metrics_%s.png' % meta_type)
    plt.show()



def model_select(model_name: str, n_class: int = 6) -> torch.nn.Module:
    """
    Sets up the model
    """
    if model_name.lower().startswith('unet'):
        # num of trainable params = 26.079.479
        print('Loaded Unet')
        model = UnetPlusPlus(n_class=n_class)
    elif model_name.lower().startswith('deep'):
        #  num of trainable params = 39.758.247
        print('Loaded Deeplab')
        model = Deeplabv3plus_resnet(n_class=n_class)
    elif model_name.lower().startswith('segnet'):
        print('Loaded Segnet')
        # num of trainable params = 12.932.295
        model = SegNet(n_class=n_class)
    elif model_name.lower().startswith('hrnet'):
        # num of trainable params = 31.990.087
        print('Loaded HRNet')
        model = HRNet(n_class=n_class)
    else:
        raise Exception('Unknown model')
    sum = 0
    for param in model.parameters():
        if param.requires_grad:
            sum += param.numel()
    print(f'Total trainable params: {sum}, model: {model_name}')
    return model

import imgviz
import copy
def visualize_segmentation(**kwargs):
    """Visualize segmentation.

    Parameters
    ----------
    img: ndarray
        Input image to predict label.
    lbl_true: ndarray
        Ground truth of the label.
    lbl_pred: ndarray
        Label predicted.
    n_class: int
        Number of classes.
    label_names: dict or list
        Names of each label value.
        Key or index is label_value and value is its name.

    Returns
    -------
    img_array: ndarray
        Visualized image.
    """
    img = kwargs.pop('img', None)
    lbl_true = kwargs.pop('lbl_true', None)
    lbl_pred = kwargs.pop('lbl_pred', None)
    n_class = kwargs.pop('n_class', None)
    label_names = kwargs.pop('label_names', None)
    if kwargs:
        raise RuntimeError(
            'Unexpected keys in kwargs: {}'.format(kwargs.keys()))

    if lbl_true is None and lbl_pred is None:
        raise ValueError('lbl_true or lbl_pred must be not None.')

    lbl_true = copy.deepcopy(lbl_true)
    lbl_pred = copy.deepcopy(lbl_pred)

    mask_unlabeled = None
    viz_unlabeled = None
    if lbl_true is not None:
        mask_unlabeled = lbl_true == -1
        lbl_true[mask_unlabeled] = 0
        viz_unlabeled = (
            np.random.random((lbl_true.shape[0], lbl_true.shape[1], 3)) * 255
        ).astype(np.uint8)
        if lbl_pred is not None:
            lbl_pred[mask_unlabeled] = 0

    vizs = []

    if lbl_true is not None:
        viz_trues = [
            img,
            imgviz.label2rgb(label=lbl_true, label_names=label_names),
            imgviz.label2rgb(lbl_true, img, label_names=label_names),
        ]
        viz_trues[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        viz_trues[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(imgviz.tile(viz_trues, (1, 3)))

    if lbl_pred is not None:
        viz_preds = [
            img,
            imgviz.label2rgb(lbl_pred, label_names=label_names),
            imgviz.label2rgb(lbl_pred, img, label_names=label_names),
        ]
        if mask_unlabeled is not None and viz_unlabeled is not None:
            viz_preds[1][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
            viz_preds[2][mask_unlabeled] = viz_unlabeled[mask_unlabeled]
        vizs.append(imgviz.tile(viz_preds, (1, 3)))

    if len(vizs) == 1:
        return vizs[0]
    elif len(vizs) == 2:
        return imgviz.tile(vizs, (2, 1))
    else:
        raise RuntimeError


def plot_confusion_matrix(confusion_matrix, class_names=None):
    if class_names is None:
        class_names = [str(i) for i in range(np.max(confusion_matrix.shape))]
    # Create the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=True)

    # Label the axes
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(class_names, rotation=0, fontsize=10)

    # Set the labels and title
    ax.set_xlabel("Predicted labels, pixels * 10^6")
    ax.set_ylabel("True labels")
    plt.title("Confusion Matrix")

    # Display the plot
    plt.tight_layout()
    plt.savefig('conf_matrix.png')

import math
import tifffile as tiff
def plot_separate_mosaic(image_paths):
    """
    Plots all images in a single dynamically-adjusted grid.

    Args:
        image_paths (list): List of file paths to the images to be plotted.
    """
    # Calculate the number of rows and columns
    n_images = len(image_paths)
    if n_images == 0:
        print("No images to display.")
        return

    cols = math.ceil(math.sqrt(n_images))  # Number of columns
    rows = math.ceil(n_images / cols)  # Number of rows

    # Create the figure and axes
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()  # Flatten in case of a single row or column

    # Plot each image
    for i, path in enumerate(image_paths):
        try:
            img = tiff.imread(path)  # Use tifffile to read the image
            img = img[:, :, :3]
            axes[i].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
            axes[i].axis('off')  # Hide the axis for cleaner visualization
        except Exception as e:
            print(f"Error loading {path}: {e}")
    # Turn off remaining axes if images < grid size
    for i in range(n_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()
def create_mosaic(image_paths, rows=4, cols=5):
    """
    Arranges images into a fixed mosaic with the specified number of rows and columns.

    Args:
        image_paths (list): List of file paths to the images to be concatenated.
        rows (int): Number of rows in the mosaic.
        cols (int): Number of columns in the mosaic.

    Returns:
        mosaic_image: The final mosaic image as a NumPy array.
    """
    images = []

    # Load and resize images
    for path in image_paths:
        try:
            img = tiff.imread(path)[:, :, :3]  # Load image and take the first three channels
            images.append(img)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    if not images:
        print("No valid images to concatenate.")
        return

    # Determine the size of each cell
    target_height = min(img.shape[0] for img in images)
    target_width = min(img.shape[1] for img in images)

    # Resize all images to the same size
    resized_images = []
    for img in images:
        img = Image.fromarray(img)
        img = img.resize((target_width, target_height),
                         Image.Resampling.LANCZOS)  # Use LANCZOS for high-quality resizing
        resized_images.append(np.array(img))

    # Fill the mosaic grid
    grid = []
    for i in range(rows):
        row_images = resized_images[i * cols:(i + 1) * cols]
        if len(row_images) < cols:
            # Fill missing images with black images
            missing = cols - len(row_images)
            black_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)
            row_images.extend([black_image] * missing)
        grid.append(np.concatenate(row_images, axis=1))  # Concatenate images horizontally

    # Combine all rows into the final mosaic
    mosaic_image = np.concatenate(grid, axis=0)  # Concatenate rows vertically

    # Display the mosaic
    plt.figure(figsize=(15, 10))
    plt.imshow(mosaic_image)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

    return mosaic_image

def save_labels_on_image_exact(image, labels, output_path, alpha=0.5):
    """
    Creates and saves an image with labels overlaid with adjustable transparency, preserving exact resolution.

    Args:
        image (ndarray): The base image as a NumPy array (H, W, 3).
        labels (ndarray): The label mask as a NumPy array (H, W).
        output_path (str): Path to save the output image.
        alpha (float): The transparency of the label overlay (0.0 - 1.0).
    """
    output_name = f'blend_{int(alpha * 100)}.png'
    output_path = os.path.join(output_path, output_name)

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError("Image should be a 3-channel (H, W, 3) array.")
    if image.shape != labels.shape:
        raise ValueError("Image and labels must have the same spatial dimensions (H, W, 3).")

    # Blend the image and labels
    blended_image = (image * (1 - alpha) + labels * alpha).astype(np.uint8)

    # Save the blended image with Pillow
    Image.fromarray(blended_image).save(output_path)
    print(f"Overlayed image saved to {output_path}")

