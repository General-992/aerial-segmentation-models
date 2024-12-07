import PIL
from torch.utils.data import Dataset, DataLoader
import math
import numpy as np
import torch


class Image_to_Tiles(Dataset):
    """
    Virtual dataset that generates non-overlapping tiles out of a high-resolution single image.
    """
    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)
    def __init__(self, feature_img, net_reso_hw):

        self._feature_img = feature_img
        img_reso_hw = tuple(feature_img.shape[:2])
        self._net_reso_hw = net_reso_hw

        # Strides are now equal to the resolution of the network (no overlap)
        self._strides_hw = self._net_reso_hw
        self._tile_reso_hw = tuple([int(math.ceil(img_d / stride_d))
                                    for img_d, stride_d in zip(img_reso_hw, self._strides_hw)])

        # Calculate the required padded resolution to ensure divisibility
        padded_reso_hw = tuple([s * d for s, d in zip(self._strides_hw, self._tile_reso_hw)])
        self._padded_img = np.zeros((*padded_reso_hw, 3), dtype=np.uint8)
        self._padded_img[0:img_reso_hw[0], 0:img_reso_hw[1], :] = feature_img

    def __len__(self):
        tr = self._tile_reso_hw
        return tr[0] * tr[1]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError(f"Index {index} out of range for tiles amount of {len(self)}")

        ih = index // self._tile_reso_hw[1]
        iw = index % self._tile_reso_hw[1]

        off_h = min(ih * self._strides_hw[0], self._padded_img.shape[0] - self._net_reso_hw[0])
        off_w = min(iw * self._strides_hw[1], self._padded_img.shape[1] - self._net_reso_hw[1])

        net_h, net_w = self._net_reso_hw
        tile = self._padded_img[off_h:off_h + net_h, off_w:off_w + net_w, :]
        tile_float = tile.astype(np.float32)
        tile_float_normal = self._normalize(tile_float)
        feature_img_chw = np.transpose(tile_float_normal, (2, 0, 1))
        feature_tensor = torch.tensor(feature_img_chw)
        return feature_tensor

    def compose(self, tile_list):
        """
        Gathers tiles back together into one big image.
        """
        big_img = np.zeros(self._padded_img.shape[:2], dtype=np.uint8)
        for index, tile_img in enumerate(tile_list):
            ih = index // self._tile_reso_hw[1]
            iw = index % self._tile_reso_hw[1]
            off_h = ih * self._strides_hw[0]
            off_w = iw * self._strides_hw[1]
            net_h, net_w = self._net_reso_hw
            big_img[off_h:off_h + net_h, off_w:off_w + net_w] = tile_img
        orig_h, orig_w = self._feature_img.shape[:2]
        big_crop = big_img[:orig_h, :orig_w]
        return big_crop

    def compose_image(self, tile_list):
        """
        Gathers tiles back together into one big RGB image.
        """
        # Initialize a blank canvas for the reconstructed RGB image
        big_img = np.zeros((*self._padded_img.shape[:2], 3), dtype=np.uint8)

        for index, tile_img in enumerate(tile_list):
            ih = index // self._tile_reso_hw[1]
            iw = index % self._tile_reso_hw[1]
            off_h = ih * self._strides_hw[0]
            off_w = iw * self._strides_hw[1]
            net_h, net_w = self._net_reso_hw

            # Ensure the tile is in HWC format and copy to the correct location
            if tile_img.shape[0] == 3:  # Convert CHW to HWC if needed

                tile_img = tile_img.transpose(1, 2, 0)
                tile_img *= self.std_rgb
                tile_img += self.mean_rgb
                tile_img = tile_img.astype(np.uint8)
            big_img[off_h:off_h + net_h, off_w:off_w + net_w, :] = tile_img

        # Crop to the original image dimensions
        orig_h, orig_w = self._feature_img.shape[:2]
        big_crop = big_img[:orig_h, :orig_w, :]
        return big_crop
    def _normalize(self, img):
        img -= self.mean_rgb
        img /= self.std_rgb
        return img




if __name__ == '__main__':
    import os.path as osp
    import PIL.Image as Image
    from matplotlib import pyplot as plt
    from scripts.utils import plot_image_label

    mean = torch.tensor([113.775, 118.081, 109.273]).view(1, 3, 1, 1)
    std = torch.tensor([52.419, 46.028, 45.260]).view(1, 3, 1, 1)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    root = osp.expanduser('~/datasets/viz_tile')

    image_path = 'img/image_mosaic.png'
    image_path = osp.join(root, image_path)
    image = PIL.Image.open(image_path).convert('RGB')
    print(image.size)
    image = np.array(image)
    print(image.shape)


    # Initialize the dataset
    image_tile = Image_to_Tiles(image, net_reso_hw=(512, 512))
    print(f"Number of Tiles: {len(image_tile)}")

    # Inference DataLoader (for batching, not strictly necessary here)
    inference_batch_size = 5
    num_workers = 2
    data_loader = DataLoader(image_tile, batch_size=inference_batch_size, num_workers=num_workers)

    # Introduce model
    from scripts.utils import model_select

    model_file = ('/home/general992/MASTERS/2024_ma_islambostanov/example/logs/hrnet_2024_12_03.20_11/model_best.pth.tar')
    model_data = torch.load(model_file)
    model = model_select(model_data['arch'], 6)
    model.eval()
    model.to(device)
    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])

    # Collect tiles for reconstruction
    tiles = []
    tiles_pred = []
    with torch.no_grad():
        for idx, batch in enumerate(data_loader):
            print(f"Processing batch {idx}, shape: {batch.shape}")

            # this is just to check whether mosaic restored corectly
            tiles.extend(batch.numpy())

            # Process batch
            batch = batch.to(device)
            pred = model(batch)

            decoded_batch = pred.detach().max(1)[1].cpu().numpy()
            tiles_pred.extend(decoded_batch)


    print(f"Collected {len(tiles)} tiles.")

    # Compose back into original image
    reconstructed_image = image_tile.compose_image(tiles)  # Convert CHW to HWC
    reconstructed_label = image_tile.compose(tiles_pred)
    plot_image_label(reconstructed_image, reconstructed_label, num_classes=6)