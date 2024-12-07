import argparse
import os
import os.path as osp


import numpy as np
from torch.utils.data import DataLoader

import torch
from torchconvs.datasets import DatasetBase
import scripts
from scripts.utils import plot_image_label_classes

class CustomDataset(DatasetBase):
    """
    Custom dataset class inherited from DatasetBase with a custom class mapping.


    :param root: Dataset root directory.

    Root directory must contain img folder, optionally should contain msk folder

    :param split: Dataset split (train/val/test).
    :param transform: Whether to apply data augmentation transformations.
    :param patch_size: Patch size for cropping.
    """
    # Original classes of FLAIR. The models are trained to segment these classes:
    # 0 - Soil, Snow, clear-cuts, herbaceous vegetation, bushes
    # 1 - Pervious, Impervious and transportation surfaces and sports fields
    # 2 - Buildings, swimming pools, Green houses
    # 3 - Define a custom class mapping specific to this dataset
    # 4 - Trees
    # 5 - Agricultural surfaces
    # 6 - Water bodies

    # overwrite them with your mappings
    class_mapping = {
        4: 0, 10: 0, 14: 0, 15: 0, 8: 0, 19: 0,  # Soil, Snow, clear-cuts, herbaceous vegetation, bushes
        2: 1, 3: 1,                              # Pervious, Impervious and transportation surfaces and sports fields
        1: 2, 13: 2, 18: 2,                      # Buildings, swimming pools, Green houses
        6: 3, 7: 3, 16: 3, 17: 3,                # Trees
        9: 4, 11: 4, 12: 4,                      # Agricultural surfaces
        5: 5,                                    # Water bodies
    }

    # Mean and std of FLAIR dataset
    # You may want to overwrite them with mean and std accordingly for your dataset
    mean_rgb = np.array([113.775, 118.081, 109.273], dtype=np.float32)
    std_rgb = np.array([52.419, 46.028, 45.260], dtype=np.float32)

    def __init__(self, root: str, transform: bool = True, patch_size: int|bool =None):
        # Call the parent class constructor
        super().__init__(root, transform, patch_size)
        if transform:
            self.transforms = transform

        ##
        ## Ensure that self.files contain images and masks in a form:
        # self.files = [
        #     {'img': 'imgpath1', 'msk': 'mask path1'}
        #     {'img': 'imgpath2', 'msk': 'mask path2'}
        # ]
        ## The self.files might not contain msk paths, if msk folder is absent




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('--repo', type=str, help='Dataset repository')
    parser.add_argument('--model-path', type=str, help='Path to model .pth file')
    parser.add_argument('--output-label-path', type=str, default=None, help='Path to output label .pth file')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_path

    root = args.repo

    cuda = torch.cuda.is_available()
    print(f'Cuda available: {cuda}')
    torch.manual_seed(1337)
    if cuda:
        torch.cuda.manual_seed(1337)
        device = torch.device('cuda')

    root = osp.expanduser('~/datasets/vizualize')

    dataset = CustomDataset(root, patch_size=None)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1, shuffle=False,
        num_workers=1, pin_memory=True)



    n_class = dataloader.dataset.n_classes
    class_names = dataloader.dataset.class_names
    if args.output_label_path is None:
        output_dir = osp.join(root, 'output')


    model_data = torch.load(model_file)
    print(model_data['arch'])
    model = scripts.utils.model_select(model_data['arch'], n_class)

    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    model.to(device)

    with torch.no_grad():
        for batch_idx, (data, target, img_names) in enumerate(dataloader):
            if cuda:
                data, target = data.cuda(), target.cuda()
            score = model(data)

            imgs = data.detach().cpu()
            lbl_pred = score.detach().max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.detach().cpu()

            for img, lp, lt, img_name in zip(imgs, lbl_pred, lbl_true, img_names):

                img, lt = dataloader.dataset.untransform(img, lt)

                """
                This is my implementation of 
                visualization, but feel free to change
                :img: numpy.array([batch_size, 3, H, W])
                :lt: = numpy.array([batch_size, H, W])
                :lp: = numpy.array([batch_size, H, W])
                """

                # Create the output filename with 'label' postfix
                base_name, ext = os.path.splitext(img_name)
                label_name = f"{base_name}_label_pred{ext}"

                plot_image_label_classes(img, lp, class_names, output_dir=os.path.join(output_dir, label_name))


if __name__ == '__main__':
    main()