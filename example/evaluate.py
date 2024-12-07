import argparse
import os
import os.path as osp
import torch

import torchconvs
import scripts

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    parser.add_argument('-repo', type=str, default='datasets/FLAIR/flair_dataset_test', help='Dataset repository')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser(f'~/{args.repo}')

    test_loader = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegBase(
            root, split='agri', transform=False, patch_size=512),
        batch_size=8, shuffle=False,
        num_workers=1, pin_memory=True)

    # # Patch-Based training and Full image Testing
    # test_dataset = torchconvs.datasets.FLAIRSegBase(root='data', split='test', transform=True,
    #                                                 patch_size=256, test=True, inference_mode='original')
    #
    # # Patch-Based training and Tile-Based image Testing
    # test_dataset = torchconvs.datasets.FLAIRSegBase(root='data', split='test', transform=True,
    #                                                 patch_size=256, test=True, inference_mode='tiles')
    #
    # # Full image training and Full image Testing
    # test_dataset = torchconvs.datasets.FLAIRSegBase(root='data', split='test', transform=True,
    #                                                 patch_size=None, test=True, inference_mode='original')



    n_class = len(test_loader.dataset.class_names)
    print(n_class)
    model_data = torch.load(model_file)
    print(model_data['arch'])
    model = scripts.utils.model_select(model_data['arch'], n_class)
    print(f'cuda: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    try:
        model.load_state_dict(model_data)
    except Exception:
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


    metrics, confusion_matrix, iou_per_class = torchconvs.validate(model, test_loader)

    scripts.utils.plot_confusion_matrix(confusion_matrix)


if __name__ == '__main__':
    main()
