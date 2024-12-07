import argparse
import os
import os.path as osp

import numpy as np
import skimage.io
import torch

import torchconvs
import scripts
import tqdm

import json






def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('model_file', help='Model path')
    parser.add_argument('-g', '--gpu', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    model_file = args.model_file

    root = osp.expanduser('~/datasets/FLAIR')

    # Specify the file path
    meta_path = osp.join(root, 'flair-1_metadata_aerial.json')
    dataset_root = osp.join(root, 'flair_dataset_test')
    with open(meta_path, 'r') as file:
        data = json.load(file)


    test_loadeer = torch.utils.data.DataLoader(
        torchconvs.datasets.FLAIRSegMeta(
            root=dataset_root,metadata=data, split='test', transform=False, patch_size=None),
        batch_size=8, shuffle=False,
        num_workers=4, pin_memory=True)


    n_class = len(test_loadeer.dataset.class_names)
    model_data = torch.load(model_file)
    print(model_data['arch'])
    model = scripts.utils.model_select(model_data['arch'], n_class)

    if torch.cuda.is_available():
        model = model.cuda()
    print('==> Loading %s model file: %s' %
          (model.__class__.__name__, model_file))

    try:
        model.load_state_dict(model_data)
    except Exception:
        # For earlier torch versions
        model.load_state_dict(model_data['model_state_dict'])
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print('==> Evaluating Metadata%s' % model.__class__.__name__)

    month_metrics = {str(i): [] for i in range(1, 13)}
    time_metrics = {str(i): [] for i in range(0, 24)}
    for batch_idx, (data, target, month, camera, time) in tqdm.tqdm(enumerate(test_loadeer),
                                                              total=len(test_loadeer),
                                                              ncols=80, leave=False):
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            score = model(data)
        # Safely detach from the computation graph and move to CPU
        imgs = data.detach().cpu()
        lbl_pred = score.detach().max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.detach().cpu()

        for img, lt, lp, m, t in zip(imgs, lbl_true, lbl_pred, month, time):
            img, lt = test_loadeer.dataset.untransform(img, lt)

            acc, acc_cls, mean_iu, fwavacc, hist, IoU = scripts.metrics.label_accuracy_score(
                label_trues=lt, label_preds=lp, n_class=n_class)
            # ensure that strings
            month_str = str(int(m))
            time_str = str(int(t))
            if month_str in month_metrics:
                month_metrics[month_str].append((acc, acc_cls, mean_iu, fwavacc))
            if time_str in time_metrics:
                time_metrics[time_str].append((acc, acc_cls, mean_iu, fwavacc))

    for month, metrics in month_metrics.items():
        if metrics:
            metrics = np.mean(metrics, axis=0)
            metrics *= 100  # Convert to percentages
            print(f"Month: {month}")
            print('''\
            Accuracy: {0}
            Accuracy Class: {1}
            Mean IU: {2}
            FWAV Accuracy: {3}'''.format(*metrics))
    scripts.utils.plot_metrics_per_meta(month_metrics, model_data['arch'], meta_type = "Month")
    scripts.utils.plot_metrics_per_meta(time_metrics, model_data['arch'], meta_type="Hour")





if __name__ == '__main__':
    main()
