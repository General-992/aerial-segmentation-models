import imgviz
import numpy as np
import skimage.io
import torch

import scripts
import tqdm


def validate(model, loader):
    print('==> Evaluating %s' % model.__class__.__name__)
    dataset_name = type(loader.dataset).__name__
    n_class = len(loader.dataset.class_names)
    visualizations = []
    metrics = []
    confusion_matrix = []
    avg_boundary_iou = []
    running_sum = np.zeros(n_class, dtype=np.float32)
    class_count = np.zeros(n_class, dtype=np.float32)
    for batch_idx, (data, target) in tqdm.tqdm(enumerate(loader),
                                               total=len(loader),
                                               ncols=80, leave=False):

        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()

        with torch.no_grad():
            score = model(data)

        # Safely detach from the computation graph and move to CPU
        imgs = data.detach().cpu()
        lbl_pred = score.detach().max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.detach().cpu()
        i = 0
        for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
            # reconstruct the original images
            img, lt = loader.dataset.untransform(img, lt)

            # discard the classes out of the range of ISPRS assigned classes
            if dataset_name.startswith('ISPRS'):
                lp[(lp == 4) | (lp == 5) | (lp == 6)] = -1

            acc, acc_cls, mean_iu, fwavacc, hist, IoU = scripts.metrics.label_accuracy_score(
                label_trues=lt, label_preds=lp, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))

            avg_boundary_iou.append(scripts.metrics.boundary_iou_multiclass(lt, lp, num_classes=6))
            # Ensure IoU is float32
            IoU = IoU.astype(np.float32)
            # Update the count only for classes where IoU is valid (not NaN)
            valid_iou_mask = ~np.isnan(IoU)
            # Use NaN-safe handling, e.g., replacing NaNs with 0 temporarily
            IoU_clean = np.nan_to_num(IoU, nan=0.0)
            # Accumulate the IoU sum for valid values
            running_sum += IoU_clean

              # True where IoU is not NaN
            class_count += valid_iou_mask

            confusion_matrix.append(hist)
            if len(visualizations) < 6:
                viz = scripts.utils.visualize_segmentation(
                    lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class,
                    label_names=loader.dataset.class_names)
                visualizations.append(viz)
                # TODO: save the visualizations

            i += 1


    confusion_matrix = np.sum(confusion_matrix, axis=0)
    confusion_matrix = (confusion_matrix / 1e6).astype(int)
    metrics = np.nanmean(metrics, axis=0)
    IoU_per_class = np.divide(running_sum, class_count, where=class_count > 0)

    avg_boundary_iou = 100 * np.nanmean(avg_boundary_iou)

    metrics *= 100
    print('Accuracy: {0:.4f}, Accuracy Class: {1:.4f}, '
          'Mean IU: {2:.4f}, FWAV Accuracy: {3:.4f}, '
          'Boundary IoU: {4:.4f}'.format(*metrics, avg_boundary_iou))
    print(IoU_per_class)
    #
    viz = imgviz.tile(visualizations)
    skimage.io.imsave('viz_evaluate.png', viz)
    return metrics, confusion_matrix, IoU_per_class

