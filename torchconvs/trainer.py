import datetime
import math
import os
import os.path as osp
import shutil

import imgviz
import numpy as np
import pytz
import skimage.io
import torch

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import tqdm

import torchconvs
import scripts


class Trainer(object):

    def __init__(self, cuda, model, optimizer,
                 train_loader, val_loader, out, max_epoch,
                 size_average=False, interval_validate=None):
        self.cuda = cuda

        self.model = model
        self.optim = optimizer
        self.loss = CrossEntropyLoss()
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.timestamp_start = \
            datetime.datetime.now(pytz.timezone('Europe/Berlin'))
        self.size_average = size_average

        if interval_validate is None:
            self.interval_validate = len(self.train_loader)
        else:
            self.interval_validate = interval_validate

        self.out = out
        if not osp.exists(self.out):
            os.makedirs(self.out)

        self.log_headers = [
            'epoch',
            'iteration',
            'train/loss',
            'train/acc',
            'train/acc_cls',
            'train/mean_iu',
            'train/fwavacc',
            'valid/loss',
            'valid/acc',
            'valid/acc_cls',
            'valid/mean_iu',
            'valid/fwavacc',
            'elapsed_time',
        ]
        if not osp.exists(osp.join(self.out, 'log.csv')):
            with open(osp.join(self.out, 'log.csv'), 'w') as f:
                f.write(','.join(self.log_headers) + '\n')

        self.epoch = 0
        self.iteration = 0
        self.max_epoch = max_epoch
        self.best_mean_iu = 0

    def validate(self):
        training = self.model.training
        self.model.eval()

        n_class = len(self.val_loader.dataset.class_names)

        val_loss = 0
        visualizations = []
        metrics = []
        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.val_loader), total=len(self.val_loader),
                desc='Validation',
                leave=True, dynamic_ncols=True, mininterval=4):
            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data, target

            with torch.no_grad():
                torch.cuda.empty_cache()

                score = self.model(data)

            loss = self.loss(score, target)
            loss_data = loss.data.item()

            if np.isnan(loss_data):
                raise ValueError('loss is nan while validating')

            val_loss += loss_data / len(data)

            imgs = data.data.cpu()
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu()
            for img, lt, lp in zip(imgs, lbl_true, lbl_pred):
                img, lt = self.val_loader.dataset.untransform(img, lt)

                acc, acc_cls, mean_iu, fwavacc, hist, _ = scripts.metrics.label_accuracy_score(
                        label_trues=lt, label_preds=lp, n_class=n_class)
                metrics.append((acc, acc_cls, mean_iu, fwavacc))
                if len(visualizations) < 9:
                    viz = scripts.visualize_segmentation(
                        lbl_pred=lp, lbl_true=lt, img=img, n_class=n_class)
                    visualizations.append(viz)
            del imgs, lbl_pred, lbl_true

        metrics = np.mean(metrics, axis=0)

        out = osp.join(self.out, 'visualization_viz')
        if not osp.exists(out):
            os.makedirs(out)
        out_file = osp.join(out, 'epoch%04d.jpg' % self.epoch)
        skimage.io.imsave(out_file, imgviz.tile(visualizations))

        val_loss /= len(self.val_loader)

        with open(osp.join(self.out, 'log.csv'), 'a') as f:
            elapsed_time = (
                datetime.datetime.now(pytz.timezone('Europe/Berlin')) -
                self.timestamp_start).total_seconds()
            log = [self.epoch, self.iteration] + [''] * 5 + \
                  [val_loss] + list(metrics) + [elapsed_time]
            log = map(str, log)
            f.write(','.join(log) + '\n')

        mean_iu = metrics[2]
        is_best = mean_iu > self.best_mean_iu
        if is_best:
            self.best_mean_iu = mean_iu
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'arch': self.model.__class__.__name__,
            'optim_state_dict': self.optim.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'best_mean_iu': self.best_mean_iu,
        }, osp.join(self.out, 'checkpoint.pth.tar'))
        print(f'Checkpoint with the model state dictionary is saved to the {osp.join(self.out, "checkpoint.pth.tar")}')
        if is_best:
            shutil.copy(osp.join(self.out, 'checkpoint.pth.tar'),
                        osp.join(self.out, 'model_best.pth.tar'))

        if training:
            self.model.train()


    def train_epoch(self):
        self.model.train()

        n_class = len(self.train_loader.dataset.class_names)

        for batch_idx, (data, target) in tqdm.tqdm(
                enumerate(self.train_loader), total=len(self.train_loader),
                desc='Train epoch=%d' % self.epoch, ncols=80,
                leave=True, dynamic_ncols=True, mininterval=4):
            # restoring from the checkpoint, ensuring that train starts where it left off
            # calculate global iteration
            iteration = batch_idx + self.epoch * len(self.train_loader)
            if self.iteration != 0 and (iteration - 1) != self.iteration:
                # check if this is not the first run of training,
                # meanining we are resuming from the ckeckpoint
                # check if the current iteration matches with the checkpoint
                # if they do not match then current batch has already been
                # processed, so we skip to the next batch
                continue  # for resuming
            self.iteration = iteration
            # start validation after a specified number of iter-s or after all train batches
            if self.iteration % self.interval_validate == 0:
                self.validate()
                torch.cuda.empty_cache()
            assert self.model.training

            if self.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = data, target

            self.optim.zero_grad()
            torch.cuda.empty_cache()


            try:
                score = self.model(data)
            except Exception as e:
                print(f"Error during forward pass: {e}")
                print(f"Shape of input data: {data.shape}")


            loss = self.loss(score, target)
            loss /= len(data)
            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')
            loss.backward()
            self.optim.step()

            metrics = []
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iu, fwavacc, hist, _ = \
                scripts.metrics.label_accuracy_score(
                    lbl_true, lbl_pred, n_class=n_class)
            metrics.append((acc, acc_cls, mean_iu, fwavacc))
            metrics = np.mean(metrics, axis=0)

            with open(osp.join(self.out, 'log.csv'), 'a') as f:
                elapsed_time = (
                    datetime.datetime.now(pytz.timezone('Europe/Berlin')) -
                    self.timestamp_start).total_seconds()
                log = [self.epoch, self.iteration] + [loss_data] + \
                    metrics.tolist() + [''] * 5 + [elapsed_time]
                log = map(str, log)
                f.write(','.join(log) + '\n')


    def train(self):
        for epoch in tqdm.trange(self.epoch, self.max_epoch, desc='Train', ncols=80):
            self.epoch = epoch
            self.train_epoch()
            if self.epoch >= self.max_epoch:
                break


