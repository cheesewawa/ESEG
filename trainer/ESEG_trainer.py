#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Segmentation training script for DSEC.


'''
from __future__ import annotations

import argparse
import os
import os.path as osp
import random
import sys
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, PolynomialLR, StepLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Third‑party / local imports.
from model.segformer_build import EncoderDecoder
import pidinet.models as models
from pidinet.models.pidinet import PiDiNet
from utils.metrics import MetricsSemseg
from utils.utils_func import IOStream
from losses.loss_func import TaskLoss
from dst.dsec_dataset import DSECEvent
from DFFpytorch.models.dff import DFF

# -----------------------------------------------------------------------------#
# Environment and reproducibility                                              #
# -----------------------------------------------------------------------------#

# Restrict to the first GPU. Modify as required.
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Append project roots to PYTHONPATH.
ROOT_PATH = osp.abspath(__file__)
ROOT_DIR = osp.sep.join(ROOT_PATH.split(osp.sep)[:-2])
sys.path.extend([
    ROOT_DIR,
    '/home/ESEG/DDFpytorch',
])

# -----------------------------------------------------------------------------#
# Helper dataclasses                                                           #
# -----------------------------------------------------------------------------#


class PiDiArgs:
    '''
    Minimal wrapper around the original argparse arguments required by PiDiNet.
    Only values actually consumed by PiDiNet are included.
    '''

    def __init__(self) -> None:
        self.savedir: str = 'results/savedir'
        self.datadir: str = '../data'
        self.only_bsds: bool = False
        self.ablation: bool = False
        self.dataset: str = 'BSDS'
        self.model: str = 'pidinet'
        self.sa: bool = False
        self.dil: bool = False
        self.config: str = 'carv4'
        self.seed: int | None = None
        self.gpu: str = ''
        self.checkinfo: bool = False
        self.epochs: int = 20
        self.iter_size: int = 24
        self.lr: float = 0.005
        self.lr_type: str = 'multistep'
        self.lr_steps: list | None = None
        self.opt: str = 'adam'
        self.wd: float = 1e-4
        self.workers: int = 4
        self.eta: float = 0.3
        self.lmbda: float = 1.1
        self.resume: bool = False
        self.print_freq: int = 10
        self.save_freq: int = 1
        self.evaluate: str = 'table6_pidinet.pth'
        self.evaluate_converted: bool = True


PIDI_ARGS = PiDiArgs()

# -----------------------------------------------------------------------------#
# Trainer                                                                      #
# -----------------------------------------------------------------------------#


class Trainer:
    '''
    Main training / evaluation loop wrapper.
    '''

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.init_time = self._timestamp()
        self.save_dir = self._make_experiment_dir()
        self.io = IOStream(osp.join(self.save_dir, 'run.log'))
        self.io.cprint(str(self.args))

        # ------------------------------------------------------------------ #
        # Reproducibility                                                    #
        # ------------------------------------------------------------------ #
        torch.manual_seed(self.args.seed)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        # ------------------------------------------------------------------ #
        # Device                                                             #
        # ------------------------------------------------------------------ #
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args.cuda = self.device.type == 'cuda'
        if self.args.cuda:
            self.io.cprint(
                f'Using GPU {torch.cuda.current_device()} / {torch.cuda.device_count()} available',
            )
            torch.cuda.manual_seed(self.args.seed)
            torch.backends.cudnn.deterministic = True
        else:
            self.io.cprint('Using CPU')

        # ------------------------------------------------------------------ #
        # Data                                                               #
        # ------------------------------------------------------------------ #
        dsec_root = '/home/ESEG/DSEC'

        self.train_loader = DataLoader(
            DSECEvent(
                dsec_dir=dsec_root,
                delta_t_per_data=50,
                mode='train',
                nr_events_data=1,
                nr_bins_per_data=3,
                fixed_duration=True,
                augmentation=True,
                random_crop=True,
            ),
            num_workers=self.args.num_workers,
            batch_size=self.args.batch_size,
            shuffle=self.args.shuffle,
            drop_last=self.args.drop_last,
        )

        self.val_loader = DataLoader(
            DSECEvent(
                dsec_dir=dsec_root,
                delta_t_per_data=50,
                mode='val',
                nr_events_data=1,
                nr_bins_per_data=3,
                fixed_duration=True,
                augmentation=False,
                random_crop=False,
            ),
            num_workers=self.args.num_workers,
            batch_size=self.args.test_batch_size,
            shuffle=False,
            drop_last=False,
        )

        # ------------------------------------------------------------------ #
        # Model components                                                   #
        # ------------------------------------------------------------------ #
        self.backbone = EncoderDecoder()
        self.pidinet = getattr(models, PIDI_ARGS.model)(PIDI_ARGS).to(self.device)
        self._freeze(self.pidinet)

        self.dff = DFF(nclass=11, pretrained=True, device=self.device).to(self.device)
        self.dff.load_state_dict(torch.load('cityscapes_fine_45_11.pth')['state_dict'])
        self._freeze(self.dff)

        # Load SegFormer pretrained weights (ADE20K).
        segformer_ckpt = (
            '/home/ESEG/ckpt/segformer.b0.512x512.ade.160k.pth'
        )
        self._load_pretrained_weights(segformer_ckpt)

        # Replace classifier heads to match our 11‑class problem.
        self.backbone.decode_head.linear_pred = nn.Conv2d(256, 11, kernel_size=1)
        self.backbone.decode_head2.linear_pred = nn.Conv2d(512, 11, kernel_size=1)

        # Enable multi‑GPU if available.
        self.backbone = nn.DataParallel(self.backbone.to(self.device))
        self.backbone = torch.compile(self.backbone)  # PyTorch 2.0

        self.io.cprint(f'Let\'s use {torch.cuda.device_count()} GPUs!')

        # ------------------------------------------------------------------ #
        # Optimizer & scheduler                                              #
        # ------------------------------------------------------------------ #
        if self.args.use_sgd:
            self.io.cprint('Using SGD optimizer')
            self.opt = optim.SGD(
                self.backbone.parameters(), lr=self.args.lr, momentum=0.9, weight_decay=1e-4
            )
        else:
            self.io.cprint('Using AdamW optimizer')
            self.opt = optim.AdamW(
                self.backbone.parameters(), lr=self.args.lr, weight_decay=0.02, betas=(0.8, 0.99)
            )

        if self.args.scheduler == 'cos':
            self.scheduler = CosineAnnealingLR(self.opt, T_max=self.args.num_epochs, eta_min=1e-4)
        elif self.args.scheduler == 'step':
            self.scheduler = StepLR(self.opt, step_size=30, gamma=0.5)
        else:
            self.scheduler = PolynomialLR(
                self.opt, total_iters=self.args.num_epochs, power=1.0, verbose=True
            )

        # Separate optimizer for PiDiNet (auxiliary network).
        self.pidinet_opt = optim.Adam(self.pidinet.parameters(), lr=1e-4)

        # ------------------------------------------------------------------ #
        # Metrics and loss                                                   #
        # ------------------------------------------------------------------ #
        self.metrics = MetricsSemseg(
            num_classes=11,
            ignore_label=255,
            class_names=[
                'background',
                'building',
                'fence',
                'person',
                'pole',
                'road',
                'sidewalk',
                'vegetation',
                'car',
                'wall',
                'traffic sign',
            ],
        )

        self.criterion = TaskLoss(
            losses=['dice', 'cross_entropy'],
            gamma=2.0,
            num_classes=11,
            ignore_index=255,
        )

    # ---------------------------------------------------------------------- #
    # Training / evaluation                                                  #
    # ---------------------------------------------------------------------- #

    def train(self) -> None:
        '''
        Full training loop covering all epochs.
        '''
        best_miou = 0.0
        for epoch in range(self.args.num_epochs):
            self.metrics.reset()
            self._train_one_epoch(epoch)

            self.metrics.reset()
            best_miou = self._validate(epoch, best_miou)

            # Adjust learning rate.
            self.scheduler.step()

    def _train_one_epoch(self, epoch: int) -> None:
        '''
        Train for a single epoch.
        '''
        self.backbone.train()
        epoch_loss, num_samples = 0.0, 0

        for idx, (rgb, label, event) in enumerate(self.train_loader):
            rgb = rgb.float().to(self.device)
            event = event.to(self.device)
            label = self._prepare_label(label, rgb.shape[0])

            # Forward pass through auxiliary networks.
            _, _, _, _, _ = self.pidinet(rgb)
            _, _, _, f2, f3, f4, f5 = self.dff(event)

            # Forward pass through main network.
            logits_orig, logits = self.backbone(
                rgb, f2, f3, f4, f5, mode='whole', rescale=True
            )

            loss = self.criterion(logits_orig, label) * 0.2 + self.criterion(logits, label) * 0.8

            # Back‑prop.
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            # Statistics.
            epoch_loss += loss.item() * rgb.size(0)
            num_samples += rgb.size(0)

            preds = torch.argmax(logits, dim=1)
            self.metrics.update_batch(
                preds.unsqueeze(1).detach(), label.unsqueeze(1).detach()
            )

            if idx % 10 == 0:
                self.io.cprint(
                    f'[Epoch {epoch:03d}][{idx:04d}/{len(self.train_loader)}] Loss: {loss.item():.4f}'
                )

        summary = self.metrics.get_metrics_summary()
        mean_loss = epoch_loss / max(num_samples, 1)
        self.io.cprint(
            f'Train {epoch:03d} | Loss: {mean_loss:.4f} mAcc: {summary['acc']:.4f} mIoU: {summary['mean_iou']:.4f}'
        )

    def _validate(self, epoch: int, best_miou: float) -> float:
        '''
        Run evaluation on the validation split.
        '''
        self.backbone.eval()
        val_loss, num_samples = 0.0, 0

        with torch.no_grad():
            for rgb, label, event in self.val_loader:
                rgb = rgb.float().to(self.device)
                event = event.to(self.device)
                label = self._prepare_label(label, rgb.shape[0])

                _, _, _, f2, f3, f4, f5 = self.dff(event)
                logits_orig, logits = self.backbone(
                    rgb, f2, f3, f4, f5, mode='whole', rescale=True
                )

                loss = self.criterion(logits, label)

                val_loss += loss.item() * rgb.size(0)
                num_samples += rgb.size(0)

                preds = torch.argmax(logits, dim=1)
                self.metrics.update_batch(
                    preds.unsqueeze(1).detach(), label.unsqueeze(1).detach()
                )

        summary = self.metrics.get_metrics_summary()
        mean_loss = val_loss / max(num_samples, 1)
        self.io.cprint(
            f'Val   {epoch:03d} | Loss: {mean_loss:.4f} mAcc: {summary['acc']:.4f} mIoU: {summary['mean_iou']:.4f}'
        )

        # Save the best model.
        if summary['mean_iou'] >= best_miou:
            best_miou = summary['mean_iou']
            self._save_checkpoint(epoch, best_miou)

        return best_miou

    # ------------------------------------------------------------------ #
    # Utility helpers                                                    #
    # ------------------------------------------------------------------ #

    def _prepare_label(self, label: torch.Tensor, batch_size: int) -> torch.Tensor:
        '''
        Ensure label tensor has the expected shape across batch sizes.
        '''
        if batch_size == 1:
            return label.to(self.device)
        return label.to(self.device).squeeze()

    def _freeze(self, module: nn.Module) -> None:
        '''
        Freeze all parameters of a module.
        '''
        for param in module.parameters():
            param.requires_grad = False

    def _load_pretrained_weights(self, ckpt_path: str) -> None:
        '''
        Load weights from an official SegFormer checkpoint, ignoring
        unmatched keys.
        '''
        ckpt = torch.load(ckpt_path)
        pretrained = ckpt['state_dict']
        own_state = self.backbone.state_dict()

        missed, loaded = [], []
        for k in own_state.keys():
            if k in pretrained:
                own_state[k] = pretrained[k]
                loaded.append(k)
            else:
                missed.append(k)

        self.backbone.load_state_dict(own_state)
        self.io.cprint(
            f'Loaded {len(loaded)} keys from pretrained checkpoint, missed {len(missed)}.'
        )

    def _make_experiment_dir(self) -> str:
        '''
        Create an experiment directory at
        /home/undergrad/AyChihuahua/SAM/RGB_only_v0/log_file/<exp_name>/<timestamp>.
        '''
        base = '/home/undergrad/AyChihuahua/SAM/RGB_only_v0/log_file'
        path = osp.join(base, self.args.exp_name, self.init_time)
        os.makedirs(path, exist_ok=True)
        return path

    def _timestamp(self) -> str:
        '''
        Human‑readable timestamp in local time.
        '''
        return time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())

    def _save_checkpoint(self, epoch: int, best_score: float) -> None:
        '''
        Persist the current model to disk.
        '''
        checkpoint = {
            'Weights': self.backbone.state_dict(),
            'Epoch_num': epoch,
            'best_score': best_score,
        }
        torch.save(checkpoint, osp.join(self.save_dir, 'BestModel.pth'))
        self.io.cprint(
            f'Checkpoint saved at epoch {epoch:03d} (mIoU={best_score:.4f}).'
        )


# -----------------------------------------------------------------------------#
# CLI                                                                          #
# -----------------------------------------------------------------------------#


def parse_args() -> argparse.Namespace:
    '''
    Parse command‑line arguments.
    '''
    parser = argparse.ArgumentParser(description='SegFormer RGB‑event training')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='RGB_SEG_DSEC', help='Name of the experiment')
    parser.add_argument('--eval', action='store_true', help='Run evaluation only')
    parser.add_argument('--seed', type=int, default=1, help='Random seed')

    # Checkpoints
    parser.add_argument('--model_path', type=str, default='', help='Checkpoint path')
    parser.add_argument('--train_from_checkpoint', action='store_true', help='Resume training')

    # Data loader
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--drop_last', action='store_true', default=False)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--test_batch_size', type=int, default=4)

    # Optimizer / scheduler
    parser.add_argument('--use_sgd', action='store_true', help='Use SGD instead of AdamW')
    parser.add_argument('--lr', type=float, default=6e-5, help='Initial learning rate')
    parser.add_argument('--num_epochs', type=int, default=60)
    parser.add_argument(
        '--scheduler', type=str, default='poly', choices=['cos', 'step', 'poly'], help='LR scheduler'
    )

    return parser.parse_args()


def main() -> None:
    '''
    Entrypoint for training or evaluation.
    '''
    args = parse_args()
    trainer = Trainer(args)

    if args.eval:
        trainer._validate(epoch=0, best_miou=0.0)
    else:
        trainer.train()


if __name__ == '__main__':
    main()
