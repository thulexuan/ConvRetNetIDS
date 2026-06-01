from __future__ import print_function

"""
One-view PaCo training for CCT-RetNet on imbalanced CAN/IDS image data.

This script is adapted from the user's SupCon/SelfCon training pipeline.
Main changes:
  1) Uses CCTRetNet as backbone to produce one embedding per input image.
  2) Uses a PaCo-style loss with learnable class centers from a classifier head.
  3) Does not require augmented views or im_k.
  4) Uses a class-balanced memory queue and optional weighted sampler for imbalance.
  5) Tracks macro-F1 as the main validation metric.

Expected project imports:
  - dataset.CANDatasetEnet
  - networks.cct_retnet.CCTRetNet
  - util.AverageMeter, util.warmup_learning_rate, util.save_model

Run example:
python train_cct_retnet_paco_imbalance.py \
  --data_folder ./data/size_64_10/TFRecord_w64_s32/2/ \
  --n_classes 10 --batch_size 128 --epochs 100 \
  --optimizer AdamW --learning_rate 3e-4 \
  --sampler weighted --queue_mode class_balanced \
  --paco_queue_size 8192 --paco_alpha 0.05 --temp 0.07
"""

import argparse
import math
import os
import random
import sys
import time
import pprint
from collections import deque
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from dataset import CANDatasetEnet as CANDataset
from networks.cct_retnet import CCTRetNet
from util import AverageMeter, warmup_learning_rate, save_model

try:
    from networks.resnet_big import LinearClassifier
except Exception:
    class LinearClassifier(nn.Module):
        """Fallback classifier if networks.resnet_big.LinearClassifier is unavailable."""
        def __init__(self, num_classes=10, feat_dim=256):
            super().__init__()
            self.fc = nn.Linear(feat_dim, num_classes)

        def forward(self, features):
            return self.fc(features)


# -----------------------------
# Utilities
# -----------------------------

def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None or value == '':
        return []
    return [int(v.strip()) for v in value.split(',') if v.strip()]


def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.benchmark = True


def safe_label_to_int(label):
    if isinstance(label, torch.Tensor):
        return int(label.detach().cpu().item())
    if isinstance(label, np.ndarray):
        return int(label.item())
    return int(label)


def infer_dataset_labels(dataset):
    """Try to retrieve labels from dataset without assuming a specific attribute name.

    Falls back to iterating over the dataset once. This may take a little time,
    but it is useful for creating weighted samplers and class priors.
    """
    candidate_attrs = [
        'labels', 'targets', 'y', 'ys', 'label', 'train_labels', 'train_targets'
    ]
    for attr in candidate_attrs:
        if hasattr(dataset, attr):
            values = getattr(dataset, attr)
            try:
                arr = np.asarray(values, dtype=np.int64)
                if arr.ndim == 1 and len(arr) == len(dataset):
                    return arr
            except Exception:
                pass

    labels = []
    print('Dataset labels were not found as an attribute. Scanning dataset once...')
    for i in tqdm(range(len(dataset)), desc='Scanning labels'):
        item = dataset[i]
        if isinstance(item, (tuple, list)) and len(item) >= 2:
            labels.append(safe_label_to_int(item[1]))
        else:
            raise RuntimeError(
                'Cannot infer labels. Dataset __getitem__ must return (image, label).'
            )
    return np.asarray(labels, dtype=np.int64)


def compute_class_counts(labels, num_classes):
    counts = np.bincount(labels.astype(np.int64), minlength=num_classes)
    return counts.astype(np.int64)


def make_class_priors(counts, eps=1e-12):
    counts = counts.astype(np.float64)
    priors = counts / max(counts.sum(), eps)
    priors = np.clip(priors, eps, 1.0)
    return priors.astype(np.float32)


def make_weighted_sampler(labels, num_classes):
    counts = compute_class_counts(labels, num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_weights = 1.0 / counts
    sample_weights = class_weights[labels.astype(np.int64)]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


# -----------------------------
# PaCo components
# -----------------------------

class ClassBalancedFeatureQueue:
    """A simple class-balanced feature queue.

    Instead of one global FIFO queue that can be dominated by majority classes,
    this queue keeps a per-class deque. The returned queue is approximately
    balanced across classes.
    """
    def __init__(self, num_classes, queue_size=8192, feature_dim=128, device='cuda'):
        self.num_classes = int(num_classes)
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.per_class_size = max(1, int(math.ceil(self.queue_size / self.num_classes)))
        self.queues = [deque(maxlen=self.per_class_size) for _ in range(self.num_classes)]

    @torch.no_grad()
    def enqueue(self, features, labels):
        features = features.detach().cpu()
        labels = labels.detach().cpu().long()
        for feat, lab in zip(features, labels):
            lab_i = int(lab.item())
            if 0 <= lab_i < self.num_classes:
                self.queues[lab_i].append(feat.clone())

    def get(self):
        feats = []
        labs = []
        for c, q in enumerate(self.queues):
            if len(q) == 0:
                continue
            feats.extend(list(q))
            labs.extend([c] * len(q))
        if len(feats) == 0:
            empty_f = torch.empty(0, self.feature_dim, device=self.device)
            empty_l = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_f, empty_l
        features = torch.stack(feats, dim=0).to(self.device, non_blocking=True)
        labels = torch.tensor(labs, dtype=torch.long, device=self.device)
        return features, labels

    def __len__(self):
        return sum(len(q) for q in self.queues)


class GlobalFeatureQueue:
    """Global FIFO queue. Provided mainly for ablation."""
    def __init__(self, queue_size=8192, feature_dim=128, device='cuda'):
        self.queue_size = int(queue_size)
        self.feature_dim = int(feature_dim)
        self.device = torch.device(device)
        self.features = deque(maxlen=self.queue_size)
        self.labels = deque(maxlen=self.queue_size)

    @torch.no_grad()
    def enqueue(self, features, labels):
        features = features.detach().cpu()
        labels = labels.detach().cpu().long()
        for feat, lab in zip(features, labels):
            self.features.append(feat.clone())
            self.labels.append(int(lab.item()))

    def get(self):
        if len(self.features) == 0:
            empty_f = torch.empty(0, self.feature_dim, device=self.device)
            empty_l = torch.empty(0, dtype=torch.long, device=self.device)
            return empty_f, empty_l
        features = torch.stack(list(self.features), dim=0).to(self.device, non_blocking=True)
        labels = torch.tensor(list(self.labels), dtype=torch.long, device=self.device)
        return features, labels

    def __len__(self):
        return len(self.features)


class OneViewPaCoLoss(nn.Module):
    """One-view PaCo-style loss.

    Inputs:
      features:   [B + Q, D], where first B rows are current batch embeddings q
      labels:     [B + Q], labels for current batch and queue embeddings
      sup_logits: [B, C], classifier logits from raw features; classifier weights
                  play the role of learnable class centers.

    Positive terms for each anchor:
      - its correct class center from sup_logits
      - same-class samples from current batch and queue

    Negative terms:
      - wrong class centers
      - different-class samples from current batch and queue
    """
    def __init__(
        self,
        num_classes=10,
        temperature=0.07,
        base_temperature=None,
        alpha=0.05,
        beta=1.0,
        gamma=1.0,
        supt=1.0,
        center_rebalance=True,
        class_priors=None,
        eps=1e-12,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.base_temperature = float(base_temperature or temperature)
        self.alpha = float(alpha)   # weight for sample-positive terms
        self.beta = float(beta)     # weight for class-center positive term
        self.gamma = float(gamma)   # denominator scaling for sample logits
        self.supt = float(supt)     # temperature for supervised/class-center logits
        self.center_rebalance = bool(center_rebalance)
        self.eps = float(eps)

        if class_priors is None:
            class_priors = torch.ones(self.num_classes) / self.num_classes
        class_priors = torch.as_tensor(class_priors, dtype=torch.float32)
        class_priors = class_priors.clamp_min(self.eps)
        self.register_buffer('log_class_priors', class_priors.log().view(1, -1))

    def forward(self, features, labels, sup_logits):
        device = features.device
        labels = labels.contiguous().view(-1).long().to(device)
        batch_size = int(sup_logits.shape[0])

        if features.ndim != 2:
            raise ValueError('features must be [B + Q, D].')
        if labels.shape[0] != features.shape[0]:
            raise ValueError('labels length must match features length.')
        if sup_logits.shape[1] != self.num_classes:
            raise ValueError('sup_logits second dimension must equal num_classes.')

        # sample-to-sample logits: anchors are only current batch features.
        anchor_features = features[:batch_size]
        sample_logits = torch.matmul(anchor_features, features.T) / self.temperature

        # sample-to-class-center logits. Balanced Softmax center rebalance is
        # implemented by adding log class priors to class-center logits.
        center_logits = sup_logits / self.supt
        if self.center_rebalance:
            center_logits = center_logits + self.log_class_priors.to(device)

        logits = torch.cat([center_logits, sample_logits], dim=1)
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # Positive sample mask: same label, excluding self-contrast for the B anchors.
        anchor_labels = labels[:batch_size].view(-1, 1)
        sample_pos_mask = torch.eq(anchor_labels, labels.view(1, -1)).float()

        sample_logits_mask = torch.ones_like(sample_pos_mask)
        sample_logits_mask[:, :batch_size].scatter_(
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0.0,
        )
        sample_pos_mask = sample_pos_mask * sample_logits_mask

        # Positive class-center mask: one-hot class center.
        center_pos_mask = nnF.one_hot(
            labels[:batch_size], num_classes=self.num_classes
        ).float().to(device)

        # Final masks.
        pos_mask = torch.cat(
            [self.beta * center_pos_mask, self.alpha * sample_pos_mask], dim=1
        )
        denom_mask = torch.cat(
            [torch.ones(batch_size, self.num_classes, device=device),
             self.gamma * sample_logits_mask], dim=1
        )

        exp_logits = torch.exp(logits) * denom_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + self.eps)

        pos_weight = pos_mask.sum(dim=1).clamp_min(self.eps)
        mean_log_prob_pos = (pos_mask * log_prob).sum(dim=1) / pos_weight

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


# -----------------------------
# Options / data / model
# -----------------------------

def parse_option():
    parser = argparse.ArgumentParser('CCT-RetNet one-view PaCo training for imbalanced data')

    # basic
    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--optimizer', type=str, default='AdamW', choices=['SGD', 'RMSprop', 'Adam', 'AdamW'])

    # model / dataset
    parser.add_argument('--model', type=str, default='CCT-RetNet-OneViewPaCo')
    parser.add_argument('--dataset', type=str, default='CAN-ML', choices=['CAN-ML', 'CAN', 'ROAD'])
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=32)

    # CCT-RetNet
    parser.add_argument('--embedding_dim', type=int, default=128)
    parser.add_argument('--projection_dim', type=int, default=64)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=2)
    parser.add_argument('--mlp_ratio', type=float, default=2.5)
    parser.add_argument('--n_conv_layers', type=int, default=2)
    parser.add_argument('--dropout_rate', type=float, default=0.0)
    parser.add_argument('--attention_dropout', type=float, default=0.1)
    parser.add_argument('--stochastic_depth_rate', type=float, default=0.1)

    # PaCo
    parser.add_argument('--temp', type=float, default=0.07, help='sample contrast temperature')
    parser.add_argument('--paco_alpha', type=float, default=0.05, help='weight for sample positives')
    parser.add_argument('--paco_beta', type=float, default=1.0, help='weight for correct class center')
    parser.add_argument('--paco_gamma', type=float, default=1.0, help='denominator weight for sample logits')
    parser.add_argument('--paco_supt', type=float, default=1.0, help='temperature for class-center logits')
    parser.add_argument('--paco_queue_size', type=int, default=8192)
    parser.add_argument('--queue_mode', type=str, default='class_balanced', choices=['class_balanced', 'global', 'none'])
    parser.add_argument('--center_rebalance', dest='center_rebalance', action='store_true')
    parser.add_argument('--no_center_rebalance', dest='center_rebalance', action='store_false')
    parser.set_defaults(center_rebalance=True)
    parser.add_argument('--ce_weight', type=float, default=0.0, help='optional CE term added to PaCo loss')

    # imbalance
    parser.add_argument('--sampler', type=str, default='weighted', choices=['weighted', 'shuffle'])
    parser.add_argument('--use_weighted_ce', action='store_true', help='use class-weighted CE when ce_weight > 0')

    # training schedule
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--early_stop_patience', type=int, default=20)
    parser.add_argument('--method', type=str, default='OneViewPaCo')

    opt = parser.parse_args()

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if opt.data_folder is None:
        opt.data_folder = './data/size_64_10/TFRecord_w64_s32/3/'

    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',') if it.strip()]

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)
    opt.model_name = '{}_{}_{}_lr_{}_wd_{}_bsz_{}_trial_{}_epochs_{}'.format(
        opt.method, opt.dataset, opt.model, opt.learning_rate,
        opt.weight_decay, opt.batch_size, opt.trial, opt.epochs
    )
    if opt.cosine:
        opt.model_name += '_cosine'
    if opt.warm:
        opt.model_name += '_warm'

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, current_time)
    os.makedirs(opt.save_folder, exist_ok=True)
    opt.log_file = os.path.join(opt.save_folder, 'log.txt')
    opt.tb_folder = os.path.join(opt.save_folder, 'runs')
    return opt


def set_loader(opt):
    train_dataset = CANDataset(
        root_dir=opt.data_folder,
        window_size=opt.window_size,
        is_train=True,
        is_test=False,
    )
    test_dataset = CANDataset(
        root_dir=opt.data_folder,
        window_size=opt.window_size,
        is_train=False,
        is_test=False,
        include_data=False,
    )

    train_labels = infer_dataset_labels(train_dataset)
    class_counts = compute_class_counts(train_labels, opt.n_classes)
    class_priors = make_class_priors(class_counts)

    print('Class counts:', class_counts.tolist())
    print('Class priors:', np.round(class_priors, 6).tolist())
    nonzero_counts = class_counts[class_counts > 0]
    if len(nonzero_counts) > 0:
        print('Imbalance ratio max/min:', float(nonzero_counts.max() / nonzero_counts.min()))

    if opt.sampler == 'weighted':
        sampler = make_weighted_sampler(train_labels, opt.n_classes)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, test_loader, class_counts, class_priors


def set_model(opt, class_priors, class_counts):
    model = CCTRetNet(
        img_size=opt.img_size,
        embedding_dim=opt.embedding_dim,
        n_conv_layers=opt.n_conv_layers,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=2,
        pooling_stride=2,
        pooling_padding=0,
        num_layers=opt.num_layers,
        num_heads=opt.num_heads,
        mlp_ratio=opt.mlp_ratio,
        num_classes=opt.n_classes,
        dropout_rate=opt.dropout_rate,
        attention_dropout=opt.attention_dropout,
        stochastic_depth_rate=opt.stochastic_depth_rate,
        positional_embedding='learnable',
        projection_head='mlp',
        projection_dim=opt.projection_dim,
    ).to(opt.device)

    classifier = LinearClassifier(num_classes=opt.n_classes, feat_dim=opt.embedding_dim).to(opt.device)

    paco_criterion = OneViewPaCoLoss(
        num_classes=opt.n_classes,
        temperature=opt.temp,
        alpha=opt.paco_alpha,
        beta=opt.paco_beta,
        gamma=opt.paco_gamma,
        supt=opt.paco_supt,
        center_rebalance=opt.center_rebalance,
        class_priors=class_priors,
    ).to(opt.device)

    if opt.use_weighted_ce:
        counts = np.maximum(class_counts.astype(np.float32), 1.0)
        ce_weights = counts.sum() / (opt.n_classes * counts)
        ce_weights = torch.tensor(ce_weights, dtype=torch.float32, device=opt.device)
        ce_criterion = nn.CrossEntropyLoss(weight=ce_weights)
    else:
        ce_criterion = nn.CrossEntropyLoss()

    print('Model:', model)
    print('Classifier/class centers:', classifier)
    print('PaCo criterion:', paco_criterion)

    total_params = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in classifier.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) + \
        sum(p.numel() for p in classifier.parameters() if p.requires_grad)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    return model, classifier, paco_criterion, ce_criterion


optimize_dict = {
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}


def set_optimizer(opt, modules):
    if not isinstance(modules, (list, tuple, nn.ModuleList)):
        modules = [modules]
    params = []
    for module in modules:
        params += list(module.parameters())

    opt_cls = optimize_dict[opt.optimizer]
    if opt.optimizer in ['Adam', 'AdamW']:
        optimizer = opt_cls(params, lr=opt.learning_rate, weight_decay=opt.weight_decay)
    else:
        optimizer = opt_cls(
            params,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay,
        )
    return optimizer


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


# -----------------------------
# Train / validate
# -----------------------------

def make_queue(opt):
    if opt.queue_mode == 'none' or opt.paco_queue_size <= 0:
        return None
    if opt.queue_mode == 'class_balanced':
        return ClassBalancedFeatureQueue(
            num_classes=opt.n_classes,
            queue_size=opt.paco_queue_size,
            feature_dim=opt.projection_dim,
            device=opt.device,
        )
    return GlobalFeatureQueue(
        queue_size=opt.paco_queue_size,
        feature_dim=opt.projection_dim,
        device=opt.device,
    )


def train_one_epoch(train_loader, model, classifier, criterion, ce_criterion,
                    optimizer, epoch, opt, queue_bank, step, logger=None):
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    paco_losses = AverageMeter()
    ce_losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True).long()
        bsz = labels.size(0)

        if opt.warm:
            warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # CCT-RetNet backbone embeddings.
        raw_features = model.encode(images)                  # [B, embedding_dim]
        q = model.projection(raw_features)                   # [B, projection_dim]
        q = nnF.normalize(q, dim=1)

        # Learnable class-center logits via classifier head.
        sup_logits = classifier(raw_features)                # [B, num_classes]

        if queue_bank is not None:
            queue_features, queue_labels = queue_bank.get()
            features_for_loss = torch.cat([q, queue_features], dim=0)
            labels_for_loss = torch.cat([labels, queue_labels], dim=0)
        else:
            features_for_loss = q
            labels_for_loss = labels

        paco_loss = criterion(features_for_loss, labels_for_loss, sup_logits)
        if opt.ce_weight > 0:
            ce_loss = ce_criterion(sup_logits, labels)
            loss = paco_loss + opt.ce_weight * ce_loss
        else:
            ce_loss = torch.zeros((), device=opt.device)
            loss = paco_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Enqueue after update. Use detached current embeddings.
        if queue_bank is not None:
            queue_bank.enqueue(q.detach(), labels.detach())

        losses.update(loss.item(), bsz)
        paco_losses.update(paco_loss.item(), bsz)
        ce_losses.update(ce_loss.item(), bsz)

        batch_time.update(time.time() - end)
        end = time.time()

        if logger is not None and step % opt.print_freq == 0:
            logger.add_scalar('loss/train_total', losses.avg, step)
            logger.add_scalar('loss/train_paco', paco_losses.avg, step)
            logger.add_scalar('loss/train_ce', ce_losses.avg, step)
            if queue_bank is not None:
                logger.add_scalar('queue/size', len(queue_bank), step)

        if (idx + 1) % opt.print_freq == 0:
            msg = (
                'Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'loss {loss.val:.5f} ({loss.avg:.5f})\t'
                'paco {paco.val:.5f} ({paco.avg:.5f})\t'
                'ce {ce.val:.5f} ({ce.avg:.5f})'.format(
                    epoch, idx + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, paco=paco_losses, ce=ce_losses,
                )
            )
            print(msg)
            sys.stdout.flush()

    return step, losses.avg, paco_losses.avg, ce_losses.avg


def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    return pred.t().detach().cpu().numpy().squeeze(0)


@torch.no_grad()
def validate(val_loader, model, classifier, ce_criterion, opt):
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    total_pred = []
    total_label = []

    for images, labels in tqdm(val_loader, desc='Validate'):
        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True).long()
        bsz = labels.size(0)

        raw_features = model.encode(images)
        outputs = classifier(raw_features)
        loss = ce_criterion(outputs, labels)
        losses.update(loss.item(), bsz)

        pred = get_predict(outputs)
        total_pred.append(pred)
        total_label.append(labels.detach().cpu().numpy())

    total_pred = np.concatenate(total_pred, axis=0)
    total_label = np.concatenate(total_label, axis=0)

    metrics = {
        'loss': losses.avg,
        'accuracy': accuracy_score(total_label, total_pred) * 100.0,
        'balanced_accuracy': balanced_accuracy_score(total_label, total_pred) * 100.0,
        'macro_f1': f1_score(total_label, total_pred, average='macro', zero_division=0),
        'weighted_f1': f1_score(total_label, total_pred, average='weighted', zero_division=0),
        'macro_precision': precision_score(total_label, total_pred, average='macro', zero_division=0),
        'macro_recall': recall_score(total_label, total_pred, average='macro', zero_division=0),
        'weighted_precision': precision_score(total_label, total_pred, average='weighted', zero_division=0),
        'weighted_recall': recall_score(total_label, total_pred, average='weighted', zero_division=0),
        'per_class_recall': recall_score(total_label, total_pred, average=None, zero_division=0),
        'per_class_f1': f1_score(total_label, total_pred, average=None, zero_division=0),
        'confusion_matrix': confusion_matrix(total_label, total_pred),
    }
    return metrics


class EarlyStoppingByMacroF1:
    def __init__(self, patience=20, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = -np.inf
        self.best_epoch = 0
        self.early_stop = False

    def step(self, score, epoch):
        if score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return False


# -----------------------------
# Main
# -----------------------------

def main():
    opt = parse_option()
    set_seed(opt.seed)
    print(opt)

    train_loader, test_loader, class_counts, class_priors = set_loader(opt)
    model, classifier, paco_criterion, ce_criterion = set_model(opt, class_priors, class_counts)
    optimizer = set_optimizer(opt, [model, classifier])
    queue_bank = make_queue(opt)

    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    log_writer = open(opt.log_file, 'w')
    pp = pprint.PrettyPrinter(indent=4)

    log_writer.write(str(opt) + '\n')
    log_writer.write('Class counts: {}\n'.format(class_counts.tolist()))
    log_writer.write('Class priors: {}\n'.format(class_priors.tolist()))

    early_stopping = EarlyStoppingByMacroF1(patience=opt.early_stop_patience)
    step = 0
    best_macro_f1 = -np.inf

    for epoch in range(1, opt.epochs + 1):
        print('\nBegin epoch {} at {}'.format(epoch, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        lr = adjust_learning_rate(opt, optimizer, epoch)
        logger.add_scalar('lr', lr, epoch)

        step, train_loss, train_paco_loss, train_ce_loss = train_one_epoch(
            train_loader, model, classifier, paco_criterion, ce_criterion,
            optimizer, epoch, opt, queue_bank, step, logger,
        )

        print('Epoch {}, train loss {:.4f}, paco {:.4f}, ce {:.4f}'.format(
            epoch, train_loss, train_paco_loss, train_ce_loss
        ))
        log_writer.write('Epoch {}, train loss {:.4f}, paco {:.4f}, ce {:.4f}\n'.format(
            epoch, train_loss, train_paco_loss, train_ce_loss
        ))

        metrics = validate(test_loader, model, classifier, ce_criterion, opt)

        logger.add_scalar('val/loss', metrics['loss'], epoch)
        logger.add_scalar('val/accuracy', metrics['accuracy'], epoch)
        logger.add_scalar('val/balanced_accuracy', metrics['balanced_accuracy'], epoch)
        logger.add_scalar('val/macro_f1', metrics['macro_f1'], epoch)
        logger.add_scalar('val/weighted_f1', metrics['weighted_f1'], epoch)
        logger.add_scalar('val/macro_recall', metrics['macro_recall'], epoch)

        print('Validation loss: {:.4f}'.format(metrics['loss']))
        print('Accuracy: {:.4f}'.format(metrics['accuracy']))
        print('Balanced accuracy: {:.4f}'.format(metrics['balanced_accuracy']))
        print('Macro-F1: {:.4f}'.format(metrics['macro_f1']))
        print('Weighted-F1: {:.4f}'.format(metrics['weighted_f1']))
        print('Macro precision: {:.4f}'.format(metrics['macro_precision']))
        print('Macro recall: {:.4f}'.format(metrics['macro_recall']))
        print('Per-class recall:', np.round(metrics['per_class_recall'], 4))
        print('Per-class F1:', np.round(metrics['per_class_f1'], 4))
        print('Confusion Matrix:')
        print(metrics['confusion_matrix'])

        log_writer.write('Validation loss: {:.4f}\n'.format(metrics['loss']))
        log_writer.write('Accuracy: {:.4f}\n'.format(metrics['accuracy']))
        log_writer.write('Balanced accuracy: {:.4f}\n'.format(metrics['balanced_accuracy']))
        log_writer.write('Macro-F1: {:.4f}\n'.format(metrics['macro_f1']))
        log_writer.write('Weighted-F1: {:.4f}\n'.format(metrics['weighted_f1']))
        log_writer.write('Macro precision: {:.4f}\n'.format(metrics['macro_precision']))
        log_writer.write('Macro recall: {:.4f}\n'.format(metrics['macro_recall']))
        log_writer.write('Per-class recall: {}\n'.format(np.round(metrics['per_class_recall'], 4).tolist()))
        log_writer.write('Per-class F1: {}\n'.format(np.round(metrics['per_class_f1'], 4).tolist()))
        log_writer.write('Confusion Matrix:\n{}\n'.format(pp.pformat(metrics['confusion_matrix'])))
        log_writer.flush()

        # Save best by macro-F1, not accuracy, because the dataset is imbalanced.
        improved = early_stopping.step(metrics['macro_f1'], epoch)
        if improved:
            best_macro_f1 = metrics['macro_f1']
            save_path = os.path.join(opt.save_folder, 'best_macro_f1.pth')
            save_model(
                model,
                optimizer,
                opt,
                epoch,
                save_path,
            )
            print('Saved best macro-F1 model at epoch {}: {:.4f}'.format(epoch, best_macro_f1))

        if epoch % opt.save_freq == 0:
            save_path = os.path.join(opt.save_folder, 'ckpt_epoch_{}.pth'.format(epoch))
            save_model(model, optimizer, opt, epoch, save_path)

        if early_stopping.early_stop:
            print('Early stopping at epoch {}. Best macro-F1 {:.4f} at epoch {}'.format(
                epoch, early_stopping.best_score, early_stopping.best_epoch
            ))
            break

    save_path = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, epoch, save_path)
    log_writer.close()
    logger.close()


if __name__ == '__main__':
    main()
