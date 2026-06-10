from __future__ import print_function

import argparse
import math
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.optim as optim
import pprint
from datetime import datetime
from torch.utils.data import DataLoader, WeightedRandomSampler
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
from ndcl_loss_imbalanced import SingleDomainImbalancedNDCL
from util import AverageMeter
from util import warmup_learning_rate
from util import save_model
from networks.cct_retnet import CCTRetNet


classes = [
    'Normal', 'Combined', 'DoS', 'Fuzzy', 'Gear',
    'Interval', 'RPM', 'Speed', 'Standstill', 'Systematic'
]


class EarlyStopping:
    def __init__(self, patience=20, delta=0.0, enabled=True):
        self.patience = patience
        self.delta = delta
        self.enabled = enabled
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.best_f1 = -np.inf
        self.early_stop = False
        self.val_loss_min = np.inf
        self.val_loss_min_epoch = 0
        self.val_loss_history = []

    def __call__(self, val_loss, val_f1, model, optimizer, opt, epoch, path=None):
        if path is None:
            path = opt.save_folder

        self.val_loss_history.append((epoch, val_loss))
        if val_loss < self.val_loss_min:
            self.val_loss_min = val_loss
            self.val_loss_min_epoch = epoch

        score = val_f1
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, val_f1, model, optimizer, opt, epoch, path)
            return True

        if score <= self.best_score + self.delta:
            self.counter += 1
            if self.enabled and self.counter >= self.patience:
                self.early_stop = True
            return False

        self.best_score = score
        self.save_checkpoint(val_loss, val_f1, model, optimizer, opt, epoch, path)
        self.counter = 0
        return True

    def save_checkpoint(self, val_loss, val_f1, model, optimizer, opt, epoch, path):
        save_file = os.path.join(path, 'best.pth')
        save_model(model, optimizer, opt, epoch, save_file)
        self.val_loss_min = val_loss
        self.best_f1 = val_f1
        self.best_epoch = epoch


def parse_option():
    parser = argparse.ArgumentParser('argument for imbalanced NDCL training')

    parser.add_argument('--print_freq', type=int, default=100)
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--resume', type=str, default=None)

    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4)
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)

    # dataset/model
    parser.add_argument('--model', type=str, default='CCT-RetNet-NDCL-Imbalanced')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dataset', type=str, default='CAN-ML', choices=['CAN-ML', 'CAN', 'ROAD'])
    parser.add_argument('--data_folder', type=str, default=None)
    parser.add_argument('--n_classes', type=int, default=10)

    # imbalanced training
    parser.add_argument('--use_weighted_sampler', type=int, default=1,
                        help='1: use inverse-frequency WeightedRandomSampler for train set')
    parser.add_argument('--sampler_power', type=float, default=1.0,
                        help='sample weight = 1 / count^sampler_power')
    parser.add_argument('--early_stop_patience', type=int, default=20)

    # NDCL loss
    parser.add_argument('--ndcl_alpha', type=float, default=0.1)
    parser.add_argument('--ndcl_beta_mix', type=float, default=0.5)
    parser.add_argument('--ndcl_min_aug_per_class', type=int, default=4)
    parser.add_argument('--ndcl_max_aug_per_class', type=int, default=32)
    parser.add_argument('--ndcl_majority_aug_ratio', type=float, default=0.0,
                        help='0.0 means no synthetic hard negatives for the majority class')
    parser.add_argument('--ndcl_inv_aug_power', type=float, default=0.5)
    parser.add_argument('--ndcl_low_pos_frac', type=float, default=0.25)
    parser.add_argument('--ndcl_hard_neg_frac', type=float, default=0.25)
    parser.add_argument('--ndcl_cb_beta', type=float, default=0.9999)
    parser.add_argument('--ndcl_warmup_ce_epochs', type=int, default=5)

    # other setting
    parser.add_argument('--method', type=str, default='NDCL_Imbalanced')
    parser.add_argument('--cosine', action='store_true')
    parser.add_argument('--warm', action='store_true')
    parser.add_argument('--trial', type=str, default='0')
    parser.add_argument('--seed', type=int, default=1)

    opt = parser.parse_args()

    opt.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if opt.data_folder is None:
        opt.data_folder = './data/size_64_10/TFRecord_w64_s32/3/'

    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)
    opt.lr_decay_epochs = [int(it) for it in opt.lr_decay_epochs.split(',') if it.strip()]

    current_time = datetime.now().strftime("%m%d%y_%H%M%S")
    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_epochs_{}'.format(
        opt.method, opt.dataset, opt.model, opt.learning_rate,
        opt.weight_decay, opt.batch_size, opt.trial, opt.epochs
    )
    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)
            ) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, current_time)
    os.makedirs(opt.save_folder, exist_ok=True)
    opt.log_file = os.path.join(opt.save_folder, 'log')
    opt.tb_folder = os.path.join(opt.save_folder, 'runs')
    return opt


def _extract_label(sample):
    # CANDataset usually returns (image, label). This helper keeps the counting
    # code robust if the dataset later returns extra fields.
    label = sample[1]
    if torch.is_tensor(label):
        label = label.item()
    return int(label)


def collect_labels_and_counts(dataset, num_classes=10):
    labels = []
    counts = torch.zeros(num_classes, dtype=torch.long)

    for idx in tqdm(range(len(dataset)), desc='Reading train labels'):
        label = _extract_label(dataset[idx])
        if label < 0 or label >= num_classes:
            raise ValueError(f'Invalid label {label} at index {idx}')
        labels.append(label)
        counts[label] += 1

    return torch.tensor(labels, dtype=torch.long), counts


def build_weighted_sampler(labels, class_counts, sampler_power=1.0):
    counts = class_counts.float().clamp_min(1.0)
    class_weight = 1.0 / torch.pow(counts, sampler_power)
    sample_weight = class_weight[labels]
    return WeightedRandomSampler(
        weights=sample_weight.double(),
        num_samples=len(sample_weight),
        replacement=True,
    )


def set_loader(opt):
    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True, is_test=False)
    test_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=False, is_test=False, include_data=False)

    train_labels, class_counts = collect_labels_and_counts(train_dataset, num_classes=opt.n_classes)

    if opt.use_weighted_sampler:
        sampler = build_weighted_sampler(train_labels, class_counts, sampler_power=opt.sampler_power)
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=opt.num_workers,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=opt.batch_size,
            shuffle=True,
            num_workers=opt.num_workers,
            pin_memory=True,
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.num_workers,
        pin_memory=True,
    )

    print(f'Train dataset size: {len(train_dataset)}')
    print(f'Validation dataset size: {len(test_dataset)}')
    print(f'Train class counts: {class_counts.tolist()}')
    print(f'Majority class: {int(torch.argmax(class_counts))} ({classes[int(torch.argmax(class_counts))] if opt.n_classes == len(classes) else ""})')
    print(f'Minority class: {int(torch.argmin(class_counts))} ({classes[int(torch.argmin(class_counts))] if opt.n_classes == len(classes) else ""})')

    return train_loader, test_loader, class_counts


def set_model(opt, class_counts):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model = CCTRetNet(
        img_size=64,
        embedding_dim=128,
        n_conv_layers=2,
        kernel_size=3,
        stride=1,
        padding=1,
        pooling_kernel_size=2,
        pooling_stride=2,
        pooling_padding=0,
        num_layers=2,
        num_heads=2,
        mlp_ratio=2.5,
        num_classes=opt.n_classes,
        positional_embedding='learnable',
    )

    criterion = SingleDomainImbalancedNDCL(
        num_classes=opt.n_classes,
        class_counts=class_counts,
        alpha=opt.ndcl_alpha,
        beta_mix=opt.ndcl_beta_mix,
        min_aug_per_class=opt.ndcl_min_aug_per_class,
        max_aug_per_class=opt.ndcl_max_aug_per_class,
        majority_aug_ratio=opt.ndcl_majority_aug_ratio,
        inv_aug_power=opt.ndcl_inv_aug_power,
        low_pos_frac=opt.ndcl_low_pos_frac,
        hard_neg_frac=opt.ndcl_hard_neg_frac,
        cb_beta=opt.ndcl_cb_beta,
        warmup_ce_epochs=opt.ndcl_warmup_ce_epochs,
    )
    criterion_validate = torch.nn.CrossEntropyLoss()

    model = model.to(opt.device)
    criterion = criterion.to(opt.device)
    criterion_validate = criterion_validate.to(opt.device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Model information here: ', model)
    print('Model device: ', next(model.parameters()).device)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'NDCL class weights: {criterion.class_weights.detach().cpu().tolist()}')

    return model, criterion, criterion_validate


optimize_dict = {
    'SGD': optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW,
}


def set_optimizer(opt, model, optim_choice='SGD'):
    optimizer_cls = optimize_dict[optim_choice]
    if optim_choice in ['Adam', 'AdamW']:
        return optimizer_cls(model.parameters(), lr=opt.learning_rate, weight_decay=opt.weight_decay)
    return optimizer_cls(
        model.parameters(),
        lr=opt.learning_rate,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay,
    )


def unpack_logits(model_output):
    # Handles both logits-only models and models returning (features, logits).
    if isinstance(model_output, (tuple, list)):
        for item in reversed(model_output):
            if torch.is_tensor(item) and item.dim() == 2:
                return item
        raise ValueError('Could not find a [B, C] logits tensor in model output tuple/list.')
    return model_output


def train(train_loader, model, criterion, optimizer, epoch, opt, step, logger=None):
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    ce_losses = AverageMeter()
    con_losses = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True).long()
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        logits = unpack_logits(model(images))
        loss_dict = criterion(logits=logits, labels=labels, epoch=epoch, return_dict=True)
        loss = loss_dict['loss']

        losses.update(loss.item(), bsz)
        ce_losses.update(float(loss_dict['loss_ce'].item()), bsz)
        con_losses.update(float(loss_dict['loss_con'].item()), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if logger is not None:
            logger.add_scalar('train/loss_total_iter', loss.item(), step)
            logger.add_scalar('train/loss_ce_iter', loss_dict['loss_ce'].item(), step)
            logger.add_scalar('train/loss_con_iter', loss_dict['loss_con'].item(), step)
            logger.add_scalar('train/aug_total_iter', loss_dict['aug_total'].item(), step)

        batch_time.update(time.time() - end)
        end = time.time()

        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'loss {loss.val:.5f} ({loss.avg:.5f})\t'
                'ce {ce.val:.5f} ({ce.avg:.5f})\t'
                'con {con.val:.5f} ({con.avg:.5f})\n'.format(
                    epoch, idx + 1, len(train_loader),
                    batch_time=batch_time, data_time=data_time,
                    loss=losses, ce=ce_losses, con=con_losses
                )
            )
            print(log_message)
            sys.stdout.flush()

    return step, losses.avg, ce_losses.avg, con_losses.avg


def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    return pred.t().detach().cpu().numpy().squeeze(0)


def validate(val_loader, model, criterion, opt):
    model.eval()

    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int)

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validate'):
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True).long()
            logits = unpack_logits(model(images))
            bsz = labels.size(0)
            loss = criterion(logits, labels)
            losses.update(loss.item(), bsz)

            pred = get_predict(logits)
            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels.detach().cpu().numpy()), axis=0)

    metrics = {
        'accuracy': accuracy_score(total_label, total_pred) * 100.0,
        'balanced_accuracy': balanced_accuracy_score(total_label, total_pred) * 100.0,
        'f1_macro': f1_score(total_label, total_pred, average='macro', zero_division=0),
        'f1_weighted': f1_score(total_label, total_pred, average='weighted', zero_division=0),
        'precision_macro': precision_score(total_label, total_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(total_label, total_pred, average='macro', zero_division=0),
        'per_class_f1': f1_score(total_label, total_pred, average=None, zero_division=0).tolist(),
    }
    conf_matrix = confusion_matrix(total_label, total_pred, labels=list(range(opt.n_classes)))
    return losses.avg, metrics, conf_matrix


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


def main():
    opt = parse_option()
    print(opt)

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    train_loader, test_loader, class_counts = set_loader(opt)
    model, criterion, criterion_validate = set_model(opt, class_counts)
    optimizer = set_optimizer(opt, model, optim_choice=opt.optimizer)

    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)
    log_writer = open(opt.log_file, 'w')
    pp = pprint.PrettyPrinter(indent=4)

    log_writer.write(f'Params information:\n{pp.pformat(vars(opt))}\n')
    log_writer.write(f'Train class counts: {class_counts.tolist()}\n')
    log_writer.write(f'Class names: {classes}\n')

    early_stopping = EarlyStopping(patience=opt.early_stop_patience, delta=0.0, enabled=True)
    start_epoch = 1
    final_epoch = start_epoch - 1
    step = 0

    for epoch in range(start_epoch, opt.epochs + 1):
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        final_epoch = epoch
        adjust_learning_rate(opt, optimizer, epoch)

        step, train_loss, train_ce, train_con = train(
            train_loader, model, criterion, optimizer, epoch, opt, step, logger=logger
        )

        val_loss, metrics, conf_matrix = validate(test_loader, model, criterion_validate, opt)

        logger.add_scalar('train/loss_total_epoch', train_loss, epoch)
        logger.add_scalar('train/loss_ce_epoch', train_ce, epoch)
        logger.add_scalar('train/loss_con_epoch', train_con, epoch)
        logger.add_scalar('val/loss_ce', val_loss, epoch)
        logger.add_scalar('val/accuracy', metrics['accuracy'], epoch)
        logger.add_scalar('val/balanced_accuracy', metrics['balanced_accuracy'], epoch)
        logger.add_scalar('val/f1_macro', metrics['f1_macro'], epoch)
        logger.add_scalar('val/f1_weighted', metrics['f1_weighted'], epoch)

        log_writer.write(
            f'Epoch: {epoch}, train_loss: {train_loss:.4f}, '
            f'train_ce: {train_ce:.4f}, train_con: {train_con:.4f}, '
            f'val_loss: {val_loss:.4f}\n'
        )
        log_writer.write(f'Metrics:\n{pp.pformat(metrics)}\n')
        log_writer.write(f'Confusion Matrix:\n{pp.pformat(conf_matrix)}\n')
        log_writer.flush()

        print(f'Epoch {epoch}, train loss {train_loss:.4f}, CE {train_ce:.4f}, Con {train_con:.4f}')
        print(f'Validation loss: {val_loss:.4f}')
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        print(f"Macro F1: {metrics['f1_macro']:.4f}")
        print(f"Weighted F1: {metrics['f1_weighted']:.4f}")
        print('Per-class F1:', metrics['per_class_f1'])
        print('Confusion Matrix:')
        print(conf_matrix)

        improved = early_stopping(
            val_loss,
            metrics['f1_macro'],
            model,
            optimizer,
            opt,
            epoch,
            opt.save_folder,
        )
        if improved:
            print(f"Best checkpoint updated: macro F1 {metrics['f1_macro']:.4f} at epoch {epoch}")

        if epoch % opt.save_freq == 0:
            save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, save_file)

        if early_stopping.early_stop:
            print(
                f'Early stopping at epoch {epoch}. '
                f'Best macro F1 {early_stopping.best_f1:.4f} at epoch {early_stopping.best_epoch}.'
            )
            log_writer.write(
                f'Early stopping at epoch {epoch}. '
                f'Best macro F1 {early_stopping.best_f1:.4f} at epoch {early_stopping.best_epoch}.\n'
            )
            break

    save_file = os.path.join(opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, final_epoch, save_file)
    log_writer.close()
    logger.close()


if __name__ == '__main__':
    main()
