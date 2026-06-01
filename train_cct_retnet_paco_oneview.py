from __future__ import print_function

import argparse
from collections import deque
import math
import os
import random
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms.functional as F
import pprint
from dataset import CANDatasetEnet as CANDataset
# from networks.vision_trans import CEViT
from datetime import datetime
from networks.resnet_big import LinearClassifier
from selfcon.losses import ConLoss
from supcon.util import accuracy
from util import AverageMeter
from util import warmup_learning_rate
from util import save_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from networks.cct_retnet import CCTRetNet


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    if value is None or value == '':
        return []
    return [int(v.strip()) for v in value.split(',') if v.strip()]


class OneViewPaCoLoss(nn.Module):
    """
    One-view PaCo loss for the case where each input sample has only one view
    and no augmented key view is available.

    Expected inputs:
        features:   Tensor [B + Q, D]
                    First B rows are current batch projected features q.
                    Remaining Q rows are queue features from previous batches.
        labels:     Tensor [B + Q]
                    Labels for current batch + queue features.
        sup_logits: Tensor [B, C]
                    Class-center/classifier logits of current batch.

    Positive targets for each anchor:
        1) its true class center via sup_logits
        2) same-class samples in the current batch/queue, excluding itself

    Negative targets:
        1) other class centers
        2) different-class samples in the current batch/queue
    """
    def __init__(self, temperature=0.07, base_temperature=None, alpha=0.05,
                 beta=1.0, gamma=1.0, supt=1.0, num_classes=10, eps=1e-12):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.alpha = alpha      # weight for sample-to-sample positives
        self.beta = beta        # weight for sample-to-class-center positive
        self.gamma = gamma      # weight for sample candidates in denominator
        self.supt = supt        # temperature for supervised/class-center logits
        self.num_classes = num_classes
        self.eps = eps

    def forward(self, features, labels, sup_logits):
        if features.dim() != 2:
            raise ValueError(f'features must be [B + Q, D], got {features.shape}')
        if labels.dim() != 1:
            labels = labels.view(-1)
        if sup_logits.dim() != 2:
            raise ValueError(f'sup_logits must be [B, C], got {sup_logits.shape}')

        device = features.device
        batch_size = sup_logits.shape[0]

        if labels.shape[0] != features.shape[0]:
            raise ValueError(
                f'labels length ({labels.shape[0]}) must match features rows ({features.shape[0]}).'
            )
        if sup_logits.shape[1] != self.num_classes:
            raise ValueError(
                f'sup_logits has {sup_logits.shape[1]} classes but num_classes={self.num_classes}.'
            )

        labels = labels.contiguous().view(-1, 1)

        # Sample-to-sample logits: anchors are current batch features only.
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature
        )

        # Add sample-to-class-center logits before sample contrast logits.
        # sup_logits are produced by the classifier head and serve as class-center similarity.
        anchor_dot_contrast = torch.cat(
            (sup_logits / self.supt, anchor_dot_contrast),
            dim=1
        )

        # Numerical stability.
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Positive mask for sample-to-sample contrast.
        sample_pos_mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # Remove self-contrast in the sample part.
        sample_logits_mask = torch.ones_like(sample_pos_mask).to(device)
        sample_logits_mask.scatter_(
            1,
            torch.arange(batch_size, device=device).view(-1, 1),
            0
        )
        sample_pos_mask = sample_pos_mask * sample_logits_mask

        # Positive mask for class centers: one-hot of the true class.
        center_pos_mask = nnF.one_hot(
            labels[:batch_size].view(-1).long(),
            num_classes=self.num_classes
        ).float().to(device)

        # Final positive mask: [class-center positives | sample positives]
        positive_mask = torch.cat(
            (center_pos_mask * self.beta, sample_pos_mask * self.alpha),
            dim=1
        )

        # Denominator mask: keep all class centers; remove self from sample candidates.
        logits_mask = torch.cat(
            (
                torch.ones(batch_size, self.num_classes, device=device),
                sample_logits_mask * self.gamma
            ),
            dim=1
        )

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + self.eps)

        mean_log_prob_pos = (positive_mask * log_prob).sum(1) / (positive_mask.sum(1) + self.eps)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        return loss.mean()


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
            self.save_checkpoint(val_loss,val_f1,  model, optimizer, opt, epoch, path)
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

    def save_checkpoint(self, val_loss,val_f1,  model, optimizer, opt, epoch, path):
        save_file = os.path.join(path, 'best.pth')
        save_model(model, optimizer, opt, epoch, save_file)
        self.val_loss_min = val_loss
        self.best_f1 = val_f1
        self.best_epoch = epoch

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=20,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='number of training epochs')
    parser.add_argument('--resume', type=str, default=None, 
                        help='path to the checkpoint to resume from')
    
    # optimization
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    # optimization classifier
    parser.add_argument('--epoch_start_classifier', type=int, default=150)
    parser.add_argument('--learning_rate_classifier', type=float, default=0.01,
                        help='learning rate classifier')
    parser.add_argument('--lr_decay_epochs_classifier', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate_classifier', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay_classifier', type=float, default=0,
                        help='weight decay_classifier')
    parser.add_argument('--momentum_classifier', type=float, default=0.9,
                        help='momentum_classifier')
    # model dataset
    parser.add_argument('--model', type=str, default='CCT-RetNet-OneViewPaCo')
    parser.add_argument('--classifier_optimizer', type=str, default='Adam')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dataset', type=str, default='CAN-ML',
                        choices=['CAN-ML', 'CAN', 'ROAD'], 
                        help='dataset')
    parser.add_argument('--mean', type=str, 
                        help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, 
                        help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, 
                        help='path to custom dataset')
    parser.add_argument('--n_classes', type=int, default=10, 
                        help='number of class')

    # method
    parser.add_argument('--method', type=str, default='OneViewPaCo', 
                        help='choose method')
    parser.add_argument('--selfcon_layers', type=str, default='1,2,3',
                        help='comma-separated 1-based RetNet hidden layers used as SelfCon auxiliary views')
    parser.add_argument('--selfcon_alpha', type=float, default=0.0,
                        help='weight for the auxiliary SelfCon term; 0 uses all views in one ConLoss')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for PaCo sample-to-sample contrast')
    parser.add_argument('--paco_alpha', type=float, default=0.05,
                        help='weight for same-class sample positives in one-view PaCo')
    parser.add_argument('--paco_beta', type=float, default=1.0,
                        help='weight for the correct class-center positive in one-view PaCo')
    parser.add_argument('--paco_gamma', type=float, default=1.0,
                        help='weight for sample candidates in the PaCo denominator')
    parser.add_argument('--paco_supt', type=float, default=1.0,
                        help='temperature divisor for class-center/classifier logits')
    parser.add_argument('--paco_queue_size', type=int, default=8192,
                        help='FIFO queue size for one-view PaCo; set 0 to disable queue')

    # PaCo
    parser.add_argument('--projection_dim', type=int, default=128)
    parser.add_argument('--queue_mode', type=str, default='class_balanced', choices=['class_balanced', 'global', 'none'])
    parser.add_argument('--center_rebalance', dest='center_rebalance', action='store_true')
    parser.add_argument('--no_center_rebalance', dest='center_rebalance', action='store_false')
    parser.set_defaults(center_rebalance=True)
    parser.add_argument('--ce_weight', type=float, default=0.0, help='optional CE term added to PaCo loss')

    # imbalance
    parser.add_argument('--sampler', type=str, default='weighted', choices=['weighted', 'shuffle'])
    parser.add_argument('--use_weighted_ce', action='store_true', help='use class-weighted CE when ce_weight > 0')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='patience for early stopping')
    opt = parser.parse_args()

    opt.device = torch.device('cuda:0')
    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './data/size_64_10/TFRecord_w64_s32/3/'
    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    current_time = datetime.now().strftime("%D_%H%M%S").replace('/', '')
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.selfcon_layers = parse_int_list(opt.selfcon_layers)

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}_epochs_{}'. \
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial, opt.epochs)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)
    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name, current_time)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    # dir to save log file
    # opt.log_file = f'{opt.model_path}/{opt.model_name}/log'
    # opt.tb_folder = f'{opt.model_path}/{opt.model_name}/runs'
    opt.log_file = os.path.join(opt.save_folder, 'log')
    opt.tb_folder = os.path.join(opt.save_folder, 'runs')
    return opt

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

def set_loader(opt):
    train_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=True, is_test=False)
    test_dataset = CANDataset(root_dir=opt.data_folder, window_size=32, is_train=False, is_test=False, include_data=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)
    train_classifier_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    return train_loader, test_loader, train_classifier_loader


def set_model(opt):
    torch.cuda.empty_cache()
    
    model = CCTRetNet(
        img_size = 64,
        embedding_dim = 128, # the number of filters in the convolutional layers = the dimension of tokens before put into transformer encoder
        n_conv_layers = 2, # number of convolutional layers
        kernel_size = 3,
        stride = 1,
        padding = 1,
        pooling_kernel_size = 2,
        pooling_stride = 2,
        pooling_padding = 0,
        num_layers = 2, # the number of transformer encoder layers
        num_heads = 2, # the number of attention heads in each transformer encoder layer
        mlp_ratio = 2.5, # dim feedforward layer = embedding_dim * mlp_ratio, current case = 128 * 2.5 = 320
        num_classes = opt.n_classes,
        positional_embedding = 'learnable', # ['sine', 'learnable', 'none']
        projection_head = 'mlp',
        projection_dim = 128,
        selfcon_layers = opt.selfcon_layers,
        selfcon_projection_dim = 128,
        selfcon_head = 'mlp',
    )

    criterion_model = OneViewPaCoLoss(
        temperature=opt.temp,
        alpha=opt.paco_alpha,
        beta=opt.paco_beta,
        gamma=opt.paco_gamma,
        supt=opt.paco_supt,
        num_classes=opt.n_classes,
    )

    classifier = LinearClassifier(num_classes=opt.n_classes, feat_dim=128)
    criterion_classifier = torch.nn.CrossEntropyLoss()

    model = model.to(opt.device)
    criterion_model = criterion_model.to(opt.device)
    classifier = classifier.to(opt.device)
    criterion_classifier = criterion_classifier.to(opt.device)
    print('Model device: ', next(model.parameters()).device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print information about the model
    print('Model information (CCTRetNet): ', model)
    print('Criterion for model (OneViewPaCoLoss): ', criterion_model)
    print('Classifier information (LinearClassifier): ', classifier)
    print('Criterion for classifier (CrossEntropyLoss): ', criterion_classifier)

    print('Model device: ', next(model.parameters()).device)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    return model, criterion_model, classifier, criterion_classifier

optimize_dict = {
    'SGD' : optim.SGD,
    'RMSprop': optim.RMSprop,
    'Adam': optim.Adam,
    'AdamW': optim.AdamW
}

def set_optimizer(opt, model, class_str='', optim_choice='SGD'):
    dict_opt = vars(opt)
    optimizer = optimize_dict[optim_choice]

    # Handle both Adam and AdamW cases
    if optim_choice in ['Adam', 'AdamW']:
        optimizer = optimizer(
            model.parameters(),
            lr=dict_opt['learning_rate' + class_str],
            weight_decay=dict_opt['weight_decay' + class_str]
        )
    else:
        # For optimizers that require momentum like SGD and RMSprop
        optimizer = optimizer(
            model.parameters(),
            lr=dict_opt['learning_rate' + class_str],
            momentum=dict_opt['momentum' + class_str],
            weight_decay=dict_opt['weight_decay' + class_str]
        )

    return optimizer

def train(train_loader, model, classifier, queue_bank, criterion, optimizer, epoch, opt, step):
    """One epoch training with one-view PaCo loss."""
    model.train()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)

        images = images.to(opt.device, non_blocking=True)
        labels = labels.to(opt.device, non_blocking=True).long()
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # CCTRetNet returns the backbone feature and the projection-head feature.
        # final_feature: raw encoder feature, used for class-center/classifier logits.
        # final_output: projected feature, used for sample-to-sample contrast + queue.
        features = model.encode(images)
        projected_features = model.projection(features)

        q = nnF.normalize(projected_features, dim=1)
        sup_logits = classifier(features)

        if queue_bank is not None:
            queue_features, queue_labels = queue_bank.get()
            features_for_loss = torch.cat([q, queue_features], dim=0)
            labels_for_loss = torch.cat([labels, queue_labels], dim=0)
        else:
            features_for_loss = q
            labels_for_loss = labels

        loss = criterion(features_for_loss, labels_for_loss, sup_logits)

        losses.update(loss.item(), bsz)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update queue after the optimization step. The queued features are detached.
        if queue_bank is not None:
            queue_bank.enqueue(q.detach(), labels.detach())

        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train-PaCo: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'queue {queue_len}\t'
                'loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, queue_len=len(queue_bank), loss=losses
                )
            )
            print(log_message)
            sys.stdout.flush()

    return step, losses.avg

def train_classifier(train_loader, model, classifier, criterion, optimizer, epoch, opt, step, logger):
    model.eval()
    classifier.train()
    
    losses = AverageMeter()
    accs = AverageMeter()
    # here, criterion is cross-entropy
    
    for idx, (images, labels) in tqdm(enumerate(train_loader)):
        step += 1
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        with torch.no_grad():
            features = model.encode(images) # shape is (batch_size, embedding_dim)
            
        output = classifier(features.detach())
        loss = criterion(output, labels)
        losses.update(loss.item(), bsz)
        acc = accuracy(output, labels, topk=(1, ))
        accs.update(acc[0], bsz)
        
        if step % opt.print_freq == 0:
            logger.add_scalar('loss_ce/train', losses.avg, step)
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return step, losses.avg, accs.avg

def get_predict(outputs):
    _, pred = outputs.topk(1, 1, True, True)
    pred = pred.t().cpu().numpy().squeeze(0)
    return pred

def validate(val_loader, model, classifier, criterion, opt):
    model.eval()
    classifier.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader): 
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
            features = model.encode(images)
            outputs = classifier(features) # shape is (batch_size, num_classes) after go through classifier
            bsz = labels.size(0)
            loss = criterion(outputs, labels)
            losses.update(loss.item(), bsz)
            
            pred = get_predict(outputs)

            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu().numpy()

            total_pred = np.concatenate((total_pred, pred), axis=0)
            total_label = np.concatenate((total_label, labels), axis=0)
    
    f1 = f1_score(total_label, total_pred, average='weighted')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred)
    accuracy = accuracy_score(total_label, total_pred)
    acc_percent = accuracy * 100 
    return losses.avg, f1, precision, recall, conf_matrix, acc_percent


def adjust_learning_rate(args, optimizer, epoch, class_str=''):
    dict_args = vars(args)
    lr = dict_args['learning_rate'+class_str]
    if args.cosine:
        eta_min = lr * (dict_args['lr_decay_rate'+class_str] ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        # lr_decay_epochs_arr = list(map(int, dict_args['lr_decay_epochs'+class_str].split(',')))
        lr_decay_epochs = dict_args['lr_decay_epochs'+class_str]
        if isinstance(lr_decay_epochs, str):
            lr_decay_epochs_arr = list(map(int, lr_decay_epochs.split(',')))
        else:
            lr_decay_epochs_arr = list(map(int, lr_decay_epochs))
        lr_decay_epochs_arr = np.asarray(lr_decay_epochs_arr)
        steps = np.sum(epoch > lr_decay_epochs_arr)
        if steps > 0:
            lr = lr * (dict_args['lr_decay_rate'+class_str] ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    opt = parse_option()
    print(opt)

    seed = 1
    torch.manual_seed(seed)

    # build data loader
    train_loader, test_loader, train_classifier_loader = set_loader(opt)
    model, criterion_model, classifier, criterion_classifier = set_model(opt)

    # build optimizer
    # PaCo updates both the encoder/projection head and the classifier/class centers.
    optimizer = set_optimizer(opt, torch.nn.ModuleList([model, classifier]), optim_choice=opt.optimizer)
    classifier_optimizer = set_optimizer(opt, classifier, optim_choice=opt.classifier_optimizer)
    queue_bank = make_queue(opt)
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    start_epoch = 1  # Default start from epoch 1
    step = 0

    log_writer = open(opt.log_file, 'w')
    # training routine
    pp = pprint.PrettyPrinter(indent=4)

    total_params = sum(p.numel() for p in model.parameters())
    log_writer.write(f'Total parameters: {total_params:,}\n')

    early_stopping = EarlyStopping(patience=opt.early_stop_patience, delta=0.0, enabled=True)
    final_epoch = start_epoch - 1

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, opt.epochs + 1):
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        final_epoch = epoch
        adjust_learning_rate(opt, optimizer, epoch)

        # train encoder/projection/class centers with one-view PaCo loss
        new_step, train_loss = train(train_loader, model, classifier, queue_bank, criterion_model, optimizer, epoch, opt, step)

        print(f'Epoch {epoch}, CCT-RetNet-OneViewPaCo Loss {train_loss:.4f}')
        log_writer.write(f'Epoch: {epoch}, CCT-RetNet-OneViewPaCo Loss: {train_loss:.4f}\n')
        logger.add_scalar('train_loss/val', train_loss, step)
        train_losses.append(train_loss)

        # train and validate classifier with cross-entropy loss
        class_epoch = epoch - opt.epoch_start_classifier + 1
        if class_epoch > 0:
            adjust_learning_rate(opt, classifier_optimizer, class_epoch, class_str='_classifier')
            new_step, classifier_loss, classifier_acc = train_classifier(train_classifier_loader, model, classifier, criterion_classifier, classifier_optimizer, class_epoch, opt, new_step, logger)
            print('Classifier: Loss: {:.4f}, Acc: {}'.format(classifier_loss, classifier_acc))
            log_writer.write('Classifier: Loss: {:.4f}, Acc: {}\n'.format(classifier_loss, classifier_acc))
            logger.add_scalar('classifier_loss/val', classifier_loss, step)
            logger.add_scalar('classifier_acc/val', classifier_acc, step)

            loss, val_f1, precision, recall, conf_matrix, accuracy = validate(test_loader, model, classifier, criterion_classifier, opt)
            logger.add_scalar('loss_ce/val', loss, step)
            logger.add_scalar('acc/val', accuracy, step)
            logger.add_scalar('val_f1/val', val_f1, step)
            val_losses.append(loss)

            log_writer.write(f'Accuracy: {accuracy:.4f}\n')
            log_writer.write(f'F1 Score: {val_f1:.4f}\n')
            log_writer.write(f'Precision: {precision:.4f}\n')
            log_writer.write(f'Recall: {recall:.4f}\n')
            log_writer.write(f'Confusion Matrix:\n{pp.pformat(conf_matrix)}\n')

            # Print results
            print(f'Validation loss: {loss:.4f}')
            print(f'Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {val_f1:.4f}')
            print(f'Precision: {precision:.4f}')
            print(f'Recall: {recall:.4f}')
            print(f'Best validation F1: {early_stopping.best_f1:.4f} at epoch {early_stopping.best_epoch}')
            print(f'Min validation loss: {early_stopping.val_loss_min:.4f} at epoch {early_stopping.val_loss_min_epoch}')
            print('Confusion Matrix:')
            print(conf_matrix)
            print('Train Losses:', train_losses)
            print('Validation Losses:', val_losses)

       

            if (class_epoch % 10 == 0):
                print('Saving model at epoch {}...'.format(epoch))
                classifier_save_file = os.path.join(opt.save_folder, f'classifier_ckpt_epoch_{class_epoch}.pth')
                save_model(classifier, classifier_optimizer, opt, class_epoch, classifier_save_file)

        step = new_step
        # Save checkpoint 
      
        if epoch % opt.save_freq == 0:
            model_save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, model_save_file)

    # save last checkpoint after training is done
    model_save_file = os.path.join(opt.save_folder, f'last.pth')
    classifier_save_file = os.path.join(opt.save_folder, f'classifier_last.pth')
    save_model(model, optimizer, opt, epoch, model_save_file)
    final_class_epoch = max(1, final_epoch - opt.epoch_start_classifier + 1)
    save_model(classifier, classifier_optimizer, opt, final_class_epoch, classifier_save_file)
    
if __name__ == '__main__':
    main()
