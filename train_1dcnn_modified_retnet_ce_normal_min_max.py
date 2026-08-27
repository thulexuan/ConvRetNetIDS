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
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import pprint
from datetime import datetime
from dataset_backup_with_no_normalize import CANMLDataset
from model import CANIDSConfig, CANIntrusionDetector

from models.time_aware_retnet import TimeAwareRetNetCANIDS
from networks.cnn_1d_modified_retnet import CNN1DModifiedRetNet
from supcon.util import accuracy
from util import AverageMeter
from util import warmup_learning_rate
from util import save_model
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter

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
    parser.add_argument('--learning_rate_classifier', type=float, default=3e-4,
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
    parser.add_argument('--model', type=str, default='CCT-RetNet-GPaCo')
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
    parser.add_argument('--method', type=str, default='GPaCo+sequence-input', 
                        help='choose method')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--early_stop_patience', type=int, default=20,
                        help='patience for early stopping')
    # model parameters 
    parser.add_argument('--embedding_dim', type=int, default=40,
                        help='dimension of the embedding layer')
    parser.add_argument('--projection_dim', type=int, default=64,
                        help='dimension of the projection layer to train feature space')
    parser.add_argument('--n_input_channels', default=9, type=int,
                    help='the number of input channels')
    parser.add_argument('--kernel_size', default=3, type=int,
                    help='the size of the kernel in the tokenizer')

    # input sliding window parameters 
    parser.add_argument('--window_size', type=int, default=64,
                        help='size of the sliding window')
    parser.add_argument('--stride_size', type=int, default=64,
                        help='size of the stride for the sliding window')

    # settings for RetNet config
    parser.add_argument('--n_layers', default=2, type=int,
                    help='the number of RetNet blocks')
    parser.add_argument('--n_heads', default=2, type=int,
                    help='the number of RetNet retention layers in each block')
    parser.add_argument('--n_conv_layers', default=2, type=int,
                    help='the number of convolutional layers in tokenizer')
    parser.add_argument('--positional_embedding', type=str, default='none',
                        choices=['sine', 'learnable', 'none', 'sine_2d', 'learnable_2d'],
                        help='type of positional embedding')

    # GPaCo loss configuration
    parser.add_argument('--alpha', default=0.05, type=float,
                    help='contrast weight among samples')
    parser.add_argument('--temp', default=0.07, type=float,
                    help='temperature for contrastive learning')
    

    opt = parser.parse_args()

    opt.device = torch.device('cuda:0')
    # set the path according to the environment
    if opt.data_folder is None:
        # opt.data_folder = './data/size_64_10/TFRecord_w64_s32/3/'
        opt.data_folder = './data/can-ml/can-ml/2017-subaru-forester/090726/preprocessed/rid2/TFRecord_w16_s1/'
    opt.model_path = './save/{}_models/{}'.format(opt.dataset, opt.method)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    current_time = datetime.now().strftime("%D_%H%M%S").replace('/', '')
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

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
    opt.log_file = os.path.join(opt.save_folder, 'log')
    opt.tb_folder = os.path.join(opt.save_folder, 'runs')
    return opt

def calculate_class_counts(dataset, num_classes=10):
        class_counts = torch.zeros(num_classes, dtype=torch.long)

        for idx in tqdm(range(len(dataset)), desc="Counting class labels"):
            _, label = dataset[idx]

            if torch.is_tensor(label):
                label = label.item()

            label = int(label)

            if label < 0 or label >= num_classes:
                raise ValueError(f"Invalid label {label} at index {idx}")

            class_counts[label] += 1

        return class_counts

def set_loader(opt):

    train_dataset = CANMLDataset(
        root_dir=opt.data_folder, window_size=opt.window_size, stride_size=opt.stride_size, is_train=True, is_test=False)
    
    test_dataset = CANMLDataset(
        root_dir=opt.data_folder, window_size=opt.window_size, stride_size=opt.stride_size, is_train=False, is_test=True)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batch_size, shuffle=False, pin_memory=True)
    train_classifier_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True)

    print('Number of training samples: {}'.format(len(train_dataset)))
    print('Number of testing samples: {}'.format(len(test_dataset)))
    # torch.set_printoptions(
    #     precision=4,
    #     threshold=float("inf"), # no truncation of tensor elements
    #     linewidth=200,
    #     sci_mode=False   
    # )
   
    return train_loader, test_loader, train_classifier_loader

def set_model(opt):
    torch.cuda.empty_cache()
    
    # model = CNN1DModifiedRetNet(
    #     window_size = opt.window_size,
    #     stride_size = opt.stride_size,
    #     # mlp_ratio = 2.5, # dim feedforward layer = embedding_dim * mlp_ratio, current case = 128 * 2.5 = 320
    #     num_classes = opt.n_classes,
    #     # projection_dim=opt.projection_dim,
    #     n_layers=opt.n_layers,
    #     # n_heads=opt.n_heads
    # )
    # ---- standard
    cfg = CANIDSConfig(num_classes=10, window_size=opt.window_size)
    print(cfg)
    model = CANIntrusionDetector(cfg)

    # ------ spatial per frame
    # model = SpatialFrameEncoderRetNetCANIDS(
    #         num_features=opt.n_input_channels,
    #         linear1_dim=32,
    #         conv_out_channels=32,
    #         kernel_size=3,
    #         d_model=128,
    #         retnet_layers=3,
    #         retnet_heads=4,
    #         ffn_dim=256,
    #         num_classes=opt.n_classes,
    #         dropout=0.1,
    #     )
    # ----- simple conv1D cho spatial
    # model = SimpleConv1DRetNetCANIDS(
    #         num_features=opt.n_input_channels,
    #         d_model=128,          # channel-expanding, giống RAD (256 > 9)
    #         conv_kernel_size=3,
    #         conv_padding=1,
    #         conv_stride=1,
    #         causal_conv=False,     # đúng thiết kế gốc: padding đối xứng
    #         retnet_layers=2,        # RAD dùng 1 layer cho SWaT/MSL/SWAN
    #         retnet_heads=4,
    #         ffn_dim=256,
    #         num_classes=opt.n_classes,
    #         dropout=0.1,
    #     )

    # --- simple conv1D + retnet + cam
    # model = SimpleConv1DRetNetCAM(
    #         num_features=opt.n_input_channels,
    #         d_model=128,
    #         conv_kernel_size=3,
    #         conv_padding=1,
    #         conv_stride=1,
    #         causal_conv=False,
    #         retnet_layers=2,
    #         retnet_heads=4,
    #         ffn_dim=256,
    #         cam_reduction=8,
    #         num_classes=opt.n_classes,
    #         dropout=0.1,
    #     )

    # ---- STDLM original
    # cfg_bin = STDLMConfig(num_classes=opt.n_classes, window_size=opt.window_size)
    # model = STDLM(cfg_bin)

    # --- only ModernTCN
    # model = ModernTCNClassifier(
    #     num_features=opt.n_input_channels,
    #     window_size=opt.window_size,
    #     num_classes=opt.n_classes,          # tùy số lớp tấn công bạn có
    #     d_model=128,             # nhỏ vì window ngắn, tránh overfit
    #     num_layers=3,
    #     ffn_ratio=2,
    #     patch_size=1,           # giữ chi tiết tối đa (không patch, giống stem thuần)
    #     stride=1,
    #     large_kernel=15,        # ~ nửa window_size, đủ bao phủ phần lớn context
    #     small_kernel=3,
    #     dropout=0.1,
    # )
    # --- ModernTCN + RetNet
    # model = ModernTCNRetNet(
    #     num_features=opt.n_input_channels,
    #     window_size=opt.window_size,
    #     num_classes=opt.n_classes,
    #     d_model=128,
    #     num_layers=2,
    #     ffn_ratio=2,
    #     patch_size=1,
    #     stride=1,
    #     large_kernel=15,
    #     small_kernel=3,
    #     retention_heads=4,
    #     fusion='concat',
    #     dropout=0.1,
    # )
    # 05-08-2026: can_id_payload_conv1d_retnet.py: CANIDPayloadRetNetIDS
    # cfg = CANIDPayloadConfig(num_classes=10, window_size=opt.window_size)
    # model = CANIDPayloadRetNetIDS(cfg)

    # model = RetNetCANIDS(
    #         num_features=opt.n_input_channels,
    #         d_model=128,
    #         num_layers=3,
    #         num_heads=4,
    #         ffn_dim=256,
    #         num_classes=opt.n_classes,
    #         dropout=0.1,
    #     )

    # model = TimeAwareRetNetCANIDS(
    #             num_features=11,
    #             d_model=128,
    #             num_layers=2,
    #             num_heads=4,
    #             ffn_dim=256,
    #             num_classes=10,
    #             dropout=0.0,
    #             min_time_scale=0.001,
    #             max_time_scale=0.5,
    #         )

    # criterion_model = PaCoLoss(alpha=opt.alpha, temperature=opt.temp)
    criterion_model = torch.nn.CrossEntropyLoss()

    criterion_ce = torch.nn.CrossEntropyLoss()

    model = model.to(opt.device)
    criterion_model = criterion_model.to(opt.device)
    criterion_ce = criterion_ce.to(opt.device)

    print('Model device: ', next(model.parameters()).device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # print information about the model
    print('Model information (MC1DCNN-RetNet): ', model)
    print('Criterion for model (CE Loss): ', criterion_model)
    print('Criterion for classifier (CrossEntropyLoss): ', criterion_ce)

    print('Model device: ', next(model.parameters()).device)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')

    return model, criterion_model, criterion_ce

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

def train(train_loader, model, criterion_model, criterion_ce, optimizer, epoch, opt, step):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    end = time.time()

    for idx, (images, labels) in enumerate(train_loader):
        step += 1
        data_time.update(time.time() - end)
        
        # images[0] = images[0].to(opt.device, non_blocking=True) # images shape: (bsz, sequence_length, embedding_dim) ~ (128, 64*64, 256)
        # images[1] = images[1].to(opt.device, non_blocking=True) # images shape: (bsz, sequence_length, embedding_dim) ~ (128, 64*64, 256)
        images = images.to(opt.device, non_blocking=True) # (128, 16, 9)
        labels = labels.to(opt.device, non_blocking=True) # labels shape: (bsz)~ (128)
        bsz = labels.shape[0]

        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # print('Learning rate at epoch {}, step {}: {}'.format(epoch, step, optimizer.param_groups[0]['lr']))

        # output_before_proj, _ = model(images) # (128, 144)
        # logits = model.fc(output_before_proj) # (128, 144) -> (128, 10)

        # code 270726 
        feats, logits = model(images) # (128, 32, 9) -> (128, 10)
        loss = criterion_model(logits, labels)
        
        losses.update(loss.item(), bsz)
  
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
        if (idx + 1) % opt.print_freq == 0:
            log_message = (
                'Train: [{0}][{1}/{2}]\t'
                'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'DT {data_time.val:.5f} ({data_time.avg:.5f})\t'
                'loss {loss.val:.5f} ({loss.avg:.5f})\n'.format(
                    epoch, idx + 1, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses
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
            images[0] = images[0].to(opt.device, non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        
        with torch.no_grad():
            features = model.encode(images[0]) # (128, 128)
            
        output = classifier(features.detach()) #(128, 10)
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

def validate(val_loader, model, criterion, opt):
    model.eval()
    # classifier.eval()
    
    losses = AverageMeter()
    total_pred = np.array([], dtype=int)
    total_label = np.array([], dtype=int) 
    
    with torch.no_grad():
        for images, labels in tqdm(val_loader): 
            images = images.to(opt.device, non_blocking=True)
            labels = labels.to(opt.device, non_blocking=True)
            # output_feature_before_proj, _ = model(images) # (128, 128)
            # outputs = model.fc(output_feature_before_proj) # (128, 128) -> (128, 10)
            # code 270726
            feats, outputs = model(images) # (128, 9) -> (128, 10)
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
    macro_f1 = f1_score(total_label, total_pred, average='macro')
    precision = precision_score(total_label, total_pred, average='weighted', zero_division=0)
    recall = recall_score(total_label, total_pred, average='weighted')
    conf_matrix = confusion_matrix(total_label, total_pred)
    accuracy = accuracy_score(total_label, total_pred)
    acc_percent = accuracy * 100 
    return losses.avg, f1, macro_f1, precision, recall, conf_matrix, acc_percent


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
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # build data loader
    train_loader, test_loader, train_classifier_loader = set_loader(opt)
    model, criterion_model, criterion_ce = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model, optim_choice=opt.optimizer)
    # classifier_optimizer = set_optimizer(opt, classifier, optim_choice=opt.classifier_optimizer)
    
    logger = SummaryWriter(log_dir=opt.tb_folder, flush_secs=2)

    start_epoch = 1  # Default start from epoch 1
    step = 0

    log_writer = open(opt.log_file, 'w')
    # training routine
    pp = pprint.PrettyPrinter(indent=4)

    total_params = sum(p.numel() for p in model.parameters())
    log_writer.write(f'Total parameters: {total_params:,}\n')
    log_writer.write(f'Print parametes: {str(opt)}\n')
    log_writer.write(f'Model information: {model}\n')

    train_classification_losses = []
    train_supcon_losses = []
    val_losses = []

    for epoch in range(start_epoch, opt.epochs + 1):
        print('Begin time: ' + str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
        adjust_learning_rate(opt, optimizer, epoch)

        # train model with CE loss
        new_step, train_loss = train(train_loader, model, criterion_model, criterion_ce, optimizer, epoch, opt, step)

        print(f'Epoch {epoch}, 1DCNN-RetNet-CE Loss {train_loss:.4f}')
        log_writer.write(f'Epoch: {epoch}, 1DCNN-RetNet-CE Loss: {train_loss:.4f}\n')
        logger.add_scalar('train_loss/val', train_loss, step)
        train_supcon_losses.append(train_loss)
        print('Train Losses:', train_supcon_losses)

        # train and validate classifier with cross-entropy loss
        # class_epoch = epoch - opt.epoch_start_classifier + 1
        # if class_epoch > 0:
        #     adjust_learning_rate(opt, classifier_optimizer, class_epoch, class_str='_classifier')
        #     new_step, classifier_loss, classifier_acc = train_classifier(train_classifier_loader, model, classifier, criterion_classifier, classifier_optimizer, class_epoch, opt, new_step, logger)
        #     train_classification_losses.append(classifier_loss)
        #     print('Classifier: Loss: {:.4f}, Acc: {}'.format(classifier_loss, classifier_acc))
        #     log_writer.write('Classifier: Loss: {:.4f}, Acc: {}\n'.format(classifier_loss, classifier_acc))
        #     logger.add_scalar('classifier_loss/val', classifier_loss, step)
        #     logger.add_scalar('classifier_acc/val', classifier_acc, step)

        loss, val_f1, macro_f1, precision, recall, conf_matrix, accuracy = validate(test_loader, model, criterion_ce, opt)
        logger.add_scalar('loss_ce/val', loss, step)
        logger.add_scalar('acc/val', accuracy, step)
        logger.add_scalar('val_f1/val', val_f1, step)
        logger.add_scalar('macro_f1/val', macro_f1, step)
        val_losses.append(loss)

        log_writer.write(f'Accuracy: {accuracy:.4f}\n')
        log_writer.write(f'F1 Score: {val_f1:.4f}\n')
        log_writer.write(f'Macro F1 Score: {macro_f1:.4f}\n')
        log_writer.write(f'Precision: {precision:.4f}\n')
        log_writer.write(f'Recall: {recall:.4f}\n')
        log_writer.write(f'Confusion Matrix:\n{pp.pformat(conf_matrix)}\n')

        # Print results
        print(f'Validation loss: {loss:.4f}')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'F1 Score: {val_f1:.4f}')
        print(f'Macro F1 Score: {macro_f1:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print('Confusion Matrix:')
        print(conf_matrix)
        # print('Train Classification Losses:', train_classification_losses)
        
        print('Validation Losses:', val_losses)

        step = new_step
        # Save checkpoint 
      
        if epoch % opt.save_freq == 0:
            model_save_file = os.path.join(opt.save_folder, f'ckpt_epoch_{epoch}.pth')
            save_model(model, optimizer, opt, epoch, model_save_file)


    # save last checkpoint after training is done
    model_save_file = os.path.join(opt.save_folder, f'last.pth')
    save_model(model, optimizer, opt, epoch, model_save_file)
    
if __name__ == '__main__':
    main()