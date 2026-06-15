import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# class PaCoLoss(nn.Module):
#     def __init__(self, alpha=0.05, beta=1.0, gamma=1.0, supt=1.0, temperature=0.07, base_temperature=None, K=8192, num_classes=10, smooth=0.0):
#         super(PaCoLoss, self).__init__()
#         self.temperature = temperature
#         self.beta = beta
#         self.gamma = gamma
#         self.supt = supt

#         self.base_temperature = temperature if base_temperature is None else base_temperature
#         self.K = K
#         self.alpha = alpha
#         self.num_classes = num_classes
#         self.smooth = smooth

#         self.weight = None

#     def cal_weight_for_classes(self, cls_num_list):
#         cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
#         self.weight = cls_num_list / cls_num_list.sum()
#         self.weight = self.weight.to(torch.device('cuda'))

#     def forward(self, features, labels=None, sup_logits=None):
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         # batch_size = features.shape[0] - self.K
#         batch_size = features.shape[0] 

#         labels = labels.contiguous().view(-1, 1)
#         mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

#         # compute logits
#         anchor_dot_contrast = torch.div(
#             torch.matmul(features[:batch_size], features.T),
#             self.temperature)

#         # add supervised logits
#         if self.weight is not None:
#             anchor_dot_contrast = torch.cat(( (sup_logits + torch.log(self.weight + 1e-9) ) / self.supt, anchor_dot_contrast), dim=1)
#         else:
#             anchor_dot_contrast = torch.cat(( (sup_logits) / self.supt, anchor_dot_contrast), dim=1)


#         # for numerical stability
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#         logits = anchor_dot_contrast - logits_max.detach()

#         # mask-out self-contrast cases
#         logits_mask = torch.scatter(
#             torch.ones_like(mask),
#             1,
#             torch.arange(batch_size).view(-1, 1).to(device),
#             0
#         )
#         mask = mask * logits_mask

#         # add ground truth 
#         one_hot_label = torch.nn.functional.one_hot(labels[:batch_size,].view(-1,), num_classes=self.num_classes).to(torch.float32)
#         one_hot_label = self.smooth / (self.num_classes - 1 ) * (1 - one_hot_label) + (1 - self.smooth) * one_hot_label
#         mask = torch.cat((one_hot_label * self.beta, mask * self.alpha), dim=1)

#         # compute log_prob
#         logits_mask = torch.cat((torch.ones(batch_size, self.num_classes).to(device), self.gamma * logits_mask), dim=1)
#         exp_logits = torch.exp(logits) * logits_mask
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
       
#         # compute mean of log-likelihood over positive
#         mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#         # loss
#         loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss = loss.mean()
#         return loss

class PaCoLoss(nn.Module):
    def __init__(
        self,
        alpha=0.05,
        beta=1.0,
        gamma=1.0,
        supt=1.0,
        temperature=0.07,
        base_temperature=None,
        K=8192,
        num_classes=10,
        smooth=0.0,
        center_weight_type='effective',
        center_effective_beta=0.999
    ):
        super(PaCoLoss, self).__init__()

        self.temperature = temperature
        self.beta = beta
        self.gamma = gamma
        self.supt = supt

        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.num_classes = num_classes
        self.smooth = smooth

        # Balanced Softmax prior
        self.weight = None

        # Class-dependent center weight
        self.center_weight = None
        self.center_weight_type = center_weight_type
        self.center_effective_beta = center_effective_beta

    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.tensor(
            cls_num_list,
            dtype=torch.float32
        ).view(1, self.num_classes)

        # For Balanced Softmax
        self.weight = cls_num_list / cls_num_list.sum()

        # For class-dependent center weight
        if self.center_weight_type == 'inverse':
            center_weight = cls_num_list.sum() / (
                self.num_classes * cls_num_list.clamp_min(1.0)
            )

        elif self.center_weight_type == 'effective':
            beta = self.center_effective_beta
            effective_num = (1.0 - torch.pow(beta, cls_num_list)) / (1.0 - beta)
            center_weight = effective_num.sum() / (
                self.num_classes * effective_num.clamp_min(1e-12)
            )

        else:
            raise ValueError(
                f"Unknown center_weight_type: {self.center_weight_type}"
            )

        # Normalize to keep average center weight around 1
        center_weight = center_weight / center_weight.mean()

        self.center_weight = center_weight.view(self.num_classes)

    def forward(self, features, labels=None, sup_logits=None):
        device = features.device

        # If you do not use queue
        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1).long()

        mask = torch.eq(
            labels[:batch_size],
            labels.T
        ).float().to(device)

        # sample-to-sample contrastive logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature
        )

        # supervised / class-center logits
        if self.weight is not None:
            class_prior = self.weight.to(device)
            sup_part = (sup_logits + torch.log(class_prior + 1e-9)) / self.supt
        else:
            sup_part = sup_logits / self.supt

        anchor_dot_contrast = torch.cat(
            (sup_part, anchor_dot_contrast),
            dim=1
        )

        # numerical stability
        logits_max, _ = torch.max(
            anchor_dot_contrast,
            dim=1,
            keepdim=True
        )
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # ground-truth class-center positive
        anchor_labels = labels[:batch_size].view(-1).long()

        one_hot_label = F.one_hot(
            anchor_labels,
            num_classes=self.num_classes
        ).to(torch.float32).to(device)

        one_hot_label = (
            self.smooth / (self.num_classes - 1) * (1 - one_hot_label)
            + (1 - self.smooth) * one_hot_label
        )

        # class-dependent center weight
        if self.center_weight is not None:
            center_weight = self.center_weight.to(device)          # [num_classes]
            beta_y = self.beta * center_weight[anchor_labels]      # [B]
            center_mask = one_hot_label * beta_y.view(-1, 1)       # [B, num_classes]
        else:
            center_mask = one_hot_label * self.beta

        # final positive mask:
        # center positives use class-dependent beta_y
        # sample positives use alpha
        mask = torch.cat(
            (center_mask, mask * self.alpha),
            dim=1
        )

        # denominator mask
        logits_mask = torch.cat(
            (
                torch.ones(batch_size, self.num_classes).to(device),
                self.gamma * logits_mask
            ),
            dim=1
        )

        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(
            exp_logits.sum(1, keepdim=True) + 1e-12
        )

        # average log-likelihood over positives
        mean_log_prob_pos = (mask * log_prob).sum(1) / (
            mask.sum(1) + 1e-12
        )

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss

class MultiTaskLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=1.0, gamma=1.0, supt=1.0, temperature=0.07, base_temperature=None, K=8192, num_classes=10):
        super(MultiTaskLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        rew = self.class_weight.squeeze()[labels[:batch_size].squeeze()]
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos * rew
        loss = loss.mean()

        loss_balancesoftmax = F.cross_entropy(sup_logits + torch.log(self.weight + 1e-9), labels[:batch_size].squeeze())
        return loss_balancesoftmax + self.alpha * loss


class MultiTaskBLoss(nn.Module):
    def __init__(self, alpha=0.05, beta=1.0, gamma=1.0, supt=1.0, temperature=0.07, base_temperature=None, K=8192, num_classes=10):
        super(MultiTaskBLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.supt = supt
        self.num_classes = num_classes
        self.effective_num_beta = 0.999


    def cal_weight_for_classes(self, cls_num_list):
        cls_num_list = torch.Tensor(cls_num_list).view(1, self.num_classes)
        self.weight = cls_num_list / cls_num_list.sum()
        self.weight = self.weight.to(torch.device('cuda'))

        if self.effective_num_beta != 0:
            effective_num = np.array(1.0 - np.power(self.effective_num_beta, cls_num_list)) / (1.0 - self.effective_num_beta)
            per_cls_weights = sum(effective_num) / len(effective_num) / effective_num
            self.class_weight  = torch.FloatTensor(per_cls_weights).to(torch.device('cuda'))

    def forward(self, features, labels=None, sup_logits=None, mask=None, epoch=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        batch_size = features.shape[0] - self.K
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        loss_ce = F.cross_entropy(sup_logits, labels[:batch_size].squeeze())
        return loss_ce + self.alpha * loss

