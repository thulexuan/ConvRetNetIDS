import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

import numpy as np

class NDCL(nn.Module):
    def __init__(self, num_classes, model, hparams):
        super(NDCL, self).__init__()
        self.n_class = num_classes
        self.model = model
        self.hparams = hparams

    @staticmethod
    def _margin_loss(pos, neg, pos_y, neg_y):
        loss = torch.tensor([0.], device=pos.device)
        for k in torch.unique(pos_y):
            num = torch.sum(pos_y == k)
            # extract instances from each class
            matrix = torch.cat([pos[pos_y == k], neg[neg_y == k]], dim=0)
            cos = 1 - torch.cosine_similarity(matrix.unsqueeze(1), matrix.unsqueeze(0), dim=-1)
            cos = torch.triu(cos, 1)[:, 1:] + torch.tril(cos, -1)[:, :-1]
            cos = F.softmax(cos[:num], dim=-1)

            # calc loss: [sum: frac => -log: con loss for each row => mean: mean con loss]
            loss += torch.mean(-torch.log(torch.sum(cos[:, num - 1:], dim=-1)))
        # end for k
        return loss / torch.unique(pos_y).size(0)

    def _aug2(self, z, y, pred): # x là input ảnh thật, z -> embeddings after backbone, y -> labels, pred -> prediction probs sau khi qua softmax
        def lamd(alpha):
            lam = np.random.beta(alpha, alpha)
            return max(lam, 0.8) 

        def obtain_num_from_pred(pred, k, thred):
            pos = torch.where(pred[:, k] >= thred)[0]
            return pos[-1].item() + 1 if pos.size(0) else 0

        if torch.unique(y).size(0) == 1:
            return None, None

        # setting
        alpha, total, num_gen = self.hparams['mixup'], z.size(0), {} # self.hparams['mixup'] = 0.5
        # total: tổng số sample trong batch
        mode = self.hparams['aug_mode'].lower() if 'aug_mode' in self.hparams.keys() else 'rev'
        assert mode in ['set', 'mean', 'rev']
        if mode != 'set':
            scaler = self.hparams['aug_scaler'] if 'aug_scaler' in self.hparams.keys() else 1.0
            scaler = 1.0

        # calc the number of data augmentation for each class
        if mode == 'set':  # tạo cùng 1 số lượng augmented negative samples cho mỗi class
            num = np.max([np.min([self.hparams['batch_size'] // 2, 24]), 8])
            num_gen = {k.item(): num for k in torch.unique(y)}
        elif mode == 'mean': # tạo số lượng augmented samples gần đều cho mỗi class, tổng số augmented sampples = total * scaler
            num = int(np.ceil(total * scaler / torch.unique(y).size(0)))
            redundant, classes, n_class = num * torch.unique(y).size(0) - int(total * scaler), [], []
            for k in torch.unique(y):
                classes.append(k), n_class.append(y[y == k].size(0))
            ind = np.argsort(n_class)
            for i in ind:
                num_gen[classes[i].item()] = num - 1 if redundant > 0 else num
                redundant -= 1
        else:
            total, classes, n_class = int(total * scaler), [], [] # total: tổng số augmented samples cần tạo
            # class nào có số lượng samples trong batch nhiều hơn thì tạo nhiều augmented negative samples hơn
            for k in torch.unique(y):
                classes.append(k), n_class.append(y[y == k].size(0) / y.size(0)) # tính tỷ lệ samples của từng class trong batch
            ratio, start, cum = np.exp(n_class) / np.exp(n_class).sum(), 0, 0
            for ind in range(len(classes)):
                cum += ratio[ind]
                num_gen[classes[ind].item()] = int(np.ceil(cum * total)) - start
                start += num_gen[classes[ind].item()]
        
        # print('print number of augmented negative samples of all classess (num_gen):', num_gen)
        # end calc

        # generate data for each class
        lam, device, aug, aug_y, n_class = lamd(alpha), z.device, [], [], pred.size(-1)
        for k in torch.unique(y):
            aug_pos, pos, neg, pred_pos, pred_neg = [], z[y == k], z[y != k], pred[y == k], pred[y != k]
            # pos: embeddings của các samples có label = k, neg: embeddings của các samples có label != k
            # pred_pos: prediction probs của các samples có label = k, pred_neg: prediction probs của các samples có label != k

            # chọn pos: chọn sample có label = k, prob = k xếp theo thứ tự tăng dần
            # chọn neg: chọn sample có label != k, prob = k xếp theo thứ tự giảm dần

            # asc sort neg though pred_neg[:, k]
            _, ind = torch.sort(pred_neg[:, k], descending=True)
            neg, pred_neg = neg[ind], pred_neg[ind]
            # dec sort pos though pred_pos[:, k]
            _, ind = torch.sort(pred_pos[:, k])
            pos, pred_pos = pos[ind], pred_pos[ind]
            # find neg mix instances (> 2/K)
            # tính số hard negative samples được tạo ra cho mỗi class
            # ví dụ với class = DoS, negative set có xác suất = {0.62, 0.4, 0.25, 0.12, 0.03} thì num = 3, large_num = 4
            num = obtain_num_from_pred(pred_neg, k, 2 / n_class)
            large_num = obtain_num_from_pred(pred_neg, k, 1 / n_class)
            if pos.size(0) == 1:
                # n_min: minimum amount of instances
                lam_1, n_sampling, n_min = lam, neg.size(0) // 2, int(np.ceil(num_gen[k.item()] / 3))
                # choose the number of sampling
                if n_min <= num:
                    n_sampling = np.min([num, n_sampling]) if n_min <= n_sampling else num
                elif n_min <= n_sampling:
                    # num < n_min <= n_sampling
                    n_sampling = n_sampling
                elif n_min <= large_num:
                    n_sampling = large_num
                else:
                    n_sampling = neg.size(0)
                # end if [choose]
                ending = 0
                while True:
                    # generate instances
                    gen = lam_1 * pos + (1 - lam_1) * neg[:n_sampling]
                    aug_pos.append(gen)
                    lam_1, ending = lamd(alpha), ending + n_sampling
                    if ending >= num_gen[k.item()]:
                        break
                # end while [generate]
            else:
                # amount of positive instances > 1
                n_sampling = 0
                for per in [0.25, 0.5, 0.75]:
                    # n_pos: number of positive instances to be mixed, is_diversity: whether to use diversity-enhanced mixup
                    n_pos, is_diversity = int(np.ceil(pos.size(0) * per)), False
                    if n_pos == 1:
                        continue
                    # number of sampling
                    if num_gen[k.item()] <= n_pos * num:
                        n_sampling = num
                    elif num_gen[k.item()] <= n_pos * large_num:
                        n_sampling = large_num
                    else:
                        is_diversity, n_neg = True, int(np.ceil(num_gen[k.item()] / n_pos))
                        if num > 0 and n_neg / num >= 0.25:
                            n_sampling = num
                        elif large_num > 0 and n_neg / large_num >= 0.25:
                            n_sampling = large_num
                    # end if num_gen
                    if n_sampling != 0:
                        break
                # end for per
                # adjust amount of sampling instances from neg
                n_sampling = n_sampling if n_sampling != 0 else \
                    (neg.size(0) if neg.size(0) < int(np.ceil(num_gen[k.item()] / n_pos)) else int(np.ceil(num_gen[k.item()] / n_pos)))
                ending, pos, neg = 0, pos[:n_pos], neg[:n_sampling]
                while True:
                    if ending != 0 and is_diversity:
                        lam_1 = lamd(alpha)
                        gen_neg = (lam_1 * neg[np.random.permutation(n_sampling)]
                                   + (1 - lam_1) * neg[np.random.permutation(n_sampling)])
                    else:
                        gen_neg = neg
                    # enf if [choose negative instances]
                    # generate instances
                    gen = []
                    for i in range(n_pos):
                        gen.append(lam * pos[i:i + 1] + (1 - lam) * gen_neg)
                    aug_pos.append(torch.cat(gen, dim=0))
                    ending += aug_pos[-1].size(0)
                    if ending >= num_gen[k.item()]:
                        break
                # end while [generate]
            # end if [whole generate phase]
            aug.append(torch.cat(aug_pos, dim=0)[:num_gen[k.item()]]), aug_y.append(k * torch.ones(num_gen[k.item()], device=device))
        # end for k
        return torch.cat(aug, dim=0), torch.cat(aug_y, dim=0)

    def forward(self, embeddings, labels, logits, unlabeled=None, opt=None):

        # print('print the number of unique labels:', torch.unique(labels).size(-1))
        # calc all predict results
        pred = logits # (128, 10), output của backbone, chưa qua softmax, còn là logits
        each_loss = F.cross_entropy(pred, labels, reduction='none') # (128,), cross entropy loss cho từng instance, chưa qua softmax để thành xác suất
        pred = F.softmax(pred, dim=-1) # đưa logits qua softmax -> tính được prediction probs, here pred (128, 10)

        cls_loss = torch.tensor([0.], device=opt.device) # 10 elements
        for k in torch.unique(labels): # for each label, có tất cả 10 labels
            li = each_loss[labels == k]
            # reweight = self.weights_per_env[all_d[labels == k], k]
            penalty = F.softmax(li.detach(), dim=-1)
            cls_loss += torch.sum(li * penalty)
        cls_loss /= torch.unique(labels).size(-1)

        # margin loss
        # margin_loss = torch.tensor([0.], device=device)
        aug, aug_y = self._aug2(embeddings, labels, pred)
        if aug is not None:
            aug_pred = F.softmax(self.model.classifier.get_logits_from_embeddings(aug), dim=-1)
            margin_loss = self._margin_loss(pred, aug_pred, labels, aug_y)
        else:
            margin_loss = torch.tensor([0.], device=opt.device)

        loss = cls_loss + self.hparams['alpha'] * margin_loss 

        # # optimize
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()

        return loss
