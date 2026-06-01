"""
One-view PaCo / MoCo-style builder.

This file is a drop-in replacement for:
    PaCo/Full-ImageNet/moco/builder.py

Main change from the original PaCo implementation:
    original: features = [q, k, queue]  # needs im_q and im_k / two augmented views
    one-view: features = [q, queue]     # only needs one input view

The queue is still kept because it provides many sample candidates for
sample-to-sample contrast. Class-center contrast is still provided through
`logits_q = self.linear(self.feat_after_avg_q)` and PaCoLoss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn import Parameter


class NormedLinear_Classifier(nn.Module):
    def __init__(self, num_classes=1000, feat_dim=2048):
        super(NormedLinear_Classifier, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_classes))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x, *args):
        out = F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))
        return out


def flatten(t):
    return t.reshape(t.shape[0], -1)


class MoCo(nn.Module):
    """
    One-view MoCo-style wrapper for PaCo.

    It keeps the same class name `MoCo` so the original training script can still call:
        model = moco.builder.MoCo(...)

    Input during training:
        im_q:   one view of the input sample
        labels: labels of im_q
        im_k:   ignored, kept only for API compatibility

    Output during training:
        features:   [B + K, dim] = query features + queue features
        target:     [B + K]      = query labels + queue labels
        logits_q:   [B, C]       = class-center/classifier logits

    Output during inference:
        logits_q:   [B, C]
    """

    def __init__(
        self,
        base_encoder,
        dim=128,
        K=65536,
        m=0.999,              # kept for API compatibility; not used in one-view mode
        T=0.07,
        mlp=False,
        normalize=True,
        num_classes=1000,
        feat_dim=2048,
    ):
        super(MoCo, self).__init__()
        self.K = K
        self.m = m
        self.T = T
        self.num_classes = num_classes
        self.feat_dim = feat_dim

        # Only the query encoder is needed for one-view PaCo.
        self.encoder_q = base_encoder(num_classes=dim)

        # Class-center / classifier layer.
        # In PaCo, class centers are represented through the classifier weights.
        self.linear = nn.Linear(feat_dim, num_classes)

        if mlp:
            dim_mlp = self.encoder_q.fc.weight.shape[1]
            self.encoder_q.fc = nn.Sequential(
                nn.Linear(dim_mlp, dim_mlp),
                nn.BatchNorm1d(dim_mlp),
                nn.ReLU(),
                self.encoder_q.fc,
            )

        # Queue stores old query features. Shape: [K, dim]
        self.register_buffer("queue", torch.randn(K, dim))
        self.queue = F.normalize(self.queue, dim=1)
        self.register_buffer("queue_l", torch.randint(0, num_classes, (K,)))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Hook for the feature before the final projection head.
        # For ResNet, layer=-2 normally points to avgpool.
        self.layer = -2
        self.feat_after_avg_q = None
        self.normalize = normalize
        self._register_hook()

    def _find_layer(self, module):
        if isinstance(self.layer, str):
            modules = dict([*module.named_modules()])
            return modules.get(self.layer, None)
        if isinstance(self.layer, int):
            children = [*module.children()]
            return children[self.layer]
        return None

    def _hook_q(self, _, __, output):
        self.feat_after_avg_q = flatten(output)
        if self.normalize:
            self.feat_after_avg_q = F.normalize(self.feat_after_avg_q, dim=1)

    def _register_hook(self):
        layer_q = self._find_layer(self.encoder_q)
        assert layer_q is not None, f"hidden layer ({self.layer}) not found"
        layer_q.register_forward_hook(self._hook_q)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, labels):
        """
        Add current query features to the queue.

        In DDP, features/labels are gathered from all GPUs before enqueueing.
        In single-GPU/non-DDP, concat_all_gather returns the input directly.
        """
        keys = concat_all_gather(keys)
        labels = concat_all_gather(labels)

        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)

        # Original MoCo assumes K % batch_size == 0.
        # Here we support the final partial wrap for convenience.
        if ptr + batch_size <= self.K:
            self.queue[ptr:ptr + batch_size, :] = keys
            self.queue_l[ptr:ptr + batch_size] = labels
        else:
            first = self.K - ptr
            self.queue[ptr:, :] = keys[:first]
            self.queue_l[ptr:] = labels[:first]
            remain = batch_size - first
            self.queue[:remain, :] = keys[first:]
            self.queue_l[:remain] = labels[first:]

        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    def _train(self, im_q, im_k=None, labels=None):
        """
        One-view training forward.
        `im_k` is ignored and only kept so old calls do not crash.
        """
        if labels is None:
            raise ValueError("labels must be provided during training")

        # Query features used for sample-to-sample contrast.
        q = self.encoder_q(im_q)
        q = F.normalize(q, dim=1)

        # features = current query + old queue
        features = torch.cat((q, self.queue.clone().detach()), dim=0)
        target = torch.cat((labels, self.queue_l.clone().detach()), dim=0)

        # Update queue with current one-view query features.
        self._dequeue_and_enqueue(q.detach(), labels.detach())

        # Class-center logits. These are used by PaCoLoss as sample-to-center similarities.
        logits_q = self.linear(self.feat_after_avg_q)

        return features, target, logits_q

    def _inference(self, image):
        _ = self.encoder_q(image)
        encoder_q_logits = self.linear(self.feat_after_avg_q)
        return encoder_q_logits

    def forward(self, im_q, im_k=None, labels=None):
        if self.training:
            return self._train(im_q=im_q, im_k=im_k, labels=labels)
        return self._inference(im_q)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    Safe for both DDP and non-DDP execution.
    """
    if not dist.is_available() or not dist.is_initialized():
        return tensor

    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    output = torch.cat(tensors_gather, dim=0)
    return output
