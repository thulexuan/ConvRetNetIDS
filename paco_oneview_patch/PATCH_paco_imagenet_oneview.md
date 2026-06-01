# Patch instructions for one-view PaCo training

This patch changes the official PaCo Full-ImageNet training path from two-view training to one-view training.

Original PaCo/MoCo training expects:

```python
images[0]  # im_q
images[1]  # im_k
features = [q, k, queue]
```

One-view PaCo expects:

```python
images     # single image tensor
features = [q, queue]
```

## 1. Replace files

Copy these files into the official folder:

```text
paco_oneview_patch/moco/builder.py  ->  PaCo/Full-ImageNet/moco/builder.py
paco_oneview_patch/losses.py        ->  PaCo/Full-ImageNet/losses.py
```

## 2. Add num_classes argument when creating the model

In `paco_imagenet.py`, replace:

```python
model = moco.builder.MoCo(
    getattr(resnet, args.arch),
    args.moco_dim,
    args.moco_k,
    args.moco_m,
    args.moco_t,
    args.mlp,
    args.normalize,
)
```

with:

```python
num_classes = 8142 if args.dataset == 'inat' else 1000

model = moco.builder.MoCo(
    getattr(resnet, args.arch),
    dim=args.moco_dim,
    K=args.moco_k,
    m=args.moco_m,
    T=args.moco_t,
    mlp=args.mlp,
    normalize=args.normalize,
    num_classes=num_classes,
)
```

For your IDS dataset with 10 classes, use:

```python
num_classes = 10
```

## 3. Pass num_classes to PaCoLoss

Replace:

```python
criterion = PaCoLoss(
    alpha=args.alpha,
    beta=args.beta,
    gamma=args.gamma,
    temperature=args.moco_t,
    K=args.moco_k,
).cuda(args.gpu)
```

with:

```python
criterion = PaCoLoss(
    alpha=args.alpha,
    beta=args.beta,
    gamma=args.gamma,
    temperature=args.moco_t,
    K=args.moco_k,
    num_classes=num_classes,
).cuda(args.gpu)
```

## 4. Use single-view transform, not TwoCropsTransform

Original code uses something like:

```python
transform_train = moco.loader.TwoCropsTransform(transforms.Compose(augmentation_regular))
```

Replace with:

```python
transform_train = transforms.Compose(augmentation_regular)
```

If you truly do not want augmentation, use only resize/crop/tensor/normalize, for example:

```python
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize,
])
```

For IDS/tabular data, your dataset should simply return:

```python
x, y
```

not:

```python
[x_view1, x_view2], y
```

## 5. Modify the train loop

Original code:

```python
for i, (images, target) in enumerate(train_loader):
    if args.gpu is not None:
        images[0] = images[0].cuda(args.gpu, non_blocking=True)
        images[1] = images[1].cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

    features, labels, logits = model(im_q=images[0], im_k=images[1], labels=target)
    loss = criterion(features, labels, logits)
```

Replace with:

```python
for i, (images, target) in enumerate(train_loader):
    if args.gpu is not None:
        images = images.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)

    features, labels, logits = model(im_q=images, labels=target)
    loss = criterion(features, labels, logits)
```

For fp16 branch, replace similarly:

```python
with autocast():
    features, labels, logits = model(im_q=images, labels=target)
    loss = criterion(features, labels, logits)
```

## 6. What changes conceptually?

Original PaCo:

```text
q = encoder_q(view1)
k = encoder_k(view2)
features = [q, k, queue]
```

One-view PaCo:

```text
q = encoder_q(x)
features = [q, queue]
```

The loss still learns:

```text
anchor close to correct class center
anchor close to same-class samples in queue
anchor far from wrong class centers
anchor far from different-class samples in queue
```
