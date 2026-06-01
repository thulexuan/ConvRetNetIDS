# One-view PaCo modification

This package modifies PaCo for the case where each input sample has only one view and no augmented second view.

Files included:

```text
moco/builder.py                 replacement model file
losses.py                       replacement PaCo loss file
PATCH_paco_imagenet_oneview.md  training-script changes
```

Core modification:

```text
Original PaCo:  features = [q, k, queue]
One-view PaCo:  features = [q, queue]
```

Why this works:

PaCo does not only rely on two image views. Its strongest part is the use of class-wise learnable centers. Even with one view, each anchor can still contrast against:

```text
positive: correct class center + same-class queue samples
negative: wrong class centers + different-class queue samples
```

For an IDS dataset with 10 classes, set:

```python
num_classes = 10
```

and create the model with:

```python
model = moco.builder.MoCo(
    base_encoder,
    dim=args.moco_dim,
    K=args.moco_k,
    m=args.moco_m,
    T=args.moco_t,
    mlp=args.mlp,
    normalize=args.normalize,
    num_classes=10,
)
```
