# TSC-1View Implementation

This folder implements a **one-view modified version** of:

**Targeted Supervised Contrastive Learning for Long-Tailed Recognition (CVPR 2022)**

## Why one-view?

Original TSC uses two views:

```text
x_i        -> v_i
aug(x_i)   -> v_i_tilde
```

This implementation assumes your pipeline has only:

```text
x_i -> z_i
```

So we remove the augmented self-positive, but keep the main TSC idea:

```text
same-class contrastive term + class-target contrastive term
```

## Files

```text
tsc_oneview_loss.py           # Core TSC-1View loss
train_tsc_oneview_stage1.py   # Stage 1: train encoder
train_linear_stage2.py        # Stage 2: freeze encoder and train linear classifier
example_usage.py              # Minimal usage example
```

## Loss input

The loss expects:

```python
features: Tensor[B, D]  # embedding from backbone
labels:   Tensor[B]     # class labels
```

It internally maintains:

```text
targets:       [C, D] uniform class targets
class_centers: [C, D] moving-average class centers
assignment:    class_id -> target_id
```

## Core idea

For each anchor `z_i`:

1. Pull it close to selected same-class features from the current batch.
2. Pull it close to the assigned target of its class.
3. Push it away from other samples and other class targets.

## Formula intuition

Original TSC denominator:

```text
augmented self-view + other batch features + all class targets
```

One-view denominator:

```text
other batch features + all class targets
```

## Important notes

- This is a **modified TSC**, not the exact original paper implementation.
- It is useful when augmentation is unavailable or unsafe for your data.
- If batch sampling rarely puts same-class positives together, the target term still provides learning signal.
- For serious experiments, use a sufficiently large batch or a class-aware sampler to improve same-class positives.
- `scipy` is recommended for Hungarian matching. If unavailable, the code uses a greedy fallback.

## Typical use

```python
from train_tsc_oneview_stage1 import train_tsc_oneview_stage1

result = train_tsc_oneview_stage1(
    encoder=encoder,
    train_loader=train_loader,
    optimizer=optimizer,
    num_classes=10,
    emb_dim=128,
    epochs=100,
    tau=0.1,
    k=6,
    lambda_target=1.0,
)
```

Then train a linear classifier:

```python
from train_linear_stage2 import train_linear_stage2

classifier, best_acc = train_linear_stage2(
    encoder=result["encoder"],
    train_loader=train_loader,
    val_loader=val_loader,
    emb_dim=128,
    num_classes=10,
)
```

## How to adapt to cct_retnet

Edit `get_embedding()` inside `train_tsc_oneview_stage1.py`.

If your model returns:

```python
logits, features = model(x)
```

the default implementation already takes `out[-1]`, so it will use `features`.

If your model only returns logits, change your model to return embeddings before the classifier.
