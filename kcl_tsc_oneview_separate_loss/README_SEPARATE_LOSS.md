# KCL/TSC One-View With Separated Loss

This version is based on the original `builder.py` implementation style, but the loss is separated from the model.

## Files

```text
builder_oneview_model.py       # model, queue, targets, masks, assignment
losses_oneview.py              # separated KCL/TSC loss
example_separate_loss_train.py # minimal training example
```

## Original style

Original `KCL.forward()` computes everything inside the model:

```python
logits, labels, q, feat_q, loss, loss_class, loss_target = model(im_q, im_k, im_labels)
```

## New separated style

Now the model only returns logits and masks:

```python
output = model(images, labels)
```

Then the criterion computes loss:

```python
loss_output = criterion(
    logits=output["logits"],
    mask_class=output["mask_class"],
    mask_target=output["mask_target"],
)
loss = loss_output["loss"]
```

## Training loop

```python
output = model(images, labels)
loss_output = criterion(output["logits"], output["mask_class"], output["mask_target"])

optimizer.zero_grad()
loss_output["loss"].backward()
optimizer.step()
```

## Why this is better

Separating the loss makes it easier to:

1. replace the loss later,
2. add CE loss or LDAM loss,
3. log `loss_class` and `loss_target`,
4. combine with your existing training pipeline,
5. keep the model focused on feature extraction, queue, and target assignment.

## One-view difference

Original KCL/TSC uses:

```text
im_q + im_k
positive = q · k
```

This one-view version uses:

```text
im only
positives = same-class features in queue + assigned class target
```

So this is not the exact original TSC. It is a one-view adaptation that preserves the TSC idea of uniform class targets and online class-target assignment.
