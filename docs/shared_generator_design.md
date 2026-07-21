# Shared generator and optimizer-step boundaries

## Minimal architectural correction

The image path intentionally retains the historical ACGAN structure:

1. concatenate 100-dimensional noise with a 10-dimensional learned class embedding;
2. project to a $256\times7\times7$ feature map;
3. upsample and decode it to a $1\times28\times28$ image.

The corrected angle head reads that same projected feature map. It applies global average pooling
followed by a small multilayer perceptron and produces 16 angles in $[-\pi,\pi]$. Therefore the
class embedding and input projection are shared parameters, while the convolutional image decoder
and angle head remain branch-specific.

This is a conservative correction: it preserves the ACGAN image generator and adds only 34,960
angle-head parameters. The shared path contains 1,392,484 parameters. A quantum-loss-only test
must produce non-zero gradients in this shared path and must change subsequent generated images
for fixed noise and labels.

## Explicit computation boundary

Image generation, angle production, and energy evaluation are separate operations:

- ordinary generation invokes only the shared path and image decoder;
- a quantum generator step computes images and angles from the shared representation once;
- a discriminator step never produces angles or invokes an energy backend;
- a zero-weight baseline never invokes an energy backend;
- all ten class energies are obtained in one batched backend call.

This boundary makes quantum call counts testable and avoids paying simulator cost during
discriminator training, sample generation, or evaluation.

## Optimizer isolation

During the discriminator step, the generator is evaluated under `no_grad` and its BatchNorm state
is held fixed. During the generator step, discriminator parameters are marked non-trainable and
its BatchNorm state is held fixed. The original train/eval mode is restored after each step.

Tests compare complete model state dictionaries, not only gradients. This catches accidental
updates of running means, running variances, and batch counters in addition to parameter changes.

## What this correction establishes

The computation graph now permits the quantum objective to affect images and prevents unrelated
optimizer steps from contaminating each model. It does not establish that this signal improves
image quality. That question remains reserved for matched, multi-seed ablations after the training
and evaluation protocol is frozen.
