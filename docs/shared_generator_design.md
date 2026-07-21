# Shared generator and optimizer-step boundaries

## Minimal architectural correction

The image path intentionally retains the historical ACGAN structure:

1. concatenate 100-dimensional noise with a 10-dimensional learned class embedding;
2. project to a $256\times7\times7$ feature map;
3. upsample and decode it to a $1\times28\times28$ image.

An initial correction attached the angle head directly to the projected feature map. A 500-step
diagnostic pilot showed that the energy target could then be predicted equally well with ordinary
noise and with zero noise. The head was reading the class embedding directly rather than depending
on image content.

The corrected angle head therefore reads only the completed $1\times28\times28$ generated image.
Two small convolutional layers, global average pooling, and a multilayer perceptron produce 16
angles in $[-\pi,\pi]$. The head receives no separate class label or latent vector. Regularizer
gradients must traverse the complete image generator, including the convolutional decoder, before
reaching the shared latent projection.

The angle head contains 29,200 parameters. The complete generator contains 1,791,989 parameters,
of which 1,762,789 belong to the image-producing path shared with the regularizer gradient.

A regularizer-only test must produce non-zero gradients in both the input projection and image
decoder and must change subsequent generated images for fixed noise and labels.

## Explicit computation boundary

Image generation, angle production, and energy evaluation are separate operations:

- ordinary generation invokes only the shared path and image decoder;
- a quantum generator step generates each image once and derives angles from that image;
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
