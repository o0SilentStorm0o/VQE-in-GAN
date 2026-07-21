# Post-hoc diagnostic protocol for the failed contrastive candidate

## Scope

This protocol records a mechanistic, post-hoc analysis of Candidate 4 after its frozen Stage 1
rejection. It cannot rehabilitate the candidate or turn the held-out seeds into development data.
Its only purpose is to answer:

> Why did a 5% modular quantum addition slightly improve mean conditional accuracy while failing
> to improve mean class-conditional FID, and why did the paired effect reverse between seeds?

No candidate constants are tuned. No additional multi-step GAN training is authorized. The only
optimizer execution in this diagnostic is an exact reconstruction of the first generator update,
used to separate the effects of classical-logit scaling and quantum-logit addition.

## Fixed artifacts

The diagnostic uses the four completed CPU checkpoints from source revision
`1917d0648ab6d9d21a217729533167807b9c8096`:

| Seed | Variant | Checkpoint SHA-256 |
|---:|---|---|
| 101 | Log-KDE | `0899f2b8f06ac29d589343663672461614b274d7ef71403d435bf1fff3a673eb` |
| 101 | Full hybrid | `c141c40a480322d34ea41de1b7bdb0256bc9d5db72641b72fa87a513ea4624f8` |
| 202 | Log-KDE | `ae459e43674f1b723b1f47f6dc81f86c099d2d1345dc2e5e892a4e318dca1871` |
| 202 | Full hybrid | `2f3675f6eca0148d8f7c47e3f5680c8887cfb31a4360e0fd4d6f3f19bf55bae3` |

The MNIST classifier digest is
`e9dcb79e44703f86afe208bca8f548fcc715dcc0c464a19e591b5a6fef0eefb8`.

## Questions and measurements

### 1. What is actually trainable and shared?

- Count trainable parameters in the contrastive circuit regularizer.
- Reconstruct the seeded generator initialization and compare every generator module with each
  final checkpoint.
- Report whether the generator `angle_head` changed at all.

This distinguishes a trainable VQE branch from a fixed image-to-state feature map whose gradient
only reaches the classical image-producing path.

### 2. Which classes create the aggregate trade-off?

Using the original 5,000 balanced generated samples and evaluation seed `91001`, report per class:

- independent-classifier accuracy and target probability;
- class-conditional feature FID;
- the FID mean/centroid and covariance components;
- intra-class feature diversity;
- the most common wrong prediction.

The hybrid-minus-KDE difference is reported for every class. Counts of improved and degraded
classes are retained; an aggregate mean alone is insufficient.

### 3. Does the modular score add a meaningful decision signal?

On 1,000 balanced generated samples, report KDE, quantum, and hybrid cross-entropy, requested-class
accuracy, target margin, and centered-logit RMS. Also report KDE/quantum agreement and how often the
hybrid changes a KDE decision beneficially or harmfully.

### 4. Is the comparison confounded by classical-logit scaling?

The frozen losses are

`L_K = CE(K, y)` and `L_H = CE(0.95 K + 0.05 Q, y)`.

The diagnostic inserts a read-only temperature-matched control

`L_S = CE(0.95 K, y)`

and decomposes the generator-parameter gradient exactly as

`grad(L_H - L_K) = grad(L_S - L_K) + grad(L_H - L_S)`.

The first term is the effect of weakening/heating the classical logits. The second is the isolated
quantum-logit addition at the frozen mixture coefficient. Norms, cosines, and the numerical
identity residual are reported separately.

### 5. What does the isolated addition align with?

On the actual first training batch and on a fixed balanced diagnostic batch of 100 samples with
noise seed `123456`, compare the scale effect, isolated quantum addition, and total hybrid-minus-KDE
gradient with:

- the ACGAN adversarial and auxiliary losses;
- an independent MNIST-classifier cross-entropy;
- squared distance to the real target-class feature centroid;
- a local diversity-promoting negative within-class variance proxy.

The latter two are local diagnostic proxies, not substitutes for FID or recall.

### 6. How is a small first perturbation amplified?

Reconstruct the exact first discriminator update and then apply three first generator updates from
the same state: full KDE, `0.95*KDE`, and the frozen hybrid. Report update-vector distances and the
fraction of the hybrid-versus-KDE difference reproduced by classical scaling alone. Then compare
the first-update distance with the final paired parameter distance and identify the first logged
step at which GAN and discriminator losses diverge.

## Interpretation boundary

These measurements may identify why this implementation failed. They cannot prove that every
trainable quantum generator branch or every quantum-assisted distribution loss will fail. Any new
mechanism proposed from this analysis must receive new development data, a matched classical
control that preserves logit scale, and a new untouched evaluation sequence.
