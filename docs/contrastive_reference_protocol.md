# Frozen contrastive-reference protocol

## Scope

This protocol freezes a new exploratory experiment before any result from its held-out seeds is
inspected. It tests a narrow claim:

> Does a small coherent, entangled modular-energy contribution improve a strong classical
> class-conditional reference loss at a fixed early-training horizon?

It does not test computational quantum advantage. The four-qubit statevector and dense modular
Hamiltonians are classically inexpensive at this scale.

## Why this candidate exists

The earlier modular-reference loss minimizes only the requested class energy. It does not require
that energy to be lower than the other nine class energies. Its reference densities are also
nearly pure at the original `0.125*pi` scale, making the gradient almost equivalent to prototype
fidelity. These two facts explain its weak semantic alignment and seed instability.

The replacement uses all class energies as logits. For class `c`, the real reference density
defines `H_c = -log(rho_c)`. The identity component of every `H_c` is removed and the remaining
observable is normalized to unit Frobenius norm. A generated image is penalized by cross-entropy
over the ten negative energy expectations, not by one target energy in isolation.

## Frozen image-to-circuit mapping

The circuit remains the existing 16-angle, four-qubit circuit with one CNOT ring. Only the fixed
assignment of pooled image values to its rotations changes.

- Images are adaptively pooled to `4 x 4`.
- Qubits follow the spatial ring top-left, top-right, bottom-right, bottom-left.
- Within each quadrant, rotation layers receive bottom-right, top-left, top-right, bottom-left.
- The resulting flattened permutation is
  `[5, 7, 15, 13, 0, 2, 10, 8, 1, 3, 11, 9, 4, 6, 14, 12]`.
- Angle scale is `0.75*pi`.
- Reference depolarization is `0.01`.
- Quantum logit temperature is `0.04`.

The mapping, scale, and temperature were selected by modular-classification negative log
likelihood on a training-only calibration split. They must not be changed after held-out GAN runs.

## Strong classical control and hybrid candidate

The classical control is a class-conditional log kernel-density estimator on exactly the same
pooled `4 x 4` input and the same balanced reference images. It uses

- squared RBF bandwidth `0.03125`;
- logit temperature `0.75`;
- `1,024` reference images per class.

Using a log density rather than a raw mean kernel prevents the classical gradient from vanishing
when generated samples are far from the reference bank.

The full candidate combines calibrated logits as

`0.95 * logit_KDE + 0.05 * logit_quantum`.

Product and dephased ablations retain the same mixture coefficient, data, circuit rotations, and
outer loss weight. Product removes the CNOT ring. Dephased retains the ring but discards every
off-diagonal density-matrix element.

## Frozen optimization settings

- Outer regularizer weight: `2e-5` for classical, full, product, and dephased variants.
- Online gradient balancing: disabled.
- The shared weight is the median-scale calibration that gives approximately a `0.10` initial
  regularizer/GAN parameter-gradient ratio on development seeds 42 and 43.
- Batch size: `64`.
- Horizon: `200` generator steps.
- Evaluation: `5,000` balanced generated images with seed `91001`.
- Neural architecture, optimizers, data order, generator labels, and evaluation classifier are
  identical across paired variants.

MPS runs are not admissible. Paired MPS replays diverged despite strict deterministic mode. CPU or
a separately replay-verified deterministic CUDA environment is required.

## Development evidence, not confirmation

The following results motivated the freeze. They are not held-out evidence.

| Seed | Variant | Accuracy | FID | Class-FID | Precision | Recall | Diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | ACGAN | 0.5404 | 0.2793 | 0.3563 | 0.4868 | 0.1254 | 0.6561 |
| 42 | Log-KDE | 0.5658 | 0.3161 | 0.3955 | 0.4982 | 0.0398 | 0.6130 |
| 42 | Full hybrid | 0.5834 | 0.2932 | 0.3706 | 0.4610 | 0.0472 | 0.6342 |
| 42 | Product hybrid | 0.5748 | 0.3174 | 0.3950 | 0.4270 | 0.0182 | 0.5915 |
| 42 | Dephased hybrid | 0.5142 | 0.2858 | 0.3659 | 0.4774 | 0.0722 | 0.6643 |
| 43 | Log-KDE | 0.7720 | 0.4844 | 0.5343 | 0.5720 | 0.0032 | 0.4590 |
| 43 | Full hybrid | 0.8006 | 0.4002 | 0.4515 | 0.4418 | 0.0044 | 0.5023 |

The full circuit improves accuracy, FID, class-FID, recall, and diversity over log-KDE on both
development seeds, while reducing precision. On seed 42, neither product nor dephased reproduces
the full circuit's conditioning improvement. This is sufficient to justify a held-out
falsification test, not a positive claim.

## Held-out sequence and stop rule

Seeds 44, 314, and 729 are excluded because earlier candidate diagnostics already inspected their
trajectories. The untouched sequence is fixed as `101, 202, 303, 404, 505`.

Stage 1 runs only log-KDE and the full hybrid on seeds 101 and 202. The candidate is rejected and
no additional tuning is allowed unless both conditions hold across the two paired runs:

1. mean conditional-accuracy difference `full - KDE` is positive;
2. mean class-FID difference `full - KDE` is negative.

Individual sign reversals must be reported even if the mean passes. If Stage 1 passes, product and
dephased ablations are run on the same two seeds, followed by the preregistered seeds 303, 404, and
505. Confirmatory claims require paired uncertainty intervals over the complete frozen set and a
longer-horizon follow-up; a single favorable seed is never sufficient.

## Stage 1 result

Stage 1 was executed on CPU from source revision
`1917d0648ab6d9d21a217729533167807b9c8096`, without changing the frozen configuration. Every row
uses the same 5,000 balanced evaluation samples and evaluation seed `91001`.

| Seed | Variant | Accuracy | FID | Class-FID | Precision | Recall | Diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 101 | Log-KDE | 0.7192 | 0.2894 | 0.3471 | 0.4972 | 0.0458 | 0.6173 |
| 101 | Full hybrid | 0.7358 | 0.3117 | 0.3662 | 0.5318 | 0.0254 | 0.6018 |
| 202 | Log-KDE | 0.7498 | 0.2646 | 0.3129 | 0.5740 | 0.0432 | 0.6001 |
| 202 | Full hybrid | 0.7486 | 0.2586 | 0.3055 | 0.5138 | 0.0348 | 0.6147 |

The paired `full - KDE` differences reverse sign between seeds:

| Seed | Accuracy difference | Class-FID difference |
|---:|---:|---:|
| 101 | +0.01660 | +0.01913 |
| 202 | -0.00120 | -0.00742 |
| **Mean** | **+0.00770** | **+0.00585** |

The mean accuracy condition passes, but the mean class-FID condition fails because positive FID
differences are worse. Candidate 4 is therefore **rejected at Stage 1**. Per the frozen rule, the
product and dephased variants and seeds 303, 404, and 505 are not run, and this candidate receives
no post-hoc tuning.

The machine-readable metrics, run times, source revision, classifier digest, and checkpoint
digests are preserved in
[`contrastive_stage1_results.json`](contrastive_stage1_results.json). The run provenance reports a
dirty worktree only because the repository contained a pre-existing untracked PDF and temporary
directory; there were no tracked source changes relative to the recorded revision during either
run, and neither untracked item participates in the experiment.

## Command

```bash
uv run python scripts/run_seed_matrix.py \
  --seed 101 \
  --output-root runs/contrastive-heldout-seed-101 \
  --dataset-root data \
  --classifier mnist_classifier.pth \
  --epochs 1 \
  --max-steps 200 \
  --evaluation-samples 5000 \
  --device cpu \
  --quantum-device cpu \
  --variants classical_log_kde_contrastive hybrid_modular_kde_contrastive
```
