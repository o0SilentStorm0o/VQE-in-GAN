# Trainable relational-coverage development protocol

## Status and scope

This document freezes the mechanism and the development decision rule before any training result
from the new candidate is inspected. The experiment is exploratory. Seeds 42 and 43 may be used
for calibration and development because they have already been used during earlier design work.
Seeds 101 and 202 are diagnostic-only after the rejected contrastive experiment and must not be
reported as fresh evidence. Seeds 303, 404, and 505 remain untouched.

The narrow question is:

> Can a trainable four-qubit branch add a useful within-class support-and-coverage signal to the
> unchanged log-KDE ACGAN objective, beyond an equally weighted classical kernel operating on the
> same angles?

This does not test computational quantum advantage. A four-qubit statevector is inexpensive to
simulate classically. It tests whether the circuit-induced geometry supplies a useful inductive
bias under a controlled optimization budget.

## Causal repair

The rejected contrastive candidate answered only “which class?” and mixed `0.95 * KDE` with a
small quantum score. The repair changes both faults without changing the ACGAN or the circuit:

1. The log-KDE loss remains exactly `CE(KDE_logits, requested_class)` with outer weight `2e-5`.
   It is not rescaled inside a hybrid logit.
2. The new term is additive and has its own fixed weight. It cannot change the KDE temperature.
3. The new term compares generated samples with many real samples of the *same* class. It asks for
   support membership and reverse coverage, not another class decision.
4. The generator's existing `angle_head` supplies a bounded trainable residual to the image-based
   circuit angles. Gradients must reach both the head and the image-producing path.
5. A periodic classical RBF kernel receives the identical reference images, base angles,
   trainable residual angles, soft-min objective, and initial shared-gradient budget.

## Frozen angle construction

The existing four-qubit, 16-angle circuit with one circular CNOT ring is retained. Images are
adaptively pooled to `4 x 4` and use the previously calibrated spatial permutation

`[5, 7, 15, 13, 0, 2, 10, 8, 1, 3, 11, 9, 4, 6, 14, 12]`.

For a generated image `x`,

`theta_G(x) = 0.75*pi*permute(pool_4x4(x)) + 0.10*angle_head(x)`.

`angle_head(x)` is already bounded to `[-pi, pi]`, so the learned correction cannot exceed
`0.10*pi` per angle. Its final linear layer is initialized to zero for both trainable-coverage
variants. The candidate therefore starts at the deterministic image encoding rather than at an
arbitrary learned offset. The direct pooled-image path remains present even if the head learns.

Real reference images use only the deterministic first term. Their encodings are computed once
from the first balanced 1,024 training examples per class and detached. No test image is used.

## Projector-energy coverage objective

Let `|g_i>` be a generated circuit state and `|r_j>` a fixed real reference state from the same
class. Their projector energy is

`E_ij = <g_i|(I - |r_j><r_j|)|g_i> = 1 - |<g_i|r_j>|^2`.

Thus the circuit is optimized end to end by an energy expectation, but the Hamiltonians are
data-derived projectors rather than class labels. For a matrix of same-class energies, define the
normalized soft minimum

`smin_tau(E_1...E_n) = -tau*log(mean(exp(-E/tau)))`, with `tau = 0.10`.

The per-class loss is the mean of:

- generated-to-real support: average row-wise soft minimum;
- real-to-generated coverage: average column-wise soft minimum.

The first term prevents samples from sitting far from all real references. The reverse term makes
duplicate generated states unhelpful and gives distinct samples an incentive to cover different
parts of the real class support. Classes are averaged equally within every minibatch.

## Matched classical control

The classical control replaces only the quantum fidelity. It uses the same combined generated
angles and deterministic real angles with periodic distance

`d(theta, phi) = sum_k(1 - cos(theta_k - phi_k))`

and kernel `k_C = exp(-d / sigma_squared)`. `sigma_squared` is fixed during reference fitting so
that the median nonzero within-class real-reference distance has kernel value `0.5`. Its energy is
`1 - k_C`, followed by the identical bidirectional soft-min objective.

Separate fixed outer weights are allowed because raw kernel-gradient scales differ. Both weights
must be chosen solely by the calibration rule below, not by GAN outcomes.

## Frozen variants

1. `classical_log_kde_contrastive`: unchanged KDE baseline, weight `2e-5`.
2. `classical_log_kde_scale_control`: the old `0.95` KDE scale without a quantum addition,
   implemented as KDE weight `1.9e-5`.
3. `classical_kde_relational_coverage`: exact KDE plus periodic-RBF relational coverage.
4. `quantum_kde_relational_coverage`: exact KDE plus entangled-state projector coverage.

Product-circuit and dephased-state controls are implemented only after the full candidate passes
the development gate. They may not be substituted post hoc for a failed full candidate.

## Weight calibration before development training

The coverage weights are initially unset. A deterministic calibration script must use only the
first four shuffled training batches for seeds 42 and 43, initialized networks, and the frozen
reference bank. It measures gradients on the image-producing generator parameters only:

`r = ||grad(coverage)|| / ||grad(GAN)||`.

For each kernel independently, its fixed outer weight is

`median_over_8_batches(0.01 / r)`.

This gives the added term a target initial shared-gradient budget of 1% of the ordinary ACGAN
gradient. It is deliberately smaller than the calibrated KDE contribution but materially larger
than the ineffective isolated contribution in the rejected experiment. The resulting two scalar
weights and the eight raw measurements must be committed before development training starts.
There is no online gradient balancing and no per-seed weight.

Calibration must also record, but not optimize against:

- coverage/GAN gradient cosine on the shared image path;
- coverage/KDE gradient cosine on the shared image path;
- raw gradient norm on `angle_head`;
- the fraction of the coverage gradient norm carried by the direct image path;
- full/product/dephased loss and gradient differences on the same batch.

### Frozen calibration result

The calibration was executed from source revision `4bc24e8f09294e6100eea95b8581ce30817f79ce`
with 1,024 real references per class. The committed outer weights are:

- periodic classical coverage: `0.0033809364895852783`;
- full quantum coverage: `0.0005662128371831583`.

These are the exact medians produced by the rule above. After applying them, the individual
initial coverage/GAN shared-gradient ratios span approximately `0.80%` to `1.65%` for the
classical control and `0.20%` to `1.50%` for the quantum candidate; both medians are `1%`.

The calibration already provides useful mechanism evidence, without any GAN outcome:

- zero residual angles are exact and both matched variants retain an identical KDE loss;
- both coverage losses reach the image-producing path and the final angle-output layer;
- full/product and full/dephased gradient differences are nonzero on both seeds;
- quantum-coverage/KDE gradient cosine is negative on all eight batches (`-0.752` to `-0.976`),
  so the quantum term is not a rescaled copy of KDE;
- quantum-coverage/GAN cosine is positive on all seed-42 batches and negative on all seed-43
  batches, warning in advance that usefulness may remain trajectory-dependent.

The complete values are preserved in
[`relational_coverage_calibration.json`](relational_coverage_calibration.json), SHA-256
`183ba751b92fed445b079650e52b783d1992634cbc637d2a051c1dd918b16e87`.
Its dirty-source flag is caused by the pre-existing untracked PDF and temporary directory; the
calibration source itself is the recorded committed revision.

## Structural gates before training

The candidate is invalid unless tests and a real MNIST calibration batch show all of the following:

1. zero trainable angle residual reproduces the fixed deterministic encoding exactly;
2. coverage gives finite, nonzero gradients to `angle_head`;
3. coverage gives finite, nonzero gradients to the image-producing generator parameters;
4. the KDE component and gradient are bitwise identical between the KDE baseline and both
   additive variants before the coverage term is applied;
5. the matched classical and quantum variants have identical neural parameter counts and initial
   image-producing weights for a paired seed;
6. the full circuit is numerically distinguishable from both product and dephased controls.

## Development run and stop rule

After calibration is committed, run all four variants on seeds 42 and 43 with:

- deterministic CPU execution;
- batch size 64;
- 200 generator steps;
- 5,000 balanced evaluation samples with seed 91001;
- identical data order, noise stream, neural initialization, optimizer settings, and classifier.

The full quantum candidate advances to product/dephased ablations only if all conditions hold:

1. its mean class-conditional FID is lower than exact KDE and is not worse on either seed;
2. its mean diversity ratio is not lower than exact KDE;
3. its mean conditional accuracy is no more than 0.005 below exact KDE;
4. its mean class-conditional FID is lower than the matched classical coverage control;
5. the trained `angle_head` changes, while post-run gradients still reach the image-producing path;
6. no NaN, reference-fit failure, or paired-protocol mismatch occurs.

Failure rejects this candidate at the development stage. The weights, temperature, residual
bound, kernel bandwidth rule, or gate must not be tuned after seeing these two outcomes. A later
mechanism may be designed under a new protocol, but it is a new candidate.

Passing this gate is not confirmatory evidence. It only licenses the causal circuit ablations and,
if those isolate a coherent entangling contribution, a separately frozen run on untouched seeds.

## Development result

The four frozen variants were run from source revision
`22c72c51aa40b5d757e7a5b05074779f8767242d`. Every run completed 200 deterministic CPU steps and
used the same 5,000-sample evaluation.

| Variant | Mean accuracy | Mean FID | Mean class-FID | Mean precision | Mean recall | Mean diversity |
|---|---:|---:|---:|---:|---:|---:|
| Exact log-KDE | 0.6689 | 0.4003 | 0.4649 | 0.5351 | 0.0215 | 0.5360 |
| `0.95` KDE scale control | 0.6710 | 0.3415 | 0.4025 | 0.4588 | 0.0290 | 0.5665 |
| Matched classical coverage | 0.6817 | 0.3450 | 0.4062 | 0.4575 | 0.0200 | 0.5614 |
| Full quantum coverage | 0.6736 | 0.3381 | 0.4023 | 0.4332 | 0.0272 | 0.5745 |

Against exact KDE, the quantum candidate changes mean accuracy by `+0.0047`, class-FID by
`-0.0626`, and diversity by `+0.0385`. Class-FID improves on seed 42 by `-0.0530` and on seed 43
by `-0.0722`. Against matched classical coverage, its mean class-FID is lower by only `0.0040`;
the sign reverses between seeds. Its mean metrics are also close to the scale-only trajectory.
Therefore this is a passed development gate, not evidence that the circuit caused the gain.

The mechanism audit confirms that all 29,200 angle-head values changed in both quantum runs. At
the final checkpoints, both the angle-head and image-producing gradients remain nonzero, and the
direct image path carries more than 99% of the full shared-gradient norm. The branch is thus
trainable and image-coupled without being able to hide the loss entirely in the angle head.

All six preregistered checks pass. Full values, configurations, checkpoint hashes, timings, and
post-run gradients are preserved in
[`relational_coverage_development_results.json`](relational_coverage_development_results.json),
SHA-256 `90b2b104e5d07f4461b07ff7ea516ae66db88c54f1286992388b5e04b81d5fab`.

## Frozen circuit-ablation follow-up

Passing the development gate licenses exactly two new controls:

- `quantum_kde_relational_product`: remove the CNOT ring from both generated and reference
  circuits;
- `quantum_kde_relational_dephased`: retain the ring but replace state overlap with squared
  Bhattacharyya overlap of computational-basis probabilities.

Both retain exact KDE weight `2e-5`, quantum coverage weight `0.0005662128371831583`, the same
angle-head initialization and residual bound, the same references, and the same 200-step/evaluation
protocol. Their weights are not recalibrated. The full variant is rerun alongside both controls on
seeds 42 and 43. Its checkpoint hashes must exactly reproduce the development checkpoints before
the ablations are interpreted.

The circuit-specific gate passes only if:

1. full quantum mean class-FID is at least `0.005` lower than both product and dephased means;
2. full quantum mean diversity is no more than `0.005` below either control;
3. full quantum mean accuracy is no more than `0.005` below either control;
4. the exact full-run replay and all structural checks pass.

If product or dephased matches or beats the full circuit under this rule, the development gain is
not isolated to coherent entangling geometry. No held-out run is licensed in that case.
