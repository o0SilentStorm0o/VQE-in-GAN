# Quantum-utility audit

## Status and question

This document records exploratory mechanism tests performed before any confirmatory experiment.
None of the numbers below are preregistered results. The purpose is to answer a narrower question:

> Does a four-qubit auxiliary circuit provide a reproducible benefit to the corrected ACGAN that
> cannot be explained by real-data access, gradient scaling, or a matched classical feature map?

The current answer is **no robust quantum-specific benefit in the tested architecture**. Several
quantum losses improve individual ACGAN trajectories, but the improvement is seed-dependent and
equal or stronger classical controls explain it. The strongest class-contrastive candidate also
failed its frozen held-out class-FID gate. This does not prove that every possible quantum-assisted
GAN must fail. It does reject the mechanisms tested here and prevents them from being promoted to
a confirmatory claim.

## Evidence required for a quantum-specific claim

A useful quantum branch must pass all of these gates:

1. Its gradient must reach the image-producing generator path and depend on generated image
   content rather than a label shortcut.
2. It must improve a distribution metric over `no_regularizer` across paired seeds, not only one
   training trajectory.
3. It must improve over a classical control with the same real data, input resolution, gradient
   budget, and parameter count.
4. Removing coherence, entanglement, or the intended class alignment must remove the benefit.
5. The result must survive a frozen confirmatory seed set and a longer convergence horizon.

Passing only the first gate establishes a working hybrid computation graph, not quantum utility.

## Audit of the original and corrected Ising branch

The original fixed diagonal Hamiltonian has several structural limitations:

- all historical class Hamiltonians share the ground state `|0000>`;
- the corrected energy vector contains only seven distinct diagonal observables and its
  class-centered score matrix has rank six;
- the final four RZ angles of the 16-angle circuit are unobservable immediately before a diagonal
  Z measurement, with numerical derivatives at approximately machine precision;
- the quantum branch never observes real data and can therefore reward generator-specific
  artifacts;
- dynamic norm balancing controls gradient magnitude, but it cannot repair an irrelevant or
  conflicting direction.

In an independent 500-step diagnostic, the fixed quantum energy classified real MNIST at 13.21%,
the generated target energy agreed with requested labels at 22.5%, and an independent image
classifier agreed at 15.65%. The mean cosine between the energy gradient and the independent
semantic gradient was 0.00665, effectively orthogonal.

## Supervised capacity check

The same small image encoder was trained on real MNIST before involving a GAN. These are
exploratory 20-epoch measurements, not final benchmarks.

| Readout | Real-MNIST accuracy |
|---|---:|
| Fixed class-encoded Ising energy | 15.9% |
| Learned quantum density-matrix linear readout | 57.1% |
| Direct ten-logit classical head | 70.4% |
| Matched classical linear readout on the same angles | 74.5% |
| Learned quantum fidelity prototypes | approximately 11% |

The circuit representation contains class information, but the fixed energy readout is weak and
the matched classical readout extracts more of it. This check rejects the premise that the fixed
VQE-inspired objective is already a strong semantic teacher for the generator.

## Candidate 1: class-conditional modular free energy

### Mechanism

Each image is deterministically pooled to 16 values and encoded by the parity-tested four-qubit
circuit. Pure image states are averaged per class into generated and real mixed states,
`rho_G,c` and `rho_R,c`. The proposed loss is

$$
L_{CMF} = \frac{1}{C_B}\sum_{c\in B}
D(\rho_{G,c}\Vert\rho_{R,c})
= \frac{1}{C_B}\sum_{c\in B}
\left(\operatorname{Tr}[\rho_{G,c}(-\log\rho_{R,c})]-S(\rho_{G,c})\right).
$$

A 1% depolarizing mixture makes the matrix logarithm finite. The real state and its modular
Hamiltonian are detached; gradients flow only through generated images. Unlike the fixed Ising
loss, the target is derived from real data, uses the full noncommuting density matrix, and includes
entropy to oppose collapse.

Quantum relative entropy, modular Hamiltonians, and quantum kernel mean embeddings are established
primitives. The candidate contribution is their specific class-conditional, image-coupled use as
an ACGAN generator regularizer and its causal ablation family. Novelty has not been established by
a complete literature review.

### Paired 500-step pilots

All variants use batch size 64, a regularizer/GAN shared-gradient ratio of 0.10, 1,000 balanced
evaluation samples, and the same real-batch labels for generator updates. Lower FID values are
better; higher accuracy, precision, recall, and diversity are better.

| Seed | Variant | Accuracy | FID | Class-FID | Precision | Recall | Diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | ACGAN | 0.201 | 0.7619 | 0.8970 | 0.059 | 0.000 | 0.1712 |
| 42 | Modular quantum | 0.197 | 0.6627 | 0.7904 | 0.153 | 0.000 | 0.1960 |
| 42 | Classical RBF-MMD | 0.387 | 0.6759 | 0.8618 | 0.076 | 0.000 | 0.1852 |
| 43 | ACGAN | 0.170 | 0.7745 | 0.8819 | 0.072 | 0.000 | 0.3323 |
| 43 | Modular quantum | 0.519 | 0.5076 | 0.6583 | 0.111 | 0.000 | 0.5144 |
| 43 | Classical RBF-MMD | 0.758 | 0.3462 | 0.4293 | 0.530 | 0.001 | 0.5375 |
| 44 | ACGAN | 0.783 | 0.1538 | 0.2533 | 0.712 | 0.197 | 0.7045 |
| 44 | Modular quantum | 0.739 | 0.2068 | 0.3604 | 0.654 | 0.029 | 0.5831 |
| 44 | Classical RBF-MMD | 0.765 | 0.2057 | 0.2947 | 0.714 | 0.105 | 0.6481 |

Across these three exploratory seeds, the modular loss improves mean FID and accuracy over the
ACGAN, but its paired effect changes sign on the already strong seed 44. RBF-MMD has the better
three-seed mean for accuracy, FID, class-FID, precision, and diversity. Both regularizers reduce
mean recall. Three inspected seeds are far too few for inference, but they are sufficient to reject
an immediate confirmatory run of this formulation.

### Causal removals on development seed 42

| Variant | Accuracy | FID | Class-FID | Precision | Diversity |
|---|---:|---:|---:|---:|---:|
| Full modular loss | 0.197 | 0.6627 | 0.7904 | 0.153 | 0.1960 |
| Dephased density matrices | 0.158 | 0.8170 | 0.9219 | 0.168 | 0.1821 |
| Product circuit, no CNOT ring | 0.077 | 0.8533 | 0.9438 | 0.093 | 0.1765 |
| Cross-energy only, no entropy | 0.138 | 0.8048 | 0.9248 | 0.021 | 0.1896 |

The full construction beats these removals on most metrics for this one seed, so coherence,
entanglement, and entropy are not inert implementation details. That causal signal is interesting,
but it does not override the seed instability or the stronger classical control.

## Candidate 2: orthogonal coherence-residual guidance

### Mechanism

To isolate information unavailable to a dephased model, define

$$
R_Q = D(\rho_G\Vert\rho_R)
- D(\Delta\rho_G\Vert\Delta\rho_R),
$$

which is non-negative by data processing for the dephasing channel `Delta`. The training step first
applies classical RBF-MMD with the same 0.10 gradient budget. It then removes from `grad R_Q` the
direction already supplied by RBF and removes any remaining component that conflicts with the GAN
gradient in the orthogonal subspace. The retained quantum-only direction receives a 0.05 budget.

The dephasing residual and gradient projection are known mathematical tools. Their combination as
an incremental quantum-only audit for this GAN is a project-specific candidate, not an established
novelty claim.

### Result and decision

On development seed 42, accuracy fell from 0.387 for RBF-MMD to 0.311, FID worsened from 0.6759 to
0.7427, class-FID from 0.8618 to 0.8783, precision from 0.076 to 0.030, and diversity from 0.1852 to
0.1622. During training, the raw quantum and RBF gradients had mean cosine 0.962. Only 23.6% of the
quantum gradient norm survived on average after projection, falling to 15.6% over the final 50
steps. The allegedly unique remainder behaved as noise and harmed training. This candidate is
rejected and must not be tuned on the same seeds.

## Candidate 3: stable modular reference memory

### Mechanism

The minibatch modular Hamiltonian is estimated from roughly six real images per class, and the
matrix logarithm amplifies sampling noise. The reference-memory variant therefore encodes the
first balanced 256 training images per class once, constructs ten fixed modular Hamiltonians, and
uses them for every generator step. A matched classical control stores the same 256 pooled images
per class and computes RBF-MMD against that fixed bank. This separates a possible quantum effect
from the benefit of simply seeing a larger and more stable real sample.

### Paired result

Seeds 42 and 43 are development seeds. The reference configuration was then frozen at 256 real
images per class, angle scale `0.125*pi`, 1% depolarization, and gradient ratio 0.10 before seeds 314
and 729 were run. The reference loader has a private fixed RNG and a regression test verifies that
reference construction does not shift the paired training RNG stream.

| Seed | Variant | Accuracy | FID | Class-FID | Precision | Recall | Diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | ACGAN | 0.201 | 0.7619 | 0.8970 | 0.059 | 0.000 | 0.1712 |
| 42 | Modular reference | 0.474 | 0.6961 | 0.8136 | 0.151 | 0.000 | 0.2007 |
| 42 | Classical RBF reference | 0.105 | 0.8131 | 0.9562 | 0.000 | 0.000 | 0.1834 |
| 43 | ACGAN | 0.170 | 0.7745 | 0.8819 | 0.072 | 0.000 | 0.3323 |
| 43 | Modular reference | 0.405 | 0.5435 | 0.6447 | 0.224 | 0.000 | 0.3547 |
| 43 | Classical RBF reference | 0.304 | 0.5168 | 0.6859 | 0.253 | 0.000 | 0.3565 |
| 314 | ACGAN | 0.333 | 0.6052 | 0.7176 | 0.130 | 0.000 | 0.4337 |
| 314 | Modular reference | 0.538 | 0.4764 | 0.6866 | 0.401 | 0.000 | 0.3167 |
| 314 | Classical RBF reference | 0.385 | 0.5451 | 0.7364 | 0.063 | 0.000 | 0.3454 |
| 729 | ACGAN | 0.601 | 0.4734 | 0.6194 | 0.017 | 0.001 | 0.5252 |
| 729 | Modular reference | 0.473 | 0.6910 | 0.8293 | 0.000 | 0.000 | 0.4460 |
| 729 | Classical RBF reference | 0.819 | 0.2826 | 0.3766 | 0.596 | 0.052 | 0.5692 |

The modular reference improves class-FID over the classical reference on both development seeds
and on held-out seed 314. It also beats both controls on accuracy, FID, class-FID, and precision on
seed 314. It fails sharply on held-out seed 729, however. Across all four inspected seeds, its mean
accuracy is higher than the classical reference (0.473 versus 0.403), but the classical reference
has better mean FID (0.539 versus 0.602), class-FID (0.689 versus 0.744), precision (0.228 versus
0.194), recall (0.013 versus 0.000), and diversity (0.364 versus 0.329).

At the final checkpoints, the image-space cosine between the modular-reference gradient and an
independent classifier's semantic gradient was 0.0281, 0.0249, 0.0304, and 0.0068 for seeds 42, 43,
314, and 729 respectively. The failed seed receives an almost semantically empty rather than an
oppositely directed signal. This is a post-hoc mechanism clue, not an admissible gate: putting the
evaluation classifier into training would turn the method into a classically supervised one.

The stable reference is therefore the strongest original candidate found in this audit and proves
that the quantum loss can help individual trajectories, including one held-out seed. It does not
yet provide a robust quantum-specific improvement and must not be presented as a positive result.

## Backend reproducibility finding

Replaying the same 50-step modular-reference configuration exposed an MPS failure that strict
PyTorch deterministic mode does not report. Two runs are identical through step 19, differ by
approximately `1e-7` at step 20, and then amplify that perturbation through adversarial training.
At step 50, generator tensors differ by up to `0.063`. A clean ACGAN replay without a distribution
regularizer remains exactly identical, while a classical RBF distribution loss also diverges.
This localizes the problem to the MPS backward path across the CPU regularizer boundary rather
than to quantum randomness.

Two independent 30-step CPU runs had byte-identical logs and exactly identical generator and
discriminator tensors. Switching from warning-only to strict deterministic mode and synchronizing
MPS after every optimizer step did not fix MPS. Moving gradient-norm reductions to CPU only delayed
the first difference. Online gradient balancing amplified the perturbation, but a fixed-weight
log-KDE hybrid still diverged by step 6 because its narrow density geometry is sensitive to tiny
image changes.

Consequently, MPS outputs in the earlier tables are exploratory trajectory samples, not exact
replays. New confirmatory work must use CPU or a separately replay-verified deterministic CUDA
environment. The runner records this limitation in provenance and strict deterministic mode is
now enabled for known unsupported operations.

## Candidate 4: centered modular contrast over a log-KDE base

### Mechanism correction

The fixed modular reference has two newly isolated failure modes. First, its target-only energy
does not require the requested class to beat any competing class. Second, at angle scale
`0.125*pi`, real class states have mean purity approximately `0.98`; centered modular gradients
are therefore almost fidelity-prototype gradients. Alternative quantum divergences have gradient
cosines around `0.98` to `0.99` with the same direction and do not repair the problem.

The replacement removes the identity component of every real-data modular Hamiltonian, normalizes
the remaining observable to unit Frobenius norm, and uses all ten negative energies as
cross-entropy logits. A spatial mapping assigns one `2 x 2` image quadrant to each qubit in CNOT
ring order. The frozen scale is `0.75*pi`, and the reference bank contains 1,024 images per class.

On held-out real MNIST images, the calibrated full quantum score classifies 72.20%, compared with
58.53% for the same rotations without the CNOT ring and 39.43% after dephasing. The circuit's full
density-map Jacobian has rank 16, the product map rank 8, and the dephased probability map rank 12;
the final four RZ angles are exactly dead only after dephasing. The useful geometry therefore
depends causally on both entanglement and coherence.

The strong classical control is not the earlier raw RBF mean. It is a stable class-conditional log
kernel density on the same pooled input and references, with squared bandwidth `0.03125`. It
classifies real MNIST at 77.72%, so it is stronger than the standalone quantum teacher. On a
training-only calibration split, adding 5% of the quantum logits slightly improves log-KDE
negative log likelihood; product and dephased additions receive zero optimal mixture weight. This
freezes the hybrid at `0.95 * KDE + 0.05 * quantum`, without claiming that the tiny calibration
gain is practically meaningful.

### Deterministic 200-step development result

All rows below use CPU, 5,000 evaluation samples, 1,024 references per class, and one fixed outer
weight `2e-5`. No online gradient balancing is used.

| Seed | Variant | Accuracy | FID | Class-FID | Precision | Recall | Diversity |
|---:|---|---:|---:|---:|---:|---:|---:|
| 42 | ACGAN | 0.5404 | 0.2793 | 0.3563 | 0.4868 | 0.1254 | 0.6561 |
| 42 | Log-KDE | 0.5658 | 0.3161 | 0.3955 | 0.4982 | 0.0398 | 0.6130 |
| 42 | Full hybrid | 0.5834 | 0.2932 | 0.3706 | 0.4610 | 0.0472 | 0.6342 |
| 42 | Product hybrid | 0.5748 | 0.3174 | 0.3950 | 0.4270 | 0.0182 | 0.5915 |
| 42 | Dephased hybrid | 0.5142 | 0.2858 | 0.3659 | 0.4774 | 0.0722 | 0.6643 |
| 43 | Log-KDE | 0.7720 | 0.4844 | 0.5343 | 0.5720 | 0.0032 | 0.4590 |
| 43 | Full hybrid | 0.8006 | 0.4002 | 0.4515 | 0.4418 | 0.0044 | 0.5023 |

The full circuit improves conditional accuracy, FID, class-FID, recall, and diversity over log-KDE
on both development seeds, while reducing precision. Product does not reproduce the balanced
improvement. Dephasing improves distribution coverage but sharply harms conditioning, indicating
that the full circuit's narrow contribution is class separation rather than generic image quality.
This is encouraging development evidence only. The complete frozen held-out sequence and stop
rule are specified in [contrastive_reference_protocol.md](contrastive_reference_protocol.md).

### Frozen Stage 1 outcome

The untouched seeds 101 and 202 were run exactly once at the frozen 200-step horizon before any
candidate change. The paired effects reverse sign:

| Seed | Accuracy, KDE | Accuracy, hybrid | Class-FID, KDE | Class-FID, hybrid |
|---:|---:|---:|---:|---:|
| 101 | 0.7192 | 0.7358 | 0.3471 | 0.3662 |
| 202 | 0.7498 | 0.7486 | 0.3129 | 0.3055 |
| **Mean** | **0.7345** | **0.7422** | **0.3300** | **0.3358** |

The hybrid improves mean conditional accuracy by 0.00770 but worsens mean class-FID by 0.00585.
The frozen rule required both an accuracy improvement and a class-FID reduction. Candidate 4
therefore fails Stage 1, and its product/dephased controls and later seeds are not run. Exact
metrics and artifact digests are in
[contrastive_stage1_results.json](contrastive_stage1_results.json).

### Hardware interpretation

The improved Hamiltonians are dense. Aggregated squared Pauli-coefficient mass is 4.2% at weight
one, 46.5% at weight two, 28.3% at weight three, and 20.9% at weight four. Approximately 185 Pauli
strings are needed for 95% of the norm and 233 for 99%. Restricting to at most two-local terms
reduces real-image classification from 72.20% to 55.95%.

An ungrouped shot model needs roughly 25,500 shots per image, or 100 shots for each nonidentity
Pauli string, to approach 67% classification on the inspected subset. Training would multiply this
cost by repeated gradient evaluations. The candidate is therefore a simulator-based inductive-bias
study, not a near-term hardware or VQE speedup claim.

## Current conclusion

The experiments support four statements:

1. A real-data-anchored quantum loss can regularize this ACGAN and can improve individual
   trajectories, including a held-out screening seed.
2. The original target-only modular loss is not robust and is explained largely by prototype
   fidelity, reference stability, and gradient scaling.
3. A centered class-contrastive score exposes a small development-set contribution that disappears
   under product and dephased removals, but it fails the frozen held-out class-FID gate against the
   strong log-KDE control.
4. With four noiselessly simulated qubits, every tested quantum layer is a small differentiable
   classical computation. These experiments can study inductive bias, but they cannot establish a
   computational quantum advantage.

The honest result for the present architecture is therefore **no robust demonstrated
quantum-specific benefit**. Candidate 4 is rejected under its frozen rule; further tuning on its
inspected seeds is prohibited.

## Defensible next decision

There are now three coherent research directions:

- Treat the corrected work as a rigorous negative benchmark. Freeze the original four-way
  ablation, run enough paired seeds to quantify equivalence, and make the causal-audit methodology
  the contribution.
- Treat Candidate 4 as a completed falsification result. Its accuracy/class-FID trade-off can be
  analyzed descriptively, but it cannot be rescued by retuning on seeds 101 or 202. A materially
  new mechanism would require a new development split and a new untouched test set.
- Start a genuinely new study in which the quantum resource is operational rather than a
  four-qubit simulator feature map: a circuit family and qubit count that are not cheaply emulated,
  an explicit hardware/noise model, and a compute- or sample-efficiency claim against strong
  classical kernels. That is a new experiment, not a cosmetic repair of the current one.

The completed frozen falsification pair can be reproduced without changing the default four-way
matrix:

```bash
uv run python scripts/run_seed_matrix.py \
  --seed 101 \
  --output-root runs/contrastive-heldout-seed-101 \
  --dataset-root data \
  --classifier mnist_classifier.pth \
  --epochs 1 --max-steps 200 --evaluation-samples 5000 \
  --device cpu \
  --quantum-device cpu \
  --variants classical_log_kde_contrastive hybrid_modular_kde_contrastive
```

Repeat with seed 202 and a distinct output root for the second pair. The remaining frozen seeds and
causal ablations are intentionally absent because Stage 1 failed.

Every run records the balanced reference size, generator label source, device placement, source
revision, configuration, and checkpoint digest. The development output directories used to design
the candidate are not treated as confirmatory artifacts. The frozen Stage 1 summary and checkpoint
digests are retained separately.

## Related primary literature

- [Data re-uploading for a universal quantum classifier](https://arxiv.org/abs/1907.02085)
- [The power of data in quantum machine learning](https://www.nature.com/articles/s41467-021-22539-9)
- [Exponential concentration in quantum kernel methods](https://www.nature.com/articles/s41467-024-49287-w)
- [Trainability barriers in low-depth quantum generative modelling](https://www.nature.com/articles/s41534-024-00902-0)
- [Benchmarking variational quantum algorithms for machine learning](https://arxiv.org/abs/2403.07059)
- [Quantum modular Hamiltonian learning](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.108.062422)
