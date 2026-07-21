# Corrected experiment protocol

## Status

This directory contains the redesign of the simulator-based ACGAN experiment. The original
notebooks and result artifacts remain unchanged as a historical record. New results must be
produced only by the modular implementation under `src/` and must record the exact configuration,
dependency lock file, random seed, source revision, and backend.

## Stage 1: quantum backend parity and performance

Before changing the GAN architecture, the quantum energy computation must satisfy all of the
following gates:

1. The Qiskit reference backend and the batched Torch statevector backend evaluate the same
   four-qubit `EfficientSU2` circuit and class-conditioned Ising Hamiltonian.
2. Forward energies agree within a documented numerical tolerance on random and boundary inputs.
3. Gradients with respect to every circuit angle agree within a documented numerical tolerance.
4. Batch ordering and class conditioning are covered by automated tests.
5. Forward and backward performance is measured for batch sizes 1, 32, and 64 after warm-up.

No GAN training result is considered valid until these gates pass.

**Current status:** all five gates pass. Reproduction commands and timings are recorded in
`quantum_backend_benchmark.md`.

## Stage 2: corrected ACGAN computation graph

The corrected generator will derive circuit angles from the completed generated image. Automated
tests must demonstrate that:

- the energy loss has a non-zero gradient with respect to shared generator parameters;
- a step using only the energy loss changes the image-producing path;
- the quantum backend is not invoked during discriminator-only or evaluation passes;
- discriminator parameters receive no gradient from the generator objective;
- all ablation variants use the same image generator, discriminator, data order, and optimizer
  protocol for a given seed.

**Current status:** the shared generator and isolated optimizer steps are implemented and covered
by gradient, call-count, parameter, and BatchNorm-state tests. The architecture and computation
boundaries are recorded in `shared_generator_design.md`. The zero-regularizer variant uses the same
runner and model definitions and skips backend construction. All variants now reuse the labels of
the real minibatch for the generator update, which gives distribution losses equal per-class
real/generated counts and preserves an identical RNG stream across paired variants.

## Stage 3: smoke run and preregistered experiment

A short MNIST subset run will validate determinism, logging, checkpointing, and metric generation.
The full hypotheses, equivalence bounds, seed list, metrics, and exclusion criteria must be frozen
before full-scale runs begin. Exploratory analyses must be labeled separately from confirmatory
tests.

**Current status:** CPU and MPS/CPU-bridge smoke runs complete successfully, but longer paired MPS
replays with a CPU distribution regularizer are not deterministic. A perturbation near `1e-7` is
amplified by adversarial training even under strict deterministic mode. Independent CPU replays
have byte-identical logs and exactly identical checkpoint tensors. Configuration, source
provenance, dependency-lock digest, device placement, backend eligibility, metrics, and atomic
checkpoints are recorded per run. New distribution experiments must use CPU or a replay-verified
deterministic CUDA environment.

## Stage 4: quantum-utility gate

Before any positive hypothesis is preregistered, a quantum candidate must beat a matched classical
control with the same real-data access, pooled input, generator architecture, and shared-gradient
budget. Dephased, product-circuit, entropy-free, and label-alignment removals are mechanism checks,
not substitutes for the classical control.

**Current status:** three original real-data-anchored candidates were explored. Class-conditional
modular free energy was seed-dependent and weaker on average than RBF-MMD; an orthogonal
coherence-residual augmentation harmed the RBF baseline; and a stable modular reference bank
improved the primary class-FID screen on three seeds, including held-out seed 314, but failed
sharply on held-out seed 729. A fourth, separate candidate fixes the target-only objective with
centered all-class modular logits and tests only a 5% quantum addition over a stronger log-KDE
control. It passed product and dephased development checks on seed 42 and improved the classical
control on development seeds 42 and 43. On the frozen seeds 101 and 202 it improved mean
conditional accuracy by 0.77 percentage points but worsened mean class-FID by 0.00585. It therefore
failed the preregistered Stage 1 gate in `contrastive_reference_protocol.md`; later seeds and causal
ablations are not authorized for this candidate. The completed post-hoc diagnosis shows that the
branch is a fixed parameter-free feature map, its loss is class-relative rather than
distribution-matching, and the frozen comparison is dominated initially by a simultaneous change
in KDE logit scale. See `contrastive_failure_diagnosis.md`.
