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
runner and model definitions and skips backend construction. A cross-variant data-order regression
test remains pending before the ablation protocol is frozen.

## Stage 3: smoke run and preregistered experiment

A short MNIST subset run will validate determinism, logging, checkpointing, and metric generation.
The full hypotheses, equivalence bounds, seed list, metrics, and exclusion criteria must be frozen
before full-scale runs begin. Exploratory analyses must be labeled separately from confirmatory
tests.

**Current status:** CPU and MPS/CPU-bridge smoke runs complete successfully. Repeated MPS runs with
the same seed produced byte-identical structured metrics. Configuration, source provenance,
dependency-lock digest, device placement, metrics, and atomic checkpoints are recorded per run.
A 500-step exploratory pilot rejected both a direct feature-map angle head and fixed regularizer
weighting. The corrected image-conditioned pilot and a candidate confirmatory protocol are recorded
in `pilot_500_steps.md` and `ablation_protocol_draft.md`. Training duration, exact statistical
tests, and equivalence bounds are not yet frozen.
