# Ablation protocol draft

## Status

This protocol is a draft informed by diagnostic pilots. It is not yet preregistered and must not
be used to label exploratory results as confirmatory. The training duration and final seed list
remain open until a convergence pilot is completed.

Subsequent real-data-anchored quantum candidates and matched classical controls are recorded in
`quantum_utility_audit.md`. None of the original candidates passed the quantum-utility gate, so
this draft must not be extended with a selected positive variant based on the inspected seeds. A
new all-class contrastive study has its own frozen development/held-out boundary and stop rule in
`contrastive_reference_protocol.md`; it does not retroactively change this original four-way
negative/equivalence protocol.

## Experimental variants

All variants instantiate the same generator, image-conditioned angle head, discriminator, data
order, optimizer settings, initialization, and evaluation pipeline.

1. `no_regularizer`: ordinary ACGAN; the angle head is present but inactive and no energy model is
   constructed.
2. `quantum_contrastive`: class-encoded four-qubit Ising energy matrix with contrastive energy
   loss.
3. `classical_prototype`: parameter-free periodic distances from the same 16 angles to ten
   deterministic class prototypes.
4. `quantum_permuted`: the same quantum computation as `quantum_contrastive`, with a fixed cyclic
   permutation of class-energy columns.

Every energy model has zero trainable parameters. Automated tests require identical generator and
discriminator state structure and parameter counts across all four variants.

## Regularizer strength

Using a shared scalar coefficient is not a fair comparison because the three energy landscapes
have different Jacobian scales and their scales change during training. A fixed initialization
calibration was also rejected after a diagnostic run showed that a nominal 10% contribution could
grow above 180% within 100 steps.

For every regularized generator step, let $g_{GAN}$ be the gradient of adversarial plus auxiliary
loss and $g_R$ the unweighted regularizer gradient, both restricted to the common image-producing
path. The detached effective coefficient is

$$
\lambda_t = \rho\frac{\lVert g_{GAN}\rVert_2}{\lVert g_R\rVert_2},
\qquad \rho=0.10.
$$

The regularizer direction is unchanged, while its norm on shared parameters is exactly 10% of the
GAN objective norm. The effective coefficient and achieved ratio are logged every step. The angle
head receives the same coefficient but is excluded from norm matching because it has no GAN
gradient.

## Candidate hypotheses

- Primary: `quantum_contrastive` lowers class-conditional MNIST feature-FID relative to
  `no_regularizer`.
- Secondary: `quantum_contrastive` lowers class-conditional feature-FID relative to
  `classical_prototype`.
- Mechanistic: the correctly aligned quantum Hamiltonian outperforms `quantum_permuted`.

Failure of the mechanistic comparison would mean that any observed gain cannot be attributed to
the proposed class-to-Hamiltonian alignment.

## Candidate metrics

The proposed primary metric is class-conditional feature-FID. The frozen independent MNIST
classifier supplies 64-dimensional convolutional features; FID is computed independently for each
requested digit and averaged equally across the ten classes.

Secondary metrics are:

- conditional accuracy from the frozen classifier;
- global 64-dimensional MNIST feature-FID;
- feature-manifold precision and recall with three neighbors;
- generated within-class feature diversity and its ratio to real MNIST diversity;
- regularizer/GAN gradient cosine and effective regularizer coefficient.

The classifier checkpoint must achieve at least 98% accuracy on the balanced real evaluation
sample. Evaluation should use 5,000 generated and 5,000 real images, exactly 500 per class, from a
fixed evaluation seed that is not used during training.

## Reproducibility and exclusions

Candidate full-run seeds are `42, 314, 729, 2025, 4096, 5003, 65537, 9001, 12011, 16001`.
Ten paired seeds permit more informative uncertainty estimates than the earlier five-seed plan.
Final seeds must be frozen before full runs.

A run may be excluded only for a recorded infrastructure failure, corrupt checkpoint, or non-finite
training value. A failed infrastructure run must be repeated with the same variant and seed. Runs
must not be excluded for poor image metrics, instability that remains finite, or disagreement with
the hypothesis.

The primary analysis will use paired per-seed differences, bootstrap confidence intervals, and an
exact paired randomization test. Secondary pairwise claims require multiplicity correction. Exact
tests and equivalence bounds will be finalized before confirmatory training.
