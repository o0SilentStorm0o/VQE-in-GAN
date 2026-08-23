# VQE-in-GAN: V3 Controlled Quantum-Utility Experiments

[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](pyproject.toml)
[![Status: V3 falsification result](https://img.shields.io/badge/status-V3%20falsification%20result-orange.svg)](docs/quantum_utility_audit.md)

This repository studies whether a small differentiable quantum circuit can provide a useful and
specifically quantum inductive bias to an Auxiliary Classifier GAN (ACGAN). V3 rebuilds the
experiment around a generator-shared image path, deterministic execution, matched classical and
circuit controls, frozen stopping rules, and optimizer-level budget matching.

The repository name is historical. V3 evaluates differentiable circuit-derived energy and
similarity objectives inside GAN training; it does not run a standalone ground-state VQE solver.

## Current result

The narrow V3 conclusion is:

> A trainable image-to-angle path improved the full coherent-entangled candidate on the two
> development seeds, but the candidate did not robustly outperform both the product-circuit and
> dephased controls. The tested mechanism therefore provides no demonstrated quantum-specific
> benefit.

This is not a claim that quantum-assisted GANs cannot work. It is a falsification result for the
specific four-qubit, noiseless-statevector mechanisms tested here. No computational quantum
advantage is claimed.

The strongest V3 diagnostic used two phases, CPU seeds 42 and 43, 200 training steps per run, and
5,000 balanced evaluation samples. Lower class-FID is better; higher conditional accuracy and
diversity are better.

| Phase | Circuit variant | Mean class-FID ↓ | Mean accuracy ↑ | Mean diversity ↑ |
| --- | --- | ---: | ---: | ---: |
| A: frozen angle head | Full coherent-entangled | 0.426880 | 0.644900 | 0.557745 |
| A: frozen angle head | Product | 0.451794 | 0.616300 | 0.551010 |
| A: frozen angle head | Dephased | 0.418598 | 0.673100 | 0.562932 |
| B: trainable angle head | Full coherent-entangled | 0.402251 | 0.673600 | 0.574536 |
| B: trainable angle head | Product | 0.410949 | 0.666700 | 0.574639 |
| B: trainable angle head | Dephased | 0.420250 | 0.718300 | 0.551205 |

Phase B improved the full branch over Phase A on all three mean metrics and passed the frozen
incremental angle-head gate. It still failed the circuit-specific gate:

- product had lower class-FID on one of the two seeds;
- dephased preserved substantially higher mean conditional accuracy; and
- neither comparison satisfied every frozen per-seed and mean requirement.

All 2,400 training records passed the technical audit, including shared raw-gradient matching,
Adam auxiliary-displacement matching, schedule integrity, and the separately controlled Phase B
angle-head budget. Because neither phase passed both circuit controls, no held-out seed was
authorized or run.

See [the complete result and stop decision](docs/reference_budget_relational_results.md) and
[the full quantum-utility audit](docs/quantum_utility_audit.md).

## What changed in V3

The historical experiment attached a class-conditioned Ising energy to the GAN, but the quantum
branch did not receive a meaningful image-derived learning target. V3 makes the causal path
explicit:

```text
noise + class
      │
      ▼
unchanged ACGAN image generator ───────────────► generated MNIST image
      │                                                │
      │ shared image-producing parameters              ├─ fixed 4×4 image encoding
      │                                                └─ bounded trainable angle residual
      │                                                        │
      │                                                        ▼
      └──── regularizer gradient ◄──── relational loss ◄──── 4-qubit circuit
                                                          │
real MNIST reference images ───── data-derived projectors ┘
```

The angle head reads only the completed generated image, not the label or latent vector directly.
Regularizer gradients must reach both the 29,200-parameter angle head and the 1,762,789-parameter
image-producing path. Discriminator-only and evaluation steps never invoke the quantum backend.

The final relational objective compares generated and real samples of the same class in both
directions:

- generated-to-real support discourages samples far from every real reference;
- real-to-generated coverage makes duplicated generated states less useful; and
- the existing class-conditional log-KDE term remains an unchanged anchor.

The full circuit uses one circular CNOT ring. The product control removes entanglement, and the
dephased control removes coherence before comparison. For every generator step, the controls
replay the full branch's shared raw-gradient ratio and actual Adam auxiliary-displacement ratio.
Phase B also matches the angle-head raw-gradient norm and realized Adam displacement.

The architectural and optimization details are specified in:

- [shared generator design](docs/shared_generator_design.md);
- [Hamiltonian and circuit design](docs/hamiltonian_design.md);
- [trainable relational-coverage protocol](docs/trainable_relational_coverage_protocol.md); and
- [full-trajectory reference-budget protocol](docs/reference_budget_relational_protocol.md).

## Evidence status

| Question | V3 evidence |
| --- | --- |
| Does the Torch statevector match the Qiskit reference? | Yes, including angle gradients and batch/class ordering. |
| Does the auxiliary gradient reach generated pixels? | Yes, through the completed image and shared generator path. |
| Are optimizer and discriminator boundaries isolated? | Yes, covered by state and call-count tests. |
| Is exact replay available? | Yes on CPU; the tested MPS distribution-loss path is not replay-deterministic. |
| Can a real-data-anchored quantum loss help an individual trajectory? | Yes. |
| Did the frozen contrastive candidate pass held-out screening? | No; accuracy improved slightly while class-FID worsened. |
| Did the trainable relational candidate beat both circuit controls? | No. |
| Is coherence or entanglement established as the cause of the improvement? | No. |
| Is computational quantum advantage established? | No; four noiseless qubits are cheaply simulated. |

Earlier candidates and their failure analyses remain part of the audit trail. In particular, the
fixed contrastive hybrid improved mean conditional accuracy by 0.00770 on frozen seeds 101 and
202, but worsened mean class-FID by 0.00585 and failed its preregistered joint gate. The subsequent
relational candidate fixed the missing trainable angle path and the optimizer-budget confound, but
still failed quantum-specific attribution.

## Reproduce the code

### Requirements

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)
- CPU execution for evidence-grade deterministic replay

### Install and verify

```bash
git clone https://github.com/o0SilentStorm0o/VQE-in-GAN.git
cd VQE-in-GAN
git switch experiment-v3-rebuild
uv sync --extra dev
uv run ruff check .
uv run pytest -q
```

The committed `uv.lock` is authoritative for V3.

### Quantum backend parity and timing

```bash
uv run python benchmarks/benchmark_quantum_backends.py \
  --backend all \
  --batch-sizes 1 32 64
```

### Minimal corrected smoke run

```bash
uv run vqe-gan-train \
  --run-name quantum-smoke \
  --dataset-limit 128 \
  --batch-size 64 \
  --max-steps 2 \
  --device cpu \
  --quantum-device cpu
```

### Full-trajectory matched-budget diagnostic

Run and audit Phase A before starting Phase B:

```bash
uv run python scripts/run_reference_budget_relational.py \
  --phase a \
  --output-root runs/reference-budget \
  --dataset-root data \
  --classifier mnist_classifier.pth

uv run python scripts/audit_reference_budget_relational.py \
  --phase a \
  --output-root runs/reference-budget

uv run python scripts/run_reference_budget_relational.py \
  --phase b \
  --output-root runs/reference-budget \
  --dataset-root data \
  --classifier mnist_classifier.pth

uv run python scripts/audit_reference_budget_relational.py \
  --phase b \
  --output-root runs/reference-budget
```

These scripts intentionally fix the evidence-grade configuration. The audit never authorizes a
held-out run automatically.

## Repository layout

```text
src/vqe_gan/       modular ACGAN, quantum backends, regularizers, training, evaluation
tests/             backend parity, gradient flow, fairness, budget, and isolation tests
scripts/           frozen experiment runners, calibrations, audits, and LUMI job templates
benchmarks/        quantum-backend and end-to-end timing tools
docs/              protocols, calibration records, result summaries, and causal diagnoses
runs/              raw V3 outputs and checkpoints (local/release snapshot, not normal Git)
data/              local MNIST cache (not normal Git)
*.ipynb            historical V1/V2 notebooks retained for traceability
ablation_results/  historical V2 classical-ablation artifacts
```

The complete local research snapshot, including ignored raw runs, checkpoints, data, environment,
preprint, and caches, is documented in
[the 2026-08-24 snapshot manifest](docs/research_snapshot_2026-08-24.md). Git-tracked protocols and
summaries remain readable without downloading that archive.

## V1/V2 historical material

The root notebooks, original result directories, and classical ablation belong to the historical
paper experiment. They are retained unchanged for provenance. Their conclusion is narrower than
some wording in the old README suggested: the tested fixed VQE-inspired regularizer did not show a
measurable benefit over its classical alternatives. That result does not prove a universal
impossibility.

V3 is a separate corrective experiment. It does not silently replace historical results, and raw
artifact provenance continues to record the source revisions under which each run was produced.
Because contributor metadata was corrected without changing file trees, historical and current
V3 revision hashes are listed in
[the identity rewrite map](docs/revision_identity_map_2026-08-24.md).

## Scope and limitations

- MNIST and one ACGAN host architecture only.
- Four noiselessly simulated qubits; no real quantum-hardware validation.
- Development diagnostics use two seeds and 200 steps, not a powered confirmatory study.
- One frozen MNIST feature classifier supplies evaluation features and class predictions.
- Matched Adam control trajectories are deliberate causal interventions, not ordinary
  unconstrained training.
- MPS was suitable for timing and smoke tests but not accepted for exact paired replay.
- No claim of novelty should be made without a dedicated, current literature review.

## Citation

The existing preprint and its V2 result refer to the historical experiment, not to the unfinished
V3 study:

```bibtex
@misc{strnadel2025vqegan,
  title  = {Differentiable Energy-Based Regularization in GANs:
            A Simulator-Based Exploration of VQE-Inspired Auxiliary Losses},
  author = {Strnadel, David},
  year   = {2025},
  note   = {V2 preprint with classical ablation study}
}
```

A separate V3 citation should be added only after its manuscript, version, and permanent identifier
are frozen.

## Author and license

David Strnadel (GitHub: [o0SilentStorm0o](https://github.com/o0SilentStorm0o))<br>
Contact: [davidstrnadel@seznam.cz](mailto:davidstrnadel@seznam.cz)

Released under the [MIT License](LICENSE).
