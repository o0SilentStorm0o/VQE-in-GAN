# Full-trajectory reference-budget relational results

## Status

Both frozen development phases completed and passed every technical audit. Neither phase passed
the circuit-specific outcome gate, so no held-out seed is authorized or was run. Phase B did pass
its separate incremental test against the Phase A full branch: training the angle head improved
the full candidate, but it did not turn the coherent-entangled circuit into a robust winner over
both circuit controls.

The protocol and all pre-outcome numerical amendments are recorded in
[`reference_budget_relational_protocol.md`](reference_budget_relational_protocol.md). The raw run
artifacts are under `runs/reference-budget/` and are intentionally excluded from version control.

## Frozen comparison

Each phase used development seeds 42 and 43, 200 CPU training steps, 5,000 balanced evaluation
samples, and one step-specific schedule recorded by the full coherent-entangled branch. Product
and dephased controls matched the full branch's shared raw-gradient ratio and Adam auxiliary
displacement ratio. Phase B additionally matched the angle-head raw-gradient norm and actual Adam
displacement norm. Lower class-conditional feature FID is better; higher conditional accuracy and
diversity ratio are better.

The circuit gate required the full branch to satisfy every condition against both controls:

1. lower class-FID on each seed and mean class-FID lower by at least `0.005`;
2. mean diversity no more than `0.005` lower; and
3. mean conditional accuracy no more than `0.005` lower.

## Phase A: frozen angle head

| Seed | Variant | Class-FID | Accuracy | Diversity |
|---:|---|---:|---:|---:|
| 42 | Full coherent-entangled | 0.373392 | 0.564800 | 0.621285 |
| 42 | Product | 0.373979 | 0.560200 | 0.624092 |
| 42 | Dephased | 0.366645 | 0.557800 | 0.639766 |
| 43 | Full coherent-entangled | 0.480367 | 0.725000 | 0.494205 |
| 43 | Product | 0.529609 | 0.672400 | 0.477927 |
| 43 | Dephased | 0.470551 | 0.788400 | 0.486098 |
| — | **Full mean** | **0.426880** | **0.644900** | **0.557745** |
| — | Product mean | 0.451794 | 0.616300 | 0.551010 |
| — | Dephased mean | 0.418598 | 0.673100 | 0.562932 |

Full passed all four checks against product. It failed all four against dephased: dephased had
lower class-FID on both seeds and better mean class-FID, accuracy, and diversity. The Phase A
circuit gate therefore failed.

## Phase B: trainable, separately budgeted angle head

| Seed | Variant | Class-FID | Accuracy | Diversity |
|---:|---|---:|---:|---:|
| 42 | Full coherent-entangled | 0.342407 | 0.566200 | 0.653641 |
| 42 | Product | 0.328683 | 0.551600 | 0.670722 |
| 42 | Dephased | 0.361122 | 0.587800 | 0.635036 |
| 43 | Full coherent-entangled | 0.462096 | 0.781000 | 0.495431 |
| 43 | Product | 0.493214 | 0.781800 | 0.478556 |
| 43 | Dephased | 0.479377 | 0.848800 | 0.467374 |
| — | **Full mean** | **0.402251** | **0.673600** | **0.574536** |
| — | Product mean | 0.410949 | 0.666700 | 0.574639 |
| — | Dephased mean | 0.420250 | 0.718300 | 0.551205 |

Against product, full improved mean class-FID by `0.008697` and preserved mean accuracy and
diversity, but product had lower class-FID on seed 42. Against dephased, full had lower class-FID on
both seeds, improved mean class-FID by `0.017998`, and had higher diversity, but its mean accuracy
was `0.044700` lower. Consequently neither control comparison passed every frozen condition and
the Phase B circuit gate failed.

### Incremental angle-head result

Relative to the Phase A full branch, the Phase B full branch changed its mean metrics by:

| Metric | Phase A full | Phase B full | Phase B minus Phase A |
|---|---:|---:|---:|
| Class-FID | 0.426880 | 0.402251 | -0.024629 |
| Accuracy | 0.644900 | 0.673600 | +0.028700 |
| Diversity | 0.557745 | 0.574536 | +0.016791 |

This passes the preregistered incremental angle-head gate. It shows that the generator-shared,
trainable angle path is useful for this full candidate under the tested budget. It does not by
itself establish a coherence-specific or quantum advantage, because the full candidate still
failed the circuit controls.

## Technical audit

| Check | Phase A maximum | Phase B maximum | Frozen limit |
|---|---:|---:|---:|
| Shared raw-ratio relative error | 2.530e-7 | 2.177e-7 | 1e-5 |
| Shared Adam-ratio relative error | 9.986e-5 | 9.996e-5 | 1e-4 |
| Adam proposal relative error | 0 | 0 | 1e-5 |
| Angle raw-norm relative error | — | 2.345e-7 | 1e-5 |
| Angle update relative error | — | 2.927e-7 | 1e-4 |
| Shared quantization moves | 90 | 210 | 512 |
| Shared quantization perturbation | 0.018397 | 0.019936 | 0.02 |
| Angle quantization moves | — | 0 | 512 |

All 2,400 training records were present, schedules were consumed exactly once and in order,
schedule hashes matched, tracked sources were clean, and all full reference updates used ordinary
Adam without quantization repair. The untracked-source flag in provenance refers only to local
user files outside the experiment outputs.

| Artifact | Source revision | SHA-256 |
|---|---|---|
| Phase A matrix | `245b7003f207c403faebd7df3b123df85f0c1f04` | `ca8c2e25a4c106ed2ada400e2abbb3a0c7521c3276345e60ef745497169fd65d` |
| Phase A audit | `245b7003f207c403faebd7df3b123df85f0c1f04` | `51db7a5d6a6a27e283e1a787160fdaf6bd6ecf65770003010042950a41e4afed` |
| Phase B matrix | `9af0f39eb5cedebbd13b8d8bd38053cea777e6c0` | `0d668adacd3a6af315197f8c19f91dd7dd888ce35392cc391413ad1d1495161c` |
| Phase B audit | `9af0f39eb5cedebbd13b8d8bd38053cea777e6c0` | `e3926549930581a6c159819607acf547d7b126c5107817bee21ffc8206f127b9` |

## Interpretation and stop decision

The matched-budget evidence is more informative than the earlier fixed-coefficient ablation:

- Full consistently outperforming product in Phase A class-FID, and on the Phase B mean, indicates
  that the factorized product geometry loses useful structure.
- Dephased outperforming full in Phase A rejects a simple claim that coherence itself supplies the
  useful fixed-map signal.
- In Phase B, full improves distribution metrics over dephased but pays a large accuracy cost. This
  is a trade-off, not a robust Pareto improvement.
- The trainable angle head is beneficial to the full candidate, but the present experiment cannot
  attribute that benefit uniquely to quantum coherence or entanglement.

The correct decision for this candidate is therefore to stop: do not run held-out seeds and do not
tune the coefficient on those seeds. A future candidate would need a newly frozen hypothesis and
controls, rather than a post-hoc relaxation of this gate.

## Limitations

- This is a two-seed, 200-step development diagnostic, not a powered confirmatory study. Its gate
  can reject the candidate but cannot establish population-level equivalence between circuits.
- Matching the realized optimizer budget requires deterministic post-Adam control displacements.
  Adam's moments still receive the raw-matched control gradient, but the resulting control
  trajectory is an experimental intervention rather than ordinary unconstrained training.
- Evaluation depends on one frozen MNIST feature classifier and a finite 5,000-sample estimate.
- The four-qubit noiseless statevector is classically inexpensive. Any observed inductive-bias
  effect would not constitute computational quantum advantage.
