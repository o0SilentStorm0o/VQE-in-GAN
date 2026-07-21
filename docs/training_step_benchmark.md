# Full training-step benchmark

## Scope

This benchmark measures one complete discriminator update followed by one complete generator
update in the final image-conditioned architecture. It includes backward passes, Adam updates, the
image-to-angle head, statevector simulation, all-class contrastive energy, and dynamic shared-path
gradient balancing at ratio 0.10.

The benchmark uses batch size 64, three warm-up iterations, and the median of ten interleaved timed
iterations. Device synchronization surrounds each measurement. It was recorded on Apple M5 with
PyTorch 2.8.0.

## Results

| Training device | Quantum device | Median D+G step | Overhead vs no regularizer |
|---|---:|---:|---:|
| MPS | none | 91.00 ms | — |
| MPS | MPS | 196.59 ms | 105.60 ms / 116.0% |
| MPS | CPU | 160.20 ms | 69.20 ms / 76.1% |

The CPU bridge reduces the complete quantum step by 18.5% relative to native MPS execution. Most
remaining overhead comes from the extra gradient measurements required to hold regularizer strength
constant, rather than from the four-qubit statevector alone.

The 500-step end-to-end pilot, which also includes data loading and periodic diagnostics, measured
53.0 s for no regularizer, 77.3 s for the classical control, 91.4 s for quantum contrastive, and
77.9 s for permuted quantum. These timings are engineering measurements, not quality results.

Device transfers remain differentiable: circuit angles move from MPS to CPU and the $B\times10$
energy matrix moves back to MPS without detaching autograd. Automated tests verify energy and
gradient preservation through the bridge.

## Reproduction

```bash
uv run python benchmarks/benchmark_training_step.py \
  --device mps \
  --quantum-device both \
  --batch-size 64 \
  --warmup 3 \
  --repeats 10 \
  --gradient-ratio 0.1
```

`quantum_device=auto` in the training CLI selects CPU when the training device is MPS and otherwise
uses the training device.

## Determinism

Two independent four-step runs of the final image-conditioned quantum configuration, including
dynamic gradient balancing and per-step diagnostics, produced byte-identical `metrics.jsonl`
files. Their shared SHA-256 digest was
`e6e7d27b7bf9e5605447bdd769e42f2a6c2c3e86a615a512a10ed86c25b307e3`.
