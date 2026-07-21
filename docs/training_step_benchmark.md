# Full training-step benchmark

## Scope

This benchmark measures one complete discriminator update followed by one complete generator
update. It therefore includes the shared generator, discriminator, backward passes, Adam updates,
angle head, statevector simulation, all-class energy matrix, and contrastive loss.

The benchmark uses batch size 64, five warm-up iterations, and the median of 20 interleaved timed
iterations. Device synchronization surrounds every measurement. It was recorded on Apple M5 with
PyTorch 2.8.0.

## Results

| Training device | Quantum device | Median D+G step | Overhead vs no regularizer |
|---|---:|---:|---:|
| MPS | none | 64.20 ms | — |
| MPS | MPS | 103.29 ms | 39.09 ms / 60.9% |
| MPS | CPU | 75.32 ms | 11.12 ms / 17.3% |
| CPU | none | 712.82 ms | — |
| CPU | CPU | 713.89 ms | 1.07 ms / 0.15% |

The CPU comparison used three warm-up and ten measured iterations. It is included to show that the
four-qubit calculation itself is small relative to CNN training. On MPS, many small complex-valued
operations incur disproportionate dispatch overhead. Moving only the statevector backend to CPU
reduces quantum overhead by approximately 3.5x and the complete quantum step by approximately 27%
relative to running all operations on MPS.

Device transfers remain differentiable: circuit angles move from MPS to CPU and the $B\times10$
energy matrix moves back to MPS without detaching autograd. Automated tests verify energy and
gradient preservation through the bridge.

## Reproduction

```bash
uv run python benchmarks/benchmark_training_step.py \
  --device mps --quantum-device both --batch-size 64 --warmup 5 --repeats 20

uv run python benchmarks/benchmark_training_step.py \
  --device cpu --quantum-device same --batch-size 64 --warmup 3 --repeats 10
```

`quantum_device=auto` in the training CLI selects CPU when the training device is MPS and otherwise
uses the training device.

## Smoke-run determinism

Two independent four-step MNIST runs on MPS with the CPU quantum bridge, batch size 16, dataset
limit 64, and seed 42 produced byte-identical `metrics.jsonl` files. Their shared SHA-256 digest
was `65e1735229d60842930590000effe7de48343698117b24941d7a50ef57c8755a`.
