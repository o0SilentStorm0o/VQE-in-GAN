# Quantum backend benchmark

This benchmark measures a complete forward and backward energy pass. It excludes backend
construction and includes warm-up iterations. The command is:

```bash
uv run python benchmarks/benchmark_quantum_backends.py --backend all --batch-sizes 1 32 64
```

## Initial result

Environment: Apple M5, macOS arm64, Python 3.12.13, PyTorch 2.8.0, Qiskit 1.4.6,
Qiskit Machine Learning 0.8.4. Values are median milliseconds per complete step.

| Backend | Batch 1 | Batch 32 | Batch 64 |
|---|---:|---:|---:|
| Batched Torch statevector | 2.48 | 2.34 | 2.06 |
| Qiskit reverse gradient | 11.95 | 406.84 | 731.51 |
| Qiskit parameter shift | 24.90 | 784.70 | 1383.44 |

At batch 64, the Torch backend is approximately 355x faster than the reverse-gradient Qiskit
reference and 671x faster than the batched parameter-shift reference on this machine. These
ratios are environment-specific and do not predict end-to-end GAN training time. The historical
notebooks were slower still because they invoked `TorchConnector` separately for every sample
and evaluated the quantum branch in code paths that discarded its output.

## Correctness gate

The benchmarked Torch implementation is accepted only while the parity suite confirms that its
statevectors, class-conditioned energies, and angle gradients match the Qiskit reference. Run:

```bash
uv run pytest -q
```
