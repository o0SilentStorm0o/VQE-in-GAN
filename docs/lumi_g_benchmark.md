# LUMI-G runtime benchmark

## Purpose and resource limit

The benchmark estimates the wall time and allocation cost of the frozen four-variant, ten-seed
experiment. It used one MI250X GCD in `dev-g`, four requested CPU cores, 16 GB of memory, and a
two-minute hard wall-time limit per submitted probe. No full experiment was submitted.

[LUMI accounts one GCD-hour as 0.5 GPU hour](https://docs.lumi-supercomputer.eu/runjobs/lumi_env/billing/).
A [complete LUMI-G node exposes eight GCDs](https://docs.lumi-supercomputer.eu/hardware/lumig/), so
eight GCD-hours equal one equivalent node-hour.

## Environment

- PyTorch: `2.10.0+rocm7.0`
- HIP: `7.0.51831`
- Device: one AMD Instinct MI250X GCD
- Container: `lumi-multitorch-full-u24r70f21m50t210-20260513_121430.sif`
- Batch size: 64
- Dynamic regularizer-to-GAN gradient ratio: 0.10
- Quantum execution device: CPU bridge

The tested image is one of the official
[LUMI AI Factory PyTorch containers](https://docs.lumi-supercomputer.eu/laif/software/ai-environment/).
The Torch statevector backend does not require Qiskit at training time. Provenance collection uses
installed package metadata without importing Qiskit so that the system ROCm PyTorch build remains
untouched.

## Steady-state complete-step results

Each result is the median of 20 interleaved complete discriminator-plus-generator steps after three
warm-up steps. Synchronization surrounds every measured step.

| Variant | Median step | Standard deviation |
|---|---:|---:|
| No regularizer | 14.313 ms | 0.273 ms |
| Classical prototype | 21.409 ms | 0.319 ms |
| Quantum contrastive, CPU bridge | 31.213 ms | 0.518 ms |
| Quantum permuted, CPU bridge | 31.453 ms | 0.558 ms |
| Quantum contrastive, GCD statevector | 38.638 ms | 0.811 ms |
| Quantum permuted, GCD statevector | 38.902 ms | 0.786 ms |

The CPU bridge is about 19% faster than executing the small statevector directly on the GCD. Full
runs therefore use `--quantum-device cpu` while the ACGAN remains on the GCD.

## End-to-end result

A second job trained all four variants for 300 real MNIST steps each, wrote per-step JSONL logs and
final checkpoints, and evaluated one checkpoint on 5,000 balanced samples. The Python workload took
105.031 seconds, including 102.005 seconds for the four training runs and 3.026 seconds for the
evaluation. The complete Slurm allocation, including container and Python startup, took 113 seconds.

The difference between the 300-step steady-state time and end-to-end training time is 72.488 seconds.
It includes one-time MIOpen kernel preparation, dataset/model construction, provenance, Lustre log
I/O, and checkpoint writes. The full-run projection retains this complete fixed overhead rather than
extrapolating only the faster steady-state steps.

## Frozen full-run projection

MNIST supplies 937 complete batches per epoch at batch size 64, or 9,370 steps over ten epochs. One
Slurm task will run all four variants for one paired seed in a single process. Models, optimizers,
data loaders, and RNG state are recreated and reseeded for every variant; only the compiled MIOpen
cache is reused. Ten array tasks cover all ten frozen seeds.

The measured projection for one seed task is:

```text
9,370 * (14.313 + 21.409 + 31.213 + 31.453) ms  = 921.890 s training steps
+ 72.488 s measured fixed training overhead
+ 4 * 3.026 s evaluation
+ 7.969 s container/Python startup
= 1,014.452 s = 16.908 min
```

Consequently, the ten-seed matrix is expected to consume:

- 2.818 GCD-hours;
- 1.409 billed GPU hours;
- 0.352 equivalent full-node hours.

With the conservative array cap of eight simultaneous tasks, compute wall time after scheduling is
approximately 33.8 minutes because the ten tasks execute in two waves. Allowing all ten tasks at once
would reduce it to approximately 16.9 minutes without changing expected billed usage. Queue waiting
is not billed and cannot be predicted from this benchmark.

The production script has a 25-minute hard limit per seed task. Even if every task reached that
limit, the array could consume at most 2.083 billed GPU hours, or 0.521 equivalent node-hours. The
full array must be submitted explicitly with an allocation account; the repository does not embed
an organization-specific project identifier.

## Probe accounting

The five short diagnostic allocations used 212 GCD-seconds in total. This equals 0.0294 billed GPU
hours or 0.00736 equivalent node-hours. Two probes measured the workload successfully; the other
three exited early while deployment paths and the per-job MIOpen cache were being validated.

## Production submission

After staging the repository, MNIST files, and classifier checkpoint in scratch, submit with:

```bash
sbatch -A <project> scripts/lumi_run_array.sbatch
```

The script pins the tested container, runs the ten frozen seeds, keeps at most eight GCDs active,
and evaluates every final checkpoint on 5,000 balanced samples.
