# VQE-in-GAN: Exploratory Integration of VQE-Inspired Energy Terms in GANs

> **Experiment redesign in progress:** The original notebooks and result artifacts are retained as
> a historical record. A modular, parity-tested implementation is being developed under `src/`,
> with the corrected protocol documented in `docs/experiment_v3_protocol.md`. New experimental
> claims will only be based on this implementation after quantum energy and gradient parity tests
> pass.

The historical and corrected Hamiltonian families, including the class-separation objective, are
specified in [`docs/hamiltonian_design.md`](docs/hamiltonian_design.md).
The corrected shared generator and optimizer isolation are specified in
[`docs/shared_generator_design.md`](docs/shared_generator_design.md).
The current exploratory pilot and draft ablation protocol are documented in
[`docs/pilot_500_steps.md`](docs/pilot_500_steps.md) and
[`docs/ablation_protocol_draft.md`](docs/ablation_protocol_draft.md).

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10–3.12](https://img.shields.io/badge/python-3.10--3.12-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)](https://qiskit.org/)
[![Status](https://img.shields.io/badge/status-V2%20Negative%20Result-red.svg)]()

> **⚠️ V2 Research Status: NEGATIVE RESULT**  
> This repository reports a **negative result**: Classical ablation experiments demonstrate that the VQE quantum module provides **no measurable causal benefit** over simple classical alternatives. The hybrid architecture trains successfully, but quantum-specific contribution is undetectable. We publish this as an honest negative result to benefit the quantum ML research community.

---

## 📋 Overview

This repository accompanies the paper:

> *"Differentiable Energy-Based Regularization in GANs: A Simulator-Based Exploration of VQE-Inspired Auxiliary Losses"* — **V2 (with Ablation Study)**

We investigated whether VQE-computed energy terms can serve as auxiliary regularization signals in GAN training. **The ablation study (V2) demonstrates that equivalent or superior results can be achieved with simple classical baselines.**

### What This Is

- ✅ **Technical feasibility demonstration** of VQE-GAN integration
- ✅ **Honest negative result** showing VQE provides no unique benefit
- ✅ **Complete ablation study** against four classical alternatives
- ✅ **Open-source reference implementation** for reproducibility

### What This Is NOT

- ❌ **NOT** a claim of quantum advantage
- ❌ **NOT** evidence of quantum-specific benefit (proven by ablation)
- ❌ **NOT** validated on real quantum hardware

---

## 🔬 Key Finding: Negative Result

### Ablation Study Summary (V2)

We tested the QACGAN against four classical baselines with pre-registered equivalence thresholds (δ_Acc=±3%, δ_FID=±5, δ_IS=±0.3, δ_LPIPS=±0.05):

| Variant | Accuracy (%) | FID (↓) | IS (↑) | LPIPS (↑) |
|---------|-------------|---------|--------|-----------|
| **MLP-Energy** | 99.1 ± 0.5 | 21.33 ± 2.97 | 2.11 ± 0.04 | 0.166 ± 0.006 |
| **Learned Bias** | 99.0 ± 0.6 | **18.43 ± 1.03** | 2.09 ± 0.06 | 0.164 ± 0.007 |
| **Random Noise** | 99.2 ± 0.4 | 20.77 ± 3.67 | 2.16 ± 0.07 | 0.165 ± 0.006 |
| **No Regularizer** | 99.0 ± 0.4 | 20.59 ± 2.72 | 2.11 ± 0.05 | 0.165 ± 0.009 |
| *QACGAN (VQE) ref* | *99.5 ± 0.5* | *27.9 ± 8.0* | *2.07 ± 0.1* | *—* |

**Statistical Conclusion:** All classical variants fall within equivalence thresholds of QACGAN. No paired t-test achieved p < 0.05. Cohen's d < 0.8 for all comparisons. **The VQE module provides no measurable causal benefit.**

### Why This Happened

The quantum module **does not see real data**—it operates solely on latent variables (z, class label). The Hamiltonian parameters are **static coefficients**, not learned from data. This means the VQE acts as a **deterministic nonlinear function** of the class label, which any simple classical function can replicate.

---

## 🏗️ Architecture

The model extends ACGAN by adding a VQE-based energy term to the generator objective:

$$\mathcal{L}_{G} = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{aux}} + \lambda_{\text{VQE}} \cdot E_c$$

where $E_c$ is the expectation value of a class-specific Ising Hamiltonian computed via a 4-qubit `EfficientSU2` ansatz.

**Critical limitation:** Since the Hamiltonian parameters depend only on class labels (not learned from data), this is functionally equivalent to a class-dependent scalar bias.

---

## 📁 Repository Structure

```
├── QACGAN_training_5Epochs.ipynb         # Exploratory run (5 epochs)
├── QACGAN_training_RUN1.ipynb            # Extended run 1 (10 epochs, seed 42)
├── QACGAN_training_RUN2.ipynb            # Extended run 2 (10 epochs, seed 2025)
├── qacgan_training_*.py                  # Python script versions
│
├── ABLATION_Classical_Baselines.ipynb    # ⭐ V2: Classical ablation study
│
├── ablation_results/                     # ⭐ V2: Ablation study outputs
│   ├── ablation_all_results.pkl          # Raw experimental data
│   ├── ablation_summary.json             # Structured results summary
│   ├── ablation_comparison.png           # Visualization (Fig. 2 in paper)
│   ├── ablation_table.tex                # LaTeX table for paper
│   └── *_logs.pkl                        # Per-variant training logs
│
├── hybrid_acgan_results_RUN1/            # Run 1 results
│   ├── hybrid_acgan_images_output/       # Generated samples per epoch
│   ├── hybrid_acgan_models_output/       # Model checkpoints
│   └── mnist_classifier.pth              # Evaluation classifier
│
├── hybrid_acgan_results_RUN2/            # Run 2 results
├── hybrid_qacgan_results_5Epochs/        # 5-epoch exploratory results
│
├── LICENSE                               # MIT License
└── README.md                             # This file
```

---

## 🚀 Usage

### Corrected implementation

- Python 3.10–3.12
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
git clone https://github.com/o0SilentStorm0o/VQE-in-GAN.git
cd VQE-in-GAN
uv sync --extra dev
uv run pytest -q
```

The dependency lock file is authoritative for the redesigned experiment. Quantum backend
benchmarks can be reproduced with:

```bash
uv run python benchmarks/benchmark_quantum_backends.py --backend all --batch-sizes 1 32 64
```

A two-step corrected MNIST smoke run can be started with:

```bash
uv run vqe-gan-train \
  --run-name quantum-smoke \
  --dataset-limit 128 \
  --batch-size 64 \
  --max-steps 2
```

On Apple Silicon, the default `auto` placement keeps the ACGAN on MPS and evaluates the small
statevector on CPU. Full-step benchmark methodology and results are recorded in
[`docs/training_step_benchmark.md`](docs/training_step_benchmark.md).

The tested LUMI-G setup, measured MI250X timings, allocation accounting, and the frozen ten-seed
job-array projection are recorded in [`docs/lumi_g_benchmark.md`](docs/lumi_g_benchmark.md). The
production array script is [`scripts/lumi_run_array.sbatch`](scripts/lumi_run_array.sbatch); it does
not embed an allocation account and therefore cannot submit work by itself.

### Historical notebooks

The root-level notebooks belong to the original experiment and are retained for traceability.
They were designed for **Google Colab** with GPU acceleration:

1. Upload any notebook to Colab
2. Enable GPU runtime (A100 recommended for ablation study)
3. Run all cells

### Reproducing Ablation Study

```bash
# Upload ABLATION_Classical_Baselines.ipynb to Colab
# Expected runtime: ~2-3 hours on A100 GPU
# Outputs saved to ablation_results/
```

### Loading Pretrained Models

```python
import torch

# Load generator (define architecture as in notebook first)
generator.load_state_dict(
    torch.load('hybrid_acgan_results_RUN1/hybrid_acgan_models_output/hybrid_generator_best.pth')
)
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dimension | 100 |
| Batch size | 64 (2×32 gradient accumulation) |
| Learning rate | 2×10⁻⁴ |
| λ_VQE | 0.1 (not tuned) |
| Qubits | 4 |
| Ansatz | EfficientSU2 (1 rep.) |
| Backend | StatevectorEstimator (noiseless) |

---

## ⚠️ Limitations Acknowledged

1. **VQE provides no unique benefit** (proven by V2 ablation study)
2. **Quantum module doesn't see data** — operates only on latent variables
3. **Hamiltonian not learned from data** — static class-dependent coefficients
4. **Simulator-only** — no real quantum hardware validation
5. **Small scale** — 4 qubits, MNIST only
6. **High overhead** — ~200× slower than classical ACGAN

---

## 📖 Lessons Learned

This project demonstrates the importance of **rigorous ablation studies** in quantum machine learning research. The initial results appeared promising, but classical controls revealed that:

1. **Any class-dependent auxiliary signal** provides similar regularization
2. **The quantum circuit's computational cost is not justified** by unique benefits
3. **Negative results are valuable** — they prevent the field from pursuing dead ends

We encourage other QML researchers to include classical ablation studies in their work.

---

## 📖 Citation

```bibtex
@misc{strnadel2025vqegan,
  title={Differentiable Energy-Based Regularization in GANs: 
         A Simulator-Based Exploration of VQE-Inspired Auxiliary Losses},
  author={Strnadel, David},
  year={2025},
  note={V2 with ablation study. Negative result: VQE provides no 
        measurable benefit over classical baselines. arXiv preprint.}
}
```

---

## 🤝 Acknowledgments

- Prof. Roman Šenkeřík (Tomas Bata University in Zlin) for supervision
- [Qiskit](https://qiskit.org/) and [PyTorch](https://pytorch.org/) teams
- The QML community for emphasizing the need for rigorous ablation studies

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Author:** David Strnadel  
**Affiliation:** Faculty of Applied Informatics, Tomas Bata University in Zlin  
**Contact:** d_strnadel@utb.cz

*This work reports a negative result. We publish it openly to benefit the quantum ML community and encourage critical evaluation of hybrid quantum-classical approaches.*
