# VQE-in-GAN: Exploratory Integration of VQE-Inspired Energy Terms in GANs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)](https://qiskit.org/)
[![Status](https://img.shields.io/badge/status-Proof%20of%20Concept-orange.svg)]()

> **⚠️ Important:** This is an **exploratory proof of concept**, not a performance benchmark.  
> We do **not** claim quantum advantage or improved performance over classical methods.

## 📋 Overview

This repository accompanies the paper:

> *"Differentiable Energy-Based Regularization in GANs: A Simulator-Based Exploration of VQE-Inspired Auxiliary Losses"*

We investigate whether VQE-computed energy terms can serve as auxiliary regularization signals in GAN training. The primary contribution is **methodological**: demonstrating the technical feasibility of integrating differentiable VQE pathways into generative model training loops.

### What This Is

- ✅ **Technical feasibility demonstration** of VQE-GAN integration
- ✅ **Boundary exploration** of hybrid quantum-classical generative models
- ✅ **Open-source reference implementation** for reproducibility

### What This Is NOT

- ❌ **NOT** a claim of quantum advantage
- ❌ **NOT** a performance improvement over classical methods
- ❌ **NOT** validated on real quantum hardware

## 🔬 Honest Results Summary

| Metric | ACGAN Baseline | Energy-Regularized (5 ep.) | Notes |
|--------|----------------|----------------------------|-------|
| Accuracy @ 5 ep. | 87.8% | 99–100% | *Consistent across runs* |
| FID @ 5 ep. | — | 19.92–35.96 | *High variance (CV≈25%)* |
| Best FID | 24.02 (@20 ep.) | ~23–24 (extended runs) | *Comparable after convergence* |

**Key observations:**
- The auxiliary energy term appears to influence class conditioning (high accuracy consistency)
- Sample quality (FID) shows high variance and converges to baseline-comparable values
- We **cannot conclude** whether observed effects are specific to VQE or would occur with any class-dependent auxiliary signal

## ⚠️ Critical Limitations

1. **No ablation against classical baselines.** We did not compare against equivalent classical regularizers (e.g., MLP-based class-dependent scalars). The observed effects may simply reflect the presence of *any* auxiliary class-dependent signal.

2. **Trivial Hamiltonian design.** The Ising Hamiltonian uses a deliberately simple linear parameterization that encodes no meaningful physics or problem structure.

3. **Simulator-only evaluation.** All results use a noiseless statevector simulator. Real quantum hardware behavior is unknown.

4. **Small scale.** Limited to 4 qubits, MNIST, and relatively few evaluation samples.

5. **High computational overhead.** ~200× slower than classical ACGAN due to Python-level simulation.

## 🏗️ Architecture

The model extends ACGAN by adding a VQE-based energy term to the generator objective:

$$\mathcal{L}_{G} = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{aux}} + \lambda_{\text{VQE}} \cdot E_c$$

where $E_c$ is the expectation value of a class-specific Ising Hamiltonian computed via a 4-qubit `EfficientSU2` ansatz using Qiskit's `EstimatorQNN`.

## 📁 Repository Structure

```
├── QACGAN_training_5Epochs.ipynb     # Exploratory run (5 epochs)
├── QACGAN_training_RUN1.ipynb        # Extended run 1 (10 epochs, seed 42)
├── QACGAN_training_RUN2.ipynb        # Extended run 2 (10 epochs, seed 2025)
├── qacgan_training_*.py              # Python script versions
│
├── hybrid_acgan_results_RUN1/        # Run 1 results
│   ├── hybrid_acgan_images_output/   # Generated samples per epoch
│   ├── hybrid_acgan_models_output/   # Model checkpoints
│   └── mnist_classifier.pth          # Evaluation classifier
│
├── hybrid_acgan_results_RUN2/        # Run 2 results
├── hybrid_qacgan_results_5Epochs/    # 5-epoch exploratory results
│
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## 🚀 Usage

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Colab (for quantum simulation)

### Installation

```bash
git clone https://github.com/o0SilentStorm0o/VQE-in-GAN.git
cd VQE-in-GAN
pip install torch torchvision qiskit qiskit-aer qiskit-machine-learning torch-fidelity
```

### Running Experiments

The notebooks are designed for **Google Colab** with GPU acceleration:

1. Upload any `QACGAN_training_*.ipynb` notebook to Colab
2. Enable GPU runtime
3. Run all cells

### Loading Pretrained Models

```python
import torch

# Load generator (define architecture as in notebook first)
generator.load_state_dict(
    torch.load('hybrid_acgan_results_RUN1/hybrid_acgan_models_output/hybrid_generator_best.pth')
)
```

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

## � Future Work (What Would Strengthen This)

1. **Ablation studies** comparing against classical regularizers with equivalent structure
2. **Statistical rigor** with confidence intervals over many more runs
3. **Noise-aware backends** to assess hardware feasibility
4. **Larger datasets** beyond MNIST
5. **More expressive Hamiltonians** encoding meaningful problem structure

## 📖 Citation

```bibtex
@misc{strnadel2025vqegan,
  title={Differentiable Energy-Based Regularization in GANs: 
         A Simulator-Based Exploration of VQE-Inspired Auxiliary Losses},
  author={Strnadel, David},
  year={2025},
  note={Exploratory proof of concept. arXiv preprint.}
}
```

## 🤝 Acknowledgments

- Prof. Roman Šenkeřík (Tomas Bata University in Zlin) for supervision
- [Qiskit](https://qiskit.org/) and [PyTorch](https://pytorch.org/) teams

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Author:** David Strnadel  
**Affiliation:** Faculty of Applied Informatics, Tomas Bata University in Zlin  
**Contact:** d_strnadel@utb.cz

*This work represents exploratory research. We encourage critical evaluation and welcome feedback.*
