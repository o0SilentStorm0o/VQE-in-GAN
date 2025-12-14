# QACGAN: Quantum-Enhanced Generative Adversarial Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-6929C4.svg)](https://qiskit.org/)

> **Official repository for the research paper:**  
> *"Quantum-Enhanced Generative Adversarial Networks: VQE-based Regularization for Improved Training Stability and Performance"*

## 📋 Abstract

This repository contains the implementation of **QACGAN** (Quantum-enhanced Auxiliary Classifier GAN), a hybrid quantum-classical architecture that integrates Variational Quantum Eigensolver (VQE)-based energy terms as auxiliary regularization signals in GANs. The method uses class-specific Ising Hamiltonians to provide physics-inspired priors that stabilize early training dynamics.

### Key Results on MNIST

| Metric | ACGAN (50 ep.) | QACGAN (5 ep.) | QACGAN (10 ep., run 1) | QACGAN (10 ep., run 2) |
|--------|----------------|----------------|------------------------|------------------------|
| Best FID ↓ | 24.02 (@20 ep.) | **19.92** (@5 ep.) | 23.91 (@10 ep.) | 23.23 (@9 ep.) |
| Best IS ↑ | 2.23 (@25 ep.) | 2.07 (@5 ep.) | 2.29 (@2 ep.) | **2.32** (@4 ep.) |
| Accuracy @ 5 ep. | 87.8% | 99.0% | 99.0% | **100.0%** |
| Training Time | 0h 27m | 7h 17m | 14h 2m | 14h 9m |

## 🏗️ Architecture

QACGAN extends the classical ACGAN by adding a VQE-based energy term to the generator objective:

$$\mathcal{L}_{G} = \mathcal{L}_{\text{adv}} + \mathcal{L}_{\text{aux}} + \lambda_{\text{VQE}} \cdot E_c$$

where $E_c$ is the expectation value of a class-specific Ising Hamiltonian:

$$H_c = -J \sum_{\langle i,j \rangle} \sigma_z^i \sigma_z^j - \sum_{i=1}^{N} h_{c,i} \sigma_z^i$$

The quantum circuit uses Qiskit's `EfficientSU2` ansatz (4 qubits, 1 repetition) with differentiable parameter encoding via `EstimatorQNN` and `TorchConnector`.

## 📁 Repository Structure

```
├── ACGAN_training.ipynb              # Classical ACGAN baseline (50 epochs)
├── QACGAN_training_5Epochs.ipynb     # QACGAN exploratory run (5 epochs, seed 42)
├── QACGAN_training_RUN1.ipynb        # QACGAN extended run 1 (10 epochs, seed 42)
├── QACGAN_training_RUN2.ipynb        # QACGAN extended run 2 (10 epochs, seed 2025)
├── MNIST_Classifier.ipynb            # External CNN classifier for evaluation
├── mnist_classifier.pth              # Pretrained classifier weights
│
├── acgan_results/                    # Classical ACGAN experiment results
│   ├── acgan_images_output/          # Generated sample images per epoch
│   ├── acgan_models_output/          # Saved model checkpoints
│   └── acgan_training_logs.pkl       # Training metrics log
│
├── hybrid_acgan_results_RUN1/        # QACGAN Run 1 results (10 epochs)
│   ├── hybrid_acgan_images_output/   # Generated samples
│   ├── hybrid_acgan_models_output/   # Model checkpoints
│   └── mnist_classifier.pth          # Classifier for evaluation
│
├── hybrid_acgan_results_RUN2/        # QACGAN Run 2 results (10 epochs)
│   └── ...
│
├── hybrid_qacgan_results_5Epochs/    # QACGAN 5-epoch exploratory results
│   └── ...
│
├── LICENSE                           # MIT License
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Google Colab (for quantum simulation)

### Installation

```bash
# Clone the repository
git clone https://github.com/o0SilentStorm0o/VQE-in-GAN.git
cd VQE-in-GAN

# Install dependencies
pip install torch torchvision qiskit qiskit-aer qiskit-machine-learning torch-fidelity
```

### Running Experiments

The notebooks are designed to run on **Google Colab** with GPU acceleration:

1. **Classical Baseline**: Open `ACGAN_training.ipynb` and run all cells
2. **QACGAN Training**: Open any `QACGAN_training_*.ipynb` notebook
3. **Evaluation**: Use `MNIST_Classifier.ipynb` to evaluate generated samples

### Quick Start (Local)

```python
# Load pretrained QACGAN generator
import torch
from models import HybridGenerator  # Define architecture as in notebook

generator = HybridGenerator(latent_dim=100, n_classes=10)
generator.load_state_dict(torch.load('hybrid_acgan_results_RUN1/hybrid_acgan_models_output/hybrid_generator_best.pth'))
generator.eval()

# Generate samples
z = torch.randn(10, 100)
labels = torch.arange(10)
fake_images = generator(z, labels)
```

## 📊 Evaluation Metrics

All experiments are evaluated using:

- **FID (Fréchet Inception Distance)**: Lower is better. Measures similarity between generated and real image distributions.
- **IS (Inception Score)**: Higher is better. Measures quality and diversity of generated samples.
- **Classification Accuracy**: Evaluated using a pretrained external CNN on 500 generated samples per epoch.

Metrics are computed using `torch-fidelity` with 1,000 generated samples against MNIST test set.

## ⚙️ Hyperparameters

| Parameter | Value |
|-----------|-------|
| Latent dimension | 100 |
| Batch size | 64 (2×32 gradient accumulation) |
| Learning rate | 2×10⁻⁴ |
| Adam β₁, β₂ | 0.5, 0.999 |
| λ_VQE | 0.1 |
| Qubits | 4 |
| Ansatz | EfficientSU2 (1 repetition) |
| Quantum backend | StatevectorEstimator (noiseless) |

## 📖 Citation

If you use this code in your research, please cite:

```bibtex
@article{strnadel2025qacgan,
  title={Quantum-Enhanced Generative Adversarial Networks: VQE-based Regularization for Improved Training Stability and Performance},
  author={Strnadel, David},
  journal={arXiv preprint},
  year={2025}
}
```

## ⚠️ Limitations

- **Simulator-only**: All quantum computations run on a noiseless statevector simulator
- **Small scale**: Limited to 4 qubits and MNIST dataset
- **Computational cost**: QACGAN training is ~200× slower than classical ACGAN due to quantum simulation overhead
- **Variance**: FID shows notable variance across runs (CV ≈ 25% at epoch 5)

## 🔮 Future Work

- Evaluation on noise-aware quantum hardware
- Extension to more complex datasets (Fashion-MNIST, CIFAR-10)
- Systematic ablation studies of λ_VQE
- Adaptive scheduling of quantum regularization

## 🤝 Acknowledgments

- Prof. Roman Šenkeřík (Tomas Bata University in Zlin) for supervision
- [Qiskit](https://qiskit.org/) team for the quantum computing framework
- [PyTorch](https://pytorch.org/) team for the deep learning framework

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Author:** David Strnadel  
**Affiliation:** Faculty of Applied Informatics, Tomas Bata University in Zlin  
**Contact:** d_strnadel@utb.cz
