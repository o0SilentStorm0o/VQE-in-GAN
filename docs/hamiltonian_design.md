# Class-conditioned Hamiltonian design

This document separates the historical Hamiltonian from the corrected experimental variant.
Both remain available so the original result can be reproduced and the new claim can be tested
without silently changing the past experiment.

## Historical family

For class $c$, the original experiment used

$$
H_c^{\mathrm{legacy}}
= -J\sum_{i=0}^{n-2} Z_i Z_{i+1}
- h_c\sum_{i=0}^{n-1} Z_i,
\qquad h_c = h_0 + c\,\Delta h.
$$

With $J>0$ and every $h_c>0$, all class Hamiltonians have the same unique ground state,
$|00\ldots0\rangle$. Increasing $h_c$ changes energy scale and gaps, but it does not create a
class-specific optimum. Consequently, minimizing only the target energy is not a meaningful
class-separation objective.

## Class-encoded family

Let $s_{c,i}\in\{-1,+1\}$ be the $Z_i$ eigenvalue encoded by bit $i$ of class index $c$:
bit zero maps to $+1$ and bit one maps to $-1$. The corrected family is

$$
H_c^{\mathrm{encoded}}
= -J\sum_{i=0}^{n-2} s_{c,i}s_{c,i+1} Z_i Z_{i+1}
- h\sum_{i=0}^{n-1}s_{c,i}Z_i.
$$

For four qubits, classes 0 through 9 map deterministically to computational-basis states
$|0000\rangle$ through $|1001\rangle$ in Qiskit's little-endian integer convention. Each class has
a different unique ground state. The Hamiltonians are related by local $X$ conjugations, so they
have identical sorted spectra. This avoids making some classes intrinsically easier merely by
assigning them a larger field magnitude.

## Training objective

The historical loss minimizes one selected scalar,

$$
\mathcal{L}_{\mathrm{raw}} = \frac{1}{B}\sum_b E_{b,y_b}.
$$

The corrected experiment evaluates the energy matrix $E\in\mathbb{R}^{B\times C}$ and uses
negative energies as class logits:

$$
\mathcal{L}_{\mathrm{contrastive}}
= \operatorname{CE}\!\left(-E/\tau, y\right), \qquad \tau>0.
$$

This objective rewards low target energy only relative to all non-target Hamiltonians. The
batched PyTorch statevector computes all class energies from one circuit state per sample; only
the inexpensive diagonal expectation calculation is repeated across classes.

## Scope of the claim

The encoded family supplies a coherent class-conditioned quantum objective. It does not by itself
establish that the quantum branch learns useful image structure or outperforms a classical map.
That requires shared generator parameters, verified gradient flow, and matched classical
ablations. Those are separate experimental stages.
