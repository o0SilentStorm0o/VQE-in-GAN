# Exploratory 500-step pilot

## Scope

These runs are engineering diagnostics with one seed (`42`), 500 optimizer steps, batch size 64,
and 1,000 balanced evaluation samples. They are not confirmatory results and were used to reject
invalid architectural and optimization choices before preregistration.

## Rejected projected-feature angle head

The first shared generator attached the angle head to the projected latent feature map. At step
500, quantum regularizer accuracy was 81.25% with ordinary noise and also 81.25% with zero noise;
the permuted quantum variant produced the same pattern. The regularizer was reading the class
embedding directly. This architecture was rejected.

The angle head now consumes only the completed generated image. Regularizer gradients therefore
traverse the full convolutional image decoder, and no label or latent tensor is supplied directly
to the angle head.

## Rejected initialization and fixed calibration

Zero-biased angle output initialized the circuit close to $|0000\rangle$, a stationary state of the
diagonal Hamiltonians. This required an effective coefficient above 40,000. The output bias is now
deterministically distributed away from zero.

A subsequent fixed 10% gradient calibration was also rejected. Within 100 steps, regularizer/GAN
gradient ratios grew as high as 1.84 and gradient cosine reached approximately -0.97. The final
implementation dynamically normalizes the shared regularizer gradient to 10% each step.

## Valid image-conditioned pilot

| Variant | Runtime | Conditional accuracy | Global feature-FID | Conditional feature-FID | Precision | Diversity ratio |
|---|---:|---:|---:|---:|---:|---:|
| No regularizer | 53.0 s | 0.296 | 0.776 | 0.916 | 0.008 | 0.170 |
| Quantum contrastive | 91.4 s | 0.154 | 0.875 | 1.011 | 0.001 | 0.184 |
| Classical prototype | 77.3 s | 0.282 | 0.741 | 0.921 | 0.041 | 0.161 |
| Quantum permuted | 77.9 s | 0.242 | 0.696 | 0.842 | 0.230 | 0.208 |

Feature recall was zero for all variants, confirming that 500 steps are insufficient for realistic
MNIST manifold coverage. The frozen classifier achieved 98.1% on the balanced real sample.

The standard quantum variant was not better in this pilot. The permuted quantum control had the
best conditional feature-FID, which argues against assigning meaning to this single run and raises
a mechanistic concern about the proposed class-Hamiltonian alignment. Multiple frozen seeds and a
longer convergence horizon are required before drawing a scientific conclusion.

At step 500, energy-target accuracies remained low: 25% for quantum, 21.9% for classical, and 25%
for permuted quantum. Zero-noise accuracies were 21.9%, 21.9%, and 35.9%, respectively. Unlike the
rejected projected-feature architecture, no regularizer rapidly solved its target through direct
label access.
