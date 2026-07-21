# Why the frozen contrastive candidate failed

## Status

This is a post-hoc mechanistic diagnosis of the rejected Candidate 4. It uses the fixed Stage 1
checkpoints for seeds 101 and 202 and does not tune or retrain the candidate. The complete numeric
output is in
[`contrastive_failure_diagnostic_results.json`](contrastive_failure_diagnostic_results.json), with
SHA-256 `30ca187478b55572919e34f49e4091f2ef343d5a2506bbeb65bb3dccec41afb9`.
The reconstructed aggregate metrics agree with the original evaluation files to at most
`3.2e-8`.

## Short answer

The experiment failed for four connected reasons:

1. The branch is not a trainable VQE branch. It is a fixed, parameter-free four-qubit feature map
   applied to a pooled generated image.
2. Its cross-entropy objective rewards relative class separation, not absolute realism, coverage,
   or within-class diversity. It therefore has no structural reason to improve class-FID.
3. `0.95*KDE + 0.05*quantum` does not isolate a quantum addition against the `1.00*KDE` control.
   It also weakens the classical logits. At the first update, almost the entire difference points
   in the classical scale/temperature direction.
4. That small, seed-dependent early perturbation is amplified by the adversarial feedback loop.
   It produces a different GAN trajectory, but not a stable quantum-specific direction. One seed
   becomes more class-discriminative and less representative; the other moves differently.

The slight mean accuracy increase is therefore not evidence that a quantum representation
consistently taught the generator better digits. It is mostly the aggregate result of one favorable
but distribution-worsening trajectory.

## 1. The implemented branch is a fixed feature map, not trainable VQE

The contrastive regularizer does not call `generator.angle_head`. It adaptively pools each final
`28 x 28` image to `4 x 4`, permutes those 16 values, multiplies them by `0.75*pi`, and feeds them
directly to a fixed circuit. The modular observables are fitted once from real images and then held
constant.

| Quantity | Result |
|---|---:|
| Trainable parameters in the contrastive regularizer | 0 |
| Trainable parameters in the generator `angle_head` | 29,200 |
| `angle_head` values changed after 200 steps, all four checkpoints | 0 |
| Image-producing generator parameters reached by the loss | 1,762,789 |

The loss is image-coupled: its gradient reaches the label embedding, input projection, and image
decoder. But no circuit parameter is optimized and the nominal quantum angle head remains exactly
at initialization. The accurate description is therefore a **fixed quantum-state feature map used
as a classical generator regularizer**, not a VQE branch sharing learned parameters with the
generator.

## 2. The loss asks “which class?”, not “is this a real sample?”

Both controls compute ten scores and minimize class cross-entropy:

`CE(score(image), requested_class)`.

This loss depends only on score differences. An image can be far from every real class and still
receive a low loss when its requested class is merely the least implausible of the ten. The quantum
construction strengthens this limitation:

- every modular Hamiltonian has its identity component removed;
- every centered Hamiltonian is normalized to unit Frobenius norm;
- only relative negative energies enter the softmax;
- each image is scored independently, with no generated-sample repulsion or coverage term;
- both branches see only the same pooled 16 values, not fine image structure.

Centering and normalization were useful for class calibration, but they remove absolute energy
offset and scale that might otherwise carry a notion of typicality. The objective can improve a
decision boundary without pulling the generated distribution toward the real density or preserving
its support. ACGAN already contains an auxiliary class loss, so this is also largely a second,
low-resolution semantic signal rather than a missing distributional signal.

## 3. The frozen comparison confounds quantum addition with KDE temperature

The compared losses are

`L_K = CE(K, y)`

and

`L_H = CE(0.95 K + 0.05 Q, y)`.

Against `L_K`, the hybrid changes two things simultaneously. The exact diagnostic inserts

`L_S = CE(0.95 K, y)`

and decomposes

`grad(L_H - L_K) = grad(L_S - L_K) + grad(L_H - L_S)`.

The first term is classical logit scaling; the second is the isolated quantum-logit addition.

### Actual first generator batch

All percentages below are weighted gradient norms relative to the ordinary ACGAN generator
gradient. The cosine compares the complete hybrid-minus-KDE change with the classical scaling
effect; one means the directions are identical.

| Seed | KDE | Hybrid | Scale effect | Isolated quantum addition | Total difference | Total/scale cosine |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 7.297% | 6.930% | 0.365% | 0.01496% | 0.367% | 0.99919 |
| 202 | 13.152% | 12.506% | 0.659% | 0.02071% | 0.646% | 0.99969 |

The standalone quantum gradient is only 3.99% and 2.00% of the KDE gradient on these first
batches. After multiplication by the 5% mixture coefficient, the genuinely added direction is
only 0.015% to 0.021% of the GAN gradient. The observed initial hybrid-minus-KDE gradient is thus
almost exactly the direction obtained by slightly weakening the classical logits.

Adam makes coordinate-level differences less linear, but the causal conclusion survives the
actual first optimizer update:

| Seed | Hybrid/KDE update distance as fraction of full KDE update | Alignment with scale-only update | Quantum residual fraction of paired difference |
|---:|---:|---:|---:|
| 101 | 2.784% | 0.9757 | 22.05% |
| 202 | 2.740% | 0.9796 | 20.15% |

The isolated quantum residual is nonzero, but the paired update difference remains approximately
98% directionally aligned with the temperature-matched classical control. Consequently the frozen
comparison cannot attribute its downstream difference specifically to the quantum circuit.

## 4. The quantum score is weak and changes few KDE decisions

On 1,000 balanced generated samples per checkpoint:

- the quantum score predicts the requested class in 56.0% to 61.5% of cases;
- KDE predicts it in 58.8% to 68.0% of cases;
- KDE and quantum predictions agree in only 64.0% to 74.3% of cases;
- adding the quantum logits changes the KDE argmax for only 0.7% to 2.1% of images;
- the effective centered-logit perturbation is 2.56% to 2.81%, not 5%, because the quantum logits
  have a smaller scale.

The hybrid sometimes lowers cross-entropy, but it very rarely changes the class decision. At the
end of training, the isolated quantum-addition gradient is only `1.6e-6` to `3.2e-6` of the GAN
gradient after weighting. It is also negatively aligned with the independent semantic,
target-centroid, and diversity-promoting proxy gradients in all four final-checkpoint diagnostics.
Thus it neither remains dynamically important nor supplies a stable external notion of realism.

## 5. The aggregate result is a class-boundary/distribution trade-off

### Seed 101

| Class | Accuracy delta | Class-FID delta | Centroid delta | Diversity delta |
|---:|---:|---:|---:|---:|
| 0 | +0.092 | +0.0112 | +0.0105 | -0.0036 |
| 1 | -0.054 | +0.0285 | +0.0289 | -0.0201 |
| 2 | +0.044 | +0.0147 | +0.0109 | -0.0153 |
| 3 | -0.080 | +0.0273 | +0.0229 | -0.0037 |
| 4 | -0.020 | +0.0074 | +0.0087 | -0.0048 |
| 5 | -0.038 | +0.0434 | +0.0398 | -0.0084 |
| 6 | +0.082 | +0.0117 | +0.0121 | -0.0082 |
| 7 | +0.044 | +0.0027 | -0.0020 | -0.0035 |
| 8 | +0.046 | +0.0186 | +0.0166 | -0.0019 |
| 9 | +0.050 | +0.0257 | +0.0211 | -0.0248 |

Accuracy improves for 6/10 classes, but class-FID and diversity worsen for **all ten**. The hybrid
makes requested classes somewhat easier to recognize while moving their feature distributions
away from real class centers and narrowing them. The centroid component explains 88.6% of the mean
class-FID deterioration. This is the direct reason seed 101 raises accuracy while worsening
distribution quality.

### Seed 202

| Class | Accuracy delta | Class-FID delta | Centroid delta | Diversity delta |
|---:|---:|---:|---:|---:|
| 0 | -0.142 | +0.0204 | +0.0162 | -0.0049 |
| 1 | -0.060 | +0.0582 | +0.0565 | +0.0088 |
| 2 | +0.034 | +0.0253 | +0.0230 | +0.0066 |
| 3 | +0.002 | -0.0192 | -0.0115 | +0.0157 |
| 4 | +0.058 | +0.0067 | +0.0078 | -0.0047 |
| 5 | -0.006 | -0.0425 | -0.0342 | +0.0305 |
| 6 | +0.042 | -0.0022 | -0.0042 | +0.0039 |
| 7 | -0.042 | -0.0052 | -0.0064 | +0.0113 |
| 8 | +0.172 | -0.0493 | -0.0484 | +0.0052 |
| 9 | -0.070 | -0.0664 | -0.0616 | +0.0170 |

Here class-FID improves for 6/10 classes and diversity for 8/10, while the large accuracy gains and
losses almost cancel. The improvement is again predominantly a centroid effect: 84.5% of the mean
class-FID change. Nothing in the fixed quantum score predicts that classes 8 and 9 should dominate
this seed while classes 0 and 1 deteriorate. This is a trajectory-specific redistribution, not a
uniform quantum mechanism.

## 6. A tiny early difference is amplified by GAN feedback

Before the first generator update, the paired runs have exactly identical adversarial, auxiliary,
and discriminator losses. Only the regularizer differs. All GAN and discriminator metrics first
diverge at step 2.

| Seed | First paired update distance | Final generator-parameter distance | Relative final distance | Amplification ratio | Final paired image RMSE | Classifier disagreement |
|---:|---:|---:|---:|---:|---:|---:|
| 101 | 0.00738 | 3.3186 | 9.36% | 449.9x | 0.2935 | 30.30% |
| 202 | 0.00726 | 3.8986 | 10.84% | 536.9x | 0.3311 | 28.12% |

After the first slightly different generator image, the discriminator receives different fake
data at step 2. It then supplies a different gradient to the next generator step, which changes the
next fake batch again. This closed feedback loop magnifies a small initial optimizer perturbation
into substantially different generators. Because the isolated quantum gradient has inconsistent
alignment across seeds, the final direction can be favorable or unfavorable.

This amplification is deterministic on CPU; it is not numerical nondeterminism. It is ordinary
sensitivity of adversarial optimization to a small but systematic change in its early trajectory.

## Causal answer to “why?”

The candidate did not fail because the circuit was too slow or because its gradients were broken.
It failed because the implemented signal was the wrong kind of signal for the desired claim:

1. A fixed low-resolution circuit produces another class discriminator.
2. Relative class cross-entropy does not constrain real-density support or sample diversity.
3. Its nominal 5% addition is much smaller in effective logit and gradient scale.
4. The comparison simultaneously changes the classical temperature, which dominates the initial
   paired difference.
5. Adversarial feedback amplifies that confounded, seed-specific perturbation into different final
   distributions.
6. The favorable accuracy component comes from sharper or shifted class boundaries, not a stable
   quantum-specific improvement in the generated distribution.

Therefore the most defensible conclusion is stronger than “two seeds were unlucky”: **this design
does not isolate, train, or test the quantum resource needed for its intended claim**. Its failure
is structurally understandable from the loss and update equations, and the checkpoints show the
predicted class/distribution trade-off.

## Requirements for a genuinely new candidate

A replacement should not retune this mixture. It should satisfy all of the following before any
new held-out run:

1. Use trainable circuit parameters that are actually reached by the optimizer and tied to the
   image-producing generator representation.
2. Target within-class distribution geometry, coverage, or sample relationships rather than
   duplicating the ACGAN class objective.
3. Keep the classical reference logits and their temperature exactly fixed when adding the quantum
   term; include a scale-only control explicitly.
4. Give the isolated quantum residual its own measured gradient budget and compare it with a
   parameter- and data-matched classical residual.
5. Include an explicit diversity or two-sample term and a causal removal of coherence and
   entanglement.
6. Develop on a new seed set and preserve a new untouched sequence. Seeds 101 and 202 may only be
   used to explain this rejected candidate.
