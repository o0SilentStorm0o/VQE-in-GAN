# Post-hoc gradient-matched circuit diagnostic

## Status

This diagnostic was designed after the frozen relational circuit-ablation gate failed. It cannot
restore that gate, authorize held-out seeds, or convert seeds 42 and 43 into confirmatory evidence.
Its sole purpose is to resolve a newly measured confound in the mechanism comparison.

The same-coefficient ablation changed the initial weighted coverage/GAN shared-gradient ratio from
roughly `0.22%`–`0.25%` for full to `1.99%`–`4.03%` for dephased and `7.70%`–`8.78%` for product on
a common diagnostic batch. Consequently, outcome differences combine circuit geometry with a
large change in optimization strength.

## Frozen question

> When product, dephased, and full kernels receive the same *initial shared image-path gradient
> budget*, does the coherent entangled kernel retain a distributional advantage?

## Calibration

The calibration exactly reuses the original development procedure:

- seeds 42 and 43;
- the first four shuffled training batches per seed;
- initialized generator and discriminator;
- 1,024 fixed real references per class;
- batch size 64;
- target weighted coverage/GAN gradient ratio `0.01` on image-producing generator parameters;
- one fixed median weight per kernel across all eight batches;
- no GAN outcome is used to select a weight.

The deterministic calibration reproduced the committed full-kernel weight exactly. The frozen
weights for this diagnostic are:

- full: `0.0005662128371831583`;
- product: `0.00007045240128892456`;
- dephased: `0.00003077098628673762`.

The exact KDE weight remains `2e-5`. The complete per-batch calibration is stored in
`docs/relational_gradient_matched_calibration.json` (SHA-256
`e7f132c369677842af35fc5de786a6f7a714ecfc5ced93942b5330c14ec2ed95`). With the single median
weight applied to all eight batches, the median weighted coverage/GAN shared-gradient ratios are
`0.01037`, `0.01000`, and `0.01046` for full, product, and dephased respectively. Their batchwise
ranges remain unequal (`0.00205`–`0.01499`, `0.00781`–`0.02085`, and
`0.00083`–`0.01424`), so the diagnostic matches the prespecified central budget rather than every
individual batch.

The angle head is reported separately. Because it receives no GAN or KDE gradient, Adam partly
normalizes away a scalar loss coefficient on its first update. Matching the shared-path gradient
therefore removes the largest observed confound but does not make all optimizer dynamics
identical. This limitation must remain in the interpretation.

## Diagnostic runs

After the weights are committed, run only gradient-matched product and dephased variants on seeds
42 and 43:

- 200 deterministic CPU steps;
- 5,000 balanced evaluation samples with seed 91001;
- all data, model, angle, reference, and optimizer settings unchanged;
- no full rerun is required because two bitwise-exact full replays are already recorded.

Compare the new means with the existing exact full means. Reapply the earlier descriptive margins:

1. full class-FID at least `0.005` lower than each control;
2. full diversity no more than `0.005` lower than each control;
3. full accuracy no more than `0.005` lower than each control.

These margins organize the diagnosis; passing them would be development-only mechanistic support,
not a new preregistered gate. Failure, especially against dephased, means this coverage design has
not isolated a useful phase-coherent contribution.

No weight, temperature, residual bound, horizon, or margin may be changed after these results are
read. Untouched seeds 303, 404, and 505 remain unused regardless of the outcome.

## Recorded outcome

The four diagnostic runs completed at 200 steps and were evaluated on 5,000 balanced samples.
The complete machine-readable audit is
`docs/relational_gradient_matched_diagnostic_results.json` (SHA-256
`8a55f5952b7767aaace375170a259f63dec2949dfbcf64ceaf9928076560d938`). Mean development
metrics were:

| Kernel | Conditional accuracy | Class-FID | Diversity ratio |
| --- | ---: | ---: | ---: |
| Full coherent entangled | 0.673600 | 0.402251 | 0.574536 |
| Gradient-matched product | 0.681700 | 0.407496 | 0.570771 |
| Gradient-matched dephased | 0.701400 | 0.416903 | 0.549076 |

Relative to product, full changed accuracy by `-0.008100`, class-FID by `-0.005244`, and
diversity by `+0.003765`. Relative to dephased, the corresponding changes were `-0.027800`,
`-0.014652`, and `+0.025461`. Thus full met the descriptive class-FID and diversity conditions
against both controls, but failed the accuracy condition against both. The mean class-FID result
was not sign-consistent: on seed 43 full was worse than product by `0.003551` and worse than
dephased by `0.001273`.

All descriptive margins therefore did **not** pass. This diagnostic cannot isolate either an
entanglement contribution or a phase-coherent contribution, and it does not authorize held-out
execution.

## Why the apparent distributional gain is not yet quantum-specific

The initial calibration removed the largest same-coefficient confound, but only at initialization.
At a common final full checkpoint, the raw shared image-path gradient directions remained very
similar:

- full versus product cosine: `0.9450`–`0.9664`;
- full versus dephased cosine: `0.9643`–`0.9725`.

Despite pointing mostly in the same direction, the calibrated weighted control gradients no
longer had comparable magnitudes. Product retained only `11.22%`–`14.33%` of the full shared
gradient norm, and dephased retained only `2.16%`–`2.39%`. A fixed coefficient calibrated on the
first batches therefore did not hold the optimization budget constant as the generator moved
through parameter space. The modest mean distributional advantage can still be explained by a
stronger late-stage regularization signal rather than by a uniquely useful coherent direction.

The trainable residual path also evolved differently. The actual residual correction RMS was
`0.264`–`0.272` radians for full, `0.244`–`0.276` for product, and only `0.006`–`0.008` for
dephased. In the full checkpoints, `99.0%`–`99.6%` of the shared coverage-gradient norm came from
the direct pooled-image path rather than indirectly through the angle head. Parameter sharing is
real, but the measured image update is dominated by the fixed spatial encoding, while Adam and
the kernel geometry give the angle-only path different dynamics across controls.

The diagnostic therefore answers the causal question conservatively: the full kernel changes the
training trajectory and shows a development-only coverage/readability trade-off, but this design
has not demonstrated that coherence or entanglement supplies the useful part of that change.

## Consequence for the next candidate

This candidate stops here. A subsequent design must freeze its rules before new outcomes and keep
the shared coverage-gradient budget matched throughout training, not only at initialization. It
must also control the angle-only optimizer path separately, because that path receives no GAN or
KDE gradient and a scalar loss coefficient is not an adequate Adam-step control. Seeds 303, 404,
and 505 remain untouched.
