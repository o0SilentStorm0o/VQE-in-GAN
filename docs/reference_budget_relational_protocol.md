# Full-trajectory reference-budget relational diagnostic

## Status and scope

This protocol was frozen before any outcome run using the new per-minibatch budget controller. It
follows
the failed fixed-weight circuit diagnostic and addresses its two measured confounds:

1. the weighted coverage gradient on the image-producing generator parameters drifted apart
   during training; and
2. the angle head has no GAN or KDE gradient, so multiplying its loss by a scalar does not control
   its Adam displacement reliably.

The diagnostic is development-only. Seeds 42 and 43 may be used because they have already been
examined. Seeds 303, 404, and 505 remain untouched. Phase A and Phase B will both be completed,
regardless of the Phase A outcome, and no threshold or optimizer rule may be changed between them.

## Frozen causal question

> When the coherent-entangled, product, and dephased circuits receive the same full-trajectory
> optimization budget, does the coherent-entangled relational direction improve conditional
> image coverage without an unacceptable loss of class readability or diversity?

The coherent-entangled kernel is the reference branch. For each seed it is run first and records a
step-indexed budget schedule. Product and dephased controls then replay that schedule using the
same seed, data order, discriminator noise, generator noise, labels, architecture, reference bank,
and number of updates. Outcomes are evaluated only after all three branches in a phase finish.

## Parameters and losses covered by the budget

The shared image-producing parameter set is exactly

`label_embedding + input_projection + image_decoder`.

The angle-only parameter set is exactly `angle_head`. Let

- `L_anchor = L_adversarial + L_auxiliary + 2e-5 L_KDE`;
- `L_Q^k` be the relational coverage loss for kernel `k`;
- `g_anchor^k = grad_shared L_anchor`;
- `g_Q,shared^k = grad_shared L_Q^k`; and
- `lambda_F = 0.0005662128371831583`, the already frozen full-kernel coefficient.

At full-reference step `t`, record

`r_t = lambda_F * ||g_Q,shared^F||_2 / ||g_anchor^F||_2`.

For a control kernel `k`, use

`lambda_t^k = r_t * ||g_anchor^k||_2 / ||g_Q,shared^k||_2`.

The assigned shared gradient is

`g_anchor^k + lambda_t^k g_Q,shared^k`.

Thus every control exactly matches the full branch's step-specific coverage-to-anchor raw-gradient
ratio while retaining its own circuit-induced gradient direction. This is deliberately not a
constant 1% controller: the measured full ratio changes by more than an order of magnitude during
training, and replacing it by 1% would define a different full candidate.

If either required norm is non-finite or the coverage norm is at or below machine epsilon, the run
is invalid. There is no clipping, smoothing, fallback coefficient, or reuse of a nearby step.

### Pre-outcome optimizer amendment

A two-step implementation smoke test, performed without image evaluation, showed that exact raw
matching alone left the counterfactual shared Adam effect 69%–83% different for product and
18%–34% different for dephased. This is an optimizer confound, not a GAN outcome. The production
runner had not been used, so the protocol was amended before either development phase began.

Let `U_anchor^k` be the Adam displacement proposed from `g_anchor^k` using the current moments, and
let `U_total^k` be the ordinary displacement after the raw-matched total gradient. The full branch
records

`s_t = ||U_total^F - U_anchor^F||_2 / ||U_anchor^F||_2`.

After Adam updates a control's moments normally, its applied shared displacement is

`U_anchor^k + alpha_t^k (U_total^k - U_anchor^k)`,

where

`alpha_t^k = s_t ||U_anchor^k||_2 / ||U_total^k - U_anchor^k||_2`.

This adds a second independent budget control: the loss coefficient matches the raw gradient norm,
while `alpha_t^k` matches the actual Adam-preconditioned auxiliary displacement. Neither operation
rotates the circuit direction it controls. Adam's moments still receive the exact raw-matched total
gradient. The full branch uses its ordinary Adam update with multiplier 1.0 and is unchanged.

A first production attempt stopped before evaluation at product step 144 because applying a large
late-stage multiplier once in float32 missed the shared Adam tolerance. No outcome was generated or
read. The applied scalar is therefore solved deterministically against the realized float32
parameter displacement: after each write, multiply it by `target / achieved`, for at most eight
iterations, stopping when relative error is at most `1e-5`. The frozen acceptance limit remains
`1e-4`; failure to meet it still invalidates the run. This numerical refinement changes neither
the target nor the update direction, performs no clipping, and is also used for the Phase B angle
displacement. Full reference updates require no refinement and remain ordinary Adam updates.

A second production attempt stopped before evaluation at full seed-43 step 147 when the Adam
proposal check narrowly exceeded `1e-5`. The controller used an algebraically equivalent Adam
formula, but its separate float32 multiply/divide operations did not reproduce PyTorch's in-place
`lerp_`, `addcmul_`, and `addcdiv_` rounding exactly. The counterfactual proposal now executes those
same operations on cloned tensors before measuring displacement. The optimizer, frozen tolerance,
budget target, and ordinary full update are unchanged.

Using the exact Adam arithmetic exposed a staircase in the realized float32 control displacement:
the multiplicative refinement then oscillated and a subsequent production attempt stopped before
evaluation at product seed-42 step 11. A bracketed bisection on that exact failed step reached
relative error `3.21e-6`, proving that the target was representable and the former solver—not the
budget definition—was at fault. Refinement now brackets values below and above the target, retains
the best realized point, and bisects with at most 33 parameter writes. It is used only when the
first analytical scale misses `1e-5`; the frozen `1e-4` acceptance limit and all outcome rules are
unchanged.

## Phase A: fixed image-to-circuit map

The entire angle head is zero-output initialized and frozen. Therefore the circuit angles are the
fixed 4x4 pooled-image encoding and the coverage gradient can reach the generator only through the
generated image. All three circuit kernels use the shared-budget rule above. This phase asks
whether circuit geometry helps when the trainable angle-only path is removed.

## Phase B: separately controlled trainable angle head

The angle head is zero-output initialized but trainable. The shared parameters use the identical
rule from Phase A. In addition, the full branch records at each step:

- `a_t = lambda_F * ||grad_angle L_Q^F||_2`, the weighted raw angle-gradient norm; and
- `u_t`, the actual L2 norm of the full angle-head parameter displacement produced by Adam.

For each control, its angle gradient is independently scaled to norm `a_t`. After Adam updates its
moments, the angle-head displacement is multiplied by `u_t / u_t^k`. This is exactly equivalent to
changing only that step's angle-head learning rate because Adam's proposed displacement is linear
in the learning rate; the per-parameter direction and moment updates are not altered. Shared
parameters use the shared Adam correction frozen above; the angle-head correction is applied
independently.

The angle-head parameters occupy their own Adam parameter group with the same frozen optimizer
settings as the shared group. The full branch keeps multiplier 1.0. A zero or non-finite proposed
angle update invalidates the run.

## Frozen technical acceptance checks

Before outcomes may be interpreted, every step in every branch must satisfy all applicable checks:

1. achieved shared raw-gradient ratio relative error at most `1e-5`;
2. achieved shared Adam auxiliary-displacement ratio relative error at most `1e-4`;
3. the analytical Adam proposal agrees with the optimizer's ordinary displacement within `1e-5`;
4. Phase B achieved weighted raw angle-gradient norm relative error at most `1e-5`;
5. Phase B actual angle-head displacement relative error at most `1e-4`;
6. no skipped, clipped, fallback, zero-norm, or non-finite step;
7. exactly 200 budget records, consumed once and in order by each control;
8. schedule metadata match seed, phase, full variant, horizon, and frozen full coefficient;
9. the schedule SHA-256 is recorded in each consuming run's provenance; and
10. a fixed-weight full replay and a schedule-recording full replay are state-identical in the
   implementation test at a short deterministic horizon.

The runner records the uncorrected shared Adam mismatch and the applied correction multiplier in
addition to the achieved value. A zero or non-finite anchor or auxiliary Adam displacement
invalidates the run; there is no clipping or fallback multiplier.

## Frozen runs and evaluation

For each phase and seed 42 and 43:

- variants: full coherent-entangled, product, dephased;
- deterministic CPU training;
- 200 generator/discriminator steps;
- batch size 64;
- 1,024 fixed real references per class;
- unchanged ACGAN, circuit, KDE, temperature, residual bound, and Adam settings;
- 5,000 balanced generated evaluation samples;
- evaluation seed 91001; and
- the existing frozen MNIST classifier and evaluation implementation.

The reference run is trained and its schedule is sealed before either control starts. The three
checkpoints are evaluated only after all training runs for that phase are complete. Phase B starts
only after the Phase A audit is written, but its execution is mandatory and its rules remain those
frozen here.

## Frozen development gates

For a phase to provide development evidence for the coherent-entangled circuit, all technical
checks must pass and full must satisfy, against both product and dephased:

1. lower class-FID on each seed and mean class-FID lower by at least `0.005`;
2. mean diversity ratio no more than `0.005` lower; and
3. mean conditional accuracy no more than `0.005` lower.

Phase B additionally tests whether training the controlled angle head adds value over Phase A.
It is considered an improvement only if its full branch lowers mean class-FID by at least `0.005`
relative to the Phase A full branch, without lowering mean diversity or accuracy by more than
`0.005`.

No held-out seed is authorized unless at least one phase passes its full circuit gate. If both
pass, Phase B is selected only if it also passes the incremental angle-head gate; otherwise Phase A
is selected. Selection and any held-out execution require a separate explicit decision after the
development audit. Failure of both phases ends this candidate without trying new coefficients on
the held-out seeds.
