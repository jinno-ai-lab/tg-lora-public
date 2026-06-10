# Component 2 Positioning

## Purpose

This note fixes the manuscript-level interpretation of the trajectory
extrapolation component after the offline predictability controls.

It is a writing guide, not a result table. Raw numbers should be read from the
generated predictability reports and runtime ablation summaries listed in
[02_source_data.md](02_source_data.md).

## Revised Claim

Do not frame Component 2 as general-purpose speculative trajectory prediction.

Use this narrower claim instead:

> When LoRA update trajectories contain a dominant low-curvature direction,
> lr-normalized EMA extrapolation can extract that direction and advance several
> optimizer-step equivalents with cheap parameter-space updates.

This claim is supported by the control structure:

- Random-direction controls are near zero, so the observed future-update cosine
  is not a random high-dimensional artifact.
- Shuffled-history controls remain close to the true temporal EMA, so most of
  the predictability comes from a low-frequency dominant direction rather than
  from accurately predicting local zig-zag order.
- Longer horizons decay, so the dominant-direction approximation has a finite
  reach. The current operating range should be treated as an empirical horizon,
  not as an unbounded extrapolation rule.

## Mathematical Decomposition

Use the update decomposition

```text
u_t = u_bar + eps_t
```

where `u_bar` is the dominant direction shared across the local training window
and `eps_t` is the high-frequency zig-zag component.

EMA suppresses the `eps_t` component and preserves `u_bar`. Shuffling destroys
temporal order in `eps_t` but leaves `u_bar` intact. Therefore, high shuffled
cosine means the method is exploiting the existence of `u_bar`, not reading the
next high-frequency turn of the trajectory.

This is a strength if stated honestly: the method identifies and exploits a
measurable precondition for successful extrapolation.

## Figure Guidance

The Component 2 mechanism figure should show three controls together:

1. future cumulative update vs EMA direction
2. future cumulative update vs random direction
3. future cumulative update vs shuffled-history EMA

The caption should explicitly state that shuffled-history cosine being close to
the true EMA cosine indicates dominant-direction structure.

## Runtime Gate

The runtime claim must be separated from the offline predictability claim.

Offline predictability answers whether the direction exists. The runtime ablation
answers whether using that signal improves `reduction_rate` and wall-clock under
fixed lr, fixed scope, random-walk disabled, and persistent Adam.

The runtime ablation entry point is:

```bash
make cosine-n-ablation
```

The current cosine-driven runtime ablation establishes the following separation:

- cosine-driven `N` selection safely raises `reduction_rate` from fixed-N
  `0.625` to about `0.752` with no observed rollbacks in the 3-seed run.
- wall-clock remains close to fixed-N because the post-extrapolation acceptance
  eval is still paid on almost every cycle.

Canonical completed artifact:

- [../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json](../../runs/cosine_n_ablation_20260603_021730/cosine_n_ablation_summary.json)

Key completed runtime values:

- fixed-N `reduction_rate`: `0.625`
- cosine-driven `N` `reduction_rate`: `0.752066`
- rollback rate: `0.0` for fixed-N and cosine-driven `N`
- selected `N` distribution for cosine-driven `N`: `{1: 3, 3: 1, 5: 1, 10: 2, 20: 3}` in each seed
- cosine-driven vs fixed-N wall-clock ratio: `0.9929x`
- best valid loss remains effectively matched: fixed-N mean `1.12146`, cosine-driven mean `1.12228`

The ideal zero-validation-cost speedup ceiling implied by a reduction rate `r`
is:

```text
speedup_max ~= 1 / (1 - r)
```

For `r = 0.752`, this gives `speedup_max ~= 4.0x`. This does not claim an
observed 4x speedup. It shows that the remaining wall-clock bottleneck is the
fixed validation/probe cost, not the absence of speculative backward-pass
replacement. The next runtime gate is therefore validation-cost removal:

```bash
make cosine-n-skip-ablation
```

That run uses a small accept-probe eval and a cosine-gated post-extrapolation
eval policy: high-consistency cycles skip the post eval, mid-consistency cycles
sample it periodically, and low-consistency or large-`N` cycles keep the
rollback probe enabled.

Completed validation-skip diagnostic:

- artifact root: [../../runs/cosine_n_skip_ablation_20260603_083151](../../runs/cosine_n_skip_ablation_20260603_083151)
- summary JSON: [../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json](../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.json)
- summary report: [../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.md](../../runs/cosine_n_skip_ablation_20260603_083151/cosine_n_ablation_summary.md)
- command surface: `make cosine-n-skip-ablation`
- seeds: `42 43 44`
- config: [../../configs/9b_tg_lora_cosine_n_skip_persistent.yaml](../../configs/9b_tg_lora_cosine_n_skip_persistent.yaml)
- accept probe: `eval.accept_eval_examples=1`
- skip policy: high `cos>=0.85` immediate accept, mid `cos>=0.70` periodic post-eval, forced post-eval for `N=20`

Key validation-skip diagnostic values:

- fixed-N `reduction_rate`: `0.54945`
- cosine-driven `N` `reduction_rate`: `0.71407`
- cosine-driven vs fixed-N `reduction_rate` delta: `+0.16462`
- fixed-N rollback rate: `0.10`
- cosine-driven `N` rollback rate: `0.06667`
- fixed-N post-extrapolation evals/skips: `2.33` / `5.00` per seed
- cosine-driven post-extrapolation evals/skips: `4.67` / `1.67` per seed
- fixed-N validation forwards: `22.33` per seed
- cosine-driven validation forwards: `24.67` per seed
- cosine-driven vs fixed-N wall-clock ratio: `1.00006x`
- best valid loss remains matched or slightly better for cosine-driven `N`:
  fixed-N mean `1.13131`, cosine-driven mean `1.12976`

Cycle-level safety check:

- skipped post-extrapolation eval cycles never triggered rollback in the
  recorded JSONL metrics.
- post-extrapolation rollbacks occurred only in low-confidence evaluated cycles.
- `N=20` cycles were force-evaluated and accepted in this run.
- several non-accepted cycles are `linearity_guard` / pilot rollback cases with
  `N=0`; do not conflate these with post-extrapolation rollback.

Interpretation:

- The cosine-driven horizon still improves backward replacement under the skip
  policy, so adaptive `N` remains the correct mechanism.
- The validation-skip policy is conservative and safe in this run, but it does
  not produce a measured wall-clock speedup over fixed-N.
- The reason is now narrower than before: post-extrapolation eval is no longer
  the dominant remaining cost. Both conditions still pay about `20` pilot
  validation forwards per seed and scheduled full validation evals. These fixed
  costs dominate wall-clock.
- Do not claim a 2x measured wall-clock speedup from Component 2. The defensible
  claim is that cosine-driven `N` increases speculative backward replacement,
  and the next engineering target is removing or amortizing pilot/full-eval
  fixed costs.

Next runtime gate:

- run a fixed-cost ablation that reduces scheduled full eval during the short
  runtime benchmark to final-eval-only, while keeping final/best-loss evaluation
  for quality accounting.
- separately test reducing pilot validation frequency or replacing pilot eval
  with a cached/smaller probe. The current `accept_eval_examples=1` already
  makes each probe small, but the frequency remains high.
- if wall-clock still does not move after those changes, inspect checkpoint I/O,
  model reload time, logging, and cache-equivalence checks.

Preliminary final-eval-only smoke:

- artifact root: [../../runs/cosine_n_skip_final_eval_only_20260603_132236](../../runs/cosine_n_skip_final_eval_only_20260603_132236)
- seed: `42`
- setting: `EVAL_POINTS=1`, `accept_eval_examples=1`, validation skip enabled
- baseline wall-clock: `833.7s`
- fixed-N wall-clock: `805.6s`
- cosine-driven wall-clock: `806.2s`
- fixed-N vs baseline wall-clock ratio: `0.9663x`
- cosine-driven vs baseline wall-clock ratio: `0.9670x`
- cosine-driven vs fixed-N wall-clock ratio: `1.0007x`
- fixed-N `reduction_rate`: `0.53846`
- cosine-driven `reduction_rate`: `0.71698`
- cosine-driven rollback rate: `0.0`
- best valid loss remains matched: fixed-N `1.13247`, cosine-driven `1.13229`

Interpretation of the smoke:

- scheduled full eval was the dominant fixed wall-clock cost in the previous
  3-seed skip diagnostic.
- removing most scheduled full eval restores TG runtime to slightly faster than
  the baseline on seed 42, while preserving the cosine-driven `reduction_rate`
  advantage over fixed-N.
- cosine-driven `N` still does not beat fixed-N on wall-clock because both share
  the same pilot validation cost and the same number of real backward passes.
- This is a one-seed diagnostic only. Expand `EVAL_POINTS=1` to 3 seeds before
  making any manuscript-level speed claim.

## Reviewer Defense and Baseline Comparison

To defend Component 2's novelty against standard optimizers (Lookahead optimizer, tuned learning rates, or simple line search), the manuscript must contextualize these comparisons:
- **Lookahead comparison**: Lookahead performs $K$ inner steps, then moves the slow weights towards fast weights ($\theta_{slow} \leftarrow \theta_{slow} + \alpha(\theta_{fast} - \theta_{slow})$). TG-LoRA performs $K$ pilot steps to estimate velocity $v = \theta_{t+K} - \theta_t$, speculatively extrapolates ($\theta_{extrap} = \theta_t + \alpha \cdot v$), checks the proposal on a Calibration/Acceptance set, and performs rollbacks on failure. The difference lies in the **bounded extrapolation, calibration-batch check, and multi-tiered rollback**.
- **Optimizer Lifecycle Confound Resolution**: Previous experiments recreated the optimizer each cycle (`recreate_per_cycle`), which reset momentum and confounded the comparison. Implementing the `persistent` optimizer lifecycle policy ensures that both baseline, cache-only, and TG-LoRA conditions share the same momentum behavior, isolating the extrapolation effect.
- **Additional Baselines**: Ongoing and future sweeps evaluate:
  1. *Cache-Only + tuned LR*: Comparing if simply increasing/tuning the learning rate replicates the validation loss gains.
  2. *Cache-Only + Lookahead*: Comparing TG-LoRA's guarded trajectory extrapolation directly with Lookahead optimizer.
  3. *Cache-Only + Random Perturbation*: Applying a random weight displacement vector of matched norm, followed by the validation acceptance check. This isolates whether the performance gain is due to the trajectory prediction ($u\_bar$ extraction) or simple validation-filtered perturbation search.

These comparisons are evaluated under matched Layer-Prefix Feature Cache environments.

## Alpha-Line Landing Plan

The paper-facing Component 2 implementation now uses the zero-order
`alpha_line_order=0` path as the main result path. The first-order
`alpha_line_order=1` path remains in the codebase for diagnostics, but it is not
used for manuscript-level results because the Qwen 4-bit/bitsandbytes backend
falls back from JVP to finite differences. That fallback is sensitive to the
finite-difference epsilon (`1e-4` is quantization-noisy, `1e-2` rejects in the
smoke, and `1e-3` is usable but has large approximation-error excursions).

The extended-forward / column-concatenation optimization is also not a mainline
path for this paper. The microbenchmark
[../../runs/bench_extended_forward_20260604_084101/bench_summary.md](../../runs/bench_extended_forward_20260604_084101/bench_summary.md)
measured one suffix layer on RTX 3060 / Qwen3.5-9B 4-bit:

- `1x suffix forward`: `193.042 ms`
- `2x extended forward`: `384.445 ms`
- `two independent 1x forwards`: `389.977 ms`
- `r = t(2x) / t(1x) = 1.992`

This is a clear no-go under the `r >= 1.7` rule. The result means that, in the
current environment, extending the token/batch column behaves like two
independent suffix forwards rather than a near-free weight-read reuse. The
paper should therefore keep the alpha-line speed ceiling conservative: the
zero-order path is evaluated as a reproducible forward-only / intermittent
backward mechanism, not as a 1F exact-JVP implementation.

The next main-result evaluation is:

- representative task only, 3 seeds
- baseline: suffix-only last25 with full backward every step
- proposal: zero-order alpha-line with intermittent full backward
- fixed backward budget
- report valid-loss trajectory, wall-clock split (`v_update` backward time vs
  alpha-step time), peak VRAM, accepted/rejected alpha steps, and theoretical
  cost ratio `3(M+1)/(2M+3)`

Future-work sidecar diagnostics are intentionally separated from the main
claim. They may be logged under the `future_work` namespace and summarized with
`scripts/summarize_component2_landing.py`, but only two observations are
eligible for the manuscript's future-work motivation:

1. corpus-redundancy diagnostics from logged loss distributions
2. projection-ratio differences between a style-like task and a knowledge-like task

The `g·v` vs loss-delta correlation is recorded only as internal next-paper
data (`paper_exclude=true`) and should not be shown in this manuscript.

## Prior-based Low-dimensional Coefficient Learning (New Design Decision - 2026-06-05)

### Diagnosis: Implementation Degeneration of Component 2
The evaluation of TG-LoRA's efficiency ceiling at $1.24\times$ (theoretical limit $1.5\times$) has been diagnosed as a degradation in the implementation rather than a limitation of the method. The previous implementation fixed the direction $v$ and heuristically adjusted the step scale at every cycle using high-variance, small-sample loss feedback on the fly. 

### Corrective Design
To resolve this, the design shifts to estimating both the trajectory direction $v$ and the trajectory scale $w_{\text{traj}}$ as a prior from the past update history. The optimization problem is then reduced to learning low-dimensional coefficients $\{\alpha, \beta_j\}$ representing corrections around this prior dynamically from training data.

1. **Prior Estimation**: Track update trajectory to compute prior direction $v$ and prior scale $w_{\text{traj}}$.
2. **Low-Dimensional Space**: Solve for coordinates in the low-dimensional subspace spanned by the prior and auxiliary orthogonal directions.
3. **Directional Derivative & Fallback**: Gradient estimation of coefficients $d\alpha$ and $d\beta_j$ requires directional derivatives. Since JVP (Jacobian-Vector Product) is unsupported in 4-bit Qwen/bitsandbytes backends, we fall back to finite differences.
4. **Numerical Conditioning Regularization**:
   - **Direction Normalization**: Unit-normalize the direction vectors to prevent scale disparity.
   - **Dimensionless Scale**: Use $w_{\text{traj}}$ to non-dimensionalize the step size, making coefficient scales comparable.
   - **Auxiliary Orthogonalization**: Orthogonalize auxiliary directions to prevent linear dependence and minimize finite-difference approximation errors.

### Implementation Checklist
- [ ] Perform offline validation to confirm the mathematical validity of the subspace prior approximation before runtime implementation.
- [ ] Implement finite-difference fallback with numerical conditioning.
- [ ] Measure the projection efficiency of trajectories onto the $\{\alpha, \beta_j\}$ subspace.

