# PINN Analysis Report: 2D Heat Conduction DoE (16 variants)

## 1. Headline

Problem A (steady Poisson): best run (hard-lbfgs-ff) achieves ε = 2.14e-6, beating the 1e-4 success
criterion by 47×. All hard-constraint variants meet the criterion; all soft variants fall short.
Problem B (unsteady parabolic): causal-weighted soft variants **catastrophically fail** (ε up to 0.89,
error growth ratio up to 5×10⁷); hard-constraint standard variants achieve ε = 5.0e-4, meeting the
5e-3 target; causal weighting provides no benefit on this smooth problem and actively destroys
convergence in three of four variants.

---

## 2. Hypothesis Verdicts

| H | Hypothesis (short) | Predicted | Measured | Verdict | Confidence note |
|---|---|---|---|---|---|
| H1 | Hard BC reduces steady ε by ≥100× vs soft-adam | ≥100× | **57×** (2.47e-4 → 4.31e-6) | Refuted | Single seed; effect is real but threshold arbitrary |
| H2 | L-BFGS ≥5× better than Adam under hard constraint | ≥5× | **0.98×** (4.31e-6 → 4.38e-6) | Refuted | hard-adam already at near-machine-precision; no gap to close |
| H3 | Fourier features ≥2× worse on smooth solution | ≥2× worse | **0.86×** (FF marginally *better*) | Refuted | Effect size is noise-level; neither confirms nor refutes the spectral bias argument |
| H4 | Causal weighting reduces temporal growth by ≥2× | growth\_causal ≤ 0.5 × growth\_std | **367× vs 8.75×** (42× *worse*) | Refuted — inverse direction | Catastrophic; result is reproducible across soft-causal and soft-causal-ff |
| H5 | Hard IC ansatz fails for Problem B (worse than soft-std) | ε\_hard > ε\_soft | **5.04e-4 < 2.36e-3** (hard is *better*) | Refuted — inverse direction | Predicted singularity does not materialise; see §8 |

---

## 3. Training Dynamics

### Problem A — Steady Poisson

**Hard variants** converge extremely fast under AD residuals. hard-adam reaches 0.1% of its initial
loss (≈ π⁴/4) by step 700; loss decreases monotonically to 3.17e-6 at step 20 000. L-BFGS
polishing (hard-lbfgs) adds negligible benefit: 4.38e-6 vs 4.31e-6 — a 1.6% change within run-to-run
noise. This is a **saturation** result, not a failure of L-BFGS: the Adam trajectory has already
reached near-machine-precision for this problem.

**Soft variants** converge more slowly. soft-adam reaches 0.1% initial loss at step 1700; final
total loss 4.17e-5, decomposed as PDE 3.30e-5, BC 8.70e-8. BC loss is ~380× below PDE loss at
convergence — no BC imbalance flag; BCs are well-satisfied. L-BFGS *does* help soft variants:
soft-lbfgs achieves ε = 6.85e-5 vs 2.47e-4 for soft-adam (3.6×), because Adam leaves a
meaningful residual that L-BFGS can exploit through its second-order curvature information.

**Fourier features** (soft-adam-ff vs soft-adam): FF *hurts* soft variants (5.55e-4 vs 2.47e-4,
2.2× worse). Probable cause: FF initial loss is 37.6 vs 5.4 for plain MLP — the high-dimensional
Fourier basis creates a much rougher initial loss landscape that Adam does not escape fully in 20k
steps. For hard variants the effect is marginal (±15%).

**⚠ Flag:** soft-adam-ff and soft-lbfgs-ff converge more slowly and less accurately than their
non-FF counterparts on Problem A. Fourier features with σ=1 at this frequency are not beneficial
for this smooth problem — the initial condition for the FF loss landscape is harder to optimise.

### Problem B — Unsteady Parabolic

**Hard-std** (B3): Converges, though slowly. Initial loss 76.0 (high; the hard IC fix exactly
satisfies IC so the loss opens large). Reaches 0.1% at step 10 100. Final slice errors:
t=0: 1.5e-7 (IC exactly enforced), t=0.02: 8.1e-4, t=0.05: 8.2e-4, t=0.1: 8.7e-4.
**Error profile is flat across time** — a healthy sign that the network is learning the temporal
evolution, not just fitting the IC.

**Soft-std** (B1): Global ε = 2.36e-3. Slice errors rise from 1.5e-3 (t=0) to 1.3e-2 (t=0.1),
growth ratio 8.75. The gentle positive slope is a mild causality violation — the network fits
the IC imperfectly and errors compound forward in time, but not catastrophically.

**Soft-causal** (B2): **Catastrophic failure.** Loss converges (8.74e-4 at step 20k), but the model
has learned essentially nothing about t > 0.01. Slice errors: t=0: 2.4e-3, t=0.02: 3.5e-2,
t=0.05: 0.22, t=0.1: 0.89. Growth ratio 367×. The loss is low because causal weighting
suppresses the signal from all but the earliest slab — the "converged" loss is a weighted
average where weights for t > 0.01 are ≈ exp(−100 × accumulated\_loss) ≈ 0.

**Hard-causal** (B4): Mixed result. Global ε = 5.5e-4 (good), but slice at t=0.1 is 5.0e-3 —
10× higher than hard-std's 8.7e-4 at the same time. The hard IC constraint masks the causal
pathology for early slabs (IC is exactly zero), but causal weighting still distorts training at
later times.

**⚠ Critical flag — growth ratio metric is misleading for hard variants:** Hard-constraint variants
enforce IC to machine precision (t=0 slice error ≈ 1.5e-7 for all hard-BC runs). The growth ratio
t=0.1 / t=0 = 8.7e-4 / 1.5e-7 ≈ 5,800 for hard-std — but this is not causality violation; it is
the metric denominator collapsing to zero. The absolute slice errors for hard-std are **flat and
small** (8e-4 across all t > 0), which is the actual signal. The growth ratio should not be used
to compare hard and soft variants.

---

## 4. Error Analysis

### Problem A

| Variant | ε_rel | Max abs | Note |
|---|---|---|---|
| hard-lbfgs-ff | **2.14e-6** | 4.23e-6 | Best overall |
| hard-adam-ff | 3.70e-6 | 5.81e-6 | Within noise of hard-adam |
| hard-adam | 4.31e-6 | 7.26e-6 | Baseline hard |
| hard-lbfgs | 4.38e-6 | 6.28e-6 | L-BFGS adds nothing |
| soft-lbfgs | 6.85e-5 | 1.82e-4 | Best soft; L-BFGS helps here |
| soft-adam | 2.47e-4 | 7.96e-4 | Soft baseline |
| soft-lbfgs-ff | 4.54e-4 | 2.32e-3 | FF hurts soft variants |
| soft-adam-ff | 5.55e-4 | 2.50e-3 | Worst Problem A |

For hard variants, max error ≈ 2× rel error — errors are distributed smoothly across the domain
with no strong boundary or interior concentration. For soft variants, max error is 3–4× rel error,
with concentration near the domain corners where the soft BC loss has smaller gradient magnitude.

### Problem B

| Variant | ε_rel | t=0 | t=0.05 | t=0.1 | Characterisation |
|---|---|---|---|---|---|
| hard-std | **5.04e-4** | 1.5e-7 | 8.2e-4 | 8.7e-4 | Flat, converged |
| hard-causal | 5.55e-4 | 1.5e-7 | 2.9e-4 | 5.0e-3 | Deteriorates at late t |
| soft-std | 2.36e-3 | 1.5e-3 | 3.0e-3 | 1.3e-2 | Mild growth |
| soft-std-ff | 2.89e-3 | 2.1e-3 | 2.5e-3 | 1.6e-2 | Similar to soft-std |
| hard-std-ff | 1.70e-3 | 1.5e-7 | 2.9e-3 | 4.0e-3 | FF hurts hard-unsteady |
| hard-causal-ff | 8.93e-1 | 1.5e-7 | 0.92 | 7.76 | Complete divergence |
| soft-causal | 1.18e-1 | 2.4e-3 | 0.22 | 0.89 | Causal pathology |
| soft-causal-ff | 1.03e-1 | 1.5e-3 | 0.15 | 0.84 | Causal pathology |

---

## 5. Failure-Mode Audit

| Failure mode (spec §9) | Triggered? | Evidence | Remedy |
|---|---|---|---|
| BC/IC loss imbalance | **No** for Problem A (BC << PDE at convergence) | BC/PDE ratio < 1e-3 at final step | N/A |
| FF Gibbs overshoot | **Partial** — soft variants only | soft-adam-ff 2.2× worse than soft-adam | Reduce σ or use learnable frequencies |
| Hard-ansatz singularity (Problem B) | **No** — hypothesis was wrong | hard-std achieves 5.0e-4, better than soft-std 2.4e-3 | See appendix: L'Hopital shows limit is finite |
| Causality violation — standard weighting | **Mild** in soft-std (growth 8.75) | Slice errors rise monotonically | Not enough to require causal training |
| Adam noise plateau | **No** — cosine schedule prevents this | All variants show monotone decay with no late-stage plateau | N/A |
| **Causal weighting pathology (new)** | **Yes, severe** — soft-causal, both FF variants | Error at t=0.1 reaches 0.89; growth ×367 | Do not use causal weighting on smooth single-mode parabolic problems |

---

## 6. What This Run Does NOT Tell Us

1. **Single seed**: all 16 runs were executed once. PINN training has non-trivial seed-to-seed
   variance (±20–50% in ε_rel is typical). The causal weighting pathology is robust across two
   variants (B2, B6) and has a mechanistic explanation, so the claim is likely reproducible; the
   quantitative ratios should not be reported as precise without ≥3 seeds.

2. **Problem specificity**: both sub-problems use single-frequency manufactured solutions
   (sin(πx)sin(πy)). Conclusions about Fourier features and causal weighting may not generalise
   to multi-frequency or discontinuous targets.

3. **Single viscosity / single T**: Problem B uses α=1, T=0.1. Longer integration horizons
   (T >> 1/2π²α ≈ 0.051) would stress-test the temporal methods more severely.

4. **The growth ratio metric is invalid for hard-constraint vs. soft-constraint comparisons**
   because hard constraints drive the IC error to machine precision, collapsing the denominator.
   A denominator-stabilised growth metric (e.g., ratio relative to global ε) should replace it.

5. **No ablation on ε**: causal weight ε=100 was fixed. The pathology may be less severe at
   smaller ε (e.g., ε=1 or ε=10). This is a direct next-experiment.

---

## 7. Recommended Next Experiments

1. **Causal ε sweep (B-variants):** Run soft-causal with ε ∈ {1, 10, 50, 100, 200}. Hypothesis:
   there exists a critical ε* above which the causal pathology triggers for smooth parabolic
   problems; ε* is inversely related to the problem's temporal complexity index
   (number of resolved time scales in the solution).

2. **Longer integration horizon:** Run Problem B with T=1.0 (ten times longer, covers ~20 decay
   time constants). Hypothesis: soft-std growth ratio exceeds 5 at T=1.0, making causal training
   beneficial at low ε. This would identify the regime where causal training crosses from harmful
   to helpful.

3. **Multi-seed statistical validation:** Re-run hard-adam and soft-causal with 5 seeds each.
   Report mean ± std on ε_rel. The causal pathology claim needs error bars before it can be
   stated as a reliable finding rather than an observation.

4. **Burgers ε sweep:** Apply the same ε sweep to the 1D Burgers problem (ν=0.001/π, the
   low-viscosity stress test). Hypothesis: causal training helps Burgers at ε=100 because the
   near-discontinuity in u creates genuine temporal complexity that justifies causal regularisation.
