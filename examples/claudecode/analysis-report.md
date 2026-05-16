# PINN Analysis Report: 2D Steady-State Heat Conduction

**Runs analysed:** standard run (N_r=4096, Adam+L-BFGS) + collocation sweep (N_r ∈ {256, 1024, 4096, 16384})
**Data files:** `training_log.csv`, `verification.json`, `sweep_collocation.json`
**Hardware:** Apple MPS, `torchmps` conda env, seed=42

> **Data quality note:** The flat `outputs/checkpoint.pt`, `training_log.csv`, and `verification.json`
> reflect the final sweep run (N_r=16384), not the standard run, because the output-directory
> restructuring landed after the runs completed. The standard run's terminal output is preserved in
> `run_standard.log`. All quantitative claims below are traceable to the files listed above.

---

## 1. Headline

The PINN met the primary success criterion on 3 of 4 collocation counts (256, 4096, 16384 pts),
achieving relative L2 error below 1×10⁻³ against the analytic reference — but the collocation
scaling hypothesis (H3) is **refuted**: error is not monotone and the sweet spot is 4096 pts,
not the largest tested count.

---

## 2. Hypothesis verdicts

| # | Hypothesis (short) | Threshold | Measured | Verdict | Confidence note |
|---|---|---|---|---|---|
| H1 | Adam-only rel-L2 < 1e-3 at N_r=4096 | < 1e-3 | **not measured** | Inconclusive | No Adam-only checkpoint saved; H1 is strictly untestable from current outputs |
| H2 | L-BFGS reduces rel-L2 by ≥5× vs Adam-only | 5× reduction | End-Adam total loss 1.84e-4 → L-BFGS 6.92e-6 (**26.6× on total loss**); rel-L2 not directly comparable without Adam-only checkpoint | Inconclusive | Total loss reduction confirmed large, but rel-L2 comparison requires separate Adam-only run |
| H3 | rel-L2 decreases monotonically 256→16384 | monotone ↓ | 9.82e-4 → **1.07e-3** → 6.94e-4 → 9.88e-4 | **Refuted** | Single seed; all four counts run with identical config |
| H4 | w_bc=10 reduces boundary residual ≥10× vs w_bc=1 | ≥10× reduction | **not measured** | Inconclusive | No w_bc=1 baseline run performed |

---

## 3. Training dynamics

*Source: `training_log.csv` (N_r=16384 sweep run; 10 000 Adam + ~1000 L-BFGS closure evaluations logged)*

**Loss at key milestones:**

| Iteration | Total | PDE (weighted) | BC (weighted) | BC/PDE ratio |
|---|---|---|---|---|
| 1 | 7.20e-1 | 5.38e-3 | 7.15e-1 | 133× |
| 1 000 | 1.38e-1 | 3.01e-2 | 1.08e-1 | 3.6× |
| 5 000 | 3.50e-4 | 1.52e-4 | 1.98e-4 | 1.3× |
| 10 000 (end Adam) | 1.84e-4 | 1.19e-4 | 6.47e-5 | 0.54× |
| ~11 000 (end L-BFGS) | 6.92e-6 | 2.64e-6 | 4.28e-6 | 1.6× |

**Convergence milestones:**
- 50% of initial loss reached: ~iteration 200
- 10% of initial loss reached: ~iteration 1300
- 1% of initial loss reached: ~iteration 2700

**Oscillatory behaviour (significant):** Adam training beyond step 2000 shows repeated loss spikes
of 10–50× above surrounding values. Notable examples: step 3000 (0.0133 vs 0.00237 at step 2900),
step 4000 (0.0218 — a 28× spike), step 7200 (0.00875), step 7800 (0.0105).
These spikes recur roughly every 500–1000 steps throughout the Adam phase.
This pattern is consistent with Adam at a fixed lr=1e-3 becoming unstable once the loss
enters the 1e-3–1e-4 range. The optimizer repeatedly overshoots a local minimum and recovers.

**L-BFGS phase:** Clean monotone decrease from 1.71e-4 to 6.92e-6 across ~1000 closure evaluations.
No stalls, no loss increases. L-BFGS reduced total loss by ~26.6×, confirming it is doing most of
the fine-scale work.

---

## 4. Error analysis

**Collocation sweep — full comparison:**

| N_r | Rel L2 | Max err | BC MSE (top) | Max PDE res | Pass? |
|---|---|---|---|---|---|
| 256 | 9.82e-4 | 1.24e-3 | 2.26e-7 | 1.61e-2 | ✓ |
| 1024 | **1.07e-3** | 1.45e-3 | 1.31e-7 | 1.14e-2 | ✗ |
| 4096 | **6.94e-4** | 8.79e-4 | 9.96e-8 | 1.11e-2 | ✓ ← best |
| 16384 | 9.88e-4 | 1.34e-3 | 1.23e-7 | 6.35e-3 | ✓ |

- The **4096-pt run achieves the best rel-L2 (6.94e-4)** — 1.4× better than 256, and better than 16384.
- The **1024-pt run is the only failure** (1.07e-3), missing the threshold by 7%.
- **BC MSE is excellent across all runs** (1e-7 to 3e-7), well below the 1e-6 criterion. Boundary
  enforcement is not the bottleneck.
- **Max PDE residual decreases with N_r** (1.61e-2 → 6.35e-3): more collocation points do enforce the
  PDE more uniformly, even when rel-L2 doesn't improve proportionally. This is the one metric
  that scales as expected.
- Max pointwise error is consistently ~30% higher than rel-L2 error, suggesting the worst errors
  are spatially concentrated rather than spread uniformly. For this BC geometry (sin(πx) on top,
  zero elsewhere), the most likely location is near the top corners (x≈0, y≈1) and (x≈1, y≈1)
  where the BC transitions sharply from sin(πx)→0 to zero-BC walls. This is a known corner
  singularity in the gradient of the analytic solution.

---

## 5. Failure-mode audit

**BC drift:** Not present at convergence. At step 1, BC loss was 133× PDE loss — by design (w_bc=10
amplifies a larger raw BC error). By end of Adam, the ratio inverts (PDE 1.8× BC). At L-BFGS end,
roughly balanced (1.6×). The w_bc=10 setting successfully prevented boundary drift. ✓

**Loss imbalance at convergence:** PDE=2.64e-6, BC=4.28e-6, ratio=1.6×. Well within acceptable range.
No remediation needed. ✓

**Oscillatory Adam loss (flagged):** Recurrent spikes throughout Adam steps 2000–10000 indicate
lr=1e-3 is too large once loss enters 1e-3 territory. The optimizer is bouncing in a flat region
rather than converging. L-BFGS rescues the run, but this is inefficient — many Adam steps after
step ~2500 are wasted. **Remediation: add learning rate decay (e.g., cosine annealing or
ReduceLROnPlateau) or switch to L-BFGS earlier (after 3000–5000 Adam steps).**

**Spectral bias near x=0, x=1:** The max pointwise error exceeds rel-L2 by ~30–40% across all
runs. The top-wall BC T=sin(πx) reaches zero at both endpoints, creating sharp gradient transitions
at corners. These corners are the most likely high-error regions — consistent with spectral bias
in tanh networks near zero-crossings. The verification plot would confirm exact location.
**Remediation: add corner collocation points or use Fourier features at frequency π.**

**Saddle-point stall in L-BFGS:** Not observed. L-BFGS is monotone and clean in all logged rows. ✓

**MPS numerical precision:** No NaN, no divergence, runs completed successfully on all four sweep
counts. ✓

---

## 6. What this run does NOT tell us

- **H1 (Adam-only) is untestable:** No Adam-only checkpoint was saved. The scaffold should add
  a `--save-adam-checkpoint` flag to capture the model state before L-BFGS begins.

- **Single seed:** All runs used seed=42. PINN training with Adam is noisy — the 1024-pt run
  failing by 7% may be a seed artefact rather than a true effect of collocation count. A 3-seed
  repeat at 256 and 1024 pts would clarify whether the non-monotonicity is real.

- **Wall time vs. spec:** The spec predicted < 5 min. Actual times were ~25 min (4096 pts) to
  ~2 h (16384 pts). The bottleneck is `torch.func.hessian + vmap`, which scales linearly with N_r.
  The spec timing assumption was based on a 256-pt smoke test and did not account for this scaling.

- **Evaluation grid resolution:** The 200×200 grid (40 000 points) may miss fine error structure
  near corners. If the worst errors are in O(1/200)-wide boundary layers, the reported max error
  could be underestimated.

- **No FEM baseline:** The spec listed a FD/FEM comparison as a stretch goal; it was not
  implemented. Without it, we cannot compare PINN accuracy-per-training-cost against a classical
  solver.

- **No w_bc sensitivity data:** H4 is entirely unmeasured. The spec's w_bc=10 default may be
  under- or over-tuned.

---

## 7. Recommended next steps

**N1 — Save Adam-only checkpoint (enables H1 and H2):**
Add `--save-adam-checkpoint` to `run.py`. After step N_adam, save `checkpoint_adam.pt` and run
`verify.py` on it. Hypothesis: Adam-only rel-L2 at N_r=4096 is between 1e-2 and 5e-3; L-BFGS
brings it below 1e-3, confirming H2 with a measured ratio > 5×.

**N2 — Repeat 256 and 1024 sweeps with 3 seeds (tests whether H3 failure is real):**
Run N_r ∈ {256, 1024, 4096} × seeds ∈ {42, 7, 123}. Hypothesis: the 1024-pt failure is
seed-dependent (variance across seeds overlaps with the 1e-3 threshold), and the true
population mean follows a monotone trend. If confirmed, the non-monotonicity is noise, not
a structural phenomenon.

**N3 — Apply cosine LR decay during Adam phase (addresses oscillation):**
Replace fixed lr=1e-3 with cosine annealing from 1e-3 to 1e-5 over 10 000 steps. Hypothesis:
loss spikes disappear after step 2000 and final Adam loss is at least 3× lower than with fixed
lr, reducing the refinement burden on L-BFGS.

**N4 — Add Fourier input features at frequency π (addresses spectral bias):**
Replace raw (x, y) inputs with [sin(πx), cos(πx), sin(πy), cos(πy), x, y]. Hypothesis: max
pointwise error near corners drops by at least 2× compared to plain MLP, without increasing
training time by more than 20%.
