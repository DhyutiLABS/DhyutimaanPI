# PINN Problem Spec: 2D Heat Conduction (Steady + Unsteady) — DoE

## 1. Problem Statement

We solve the two-dimensional heat (diffusion) equation on the unit square using physics-informed neural networks (PINNs), and characterise the empirical effect of three training interventions in a 2³ full-factorial design of experiments: (i) hard-constraint output transform vs. soft boundary-condition loss, (ii) L-BFGS polishing (steady) / causal temporal weighting (unsteady) vs. Adam-only / standard weighting, and (iii) random Fourier-feature input embedding vs. plain coordinate inputs. Two sub-problems are studied: a steady Poisson form (Problem A) and an unsteady parabolic form (Problem B). Both use manufactured solutions with closed-form references, enabling exact error measurement. The goal is to produce a clean, falsifiable measurement of which interventions actually buy accuracy on this canonical benchmark under a fixed compute budget (wall-clock parity), and to identify at what point hard-constraint ansatz constructions become ill-conditioned for parabolic problems.

---

## 2. Governing Equations

### Problem A — Steady 2D Heat (Poisson Form)

Find $u : \Omega \to \mathbb{R}$, $\Omega = (0,1)^2$, satisfying

$$-\Delta u(x,y) = f(x,y) \quad \text{on } \Omega$$
$$u = 0 \quad \text{on } \partial\Omega$$

with source term

$$f(x,y) = 2\pi^2 \sin(\pi x)\sin(\pi y)$$

### Problem B — Unsteady 2D Heat (Parabolic Form)

Find $u : \Omega \times [0,T] \to \mathbb{R}$, $\Omega = (0,1)^2$, $T = 0.1$, $\alpha = 1$, satisfying

$$u_t - \alpha (u_{xx} + u_{yy}) = 0 \quad \text{on } \Omega \times (0,T]$$
$$u(x,y,0) = \sin(\pi x)\sin(\pi y) \quad \text{on } \Omega$$
$$u(x,y,t) = 0 \quad \text{on } \partial\Omega \times [0,T]$$

**Symbol glossary:** $x, y$ — spatial coordinates; $t$ — time; $\alpha$ — thermal diffusivity; $\Delta = \partial_{xx} + \partial_{yy}$ — Laplacian; $f$ — volumetric source.

---

## 3. Domain and Boundary / Initial Conditions

| Quantity | Problem A | Problem B |
|---|---|---|
| Spatial domain $\Omega$ | $(0,1)^2$ | $(0,1)^2$ |
| Temporal domain | — | $[0, 0.1]$ |
| BC type | Homogeneous Dirichlet on all 4 sides | Homogeneous Dirichlet on all 4 sides $\times [0,T]$ |
| IC | — | $u(x,y,0) = \sin(\pi x)\sin(\pi y)$ |
| BCs (explicit) | $u(0,y)=u(1,y)=u(x,0)=u(x,1)=0$ | same, plus $u(x,y,0)=\sin(\pi x)\sin(\pi y)$ |

---

## 4. Reference / Ground Truth

### Problem A — Analytic solution (exact)

$$u_{\text{ref}}(x,y) = \sin(\pi x)\sin(\pi y)$$

Verification: this satisfies $-\Delta u = 2\pi^2 \sin(\pi x)\sin(\pi y)$ with $u=0$ on $\partial\Omega$.

### Problem B — Analytic solution (exact)

$$u_{\text{ref}}(x,y,t) = \sin(\pi x)\sin(\pi y) \exp(-2\pi^2 \alpha t)$$

Verification: exponential decay of the first eigenmode of the Laplacian; satisfies parabolic PDE, IC, and BCs exactly.

**Evaluation grid:** $100 \times 100$ spatial grid (for Problem A); $100 \times 100 \times 11$ space-time grid for Problem B (11 time slices at $t \in \{0, 0.01, \ldots, 0.1\}$).

**Error metric (primary):** Relative $L^2$ error

$$\varepsilon = \frac{\|u_{\text{pred}} - u_{\text{ref}}\|_2}{\|u_{\text{ref}}\|_2}$$

**Secondary metric:** Max absolute error on the evaluation grid.

---

## 5. Architecture and Training Plan

### Network Architecture

| Component | Setting |
|---|---|
| Base architecture | MLP, 4 hidden layers × 64 units, tanh activation |
| Inputs (Problem A) | $(x, y)$ — 2 inputs |
| Inputs (Problem B) | $(x, y, t)$ — 3 inputs |
| Output | $\hat{u}$ — scalar |
| Parameters (approx.) | ~20k (A), ~21k (B) |
| Fourier variant | Random Fourier features: $\gamma(\mathbf{x}) = [\cos(2\pi B\mathbf{x}), \sin(2\pi B\mathbf{x})]$, $B \sim \mathcal{N}(0, \sigma^2 I)$, $\sigma=1$, 32 frequencies → 64 input features; fixed during training |

### Hard-Constraint Ansatze

**Problem A — hard BC:**
$$u(x,y) = x(1-x)\,y(1-y) \cdot \text{NN}(x,y;\theta)$$
BC loss term dropped from training objective.

**Problem B — hard IC+BC (variant B-hard):**
$$u(x,y,t) = \sin(\pi x)\sin(\pi y) + t \cdot x(1-x)\,y(1-y) \cdot \text{NN}(x,y,t;\theta)$$
*Warning: this ansatz implies a singular NN target near $(x,y) \to (0,0), (1,1)$ at $t=0$ — see §8.*

### Collocation Sampling

| Points | Problem A | Problem B |
|---|---|---|
| Interior collocation | 4 096 (LHS) | 8 192 (LHS in $x,y,t$) |
| Boundary points | 400 (100 per side) | 800 ($\times T$ uniform) |
| IC points | — | 1 024 |
| Evaluation grid | $100 \times 100$ | $100 \times 100 \times 11$ |

Sampling: Latin Hypercube Sampling (LHS) for interior; uniform for boundaries. Fixed (no resampling).

### Loss Functions

**Soft-constraint variants:**

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + w_{\text{bc}}\,\mathcal{L}_{\text{BC}} + w_{\text{ic}}\,\mathcal{L}_{\text{IC}}$$

- Soft baseline: $w_{\text{bc}} = w_{\text{ic}} = 1$
- Soft-weighted: $w_{\text{bc}} = w_{\text{ic}} = 100$

**Hard-constraint variants:** $\mathcal{L} = \mathcal{L}_{\text{PDE}}$ only.

**Causal weighting (Problem B, unsteady):**

Sort residual points by time; weight PDE loss at time $t_i$ by $w_i = \exp\!\left(-\varepsilon \sum_{j: t_j < t_i} \mathcal{L}_{\text{PDE}}(t_j)\right)$, $\varepsilon = 100$.

### Optimiser Schedule

| Stage | Optimiser | LR | Steps |
|---|---|---|---|
| Stage 1 (all variants) | Adam | 1e-3 | 5 000 |
| Stage 2 — L-BFGS variants only | L-BFGS | line-search | 500 iterations |

**Hardware target:** PyTorch MPS (Apple Silicon) with CPU fallback.

---

## 6. Design of Experiments

Full $2^3$ factorial. Factor levels:

| Factor | Level 0 (−) | Level 1 (+) |
|---|---|---|
| **F1**: Constraint | Soft ($w_{\text{bc}}=100$) | Hard output-transform |
| **F2**: Optimiser/temporal | Adam-only / std. weighting | +L-BFGS (steady) / causal weighting (unsteady) |
| **F3**: Fourier features | No (plain coords) | Yes ($\sigma=1$, 32 freq) |

### Run Table — Problem A (Steady)

| Run | F1 | F2 | F3 | Label |
|---|---|---|---|---|
| A1 | Soft | Adam-only | No | soft-adam |
| A2 | Soft | +L-BFGS | No | soft-lbfgs |
| A3 | Hard | Adam-only | No | hard-adam |
| A4 | Hard | +L-BFGS | No | hard-lbfgs |
| A5 | Soft | Adam-only | Yes | soft-adam-ff |
| A6 | Soft | +L-BFGS | Yes | soft-lbfgs-ff |
| A7 | Hard | Adam-only | Yes | hard-adam-ff |
| A8 | Hard | +L-BFGS | Yes | hard-lbfgs-ff |

### Run Table — Problem B (Unsteady)

| Run | F1 | F2 | F3 | Label |
|---|---|---|---|---|
| B1 | Soft | Std. weight | No | soft-std |
| B2 | Soft | Causal | No | soft-causal |
| B3 | Hard | Std. weight | No | hard-std |
| B4 | Hard | Causal | No | hard-causal |
| B5 | Soft | Std. weight | Yes | soft-std-ff |
| B6 | Soft | Causal | Yes | soft-causal-ff |
| B7 | Hard | Std. weight | Yes | hard-std-ff |
| B8 | Hard | Causal | Yes | hard-causal-ff |

Total: **16 runs**. Each run saves to `runs/<label>/` with `training_log.csv`, `checkpoint.pt`, `verification.json`.

---

## 7. Hypotheses (Falsifiable)

**H1 — Hard constraint dominates (Problem A, steady).**
Hard-constraint runs (A3, A4, A7, A8) achieve relative $L^2$ error ≥100× lower than their soft-constraint counterparts (A1, A2, A5, A6) within the same Adam step budget (5 000 steps). *Confirmed if: mean($\varepsilon_{\text{soft-adam}}) / \varepsilon_{\text{hard-adam}} \geq 100$.*

**H2 — L-BFGS polishing provides meaningful improvement under hard constraint.**
Adding 500 L-BFGS steps to the hard-constraint run (A4 vs. A3) reduces relative $L^2$ error by ≥5×. *Confirmed if: $\varepsilon_{\text{hard-adam}} / \varepsilon_{\text{hard-lbfgs}} \geq 5$.*

**H3 — Fourier features hurt smooth low-frequency solutions.**
For the hard-constraint Adam baseline, adding Fourier features (A7 vs. A3) increases relative $L^2$ error by ≥2×. *Confirmed if: $\varepsilon_{\text{hard-adam-ff}} / \varepsilon_{\text{hard-adam}} \geq 2$.*

**H4 — Causal weighting reduces temporal error growth (Problem B).**
Per-time-slice relative $L^2$ error growth (ratio of $\varepsilon$ at $t=T$ to $\varepsilon$ at $t=0$) is ≤50% of the soft-standard baseline growth for causal-weighted runs. *Confirmed if: growth$_{\text{causal}} \leq 0.5 \times$ growth$_{\text{std}}$.*

**H5 — Hard IC ansatz fails for Problem B due to boundary singularity.**
Run B3 (hard std.) achieves worse final relative $L^2$ error than B1 (soft std.) — the hard ansatz introduces a singular NN target near $\partial\Omega$ at $t=0$ that degrades optimisation. *Confirmed if: $\varepsilon_{\text{hard-std}} > \varepsilon_{\text{soft-std}}$.*

---

## 8. Success Criteria

| Problem | Metric | Target |
|---|---|---|
| A — best run | Relative $L^2$ error | ≤ 1e-4 |
| A — soft baseline | Relative $L^2$ error | ≤ 5e-2 |
| B — best run | Relative $L^2$ error (global) | ≤ 5e-3 |
| B — causal run | Per-slice $L^2$ growth ratio ($t=T$ / $t=0$) | ≤ 2 |
| All runs | Loss components logged separately in `training_log.csv` | Required |
| All runs | `verification.json` written with `rel_l2`, `max_abs_error`, hypothesis verdicts | Required |

---

## 9. Known Failure Modes to Watch For

1. **Loss-component imbalance (A1, B1):** PDE gradient >> BC/IC gradient → network fits PDE while ignoring BCs. Check: $\mathcal{L}_{\text{BC}} / \mathcal{L}_{\text{PDE}}$ ratio at step 500. Flag if > 10.

2. **Spectral bias + Fourier feature Gibbs overshoot (A5–A8, B5–B8):** On the smooth sin·sin target, Fourier features can introduce high-frequency error. Check: error distribution on evaluation grid — look for oscillatory residual not aligned with the true solution.

3. **Singular NN target in hard-BC ansatz for Problem B (B3, B4, B7, B8):** The envelope $x(1-x)y(1-y)$ vanishes faster than $\sin(\pi x)\sin(\pi y)$ near the boundary, implying a divergent NN target. Monitor: PDE loss plateau > 1.0 at step 2000 indicates this failure.

4. **Causality violation (B1, B3, B5, B7 — standard weighting):** Later-time residuals are fit before early-time residuals converge. Check: plot per-slice $L^2$ error vs. time — if slope is positive the network is fitting late-time data first.

5. **Adam noise plateau without L-BFGS (all Adam-only runs):** Adam loss may still be decreasing at step 5000 for some runs; do not conflate "low loss" with "converged." Report best-during-training $\varepsilon$ alongside final-step $\varepsilon$.

---

## 10. Out of Scope

- Non-homogeneous or time-varying Dirichlet BCs
- Neumann or Robin boundary conditions
- Domains other than the unit square
- Source terms with high-wavenumber content
- Inverse problems (parameter identification)
- Neural-operator comparisons (DeepONet, FNO)
- XPINN / domain-decomposition variants
- $\alpha \neq 1$ or $T > 0.1$
- 3D heat conduction
