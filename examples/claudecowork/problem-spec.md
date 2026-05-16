# Problem Specification: PINN for 2D Heat Conduction

## 1. Motivation and goal

Solve the two-dimensional heat (diffusion) equation with a physics-informed neural network and characterize the empirical effect of three commonly recommended interventions from the literature: (i) hard-constraint output transforms for Dirichlet BCs, (ii) Fourier-feature input embeddings for spectral-bias mitigation, and (iii) loss-component reweighting (BC-loss upweighting) when soft constraints are used. The goal is *not* to beat a finite-difference solver — that battle is lost on a unit square — but to produce a clean, falsifiable measurement of which interventions actually buy accuracy on this canonical problem.

## 2. Governing equations

### Problem A — Steady 2D heat conduction (Poisson form)

Find `u: Ω → R` with `Ω = (0, 1)²` satisfying

```
-Δu(x, y)  =  f(x, y)         on Ω
       u   =  0               on ∂Ω
```

with source term `f(x, y) = 2 π² sin(π x) sin(π y)` chosen so that the closed-form reference solution is

```
u_ref(x, y) = sin(π x) sin(π y)
```

(Standard manufactured solution, used in DeepXDE Poisson tutorial and dozens of PINN papers.)

### Problem B — Unsteady 2D heat conduction (parabolic form)

Find `u: Ω × [0, T] → R` with `Ω = (0, 1)²`, `T = 0.1`, `α = 1`, satisfying

```
u_t  -  α (u_xx + u_yy)  =  0       on Ω × (0, T]
            u(x, y, 0)   =  sin(π x) sin(π y)
            u(x, y, t)   =  0       on ∂Ω × [0, T]
```

The closed-form reference is

```
u_ref(x, y, t) = sin(π x) sin(π y) exp(-2 π² α t)
```

(Heat-kernel-style exponential decay of the first eigenmode; canonical verification case.)

## 3. Domain and discretization

| Quantity                        | Problem A           | Problem B                     |
|--------------------------------|---------------------|-------------------------------|
| Spatial domain                  | (0, 1)²             | (0, 1)²                       |
| Temporal domain                 | —                   | [0, 0.1]                      |
| Interior collocation points     | 4 096 (Latin hypercube + uniform mix) | 8 192 in (x,y,t)             |
| Boundary collocation points     | 512 (128 per edge)  | 1 024 (in space × time)       |
| Initial-condition points        | —                   | 1 024 in (x, y) at t=0        |
| Reference grid for L2 error     | 101 × 101 uniform   | 51 × 51 × 11 uniform          |

## 4. Network architecture (baseline)

- MLP, 4 hidden layers, width 50, smooth activation (`tanh`).
- Inputs: `(x, y)` for Problem A, `(x, y, t)` for Problem B.
- Output: scalar `u`.
- Initialization: Glorot / Xavier uniform.
- Total parameters ≈ 8 000.

## 5. Loss

For the soft-constraint baseline:

```
L = w_pde · L_pde  +  w_bc · L_bc  [+ w_ic · L_ic]
```

with `L_pde = (1/N_c) Σ r²` over interior collocation points, `L_bc = (1/N_b) Σ (u − g)²` on the boundary, and (Problem B) `L_ic = (1/N_0) Σ (u(x, y, 0) − u₀)²` at `t = 0`. Default weights: `w_pde = 1`, `w_bc = 1`, `w_ic = 10`.

For the hard-constraint variant (Problem A):

```
u(x, y) = x (1 − x) y (1 − y) · NN(x, y; θ)
```

so that `u = 0` on `∂Ω` is satisfied by construction. Loss reduces to `L = L_pde`.

For the hard-constraint variant (Problem B):

```
u(x, y, t) = sin(π x) sin(π y) · exp(-c · t) · [1 + t · NN(x, y, t; θ)]
```

— a soft *initialization* of the network around the analytic decay, which the network refines. (For a strictly hard variant we use `u(x, y, t) = x(1−x)y(1−y) · [u₀(x,y) + t · NN(x,y,t;θ)]` which satisfies both `u=0` on ∂Ω and `u(·,·,0)=u₀(x,y) · x(1−x)y(1−y)` only if `u₀` already vanishes on the boundary, which it does for the chosen IC.)

## 6. Differentiation strategy

We use a **finite-difference stencil PINN** ("FD-PINN", a known variant; see CAN-PINN, Chiu et al. 2022, and Sirignano-class hybrid finite-difference PINN methods). Specifically, at each interior collocation point `(x, y)` (or `(x, y, t)`), the Laplacian is computed via the standard 5-point central stencil with stencil width `h = 1e-2`, and the time derivative (Problem B) via a central difference with step `Δt = 1e-3`:

```
∂²u/∂x² ≈ [u(x+h, y) − 2u(x, y) + u(x−h, y)] / h²
∂²u/∂y² ≈ [u(x, y+h) − 2u(x, y) + u(x, y−h)] / h²
∂u/∂t   ≈ [u(x, y, t+Δt) − u(x, y, t−Δt)] / (2 Δt)
```

For each interior collocation point we therefore evaluate the network at the centre point plus 4 (Problem A) or 6 (Problem B) stencil neighbours. This avoids the cost and stiffness of higher-order automatic differentiation while remaining a standard recipe in the recent literature. Stencil truncation error is `O(h²) = 1e-4`, well below our target accuracy.

## 7. Optimizer

- Adam, learning rate `1e-3`, 6 000 iterations on the full collocation set per epoch (full-batch).
- (No L-BFGS in this study — we deliberately stop with Adam to make the comparisons across ablations cleaner; we expect this to leave 0.5–1 order of magnitude of accuracy on the table relative to a full Adam-then-L-BFGS recipe, but it makes the ablation comparison fair.)

## 8. Verification

For every run we compute the relative L2 error on the held-out reference grid:

```
E_rel  =  ‖ u_θ − u_ref ‖_2  /  ‖ u_ref ‖_2
```

evaluated on the grid in §3. We additionally compute the maximum pointwise error and a per-step training-loss curve.

Success criterion (from literature norms for this benchmark): `E_rel ≤ 1e-2` for the soft baseline; `E_rel ≤ 1e-3` for the best variant.

## 9. Hypotheses (falsifiable)

- **H1 (hard constraint dominates).** For Problem A on the unit square, the hard-constraint output transform reduces relative L2 error by ≥ 5× compared to the soft-constraint baseline at the same training budget. *Source: Lagaris et al. 1998; Wang, Teng, Perdikaris 2021.*
- **H2 (Fourier features do not help on smooth solutions).** For both Problem A and Problem B with the chosen smooth manufactured solutions, adding a Fourier-feature embedding (`σ = 1.0`, 32 features) does not reduce L2 error by more than 2×; it may slightly increase it. *Source: Tancik et al. 2020 caveats; feature-mapping PINN follow-ups, arXiv:2402.06955.*
- **H3 (BC-loss upweighting helps the soft baseline but is dominated by the hard constraint).** Increasing `w_bc` from 1 to 100 in the soft baseline reduces relative L2 error by at least 2×, but does not reach the accuracy of the hard-constraint variant. *Source: Wang, Teng, Perdikaris 2021.*
- **H4 (error grows in time without intervention).** For Problem B, the soft-baseline relative error at `t = T` is at least 2× the relative error at `t = T/2`, illustrating the time-causality issue (in mild form for this smooth, decaying problem). *Source: Wang, Sankaran, Perdikaris 2024.*

## 10. Ablation matrix

For Problem A (steady):

| Run | Variant                                                        | Expected outcome              |
|-----|----------------------------------------------------------------|-------------------------------|
| A1  | Soft baseline, `w_bc = 1`                                       | reference, ~1e-2 rel L2       |
| A2  | Soft, `w_bc = 100`                                              | ~3–5× better than A1 (H3)     |
| A3  | Hard constraint                                                 | ≥5× better than A1 (H1)       |
| A4  | Hard + Fourier features (σ=1, 32 freqs)                         | not better than A3 (H2)       |

For Problem B (unsteady):

| Run | Variant                                                        | Expected outcome              |
|-----|----------------------------------------------------------------|-------------------------------|
| B1  | Soft baseline, `w_ic = 10`, `w_bc = 1`                          | reference                     |
| B2  | Soft, `w_ic = 100`, `w_bc = 100`                                | better than B1                |
| B3  | Hard constraint (vanishing on ∂Ω, matching IC at t=0)           | best                          |

## 11. Outputs

For each run, save to `runs/<run_name>/`:

- `training_log.csv` with columns `step, loss_total, loss_pde, loss_bc, loss_ic, rel_l2`
- `prediction.npy` and `reference.npy` (final-time slice for Problem B; full field for Problem A)
- `verification.json` with `{rel_l2_final, max_err, n_params, wallclock_s}`

Plots:
- Training loss curves (one figure with one panel per run).
- Final solution + error map (heatmaps for Problem A; final-time slice + space-time RMSE-vs-t curve for Problem B).

## 12. Out of scope for this study

- L-BFGS final polishing (would compress all runs toward the same accuracy and obscure the comparison).
- NTK-based weighting (expensive, separate study).
- Domain decomposition / XPINN (no irregular geometry to motivate it).
- Causal weighting (Wang, Sankaran, Perdikaris 2024). *Conceded gap.* For a longer time horizon or sharper IC this would be a fifth ablation; for `αT = 0.1` and a smooth IC the temporal-causality effect is mild and we do not include it as a variant.
- Comparison to an FD/FEM solver. The manufactured-solution reference is the ground truth.

## 13. Handoff to scaffold

The next skill (`pinn-scaffold`) should produce:

- A self-contained Python module implementing the FD-PINN training loop in NumPy (or PyTorch if available).
- A configuration object that selects between variants A1–A4 and B1–B3.
- A `run_all.py` driver that runs all 7 variants and writes the outputs above.
- A verification harness that computes `E_rel` against the closed-form reference.
