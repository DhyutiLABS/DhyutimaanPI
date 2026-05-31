# PINN Problem Spec: 1D Nonlinear Viscous Burgers Equation — DoE

## 1. Problem Statement

We solve the one-dimensional nonlinear viscous Burgers equation — the canonical benchmark for PINN research on time-dependent, convection-dominated PDEs — using physics-informed neural networks on the standard Raissi 2019 configuration: domain $x \in [-1,1]$, $t \in [0,1]$, sinusoidal initial condition $u(x,0) = -\sin(\pi x)$, homogeneous Dirichlet BCs, and viscosity $\nu = 0.01/\pi \approx 0.00318$. This configuration produces a mild but visible shock near $x=0$ around $t \approx 0.3$–$0.5$, making it a meaningful stress-test for temporal causality, hard-constraint ansatz design, and shock-region collocation. A full $2^3$ factorial DoE is run over three factors: (i) hard IC+BC constraint vs. soft loss, (ii) causal temporal weighting vs. uniform weighting, and (iii) Fourier-feature embedding vs. plain coordinates. A viscosity stress-test sweep ($\nu \in \{0.1/\pi, 0.01/\pi, 0.001/\pi\}$) is run post-hoc on the best-performing variant to characterise where the method degrades.

---

## 2. Governing Equations

$$u_t + u \, u_x - \nu \, u_{xx} = 0, \quad x \in [-1,1],\ t \in [0,1]$$

**Non-dimensionalised form:** The equation is already in non-dimensional form with the Reynolds-number-like parameter $1/\nu$. For $\nu = 0.01/\pi$ the effective Reynolds number is $Re = 1/\nu \approx 314$.

**Symbol glossary:**

| Symbol | Meaning |
|---|---|
| $u(x,t)$ | velocity / transported scalar |
| $\nu$ | kinematic viscosity (diffusion coefficient) |
| $u_t$ | time derivative |
| $u_x, u_{xx}$ | first and second spatial derivatives |
| $u\,u_x$ | nonlinear convective term (shock-forming) |

---

## 3. Domain and Boundary / Initial Conditions

- **Spatial domain:** $x \in [-1, 1]$
- **Temporal domain:** $t \in [0, 1]$
- **Initial condition (IC):** $u(x,0) = -\sin(\pi x)$
- **Left BC (Dirichlet):** $u(-1, t) = 0$ for all $t \in [0,1]$
- **Right BC (Dirichlet):** $u(+1, t) = 0$ for all $t \in [0,1]$

**Note on solution behaviour:** The IC is antisymmetric; the nonlinear term drives the solution toward a shock-like structure near $x=0$ as $t$ increases. At $\nu = 0.01/\pi$ the solution transitions from smooth to steep-gradient around $t \approx 0.3$; the steepest gradient at $t=1$ is $O(1/\nu)$. At $\nu = 0.001/\pi$ this gradient becomes $O(10^4)$ and vanilla PINNs are expected to fail.

---

## 4. Reference / Ground Truth

### Primary reference — High-fidelity finite-difference (FD)

For $\nu = 0.01/\pi$, an exact analytic solution exists via the Cole-Hopf transformation, but its series expansion converges slowly near the shock. We therefore use a high-fidelity FD reference as the primary metric:

- **Method:** 2nd-order upwind scheme for $u\,u_x$; central difference for $u_{xx}$; explicit RK4 time integration
- **Spatial resolution:** $N_x = 512$ uniform points on $[-1,1]$, $\Delta x \approx 0.0039$
- **Temporal resolution:** $\Delta t = 10^{-4}$, CFL-stable for all tested $\nu$
- **Output:** $u_{\text{ref}}(x_i, t_j)$ at the PINN evaluation grid

The FD reference is generated once at the start and saved to `runs/fd_reference.npy`. The PINN evaluation grid uses $N_x = 256$ points and $N_t = 101$ time slices ($t = 0, 0.01, \ldots, 1.0$).

### Secondary reference — Cole-Hopf (ν = 0.01/π only)

$$u(x,t) = -\frac{\sum_{n=1}^{N_{\max}} a_n \, e^{-n^2\pi^2\nu t} \sin(n\pi x)}{\frac{1}{2} + \sum_{n=1}^{N_{\max}} b_n \, e^{-n^2\pi^2\nu t} \cos(n\pi x)}$$

where $a_n, b_n$ are Fourier coefficients of $\exp\!\left(-\tfrac{1}{2\nu}\int_0^x u_0(s)\,ds\right)$. Used as a consistency check against the FD reference at $t = 0.1, 0.5, 1.0$.

**Error metric (primary):**

$$\varepsilon = \frac{\|u_{\text{pred}} - u_{\text{ref}}\|_2}{\|u_{\text{ref}}\|_2}$$

evaluated on the $256 \times 101$ evaluation grid. **Secondary:** per-time-slice relative $L^2$ error to characterise temporal error growth.

---

## 5. Architecture and Training Plan

### Network Architecture

| Component | Setting |
|---|---|
| Base architecture | MLP, 4 hidden layers × 64 units, tanh activation |
| Inputs | $(x, t)$ — 2 inputs |
| Output | $\hat{u}(x,t)$ — scalar |
| Parameters (approx.) | ~21k |
| Fourier variant | Random Fourier features: $\gamma(\xi) = [\cos(2\pi B\xi), \sin(2\pi B\xi)]$, $B \sim \mathcal{N}(0, \sigma^2 I)$, $\sigma = 1$, 32 frequencies → 64 input features; fixed |

### Hard-Constraint Ansatz

For hard IC+BC variants, enforce both the IC and Dirichlet BCs simultaneously:

$$u(x,t;\theta) = \underbrace{-\sin(\pi x)}_{\text{IC at }t=0} + \underbrace{t \cdot (1-x^2)}_{\text{envelope: =0 at }t=0\text{ and }x=\pm 1} \cdot \text{NN}(x,t;\theta)$$

This satisfies:
- $u(x,0) = -\sin(\pi x)$ exactly (IC, for all $x$)
- $u(\pm 1, t) = -\sin(\pm\pi) + t\cdot 0 \cdot \text{NN} = 0$ exactly (BCs)

IC and BC loss terms are dropped from the training objective.

**Potential issue:** The implied NN target at $t = 0$ is:

$$\text{NN}(x,0;\theta) = \frac{u(x,0) + \sin(\pi x)}{0 \cdot (1-x^2)} \to \text{undefined}$$

The NN is not evaluated at $t=0$ by the PDE residual (since the IC term handles $t=0$ directly), so this singularity is not active during training. However, the PDE residual at $t \approx 0^+$ requires:

$$u_t|_{t=0} = (1-x^2)\cdot\text{NN}(x,0) + t\cdot(1-x^2)\cdot\text{NN}_t|_{t=0}$$

which is well-posed as long as the network output at $t=0$ is bounded — unlike the heat-equation B3 failure where the envelope vanished faster than the source. *(This is a key difference from the Heat 2D B3 failure mode — H2 tests whether this distinction matters in practice.)*

### Collocation Sampling

| Points | Count | Strategy |
|---|---|---|
| Interior PDE collocation $(x,t)$ | 10 000 | LHS in $[-1,1]\times[0,1]$ |
| IC points $(x,0)$ | 256 | Uniform on $[-1,1]$ |
| BC points $(\pm 1, t)$ | 200 (100 each side) | Uniform on $[0,1]$ |
| Evaluation grid | $256 \times 101$ | Uniform |

### Loss Functions

**Soft-constraint variants:**

$$\mathcal{L} = \mathcal{L}_{\text{PDE}} + w_{\text{ic}}\,\mathcal{L}_{\text{IC}} + w_{\text{bc}}\,\mathcal{L}_{\text{BC}}$$

- Soft baseline: $w_{\text{ic}} = 10$, $w_{\text{bc}} = 10$ (slightly elevated to avoid vanilla collapse)

**Hard-constraint variants:** $\mathcal{L} = \mathcal{L}_{\text{PDE}}$ only.

**Causal weighting (F2 = +):**

Sort collocation points by $t$; partition into $N_c = 100$ uniform time slabs. Weight PDE loss in slab $k$ by:

$$w_k = \exp\!\left(-\varepsilon \sum_{j=1}^{k-1} \overline{\mathcal{L}}_{\text{PDE},j}\right), \quad \varepsilon = 100$$

where $\overline{\mathcal{L}}_{\text{PDE},j}$ is the mean PDE residual in slab $j$. Log $w_k$ trajectory to `training_log.csv` for analysis.

### Optimiser Schedule

| Stage | Optimiser | LR | Steps |
|---|---|---|---|
| All 8 DoE variants | Adam | 1e-3 | 10 000 |
| Viscosity stress test | Adam | 1e-3 | 10 000 |

10k steps (vs 5k for heat) because the Burgers shock requires more training iterations. No L-BFGS in the main DoE (to keep the causality-weighting effect interpretable); L-BFGS finishing is noted as a follow-up.

**Hardware target:** PyTorch MPS (Apple Silicon), CPU fallback.

---

## 6. Design of Experiments

Full $2^3$ factorial on $\nu = 0.01/\pi$:

| Factor | Level 0 (−) | Level 1 (+) |
|---|---|---|
| **F1**: Constraint | Soft ($w_{\text{ic}}=10, w_{\text{bc}}=10$) | Hard IC+BC ansatz |
| **F2**: Temporal | Uniform weighting | Causal weighting ($\varepsilon=100$) |
| **F3**: Input features | Plain $(x,t)$ | Fourier features ($\sigma=1$, 32 freq) |

### Run Table — ν = 0.01/π

| Run | F1 | F2 | F3 | Label |
|---|---|---|---|---|
| C1 | Soft | Uniform | No | soft-uniform |
| C2 | Soft | Causal | No | soft-causal |
| C3 | Hard | Uniform | No | hard-uniform |
| C4 | Hard | Causal | No | hard-causal |
| C5 | Soft | Uniform | Yes | soft-uniform-ff |
| C6 | Soft | Causal | Yes | soft-causal-ff |
| C7 | Hard | Uniform | Yes | hard-uniform-ff |
| C8 | Hard | Causal | Yes | hard-causal-ff |

### Viscosity Stress-Test — Best Variant Only

After main DoE, run the best-performing variant at:

| Run | ν | Label |
|---|---|---|
| V1 | $0.1/\pi$ | best-nu-high |
| V2 | $0.01/\pi$ | best-nu-mid (= best DoE run, reuse) |
| V3 | $0.001/\pi$ | best-nu-low |

Total: **8 DoE runs + 2 stress-test runs = 10 runs**. Each run saves to `runs/<label>/` with `training_log.csv`, `checkpoint.pt`, `verification.json`.

---

## 7. Hypotheses (Falsifiable)

**H1 — Causal weighting is necessary for sub-1e-2 accuracy on Burgers.**
Soft-causal (C2) achieves relative $L^2 \leq 5 \times 10^{-3}$ while soft-uniform (C1) plateaus at $\geq 1 \times 10^{-2}$ at 10k Adam steps. *Confirmed if: $\varepsilon_{\text{C1}} / \varepsilon_{\text{C2}} \geq 2$ AND $\varepsilon_{\text{C2}} \leq 5\times10^{-3}$.*

**H2 — Hard IC+BC ansatz provides a ≥5× error reduction over soft when combined with causal weighting.**
Hard-causal (C4) achieves relative $L^2 \leq 5\times$ lower than soft-causal (C2). *Confirmed if: $\varepsilon_{\text{C2}} / \varepsilon_{\text{C4}} \geq 5$.*

**H3 — Hard-causal (C4) is the best of the 8 DoE variants.**
C4 has the lowest global relative $L^2$ error among all 8 runs. *Confirmed if: $\varepsilon_{\text{C4}} = \min_i \varepsilon_{C_i}$.*

**H4 — Fourier features reduce error for Burgers (in contrast to smooth heat).**
For a fixed constraint and temporal strategy, adding Fourier features reduces relative $L^2$ by ≥1.5×. *Confirmed if: $\varepsilon_{\text{soft-uniform}} / \varepsilon_{\text{soft-uniform-ff}} \geq 1.5$ OR $\varepsilon_{\text{hard-causal}} / \varepsilon_{\text{hard-causal-ff}} \geq 1.5$.*

**H5 — Method degrades catastrophically at ν = 0.001/π.**
Best DoE variant at $\nu = 0.001/\pi$ (V3) achieves relative $L^2 > 0.1$. *Confirmed if: $\varepsilon_{\text{V3}} > 0.1$.*

---

## 8. Success Criteria

| Run / metric | Target |
|---|---|
| Best DoE run (global $\varepsilon$) | ≤ 5e-3 |
| Soft-uniform baseline (C1) | Reportable (expected 1e-2 to 5e-2) |
| Per-time-slice $L^2$ at $t = 1.0$ (best run) | ≤ 1e-2 |
| Causal weight trajectory | Log $w_k$ saved; $w_k$ should increase over training |
| All loss components | Logged separately in `training_log.csv` |
| `verification.json` | Contains `rel_l2`, `max_abs_error`, per-slice errors at $t \in \{0.25, 0.5, 0.75, 1.0\}$, hypothesis verdicts |

---

## 9. Known Failure Modes to Watch For

1. **Temporal causality violation (C1, C3, C5, C7 — uniform weighting):** Network fits $t \approx 0.5$–$1.0$ region before the IC has converged. Diagnostic: per-time-slice $L^2$ error should be lowest at $t=0$ and grow; if it is **not** monotone, causality is violated. Flag if error at $t=0.25$ > error at $t=0.75$.

2. **Shock-front under-resolution:** Vanilla collocation misses the high-gradient region near $x=0$, $t\approx 0.3$–$0.5$. Diagnostic: plot absolute error map — if peak error is at the shock location, collocation density is insufficient. The uniform 10k LHS budget may be tight; note this for adaptive-collocation follow-up.

3. **Globally incorrect low-residual solution (Burgers failure mode):** Network achieves low PDE residual but the solution is qualitatively wrong (e.g., zero everywhere, or symmetric when the IC is antisymmetric). Diagnostic: check $\varepsilon_{\text{IC}}$ separately from $\varepsilon_{\text{global}}$; compare solution sign at $t=0$ against $-\sin(\pi x)$.

4. **Loss spike at shock transition:** Adam step with very large PDE residual gradient can cause loss spike around steps 2000–5000. Monitor loss trajectory; if spike occurs and loss does not recover, reduce LR or switch to gradient clipping.

5. **Hard IC ansatz ill-conditioning (analogous to Heat B3):** At $\nu = 0.001/\pi$ the PDE residual at $t \approx 0^+$ demands $\text{NN}$ to reproduce a near-singular gradient. Monitor PDE loss plateau. If plateau $> 10$ at step 5000, diagnose.

6. **Fourier feature frequency mismatch:** At $\nu = 0.1/\pi$ (smooth solution), Fourier features may introduce unwanted high-frequency error (same as heat A4 failure). Diagnostic: compare C5 vs C1 at $\nu = 0.1/\pi$.

---

## 10. Out of Scope

- Inviscid Burgers ($\nu = 0$) — requires special shock-capturing; out of scope for PINN comparison
- ν < 0.001/π — expected total failure; documented as known limitation, not tested
- Non-zero Dirichlet or Neumann BCs
- 2D or 3D Burgers
- Burgers as a sub-step in Navier-Stokes
- Inverse problems (recovering ν from data)
- Neural-operator comparisons (DeepONet, FNO)
- Adaptive collocation / residual-guided resampling (flagged for follow-up, not in main DoE)
- XPINN domain decomposition
