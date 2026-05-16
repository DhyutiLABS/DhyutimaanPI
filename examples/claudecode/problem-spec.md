# PINN Problem Spec: 2D Steady-State Heat Conduction

## 1. Problem statement

We solve the 2D steady-state heat conduction equation on the unit square $[0,1]^2$.
The temperature field $T(x,y)$ satisfies Laplace's equation $\nabla^2 T = 0$ everywhere in the interior,
subject to Dirichlet boundary conditions that are zero on three walls and sinusoidal on the top wall.
An exact closed-form solution exists, making this an ideal first benchmark: PINN predictions can be
rigorously validated against the analytic reference, and we can measure how L2 error scales with
collocation-point count. A secondary goal is to establish a FEM baseline (SciPy/FEniCS or finite
differences) to compare PINN accuracy per training cost.

## 2. Governing equations

$$\nabla^2 T = \frac{\partial^2 T}{\partial x^2} + \frac{\partial^2 T}{\partial y^2} = 0, \quad (x,y) \in \Omega = (0,1)^2$$

**Symbol definitions:**
- $T(x,y)$: temperature field [K or non-dimensional]
- $x, y$: spatial coordinates $\in [0,1]$
- $\Omega$: open unit square domain
- $\partial\Omega$: boundary of $\Omega$ (four edges)

The problem is already non-dimensional (unit square, $T$ is the only field variable).

## 3. Domain and boundary/initial conditions

- **Spatial domain:** $\Omega = [0,1] \times [0,1]$, open interior; $\partial\Omega$ = four edges.
- **Temporal domain:** N/A — steady-state problem.

**Boundary conditions (all Dirichlet):**

| Boundary          | Condition                         |
|-------------------|-----------------------------------|
| Left:   $x=0$, $y \in [0,1]$ | $T = 0$                 |
| Right:  $x=1$, $y \in [0,1]$ | $T = 0$                 |
| Bottom: $y=0$, $x \in [0,1]$ | $T = 0$                 |
| Top:    $y=1$, $x \in [0,1]$ | $T = \sin(\pi x)$       |

No initial conditions (steady-state).

## 4. Reference / ground truth

**Analytic solution (closed form):**

$$T^*(x,y) = \frac{\sin(\pi x)\,\sinh(\pi y)}{\sinh(\pi)}$$

**Derivation note:** separation of variables with $T = X(x)Y(y)$, $X = \sin(\pi x)$, $Y = A\sinh(\pi y)$, normalised so that $T(x,1) = \sin(\pi x)$.

**Evaluation grid:** $200 \times 200$ uniform grid on $[0,1]^2$ (40 000 points).

**Error metric:**

$$\varepsilon_{L^2} = \frac{\|T_\text{PINN} - T^*\|_2}{\|T^*\|_2}$$

## 5. Architecture and training plan

**Network:**
- Architecture: MLP, fully-connected
- Depth × Width: 5 hidden layers × 64 neurons
- Activation: $\tanh$ (smooth, suited to Laplace operator)
- Inputs: $(x, y) \in \mathbb{R}^2$
- Output: $T(x, y) \in \mathbb{R}^1$
- Weight init: Xavier uniform (default PyTorch)

**Sampling:**
- Interior collocation points: 4096 (LHS / uniform random; ablation will vary this to 256, 1024, 4096, 16384)
- Boundary points: 400 total (100 per edge, uniformly spaced)
- No resampling during training (fixed point set; note as a hypothesis axis if desired)

**Loss:**

$$\mathcal{L} = w_r \mathcal{L}_r + w_b \mathcal{L}_b$$

$$\mathcal{L}_r = \frac{1}{N_r}\sum_{i=1}^{N_r} \left(\nabla^2 T(\mathbf{x}_i)\right)^2, \quad \mathcal{L}_b = \frac{1}{N_b}\sum_{j=1}^{N_b}\left(T(\mathbf{x}_j) - T_j^{\text{BC}}\right)^2$$

- Default weights: $w_r = 1.0$, $w_b = 10.0$ (higher BC weight to suppress boundary drift — a hypothesis axis)
- Residual computed via `torch.autograd.grad` or `torch.func.grad`

**Optimizer:**
- Stage 1: Adam, lr = 1e-3, 10 000 iterations
- Stage 2: L-BFGS, max 2000 iterations, strong Wolfe line search
- Total wall time target: < 5 min on CPU/MPS

**Hardware target:** Apple MPS (`torch.device("mps")`) via `torchmps` conda env; fallback to CPU automatically.

## 6. Hypotheses (falsifiable)

**H1 — Baseline convergence:**
The PINN trained with 4096 collocation points achieves relative L2 error $\varepsilon_{L^2} < 10^{-3}$ on the $200 \times 200$ evaluation grid after Stage 1 (Adam only).

**H2 — L-BFGS refinement:**
Adding the L-BFGS Stage 2 reduces $\varepsilon_{L^2}$ by at least $5\times$ compared to Adam-only training, without increasing boundary loss $\mathcal{L}_b$.

**H3 — Collocation scaling:**
Relative L2 error decreases monotonically as collocation count increases from 256 → 1024 → 4096 → 16384, and the improvement from 4096 → 16384 is less than $2\times$ (diminishing returns past 4k).

**H4 — BC weight sensitivity:**
Increasing the BC weight from $w_b = 1$ to $w_b = 10$ reduces boundary residual by at least an order of magnitude, at the cost of slowing interior residual convergence by no more than $2\times$ in wall-clock time.

## 7. Success criteria

- Relative L2 error $\varepsilon_{L^2} \leq 10^{-3}$ on the $200 \times 200$ evaluation grid (two-stage training, 4096 collocation points).
- Pointwise max error $\leq 5 \times 10^{-3}$ over the evaluation grid.
- Boundary residual $\mathcal{L}_b \leq 10^{-6}$ at end of training.
- Training completes in < 5 minutes on Apple MPS.
- Verification harness produces `verification.json` and a comparison plot without errors.

## 8. Known failure modes to watch for

- **BC drift:** If $w_b$ is too low, the network learns a smooth interior but violates BCs. Watch: $\mathcal{L}_b$ plateau above $10^{-4}$.
- **Loss imbalance:** If $\mathcal{L}_r \gg \mathcal{L}_b$ at step 0, the Adam gradient is dominated by the interior residual and BCs never converge. Monitor both terms separately.
- **Spectral bias near $x=0,1$:** $\sin(\pi x)$ has zero-crossings at both ends; the network may over-smooth the gradient there. Check pointwise error map — it should peak near corners, not mid-edges.
- **Saddle-point stall in L-BFGS:** If L-BFGS increases total loss, reduce max line-search steps or restart from Adam solution. Log loss at every L-BFGS step.
- **MPS numerical precision:** `torch.float32` on MPS can accumulate gradient noise. If residuals stall, check against CPU run with same seeds.

## 9. Out of scope

- Time-dependent (transient) heat equation — not this iteration.
- Non-homogeneous source term ($\nabla^2 T = -f$) — trivially addable but excluded to keep the reference clean.
- Non-unit or non-square geometries.
- Mixed Neumann/Robin BCs.
- Adaptive collocation (RAR, residual-based refinement).
- Full FEM comparison (a FD reference on a $200 \times 200$ grid may be included in the verification harness for qualitative comparison only).
- GPU (CUDA) targets — MPS + CPU only.
- Uncertainty quantification or Bayesian PINNs.
