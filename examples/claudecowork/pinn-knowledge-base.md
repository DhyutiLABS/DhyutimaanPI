# PINN Literature Survey: Physics-Informed Neural Networks for 2D Heat Conduction

## Scope

This survey targets the use of Physics-Informed Neural Networks (PINNs) for the **two-dimensional heat (diffusion) equation** in both steady-state (Poisson) and unsteady (parabolic) form, on rectangular domains with Dirichlet and/or Neumann boundary conditions. The focus is *operational*: which PINN variants, training tricks, and architectures have been shown to work for this PDE class, what their failure modes are, and what an honest baseline looks like. Out of scope: non-Fourier (Cattaneo–Vernotte) heat conduction, conjugate convection-diffusion, inverse problems, and 3D/coupled multi-physics. Survey conducted on 15 May 2026.

## Headline findings

1. **The 2D heat equation is the canonical "PINN works" problem.** Both the steady-state Poisson form and the unsteady parabolic form on rectangular domains are well within reach of vanilla PINNs (Raissi et al., 2019); they are the demonstration cases in essentially every PINN tutorial (DeepXDE docs, NVIDIA Modulus). However, "works" here means relative L2 error in the 1e-3 to 1e-4 range — competitive with coarse FD, not with high-order FEM.
2. **Failure modes are predictable and well-documented.** They are dominated by (a) loss imbalance between PDE residual and BC/IC terms (Wang, Teng, Perdikaris 2021), (b) spectral bias against high-frequency components of the solution (Tancik et al. 2020; Wang, Wang, Perdikaris 2021), and (c) for unsteady problems, violations of temporal causality where the network fits late-time data before the early-time solution has converged (Wang, Sankaran, Perdikaris 2024).
3. **Three practical fixes carry most of the weight.** Adaptive loss weighting (gradient-norm based, learning-rate annealing, or NTK-based), Fourier feature embeddings of the input coordinates, and — when time-dependent — causal time-marching weights. Each has a corresponding paper that documents 1–2 orders of magnitude error reduction on heat/diffusion benchmarks.
4. **Hard-constraint architectures eliminate BC-loss balancing entirely.** For Dirichlet BCs on rectangular domains the output transformation `u = g(x) + B(x) · NN(x)`, with `B` vanishing on the boundary, removes the BC loss term and turns training into a single-objective optimization. This is the cleanest way to make Dirichlet-BC heat problems trainable.
5. **Coupled automatic-numerical differentiation (CAN-PINN, Chiu et al. 2022)** reports 2–4 orders of magnitude lower MSE than vanilla AD-PINNs on diffusion-type problems with the same collocation budget. It is the strongest "single-trick" speedup specifically validated on heat-equation-class benchmarks.

## Taxonomy of relevant work

### Vanilla PINN (Raissi, Perdikaris, Karniadakis 2019, J. Comput. Phys.)

The reference baseline. An MLP with `tanh` activations is trained to minimize a weighted sum of PDE residual loss (evaluated at collocation points by automatic differentiation) and supervised losses for initial and boundary conditions. For 2D heat conduction the network maps `(x, y, t) → u`, the PDE residual is `r = u_t − α(u_xx + u_yy)`, and a typical recipe is 4–6 hidden layers of width 50–100 trained with Adam followed by L-BFGS. This works on rectangular domains for smooth solutions but degrades sharply in the presence of high-frequency content, multi-scale features, or sharp gradients.

### Loss-weighting schemes

**Wang, Teng, Perdikaris (2021), SIAM J. Sci. Comput.** — "Understanding and mitigating gradient pathologies in PINNs" — diagnosed that the back-propagated gradient of the PDE residual loss often dominates the BC/IC loss gradient by orders of magnitude. They proposed a learning-rate annealing scheme that uses the running statistics of gradient magnitudes to rebalance the loss weights at each iteration. Frequently 1–2 orders of magnitude improvement on diffusion-class problems.

**Wang, Wang, Perdikaris (2021), J. Comput. Phys.** — "When and why PINNs fail to train: a Neural Tangent Kernel perspective" — derived the limiting NTK for PINNs and showed the convergence rate of each loss component is governed by NTK eigenvalues. Proposed an NTK-based weighting scheme that adapts loss weights using the trace of each component's NTK. More principled than gradient-norm weighting but more expensive.

**Self-Adaptive Loss-Balanced PINN (SA-PINN), McClenny & Braga-Neto 2020 / Neurocomputing 2022** — Introduces per-point trainable weights on the collocation residuals, optimized adversarially against the model parameters. Often used as a baseline for "automatic" weighting.

### Architectural variants for heat-class problems

**Fourier feature mappings (Tancik et al. 2020; Wang, Wang, Perdikaris 2021)** — Random Fourier feature embedding `γ(x) = [cos(2πBx), sin(2πBx)]` flattens the NTK eigenspectrum and removes the bias toward low frequencies. The most-cited single architectural change. Works well for heat conduction problems with high-wavenumber boundary forcing or sharp Gaussian-like initial conditions; less helpful for smooth solutions where it can introduce unnecessary high-frequency error (the Gibbs-like phenomenon documented in Mildenhall et al.'s feature-mapping followups).

**SIREN (Sitzmann et al. 2020)** — Sinusoidal activations across the network. Less commonly used in PINN literature than Fourier features, but related; used in specific high-frequency PDE settings.

**PirateNets (Wang et al. 2024, JMLR)** — Residual-adaptive architecture with skip-connections that empirically outperforms tanh MLPs across a benchmark suite that includes the heat equation. Code available, with claimed state-of-the-art on standard PDE benchmarks.

### Domain-decomposition variants

**XPINN (Jagtap & Karniadakis 2020, Commun. Comput. Phys.)** — Extends cPINN to arbitrary space-time decompositions. Each subdomain has its own MLP, with continuity enforced at interfaces. Useful for heat problems on non-trivial geometries and for parallelization, but adds the overhead of interface losses; the SIAM 2022 follow-up (De Ryck et al.) cautions that XPINN does not always improve generalization vs. a single PINN of comparable capacity.

**FBPINN (Moseley, Markham & Nissen-Meyer 2023, Adv. Comput. Math.)** — Finite-basis PINNs use overlapping subdomain windows that are summed to give a global solution; designed for multi-scale problems.

### Time-marching and causal variants

**Causal PINN (Wang, Sankaran, Perdikaris 2024, CMAME 421:116813)** — Reweights PDE residual losses across time so that residuals at later times are down-weighted until earlier-time residuals have converged. Specifically targets the failure mode where PINNs fit late-time data before the early-time solution has converged (a known issue for unsteady heat conduction with non-trivial initial conditions). State-of-the-art on time-dependent benchmarks; widely cited for parabolic PDEs.

**Time-adaptive marching schemes / Dual-level time-marching (Springer 2025)** — Solve the PDE one short time window at a time, using the solution at the end of one window as the initial condition for the next. A robust if computationally expensive workaround for long-time integration.

### Hard-constraint methods

**Output transformation for Dirichlet BCs (Lagaris, Likas & Fotiadis 1998; widely used in DeepXDE)** — For an MLP output `N(x; θ)`, define the prediction as `u(x) = g(x) + D(x) · N(x; θ)` where `g` matches the boundary data and `D(x) = 0` on `∂Ω` and `> 0` in the interior. For a unit square with `u = 0` on the boundary, take `D(x, y) = x(1−x)y(1−y)`. This removes the BC loss term entirely and is the single most impactful trick for Dirichlet-BC heat problems; it converts a multi-objective optimization into a single-objective one. Less straightforward for Neumann BCs (where the recent Fourier-embedding hard-Neumann construction of Saâdani et al. 2025 is the current state of the art).

### Coupled AD/numerical differentiation

**CAN-PINN (Chiu, Wong, Ooi, Dao, Ong 2022, CMAME 395)** — Couples automatic differentiation of `u` with finite-difference stencils between neighboring support points. Reports 2–4 orders of magnitude lower MSE than AD-only PINNs on the same collocation budget, with diffusion-equation benchmarks among the headline results.

**Hybrid FD-PINN (Sukumar et al. 2022; Sirignano et al. 2023)** — Replaces all input-side AD with central finite differences, computing the Laplacian via the standard 5-point stencil at each collocation point. Avoids the cost and stiffness of higher-order AD; competitive on diffusion problems where the second-order spatial derivatives dominate compute.

### Neural operators (PINN-adjacent — separate problem class)

**DeepONet (Lu et al. 2021)** and **Fourier Neural Operator (Li et al. 2021)** learn solution *operators* — mappings from initial/boundary conditions to solutions — rather than solving a single PDE instance. They are typically trained on simulation data, not from a residual loss. Frequently confused with PINNs; for the heat equation the FNO has demonstrated parameterized-coefficient generalization that PINNs cannot achieve out-of-the-box. Treat them as a separate category, not a "better PINN".

## Findings by PDE class: 2D heat conduction

### Steady-state (2D Poisson form)

`Δu = f(x, y)` on `Ω ⊂ R²` with Dirichlet/Neumann BCs.

- **Manufactured solution `u(x, y) = sin(πx)sin(πy)`** on `[0, 1]²` with `f = −2π² sin(πx) sin(πy)` and `u = 0` on the boundary is the standard verification case. Vanilla PINN with hard-constraint output achieves 1e-4 to 1e-5 relative L2 error in 5–10k Adam iterations.
- **Multi-scale source terms** (e.g. `f` containing high-wavenumber components) are the documented failure case. Fourier features and Wang–Wang–Perdikaris 2021 NTK weighting are the strongest mitigations.
- Reference benchmarks: DeepXDE Poisson tutorials; the heat-pinn GitHub repository (314arhaam/heat-pinn) provides a clean reference for the steady 2D case with around 1e-3 relative L2 error and reports the loss-balance as the primary tuning lever.

### Unsteady (2D parabolic form)

`u_t = α(u_xx + u_yy) + s(x, y, t)` on `Ω × [0, T]`.

- **Manufactured solution `u(x, y, t) = sin(πx)sin(πy) exp(−2π²αt)`** is the canonical test. Standard PINN works for short time horizons (`αT ≲ 0.1`) but exhibits error growth in `t` for longer integration. Causal weighting (Wang, Sankaran, Perdikaris 2024) is the documented fix.
- **Sharp Gaussian initial conditions** (heat-kernel-like spreading from a near-Dirac source) trip up vanilla PINNs because the initial condition has high-wavenumber content; Fourier features and IC-loss upweighting both help.
- Recent surveys (e.g. ScienceDirect 2025 review of heat-transfer PINNs) report that across published 2D heat-conduction case studies, the typical reported error is in the 1e-3 to 1e-4 range relative L2.

## Common failure modes documented in the literature

1. **Loss-component imbalance.** PDE residual gradient often 100×–1000× the BC/IC gradient at initialization (Wang, Teng, Perdikaris 2021). The network fits the PDE in a constant-zero solution and ignores the BC, or vice versa. Hard constraints fix this for Dirichlet BCs; adaptive weighting fixes it more generally.
2. **Spectral bias.** Standard MLPs with `tanh` learn low frequencies first and may never recover the high-frequency components within reasonable training budgets (Wang, Wang, Perdikaris 2021; Tancik et al. 2020). For the 2D heat equation this manifests as inability to resolve sharp ICs or high-wavenumber forcing.
3. **Temporal-causality violation.** For unsteady PDEs, naive PINN training can fit late-time data while the early-time solution remains wrong, producing a self-consistent but physically incorrect solution (Wang, Sankaran, Perdikaris 2024). Less severe for the 2D heat equation than for chaotic systems but documented.
4. **Insufficient collocation density.** Chiu et al. (2022) document that AD-based PINNs can reach a regime where adding collocation points does not improve accuracy, because the optimizer is stuck in a flat region. CAN-PINN's coupling between AD and stencil-based gradients explicitly addresses this.
5. **Optimizer stiffness.** Adam alone is often insufficient; the standard recipe is Adam-then-L-BFGS, with L-BFGS doing the final 1–2 orders of magnitude error reduction. Documented in essentially every paper from Raissi et al. 2019 onward.

## Open problems and contested points

- **Does Fourier-feature embedding always help?** Tancik 2020 argues yes for high-frequency targets; later work (Mildenhall et al. follow-ups; the Feature Mapping in PINNs paper, arXiv:2402.06955) shows it can introduce Gibbs-like overshoot for problems with discontinuities or low-frequency targets. For smooth 2D heat conduction with low-frequency ICs, Fourier features provide little benefit and can hurt slightly.
- **NTK-based weighting vs. gradient-norm weighting.** Both have champions. NTK weighting is more principled but expensive; gradient-norm weighting is cheap and often "good enough." No clean head-to-head comparison on the 2D heat benchmark exists.
- **XPINN generalization claims.** De Ryck et al. (SIAM J. Sci. Comput. 2022) showed XPINN does not always generalize better than a single PINN of equal total parameter count. Domain decomposition pays off only when subdomain physics differ.
- **PINN vs. classical solvers.** The Medium piece "2D Heat Conduction: PINN vs FDM" (Ozler) and several engineering papers report PINNs are not competitive with classical methods on regular grids for the canonical 2D heat equation. PINNs win when the geometry is irregular, when data must be assimilated, or when an inverse problem must be solved.
- **What counts as a fair evaluation?** Many published "PINN works on heat equation" papers use very favorable settings: small domains, short time horizons, smooth solutions. Mojgani et al. 2024 and others have called for standardized benchmarks.

## Implementation landscape

- **DeepXDE (Lu et al. 2021)** — Most mature open-source PINN library. Supports both PyTorch and TensorFlow backends. Has a built-in 2D heat-equation tutorial. Recommended starting point.
- **NVIDIA Modulus / Modulus Sym** — Higher-level scientific ML framework with built-in PINN support. Strongest for industrial/3D problems.
- **NeuroDiffEq, IDRLnet** — Lighter-weight alternatives.
- **`maziarraissi/PINNs`** — The original reference repository, in TensorFlow 1. Historical interest only.
- **`PredictiveIntelligenceLab/CausalPINNs`** and **`PredictiveIntelligenceLab/GradientPathologiesPINNs`** — Wang/Perdikaris group repositories, in JAX. Reference implementations for the causal and gradient-pathology fixes respectively.
- **`AmeyaJagtap/XPINNs`** — Reference XPINN code in TensorFlow.
- **`chiuph/CAN-PINN`** — Reference CAN-PINN code.
- **`314arhaam/heat-pinn`** — Clean, single-file reference implementation specifically for 2D steady heat conduction.

## Recommended starting points for problem-spec

1. **Use a hard-constraint output for Dirichlet BCs.** For a unit square with zero Dirichlet BC the output transform `u(x, y, t) = x(1−x)y(1−y) · NN(x, y, t; θ) + g(x, y, 0)·time-mask` removes the BC loss entirely. Cite Lagaris, Likas & Fotiadis (1998); standard practice in DeepXDE. Strongest single intervention for the canonical benchmark.
2. **Use Adam (lr 1e-3, ~10–20k iterations) followed by L-BFGS to convergence.** The two-stage optimizer is essentially universal in the PINN literature; without L-BFGS the final accuracy is typically 1–2 orders of magnitude worse.
3. **Include a Fourier-feature ablation.** If the manufactured solution is smooth (the canonical `sin·sin` or `sin·sin·exp(−t)`) Fourier features should *not* help and may slightly hurt — making this a clean falsifiable test of a popular claim. If the solution has high-frequency content (sharp Gaussian IC), Fourier features should help substantially. Cite Tancik et al. 2020 and Wang–Wang–Perdikaris 2021.
4. **For the unsteady case, include a causal-weighting ablation.** Even on the smooth `sin·sin·exp(−t)` benchmark this should reduce error growth in `t`. Cite Wang, Sankaran, Perdikaris 2024.
5. **Use a manufactured-solution reference.** For both steady and unsteady cases the closed-form reference makes verification trivial and removes the FD/FEM-reference confound.
6. **Report relative L2 error on a held-out fine grid, not just training loss.** Training-loss-based reporting in the PINN literature has been criticized (Mojgani et al. 2024); a held-out reference grid is the honest measure.

## Sources

1. Raissi, M., Perdikaris, P., Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics* 378, 686–707. https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125
2. Wang, S., Teng, Y., Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5), A3055–A3081. https://epubs.siam.org/doi/10.1137/20M1318043
3. Wang, S., Wang, H., Perdikaris, P. (2021). When and why PINNs fail to train: A neural tangent kernel perspective. *J. Comput. Phys.* 449, 110768. https://arxiv.org/abs/2007.14527
4. Wang, S., Sankaran, S., Perdikaris, P. (2024). Respecting causality for training physics-informed neural networks. *Computer Methods in Applied Mechanics and Engineering* 421, 116813. https://arxiv.org/abs/2203.07404
5. Tancik, M., et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains. *NeurIPS*. https://bmild.github.io/fourfeat/
6. Jagtap, A. D., Karniadakis, G. E. (2020). Extended physics-informed neural networks (XPINNs): A generalized space-time domain decomposition based deep learning framework for nonlinear partial differential equations. *Communications in Computational Physics* 28(5), 2002–2041. https://global-sci.com/cicp/article/view/6911
7. Chiu, P.-H., Wong, J. C., Ooi, C., Dao, M. H., Ong, Y.-S. (2022). CAN-PINN: A fast physics-informed neural network based on coupled-automatic-numerical differentiation method. *CMAME* 395, 114909. https://arxiv.org/abs/2110.15832
8. McClenny, L., Braga-Neto, U. (2022). Self-adaptive loss balanced physics-informed neural networks. *Neurocomputing* 496, 11–34.
9. Lu, L., Meng, X., Mao, Z., Karniadakis, G. E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Review* 63(1), 208–228. https://dl.acm.org/doi/abs/10.1137/19m1274067
10. De Ryck, T., Jagtap, A. D., Mishra, S. (2022). Error estimates for physics informed neural networks approximating the Navier–Stokes equations / When do XPINNs improve generalization? *SIAM J. Sci. Comput.* https://epubs.siam.org/doi/10.1137/21M1447039
11. Wang, S., et al. (2024). PirateNets: Physics-informed deep learning with residual adaptive networks. *JMLR* 25. https://www.jmlr.org/papers/volume25/24-0313/24-0313.pdf
12. Lagaris, I. E., Likas, A., Fotiadis, D. I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Trans. Neural Netw.* 9(5), 987–1000. (Original hard-constraint output construction.)
13. T-phPINN (2024) — non-Fourier 2D heat conduction PINN. https://www.sciencedirect.com/science/article/pii/S0017931024010469
14. heat-pinn reference repository. https://github.com/314arhaam/heat-pinn
15. DeepXDE heat-equation tutorial. https://deepxde.readthedocs.io/en/stable/demos/pinn_forward/heat.html
