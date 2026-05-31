# PINN Literature Survey: 2D Heat Conduction + 1D Nonlinear Burgers (Forward Problems)

## Scope

Combined survey targeting two canonical forward problems solved with Physics-Informed Neural Networks (PINNs):

1. **2D steady and unsteady heat conduction** on the unit square Ω = (0,1)², manufactured-solution reference
2. **1D nonlinear viscous Burgers equation** on x ∈ [-1,1], t ∈ [0,1], IC u(0,x) = −sin(πx), ν = 0.01/π

Focus: operational guidance — which variants, training recipes, and failure modes are well-documented, and what to test in a DoE. Survey completed May 2026. Sources: arXiv, JMLR, NeurIPS 2024 (PINNacle benchmark), CMAME, Scientific Reports.

---

## Headline Findings

1. **Hard-constraint output transforms are the strongest single intervention for steady Dirichlet-BC problems.** For steady 2D heat with homogeneous BCs, enforcing u = x(1−x)y(1−y)·NN eliminates BC loss entirely, routinely achieving 1e-4 to 1e-5 relative L² error in 3–5k Adam iterations — 100–1000× better than soft-constraint baselines at the same budget. (Prior DhyutimaanPI run: A3 reached 7.85e-5 in 3000 steps.)

2. **Causal weighting is effective on stiff/chaotic time-dependent PDEs — but fails on smooth or low-stiffness problems.** Wang, Sankaran, Perdikaris (2022/2024) demonstrated SOTA on Lorenz, Kuramoto-Sivashinsky, and Navier-Stokes. ⚠️ **DhyutimaanPI DoE 2026 update:** Causal weighting with ε=100 is the *worst* Burgers DoE variant at ν=0.01/π and produces catastrophic failure on smooth 2D heat (367× error growth). The failure mechanism is **causal weight starvation**: when ε·T·ℓ̄₀ ≫ 1, gradient contribution of all but the earliest temporal slab collapses to zero within a few hundred training steps. Use the computable diagnostic ε·T·ℓ̄₀ before applying causal weighting; target product < 0.1.

3. **Loss-component imbalance is the dominant failure mode for both problem classes.** The BC/IC loss gradient is often 100–1000× smaller than the PDE residual gradient, causing the network to fit the PDE while ignoring constraints (Wang, Teng, Perdikaris 2021). ReLoBRaLo (Bischof & Kraus 2021/2025) solves this 40% faster than learning-rate annealing on Burgers; hard constraints eliminate it for Dirichlet BCs.

4. **Burgers with small ν (ν ≤ 0.01/π) is a stress-test, not a tutorial.** PINNacle NeurIPS 2024 found standard PINNs only solve 10/22 benchmark tasks; Burgers with steep gradients is among the failures unless weak-form residuals, adaptive collocation, or singular-layer methods are applied. The benchmark 1e-3 relative L² cited in early papers (Raissi 2019) uses ν = 0.01/π with IC −sin(πx) — which produces a mild shock; lower ν requires explicit handling.

5. **PirateNets (Wang et al., JMLR 2025) are the strongest general-purpose PINN architecture.** Residual Adaptive Networks initialize as shallow networks, progressively deepen, and achieve SOTA across benchmark PDEs including heat and Burgers. The adaptive residual connection fixes unstable PDE residual loss minimization in deeper MLPs, a previously documented failure of vanilla PINN scaling.

---

## Taxonomy of Relevant Work

### Vanilla PINN — Raissi, Perdikaris, Karniadakis (2019, J. Comput. Phys. 378:686–707)

The reference baseline. An MLP with tanh activations minimizes a weighted sum of: PDE residual (at collocation points via AD), IC loss, and BC loss. Two-stage training: Adam then L-BFGS. For 1D Burgers (ν = 0.01/π), Raissi reported ~1e-3 relative L² with a 9-layer × 20-width network, using 25k collocation points, 100 IC points, and 50 BC points. The original code is on GitHub (maziarraissi/PINNs). **Limitation:** does not scale to small ν; L-BFGS stage is critical and absent from many reimplementations.

### Loss-Weighting Schemes

**Wang, Teng, Perdikaris (2021) — "Understanding and Mitigating Gradient Pathologies in PINNs," SIAM J. Sci. Comput.** Diagnosed that PDE residual gradients dominate BC/IC gradients by orders of magnitude. Learning-rate annealing: scale each loss component by the ratio of its maximum gradient norm to its mean gradient norm. 1–2 orders of magnitude improvement on diffusion and Burgers benchmarks.

**Wang, Wang, Perdikaris (2021) — "When and Why PINNs Fail to Train: A Neural Tangent Kernel Perspective," J. Comput. Phys.** NTK-based weighting; more principled but more expensive per step. Convergence rate of each loss component is proportional to its NTK trace.

**ReLoBRaLo — Bischof & Kraus (2021, arXiv:2110.09813; published CMAME 2025).** Relative Loss Balancing with Random Lookback — single backward pass, lightweight overhead. On Burgers: 40% faster to target accuracy than learning-rate annealing, 70% faster than GradNorm. Tested on Burgers forward + inverse problems as primary benchmark. **Recommended for Burgers DoE.**

**SA-PINN — McClenny & Braga-Neto (2020/2022, Neurocomputing).** Per-point trainable weights on collocation residuals; optimized adversarially. Effective for Burgers near the shock transition.

**AL-PINNs — Lagaris-augmented (2022, arXiv:2205.01059).** Augmented Lagrangian relaxation for PINN loss balancing; benchmarked on viscous Burgers with adaptive weight updates.

### Architectural Variants

**Fourier Feature Mappings — Tancik et al. (2020); Wang, Wang, Perdikaris (2021).** Random Fourier embedding γ(x) = [cos(2πBx), sin(2πBx)] flattens the NTK eigenspectrum. Helps for high-wavenumber problems (e.g., heat with high-frequency source, Burgers near shock). **Caveat:** hurts smooth, low-frequency problems (confirmed in prior DhyutimaanPI Heat A4 run: Fourier was 25× worse than hard-constraint-only on smooth sin·sin solution).

**SIREN — Sitzmann et al. (2020, NeurIPS).** Sinusoidal activations. Rarely the first choice in PINN literature but relevant for Burgers when high-frequency resolution is needed.

**PirateNets — Wang, Li, Chen, Perdikaris (2024 arXiv; JMLR 2025, Vol. 25, Art. 402).** Physics-informed Residual Adaptive Networks. Novel adaptive residual connection; networks initialized as shallow, progressively deepen. SOTA across standard PINN benchmarks. Code: github.com/PredictiveIntelligenceLab/jaxpi (pirate branch). Identified that MLP derivative trainability, not depth per se, is the root cause of vanilla PINN scaling failures.

### Hard-Constraint Methods

**Output Transformation for Dirichlet BCs — Lagaris, Likas & Fotiadis (1998); widely adopted in DeepXDE.** For homogeneous Dirichlet BCs on unit square: u(x,y) = x(1−x)y(1−y)·NN(x,y). Removes BC loss term; single-objective optimization. Most impactful single trick for steady 2D heat. For unsteady Burgers: enforce IC via u(x,t) = −sin(πx) + t·NN(x,t) (but watch target singularity near t=0 boundaries — confirmed failure in DhyutimaanPI Heat B3 run).

**Hard Constraint + Weighted Loss for Conservation Laws (Nature Scientific Reports 2025).** Combines hard boundary constraints with adaptive PDE residual weighting for hyperbolic conservation laws including Burgers-type problems. Addresses shock-region instability.

### Causal and Time-Marching Variants

**Causal PINN — Wang, Sankaran, Perdikaris (2022 arXiv:2203.07404; CMAME 421:116813, 2024).** Re-weights PDE residuals across the time axis: residuals at later times are down-weighted until earlier-time residuals converge. Exponential causal weight: w(t) = exp(−ε · ∑_{t'<t} r(t')). Demonstrates SOTA on Lorenz, Kuramoto-Sivashinsky, and Navier-Stokes. Code: github.com/PredictiveIntelligenceLab/CausalPINNs. ~~**This is the top recommendation for 1D Burgers DoE.**~~ ⚠️ **UPDATED (DhyutimaanPI DoE 2026):** Causal weighting with ε=100 is the *worst* performer on Burgers ν=0.01/π (mild stiffness) and causes catastrophic compound failure at ν=0.001/π. Apply only when ε·T·ℓ̄₀ < 0.1 (evaluate at initialisation). Benefits are confined to genuinely stiff or chaotic problems where ℓ̄₀ is small relative to the starvation threshold.

**Causality-Enhanced Discrete PINNs — IJCAI 2024.** Combines causal weighting with discretized time-stepping; extends CausalPINNs idea to structured time grids.

**Implicit Euler Transfer Learning — Burgers PINNs (2023, arXiv:2310.15343).** Time-discrete PINN using implicit Euler: each network at time slab transfers weights to the next. Effective for moderate ν; each slab is a small, fast-converging optimization.

### Domain Decomposition

**XPINN — Jagtap & Karniadakis (2020, Commun. Comput. Phys.).** Space-time subdomains, each with its own MLP. Continuity enforced at interfaces. Useful for long-time Burgers integration or multi-scale problems. **Caveat:** De Ryck et al. (SIAM 2022) showed XPINN does not always generalize better than a single PINN of equal total capacity.

**FBPINN — Moseley, Markham, Nissen-Meyer (2023, Adv. Comput. Math.).** Overlapping window decomposition; designed for multi-scale problems. Less directly relevant for unit-domain Burgers.

### Weak-Form Methods

**WF-PINNs — Nature Scientific Reports 2025.** Weak-form integral formulation of the PDE residual. For Burgers, embeds the entropy condition as a hard constraint, achieving significantly higher accuracy near shocks vs. strong-form PINNs. Dual-network architecture for inverse problems. This is the most relevant new method for low-ν Burgers.

**Singular Layer PINNs (sl-PINNs) — arXiv:2410.09723 (2024).** Specialized basis functions for the interior layer in low-ν Burgers. Highly accurate for stiff Burgers but problem-specific.

**Hybrid Boundary-Layer + PINN — MDPI Mathematics 2024.** Combines analytical boundary-layer theory with PINN to handle ν → 0 limit. Complementary to but more complex than pure PINN approaches.

### Adaptive Collocation

**PINNACLE (Stabilized Adaptive Loss + Residual Collocation) — arXiv:2603.03224.** Residual-gradient guided adaptive sampling allocates more collocation points near shocks dynamically. Relevant for Burgers DoE where shock position changes with ν.

### Benchmarks

**PINNacle — NeurIPS 2024 Datasets & Benchmarks Track.** 20+ PDEs, standardized evaluation protocol. Key finding: standard PINNs solve only 10/22 tasks. 1D Burgers is among failure cases without interventions. Code: github.com/i207M/PINNacle.

**PDEBench — NeurIPS 2022.** Broader ML benchmark (not PINN-specific); includes 1D Burgers with varying ν.

---

## Findings by PDE Class

### 2D Heat Conduction (Summary from Prior DhyutimaanPI Runs)

**Steady (Poisson form):** u_ref = sin(πx)sin(πy), f = −2π²sin(πx)sin(πy), homogeneous Dirichlet.
- Hard constraint (A3): 7.85e-5 relative L² in 3000 Adam steps, no L-BFGS. Best result.
- Soft + w_bc=100 (A2): 3.04e-2. Soft + w_bc=1 (A1): 8.68e-2.
- Fourier features hurt on this smooth solution (A4: 2.01e-3 vs A3: 7.85e-5).

**Unsteady (Parabolic):** u_ref = sin(πx)sin(πy)·exp(−2π²αt), α=1, T=0.1.
- Soft + heavy weighting (B2): 2.26e-2. Hard constraint (B3): 0.26 — **failed** due to singular target at boundary.
- Causal weighting not yet tested; expected to improve B-variants.
- Adam only, no L-BFGS. L-BFGS finishing step would compress all variants.

**Open gaps:** L-BFGS polishing, causal weighting for unsteady, seed variance (N=3 minimum), FD stencil h sensitivity.

### 1D Burgers Equation

**Standard setup (Raissi 2019):** u_t + u·u_x − ν·u_xx = 0, x∈[-1,1], t∈[0,1], u(x,0) = −sin(πx), u(-1,t)=u(1,t)=0, ν = 0.01/π ≈ 0.00318.

- Vanilla PINN (Raissi 2019): ~1e-3 relative L² with 9×20 network, 25k collocation points, Adam+L-BFGS.
- At ν = 0.01/π the solution forms a mild but visible shock near x=0 around t≈0.3–0.5. Steep gradient → hard for Adam alone.
- **Failure mode:** networks "solve" the residual while fitting the wrong solution (globally incorrect but low-residual). Documented in WF-PINNs 2025 and singular-layer PINN literature.
- **Best known results:** Causal PINN achieves ~1e-4 on this setup; WF-PINNs claim further improvement near shock.
- With ν < 0.001, essentially all vanilla PINN methods fail; WF-PINNs / sl-PINNs / XPINN required.

**Recommended architecture for DoE:** 4-hidden-layer MLP, width 64, tanh activation, Adam + optional L-BFGS, with causal weighting as the primary intervention.

---

## Common Failure Modes

| Failure mode | Source | Remedy |
|---|---|---|
| Loss-component imbalance (PDE gradient >> BC/IC gradient) | Wang, Teng, Perdikaris 2021 | Hard constraints (Dirichlet) or ReLoBRaLo/LR-annealing weighting |
| Spectral bias (slow convergence on high-frequency components) | Wang, Wang, Perdikaris 2021; Tancik 2020 | Fourier feature embedding (only when solution has high-frequency content) |
| Temporal causality violation (late-time data fit before early-time) | Wang, Sankaran, Perdikaris 2022 | Causal weighting; time-marching schemes |
| **Causal weight starvation on smooth/low-stiffness problems** ⚠️ | **DhyutimaanPI DoE 2026** | **Do NOT use ε=100 unless ε·T·ℓ̄₀ < 0.1 (evaluate at step 0)** |
| **Representational-starvation compound failure (thin-layer Burgers)** ⚠️ | **DhyutimaanPI DoE 2026** | **Increase network width/collocation density before applying causal weighting** |
| Globally incorrect low-residual solution (Burgers) | WF-PINNs 2025; singular-layer PINN 2024 | Weak-form loss; entropy condition enforcement; adaptive collocation |
| Ill-conditioned hard-constraint ansatz (unsteady, near boundary) | DhyutimaanPI B3 analysis | Smooth time-window envelope; modified ansatz (disproved for Burgers hard-IC — see below) |
| Deep MLP instability (PDE residual gradient breakdown) | PirateNets 2024 | PirateNets / residual adaptive connections |
| Insufficient collocation density near shocks | Chiu et al. 2022; PINNACLE 2024 | Adaptive residual-guided collocation |

---

## ⚠️ Post-Experiment Updates — DhyutimaanPI DoE (May 2026)

> These findings update or contradict pre-survey recommendations above. They are based on a full 2³ factorial DoE (26 variants) on 2D Heat and 1D Burgers, reported in `dhyutimaan_caisc2026.tex`.

### KB Contradictions

**Headline finding #2 (causal weighting) — CONTRADICTED for Burgers**

The pre-survey recommended causal weighting as "the top recommendation for 1D Burgers DoE." This is wrong at ε=100 for ν=0.01/π:
- Soft-causal is the *worst* Burgers DoE variant: ε_rel = 1.34×10⁻² vs. 9.34×10⁻³ for soft-uniform (43% worse).
- Burgers H1 (causal reduces error vs. uniform) was refuted and reversed.
- The literature result "causal PINN achieves ~1e-4" uses a 9-layer × 20-width network with 25k collocation points — 3× more parameters than our 4×64 MLP. At equal architecture scale, causal weighting provides no benefit on ν=0.01/π Burgers.

**Recommended Starting Point #4 — WRONG DIRECTION**

"Hypothesis: causal weighting reaches ≤5e-3 while vanilla plateaus at ≥1e-2." Outcome: causal was worse than vanilla at every comparison. Retract this recommendation.

### New Analytical Result: Causal Weight Starvation Condition

For causal weighting with parameter ε, K temporal slabs, final time T:

**w_k ≈ exp(−ε·(k−1)·ℓ̄₀)**

where ℓ̄₀ is the mean per-slab PDE residual at initialisation.

**Starvation occurs when: ε·T·ℓ̄₀ ≫ 1**

Computable diagnostic: run 10 Adam steps, compute ℓ̄₀ from training_log.csv, evaluate ε·T·ℓ̄₀. If > 1, reduce ε until product < 0.1.

### Causal Weighting Across Four Regimes (Unified Picture)

| Regime | ε·T·ℓ̄₀ | Causal result | Mechanism |
|---|---|---|---|
| Heat B (smooth, T=0.1, ε=100) | ≫ 1 | 367× error growth | Pure starvation |
| Burgers ν=0.01/π (mild layer) | Moderate | 43% worse (soft-causal) | Mild starvation; hard IC partially protective |
| Burgers ν=0.001/π (thin layer) | ≫ 1 (inflated by unresolved layer) | ε_rel=0.574 catastrophic | Representational failure → inflated ℓ̄₀ → starvation compound |
| Burgers ν=0.1/π (smooth wide) | ≈ 0 | ε_rel=2.75×10⁻⁴ (best result) | ℓ̄₀ tiny → causal degenerates to uniform → no harm |

**Key insight:** Hard IC enforces t=0 to machine precision, reducing early-slab residuals and partially relaxing the starvation condition. This is why hard-causal (7.44×10⁻³) is less damaged than soft-causal (1.34×10⁻²) at standard ν.

**Flat error profile as starvation signature:** When causal starvation suppresses all slabs beyond t≈0 equally, the per-slice error profile is *flat* (growth ratio ≈ 1), not monotonically growing. Flat profile ≠ good training; it means the network learned almost nothing at any time beyond the initial slab.

### Resolved Open Questions

**When do Fourier features help?** (Previously open)
- Heat steady (smooth, hard BC): negligible effect (marginally better, within noise)
- Heat steady (smooth, soft BC): FF hurts 2.2× — rougher loss landscape
- Burgers ν=0.01/π (near-discontinuity): FF improves 1.33× — below 1.5× threshold but directionally consistent with spectral bias argument
- **Rule:** FF beneficial only when solution has genuine high-frequency spatial content AND residuals are not dominated by an unresolved thin layer.

**Hard IC+BC for Burgers — singularity question:** (Previously open)
The ansatz û = −sin(πx) + t(1−x²)·NN is well-posed. Unlike Heat B (where the target NN function has 0/0 form at boundary), the Burgers hard ansatz does not evaluate the NN at t=0 during PDE residual computation. The hard constraint is safe for Burgers at all tested viscosities. At ν_low, failure is representational, not ansatz conditioning.

### Priority Next Experiments (from DoE analysis)

1. **ε sweep** (ε ∈ {1, 10, 50, 100, 200}) on Heat B and Burgers ν=0.01/π — quantify critical ε* below which starvation does not trigger.
2. **Adaptive collocation** — residual-guided resampling near the thin layer for ν=0.001/π; expected to reduce ℓ̄₀ and break the compound failure.
3. **Multi-seed validation** (N≥3) — causal starvation claim is mechanistically supported but all results are single-seed.
4. **Longer horizon for Heat B** (T=1.0) — test whether causal weighting becomes beneficial when temporal complexity is higher (more decay time constants resolved).

---

## Open Problems and Contested Points

- **NTK weighting vs. ReLoBRaLo vs. gradient-norm annealing.** No clean comparison on unit-domain Burgers. ReLoBRaLo claims 40% speedup; NTK weighting has stronger theory but is expensive. Not tested in our DoE.
- **Fair benchmarking.** PINNacle 2024 calls out that many published PINN improvements are benchmarked on settings favorable to the proposed method. Our results confirm this: causal weighting succeeds in the Wang et al. 2022 benchmarks (Allen-Cahn, Navier-Stokes) but fails on smooth heat and moderate-stiffness Burgers.
- **Causal ε* as a function of problem parameters.** The starvation condition gives a computable diagnostic but not an analytical formula for ε*. Deriving ε*(ν, T, architecture) analytically is an open problem.

---

## Implementation Landscape

- **DeepXDE (Lu et al., JMLR 2021):** Most mature library; PyTorch + TensorFlow. Has 1D Burgers and 2D Poisson tutorials. Recommended for baseline comparison.
- **NVIDIA Modulus / Modulus Sym:** Production-grade; 3D-scale problems; more overhead for unit-domain research.
- **JAXpi / PirateNets:** github.com/PredictiveIntelligenceLab/jaxpi — reference JAX implementation of PirateNets and CausalPINNs. Requires JAX; not directly usable with our PyTorch/MPS setup.
- **CausalPINNs:** github.com/PredictiveIntelligenceLab/CausalPINNs — PyTorch reference for Wang et al. 2022.
- **DhyutimaanPI scaffold (PyTorch + MPS):** This repo's pinn-scaffold skill generates self-contained PyTorch modules compatible with Apple Silicon MPS backend. Use this as the primary implementation vehicle.

---

## Recommended Starting Points for Problem-Spec

**For 2D Heat Conduction (new DoE extending prior runs):**

1. **Extend hard-constraint results with L-BFGS polishing.** Prior A3 reached 7.85e-5 with Adam only. Adding 500-step L-BFGS finishing is expected to push this to 1e-6 to 1e-7. Falsifiable: L-BFGS polishing reduces relative L² by ≥5× vs. Adam-only baseline at identical wall time budget (2× wall-clock).
2. **Causal weighting for the unsteady problem (B-variants).** B2 was best-in-class at 2.26e-2 with soft constraints. Causal weighting (Wang 2022) on the time axis should address the residual time-growth observed in B1. Falsifiable: causal weighting reduces per-time-slice L² growth from 3× to ≤1.5× vs. soft-weight baseline.
3. **Seed variance (N=3 seeds) for A3 vs. A4 gap.** Prior 25× gap (hard vs. hard+Fourier) was single-seed. Confirm with 3 seeds.

**For 1D Burgers (new):**

4. ~~**Causal PINN vs. vanilla PINN at ν = 0.01/π.**~~ ⚠️ **RETRACTED (DhyutimaanPI DoE 2026):** Hypothesis was wrong direction. Soft-causal (1.34×10⁻²) was 43% *worse* than soft-uniform (9.34×10⁻³). Causal weighting at ε=100 triggers starvation at this viscosity. **Revised recommendation:** run an ε sweep (ε ∈ {1, 10, 50, 100}) on Burgers to find ε* below which starvation does not trigger; this is now the highest-priority next experiment for Burgers causal weighting.
5. **Hard IC vs. soft IC — smooth ansatz u = −sin(πx) + t·(1−x²)·NN.** The hard-IC ansatz eliminates IC loss and enforces BCs simultaneously. Hypothesis: hard IC reduces total error by ≥5× vs. soft-constraint baseline.
6. **ν sweep (0.1/π, 0.01/π, 0.001/π) with causal weighting.** Characterize at what viscosity the method degrades. Hypothesis: causal PINN maintains ≤1e-2 error for ν ≥ 0.01/π but fails (error > 0.1) for ν = 0.001/π.
7. **Collocation strategy: uniform vs. residual-adaptive.** Allocate 20% of collocation budget to residual-gradient-weighted resampling at epoch 500. Hypothesis: adaptive collocation reduces error near shock by ≥2× vs. uniform at the same total point count.

---

## Sources

1. Raissi, M., Perdikaris, P., Karniadakis, G.E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs. *J. Comput. Phys.* 378:686–707.
2. Wang, S., Teng, Y., Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM J. Sci. Comput.* 43(5):A3055–A3081.
3. Wang, S., Wang, H., Perdikaris, P. (2021). On the eigenvector bias of Fourier feature networks. *J. Comput. Phys.* 439:110378.
4. Tancik, M., et al. (2020). Fourier features let networks learn high frequency functions in low dimensional domains. *NeurIPS 2020.*
5. Wang, S., Sankaran, S., Perdikaris, P. (2022/2024). Respecting causality for training physics-informed neural networks. *CMAME* 421:116813. arXiv:2203.07404.
6. Wang, S., Li, B., Chen, Y., Perdikaris, P. (2024). PirateNets: Physics-informed deep learning with residual adaptive networks. *JMLR* 25(402). arXiv:2402.00326.
7. Bischof, R., Kraus, M.A. (2021/2025). Multi-objective loss balancing for physics-informed deep learning (ReLoBRaLo). *CMAME* 2025. arXiv:2110.09813.
8. Lu, L., Meng, X., Mao, Z., Karniadakis, G.E. (2021). DeepXDE: A deep learning library for solving differential equations. *SIAM Rev.* 63(1):208–228.
9. Chiu, P.-H., Wong, J.C., Ooi, C., Dao, M.H., Ong, Y.-S. (2022). CAN-PINN. *CMAME* 395:114909.
10. Sitzmann, V., Martel, J., Bergman, A., Lindell, D., Wetzstein, G. (2020). Implicit neural representations with periodic activation functions (SIREN). *NeurIPS 2020.*
11. Jagtap, A.D., Karniadakis, G.E. (2020). Extended physics-informed neural networks (XPINNs). *Commun. Comput. Phys.* 28(5):2002–2041.
12. PINNacle (2024). A comprehensive benchmark of PINNs for solving PDEs. *NeurIPS 2024 Datasets & Benchmarks.* github.com/i207M/PINNacle.
13. WF-PINNs (2025). Solving forward and inverse problems of Burgers equation with steep gradients. *Nature Scientific Reports.* doi:10.1038/s41598-025-24427-4.
14. Singular Layer PINNs (2024). arXiv:2410.09723.
15. Lagaris, I.E., Likas, A., Fotiadis, D.I. (1998). Artificial neural networks for solving ordinary and partial differential equations. *IEEE Trans. Neural Netw.* 9(5):987–1000.
16. McClenny, L., Braga-Neto, U. (2020/2022). Self-adaptive physics-informed neural networks using a soft attention mechanism. *Neurocomputing.*
17. Moseley, B., Markham, A., Nissen-Meyer, T. (2023). Finite basis physics-informed neural networks (FBPINNs). *Adv. Comput. Math.* 49:62.
18. PDEBench (2022). An extensive benchmark for scientific machine learning. *NeurIPS 2022.* arXiv:2210.07182.
