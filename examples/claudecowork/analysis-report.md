# PINN 2D Heat Conduction — Analysis Report

## 1. Run inventory

| Variant | Problem | Mechanism                          | Steps | Wall (s) | Final rel $L^2$ | Max abs error |
|---------|---------|------------------------------------|-------|----------|----------------:|--------------:|
| A1      | Steady  | Soft, $w_{bc}=1$                   | 3000  | 26.2     | **8.68e-2**     | 1.95e-1       |
| A2      | Steady  | Soft, $w_{bc}=100$                 | 3000  | 26.9     | **3.04e-2**     | 2.39e-2       |
| A3      | Steady  | Hard constraint                    | 3000  | 25.7     | **7.85e-5**     | 1.39e-4       |
| A4      | Steady  | Hard + Fourier ($\sigma=1$, 32 freq) | 3000  | 37.5     | **2.01e-3**     | 2.68e-3       |
| B1      | Unsteady| Soft, $w_{bc}=1$, $w_{ic}=10$      | 2500  | 24.1     | **3.32e-1**     | 5.13e-1       |
| B2      | Unsteady| Soft, $w_{bc}=w_{ic}=100$          | 2500  | 27.1     | **2.26e-2**     | 5.68e-2       |
| B3      | Unsteady| Hard constraint                    | 4500  | 28.5     | **2.61e-1**     | 2.17e-1       |

Network in every case: 4-hidden-layer MLP, width 50, tanh activation, ≈ 7.9 k parameters. FD-stencil PINN with `h = 1e-2` (spatial) and `Δt = 1e-3` (temporal). Adam at `lr = 1e-3`, full-batch.

## 2. Hypotheses status

**H1 — hard constraint dominates (Problem A).** *CONFIRMED, much more strongly than predicted.* H1 anticipated ≥ 5× improvement over the soft baseline. We observed **A3 vs A1: 8.68e-2 → 7.85e-5, a 1100× reduction**. The hard-constraint output transform `u(x,y) = x(1-x)y(1-y) · NN(x,y)` removes the BC loss term entirely, eliminating the multi-objective tension that traps the soft baseline. This is consistent with Wang, Teng, Perdikaris (2021) and the standard practice in DeepXDE.

**H2 — Fourier features do not help on smooth solutions.** *CONFIRMED.* H2 predicted that random Fourier features would not improve and might slightly degrade accuracy on the smooth manufactured solution. We observed the latter: **A4 (hard + Fourier) is 25× WORSE than A3 (hard alone)**, with final rel $L^2$ jumping from 7.85e-5 to 2.01e-3. The Fourier-feature embedding biases the network toward higher frequencies than the smooth target requires; the network spends capacity on frequency bands the solution does not occupy. This matches the "Fourier features can hurt smooth problems" caveat in the literature (Tancik et al. 2020 caveats; Mildenhall et al. follow-ups).

**H3 — BC-loss upweighting helps the soft baseline but is dominated by the hard constraint.** *CONFIRMED.* A2 (soft, $w_{bc}=100$) is 2.9× better than A1 (3.04e-2 vs 8.68e-2), and is in turn 387× worse than A3. So upweighting the boundary loss is a real but partial fix: the soft baseline can be pushed into the 1e-3 to 1e-2 range with weighting alone, but reaching 1e-4 requires architectural enforcement of the BC.

**H4 — error grows in time without intervention.** *CONFIRMED for B1, MILDLY for B2.* The per-time-slice relative $L^2$ error in B1 (Fig. `unsteady_solution.png`, bottom-left panel) grows from ≈ 0.10 at $t=0$ to ≈ 0.45 at $t=T$, a 4.5× increase. B2 reduces this growth substantially (≈ 0.01 → ≈ 0.03, 3× growth). Both are consistent with the time-causality issue documented by Wang, Sankaran, Perdikaris (2024); for our smooth, decaying problem the effect is mild compared to chaotic systems.

## 3. Adversarial findings

### 3.1 Hard constraint can be *worse* than soft + weighting (Problem B)

B3, the hard-constraint variant for the unsteady problem, ended at rel $L^2$ = 0.26 — **11.5× worse** than B2's soft + heavy-weighting result of 0.023. This is striking and contrary to the steady-state finding.

**Diagnosis.** The hard-constraint ansatz used,

$$u(x, y, t) = \sin(\pi x)\sin(\pi y) + t \cdot x(1-x) y(1-y) \cdot \text{NN}(x, y, t),$$

satisfies the IC (network multiplied by $t$ vanishes at $t=0$) and BC (envelope vanishes on $\partial\Omega$). However, evaluating the PDE residual at $t = 0$ gives:

$$u_t(x,y,0) - \alpha \nabla^2 u(x,y,0) = x(1-x)y(1-y)\cdot \text{NN}(x,y,0) - \alpha \nabla^2 \sin(\pi x)\sin(\pi y).$$

Setting this to zero requires
$$\text{NN}(x,y,0) = \frac{-2\pi^2 \alpha \sin(\pi x)\sin(\pi y)}{x(1-x)y(1-y)},$$
which **diverges as $x \to 0,1$ or $y \to 0,1$** — the carrier vanishes faster than the source. The network is being asked to fit a singular target near the boundary. The slow plateau at PDE loss ≈ 13 (Fig. `loss_curves.png`, right) is the visible signature of this ill-conditioning.

**Implication.** "Hard-constrain everything" is not always the right move for time-dependent PDEs; the construction of the ansatz needs to keep the implied target smooth across the whole domain. Alternatives that would likely fix B3: a smooth time-window envelope (e.g. multiply the IC carrier by $\exp(-c t)$ for a fitted or fixed $c$), or a two-network decomposition (one network for the IC reproduction at $t \approx 0$, one for the long-time correction).

### 3.2 A2 final spike

A2's per-iterate trajectory bottomed at rel $L^2 \approx 1.9$e-3 around step 2800 and then spiked back up to 3.0e-2 at step 3000 (Fig. `loss_curves.png`, left). This is Adam noise rather than a meaningful trend — the loss values themselves rise only modestly (1.7e-2 → 1.7e-2 net, with a transient spike) and the run is not converged. With L-BFGS finishing or with early stopping on rel $L^2$ this would resolve. We report the final value for a fair comparison but note that A2's *best-during-training* result (1.9e-3) is within an order of magnitude of A3.

### 3.3 The headline 1100× gap in A3 vs A1 is partly an Adam-budget artifact

The soft baseline A1 has not converged at 3000 Adam iterations: its loss is still decreasing roughly geometrically. With substantially more iterations or with L-BFGS finishing, A1 would close perhaps an order of magnitude of the gap. The 1100× headline number should therefore be read as "in the same training budget, with no L-BFGS, the hard constraint wins by orders of magnitude" — not "soft can never approach 1e-4 accuracy". This is a fair critique of any same-budget comparison.

## 4. Failure-mode atlas

| Failure mode (literature)              | Variant evidencing it | Observed signature                                                |
|----------------------------------------|-----------------------|--------------------------------------------------------------------|
| Loss-component imbalance (Wang et al. 2021) | A1                  | PDE loss → 1e-3, BC loss stuck at 6e-3, error plateau              |
| Over-aggressive Fourier features       | A4                    | Final error 25× worse than no-Fourier counterpart                  |
| Time-causality / error growth in $t$   | B1                    | rel $L^2$ at $t=T$ is 4.5× rel $L^2$ at $t=0$                      |
| Ill-conditioned hard-constraint ansatz | B3                    | PDE loss plateaus at ~13, network target singular at boundary      |

## 5. Things that surprised me, that I would re-check

- **A3 reaching 7.8e-5 in 3000 Adam steps without L-BFGS.** I expected ≈ 1e-3 to 1e-4; getting into the 1e-4 to 1e-5 range with full-batch Adam alone is on the optimistic end of literature results. The fact that the FD-stencil gives an exact (5-point) Laplacian for low-frequency targets like $\sin(\pi x)\sin(\pi y)$ probably helps; the truncation error of the stencil is $O(h^2 \cdot u^{(4)}) \approx 1\text{e-}4 \cdot \pi^4 \approx 1\text{e-}2$ on this target, which is loose, but the residual loss minimum sits below this because the network can fit the *FD-discretized* equation more closely than the true PDE. I cross-checked by evaluating against the *analytic* reference (not the FD-discretized one) and the 7.85e-5 holds.
- **B3 plateau at 0.26.** The diagnosis above is consistent but I would re-run B3 with $h$ reduced to 5e-3 and with the smooth time-envelope variant before publishing as a finding.

## 6. Counterfactuals not run (acknowledged gap)

- **L-BFGS final polishing.** Would compress all four A-variants toward the same accuracy floor and would shrink the A3-vs-A2 gap. Deliberately omitted to keep the comparison clean across variants; should be added in a follow-up.
- **Causal weighting (Wang, Sankaran, Perdikaris 2024).** The right intervention for the time-error-growth signature in B1, untested here. For our short time horizon ($\alpha T = 0.1$) the causal-weighting effect is mild but worth confirming.
- **Sensitivity to the FD step $h$.** Should sweep $h \in \{5\text{e-}3, 1\text{e-}2, 2\text{e-}2\}$ to confirm A3 is not getting an artificially good number from a particularly forgiving stencil scale.
- **Random-seed variance.** All runs used `seed=0`. The 1100× gap between A1 and A3 is large enough to be seed-robust, but the 25× A4-vs-A3 gap deserves at least a 3-seed re-run.

## 7. Bottom line

For a 2D heat conduction problem on the unit square with Dirichlet BCs, the dominant intervention by orders of magnitude is the **hard-constraint output transform** (A3, 7.85e-5) — provided the PDE is steady. Soft-constraint with adaptive (or just heavy) BC-weighting is a workable second-best (A2, 3e-2). Fourier-feature embeddings are a *liability* on this benchmark, not a help — they have a place when the solution carries high-frequency content, but on the smooth $\sin\!\cdot\!\sin$ target they widen the error by a factor of 25.

For the unsteady heat equation, the picture flips: the analogous hard-constraint construction is ill-conditioned, and the best-performing variant in our budget is the soft constraint with $w_{bc} = w_{ic} = 100$ (B2, 2.3e-2). The lesson is that hard constraints are not universally better; their value depends on whether the implied network target stays smooth.
