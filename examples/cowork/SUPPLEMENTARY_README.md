# Supplementary Material — DhyutimaanPI (CAISc 2026)

**Paper:** *DhyutimaanPI: An Agentic Pipeline for Reproducible PINN Failure-Mode Discovery
via Systematic Factorial Experimentation*

This archive contains everything needed to inspect, reproduce, and extend the experiments
reported in the paper: skill definitions, problem specifications, training code, per-run
artefacts (training logs, checkpoints, verification metrics), composite figures, and the
updated PINN knowledge base.

---

## Archive Layout

```
DhyutimaanPI-supplementary/
├── SUPPLEMENTARY_README.md          ← this file
├── paper/
│   ├── dhyutimaan_caisc2026.pdf     ← compiled paper (19 pages)
│   └── dhyutimaan_caisc2026.tex     ← LaTeX source
├── skills/                          ← the four DhyutimaanPI skill definitions
│   ├── literature-survey-pinn/SKILL.md
│   ├── pinn-problem-spec/SKILL.md
│   ├── pinn-scaffold/SKILL.md
│   └── pinn-analysis-report/SKILL.md
├── experiments/
│   ├── heat2D/
│   │   ├── problem-spec.md          ← Framer output: DoE design + hypotheses H1–H5
│   │   ├── analysis-report.md       ← Analyst output: adversarial verdict on all 16 variants
│   │   ├── pinn_run/                ← Implementer output: model, train, verify scripts
│   │   │   ├── model.py
│   │   │   ├── problem.py
│   │   │   ├── train.py
│   │   │   ├── run.py
│   │   │   └── verify.py
│   │   └── runs/                    ← one subdirectory per DoE variant (16 total)
│   │       ├── hard-adam/
│   │       ├── hard-adam-ff/
│   │       ├── hard-causal/
│   │       ├── hard-causal-ff/
│   │       ├── hard-lbfgs/
│   │       ├── hard-lbfgs-ff/
│   │       ├── hard-std/
│   │       ├── hard-std-ff/
│   │       ├── soft-adam/
│   │       ├── soft-adam-ff/
│   │       ├── soft-causal/
│   │       ├── soft-causal-ff/
│   │       ├── soft-lbfgs/
│   │       ├── soft-lbfgs-ff/
│   │       ├── soft-std/
│   │       ├── soft-std-ff/
│   │       └── hypotheses_summary.json   ← machine-readable verdict for all H1–H5
│   ├── burgers1D/
│   │   ├── problem-spec.md          ← Framer output: DoE design + hypotheses H1–H5
│   │   ├── pinn_run/                ← Implementer output: model, train, verify scripts
│   │   │   ├── model.py
│   │   │   ├── problem.py
│   │   │   ├── train.py
│   │   │   └── verify.py
│   │   └── runs/                    ← 8 DoE variants + 2 viscosity stress tests
│   │       ├── hard-causal/
│   │       ├── hard-causal-ff/
│   │       ├── hard-uniform/
│   │       ├── hard-uniform-ff/
│   │       ├── soft-causal/
│   │       ├── soft-causal-ff/
│   │       ├── soft-uniform/
│   │       ├── soft-uniform-ff/
│   │       ├── best-nu-high/        ← ν = 0.1/π stress test
│   │       ├── best-nu-low/         ← ν = 0.001/π stress test (ε = 0.574, confirmed H5)
│   │       └── hypotheses_summary.json
│   └── figures/
│       ├── figure_heat_A.pdf        ← Heat Problem A comparison (paper Fig. 1)
│       ├── figure_heat_B.pdf        ← Heat Problem B comparison (paper Fig. 2)
│       └── figure_burgers.pdf       ← Burgers DoE + stress-test comparison (paper Fig. 3)
└── knowledge-base/
    └── pinn-knowledge-base.md       ← Surveyor output + post-experiment updates
```

---

## Each Run Directory Contains

| File | Description |
|---|---|
| `training_log.csv` | Per-iteration: `step`, `loss_pde`, `loss_ic`, `loss_bc`, `loss_total`, `lr` |
| `verification.json` | Final metrics: `rel_l2`, `max_abs_error`, per-slice errors, hypothesis verdicts |
| `checkpoint.pt` | PyTorch model weights at final iteration |
| `loss_curve.png` | Training loss vs. iteration plot |
| `solution_comparison.png` | Predicted vs. reference solution side-by-side |

---

## The Four-Skill Pipeline

The paper's experiments were produced end-to-end by chaining four skills in sequence.
Each skill's contract (inputs → outputs) is defined in its `SKILL.md`.

```
Surveyor  →  Framer  →  Implementer  →  Analyst
(literature   (problem-    (pinn-          (pinn-
 survey)       spec)        scaffold)       analysis-report)
```

| Skill | SKILL.md location | Output artefact |
|---|---|---|
| Surveyor | `skills/literature-survey-pinn/SKILL.md` | `pinn-knowledge-base.md` |
| Framer | `skills/pinn-problem-spec/SKILL.md` | `problem-spec.md` |
| Implementer | `skills/pinn-scaffold/SKILL.md` | `pinn_run/` code + `verify.py` |
| Analyst | `skills/pinn-analysis-report/SKILL.md` | `analysis-report.md` |

---

## Reproducing the Experiments

**Environment:** Python 3.11, PyTorch ≥ 2.1, Apple Silicon MPS backend
(conda environment `torchmps`; CPU fallback works on any platform).

```bash
# Heat 2D — all 16 variants
cd experiments/heat2D
bash ../../run_all_experiments.sh heat2D

# Burgers 1D — all 8 DoE + 2 stress tests
cd experiments/burgers1D
bash ../../run_all_experiments.sh burgers1D
```

Each run writes to `runs/<label>/` and takes 5–30 minutes depending on variant and hardware.
The `verify.py` script generates `verification.json` and comparison plots automatically at
the end of each run.

---

## Key Numerical Results at a Glance

### Heat 2D (Problem A — Steady Poisson)

| Best variant | ε_rel |
|---|---|
| hard-lbfgs-ff | **2.14 × 10⁻⁶** |
| soft-adam (baseline) | 2.47 × 10⁻⁴ |

### Heat 2D (Problem B — Unsteady Parabolic)

| Variant | ε_rel | Character |
|---|---|---|
| hard-std | **5.04 × 10⁻⁴** | Flat error profile |
| soft-causal | 1.18 × 10⁻¹ | Causal starvation — 367× error growth |

### Burgers 1D (ν = 0.01/π, DoE)

| Best variant | ε_rel |
|---|---|
| hard-uniform-ff | **7.01 × 10⁻³** |

### Burgers 1D (Viscosity Stress Test)

| ν | ε_rel | Outcome |
|---|---|---|
| 0.1/π | 2.75 × 10⁻⁴ | Smooth; all methods work |
| 0.001/π | 0.574 | **H5 confirmed — method fails** |

---

## Hypothesis Verdicts Summary

All 10 pre-registered hypotheses are recorded in `runs/hypotheses_summary.json`
for each problem. **9 of 10 are refuted; 1 is confirmed** (Burgers H5: catastrophic
failure at ν = 0.001/π). See paper Section 4 and the `analysis-report.md` files for
full mechanistic explanations.
