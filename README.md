# DhyutimaanPI

**A four-skill agentic pipeline for reproducible Physics-Informed Neural Network experimentation.**

DhyutimaanPI converts informal PDE intuitions into pre-registered, falsifiable experiments and traces every failure to a specific mechanism. Applied to 26 factorial variants across 2D heat conduction and 1D Burgers, 9 of 10 pre-registered hypotheses are refuted — and every refutation is explained analytically.

> *"DhyutimaanPI is a four-skill agentic pipeline for reproducible PINN experimentation; across 26 factorial variants on 2D heat and 1D Burgers, 9 of 10 pre-registered hypotheses are refuted and every failure traced to a mechanism."*
> — CAISc 2026 submission

---

## The pipeline

```
Surveyor            Framer              Implementer         Analyst
─────────────────   ─────────────────   ─────────────────   ─────────────────
literature-survey-  pinn-problem-spec   pinn-scaffold       pinn-analysis-
pinn                                                        report

Input: topic/PDE    Input: KB or desc.  Input: spec         Input: run outputs
Output: KB.md       Output: spec.md     Output: pinn_run/   Output: report.md
```

Each stage hands off a human-readable artefact. The contract between skills is **filenames** — any skill can be re-run independently as long as its upstream artefact exists.

---

## Repository layout

```
DhyutimaanPI/
├── README.md
├── skills/                            ← four skill definitions (SKILL.md each)
│   ├── literature-survey-pinn/
│   ├── pinn-problem-spec/
│   ├── pinn-scaffold/
│   └── pinn-analysis-report/
├── examples/
│   └── cowork/                        ← CAISc 2026 study
│       ├── dhyutimaan_caisc2026.pdf   ← paper (19 pages)
│       ├── dhyutimaan_caisc2026.tex   ← LaTeX source
│       ├── pinn-knowledge-base.md     ← Surveyor output + post-experiment updates
│       ├── SUPPLEMENTARY_README.md    ← reviewer navigation guide
│       ├── figures/                   ← figure_heat_A/B.pdf, figure_burgers.pdf
│       ├── heat2D/
│       │   ├── problem-spec.md        ← Framer output (DoE, 5 hypotheses)
│       │   ├── analysis-report.md     ← Analyst output (16 variants)
│       │   ├── pinn_run/              ← model.py, train.py, verify.py
│       │   └── runs/                  ← 16 variant dirs (training_log, verification, checkpoint)
│       └── burgers1D/
│           ├── problem-spec.md        ← Framer output (DoE, 5 hypotheses)
│           ├── pinn_run/              ← model.py, train.py, verify.py
│           └── runs/                  ← 8 DoE + 2 stress-test dirs
└── docs/
    ├── using-with-claude-code.md
    ├── using-with-claude-ai.md
    ├── using-with-cursor.md
    └── using-with-antigravity.md
```

---

## Quick start

```bash
# 1. Run the surveyor
/literature-survey-pinn   # produces pinn-knowledge-base.md

# 2. Frame the problem
/pinn-problem-spec        # produces problem-spec.md

# 3. Scaffold and run
/pinn-scaffold            # produces pinn_run/ with train.py + verify.py
conda activate torchmps
python pinn_run/run.py

# 4. Analyse
/pinn-analysis-report     # produces analysis-report.md
```

See `docs/` for environment-specific guides (Claude Code, Claude.ai, Cursor, Antigravity).

---

## Why this architecture

**Separable failure.** When something goes wrong, you know which stage failed.

**Verifiable handoffs.** Every artefact (`problem-spec.md`, `verification.json`, `analysis-report.md`) is human-readable and editable. You can stop, inspect, override, and resume at any stage boundary.

**Adversarial analysis.** The Analyst skill does not celebrate convergence. It audits every failure mode the spec listed, checks every hypothesis including the ones that look obviously confirmed, and flags every claim that rests on a single seed.

**Pre-registration enforces honesty.** Hypotheses are written and committed before training runs. The spec cannot be edited post-hoc to match results.

---

## Design principles enforced by the skills

| Practice | Where enforced |
|---|---|
| Falsifiable, threshold-quantified hypotheses | `pinn-problem-spec` rejects vague predictions |
| Reference solution required | `pinn-scaffold` generates MMS or FD reference before training |
| Loss components logged separately | `train.py` writes `loss_pde`, `loss_ic`, `loss_bc` as separate CSV columns |
| Per-slice temporal error | `verify.py` computes error at each time slice, not just global L² |
| Failure-mode audit | `pinn-analysis-report` walks every spec §9 item and reports whether it triggered |

---

## Hardware and stack

- **Backend:** PyTorch ≥ 2.1 with `torch.func` (grad, vmap, jacrev, hessian)
- **Primary target:** Apple Silicon MPS (`torchmps` conda environment)
- **Fallback:** CPU or CUDA; behaviour is identical

---

## CAISc 2026 results at a glance

| Problem | Best variant | ε_rel |
|---|---|---|
| Heat 2D — Poisson (steady) | hard-lbfgs-ff | 2.14 × 10⁻⁶ |
| Heat 2D — Parabolic (unsteady) | hard-std | 5.04 × 10⁻⁴ |
| Burgers 1D — ν = 0.01/π (DoE) | hard-uniform-ff | 7.01 × 10⁻³ |
| Burgers 1D — ν = 0.001/π (stress) | hard-causal | 0.574 (**H5 confirmed**) |

Key finding: causal temporal weighting produces 367× error growth when applied to a smooth parabolic problem — explained analytically by the weight starvation condition ε·T·ℓ̄₀ ≫ 1.

---

## Citation

```bibtex
@inproceedings{sundar2026dhyutimaanpi,
  title     = {{DhyutimaanPI}: An Agentic Pipeline for Reproducible {PINN}
               Failure-Mode Discovery via Systematic Factorial Experimentation},
  author    = {Sundar, Rahul},
  booktitle = {Proceedings of CAISc 2026},
  year      = {2026}
}
```

---

## License

MIT License
