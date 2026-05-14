# AGENTS.md

Cross-tool agent instructions for the DhyutimaanPI PINN research workflow. Read by Antigravity, Cursor, Claude Code, and other IDEs that respect the `AGENTS.md` standard.

## Project overview

This repository contains four chained skills for conducting Physics-Informed Neural Network (PINN) research:

1. **Surveyor** — `skills/literature-survey-pinn/SKILL.md` → produces `pinn-knowledge-base.md`
2. **Framer** — `skills/pinn-problem-spec/SKILL.md` → produces `problem-spec.md`
3. **Implementer** — `skills/pinn-scaffold/SKILL.md` → produces `pinn_run/` (Python module + verification harness)
4. **Analyst** — `skills/pinn-analysis-report/SKILL.md` → produces `analysis-report.md`

The skills chain via stable artifact filenames. Filenames are the contract; do not rename them when modifying the skills.

## When to invoke which skill

| Trigger phrases | Skill |
|---|---|
| "literature survey", "lit review", "state of the art on PINNs", "what's been tried for this PDE" | `literature-survey-pinn` |
| "problem spec", "frame this PINN problem", "draft a problem statement", "set up a PINN for X" | `pinn-problem-spec` |
| "scaffold the PINN", "implement the code", "PyTorch version", "code up this spec" | `pinn-scaffold` |
| "analyze the run", "produce a report", "did it work", "review the training output" | `pinn-analysis-report` |

When a user request matches a skill's `description` field in its SKILL.md frontmatter, read that SKILL.md before producing output and follow its workflow exactly.

## Conventions

- **PyTorch with `torch.func`** is the default implementation backend. DeepXDE is the supported alternate.
- **Apple Silicon (MPS)** is a supported target; fall back to CPU when `torch.func` ops fail on a given PyTorch version.
- **The method of manufactured solutions** is the default reference strategy when no analytic solution is available.
- **Loss components are always logged separately** to `training_log.csv` — never aggregate them into a single column.
- **The Analyst is adversarial by design.** Do not soften its tone or convert "inconclusive" verdicts to "confirmed".

## Working directory layout (after a full run)

```
.
├── pinn-knowledge-base.md       # from Surveyor
├── problem-spec.md              # from Framer
├── pinn_run/                    # from Implementer
│   ├── problem.py
│   ├── model.py
│   ├── train.py
│   ├── verify.py
│   ├── run.py
│   └── outputs/                 # produced by `python pinn_run/run.py`
│       ├── training_log.csv
│       ├── checkpoint.pt
│       ├── verification.json
│       └── *.png
└── analysis-report.md           # from Analyst
```

## Pause-for-review boundaries

The natural review points are after each artifact is produced:

1. After `pinn-knowledge-base.md` — verify the survey covered the right ground
2. After `problem-spec.md` — verify the hypotheses are falsifiable and the reference is solid
3. After `pinn_run/` is scaffolded — verify the code matches the spec before training
4. After training completes — run the Analyst

In tools that support it (Antigravity, Claude Code), set Artifact Review Policy to "Asks for Review" or equivalent.

## See also

- `README.md` — project overview
- `docs/using-with-claude-code.md` — Claude Code setup
- `docs/using-with-claude-ai.md` — Claude.ai setup
- `docs/using-with-cursor.md` — Cursor setup
- `docs/using-with-antigravity.md` — Antigravity setup
