# DhyutimaanPI

**Automated research and discovery agents for science and engineering, grounded in Physics-Informed Neural Networks (PINNs).**

This repository contains a chain of four Claude skills that together cover the full research loop for a computational mechanics study:

```
literature-survey-pinn  →  pinn-problem-spec  →  pinn-scaffold  →  pinn-analysis-report
        (survey)             (frame)               (implement)        (analyze + report)
```

Each skill is a self-contained `SKILL.md` file under `skills/`. They are designed to be invoked individually or as a pipeline, and to hand off structured artifacts (`pinn-knowledge-base.md`, `problem-spec.md`, training outputs, `analysis-report.md`) between stages.

---

## What's in this repo

```
DhyutimaanPI/
├── README.md                        ← you are here
├── skills/
│   ├── literature-survey-pinn/
│   │   └── SKILL.md
│   ├── pinn-problem-spec/
│   │   └── SKILL.md
│   ├── pinn-scaffold/
│   │   └── SKILL.md
│   └── pinn-analysis-report/
│       └── SKILL.md
└── docs/
    ├── using-with-claude-code.md
    ├── using-with-claude-ai.md
    ├── using-with-cursor.md
    └── using-with-antigravity.md
```

## The skill chain at a glance

| Skill | Input | Output | When to invoke |
|---|---|---|---|
| `literature-survey-pinn` | A topic or PDE class | `pinn-knowledge-base.md` | Starting a new study; need grounding in prior work |
| `pinn-problem-spec` | Knowledge base or informal description | `problem-spec.md` | Ready to frame a concrete experiment |
| `pinn-scaffold` | `problem-spec.md` | `pinn_run/` (Python module + verification harness) | Ready to write code |
| `pinn-analysis-report` | `pinn_run/outputs/*` + spec | `analysis-report.md` | Training has finished |

The contract between skills is **filenames**. Each skill reads specific files from the working directory and writes specific files back. This means any skill can be invoked independently as long as its upstream artifact exists.

## Quick start

Pick the doc that matches your environment:

- **[Claude Code](docs/using-with-claude-code.md)** — primary supported workflow; skills auto-load
- **[Claude.ai](docs/using-with-claude-ai.md)** — upload the `.skill` files in chat or use Projects
- **[Cursor](docs/using-with-cursor.md)** — via Claude integration; manual skill invocation
- **[Antigravity](docs/using-with-antigravity.md)** — Google's agentic IDE; adapt skills as agent instructions
- **VS Code (generic)** — use the Claude extension; same flow as Cursor

## Why a chain of skills rather than one big agent

Three reasons:

1. **Separable failure.** When something goes wrong, you know which stage failed. A monolithic agent collapses survey + framing + implementation + analysis into one opaque output.
2. **Verifiable handoffs.** Each artifact (`problem-spec.md`, `verification.json`, etc.) is human-readable and reviewable. You can stop, edit, and resume at any boundary.
3. **Reusability.** The analysis skill works on *any* PINN run that produces the expected output schema, not just runs from this scaffold. The scaffold works for any spec that matches the contract.

This mirrors how a competent human research workflow is structured — survey, frame, implement, analyze — and treats each step as a first-class artifact rather than a transient prompt.

## Design principles

The skills enforce four practices that PINN papers commonly skip:

- **Falsifiable hypotheses.** `pinn-problem-spec` rejects vague hypotheses ("the PINN will work") and forces specific, measurable predictions.
- **Reference solutions are non-negotiable.** Method of manufactured solutions is the default; the analysis skill cannot operate without a ground truth.
- **Loss components are reported separately.** A low total loss can hide a high boundary error. The scaffold logs every term to its own CSV column.
- **Adversarial analysis.** `pinn-analysis-report` is not a celebrator. It audits every failure mode the spec listed and flags every confirmation that rests on shaky evidence.

## Stack and hardware

Default implementation backend: **PyTorch ≥ 2.0 with `torch.func`** (grad, vmap, jacrev, hessian). DeepXDE is supported as an alternate backend when the user wants a higher-level interface.

Tested target: Apple Silicon (M-series, including M5 Pro) with the MPS backend, and Linux/CUDA. Small PINNs often train fastest on CPU due to kernel-launch overhead on small batches — this is normal.

## Contributing

Each skill follows Anthropic's [skill format](https://docs.claude.com): YAML frontmatter (name, description) followed by markdown instructions. When modifying:

- Keep `SKILL.md` under 500 lines (current files are 170–200)
- Preserve the artifact filenames in the chain (`problem-spec.md`, `pinn_run/outputs/verification.json`, etc.) — downstream skills depend on them
- The skill `description` field is the primary trigger mechanism; make it concrete

## License

TBD — add a LICENSE file before public use.

## Acknowledgments

Built for the DhyutiLABS / CAISC research workflow. Skill structure follows Anthropic's `skill-creator` conventions.
