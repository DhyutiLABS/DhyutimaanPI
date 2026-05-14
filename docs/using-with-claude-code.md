# Using DhyutimaanPI Skills with Claude Code

Claude Code is the primary supported environment for these skills. They auto-load when relevant, can be triggered explicitly, and the project-level installation travels with the repo.

## Installation

You have two choices: **project-scoped** (the skills live in this repo, every collaborator gets them via git) or **personal** (the skills live in your home directory, available across all your projects).

### Option A — Project-scoped (recommended for teams)

If you've cloned this repo, the skills are already at `skills/<name>/SKILL.md`. To make Claude Code pick them up, symlink or copy them into `.claude/skills/` at the repo root:

```bash
cd /path/to/DhyutimaanPI
mkdir -p .claude/skills
ln -s ../../skills/literature-survey-pinn .claude/skills/literature-survey-pinn
ln -s ../../skills/pinn-problem-spec      .claude/skills/pinn-problem-spec
ln -s ../../skills/pinn-scaffold          .claude/skills/pinn-scaffold
ln -s ../../skills/pinn-analysis-report   .claude/skills/pinn-analysis-report
```

(Use `cp -r` instead of `ln -s` on Windows or if you prefer copies.)

Commit `.claude/skills/` so collaborators get them automatically:

```bash
git add .claude/skills
git commit -m "Install PINN skills for Claude Code"
```

### Option B — Personal (available in all your projects)

Copy or symlink to `~/.claude/skills/`:

```bash
cp -r skills/literature-survey-pinn ~/.claude/skills/
cp -r skills/pinn-problem-spec      ~/.claude/skills/
cp -r skills/pinn-scaffold          ~/.claude/skills/
cp -r skills/pinn-analysis-report   ~/.claude/skills/
```

### Verify

Start Claude Code in any directory and run:

```
/skills
```

You should see the four PINN skills listed. If they don't appear, check that `SKILL.md` is exactly one level deep inside each skill folder (a common error is nesting it under an extra subfolder).

## How to use them

Skills auto-load based on their `description` field — you typically don't need to invoke them by name. Just describe what you want to do, and Claude Code consults the right skill.

### Full pipeline (the typical session)

A natural sequence in one Claude Code conversation:

```
> Do a literature survey on PINNs for the 1D wave equation, focusing on
  causal weighting and time-marching variants.

[literature-survey-pinn fires; produces pinn-knowledge-base.md]

> Using that knowledge base, draft a problem spec for a 1D wave equation
  PINN. Use the method of manufactured solutions.

[pinn-problem-spec fires; produces problem-spec.md]

> Now scaffold the implementation in PyTorch with torch.func.

[pinn-scaffold fires; produces pinn_run/ with all five Python files]

> python pinn_run/run.py

[training runs locally, outputs land in pinn_run/outputs/]

> Analyze the run and produce a report.

[pinn-analysis-report fires; produces analysis-report.md]
```

### Invoking a single skill

Each skill can also be invoked in isolation. For instance, if you already have a `problem-spec.md` from a colleague and just want code:

```
> Read problem-spec.md and scaffold the PyTorch implementation.
```

The `pinn-scaffold` skill will trigger. The other skills don't run unless their inputs and triggers match.

### Explicit invocation

If you want to force a specific skill (useful when triggering is ambiguous):

```
> /pinn-problem-spec
```

This works on most Claude Code versions; if it doesn't, just include the skill name in your prompt naturally ("use the pinn-problem-spec skill to ...").

## Recommended Claude Code settings for this workflow

In your Claude Code settings:

- **Code execution**: enabled. The scaffold skill smoke-tests generated code, and the analysis skill loads CSVs with pandas.
- **File creation**: enabled. Every skill produces markdown artifacts.
- **Auto-approve `python` and `pip`**: helpful for smooth training runs. Be more conservative with shell commands that touch your filesystem outside the project.

## Running on Apple Silicon (M-series)

PyTorch's MPS backend works for these skills. The scaffold skill is aware of MPS and will fall back to CPU if `torch.func` ops fail on a given PyTorch version. For the small demo problems (2D Poisson, 1D wave), CPU is often faster than MPS due to kernel-launch overhead — don't be surprised.

Before your first run, verify your environment:

```bash
python -c "import torch; print('mps:', torch.backends.mps.is_available(), 'version:', torch.__version__)"
python -c "from torch.func import grad, vmap, hessian; print('torch.func ok')"
```

If both print successfully, you're set.

## Tips for the workshop / live demo

1. **Pre-run the survey skill** the night before and ship the resulting `pinn-knowledge-base.md` with the repo. The survey is the slowest step (multiple web searches); for live demos, you want it cached.

2. **Use a fresh working directory** for each demo. The skills read/write files in the current directory, and stale artifacts can confuse the chain.

3. **Show the SKILL.md once** at the start. Open `skills/pinn-problem-spec/SKILL.md` in the editor and walk through the structure — it demystifies what's actually happening.

4. **Don't auto-approve everything.** Audiences appreciate seeing the agent ask before executing. Keep "Request Review" on for shell commands.

5. **Watch the artifact handoff.** When `problem-spec.md` is written, open it in the editor side-by-side with the chat. Same for `analysis-report.md`. This is the most important pedagogical moment — it makes the chain concrete.

## Updating the skills

When you modify a `SKILL.md`, Claude Code picks up the change in the next session. No restart needed for project-scoped skills. For personal skills, restart Claude Code to be safe.

## Troubleshooting

- **Skill doesn't trigger when I expect it to.** Read the skill's `description` field — that's what Claude Code uses to decide. If your phrasing doesn't match any of the trigger phrases, either rephrase or invoke the skill by name.
- **`/skills` doesn't list the skill.** Check the path: `.claude/skills/pinn-scaffold/SKILL.md` should exist *exactly* — no extra nesting.
- **`torch.func` errors on MPS.** Try CPU: `export PYTORCH_ENABLE_MPS_FALLBACK=1` or modify the scaffold's `run.py` to use `device='cpu'`.
- **Training diverges.** Almost always loss-weight imbalance. The analysis skill will tell you which term dominates; raise the under-weighted term by 10× and re-run.
