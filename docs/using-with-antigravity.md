# Using DhyutimaanPI Skills with Google Antigravity

Antigravity is Google's agent-first IDE. It supports the SKILL.md format natively via its `.agents/skills/` directory, and it reads the cross-tool `AGENTS.md` standard. The PINN skills work in Antigravity with one small adjustment: Antigravity is built around the Agent Manager and multi-agent workflows, so the skills are best installed both as skills *and* registered in an `AGENTS.md` to define how they chain.

## Installation

Antigravity reads skills from `.agents/skills/<name>/SKILL.md` (project-scoped) and reads `AGENTS.md` for cross-tool agent instructions.

### Step 1 — Install the skills

```bash
cd /path/to/DhyutimaanPI
mkdir -p .agents/skills
cp -r skills/literature-survey-pinn .agents/skills/
cp -r skills/pinn-problem-spec      .agents/skills/
cp -r skills/pinn-scaffold          .agents/skills/
cp -r skills/pinn-analysis-report   .agents/skills/
```

Commit `.agents/` so collaborators get the same setup.

### Step 2 — Create AGENTS.md

Antigravity's strength is multi-agent orchestration. Define a small "team" in `AGENTS.md` at the repo root that maps the skills to agent roles:

```markdown
# AGENTS.md — PINN research team

This project conducts physics-informed neural network (PINN) studies
using a chain of four specialized agents, each backed by a SKILL.md
file under `.agents/skills/`.

## Agents

### Surveyor
- Skill: `.agents/skills/literature-survey-pinn/SKILL.md`
- Role: Conducts literature surveys on PINN topics
- Input: A topic, PDE, or research question
- Output: `pinn-knowledge-base.md`

### Framer
- Skill: `.agents/skills/pinn-problem-spec/SKILL.md`
- Role: Turns an informal problem into a falsifiable spec
- Input: A research question (and optionally `pinn-knowledge-base.md`)
- Output: `problem-spec.md`

### Implementer
- Skill: `.agents/skills/pinn-scaffold/SKILL.md`
- Role: Generates PyTorch implementation from a spec
- Input: `problem-spec.md`
- Output: `pinn_run/` (Python module + verification harness)

### Analyst
- Skill: `.agents/skills/pinn-analysis-report/SKILL.md`
- Role: Adversarial analysis of completed training runs
- Input: `problem-spec.md`, `pinn_run/outputs/*`
- Output: `analysis-report.md`

## Workflow

The standard pipeline: Surveyor → Framer → Implementer → (user runs training) → Analyst.

Each agent should:
1. Read its SKILL.md before doing anything else.
2. Read its declared input files.
3. Produce its declared output files exactly (downstream agents depend on these names).
4. Pause for review before handing off (the Artifact Review Policy is set to "Asks for Review").

## Conventions

- All artifacts live in the project root or under `pinn_run/`.
- Filenames in the SKILL.md files are contracts — do not rename them.
- The Analyst is adversarial by design; do not soften its tone.
```

### Step 3 — Configure Antigravity settings

For PINN work, the recommended Antigravity settings:

- **Artifact Review Policy**: `Asks for Review` (you want to inspect each handoff)
- **Terminal Command Auto Execution**: `Request Review` initially. Add `python` and `pip` to the Allow List once you trust the workflow.
- **Enable Terminal Sandbox**: on
- **Model**: Claude Sonnet 4.5 if available (these skills were written with Claude in mind), otherwise Gemini 3 Pro

## How to use them

### Single-agent mode (Editor view)

In Editor mode, open the chat and prompt naturally:

```
Run the Surveyor agent: do a literature survey on PINNs for the 1D wave
equation with focus on causal weighting.
```

Antigravity will load the Surveyor's SKILL.md and follow it.

### Multi-agent mode (Agent Manager)

Antigravity's Agent Manager is the bigger win. Open it (Cmd/Ctrl+E to toggle from editor), create a new task, and prompt at the workflow level:

```
Execute the full PINN research pipeline for the 2D Poisson equation
with manufactured solution. Run Surveyor, then Framer, then Implementer.
Pause for my review at each handoff. After I run training locally and
upload the outputs, run the Analyst.
```

Antigravity will plan the task, spawn each agent in sequence, produce Artifacts for review at each step, and pause for your approval. This is the closest match to a "research pipeline" experience among current IDEs.

### Workflow as a slash command

Antigravity supports custom workflows (saved prompts). Save the multi-agent prompt above as a workflow:

1. Open Customizations → Workflows
2. New workflow named `pinn-pipeline`
3. Paste the prompt
4. Save

Now in any project with these skills installed, type `/pinn-pipeline` and Antigravity runs the whole chain.

## Antigravity-specific considerations

1. **Artifacts vs. files.** Antigravity treats agent outputs as Artifacts (reviewable, commentable, versioned). The SKILL.md files specify *file* outputs (`problem-spec.md`, etc.). Both work — Antigravity will produce the files and also surface them as Artifacts in the UI. You can leave Google-Docs-style comments on Artifacts; the agent will incorporate the comments on the next iteration.

2. **Plan mode vs. Fast mode.** For the Framer and Analyst (which require thinking), Plan mode is better. For the Implementer (mostly mechanical code generation), Fast mode is fine.

3. **Model selection per agent.** Antigravity lets you assign different models to different agents. A reasonable split:
   - Surveyor: Gemini 3 Pro (2M-token context helps when reading many papers)
   - Framer: Claude Sonnet 4.5 (good at structured reasoning)
   - Implementer: Claude Sonnet 4.5 or Gemini (either works for code)
   - Analyst: Claude Sonnet 4.5 (the adversarial-reading prompt was tuned for Claude)

4. **Browser tool.** The Surveyor uses web search. In Antigravity's settings, make sure the agent has browser access and the URL allowlist includes arxiv.org, semanticscholar.org, openreview.net, and github.com.

5. **MCP integration.** If you have arXiv or Semantic Scholar MCP servers configured, the Surveyor will prefer them over web search. Worth setting up if you do many surveys.

## Running training

Antigravity has a built-in terminal. After the Implementer produces `pinn_run/`, run:

```bash
python pinn_run/run.py
```

The Analyst can then read `pinn_run/outputs/*` directly from the workspace.

For larger runs, run training outside Antigravity (e.g., on a remote machine) and copy outputs back before invoking the Analyst.

## Troubleshooting

- **Agent doesn't read SKILL.md.** Make sure `.agents/skills/<name>/SKILL.md` exists at the right depth (one level under `.agents/skills/`). Also confirm Antigravity is reading from `AGENTS.md` — check Customizations → Rules.
- **Multi-agent task gets stuck mid-chain.** Check the Artifact Review Policy. If it's `Asks for Review`, the agent is waiting for your approval — open the Agent Manager and approve.
- **Model produces generic code, ignoring the SKILL.md.** Switch to Plan mode and re-prompt: "First read the Implementer's SKILL.md, then write the implementation plan, then write code."
- **Browser access denied.** Add the relevant domains to the URL allowlist in Antigravity settings.

## VS Code (and other VS Code forks)

The same setup works for any VS Code-based agentic IDE that respects the SKILL.md / AGENTS.md conventions (Windsurf, Continue.dev, etc.). Copy skills to `.agents/skills/`, drop an `AGENTS.md` at the root, and the same prompts apply. Triggering reliability varies by tool — explicit invocation ("use the Framer agent") is the safest bet across IDEs.
