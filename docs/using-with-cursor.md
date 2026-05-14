# Using DhyutimaanPI Skills with Cursor

Cursor supports the SKILL.md format via its agent rules and skills features. The setup is straightforward and lets you use these skills with Cursor's built-in Claude integration (or any other model Cursor supports).

## Installation

Cursor looks for skills in `.cursor/skills/<name>/SKILL.md` (project-scoped). Copy or symlink the skills there:

```bash
cd /path/to/DhyutimaanPI
mkdir -p .cursor/skills
cp -r skills/literature-survey-pinn .cursor/skills/
cp -r skills/pinn-problem-spec      .cursor/skills/
cp -r skills/pinn-scaffold          .cursor/skills/
cp -r skills/pinn-analysis-report   .cursor/skills/
```

Commit `.cursor/skills/` so collaborators get them via git.

Cursor also reads `.claude/skills/` (cross-tool compatibility), so if you've already installed for Claude Code, you don't need a separate copy — Cursor will pick those up too.

## Alternative: use as Cursor Rules

If your Cursor version doesn't pick up skills automatically, you can install them as Cursor Rules. Cursor reads rules from `.cursor/rules/` (project) and supports the `AGENTS.md` cross-tool standard.

Add a top-level `AGENTS.md` to your project root:

```markdown
# AGENTS.md

This project includes four PINN research skills under `skills/`.
When the user requests work matching these triggers, follow the
corresponding `skills/<name>/SKILL.md`:

- "literature survey", "lit review", "state of the art on PINNs"
  → follow `skills/literature-survey-pinn/SKILL.md`

- "problem spec", "frame this PINN problem", "draft a problem statement"
  → follow `skills/pinn-problem-spec/SKILL.md`

- "scaffold the PINN", "implement", "write the code", "PyTorch version"
  → follow `skills/pinn-scaffold/SKILL.md`

- "analyze the run", "produce a report", "did it work"
  → follow `skills/pinn-analysis-report/SKILL.md`

Each SKILL.md specifies inputs, outputs, and workflow. The skills chain:
survey → problem-spec → scaffold → analysis. Artifacts pass between them
as files in the working directory.
```

`AGENTS.md` is read by Cursor, Antigravity, Claude Code, and several other tools — installing once gives broad coverage.

## How to use them

Open the Cursor chat panel (Cmd+L / Ctrl+L) and describe what you want:

```
> Do a literature survey on PINNs for the 2D Poisson equation. Use the
  literature-survey-pinn skill in this project.
```

Cursor's agent (using whichever model is selected — Claude Sonnet/Opus, GPT, Gemini) will read the SKILL.md and follow its workflow.

For the full chain, the prompts are the same as Claude Code; see [using-with-claude-code.md](using-with-claude-code.md#full-pipeline-the-typical-session).

## Cursor-specific tips

1. **Pin Claude as the model** for these skills. They were written and tested with Claude in mind; other models will work but may interpret the instructions differently.

2. **Use Agent mode, not chat mode**, for multi-step work. The scaffold skill writes multiple files; agent mode handles that flow better than chat.

3. **Add the skills folder to `.cursorignore` exemptions** if you have a broad ignore pattern. You want Cursor to read the SKILL.md files even though they're "documentation."

4. **Composer / multi-file edits** are useful when the scaffold skill produces 5 Python files at once.

## Running training

Cursor's terminal is the standard place to run training:

```bash
python pinn_run/run.py
```

Cursor will pick up the resulting `pinn_run/outputs/` files automatically when you ask the analysis skill to read them.

## Limitations

- Cursor's skill triggering is less robust than Claude Code's — you'll often need to mention the skill by name explicitly.
- Cursor doesn't have Anthropic's `code_execution` / `file_creation` sandbox; everything runs in your local terminal, which is fine but means you should commit your environment file (`requirements.txt` or `environment.yml`).
- Cursor's context window varies by model and plan; for long chains, the analysis skill may need the spec re-uploaded mid-conversation.

## Troubleshooting

- **Skill not invoked even when I mention it.** Make sure `.cursor/skills/<name>/SKILL.md` exists (or `.claude/skills/<name>/SKILL.md`). Run `ls -la .cursor/skills/` to verify.
- **Agent ignores SKILL.md and writes generic code.** Be explicit: "Read `.cursor/skills/pinn-scaffold/SKILL.md` and follow its workflow." Once the SKILL.md is in context, subsequent turns work normally.
- **Model differences.** GPT-based models tend to need more explicit step-by-step prompting than Claude. If outputs feel rushed, add "follow the workflow step by step" to your prompt.
