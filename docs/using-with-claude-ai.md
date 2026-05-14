# Using DhyutimaanPI Skills with Claude.ai

Claude.ai (the web app and mobile app) supports skills as `.skill` packages or by including their content directly in a Project. The workflow is slightly different from Claude Code — there's no filesystem to share with the model, but Projects, file uploads, and Claude's code execution environment give you the same end-to-end capability.

## Two installation paths

### Path A — Upload as Skills (Pro/Team/Enterprise plans)

Claude.ai supports installable skills on Pro and above. The Skills feature must be enabled in your settings (Customize → Skills). For Enterprise, the owner must enable it organization-wide first.

1. Package each skill folder as a `.skill` file (a zipped folder with `SKILL.md` at the root). Anthropic ships a packager with the `skill-creator` skill; alternatively just zip each folder:

   ```bash
   cd skills
   zip -r literature-survey-pinn.skill literature-survey-pinn
   zip -r pinn-problem-spec.skill      pinn-problem-spec
   zip -r pinn-scaffold.skill          pinn-scaffold
   zip -r pinn-analysis-report.skill   pinn-analysis-report
   ```

2. In Claude.ai, go to **Customize → Skills → Upload skill** and upload each `.skill` file. Toggle each one on.

3. (Optional, for orgs) An admin can publish them to the organization directory so everyone gets them.

Once installed, the skills auto-trigger based on their `description` field — same as in Claude Code.

### Path B — Use a Project (works on all plans)

If you can't install skills (e.g., on Free) or want a quick try-out, use a Claude.ai Project:

1. Create a new Project in claude.ai.
2. Add the four `SKILL.md` files to the Project's knowledge base (drag and drop, or paste).
3. In the Project's custom instructions, add:

   > You have access to four PINN skills in this project's knowledge:
   > literature-survey-pinn, pinn-problem-spec, pinn-scaffold, pinn-analysis-report.
   > Treat each SKILL.md as instructions to follow when its description matches.
   > When the user invokes a skill (explicitly or implicitly), read that SKILL.md
   > first and follow it exactly. Outputs (problem-spec.md, pinn_run/, etc.)
   > should be produced as files using Claude's code execution and file creation
   > capabilities.

4. Start chatting in the project. The skills aren't "installed" the way Path A installs them, but Claude will consult them as in-context instructions.

## How to use them

The interaction pattern is the same regardless of path:

```
You: Do a literature survey on PINNs for the 1D wave equation,
     focusing on causal weighting.

Claude: [consults literature-survey-pinn; uses web search; produces
         pinn-knowledge-base.md as a file]

You: Draft a problem spec for a 1D wave PINN using MMS.

Claude: [consults pinn-problem-spec; writes problem-spec.md]

You: Now scaffold the implementation in PyTorch.

Claude: [consults pinn-scaffold; writes pinn_run/*.py files,
         possibly runs a smoke test in the code execution sandbox]

You: [download files, run training locally on your laptop]
     [upload training_log.csv and verification.json back to Claude.ai]

You: Analyze the run.

Claude: [consults pinn-analysis-report; reads the uploaded files;
         produces analysis-report.md]
```

## Running training: locally vs. in Claude.ai

Claude.ai has a code execution sandbox, but it's ephemeral and time-limited. For real PINN training, two options:

- **Train locally** (recommended). The scaffold skill produces a `pinn_run/` directory you can download. Run `python pinn_run/run.py` on your laptop, then upload the contents of `pinn_run/outputs/` back to Claude.ai for analysis.

- **Train in the sandbox** (for small demos only). The 2D Poisson MMS converges in well under a minute and fits comfortably in the sandbox. The wave equation may run slow; expect ~1-2 min for a meaningful demo.

For workshops and the 1-hour session, **local training is the better choice** — the audience sees real timing and you avoid sandbox quirks.

## Tips for Claude.ai workflows

1. **Use Projects for ongoing work.** A single project per PDE/study keeps all the artifacts (knowledge base, spec, training logs, reports) in one place.

2. **Upload `problem-spec.md` explicitly when starting a new chat.** Even within a Project, fresh chats start without prior artifacts in context; uploading the spec ensures the scaffold or analysis skill has what it needs.

3. **Skills + Projects together.** If you have Pro+, install the skills (Path A) *and* use Projects for organization. The skills handle the workflow logic, the project handles the artifact history.

4. **Toggle skills off when not needed.** If you're using the project for unrelated work, toggle the PINN skills off so they don't interfere with other requests.

5. **Web search and code execution must be enabled** for the survey and scaffold skills respectively. Check Settings → Capabilities.

## Limitations vs. Claude Code

| Capability | Claude Code | Claude.ai |
|---|---|---|
| Skills auto-load from `.claude/skills/` | yes | n/a |
| Skills auto-load when uploaded | n/a | yes (Pro+) |
| Persistent filesystem | yes (your machine) | no (sandbox is ephemeral) |
| Multi-skill chain in one session | seamless | works, but uploading artifacts between steps is more manual |
| Local GPU/MPS for training | yes | no (sandbox is CPU-only) |
| Inspecting generated code in an editor | yes (your IDE) | download to view |

If you're doing serious research with this workflow, Claude Code is the better home. Claude.ai is excellent for the survey and analysis stages and for first-pass framing.

## Troubleshooting

- **Skill doesn't appear after upload.** Make sure the `.skill` file has `SKILL.md` at the *root* of the zip, not nested under a folder. Re-zip from inside the skill folder if needed.
- **Project knowledge doesn't seem to be consulted.** Be explicit in the first turn: "Consult `pinn-problem-spec/SKILL.md` from this project's knowledge and follow its workflow." Once Claude has read it, subsequent turns pick up naturally.
- **File creation produces nothing.** Code Execution and File Creation must both be enabled. Check Settings.
