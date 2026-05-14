---
name: pinn-analysis-report
description: Analyze the output of a trained PINN run and produce a structured report comparing results against the problem spec's hypotheses and success criteria. Use this skill whenever the user has finished training a PINN and wants to know "did it work?", asks to "analyze", "review", "evaluate", or "report on" PINN training results, has a `training_log.csv` / `verification.json` / `outputs/` folder from a previous PINN run, or wants ablation comparisons across PINN runs. Be adversarial: flag suspicious training curves, loss imbalances, under-resolved regions, and hypothesis confirmations that look like coincidence. Output is `analysis-report.md`.
---

# PINN Analysis & Report

This skill is the adversarial reader of PINN training output. Its job is not to celebrate that loss went down — it is to figure out whether the model actually learned the right thing, and to surface what *didn't* work as clearly as what did.

A good PINN analysis is suspicious by default. Low loss can mean the model converged on a trivial fit (e.g., a constant), boundary error can be hidden inside an aggregated total loss, and a "successful" hypothesis can be confirmed by a measurement that was never going to fail. This skill exists to catch those patterns before they end up in a paper or talk.

## When to use this skill

Trigger this skill when:

- A PINN training run has produced `training_log.csv`, `verification.json`, and/or plots
- The user asks "did the PINN work?" or "how did training go?"
- The user wants to compare runs (e.g., vanilla vs. causal weighting)
- The user wants a writeup of a completed experiment

Do not trigger this skill for active training questions (use `pinn-scaffold`), problem framing (`pinn-problem-spec`), or literature questions.

## Inputs

Expected inputs from a completed PINN run:

- `problem-spec.md` — the original spec; the report must compare results against its hypotheses and success criteria
- `pinn_run/outputs/training_log.csv` — per-iteration loss components and learning rate
- `pinn_run/outputs/verification.json` — final error metrics and hypothesis verdicts from the scaffold's verification harness
- `pinn_run/outputs/*.png` — any plots produced by `verify.py`

If any of these are missing, ask the user before fabricating analysis. Specifically: never produce numbers that aren't grounded in the input files.

## Output: `analysis-report.md`

Always produce a single markdown file with this exact structure:

```markdown
# PINN Analysis Report: <run name>

## 1. Headline
One sentence: did the run meet its success criteria, yes or no, with the key number.

## 2. Hypothesis verdicts
A table with one row per hypothesis from the spec. Columns:
- Hypothesis (short)
- Predicted threshold
- Measured value
- Verdict (confirmed / refuted / inconclusive)
- Confidence note (e.g., "single run", "MMS reference", "limited evaluation grid")

## 3. Training dynamics
- Final loss values (each component, separately)
- Loss balance over training (which term dominated, when)
- Convergence pattern (monotone, oscillatory, plateaus, spikes)
- Anything anomalous (NaN, exploding gradients, sudden drops)

## 4. Error analysis
- Relative L2 error against reference
- Max pointwise error and where it occurs (interior, near boundary, at a specific feature)
- Residual field: max, mean, where the residual is highest
- A note on whether the error is dominated by interior, boundary, or initial-condition violation

## 5. Failure-mode audit
Walk through each failure mode listed in the spec (section 8) and report whether it occurred.

## 6. What this run does NOT tell us
Honest limitations. E.g., "single seed, no statistical claim about variance", "evaluated only on smooth manufactured solution", "no test of generalization beyond the training domain".

## 7. Recommended next steps
2–4 concrete next experiments, each phrased as a hypothesis the next iteration could test.
```

## Workflow

### Step 1: Read all inputs

Load `problem-spec.md`, `training_log.csv`, and `verification.json`. View any plots in the outputs directory. If you have a code-execution environment, load the CSV with pandas and *actually compute* the summary statistics — don't eyeball the file.

If `training_log.csv` has columns the spec didn't promise (or is missing columns the spec did promise), note this as a data-quality issue at the top of the report.

### Step 2: Compute training-dynamics signals

From `training_log.csv`, compute:

- Final value of each loss column
- Ratio of final loss components (which dominates)
- Iteration at which loss reaches 50%, 10%, 1% of its initial value (per component)
- Whether any column shows non-monotone behavior in its final 20% of iterations
- Whether learning rate changed (e.g., L-BFGS phase started)

Flag if:

- One loss term is more than 100× larger than another at convergence — weighting is probably off
- The PDE loss converges but BC loss does not — the model is satisfying the equation away from where we care
- The loss plateaus early (first 10% of iterations) — likely a bad initialization or too small a network
- The loss has spikes that recur — possible learning rate too high or sampling issue

### Step 3: Compare to hypotheses

For each hypothesis in the spec:

1. Find the measurement it predicts (read carefully — many hypotheses specify a *specific region or condition*, e.g., "at t > 0.7").
2. Compute or extract that measurement from the verification output.
3. Apply the threshold from the hypothesis.
4. Mark confirmed, refuted, or inconclusive.

Mark "inconclusive" — not "confirmed" — when:

- The measurement is in the right direction but the threshold wasn't quantitative ("works better")
- A single seed isn't enough to support the claim
- The measurement was made on a region the hypothesis didn't specify

Be willing to refute hypotheses the user is rooting for. The whole point of a falsifiable hypothesis is that it can fail.

### Step 4: Failure-mode audit

For each failure mode in spec section 8, check the corresponding signal:

- **BC drift**: compute BC loss at the end of training. If it's > 10× the PDE loss, flag it.
- **Spectral bias / high-frequency underfitting**: look at the pointwise error plot — is the error concentrated in regions where the reference has high gradients?
- **Causality violation**: if time-dependent, compute error as a function of $t$. If error grows monotonically with time, flag it.
- **IC drift**: for time-dependent problems, compute error at $t=0$. If it's not the smallest, flag it.
- **Loss imbalance**: ratio of largest to smallest loss component at convergence > 100 → flag.

For each flagged issue, give one concrete remediation (e.g., "increase BC weight from 1.0 to 10.0", "add causal weighting with epsilon=100", "switch to Fourier features at frequencies {1,2,4}").

### Step 5: Write the limitations section

This is the section users skip and the section that matters most. Include at minimum:

- Single seed vs. multiple seeds (PINN training is noisy; a single run is anecdote, not evidence)
- Reference quality (MMS gives perfect ground truth; FD/FEM refs have their own error)
- Evaluation grid resolution (200×200 might miss fine structure)
- In-domain vs. out-of-domain (PINNs often fail outside the sampling region)
- What the success criterion actually tested (a low L2 error can coexist with bad behavior in a small region)

### Step 6: Propose next experiments

Each suggestion should be a hypothesis the next iteration's `problem-spec.md` could test. Don't recommend "try harder" — recommend specific interventions with specific predictions.

Good suggestion: "Switch to causal weighting (epsilon=100) and re-run; predict relative L2 error at t > 0.7 drops by at least 2×."

Bad suggestion: "Train for longer."

### Step 7: Write `analysis-report.md`

Write the file to the working directory. Keep the report tight — aim for 400–800 words plus the hypothesis table. The user should be able to read it in 2–3 minutes and walk away knowing exactly what happened and what to do next.

## Adversarial reading checklist

Before declaring the analysis complete, ask yourself:

- Is the headline supported by a number in the inputs, or am I inferring it?
- Did I check every hypothesis, including the ones that look obviously confirmed?
- Did I audit every failure mode the spec listed?
- Did I report at least one thing that didn't work, or that we can't conclude?
- Did I avoid praising the model? (Praise is the user's job; your job is to report.)

If any answer is no, fix it before handing over.

## Worked example output excerpt

Given a 2D Poisson MMS run with rel L2 = 7e-4, hypothesis was rel L2 < 1e-3:

> ## 1. Headline
> Run met its success criterion: relative L2 error 7.0e-4 against the MMS reference, below the 1e-3 threshold.
>
> ## 2. Hypothesis verdicts
> | Hypothesis | Predicted | Measured | Verdict | Note |
> |---|---|---|---|---|
> | rel L2 < 1e-3 in 5k Adam steps | < 1e-3 | 7.0e-4 | Confirmed | Single seed; MMS reference |
>
> ## 3. Training dynamics
> Final losses: PDE 2.1e-5, BC 1.4e-3. BC loss is ~70× the PDE loss, suggesting the BC weight is too low.
> Total loss decreased monotonically until iter ~3500, then plateaued. L-BFGS phase (iter 5000–) reduced PDE loss by another order of magnitude but did not reduce BC loss.
>
> ## 4. Error analysis
> ...

Note the report flags the BC/PDE imbalance even though the run "succeeded." That's the job.

## Comparing multiple runs

If the user provides multiple run directories (e.g., `pinn_run_vanilla/` and `pinn_run_causal/`), produce a *comparative* report with the same structure but add:

- A side-by-side table of final metrics
- A paired comparison for each hypothesis
- An explicit statement about whether the comparison is fair (same seed? same data? same network size?)

If the runs are not directly comparable (different network sizes, different sampling), say so. Don't average across incomparable runs.

## Anti-patterns

- **Reporting numbers you didn't load from a file.** Every quantitative claim must be traceable to an input.
- **Praising the model.** "The PINN learned the solution beautifully" is not analysis.
- **Confirming a hypothesis on a single seed without flagging the noise.** PINN training has run-to-run variance; one number is not a trend.
- **Ignoring failure modes the spec listed.** They're there for a reason — audit each one.
- **Recommending "train for longer" or "tune hyperparameters."** Those aren't experiments.
