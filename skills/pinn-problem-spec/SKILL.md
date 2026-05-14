---
name: pinn-problem-spec
description: Turn an informal description of a physics problem (PDE, BCs, domain, goals) into a structured, falsifiable problem specification that downstream scaffolding and analysis agents can act on. Use this skill whenever the user wants to set up a Physics-Informed Neural Network (PINN) problem, asks to "frame", "specify", "draft", or "set up" a PINN problem, mentions a governing equation they want to solve with a neural network, or hands over literature-survey notes and asks "what should we try?". Also trigger when the user says things like "I want to solve Burgers' / Poisson / wave / Navier-Stokes with a PINN" or asks for hypotheses, ablations, or a problem statement for a computational mechanics study. The output is a markdown problem-spec document that the pinn-scaffold skill consumes as input.
---

# PINN Problem Spec

This skill turns a fuzzy "I want to solve X with a PINN" request into a concrete, machine-readable problem spec. The spec is the contract: every downstream agent (scaffolding, training, analysis, reporting) reads it and acts on its fields. Vague specs produce vague code and useless analyses, so the work here is to *force* precision early — even at the cost of asking clarifying questions or stating assumptions explicitly.

## When to use this skill

Trigger this skill when:

- The user wants to set up a PINN problem, even casually ("let's try a PINN on the heat equation")
- The user has literature-survey notes and asks what to try next
- The user is unclear on a PDE setup and wants help framing it
- A previous spec needs revision (new BCs, different domain, new hypothesis)

Do *not* trigger this skill for pure literature questions, pure implementation questions on an already-specified problem, or analysis of completed runs. Those are different skills.

## Output: the problem spec document

Always produce a single markdown file named `problem-spec.md` with the following exact section structure. Downstream skills parse these headers, so do not rename or reorder them.

```markdown
# PINN Problem Spec: <short name>

## 1. Problem statement
One paragraph in plain language. What are we solving and why.

## 2. Governing equations
LaTeX-rendered PDE(s), with every symbol defined. Use $$ for display math.
Include the non-dimensional form if relevant.

## 3. Domain and boundary/initial conditions
- Spatial domain: ...
- Temporal domain (if applicable): ...
- BCs: list each boundary and the condition (Dirichlet, Neumann, periodic, Robin)
- ICs: list each initial condition

## 4. Reference / ground truth
How we will know the PINN is right. One of:
- Analytic solution (give the closed form)
- Method of manufactured solutions (give f such that a chosen u solves the PDE)
- High-fidelity numerical reference (FEM, FD, spectral) — specify the method and resolution
- None available (flag this — it limits what the analysis agent can do)

## 5. Architecture and training plan
- Network: MLP / Fourier features / SIREN / other; depth, width, activation
- Inputs and outputs: explicit list
- Sampling: number of collocation, boundary, initial points; resampling strategy
- Loss: list each term and its weight (or weighting scheme: fixed, NTK-based, causal, GradNorm, etc.)
- Optimizer: Adam / L-BFGS / two-stage; learning rate; number of iterations
- Hardware target: CPU / MPS / CUDA

## 6. Hypotheses (falsifiable)
2–4 hypotheses, each phrased so it can be confirmed or refuted by a specific measurement.
Bad: "the PINN will work well."
Good: "Causal weighting reduces relative L2 error at t > 0.7 by at least 30% vs. uniform weighting on this 1D wave problem."

## 7. Success criteria
Specific numbers. E.g., "relative L2 error against the analytic solution below 1e-3 on a 200x200 evaluation grid."

## 8. Known failure modes to watch for
PINN-specific things the analysis agent should explicitly check. Examples:
- Spectral bias (high-frequency components underfit)
- Loss imbalance (one term dominates)
- Causality violation (later times trained before earlier ones converge)
- BC/IC drift (residual low in interior, high at boundaries)

## 9. Out of scope
What we are explicitly NOT doing in this iteration. Keeps follow-on questions grounded.
```

## Workflow

Follow these steps in order. Do not skip ahead.

### Step 1: Establish the problem

Read the conversation for the PDE, domain, and goal. If any of the following are missing, ask the user — but ask in a single batched question, not one at a time:

- The PDE itself (or the physical system it comes from)
- The spatial and temporal domain
- The boundary and initial conditions
- Whether they have a reference solution or want one constructed via manufactured solutions

If the user is exploring and doesn't know yet ("I want to try a PINN on something simple"), suggest 2–3 canonical options with their tradeoffs and let them pick:

- **2D Poisson with manufactured solution** — fastest, cleanest, no time dimension. Best for first contact.
- **1D viscous Burgers'** — has shock formation, shows spectral bias, has known analytic/reference solutions.
- **1D wave equation** — exposes causality issues, good for comparing vanilla vs. causal PINN.
- **2D steady heat with source** — like Poisson but with a non-trivial forcing term.

### Step 2: Choose the reference strategy

This is the most important decision. Without a reference, the analysis skill cannot tell training success from confident failure. Prefer in this order:

1. **Analytic solution** if one exists for the chosen BCs.
2. **Method of manufactured solutions (MMS)** — pick a smooth function $u(x,t)$, plug into the PDE to derive the source term $f$, use that $f$ in training. This always works and gives a perfect reference.
3. **High-fidelity numerical reference** if neither is available.
4. **No reference** — only acceptable if the user explicitly accepts that the analysis will be qualitative.

If you construct an MMS, write out the manufactured $u$ and the derived $f$ in section 4 of the spec, fully evaluated. Downstream code will use these directly.

### Step 3: Propose architecture and training plan

Default starting points (the user can override):

- **Poisson / steady problems**: MLP, 4 layers × 64 units, tanh activation, ~10k collocation points, ~400 boundary points, Adam at 1e-3 for 5k iterations followed by L-BFGS to convergence.
- **Time-dependent problems (Burgers', wave)**: same MLP defaults, plus initial-condition points and a causal or time-marching weighting scheme. Note: for the wave equation, vanilla weighting often fails — flag this in section 8.

Loss weighting: start with equal weights and flag that they should be re-examined in analysis. If the user has read about NTK or causal weighting, surface it as a hypothesis (section 6) rather than baking it into the default.

### Step 4: Draft falsifiable hypotheses

This is where the skill earns its keep. Push back if the user gives a vague hypothesis. Each hypothesis must specify:

- The intervention or comparison (what changes)
- The metric (what we measure)
- The threshold or direction (what counts as confirmation)

Examples of well-formed hypotheses:

- "Adding Fourier features at frequencies {1, 2, 4, 8} reduces relative L2 error on the 2D Poisson MMS by at least 2× compared to a plain MLP of the same parameter count."
- "L-BFGS post-training reduces the boundary-loss term by an order of magnitude relative to Adam-only training, without increasing interior residual."
- "Causal weighting (epsilon=100) reduces relative L2 error at t > 0.7 on the 1D wave equation by at least 30% versus uniform weighting."

Examples that need rewriting:

- "The PINN will give good results" → not falsifiable, no threshold.
- "Adam works better than L-BFGS" → too broad, what metric, on what problem.

### Step 5: Specify known failure modes to watch for

This section feeds directly into the analysis skill. For the canonical problems, the typical entries:

- **2D Poisson**: BC drift if boundary loss is under-weighted; high-frequency underfitting if the manufactured solution has fine structure.
- **1D Burgers'**: shock front under-resolution; loss spike at the shock; instability at high Reynolds number.
- **1D wave**: error growth in time (causality violation); standing-wave-only fits when traveling waves are expected.
- **Any time-dependent problem**: IC loss low but trajectory diverges; this points to a need for IC weighting or time-marching.

### Step 6: Write out the spec and confirm

Write `problem-spec.md` to the working directory. Then summarize the spec to the user in 4–6 lines covering: problem, reference strategy, architecture choice, the headline hypothesis, and the success threshold. Ask if anything should change before scaffolding begins.

## A worked example

User says: *"Let's do a 2D Poisson with a PINN for the demo."*

The skill should:

1. Pick MMS as the reference (clean, no external dependencies). Choose a manufactured solution like $u(x,y) = \sin(\pi x)\sin(\pi y)$, derive $f = 2\pi^2 \sin(\pi x)\sin(\pi y)$.
2. Default architecture: MLP, 4×64, tanh.
3. Headline hypothesis: "The PINN achieves relative L2 error below 1e-3 within 5000 Adam steps on a 100×100 evaluation grid."
4. Failure modes to watch: BC drift, slow convergence if loss weights are skewed.
5. Out of scope: any non-trivial geometry, time dependence, or coefficient variation.

Write the spec, confirm with the user, hand off.

## Anti-patterns

- **Specifying without a reference.** If you can't measure error, you can't claim success. Push for MMS at minimum.
- **Over-specifying architecture before the problem is clear.** Architecture is a hypothesis, not a given.
- **Writing hypotheses that can't fail.** "The PINN will learn the solution" is not a hypothesis.
- **Treating loss weights as fixed.** They are almost always the wrong default; flag them for examination in analysis.
- **Mixing literature review into the spec.** That belongs in survey notes; the spec is operational.

## Handoff

When the spec is written and confirmed, tell the user: "Spec is at `problem-spec.md`. Ready for the scaffolding skill, which will generate the PyTorch code from this." Do not invoke the next skill automatically — give the user a chance to refine.
