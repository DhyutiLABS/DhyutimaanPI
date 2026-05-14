---
name: pinn-scaffold
description: Generate verifiable Physics-Informed Neural Network (PINN) implementation code from a problem-spec.md file. Use this skill whenever the user wants to implement, scaffold, or write code for a PINN, asks to "build", "code up", "implement", or "set up" a PINN, or hands over a problem spec and asks for an implementation. Also trigger when the user asks for a PyTorch PINN, a `torch.func` residual, a collocation-point sampler, a PINN training loop, or a DeepXDE version of a problem. Produces a self-contained Python module with the network, residual, sampler, training loop, and a verification harness that compares against the spec's reference solution.
---

# PINN Scaffold

This skill generates working, verifiable PINN code from a problem spec. The default backend is **PyTorch with `torch.func`** because it keeps the physics-loss computation transparent — the residual is a few lines of plain Python, not hidden inside a library. A DeepXDE alternate path is available when the user asks for a more abstracted implementation.

The generated code must be runnable end-to-end and must include a verification harness that compares training output against the reference solution defined in the spec.

## When to use this skill

Trigger this skill when:

- The user has a `problem-spec.md` (or a clear equivalent) and wants the implementation
- The user asks to "scaffold", "implement", "build", or "code up" a PINN
- The user wants to swap backends on an existing problem (e.g., "give me the DeepXDE version")
- The user wants to extend an existing PINN with a specific architectural variant (Fourier features, SIREN, causal weighting)

Do not trigger this skill for problem framing (use `pinn-problem-spec`) or for analysis of already-trained models (use `pinn-analysis-report`).

## Backend selection

There are two supported backends. Choose by reading the spec's section 5 ("Architecture and training plan") and by the user's stated preference:

- **`torch_func`** (default) — PyTorch with `torch.func.grad`, `torch.func.vmap`, `torch.func.jacrev`. Residual is hand-written and visible. Best for teaching and for unusual operators.
- **`deepxde`** (alternate) — DeepXDE's high-level API. Better when the user wants minimal code, the problem fits a standard template, or they're comparing to published DeepXDE results.

If the user has not specified, default to `torch_func` and tell them why. Offer to also emit a DeepXDE version as a second file for comparison.

## Output structure

Generate the following files in the working directory. Names matter — downstream skills (analysis, reporting) look for them.

```
pinn_run/
├── problem.py          # Problem definition: PDE residual, BCs, ICs, reference solution
├── model.py            # Network architecture (MLP, optionally with Fourier features)
├── train.py            # Training loop with logging
├── verify.py           # Verification harness: error vs. reference, residual fields
├── run.py              # Top-level entry point: trains and verifies
└── outputs/            # Created at runtime; holds checkpoints, logs, plots
    ├── training_log.csv
    ├── checkpoint.pt
    └── verification.json
```

Each file should be self-contained and runnable. Imports at the top, no hidden dependencies.

## Workflow

### Step 1: Read the spec

Read `problem-spec.md` from the working directory. If it does not exist, ask the user for it before generating any code — do not hallucinate a spec. Extract:

- The PDE residual (section 2)
- Domain and BCs/ICs (section 3)
- Reference solution (section 4)
- Architecture and training plan (section 5)
- Hypotheses to verify (section 6) — these become test cases in `verify.py`
- Success criteria (section 7) — this becomes the pass/fail threshold

### Step 2: Generate `problem.py`

This file is the math, isolated from the ML. It should contain:

- A function `pde_residual(model, x)` that takes the network and a batch of input points, and returns the residual values. Use `torch.func.grad`/`jacrev` for the derivatives, not `torch.autograd.grad`, because `torch.func` is composable and avoids the "create_graph" footguns.
- A function `boundary_loss(model, x_b, u_b)` for each BC type
- A function `initial_loss(model, x_0, u_0)` if the problem is time-dependent
- A function `reference_solution(x)` returning the analytic or MMS reference
- The manufactured source term (if MMS), evaluated as a function of inputs
- Domain bounds as module-level constants

For the 2D Poisson MMS case ($u = \sin\pi x \sin\pi y$, $f = 2\pi^2 \sin\pi x \sin\pi y$), the residual is:

```python
import torch
from torch.func import grad, vmap

def laplacian(model):
    def u(x):
        return model(x).squeeze()
    def lap_at(x):
        # Hessian diagonal sum
        H = torch.func.hessian(u)(x)
        return torch.diagonal(H).sum()
    return vmap(lap_at)

def pde_residual(model, x, f_source):
    return laplacian(model)(x) + f_source(x)  # -Δu = f → residual = Δu + f
```

This pattern — wrap the model, take derivatives with `torch.func`, vmap over the batch — is the core idiom. Use it for every PDE.

### Step 3: Generate `model.py`

Default network: MLP with configurable depth and width, tanh activation, optional Fourier feature input encoding. The model must accept a tensor of shape `(batch, d_in)` and return `(batch, d_out)`. Keep it simple — a custom `nn.Module` subclass with a `forward` method.

If the spec calls for Fourier features, include them as a fixed (non-trainable) projection at the input. Document the frequencies inline.

### Step 4: Generate `train.py`

The training loop must:

- Sample collocation, boundary, and (if applicable) initial points. Default: 10000 collocation, 400 boundary, 200 initial.
- Compute the weighted loss (residual + BC + IC terms, weights from the spec).
- Log loss components separately to `outputs/training_log.csv` every N iterations — *separate columns* for each loss term. The analysis skill depends on this granularity.
- Save a checkpoint to `outputs/checkpoint.pt` at the end.
- Use Adam first; if the spec calls for L-BFGS post-training, run it after Adam converges or after a fixed iteration count.

Logging schema (CSV header):
```
iteration,total_loss,pde_loss,bc_loss,ic_loss,lr,time_elapsed_s
```

### Step 5: Generate `verify.py`

This is the verification harness. It must:

- Load the checkpoint.
- Evaluate the network on a dense grid (default: 200×200 for 2D, 1000 for 1D).
- Compute the relative L2 error against `reference_solution`.
- Compute the residual field on the grid (helps spot under-resolved regions).
- Compare against each hypothesis in the spec (section 6) and write a JSON verdict.
- Save plots to `outputs/`: predicted solution, reference, pointwise error, residual field.

Verification output schema (`outputs/verification.json`):

```json
{
  "relative_l2_error": 0.0007,
  "max_pointwise_error": 0.003,
  "max_residual": 0.012,
  "success_criteria_met": true,
  "hypotheses": [
    {"id": 1, "statement": "...", "result": "confirmed", "evidence": "rel_l2 = 7e-4 < 1e-3"}
  ]
}
```

The analysis skill reads this file. Keep the keys stable.

### Step 6: Generate `run.py`

A thin entry point that runs `train.py` then `verify.py`. No surprises — just orchestration.

### Step 7: Smoke-test the generated code

Before handing back to the user, attempt to run `run.py` with a reduced iteration count (e.g., 200 Adam steps) just to confirm it doesn't crash. If you can't execute (no GPU/MPS available, missing deps), at minimum run a Python syntax check on each file. Report explicitly which checks ran and which were skipped.

## Backend-specific details

### `torch_func` backend

- Use `from torch.func import grad, vmap, jacrev, hessian` — these are stable in PyTorch 2.0+.
- Wrap scalar-output functions for `grad`, vector-output for `jacrev`.
- For second derivatives in 2D+, prefer `hessian` over nested `grad` — clearer and faster.
- For Apple Silicon (MPS): keep model on `mps` device but **fall back to CPU if `torch.func` ops fail on MPS for your PyTorch version**. Small PINNs often train faster on CPU anyway due to kernel-launch overhead.
- Avoid `torch.autograd.grad(..., create_graph=True)` patterns — they work but are harder to read and prone to graph-retention bugs.

### `deepxde` backend

- Use `dde.data.PDE` for steady problems, `dde.data.TimePDE` for time-dependent.
- Define the residual as a function `(x, y)` returning the residual; DeepXDE handles the derivatives via `dde.grad.jacobian` and `dde.grad.hessian`.
- Backend choice within DeepXDE: prefer the `pytorch` backend (set `DDE_BACKEND=pytorch` env var) for consistency.
- The reference solution is passed via the `solution=` argument to `dde.data.PDE`.

## Common failures and how to avoid them

- **Residual computed without `vmap`**: works but is slow. Always vmap over the batch dimension.
- **Boundary points sampled in the interior**: check that boundary samplers actually live on the boundary. Plot them once during generation.
- **Loss explodes in the first iterations**: usually means the network output is initialized far from the BC values. Adding a hard BC enforcement (output transformation) is the cleanest fix; mention this as an option to the user.
- **`torch.func` with batch norm / dropout**: these modules don't play well with `vmap`. Don't include them in the MLP.
- **Forgetting `model.eval()` in `verify.py`**: rarely matters for PINNs (no batch norm) but include it anyway as a habit.

## Worked example: 2D Poisson MMS

Given a spec that specifies:
- $-\nabla^2 u = f$ on $[0,1]^2$, $u=0$ on the boundary
- MMS with $u = \sin\pi x \sin\pi y$, $f = 2\pi^2 \sin\pi x \sin\pi y$
- MLP 4×64 tanh, Adam 5000 iters then L-BFGS, success at relative L2 < 1e-3

Generate the five files above with that math baked in. The residual in `problem.py` becomes 5-10 lines using `torch.func.hessian` + `vmap`. The training loop in `train.py` is ~60 lines. `verify.py` produces the L2 error and a 4-panel plot. Total scaffold: ~250 lines, all human-readable.

## Handoff

After the files are written and smoke-tested, tell the user:

> Scaffold is at `pinn_run/`. To train and verify: `python pinn_run/run.py`. Outputs land in `pinn_run/outputs/`. Once training finishes, the analysis skill will pick up `training_log.csv` and `verification.json` to produce the report.

Do not auto-invoke the analysis skill. Let the user run training first.

## Anti-patterns

- **Generating code that doesn't actually use the spec.** If you find yourself writing generic boilerplate, re-read the spec and bake in the specific PDE, BCs, and reference.
- **Skipping the verification harness.** A PINN with no error measurement is a confident hallucination machine. The harness is non-negotiable.
- **Writing code you didn't even syntax-check.** Always at minimum compile the files before declaring done.
- **Hiding the residual inside a library when the spec uses an unusual operator.** If the PDE has a non-standard term, `torch_func` keeps it visible; don't paper over it with DeepXDE.
