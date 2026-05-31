"""
train.py — Training loop for 2D Heat PINN DoE (Problems A and B).

All 16 variants (A1-A8 steady, B1-B8 unsteady) share this module.
Run via run.py --variant <label>.

Logging: training_log.csv with columns:
  step, total_loss, pde_loss, bc_loss, ic_loss, lr, wall_s
"""

import csv
import math
import time
from pathlib import Path

import torch
import torch.optim as optim

from model import build_model
from problem import (
    pde_residual_A, pde_residual_B,
    bc_loss_A, bc_loss_B, ic_loss_B,
    apply_hard_A, apply_hard_B,
    sample_interior_A, sample_boundary_A,
    sample_interior_B, sample_boundary_B, sample_ic_B,
    ref_A, ref_B,
    T_MAX, T_MIN,
)

# ── Variant config table ──────────────────────────────────────────────────────
#  label : (problem, hard, lbfgs/causal, fourier)
VARIANTS = {
    # Problem A (steady)
    "soft-adam":       ("A", False, False, False),
    "soft-lbfgs":      ("A", False, True,  False),
    "hard-adam":       ("A", True,  False, False),
    "hard-lbfgs":      ("A", True,  True,  False),
    "soft-adam-ff":    ("A", False, False, True),
    "soft-lbfgs-ff":   ("A", False, True,  True),
    "hard-adam-ff":    ("A", True,  False, True),
    "hard-lbfgs-ff":   ("A", True,  True,  True),
    # Problem B (unsteady)
    "soft-std":        ("B", False, False, False),
    "soft-causal":     ("B", False, True,  False),
    "hard-std":        ("B", True,  False, False),
    "hard-causal":     ("B", True,  True,  False),
    "soft-std-ff":     ("B", False, False, True),
    "soft-causal-ff":  ("B", False, True,  True),
    "hard-std-ff":     ("B", True,  False, True),
    "hard-causal-ff":  ("B", True,  True,  True),
}

# Hyperparameters
ADAM_STEPS   = 20_000    # increased from 5k — 5k is insufficient for O(1e-3) ε_rel
LBFGS_STEPS  = 500
LR_ADAM      = 1e-3
LR_MIN       = 1e-5      # cosine decay floor
W_BC_SOFT    = 100.0
W_IC_SOFT    = 100.0
N_COL_A      = 4_096
N_BC_SIDE_A  = 100
N_COL_B      = 8_192
N_BC_SIDE_B  = 25     # per side × 20 time slices = 2000 per side total
N_BC_T_B     = 20
N_IC_B       = 1_024
LOG_EVERY    = 100
RESAMPLE_EVERY = 2_000   # re-draw collocation points periodically
CAUSAL_EPS   = 100.0
N_TIME_SLABS = 50     # causal weighting: number of time slabs


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Wrapped model: applies hard-constraint ansatz ────────────────────────────

class WrappedModelA(torch.nn.Module):
    """Optionally applies x(1-x)y(1-y)·NN transform."""
    def __init__(self, base_model, hard: bool):
        super().__init__()
        self.base = base_model
        self.hard = hard

    def forward(self, xy):
        out = self.base(xy)
        if self.hard:
            out = apply_hard_A(out, xy)
        return out


class WrappedModelB(torch.nn.Module):
    """Optionally applies sin(πx)sin(πy) + t·x(1-x)y(1-y)·NN transform."""
    def __init__(self, base_model, hard: bool):
        super().__init__()
        self.base = base_model
        self.hard = hard

    def forward(self, xyt):
        out = self.base(xyt)
        if self.hard:
            out = apply_hard_B(out, xyt)
        return out


# ── Causal weight computation ────────────────────────────────────────────────

def causal_weights(xyt: torch.Tensor, pde_res: torch.Tensor,
                   n_slabs: int, eps: float) -> torch.Tensor:
    """
    Wang, Sankaran, Perdikaris 2022 causal weighting.
    Sort collocation points by t into n_slabs; weight slab k by
      w_k = exp(-eps * sum_{j<k} mean_pde_loss_j)
    Returns per-point weights (same shape as pde_res).
    """
    t_vals = xyt[:, 2]
    t_min, t_max = T_MIN, T_MAX
    slab_size = (t_max - t_min) / n_slabs
    weights = torch.ones_like(pde_res)
    cum_loss = torch.tensor(0.0, device=xyt.device)
    for k in range(n_slabs):
        t_lo = t_min + k * slab_size
        t_hi = t_min + (k + 1) * slab_size
        mask = (t_vals >= t_lo) & (t_vals < t_hi)
        if mask.sum() == 0:
            continue
        w_k = torch.exp(-eps * cum_loss)
        weights[mask] = w_k
        cum_loss = cum_loss + pde_res[mask].mean().detach()
    return weights


# ── Training function ─────────────────────────────────────────────────────────

def train_variant(label: str, out_dir: Path):
    """Train a single DoE variant and save all outputs to out_dir."""
    prob, hard, extra, fourier = VARIANTS[label]
    device = get_device()
    print(f"\n{'='*60}")
    print(f"Variant: {label}  |  Problem: {prob}  |  Hard: {hard}  |  "
          f"Extra: {'lbfgs' if prob=='A' and extra else 'causal' if prob=='B' and extra else 'none'}  |  "
          f"Fourier: {fourier}  |  Device: {device}")
    print(f"{'='*60}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Build model ────────────────────────────────────────────────────────
    d_in = 2 if prob == "A" else 3
    base = build_model(fourier=fourier, d_in=d_in, hidden=64, n_layers=4)
    if prob == "A":
        model = WrappedModelA(base, hard).to(device)
    else:
        model = WrappedModelB(base, hard).to(device)

    # ── Sample points ──────────────────────────────────────────────────────
    if prob == "A":
        xy_col  = sample_interior_A(N_COL_A, device)
        xy_bc   = sample_boundary_A(N_BC_SIDE_A, device)
    else:
        xyt_col = sample_interior_B(N_COL_B, device)
        xyt_bc  = sample_boundary_B(N_BC_SIDE_B, N_BC_T_B, device)
        xy_ic   = sample_ic_B(N_IC_B, device)

    # ── Optimiser + cosine LR schedule ────────────────────────────────────
    opt_adam   = optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
        opt_adam, T_max=ADAM_STEPS, eta_min=LR_MIN)

    # ── CSV logger ─────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    csv_file = open(log_path, "w", newline="")
    writer   = csv.DictWriter(csv_file, fieldnames=[
        "step", "total_loss", "pde_loss", "bc_loss", "ic_loss", "lr", "wall_s"])
    writer.writeheader()

    t_start = time.time()

    # ── Adam loop ──────────────────────────────────────────────────────────
    for step in range(1, ADAM_STEPS + 1):

        # Periodic collocation resampling (keeps network from memorising a fixed set)
        if step > 1 and (step - 1) % RESAMPLE_EVERY == 0:
            if prob == "A":
                xy_col = sample_interior_A(N_COL_A, device)
                xy_bc  = sample_boundary_A(N_BC_SIDE_A, device)
            else:
                xyt_col = sample_interior_B(N_COL_B, device)
                xyt_bc  = sample_boundary_B(N_BC_SIDE_B, N_BC_T_B, device)
                xy_ic   = sample_ic_B(N_IC_B, device)

        opt_adam.zero_grad()

        if prob == "A":
            pde_res = pde_residual_A(model, xy_col)
            pde_l   = (pde_res ** 2).mean()
            bc_l    = bc_loss_A(model, xy_bc) if not hard else torch.tensor(0.0, device=device)
            ic_l    = torch.tensor(0.0, device=device)
            loss    = pde_l + (W_BC_SOFT * bc_l if not hard else 0.0)
        else:
            pde_res = pde_residual_B(model, xyt_col)
            if extra:  # causal weighting
                w = causal_weights(xyt_col, pde_res.detach() ** 2,
                                   N_TIME_SLABS, CAUSAL_EPS)
                pde_l = (w * pde_res ** 2).mean()
            else:
                pde_l = (pde_res ** 2).mean()
            bc_l = bc_loss_B(model, xyt_bc) if not hard else torch.tensor(0.0, device=device)
            ic_l = ic_loss_B(model, xy_ic)  if not hard else torch.tensor(0.0, device=device)
            loss = pde_l + (W_BC_SOFT * bc_l + W_IC_SOFT * ic_l if not hard else 0.0)

        loss.backward()
        opt_adam.step()
        scheduler.step()

        if step % LOG_EVERY == 0:
            current_lr = scheduler.get_last_lr()[0]
            row = {
                "step":       step,
                "total_loss": loss.item(),
                "pde_loss":   pde_l.item(),
                "bc_loss":    bc_l.item(),
                "ic_loss":    ic_l.item(),
                "lr":         current_lr,
                "wall_s":     round(time.time() - t_start, 2),
            }
            writer.writerow(row)
            csv_file.flush()
            if step % 2000 == 0:
                print(f"  step {step:5d}  total={loss.item():.3e}  "
                      f"pde={pde_l.item():.3e}  bc={bc_l.item():.3e}  "
                      f"ic={ic_l.item():.3e}  lr={current_lr:.2e}")

    # ── Optional L-BFGS polishing (Problem A only) ─────────────────────────
    if prob == "A" and extra:
        print(f"  Running L-BFGS ({LBFGS_STEPS} max iterations) …")
        opt_lbfgs = optim.LBFGS(model.parameters(), max_iter=LBFGS_STEPS,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=50, line_search_fn="strong_wolfe")
        lbfgs_step = [ADAM_STEPS]

        def closure():
            opt_lbfgs.zero_grad()
            # torch.enable_grad() overrides the no_grad context that L-BFGS's
            # strong Wolfe line search uses for function-value-only evaluations.
            # Without it, model(xy_g) has no grad_fn and _ag() crashes.
            with torch.enable_grad():
                pde_r  = pde_residual_A(model, xy_col)
                pde_l_ = (pde_r ** 2).mean()
                bc_l_  = (bc_loss_A(model, xy_bc) if not hard
                          else torch.tensor(0.0, device=device))
                loss_  = pde_l_ + (W_BC_SOFT * bc_l_ if not hard else 0.0)
                loss_.backward()
            return loss_.detach()

        opt_lbfgs.step(closure)

        # log final L-BFGS state (no_grad incompatible with AD residual — use enable_grad)
        with torch.enable_grad():
            pde_r  = pde_residual_A(model, xy_col)
            pde_l_ = (pde_r ** 2).mean()
            bc_l_  = (bc_loss_A(model, xy_bc) if not hard
                      else torch.tensor(0.0, device=device))
            total_ = pde_l_ + (W_BC_SOFT * bc_l_ if not hard else 0.0)
        writer.writerow({
            "step":       ADAM_STEPS + LBFGS_STEPS,
            "total_loss": total_.item(),
            "pde_loss":   pde_l_.item(),
            "bc_loss":    bc_l_.item(),
            "ic_loss":    0.0,
            "lr":         0.0,
            "wall_s":     round(time.time() - t_start, 2),
        })
        print(f"  L-BFGS done  total={total_.item():.3e}  pde={pde_l_.item():.3e}")

    csv_file.close()

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = out_dir / "checkpoint.pt"
    torch.save({"model_state": model.state_dict(),
                "variant": label,
                "hard": hard,
                "fourier": fourier,
                "problem": prob,
                "extra": extra}, ckpt_path)

    wall = round(time.time() - t_start, 1)
    print(f"  Saved → {out_dir}  [{wall}s]")
    return model, prob, hard
