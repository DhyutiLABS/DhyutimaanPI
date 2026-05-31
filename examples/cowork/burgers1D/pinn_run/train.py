"""
train.py — Training loop for 1D Burgers PINN DoE.

DoE variants C1–C8 (ν=0.01/π) + viscosity stress tests V1, V3.
CSV log columns: step, total_loss, pde_loss, bc_loss, ic_loss, lr, wall_s
"""

import csv
import math
import time
from pathlib import Path

import torch
import torch.optim as optim

from model import build_model
from problem import (
    NU_DEFAULT, X_MIN, X_MAX, T_MIN, T_MAX,
    pde_residual, soft_ic_loss, soft_bc_loss, causal_pde_loss,
    sample_collocation, sample_ic, sample_bc,
    build_fd_reference,
)

# ─── Variant table ────────────────────────────────────────────────────────────
# label : (hard, causal, fourier, nu)
VARIANTS = {
    # Main 2³ DoE  (ν = 0.01/π)
    "soft-uniform":    (False, False, False, NU_DEFAULT),
    "soft-causal":     (False, True,  False, NU_DEFAULT),
    "hard-uniform":    (True,  False, False, NU_DEFAULT),
    "hard-causal":     (True,  True,  False, NU_DEFAULT),
    "soft-uniform-ff": (False, False, True,  NU_DEFAULT),
    "soft-causal-ff":  (False, True,  True,  NU_DEFAULT),
    "hard-uniform-ff": (True,  False, True,  NU_DEFAULT),
    "hard-causal-ff":  (True,  True,  True,  NU_DEFAULT),
    # Viscosity stress tests — best variant repeated at other ν
    "best-nu-high":    (True,  True,  False, 0.1  / math.pi),
    "best-nu-low":     (True,  True,  False, 0.001 / math.pi),
}

# ─── Hyper-parameters ─────────────────────────────────────────────────────────
ADAM_STEPS     = 30_000    # increased from 10k — Burgers needs more steps for shock region
LBFGS_STEPS    = 500       # L-BFGS polish for hard variants after Adam
LR_ADAM        = 1e-3
LR_MIN         = 1e-5      # cosine decay floor
W_IC_SOFT      = 10.0
W_BC_SOFT      = 10.0
N_COL          = 10_000
N_IC           = 256
N_BC_T         = 200       # points per BC side
N_TIME_SLABS   = 50
CAUSAL_EPS     = 100.0
LOG_EVERY      = 200
RESAMPLE_EVERY = 3_000     # re-draw collocation points periodically


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_variant(label: str, out_dir: Path):
    hard, causal, fourier, nu = VARIANTS[label]
    device = get_device()

    print(f"\n{'='*64}")
    print(f"Variant : {label}")
    print(f"  hard={hard}  causal={causal}  fourier={fourier}  ν={nu:.5f}")
    print(f"  device={device}")
    print(f"{'='*64}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── Model ────────────────────────────────────────────────────────────
    model = build_model(fourier=fourier, hard=hard).to(device)

    # ─── Collocation points (fixed throughout training) ────────────────────
    xt_col = sample_collocation(N_COL, device)
    x_ic   = sample_ic(N_IC, device)
    xt_bc  = sample_bc(N_BC_T, device)

    # ─── Optimiser + cosine LR schedule ──────────────────────────────────
    opt       = optim.Adam(model.parameters(), lr=LR_ADAM)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=ADAM_STEPS, eta_min=LR_MIN)

    # ─── CSV logger ────────────────────────────────────────────────────────
    log_path = out_dir / "training_log.csv"
    csv_fh   = open(log_path, "w", newline="")
    writer   = csv.DictWriter(csv_fh, fieldnames=[
        "step", "total_loss", "pde_loss", "bc_loss", "ic_loss", "lr", "wall_s"])
    writer.writeheader()

    t0 = time.time()

    for step in range(1, ADAM_STEPS + 1):

        # Periodic collocation resampling
        if step > 1 and (step - 1) % RESAMPLE_EVERY == 0:
            xt_col = sample_collocation(N_COL, device)
            x_ic   = sample_ic(N_IC, device)
            xt_bc  = sample_bc(N_BC_T, device)

        opt.zero_grad()

        # PDE residual (needs grad through xt_col)
        xt_col_g = xt_col.clone().detach().requires_grad_(True)
        res = pde_residual(model, xt_col_g, nu)         # (N,1)

        if causal:
            t_col = xt_col[:, 1]
            pde_l = causal_pde_loss(res.squeeze(), t_col,
                                    n_slabs=N_TIME_SLABS, eps=CAUSAL_EPS)
        else:
            pde_l = (res ** 2).mean()

        # BC / IC losses
        if hard:
            bc_l = torch.tensor(0.0, device=device)
            ic_l = torch.tensor(0.0, device=device)
        else:
            bc_l = soft_bc_loss(model, xt_bc)
            ic_l = soft_ic_loss(model, x_ic)

        loss = pde_l + W_BC_SOFT * bc_l + W_IC_SOFT * ic_l
        loss.backward()
        opt.step()
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
                "wall_s":     round(time.time() - t0, 2),
            }
            writer.writerow(row)
            csv_fh.flush()
            if step % 5000 == 0:
                print(f"  step {step:6d}  total={loss.item():.3e}  "
                      f"pde={pde_l.item():.3e}  bc={bc_l.item():.3e}  "
                      f"ic={ic_l.item():.3e}  lr={current_lr:.2e}")

    # ─── L-BFGS polishing for hard variants ───────────────────────────────
    if hard:
        print(f"  Running L-BFGS polish ({LBFGS_STEPS} max iters) …")
        opt_lbfgs = optim.LBFGS(model.parameters(), max_iter=LBFGS_STEPS,
                                  tolerance_grad=1e-9, tolerance_change=1e-12,
                                  history_size=50, line_search_fn="strong_wolfe")

        def closure():
            opt_lbfgs.zero_grad()
            xt_g = xt_col.clone().detach().requires_grad_(True)
            res_ = pde_residual(model, xt_g, nu)
            if causal:
                t_c = xt_col[:, 1]
                l_  = causal_pde_loss(res_.squeeze(), t_c, N_TIME_SLABS, CAUSAL_EPS)
            else:
                l_  = (res_ ** 2).mean()
            l_.backward()
            return l_

        opt_lbfgs.step(closure)

        # log final L-BFGS state
        with torch.no_grad():
            xt_g = xt_col.clone().detach().requires_grad_(True)
        res_ = pde_residual(model, xt_g, nu)
        if causal:
            pde_lf = causal_pde_loss(res_.squeeze(), xt_col[:, 1], N_TIME_SLABS, CAUSAL_EPS)
        else:
            pde_lf = (res_ ** 2).mean()
        writer.writerow({
            "step": ADAM_STEPS + LBFGS_STEPS,
            "total_loss": pde_lf.item(),
            "pde_loss":   pde_lf.item(),
            "bc_loss":    0.0,
            "ic_loss":    0.0,
            "lr":         0.0,
            "wall_s":     round(time.time() - t0, 2),
        })
        print(f"  L-BFGS done  pde={pde_lf.item():.3e}")

    csv_fh.close()

    # ─── Save checkpoint ───────────────────────────────────────────────────
    ckpt = {
        "model_state": model.state_dict(),
        "variant": label,
        "hard": hard,
        "causal": causal,
        "fourier": fourier,
        "nu": nu,
    }
    torch.save(ckpt, out_dir / "checkpoint.pt")

    wall = round(time.time() - t0, 1)
    print(f"  Done [{wall}s]  →  {out_dir}")
    return model, nu
