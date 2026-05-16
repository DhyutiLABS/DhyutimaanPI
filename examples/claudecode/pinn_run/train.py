"""
train.py — Two-stage training loop for the 2D Heat PINN.

Stage 1 : Adam    (default 10 000 iterations)
Stage 2 : L-BFGS  (default 2 000 iterations)

Loss:
    L = w_pde * mean(∇²T)²  +  w_bc * mean((T - T_bc)²)

Outputs written to pinn_run/outputs/<run_name>/:
    training_log.csv   — per-step loss breakdown
    checkpoint.pt      — model state dict + config
"""

import csv
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim

# Allow running as  python pinn_run/train.py  from project root
sys.path.insert(0, str(Path(__file__).parent))

from model import build_model
from problem import pde_residual, bc_loss, sample_interior, sample_boundary

OUTPUTS_ROOT = Path(__file__).parent / "outputs"


# ── Device selection ──────────────────────────────────────────────────────────

def select_device() -> torch.device:
    """Try MPS (Apple Silicon); fall back to CPU if torch.func is unsupported."""
    if torch.backends.mps.is_available():
        try:
            t = torch.zeros(2, device="mps")
            torch.func.hessian(lambda x: x.sum())(t)
            print("[device] Using MPS.")
            return torch.device("mps")
        except Exception:
            print("[device] torch.func not fully supported on this MPS build; using CPU.")
    return torch.device("cpu")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    n_collocation: int = 4096,
    n_per_edge: int = 100,
    n_adam: int = 10_000,
    n_lbfgs: int = 2_000,
    w_pde: float = 1.0,
    w_bc: float = 10.0,
    lr_adam: float = 1e-3,
    hidden: int = 64,
    depth: int = 5,
    device: torch.device | None = None,
    use_torch_func: bool = True,
    seed: int = 42,
    log_every: int = 100,
    run_name: str | None = None,
) -> tuple[torch.nn.Module, Path]:
    """Train the PINN and save checkpoint + training log.

    Returns the trained model.
    """
    torch.manual_seed(seed)

    if run_name is None:
        lbfgs_tag = f"_lbfgs{n_lbfgs}" if n_lbfgs > 0 else "_nolbfgs"
        run_name = f"nr{n_collocation}_adam{n_adam}{lbfgs_tag}"
    output_dir = OUTPUTS_ROOT / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[train] Output dir: {output_dir}")

    if device is None:
        device = select_device()
    print(f"[train] device={device}  N_r={n_collocation}  N_bc={4*n_per_edge}")
    print(f"[train] Adam={n_adam} iters  L-BFGS={n_lbfgs} iters  w_pde={w_pde}  w_bc={w_bc}")

    model = build_model(hidden=hidden, depth=depth, device=device)
    print(f"[train] Model params: {model.n_params:,}")

    # Fixed point sets (sampled once; see spec §9 for resampling discussion)
    x_r          = sample_interior(n_collocation, device)
    x_bc, T_bc   = sample_boundary(n_per_edge, device)

    # ── Loss function ──────────────────────────────────────────────────────
    def compute_loss():
        r      = pde_residual(model, x_r, use_torch_func=use_torch_func)
        l_pde  = w_pde * (r ** 2).mean()
        l_bc   = w_bc  * bc_loss(model, x_bc, T_bc)
        total  = l_pde + l_bc
        return total, l_pde.item(), l_bc.item()

    log_rows: list[list] = []
    t0 = time.perf_counter()

    # ── Stage 1: Adam ──────────────────────────────────────────────────────
    print("\n── Stage 1: Adam ──────────────────────────────────────────────")
    optimizer_adam = optim.Adam(model.parameters(), lr=lr_adam)

    for step in range(1, n_adam + 1):
        optimizer_adam.zero_grad()
        total, l_pde, l_bc = compute_loss()
        total.backward()
        optimizer_adam.step()

        if step % log_every == 0 or step == 1:
            elapsed = time.perf_counter() - t0
            lr      = optimizer_adam.param_groups[0]["lr"]
            log_rows.append([step, total.item(), l_pde, l_bc, 0.0, lr, f"{elapsed:.2f}"])
            print(
                f"  step {step:6d}  total={total.item():.4e}"
                f"  pde={l_pde:.4e}  bc={l_bc:.4e}  t={elapsed:.1f}s"
            )

    # ── Stage 2: L-BFGS ───────────────────────────────────────────────────
    if n_lbfgs > 0:
        print("\n── Stage 2: L-BFGS ────────────────────────────────────────────")
        lbfgs_call_count = [0]

        optimizer_lbfgs = optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=n_lbfgs,          # all iterations in one step() call
            max_eval=int(n_lbfgs * 1.25),
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=100,
            line_search_fn="strong_wolfe",
        )

        def closure():
            optimizer_lbfgs.zero_grad()
            total, l_pde, l_bc = compute_loss()
            total.backward()

            lbfgs_call_count[0] += 1
            n = lbfgs_call_count[0]
            if n % 50 == 0 or n == 1:
                elapsed = time.perf_counter() - t0
                log_rows.append(
                    [n_adam + n, total.item(), l_pde, l_bc, 0.0, 0.0, f"{elapsed:.2f}"]
                )
                print(
                    f"  L-BFGS eval {n:4d}  total={total.item():.4e}"
                    f"  pde={l_pde:.4e}  bc={l_bc:.4e}  t={elapsed:.1f}s"
                )
            return total

        optimizer_lbfgs.step(closure)

    # ── Save checkpoint ────────────────────────────────────────────────────
    ckpt_path = output_dir / "checkpoint.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "hidden": hidden,
                "depth": depth,
                "in_dim": 2,
                "out_dim": 1,
            },
            "train_config": {
                "n_collocation": n_collocation,
                "n_per_edge": n_per_edge,
                "n_adam": n_adam,
                "n_lbfgs": n_lbfgs,
                "w_pde": w_pde,
                "w_bc": w_bc,
                "seed": seed,
            },
        },
        ckpt_path,
    )
    print(f"\n[train] Checkpoint saved → {ckpt_path}")

    # ── Write CSV log ──────────────────────────────────────────────────────
    csv_path = output_dir / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["iteration", "total_loss", "pde_loss", "bc_loss", "ic_loss", "lr", "time_elapsed_s"]
        )
        writer.writerows(log_rows)
    print(f"[train] Training log → {csv_path}")

    total_time = time.perf_counter() - t0
    print(f"[train] Total wall time: {total_time:.1f}s")

    return model, output_dir


if __name__ == "__main__":
    train()
