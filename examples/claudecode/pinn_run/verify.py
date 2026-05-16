"""
verify.py — Verification harness for the 2D Heat PINN.

Loads pinn_run/outputs/checkpoint.pt, evaluates on a 200×200 grid,
computes error metrics against the analytic solution, saves a 4-panel
plot, and writes pinn_run/outputs/verification.json.

Success criteria (from problem-spec §7):
  • Relative L2 error  ≤ 1e-3
  • Max pointwise error ≤ 5e-3
  • BC residual (top wall MSE) ≤ 1e-6
"""

import json
import math
import sys
from pathlib import Path

import torch
import matplotlib

matplotlib.use("Agg")   # non-interactive backend; safe on headless systems
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))

from model import build_model
from problem import reference_solution, pde_residual, PI

OUTPUTS_ROOT = Path(__file__).parent / "outputs"
GRID_N  = 200    # evaluation grid resolution (200×200 = 40 000 points)
RES_N   = 64     # coarser grid for residual field (64×64)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _meshgrid_points(n: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (xy_flat, (XX, YY)) for an n×n grid on [0,1]²."""
    xs = torch.linspace(0.0, 1.0, n, device=device)
    ys = torch.linspace(0.0, 1.0, n, device=device)
    XX, YY = torch.meshgrid(xs, ys, indexing="ij")
    xy_flat = torch.stack([XX.flatten(), YY.flatten()], dim=1)
    return xy_flat, (XX, YY)


def _to_np(t: torch.Tensor, n: int) -> "np.ndarray":
    import numpy as np
    return t.detach().cpu().numpy().reshape(n, n)


# ── Main verification ─────────────────────────────────────────────────────────

def verify(
    device: torch.device | None = None,
    use_torch_func: bool = True,
    output_dir: Path | None = None,
) -> dict:
    """Run the full verification suite and return the results dict."""
    import numpy as np

    out = output_dir or OUTPUTS_ROOT
    ckpt_path = out / "checkpoint.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Run train.py first.")

    if device is None:
        device = torch.device("cpu")

    raw = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = raw["config"]
    model = build_model(hidden=cfg["hidden"], depth=cfg["depth"], device=device)
    model.load_state_dict(raw["model_state"])
    model.eval()

    train_cfg = raw.get("train_config", {})
    print(f"[verify] Loaded checkpoint: {ckpt_path}")
    print(f"         Train config: {train_cfg}")

    # ── Dense solution grid ────────────────────────────────────────────────
    xy_grid, (XX, YY) = _meshgrid_points(GRID_N, device)

    with torch.no_grad():
        T_pred = model(xy_grid)                   # (N², 1)
        T_ref  = reference_solution(xy_grid)      # (N², 1)

    err_abs  = (T_pred - T_ref).abs()             # (N², 1)
    rel_l2   = (
        torch.norm(T_pred - T_ref) / torch.norm(T_ref)
    ).item()
    max_err  = err_abs.max().item()

    # ── Boundary residual check (top wall: y=1, T=sin(πx)) ─────────────
    t_vals  = torch.linspace(0.0, 1.0, 200, device=device)
    x_top   = torch.stack([t_vals, torch.ones_like(t_vals)], dim=1)
    with torch.no_grad():
        T_top_pred = model(x_top).squeeze()
    T_top_true = torch.sin(PI * t_vals)
    bc_mse_top = ((T_top_pred - T_top_true) ** 2).mean().item()

    # Same check for the three zero-BC walls
    zero_walls = []
    for wall_pts in [
        torch.stack([torch.zeros(200, device=device), t_vals], dim=1),   # left
        torch.stack([torch.ones(200, device=device),  t_vals], dim=1),   # right
        torch.stack([t_vals, torch.zeros(200, device=device)], dim=1),   # bottom
    ]:
        with torch.no_grad():
            pred = model(wall_pts).squeeze()
        zero_walls.append((pred ** 2).mean().item())
    bc_mse_zero = float(np.mean(zero_walls))

    # ── PDE residual field (interior coarse grid) ─────────────────────
    xy_res, _ = _meshgrid_points(RES_N, device)
    # Strip boundary to avoid edge effects in residual computation
    mask  = (xy_res[:, 0] > 0.02) & (xy_res[:, 0] < 0.98) & \
            (xy_res[:, 1] > 0.02) & (xy_res[:, 1] < 0.98)
    xy_int = xy_res[mask]

    try:
        residuals = pde_residual(model, xy_int, use_torch_func=use_torch_func)
        max_residual = residuals.abs().max().item()
        rms_residual = residuals.pow(2).mean().sqrt().item()
    except Exception as e:
        print(f"[verify] Residual computation failed: {e}. Setting to NaN.")
        residuals    = torch.zeros(xy_int.shape[0], device=device)
        max_residual = float("nan")
        rms_residual = float("nan")

    # ── Success criteria ───────────────────────────────────────────────────
    crit_l2  = rel_l2  <= 1e-3
    crit_max = max_err <= 5e-3
    crit_bc  = bc_mse_top <= 1e-6
    success  = crit_l2 and crit_max

    # ── Hypothesis verdicts ────────────────────────────────────────────────
    nc = train_cfg.get("n_collocation", "?")
    hypotheses = [
        {
            "id": 1,
            "statement": f"PINN achieves rel-L2 < 1e-3 with {nc} collocation points (Adam+L-BFGS)",
            "result": "confirmed" if crit_l2 else "refuted",
            "evidence": f"rel_l2 = {rel_l2:.3e}  (threshold 1e-3)",
        },
        {
            "id": 2,
            "statement": "L-BFGS reduces rel-L2 by ≥5× vs Adam-only",
            "result": "not_measured",
            "evidence": (
                "Requires a separate Adam-only checkpoint. "
                "Re-run: python pinn_run/run.py --no-lbfgs, then compare."
            ),
        },
        {
            "id": 3,
            "statement": "Rel-L2 decreases monotonically across 256→1024→4096→16384 collocation pts",
            "result": "not_measured",
            "evidence": "Run: python pinn_run/run.py --sweep-collocation",
        },
        {
            "id": 4,
            "statement": "BC weight w_bc=10 reduces boundary residual vs w_bc=1",
            "result": "not_measured",
            "evidence": "Run: python pinn_run/run.py --w-bc 1.0 and compare bc_mse_top.",
        },
    ]

    result = {
        "relative_l2_error":  rel_l2,
        "max_pointwise_error": max_err,
        "bc_mse_top_wall":    bc_mse_top,
        "bc_mse_zero_walls":  bc_mse_zero,
        "max_pde_residual":   max_residual,
        "rms_pde_residual":   rms_residual,
        "success_criteria_met": success,
        "criteria_detail": {
            "rel_l2_leq_1e-3":   crit_l2,
            "max_err_leq_5e-3":  crit_max,
            "bc_top_mse_leq_1e-6": crit_bc,
        },
        "train_config": train_cfg,
        "hypotheses": hypotheses,
    }

    # ── Console summary ────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  Verification Results")
    print("═" * 55)
    print(f"  Relative L2 error     : {rel_l2:.4e}  {'✓' if crit_l2 else '✗'}  (≤ 1e-3)")
    print(f"  Max pointwise error   : {max_err:.4e}  {'✓' if crit_max else '✗'}  (≤ 5e-3)")
    print(f"  BC MSE — top wall     : {bc_mse_top:.4e}  {'✓' if crit_bc else '✗'}  (≤ 1e-6)")
    print(f"  BC MSE — zero walls   : {bc_mse_zero:.4e}")
    print(f"  Max PDE residual      : {max_residual:.4e}")
    print(f"  RMS PDE residual      : {rms_residual:.4e}")
    print("─" * 55)
    print(f"  Overall : {'PASS ✓' if success else 'FAIL ✗'}")
    print("═" * 55 + "\n")

    # ── Save JSON ──────────────────────────────────────────────────────────
    json_path = out / "verification.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"[verify] Results → {json_path}")

    # ── Plots ──────────────────────────────────────────────────────────────
    _save_plots(T_pred, T_ref, err_abs, residuals, xy_int, train_cfg, out)

    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def _save_plots(
    T_pred:    torch.Tensor,
    T_ref:     torch.Tensor,
    err_abs:   torch.Tensor,
    residuals: torch.Tensor,
    xy_int:    torch.Tensor,
    train_cfg: dict,
    out:       Path,
) -> None:
    import numpy as np

    nc   = train_cfg.get("n_collocation", "?")
    na   = train_cfg.get("n_adam", "?")
    nlb  = train_cfg.get("n_lbfgs", "?")
    title = f"2D Heat PINN  |  N_r={nc}  Adam={na}  L-BFGS={nlb}"

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    def show(ax, data, label, cmap="hot", n=GRID_N):
        arr = _to_np(data, n)
        im  = ax.imshow(arr.T, origin="lower", extent=[0, 1, 0, 1], cmap=cmap, aspect="equal")
        ax.set_title(label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    show(axes[0, 0], T_pred,   "PINN prediction  $T_{pred}$",      "hot")
    show(axes[0, 1], T_ref,    "Analytic solution  $T^*$",          "hot")
    show(axes[1, 0], err_abs,  "Pointwise absolute error  $|e|$",   "Blues")

    # Residual field — scatter since grid has boundary mask
    import numpy as np
    xn = xy_int[:, 0].detach().cpu().numpy()
    yn = xy_int[:, 1].detach().cpu().numpy()
    rn = residuals.abs().detach().cpu().numpy()
    sc = axes[1, 1].scatter(xn, yn, c=rn, cmap="Reds", s=4, vmin=0)
    axes[1, 1].set_title("PDE residual  $|\\nabla^2 T|$")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("y")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    plt.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_path = out / "verification_plots.png"
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[verify] Plot → {plot_path}")

    # ── Learning curve (if training log exists) ────────────────────────
    log_path = out / "training_log.csv"
    if log_path.exists():
        import csv
        iters, totals, pdes, bcs = [], [], [], []
        with open(log_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                iters.append(int(row["iteration"]))
                totals.append(float(row["total_loss"]))
                pdes.append(float(row["pde_loss"]))
                bcs.append(float(row["bc_loss"]))

        fig2, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(iters, totals, label="total", color="k",      linewidth=1.5)
        ax.semilogy(iters, pdes,   label="PDE",   color="steelblue", linewidth=1.2)
        ax.semilogy(iters, bcs,    label="BC",    color="coral",   linewidth=1.2)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (log scale)")
        ax.set_title("Training loss curve")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()
        curve_path = out / "loss_curve.png"
        plt.savefig(curve_path, dpi=150, bbox_inches="tight")
        plt.close(fig2)
        print(f"[verify] Loss curve → {curve_path}")


if __name__ == "__main__":
    verify()
