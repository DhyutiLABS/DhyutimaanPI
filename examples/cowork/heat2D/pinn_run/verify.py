"""
verify.py — Verification harness for 2D Heat PINN DoE.

Loads checkpoint, evaluates on dense grid, computes rel-L² vs analytic,
writes verification.json, and saves plots.

Usage:
    python verify.py --variant hard-adam --run_dir examples/cowork/heat2D/runs/hard-adam
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import build_model
from train import WrappedModelA, WrappedModelB, VARIANTS
from problem import ref_A, ref_B, X_MIN, X_MAX, Y_MIN, Y_MAX, T_MIN, T_MAX


def load_model(run_dir: Path, device: torch.device):
    ckpt = torch.load(run_dir / "checkpoint.pt", map_location=device)
    label   = ckpt["variant"]
    hard    = ckpt["hard"]
    fourier = ckpt["fourier"]
    prob    = ckpt["problem"]
    d_in    = 2 if prob == "A" else 3
    base    = build_model(fourier=fourier, d_in=d_in)
    if prob == "A":
        model = WrappedModelA(base, hard).to(device)
    else:
        model = WrappedModelB(base, hard).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, prob, hard, fourier, label


def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def verify_A(model, device, run_dir: Path, label: str, hard: bool):
    N = 100
    x = np.linspace(X_MIN, X_MAX, N)
    y = np.linspace(Y_MIN, Y_MAX, N)
    X, Y = np.meshgrid(x, y)                     # (N,N)
    xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_t    = torch.tensor(xy_flat, dtype=torch.float32, device=device)

    with torch.no_grad():
        u_pred = model(xy_t).cpu().numpy().reshape(N, N)

    u_ref = np.sin(math.pi * X) * np.sin(math.pi * Y)

    err_rel  = rel_l2(u_pred.ravel(), u_ref.ravel())
    err_max  = float(np.max(np.abs(u_pred - u_ref)))

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    vmin, vmax = u_ref.min(), u_ref.max()
    im0 = axes[0].contourf(X, Y, u_pred, 50, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0].set_title(f"PINN [{label}]"); plt.colorbar(im0, ax=axes[0])
    im1 = axes[1].contourf(X, Y, u_ref,  50, vmin=vmin, vmax=vmax, cmap="viridis")
    axes[1].set_title("Reference"); plt.colorbar(im1, ax=axes[1])
    err_map = np.abs(u_pred - u_ref)
    im2 = axes[2].contourf(X, Y, err_map, 50, cmap="hot")
    axes[2].set_title(f"|Error|  rel-L²={err_rel:.2e}"); plt.colorbar(im2, ax=axes[2])
    fig.suptitle(f"2D Steady Heat — {label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(run_dir / "solution_comparison.png", dpi=120)
    plt.close()

    return err_rel, err_max


def verify_B(model, device, run_dir: Path, label: str, hard: bool):
    N_x, N_y = 50, 50
    T_SLICES  = [0.0, 0.02, 0.05, 0.1]
    x = np.linspace(X_MIN, X_MAX, N_x)
    y = np.linspace(Y_MIN, Y_MAX, N_y)
    X, Y = np.meshgrid(x, y)

    all_pred, all_ref, slice_errs = [], [], {}
    for t_val in T_SLICES:
        t_col  = np.full((N_x * N_y, 1), t_val)
        xy_f   = np.stack([X.ravel(), Y.ravel()], axis=1)
        xyt_f  = np.concatenate([xy_f, t_col], axis=1)
        xyt_t  = torch.tensor(xyt_f, dtype=torch.float32, device=device)
        with torch.no_grad():
            u_p = model(xyt_t).cpu().numpy().reshape(N_x, N_y)
        u_r = np.sin(math.pi * X) * np.sin(math.pi * Y) * math.exp(-2 * math.pi**2 * t_val)
        all_pred.append(u_p.ravel()); all_ref.append(u_r.ravel())
        slice_errs[f"t_{t_val:.3f}"] = rel_l2(u_p.ravel(), u_r.ravel())

    err_rel = rel_l2(np.concatenate(all_pred), np.concatenate(all_ref))
    err_max = float(max(
        np.max(np.abs(p - r))
        for p, r in zip(all_pred, all_ref)
    ))

    # error growth ratio
    t0_err = slice_errs[f"t_{T_SLICES[0]:.3f}"]
    tT_err = slice_errs[f"t_{T_SLICES[-1]:.3f}"]
    growth = tT_err / (t0_err + 1e-12)

    # ── Plots ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, len(T_SLICES), figsize=(4*len(T_SLICES), 7))
    for i, t_val in enumerate(T_SLICES):
        t_col = np.full((N_x * N_y, 1), t_val)
        xy_f  = np.stack([X.ravel(), Y.ravel()], axis=1)
        xyt_f = np.concatenate([xy_f, t_col], axis=1)
        xyt_t = torch.tensor(xyt_f, dtype=torch.float32, device=device)
        with torch.no_grad():
            u_p = model(xyt_t).cpu().numpy().reshape(N_x, N_y)
        u_r = (np.sin(math.pi * X) * np.sin(math.pi * Y)
               * math.exp(-2 * math.pi**2 * t_val))
        axes[0, i].contourf(X, Y, u_p, 30, cmap="viridis")
        axes[0, i].set_title(f"PINN t={t_val:.2f}")
        axes[1, i].contourf(X, Y, np.abs(u_p - u_r), 30, cmap="hot")
        axes[1, i].set_title(f"|Err| {slice_errs[f't_{t_val:.3f}']:.1e}")
    fig.suptitle(f"2D Unsteady Heat — {label}  |  global rel-L²={err_rel:.2e}", fontsize=11)
    plt.tight_layout()
    plt.savefig(run_dir / "solution_comparison.png", dpi=110)
    plt.close()

    return err_rel, err_max, slice_errs, growth


def plot_loss_curve(run_dir: Path, label: str):
    import csv
    log_file = run_dir / "training_log.csv"
    if not log_file.exists():
        return
    steps, total, pde, bc, ic = [], [], [], [], []
    with open(log_file) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            total.append(float(row["total_loss"]))
            pde.append(float(row["pde_loss"]))
            bc.append(float(row["bc_loss"]))
            ic.append(float(row["ic_loss"]))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, total, label="total", lw=2)
    ax.semilogy(steps, pde,   label="pde",   lw=1.5, ls="--")
    ax.semilogy(steps, bc,    label="bc",    lw=1.5, ls=":")
    if any(v > 0 for v in ic):
        ax.semilogy(steps, ic, label="ic",   lw=1.5, ls="-.")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"Training Loss — {label}")
    ax.legend(); ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=110)
    plt.close()


def verify_variant(label: str, runs_root: Path):
    run_dir = runs_root / label
    if not (run_dir / "checkpoint.pt").exists():
        print(f"  [skip] no checkpoint for {label}")
        return

    device = torch.device("cpu")   # verification always on CPU for stability
    model, prob, hard, fourier, _ = load_model(run_dir, device)
    plot_loss_curve(run_dir, label)

    if prob == "A":
        err_rel, err_max = verify_A(model, device, run_dir, label, hard)
        result = {
            "variant":       label,
            "problem":       "A_steady",
            "rel_l2":        err_rel,
            "max_abs_error": err_max,
            "slice_errors":  {},
            "growth_ratio":  None,
        }
    else:
        err_rel, err_max, slice_errs, growth = verify_B(model, device, run_dir, label, hard)
        result = {
            "variant":       label,
            "problem":       "B_unsteady",
            "rel_l2":        err_rel,
            "max_abs_error": err_max,
            "slice_errors":  slice_errs,
            "growth_ratio":  growth,
        }

    print(f"  {label:20s}  rel-L²={err_rel:.2e}  max-abs={err_max:.2e}"
          + (f"  growth={growth:.2f}" if prob == "B" else ""))
    with open(run_dir / "verification.json", "w") as f:
        json.dump(result, f, indent=2)


def evaluate_hypotheses(runs_root: Path) -> dict:
    """
    Load all verification.json files and evaluate the 5 hypotheses from the spec.
    """
    verdicts = {}

    def load(label):
        p = runs_root / label / "verification.json"
        if p.exists():
            with open(p) as f:
                return json.load(f)
        return None

    # H1: hard-adam vs soft-adam (Problem A) — ≥100× improvement
    r_hard = load("hard-adam")
    r_soft = load("soft-adam")
    if r_hard and r_soft:
        ratio = r_soft["rel_l2"] / r_hard["rel_l2"]
        verdicts["H1"] = {
            "hypothesis": "Hard constraint reduces steady rel-L² by ≥100× vs soft-adam",
            "soft_rel_l2": r_soft["rel_l2"],
            "hard_rel_l2": r_hard["rel_l2"],
            "ratio": ratio,
            "result": "confirmed" if ratio >= 100 else "refuted",
            "evidence": f"ratio = {ratio:.1f}x"
        }

    # H2: hard-lbfgs vs hard-adam — ≥5× improvement
    r_hl = load("hard-lbfgs")
    r_ha = load("hard-adam")
    if r_hl and r_ha:
        ratio = r_ha["rel_l2"] / r_hl["rel_l2"]
        verdicts["H2"] = {
            "hypothesis": "L-BFGS polishing reduces hard-constraint error by ≥5×",
            "hard_adam_rel_l2": r_ha["rel_l2"],
            "hard_lbfgs_rel_l2": r_hl["rel_l2"],
            "ratio": ratio,
            "result": "confirmed" if ratio >= 5 else "refuted",
            "evidence": f"ratio = {ratio:.1f}x"
        }

    # H3: hard-adam-ff vs hard-adam — Fourier features hurt (≥2× worse)
    r_hff = load("hard-adam-ff")
    if r_hff and r_ha:
        ratio = r_hff["rel_l2"] / r_ha["rel_l2"]
        verdicts["H3"] = {
            "hypothesis": "Fourier features hurt smooth solution (hard+FF ≥2× worse than hard)",
            "hard_rel_l2": r_ha["rel_l2"],
            "hard_ff_rel_l2": r_hff["rel_l2"],
            "ratio": ratio,
            "result": "confirmed" if ratio >= 2 else "refuted",
            "evidence": f"ratio = {ratio:.1f}x"
        }

    # H4: soft-causal growth ≤ 0.5 × soft-std growth
    r_sc = load("soft-causal")
    r_ss = load("soft-std")
    if r_sc and r_ss and r_sc.get("growth_ratio") and r_ss.get("growth_ratio"):
        g_causal = r_sc["growth_ratio"]
        g_std    = r_ss["growth_ratio"]
        verdicts["H4"] = {
            "hypothesis": "Causal weighting reduces temporal error growth by ≥2× vs soft-std",
            "soft_std_growth":    g_std,
            "soft_causal_growth": g_causal,
            "result": "confirmed" if g_causal <= 0.5 * g_std else "refuted",
            "evidence": f"growth_std={g_std:.2f}  growth_causal={g_causal:.2f}"
        }

    # H5: hard-std (B) worse than soft-std (B) — singular ansatz
    r_bs  = load("soft-std")
    r_bhs = load("hard-std")
    if r_bs and r_bhs:
        verdicts["H5"] = {
            "hypothesis": "Hard IC ansatz for Problem B fails (worse than soft-std)",
            "soft_std_rel_l2": r_bs["rel_l2"],
            "hard_std_rel_l2": r_bhs["rel_l2"],
            "result": "confirmed" if r_bhs["rel_l2"] > r_bs["rel_l2"] else "refuted",
            "evidence": f"hard={r_bhs['rel_l2']:.2e}  soft={r_bs['rel_l2']:.2e}"
        }

    return verdicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default="all",
                        help="Label of variant to verify, or 'all'")
    parser.add_argument("--runs_dir",
                        default="examples/cowork/heat2D/runs")
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)

    from train import VARIANTS
    labels = list(VARIANTS.keys()) if args.variant == "all" else [args.variant]

    print("\n=== Verification ===")
    for lbl in labels:
        verify_variant(lbl, runs_root)

    # Evaluate and save hypotheses
    verdicts = evaluate_hypotheses(runs_root)
    out_path = runs_root / "hypotheses_summary.json"
    with open(out_path, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"\nHypotheses summary → {out_path}")
    for hid, v in verdicts.items():
        print(f"  {hid}: {v['result'].upper()}  — {v['evidence']}")
