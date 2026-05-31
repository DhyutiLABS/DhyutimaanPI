"""
verify.py — Verification harness for 1D Burgers PINN DoE.

Compares PINN predictions against the high-fidelity FD reference.
Writes verification.json and plots per variant.

Usage:
    python verify.py --variant soft-causal --runs_dir examples/cowork/burgers1D/runs
    python verify.py --variant all  --runs_dir examples/cowork/burgers1D/runs
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
from problem import (
    X_MIN, X_MAX, T_MIN, T_MAX, NU_DEFAULT,
    build_fd_reference, ic_fn,
)
from train import VARIANTS


# ─── Helper ───────────────────────────────────────────────────────────────────

def rel_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12))


def load_fd_reference(runs_root: Path, nu: float) -> tuple:
    """Load or build FD reference for given ν."""
    tag  = f"{nu:.6f}".replace(".", "p")
    path = runs_root / f"fd_reference_{tag}.npy"
    if path.exists():
        data = np.load(path, allow_pickle=True).item()
        return data["x"], data["t"], data["U"]
    print(f"  Building FD reference (ν={nu:.5f}) — takes ~30s …")
    x, t, U = build_fd_reference(nu=nu)
    np.save(path, {"x": x, "t": t, "U": U})
    print(f"  FD reference saved → {path}")
    return x, t, U


def load_model_from_ckpt(run_dir: Path, device: torch.device):
    ckpt    = torch.load(run_dir / "checkpoint.pt", map_location=device)
    hard    = ckpt["hard"]
    fourier = ckpt["fourier"]
    nu      = ckpt["nu"]
    label   = ckpt["variant"]
    model   = build_model(fourier=fourier, hard=hard).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, hard, fourier, nu, label


# ─── Verification for one variant ─────────────────────────────────────────────

def verify_variant(label: str, runs_root: Path):
    run_dir = runs_root / label
    ckpt_p  = run_dir / "checkpoint.pt"
    if not ckpt_p.exists():
        print(f"  [skip] no checkpoint: {label}")
        return

    device = torch.device("cpu")    # verification always on CPU
    model, hard, fourier, nu, _ = load_model_from_ckpt(run_dir, device)

    # ─── FD reference ─────────────────────────────────────────────────────
    x_fd, t_fd, U_fd = load_fd_reference(runs_root, nu)

    # ─── Evaluation grid ──────────────────────────────────────────────────
    N_EVAL_X = 256
    N_EVAL_T = 101
    x_eval = np.linspace(X_MIN, X_MAX, N_EVAL_X)
    t_eval = np.linspace(T_MIN, T_MAX, N_EVAL_T)
    XX, TT = np.meshgrid(x_eval, t_eval)        # (N_T, N_X)
    xt_flat = np.stack([XX.ravel(), TT.ravel()], axis=1).astype(np.float32)
    xt_t    = torch.tensor(xt_flat, device=device)

    with torch.no_grad():
        u_pred = model(xt_t).cpu().numpy().reshape(N_EVAL_T, N_EVAL_X)

    # Interpolate FD onto evaluation grid
    from scipy.interpolate import RegularGridInterpolator
    interp  = RegularGridInterpolator((t_fd, x_fd), U_fd,
                                       method="linear", bounds_error=False,
                                       fill_value=None)
    u_ref   = interp(np.stack([TT.ravel(), XX.ravel()], axis=1)).reshape(N_EVAL_T, N_EVAL_X)

    # ─── Global error ─────────────────────────────────────────────────────
    err_global = rel_l2(u_pred.ravel(), u_ref.ravel())
    err_max    = float(np.max(np.abs(u_pred - u_ref)))

    # ─── Per-time-slice errors ────────────────────────────────────────────
    slice_times  = [0.0, 0.25, 0.50, 0.75, 1.0]
    slice_errors = {}
    for t_s in slice_times:
        idx = int(round(t_s * (N_EVAL_T - 1)))
        slice_errors[f"t_{t_s:.2f}"] = rel_l2(u_pred[idx], u_ref[idx])

    # Error growth ratio: t=1.0 / (t=0.0 + ε)
    growth = slice_errors["t_1.00"] / (slice_errors["t_0.00"] + 1e-12)

    # ─── Plots ────────────────────────────────────────────────────────────
    _plot_solution(u_pred, u_ref, x_eval, t_eval, slice_times,
                   slice_errors, err_global, label, run_dir)
    _plot_loss_curve(run_dir, label)

    # ─── Write verification.json ──────────────────────────────────────────
    result = {
        "variant":       label,
        "nu":            nu,
        "hard":          hard,
        "causal":        VARIANTS[label][1] if label in VARIANTS else None,
        "fourier":       fourier,
        "rel_l2":        err_global,
        "max_abs_error": err_max,
        "slice_errors":  slice_errors,
        "growth_ratio":  growth,
    }
    with open(run_dir / "verification.json", "w") as f:
        json.dump(result, f, indent=2)

    print(f"  {label:22s}  ν={nu:.5f}  rel-L²={err_global:.2e}  "
          f"max={err_max:.2e}  growth={growth:.2f}")
    return result


# ─── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_solution(u_pred, u_ref, x_eval, t_eval, slice_times,
                   slice_errors, err_global, label, run_dir):
    """x-t heatmap + slice comparisons."""
    fig = plt.figure(figsize=(16, 8))
    n_slices = len(slice_times)

    # Heatmap row
    ax_pred = fig.add_subplot(2, n_slices + 1, 1)
    ax_ref  = fig.add_subplot(2, n_slices + 1, 2)
    ax_err  = fig.add_subplot(2, n_slices + 1, 3)

    vmin = min(u_pred.min(), u_ref.min())
    vmax = max(u_pred.max(), u_ref.max())
    TT, XX = np.meshgrid(t_eval, x_eval)

    ax_pred.contourf(TT, XX, u_pred.T, 50, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax_pred.set_title(f"PINN [{label}]"); ax_pred.set_xlabel("t"); ax_pred.set_ylabel("x")

    ax_ref.contourf(TT, XX, u_ref.T, 50, vmin=vmin, vmax=vmax, cmap="RdBu_r")
    ax_ref.set_title("FD Reference"); ax_ref.set_xlabel("t")

    im = ax_err.contourf(TT, XX, np.abs(u_pred - u_ref).T, 50, cmap="hot")
    ax_err.set_title(f"|Error|  ε={err_global:.2e}"); ax_err.set_xlabel("t")
    plt.colorbar(im, ax=ax_err)

    # Per-slice row
    for i, t_s in enumerate(slice_times):
        ax = fig.add_subplot(2, n_slices + 1, n_slices + 2 + i)
        idx = int(round(t_s * (len(t_eval) - 1)))
        ax.plot(x_eval, u_ref[idx],  "k-",  lw=2,   label="FD ref")
        ax.plot(x_eval, u_pred[idx], "r--", lw=1.5, label="PINN")
        ax.set_title(f"t={t_s:.2f}  ε={slice_errors[f't_{t_s:.2f}']:.1e}")
        ax.set_xlabel("x"); ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"1D Burgers PINN — {label}", fontsize=12)
    plt.tight_layout()
    plt.savefig(run_dir / "solution_comparison.png", dpi=120)
    plt.close()


def _plot_loss_curve(run_dir: Path, label: str):
    import csv
    log_f = run_dir / "training_log.csv"
    if not log_f.exists():
        return
    steps, total, pde, bc, ic = [], [], [], [], []
    with open(log_f) as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            total.append(float(row["total_loss"]))
            pde.append(float(row["pde_loss"]))
            bc.append(float(row["bc_loss"]))
            ic.append(float(row["ic_loss"]))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(steps, total, label="total", lw=2)
    ax.semilogy(steps, pde,   label="pde",   lw=1.5, ls="--")
    if any(v > 0 for v in bc):
        ax.semilogy(steps, bc, label="bc",   lw=1.5, ls=":")
    if any(v > 0 for v in ic):
        ax.semilogy(steps, ic, label="ic",   lw=1.5, ls="-.")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss (log)")
    ax.set_title(f"Training Loss — {label}")
    ax.legend(); ax.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=110)
    plt.close()


# ─── Hypothesis evaluation ────────────────────────────────────────────────────

def evaluate_hypotheses(runs_root: Path) -> dict:
    def load(lbl):
        p = runs_root / lbl / "verification.json"
        return json.load(open(p)) if p.exists() else None

    verdicts = {}

    # H1: causal weighting reaches ≤5e-3; vanilla ≥1e-2
    r_su = load("soft-uniform")
    r_sc = load("soft-causal")
    if r_su and r_sc:
        ratio = r_su["rel_l2"] / r_sc["rel_l2"]
        verdicts["H1"] = {
            "hypothesis": "Causal weighting ≤5e-3 while vanilla ≥1e-2 (soft variants)",
            "soft_uniform_l2": r_su["rel_l2"],
            "soft_causal_l2":  r_sc["rel_l2"],
            "ratio": ratio,
            "result": "confirmed" if r_sc["rel_l2"] <= 5e-3 and r_su["rel_l2"] >= 1e-2
                      else ("partial" if ratio >= 2 else "refuted"),
            "evidence": f"ratio={ratio:.1f}x  causal={r_sc['rel_l2']:.2e}  uniform={r_su['rel_l2']:.2e}"
        }

    # H2: hard-causal ≥5× better than soft-causal
    r_hc = load("hard-causal")
    if r_hc and r_sc:
        ratio = r_sc["rel_l2"] / r_hc["rel_l2"]
        verdicts["H2"] = {
            "hypothesis": "Hard IC+BC reduces error by ≥5× vs soft-causal",
            "soft_causal_l2": r_sc["rel_l2"],
            "hard_causal_l2": r_hc["rel_l2"],
            "ratio": ratio,
            "result": "confirmed" if ratio >= 5 else "refuted",
            "evidence": f"ratio={ratio:.1f}x"
        }

    # H3: hard-causal is best of 8 DoE variants
    main_labels = [k for k in VARIANTS if not k.startswith("best-nu")]
    results = {lbl: load(lbl) for lbl in main_labels}
    valid   = {k: v for k, v in results.items() if v is not None}
    if valid and r_hc:
        best_lbl = min(valid, key=lambda k: valid[k]["rel_l2"])
        verdicts["H3"] = {
            "hypothesis": "hard-causal is best of 8 DoE variants",
            "best_variant": best_lbl,
            "best_l2": valid[best_lbl]["rel_l2"],
            "hard_causal_l2": r_hc["rel_l2"] if r_hc else None,
            "result": "confirmed" if best_lbl == "hard-causal" else "refuted",
            "evidence": f"best={best_lbl}  l2={valid[best_lbl]['rel_l2']:.2e}"
        }

    # H4: Fourier features help Burgers — soft-uniform-ff vs soft-uniform
    r_suff = load("soft-uniform-ff")
    if r_su and r_suff:
        ratio = r_su["rel_l2"] / r_suff["rel_l2"]
        verdicts["H4"] = {
            "hypothesis": "Fourier features reduce error by ≥1.5× for Burgers",
            "soft_uniform_l2":    r_su["rel_l2"],
            "soft_uniform_ff_l2": r_suff["rel_l2"],
            "ratio": ratio,
            "result": "confirmed" if ratio >= 1.5 else "refuted",
            "evidence": f"ratio={ratio:.2f}x"
        }

    # H5: best variant fails at ν=0.001/π (error > 0.1)
    r_low = load("best-nu-low")
    if r_low:
        verdicts["H5"] = {
            "hypothesis": "Method fails catastrophically at ν=0.001/π (error > 0.1)",
            "nu":     r_low["nu"],
            "rel_l2": r_low["rel_l2"],
            "result": "confirmed" if r_low["rel_l2"] > 0.1 else "refuted",
            "evidence": f"rel_l2={r_low['rel_l2']:.2e}"
        }

    return verdicts


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant",  default="all")
    parser.add_argument("--runs_dir", default="examples/cowork/burgers1D/runs")
    args = parser.parse_args()

    runs_root = Path(args.runs_dir)

    labels = (list(VARIANTS.keys()) if args.variant == "all"
              else [args.variant])

    print("\n=== Burgers 1D Verification ===")
    for lbl in labels:
        verify_variant(lbl, runs_root)

    verdicts = evaluate_hypotheses(runs_root)
    out_path = runs_root / "hypotheses_summary.json"
    with open(out_path, "w") as f:
        json.dump(verdicts, f, indent=2)
    print(f"\nHypotheses → {out_path}")
    for hid, v in verdicts.items():
        print(f"  {hid}: {v['result'].upper()}  — {v['evidence']}")
