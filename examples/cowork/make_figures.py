"""
make_figures.py  —  Compose publication-quality multi-panel figures for the CAISc 2026 paper.

Does NOT require PyTorch: uses existing per-run PNGs + verification.json / training_log.csv.

Produces:
  figures/figure_heat_A.pdf   — Prob A: solution+error panels + training loss + bar chart
  figures/figure_heat_B.pdf   — Prob B: temporal-slice error lines + causal pathology panels
  figures/figure_burgers.pdf  — Burgers: best vs worst solution panels + error bar chart
"""

import json, csv, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg

ROOT      = Path(__file__).resolve().parent.parent.parent
HEAT_RUN  = ROOT / "examples/cowork/heat2D/runs"
BURG_RUN  = ROOT / "examples/cowork/burgers1D/runs"
FIG_DIR   = Path(__file__).parent / "figures"
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   7.5,
    "xtick.labelsize":  6.5,
    "ytick.labelsize":  6.5,
    "legend.fontsize":  6.5,
    "figure.dpi":       150,
    "axes.linewidth":   0.6,
    "xtick.major.width":0.5,
    "ytick.major.width":0.5,
})

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path):
    with open(path) as f:
        return json.load(f)

def load_csv_loss(path):
    steps, cols = [], {}
    with open(path) as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            steps.append(int(row["step"]))
            for k in fields:
                if k == "step": continue
                cols.setdefault(k, []).append(float(row[k]))
    return np.array(steps), {k: np.array(v) for k, v in cols.items()}

def show_img(ax, path, title=""):
    img = mpimg.imread(path)
    ax.imshow(img)
    ax.axis("off")
    if title:
        ax.set_title(title, pad=3)

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — Heat 2D Problem A
#   Row 0: reference img (from hard-adam panel), hard-adam solution img
#   Row 1: soft-adam solution img,  hard-adam error img
#   Col 3 (spanning rows): training loss curves
#   Col 4 (separate): bar chart of all 8 A-variants
# ─────────────────────────────────────────────────────────────────────────────

def fig_heat_A():
    variants_A = [
        ("hard-lbfgs-ff", "Hard-L-BFGS-FF"),
        ("hard-adam-ff",  "Hard-Adam-FF"),
        ("hard-adam",     "Hard-Adam"),
        ("hard-lbfgs",    "Hard-L-BFGS"),
        ("soft-lbfgs",    "Soft-L-BFGS"),
        ("soft-adam",     "Soft-Adam"),
        ("soft-lbfgs-ff", "Soft-L-BFGS-FF"),
        ("soft-adam-ff",  "Soft-Adam-FF"),
    ]

    rel_l2s, labels_b = [], []
    for label, disp in variants_A:
        v = load_json(HEAT_RUN / label / "verification.json")
        rel_l2s.append(v["rel_l2"])
        labels_b.append(disp)

    steps_h, loss_h = load_csv_loss(HEAT_RUN / "hard-adam" / "training_log.csv")
    steps_s, loss_s = load_csv_loss(HEAT_RUN / "soft-adam" / "training_log.csv")

    fig = plt.figure(figsize=(9.5, 4.8))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.55, wspace=0.28,
                             width_ratios=[1.2, 1.2, 1.6, 1.6])

    # ── Image panels ──────────────────────────────────────────────────────────
    img_ha = mpimg.imread(HEAT_RUN / "hard-adam" / "solution_comparison.png")
    img_sa = mpimg.imread(HEAT_RUN / "soft-adam" / "solution_comparison.png")

    ax_ha = fig.add_subplot(gs[0, :2])
    ax_ha.imshow(img_ha); ax_ha.axis("off")
    ax_ha.set_title(r"Hard-Adam  ($\varepsilon_{\rm rel}=4.31\times10^{-6}$)"
                    "  |  PINN · Reference · |Error|", pad=3)

    ax_sa = fig.add_subplot(gs[1, :2])
    ax_sa.imshow(img_sa); ax_sa.axis("off")
    ax_sa.set_title(r"Soft-Adam  ($\varepsilon_{\rm rel}=2.47\times10^{-4}$)"
                    "  |  PINN · Reference · |Error|", pad=3)

    # ── Training loss ─────────────────────────────────────────────────────────
    ax_loss = fig.add_subplot(gs[:, 2])
    ax_loss.semilogy(steps_h, loss_h["total_loss"],  lw=1.5, color="C0",
                     label="Hard-Adam (total)")
    ax_loss.semilogy(steps_s, loss_s["total_loss"],  lw=1.5, color="C1",
                     label="Soft-Adam (total)")
    ax_loss.semilogy(steps_s, loss_s["pde_loss"],    lw=1.0, ls="--", color="C1",
                     alpha=0.7, label="Soft-Adam (PDE)")
    ax_loss.semilogy(steps_s[loss_s["bc_loss"] > 0],
                     loss_s["bc_loss"][loss_s["bc_loss"] > 0] * 100,
                     lw=1.0, ls=":", color="C3", alpha=0.7,
                     label=r"Soft-Adam ($100\times$BC)")
    ax_loss.axhline(4.31e-6, color="C0", lw=0.7, ls="-.", alpha=0.5,
                    label=r"$\varepsilon_{\rm rel}$ Hard-Adam")
    ax_loss.set_xlabel("Optimisation step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training loss — Problem A")
    ax_loss.legend(loc="upper right", framealpha=0.8)
    ax_loss.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_loss.set_xlim(0)

    # ── Bar chart all variants ────────────────────────────────────────────────
    ax_bar = fig.add_subplot(gs[:, 3])
    colors = ["#2166ac"]*4 + ["#d73027"]*4
    bars   = ax_bar.barh(range(len(labels_b)), rel_l2s, color=colors, height=0.55)
    ax_bar.set_xscale("log")
    ax_bar.set_yticks(range(len(labels_b)))
    ax_bar.set_yticklabels(labels_b, fontsize=6.5)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(r"Relative $L^2$ error $\varepsilon_{\rm rel}$")
    ax_bar.set_title("All Problem A variants")
    ax_bar.axvline(1e-4, color="gray", lw=0.8, ls="--", alpha=0.6)
    ax_bar.text(1e-4*1.1, -0.7, "target", fontsize=5.5, color="gray")
    for bar, v in zip(bars, rel_l2s):
        ax_bar.text(v * 1.2, bar.get_y() + bar.get_height()/2,
                    f"{v:.1e}", va="center", fontsize=5.5)
    ax_bar.grid(True, axis="x", alpha=0.25, lw=0.4)
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(color="#2166ac", label="Hard BC"),
                            Patch(color="#d73027", label="Soft BC")],
                  loc="lower right", fontsize=6)

    fig.suptitle("2D Steady Poisson (Problem A): Hard-constraint vs. Soft-constraint",
                 fontsize=10, y=1.01)
    fig.savefig(FIG_DIR / "figure_heat_A.pdf", bbox_inches="tight")
    plt.close()
    print("  figure_heat_A.pdf written")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Heat 2D Problem B
#   Left half: hard-std vs soft-causal per-slice error line plot
#   Middle: solution comparison images (hard-std row, soft-causal row)
#   Right: training loss comparison
# ─────────────────────────────────────────────────────────────────────────────

def fig_heat_B():
    T_SLICES  = [0.0, 0.02, 0.05, 0.1]
    t_labels  = [f"$t={t}$" for t in T_SLICES]

    def slice_errs(label):
        v = load_json(HEAT_RUN / label / "verification.json")
        return [v["slice_errors"].get(f"t_{t:.3f}", np.nan) for t in T_SLICES]

    # Collect errors for all B variants
    b_variants = [
        ("hard-std",       "Hard-Std",       "C0",    "-",  2.0),
        ("hard-causal",    "Hard-Causal",    "C0",    "--", 1.2),
        ("hard-std-ff",    "Hard-Std-FF",    "C4",    "-",  1.0),
        ("soft-std",       "Soft-Std",       "C1",    "-",  2.0),
        ("soft-std-ff",    "Soft-Std-FF",    "C1",    "--", 1.0),
        ("soft-causal",    "Soft-Causal",    "C3",    "-",  2.0),
        ("soft-causal-ff", "Soft-Causal-FF", "C3",    "--", 1.2),
        ("hard-causal-ff", "Hard-Causal-FF", "darkred",":",  1.5),
    ]

    steps_hs, loss_hs = load_csv_loss(HEAT_RUN / "hard-std"    / "training_log.csv")
    steps_sc, loss_sc = load_csv_loss(HEAT_RUN / "soft-causal" / "training_log.csv")

    img_hs = mpimg.imread(HEAT_RUN / "hard-std"    / "solution_comparison.png")
    img_sc = mpimg.imread(HEAT_RUN / "soft-causal" / "solution_comparison.png")

    fig = plt.figure(figsize=(9.5, 5.0))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35,
                             width_ratios=[1.5, 2.0, 1.5])

    # Slice error lines
    ax_sl = fig.add_subplot(gs[:, 0])
    for (lbl, disp, col, ls, lw) in b_variants:
        errs = slice_errs(lbl)
        ax_sl.semilogy(T_SLICES, errs, color=col, ls=ls, lw=lw,
                       marker="o", ms=3, label=disp)
    ax_sl.set_xlabel("Time $t$")
    ax_sl.set_ylabel(r"Slice rel-$L^2$ error")
    ax_sl.set_title("Per-slice errors — all Problem B variants")
    ax_sl.legend(fontsize=5.5, loc="lower right", ncol=1, framealpha=0.8)
    ax_sl.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_sl.set_xticks(T_SLICES)
    ax_sl.set_xticklabels([str(t) for t in T_SLICES])

    # Image panels
    ax_img_hs = fig.add_subplot(gs[0, 1])
    ax_img_hs.imshow(img_hs); ax_img_hs.axis("off")
    ax_img_hs.set_title(r"Hard-Std  ($\varepsilon_{\rm rel}=5.04\times10^{-4}$, flat profile)"
                        " — PINN at $t\in\{0,0.02,0.05,0.1\}$ and |Error|", pad=3)

    ax_img_sc = fig.add_subplot(gs[1, 1])
    ax_img_sc.imshow(img_sc); ax_img_sc.axis("off")
    ax_img_sc.set_title(r"Soft-Causal  ($\varepsilon_{\rm rel}=1.18\times10^{-1}$, growth $367\times$)"
                        " — causal weight starvation", pad=3)

    # Training loss
    ax_loss = fig.add_subplot(gs[:, 2])
    ax_loss.semilogy(steps_hs, loss_hs["total_loss"], lw=1.5, color="C0",
                     label="Hard-Std (total)")
    ax_loss.semilogy(steps_sc, loss_sc["total_loss"], lw=1.5, color="C3",
                     label="Soft-Causal (total)")
    ax_loss.semilogy(steps_hs, loss_hs["pde_loss"],   lw=1.0, ls="--", color="C0",
                     alpha=0.7, label="Hard-Std (PDE)")
    ax_loss.semilogy(steps_sc, loss_sc["pde_loss"],   lw=1.0, ls="--", color="C3",
                     alpha=0.7, label="Soft-Causal (PDE)")
    ax_loss.set_xlabel("Optimisation step")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training loss — Problem B")
    ax_loss.legend(loc="lower left", framealpha=0.8)
    ax_loss.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_loss.set_xlim(0)

    fig.suptitle("2D Unsteady Heat (Problem B): Hard-Std vs. Soft-Causal temporal pathology",
                 fontsize=10, y=1.01)
    fig.savefig(FIG_DIR / "figure_heat_B.pdf", bbox_inches="tight")
    plt.close()
    print("  figure_heat_B.pdf written")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Burgers 1D
# ─────────────────────────────────────────────────────────────────────────────

def fig_burgers():
    burgers_variants = [
        ("hard-uniform-ff", "Hard-Uniform-FF"),
        ("hard-causal-ff",  "Hard-Causal-FF"),
        ("soft-uniform-ff", "Soft-Uniform-FF"),
        ("hard-causal",     "Hard-Causal"),
        ("hard-uniform",    "Hard-Uniform"),
        ("soft-causal-ff",  "Soft-Causal-FF"),
        ("soft-uniform",    "Soft-Uniform"),
        ("soft-causal",     "Soft-Causal"),
    ]

    rel_l2s, labels_b, colors_b = [], [], []
    for label, disp in burgers_variants:
        v = load_json(BURG_RUN / label / "verification.json")
        rel_l2s.append(v["rel_l2"])
        labels_b.append(disp)
        colors_b.append("#2166ac" if "hard" in label else "#d73027")

    T_SLICES_B = [0.0, 0.25, 0.5, 0.75, 1.0]

    def slice_errs(label):
        v = load_json(BURG_RUN / label / "verification.json")
        return [v["slice_errors"].get(f"t_{t:.2f}", np.nan) for t in T_SLICES_B]

    steps_huf, loss_huf = load_csv_loss(BURG_RUN / "hard-uniform-ff" / "training_log.csv")
    steps_sc,  loss_sc  = load_csv_loss(BURG_RUN / "soft-causal"     / "training_log.csv")
    steps_su,  loss_su  = load_csv_loss(BURG_RUN / "soft-uniform"    / "training_log.csv")

    img_huf = mpimg.imread(BURG_RUN / "hard-uniform-ff" / "solution_comparison.png")
    img_sc  = mpimg.imread(BURG_RUN / "soft-causal"     / "solution_comparison.png")

    fig = plt.figure(figsize=(9.5, 5.0))
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.35,
                             width_ratios=[2.0, 1.5, 1.5])

    # Image panels
    ax_img_huf = fig.add_subplot(gs[0, 0])
    ax_img_huf.imshow(img_huf); ax_img_huf.axis("off")
    ax_img_huf.set_title(r"Hard-Uniform-FF  ($\varepsilon_{\rm rel}=7.01\times10^{-3}$)"
                         " — PINN · FD Reference · |Error|", pad=3)

    ax_img_sc = fig.add_subplot(gs[1, 0])
    ax_img_sc.imshow(img_sc); ax_img_sc.axis("off")
    ax_img_sc.set_title(r"Soft-Causal  ($\varepsilon_{\rm rel}=1.34\times10^{-2}$)"
                        " — worst variant", pad=3)

    # Slice error lines
    ax_sl = fig.add_subplot(gs[:, 1])
    b_line_variants = [
        ("hard-uniform-ff", "Hard-Uniform-FF", "C0",    "-",   2.0),
        ("hard-causal-ff",  "Hard-Causal-FF",  "C0",    "--",  1.2),
        ("soft-uniform-ff", "Soft-Uniform-FF", "C2",    "-",   1.5),
        ("soft-uniform",    "Soft-Uniform",    "C1",    "-",   1.5),
        ("soft-causal-ff",  "Soft-Causal-FF",  "C3",    "--",  1.2),
        ("soft-causal",     "Soft-Causal",     "C3",    "-",   2.0),
    ]
    for (lbl, disp, col, ls, lw) in b_line_variants:
        errs = slice_errs(lbl)
        ax_sl.semilogy(T_SLICES_B, errs, color=col, ls=ls, lw=lw,
                       marker="o", ms=3, label=disp)
    ax_sl.set_xlabel("Time $t$")
    ax_sl.set_ylabel(r"Slice rel-$L^2$ error")
    ax_sl.set_title("Per-slice errors — all Burgers variants")
    ax_sl.legend(fontsize=5.5, loc="upper left", framealpha=0.8)
    ax_sl.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_sl.set_xticks(T_SLICES_B)

    # Bar + loss
    ax_bar = fig.add_subplot(gs[0, 2])
    bars = ax_bar.barh(range(len(labels_b)), rel_l2s, color=colors_b, height=0.55)
    ax_bar.set_xscale("log")
    ax_bar.set_yticks(range(len(labels_b)))
    ax_bar.set_yticklabels(labels_b, fontsize=6)
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel(r"$\varepsilon_{\rm rel}$")
    ax_bar.set_title("All Burgers variants")
    for bar, v in zip(bars, rel_l2s):
        ax_bar.text(v * 1.15, bar.get_y() + bar.get_height()/2,
                    f"{v:.2e}", va="center", fontsize=5)
    ax_bar.grid(True, axis="x", alpha=0.25, lw=0.4)
    from matplotlib.patches import Patch
    ax_bar.legend(handles=[Patch(color="#2166ac", label="Hard BC"),
                            Patch(color="#d73027", label="Soft BC")],
                  loc="lower right", fontsize=6)

    ax_loss = fig.add_subplot(gs[1, 2])
    ax_loss.semilogy(steps_huf, loss_huf["total_loss"], lw=1.5, color="C0",
                     label="Hard-Uniform-FF")
    ax_loss.semilogy(steps_sc,  loss_sc["total_loss"],  lw=1.5, color="C3",
                     label="Soft-Causal")
    ax_loss.semilogy(steps_su,  loss_su["total_loss"],  lw=1.0, ls="--", color="C1",
                     alpha=0.8, label="Soft-Uniform")
    ax_loss.set_xlabel("Step"); ax_loss.set_ylabel("Loss")
    ax_loss.set_title("Training loss")
    ax_loss.legend(fontsize=6, loc="upper right", framealpha=0.8)
    ax_loss.grid(True, which="both", alpha=0.25, lw=0.4)
    ax_loss.set_xlim(0)

    fig.suptitle(r"1D Viscous Burgers ($\nu=0.01/\pi$): all $2^3$ DoE variants",
                 fontsize=10, y=1.01)
    fig.savefig(FIG_DIR / "figure_burgers.pdf", bbox_inches="tight")
    plt.close()
    print("  figure_burgers.pdf written")


if __name__ == "__main__":
    print("Generating figures …")
    fig_heat_A()
    fig_heat_B()
    fig_burgers()
    print("Done.  Output in:", FIG_DIR)
