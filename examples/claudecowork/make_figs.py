"""Generate figures for the PINN 2D heat conduction report."""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUNS = "runs"
FIGS = "figs"
os.makedirs(FIGS, exist_ok=True)

VARIANTS_A = ["A1", "A2", "A3", "A4"]
VARIANTS_B = ["B1", "B2", "B3"]
LABELS = {
    "A1": "A1 soft, $w_{bc}=1$",
    "A2": "A2 soft, $w_{bc}=100$",
    "A3": "A3 hard constraint",
    "A4": "A4 hard + Fourier",
    "B1": "B1 soft, $w_{ic}=10$, $w_{bc}=1$",
    "B2": "B2 soft, $w_{ic}=w_{bc}=100$",
    "B3": "B3 hard constraint",
}


def load_log(v):
    path = os.path.join(RUNS, v, "training_log.csv")
    if not os.path.exists(path):
        return None
    arr = np.genfromtxt(path, delimiter=",", names=True)
    return arr


def fig_loss_curves():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for v in VARIANTS_A:
        a = load_log(v)
        if a is None: continue
        axes[0].semilogy(a["step"], a["rel_l2"], label=LABELS[v])
    axes[0].set_xlabel("Adam iteration"); axes[0].set_ylabel("Relative $L^2$ error")
    axes[0].set_title("Problem A (steady): error vs. iteration")
    axes[0].legend(loc="upper right", fontsize=9); axes[0].grid(True, which="both", alpha=0.3)

    for v in VARIANTS_B:
        a = load_log(v)
        if a is None: continue
        axes[1].semilogy(a["step"], a["rel_l2"], label=LABELS[v])
    axes[1].set_xlabel("Adam iteration"); axes[1].set_ylabel("Relative $L^2$ error")
    axes[1].set_title("Problem B (unsteady): error vs. iteration")
    axes[1].legend(loc="upper right", fontsize=9); axes[1].grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "loss_curves.png"), dpi=150)
    plt.close()


def fig_steady_solution():
    """Heatmaps: reference, A3 prediction, A3 abs error, A1 abs error."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    ref = np.load(os.path.join(RUNS, "A3", "reference.npy"))
    A3 = np.load(os.path.join(RUNS, "A3", "prediction.npy"))
    A1 = np.load(os.path.join(RUNS, "A1", "prediction.npy"))

    extent = [0, 1, 0, 1]
    im0 = axes[0, 0].imshow(ref.T, origin="lower", extent=extent, cmap="viridis")
    axes[0, 0].set_title("Reference $u_{ref}=\\sin(\\pi x)\\sin(\\pi y)$")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(A3.T, origin="lower", extent=extent, cmap="viridis")
    axes[0, 1].set_title("A3 hard-constraint prediction")
    plt.colorbar(im1, ax=axes[0, 1])

    err1 = np.abs(A1 - ref)
    im2 = axes[1, 0].imshow(err1.T, origin="lower", extent=extent, cmap="magma")
    axes[1, 0].set_title(f"A1 |error| (max {err1.max():.3f})")
    plt.colorbar(im2, ax=axes[1, 0])

    err3 = np.abs(A3 - ref)
    im3 = axes[1, 1].imshow(err3.T, origin="lower", extent=extent, cmap="magma")
    axes[1, 1].set_title(f"A3 |error| (max {err3.max():.2e})")
    plt.colorbar(im3, ax=axes[1, 1])

    for ax in axes.ravel():
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "steady_solution.png"), dpi=150)
    plt.close()


def fig_unsteady_slices():
    """Final-time slice: reference vs B2 vs B3."""
    ref = np.load(os.path.join(RUNS, "B2", "reference.npy"))   # (51,51,11)
    B2 = np.load(os.path.join(RUNS, "B2", "prediction.npy"))
    B1 = np.load(os.path.join(RUNS, "B1", "prediction.npy"))
    B3 = np.load(os.path.join(RUNS, "B3", "prediction.npy"))

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    extent = [0, 1, 0, 1]
    # top row: solutions at t=T
    im0 = axes[0, 0].imshow(ref[:, :, -1].T, origin="lower", extent=extent, cmap="viridis")
    axes[0, 0].set_title("Reference at $t=T$")
    plt.colorbar(im0, ax=axes[0, 0])

    im1 = axes[0, 1].imshow(B2[:, :, -1].T, origin="lower", extent=extent, cmap="viridis")
    axes[0, 1].set_title("B2 prediction at $t=T$")
    plt.colorbar(im1, ax=axes[0, 1])

    im2 = axes[0, 2].imshow(B3[:, :, -1].T, origin="lower", extent=extent, cmap="viridis")
    axes[0, 2].set_title("B3 prediction at $t=T$")
    plt.colorbar(im2, ax=axes[0, 2])

    # bottom: error vs t
    t_grid = np.linspace(0, 0.1, 11)
    rmse_t = lambda U: np.sqrt(np.mean((U - ref) ** 2, axis=(0, 1)))
    rel_t = lambda U: np.linalg.norm((U - ref).reshape(-1, 11), axis=0) / np.linalg.norm(ref.reshape(-1, 11), axis=0)
    axes[1, 0].plot(t_grid, rel_t(B1), "o-", label="B1")
    axes[1, 0].plot(t_grid, rel_t(B2), "s-", label="B2")
    axes[1, 0].plot(t_grid, rel_t(B3), "d-", label="B3")
    axes[1, 0].set_xlabel("$t$"); axes[1, 0].set_ylabel("Relative $L^2$ at slice")
    axes[1, 0].set_title("Error growth in time"); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].legend()

    err = np.abs(B2[:, :, -1] - ref[:, :, -1])
    im4 = axes[1, 1].imshow(err.T, origin="lower", extent=extent, cmap="magma")
    axes[1, 1].set_title(f"B2 |error| at $t=T$ (max {err.max():.3f})")
    plt.colorbar(im4, ax=axes[1, 1])

    err = np.abs(B3[:, :, -1] - ref[:, :, -1])
    im5 = axes[1, 2].imshow(err.T, origin="lower", extent=extent, cmap="magma")
    axes[1, 2].set_title(f"B3 |error| at $t=T$ (max {err.max():.3f})")
    plt.colorbar(im5, ax=axes[1, 2])

    for ax in axes[0, :]:
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    for ax in axes[1, 1:]:
        ax.set_xlabel("$x$"); ax.set_ylabel("$y$")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "unsteady_solution.png"), dpi=150)
    plt.close()


def fig_summary_bar():
    """Bar chart of final relative L2 errors."""
    rel = []
    labels = []
    for v in VARIANTS_A + VARIANTS_B:
        d = json.load(open(os.path.join(RUNS, v, "verification.json")))
        rel.append(d["rel_l2_final"])
        labels.append(v)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#aaa", "#66c", "#3a3", "#c33", "#aaa", "#66c", "#3a3"]
    ax.bar(labels, rel, color=colors)
    ax.set_yscale("log")
    ax.set_ylabel("Final relative $L^2$ error")
    ax.set_title("All variants: final accuracy on held-out 101$\\times$101 reference grid")
    ax.grid(True, axis="y", which="both", alpha=0.3)
    for i, r in enumerate(rel):
        ax.text(i, r * 1.15, f"{r:.1e}", ha="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS, "summary_bar.png"), dpi=150)
    plt.close()


if __name__ == "__main__":
    fig_loss_curves()
    fig_steady_solution()
    fig_unsteady_slices()
    fig_summary_bar()
    print("figures written to", FIGS)
