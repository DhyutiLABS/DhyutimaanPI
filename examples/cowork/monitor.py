#!/usr/bin/env python3
"""
monitor.py — Live terminal monitor for DhyutimaanPI DoE experiments.

Run in a second terminal WHILE experiments are running:
    /Users/rahulsundar/anaconda3/envs/torchmps/bin/python examples/cowork/monitor.py

Refreshes every 5 seconds. Shows:
  • per-variant progress bar + current loss
  • estimated time remaining
  • hypothesis verdicts (as verification.json files appear)

Press Ctrl-C to exit.
"""

import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).parent.parent.parent   # DhyutimaanPI root
HEAT_RUNS  = REPO / "examples/cowork/heat2D/runs"
BURGERS_RUNS = REPO / "examples/cowork/burgers1D/runs"

HEAT_VARIANTS = [
    "soft-adam","soft-lbfgs","hard-adam","hard-lbfgs",
    "soft-adam-ff","soft-lbfgs-ff","hard-adam-ff","hard-lbfgs-ff",
    "soft-std","soft-causal","hard-std","hard-causal",
    "soft-std-ff","soft-causal-ff","hard-std-ff","hard-causal-ff",
]
BURGERS_VARIANTS = [
    "soft-uniform","soft-causal","hard-uniform","hard-causal",
    "soft-uniform-ff","soft-causal-ff","hard-uniform-ff","hard-causal-ff",
    "best-nu-high","best-nu-low",
]
HEAT_STEPS    = {v: 20500 if "lbfgs" in v else 20000 for v in HEAT_VARIANTS}
_HARD_BURGERS = {"hard-uniform","hard-causal","hard-uniform-ff","hard-causal-ff",
                 "best-nu-high","best-nu-low"}
BURGERS_STEPS = {v: 30500 if v in _HARD_BURGERS else 30000 for v in BURGERS_VARIANTS}

CLEAR = "\033[2J\033[H"
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; RESET = "\033[0m"
BOLD  = "\033[1m"; DIM = "\033[2m"

BAR_W = 28


def bar(frac: float, width: int = BAR_W) -> str:
    filled = int(frac * width)
    return "[" + "█" * filled + "░" * (width - filled) + "]"


def read_last_step(log_path: Path) -> tuple:
    """Return (last_step, total_loss, pde_loss, wall_s) from training_log.csv."""
    if not log_path.exists():
        return 0, None, None, 0
    try:
        with open(log_path) as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return 0, None, None, 0
        r = rows[-1]
        return (int(r["step"]), float(r["total_loss"]),
                float(r["pde_loss"]), float(r["wall_s"]))
    except Exception:
        return 0, None, None, 0


def read_verdict(vj_path: Path) -> tuple:
    """Return (rel_l2, done) from verification.json."""
    if not vj_path.exists():
        return None, False
    try:
        with open(vj_path) as f:
            d = json.load(f)
        return d.get("rel_l2") or d.get("relative_l2_error"), True
    except Exception:
        return None, False


def render():
    lines = []
    now = time.time()

    lines.append(f"{BOLD}{'='*72}{RESET}")
    lines.append(f"{BOLD}  DhyutimaanPI DoE Monitor   {time.strftime('%H:%M:%S')}{RESET}")
    lines.append(f"{BOLD}{'='*72}{RESET}")

    def section(title, variants, runs_dir, steps_map):
        lines.append(f"\n{BOLD}{YELLOW}{title}{RESET}")
        lines.append(f"  {'Variant':<22} {'Progress':<32} {'Loss':>10}  {'ε_rel':>10}")
        lines.append("  " + "-"*70)
        for v in variants:
            log  = runs_dir / v / "training_log.csv"
            vj   = runs_dir / v / "verification.json"
            ckpt = runs_dir / v / "checkpoint.pt"
            total_steps = steps_map[v]

            step, tot_loss, pde_loss, wall = read_last_step(log)
            rel_l2, done = read_verdict(vj)

            frac = min(step / total_steps, 1.0)

            if done:
                status = f"{GREEN}✓{RESET}"
                prog   = f"{GREEN}{bar(1.0)}{RESET}"
                l2_str = f"{GREEN}{rel_l2:.2e}{RESET}" if rel_l2 else "---"
            elif step > 0:
                status = f"{YELLOW}▶{RESET}"
                prog   = f"{YELLOW}{bar(frac)}{RESET} {step:>5}/{total_steps}"
                l2_str = "---"
            else:
                status = f"{DIM}·{RESET}"
                prog   = f"{DIM}{bar(0.0)}{RESET} pending"
                l2_str = "---"

            loss_str = f"{tot_loss:.2e}" if tot_loss else "---"
            lines.append(f"  {status} {v:<21} {prog:<40} {loss_str:>10}  {l2_str:>10}")

    section("2D Heat Conduction (16 variants)", HEAT_VARIANTS, HEAT_RUNS, HEAT_STEPS)
    section("1D Burgers (10 variants)", BURGERS_VARIANTS, BURGERS_RUNS, BURGERS_STEPS)

    # Summary
    all_done = []
    all_verts = ([(v, HEAT_RUNS) for v in HEAT_VARIANTS]
                 + [(v, BURGERS_RUNS) for v in BURGERS_VARIANTS])
    n_done = sum(1 for v, d in all_verts if (d / v / "verification.json").exists())
    n_total = len(all_verts)
    pct = 100 * n_done / n_total

    lines.append(f"\n{BOLD}  Overall: {n_done}/{n_total} complete ({pct:.0f}%)  {bar(n_done/n_total, 40)}{RESET}")

    # Show hypothesis verdicts if summaries exist
    for path, label in [(HEAT_RUNS/"hypotheses_summary.json","Heat"),
                        (BURGERS_RUNS/"hypotheses_summary.json","Burgers")]:
        if path.exists():
            try:
                hs = json.load(open(path))
                lines.append(f"\n  {BOLD}Hypotheses [{label}]:{RESET}")
                for hid, v in hs.items():
                    icon = f"{GREEN}✓{RESET}" if v["result"]=="confirmed" else f"{RED}✗{RESET}"
                    lines.append(f"    {icon} {hid}: {v['result'].upper()}  {DIM}{v['evidence']}{RESET}")
            except Exception:
                pass

    lines.append(f"\n{DIM}  Refreshing every 5s — Ctrl-C to quit{RESET}")
    return "\n".join(lines)


def main():
    print("Starting DhyutimaanPI monitor... (Ctrl-C to quit)")
    try:
        while True:
            output = render()
            print(CLEAR + output, flush=True)
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


if __name__ == "__main__":
    main()
