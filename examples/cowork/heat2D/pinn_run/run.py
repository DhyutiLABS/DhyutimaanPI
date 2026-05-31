"""
run.py — Top-level entry point for 2D Heat PINN DoE.

Usage:
    # Run a single variant
    python run.py --variant hard-adam

    # Run all 16 variants sequentially
    python run.py --variant all

    # Run only Problem A variants
    python run.py --variant all_A

    # Run only Problem B variants
    python run.py --variant all_B

    # Verify only (skip training, load existing checkpoints)
    python run.py --variant all --verify_only

Outputs are written to:
    examples/cowork/heat2D/runs/<label>/
"""

import argparse
import sys
import time
from pathlib import Path

# Resolve project root so imports work from any CWD
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

RUNS_DIR = HERE.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description="2D Heat PINN DoE runner")
    parser.add_argument("--variant", default="all",
                        help="Variant label, 'all', 'all_A', or 'all_B'")
    parser.add_argument("--verify_only", action="store_true",
                        help="Skip training; only run verification on existing checkpoints")
    parser.add_argument("--runs_dir", default=str(RUNS_DIR),
                        help="Root directory for run outputs")
    args = parser.parse_args()

    from train import VARIANTS, train_variant
    from verify import verify_variant, evaluate_hypotheses

    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    # ── Select labels ──────────────────────────────────────────────────────
    if args.variant == "all":
        labels = list(VARIANTS.keys())
    elif args.variant == "all_A":
        labels = [k for k, v in VARIANTS.items() if v[0] == "A"]
    elif args.variant == "all_B":
        labels = [k for k, v in VARIANTS.items() if v[0] == "B"]
    else:
        if args.variant not in VARIANTS:
            print(f"Unknown variant '{args.variant}'. "
                  f"Available: {list(VARIANTS.keys())}")
            sys.exit(1)
        labels = [args.variant]

    print(f"\n{'='*60}")
    print(f"2D Heat PINN DoE  —  {len(labels)} variant(s)")
    print(f"Runs dir: {runs_root}")
    print(f"{'='*60}")

    t_wall = time.time()

    for label in labels:
        out_dir = runs_root / label
        if not args.verify_only:
            train_variant(label, out_dir)
        verify_variant(label, runs_root)

    # ── Hypothesis evaluation ──────────────────────────────────────────────
    verdicts = evaluate_hypotheses(runs_root)
    summary_path = runs_root / "hypotheses_summary.json"
    import json
    with open(summary_path, "w") as f:
        json.dump(verdicts, f, indent=2)

    print(f"\n{'='*60}")
    print(f"All done in {time.time()-t_wall:.1f}s")
    print(f"\nHypotheses summary:")
    for hid, v in verdicts.items():
        status = "✓ CONFIRMED" if v["result"] == "confirmed" else "✗ REFUTED"
        print(f"  {hid}: {status}  —  {v['evidence']}")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
