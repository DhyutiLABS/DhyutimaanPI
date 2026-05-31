"""
run.py — Top-level entry point for 1D Burgers PINN DoE.

Usage:
    python run.py --variant hard-causal         # single variant
    python run.py --variant all                 # all 10 variants
    python run.py --variant all --verify_only   # re-verify existing checkpoints

Outputs per variant: examples/cowork/burgers1D/runs/<label>/
  training_log.csv  checkpoint.pt  verification.json
  loss_curve.png    solution_comparison.png
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

RUNS_DIR = HERE.parent / "runs"


def main():
    parser = argparse.ArgumentParser(description="1D Burgers PINN DoE runner")
    parser.add_argument("--variant",     default="all")
    parser.add_argument("--verify_only", action="store_true")
    parser.add_argument("--runs_dir",    default=str(RUNS_DIR))
    args = parser.parse_args()

    from train  import VARIANTS, train_variant
    from verify import verify_variant, evaluate_hypotheses

    runs_root = Path(args.runs_dir)
    runs_root.mkdir(parents=True, exist_ok=True)

    if args.variant == "all":
        labels = list(VARIANTS.keys())
    else:
        if args.variant not in VARIANTS:
            print(f"Unknown variant '{args.variant}'.")
            print(f"Available: {list(VARIANTS.keys())}")
            sys.exit(1)
        labels = [args.variant]

    print(f"\n{'='*64}")
    print(f"1D Burgers PINN DoE  —  {len(labels)} variant(s)")
    print(f"Runs dir: {runs_root}")
    print(f"{'='*64}")

    t_wall = time.time()

    for label in labels:
        out_dir = runs_root / label
        if not args.verify_only:
            train_variant(label, out_dir)
        verify_variant(label, runs_root)

    # Evaluate all hypotheses
    verdicts = evaluate_hypotheses(runs_root)
    summary_path = runs_root / "hypotheses_summary.json"
    with open(summary_path, "w") as f:
        json.dump(verdicts, f, indent=2)

    print(f"\n{'='*64}")
    print(f"All done in {time.time()-t_wall:.1f}s")
    print("\nHypothesis verdicts:")
    for hid, v in verdicts.items():
        icon = "✓" if v["result"] == "confirmed" else "✗"
        print(f"  {icon} {hid}: {v['result'].upper()}  —  {v['evidence']}")
    print(f"\nSummary → {summary_path}")


if __name__ == "__main__":
    main()
