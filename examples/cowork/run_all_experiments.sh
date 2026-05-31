#!/usr/bin/env bash
# ============================================================
# run_all_experiments.sh
# Master runner for DhyutimaanPI CAISc 2026 experiments.
# Uses conda environment: torchmps
#
# Usage (from repo root):
#   chmod +x examples/cowork/run_all_experiments.sh
#   ./examples/cowork/run_all_experiments.sh
#
# To run a single variant:
#   HEAT_VARIANT=hard-adam BURGERS_VARIANT=soft-causal ./examples/cowork/run_all_experiments.sh
# ============================================================

set -euo pipefail

PYTHON="/Users/rahulsundar/anaconda3/envs/torchmps/bin/python"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HEAT_DIR="$REPO_ROOT/examples/cowork/heat2D/pinn_run"
BURGERS_DIR="$REPO_ROOT/examples/cowork/burgers1D/pinn_run"

echo "============================================================"
echo "DhyutimaanPI CAISc 2026 Experiment Runner"
echo "Python : $PYTHON"
echo "Repo   : $REPO_ROOT"
echo "Started: $(date)"
echo "============================================================"

# ── Heat 2D DoE ──────────────────────────────────────────────
echo ""
echo ">>> Heat 2D: running ${HEAT_VARIANT:-all} variants ..."
cd "$HEAT_DIR"
"$PYTHON" run.py --variant "${HEAT_VARIANT:-all}" \
                 --runs_dir "$REPO_ROOT/examples/cowork/heat2D/runs"

# ── Burgers 1D DoE ───────────────────────────────────────────
echo ""
echo ">>> Burgers 1D: running ${BURGERS_VARIANT:-all} variants ..."
cd "$BURGERS_DIR"
"$PYTHON" run.py --variant "${BURGERS_VARIANT:-all}" \
                 --runs_dir "$REPO_ROOT/examples/cowork/burgers1D/runs"

echo ""
echo "============================================================"
echo "All experiments complete: $(date)"
echo "Results in:"
echo "  $REPO_ROOT/examples/cowork/heat2D/runs/"
echo "  $REPO_ROOT/examples/cowork/burgers1D/runs/"
echo "============================================================"
