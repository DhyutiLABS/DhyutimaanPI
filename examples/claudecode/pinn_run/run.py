"""
run.py — Top-level entry point for the 2D Heat PINN.

Usage examples
--------------
# Standard full run (10k Adam + 2k L-BFGS, 4096 collocation pts):
    python pinn_run/run.py

# Quick smoke test (200 Adam steps, 256 pts — confirms no crashes):
    python pinn_run/run.py --smoke-test

# Collocation ablation (256 → 1024 → 4096 → 16384, tests H3):
    python pinn_run/run.py --sweep-collocation

# Disable L-BFGS (Adam-only checkpoint for H2):
    python pinn_run/run.py --no-lbfgs

# Force autograd residual (skip torch.func, works on all MPS builds):
    python pinn_run/run.py --no-torch-func

# Full hyperparameter control:
    python pinn_run/run.py --n-collocation 8192 --n-adam 20000 --w-bc 20.0
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from train import train, select_device
from verify import verify


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="2D Steady-State Heat PINN — train and verify",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Training knobs ─────────────────────────────────────────────────────
    p.add_argument("--n-collocation", type=int, default=4096,
                   help="Interior collocation points")
    p.add_argument("--n-per-edge",    type=int, default=100,
                   help="Boundary points per edge (×4 total)")
    p.add_argument("--n-adam",        type=int, default=10_000,
                   help="Adam iterations")
    p.add_argument("--n-lbfgs",       type=int, default=2_000,
                   help="L-BFGS max iterations (Stage 2)")
    p.add_argument("--w-pde",         type=float, default=1.0,
                   help="PDE residual loss weight")
    p.add_argument("--w-bc",          type=float, default=10.0,
                   help="BC loss weight")
    p.add_argument("--lr-adam",       type=float, default=1e-3)
    p.add_argument("--hidden",        type=int,   default=64)
    p.add_argument("--depth",         type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--log-every",     type=int,   default=100,
                   help="Print/log interval (iterations)")

    # ── Mode flags ─────────────────────────────────────────────────────────
    p.add_argument("--no-lbfgs",       action="store_true",
                   help="Adam-only training (skip L-BFGS stage 2)")
    p.add_argument("--no-torch-func",  action="store_true",
                   help="Use torch.autograd residual instead of torch.func")
    p.add_argument("--smoke-test",     action="store_true",
                   help="200 Adam steps + 0 L-BFGS with 256 pts; checks for crashes")
    p.add_argument("--sweep-collocation", action="store_true",
                   help="Ablation: train+verify at 256/1024/4096/16384 collocation pts")
    p.add_argument("--verify-only",    action="store_true",
                   help="Skip training; run verification on existing checkpoint")

    return p.parse_args()


def run_single(args, n_collocation=None, run_name=None, device=None) -> dict:
    """Train + verify with given args. Overrides n_collocation/run_name if provided."""
    nc             = n_collocation or args.n_collocation
    n_lbfgs        = 0 if args.no_lbfgs else args.n_lbfgs
    use_torch_func = not args.no_torch_func

    _, output_dir = train(
        n_collocation  = nc,
        n_per_edge     = args.n_per_edge,
        n_adam         = args.n_adam,
        n_lbfgs        = n_lbfgs,
        w_pde          = args.w_pde,
        w_bc           = args.w_bc,
        lr_adam        = args.lr_adam,
        hidden         = args.hidden,
        depth          = args.depth,
        device         = device,
        use_torch_func = use_torch_func,
        seed           = args.seed,
        log_every      = args.log_every,
        run_name       = run_name,
    )
    return verify(device=device, use_torch_func=use_torch_func, output_dir=output_dir)


def main() -> None:
    args   = parse_args()
    device = select_device()

    # ── Smoke test ─────────────────────────────────────────────────────────
    if args.smoke_test:
        print("\n" + "═" * 55)
        print("  SMOKE TEST  (200 Adam steps, 256 pts, no L-BFGS)")
        print("═" * 55)
        import copy
        smoke_args        = copy.copy(args)
        smoke_args.n_adam = 200
        smoke_args.no_lbfgs = True
        result = run_single(smoke_args, n_collocation=256, run_name="smoke", device=device)
        status = "PASS" if result["success_criteria_met"] else "expected-fail (200 steps insufficient)"
        print(f"\nSmoke test completed — {status}.")
        return

    # ── Verify-only ────────────────────────────────────────────────────────
    if args.verify_only:
        print("\n[run] Skipping training; verifying existing checkpoint.")
        print("[run] Use --run-name to specify which output dir to verify (default: outputs/).")
        verify(device=device, use_torch_func=not args.no_torch_func)
        return

    # ── Collocation sweep ──────────────────────────────────────────────────
    if args.sweep_collocation:
        sweep_counts = [256, 1024, 4096, 16384]
        print(f"\n[run] Collocation sweep: {sweep_counts}")
        sweep_results = []
        for nc in sweep_counts:
            print(f"\n{'═'*55}\n  N_r = {nc}\n{'═'*55}")
            result = run_single(args, n_collocation=nc, run_name=f"sweep_nr{nc}", device=device)
            sweep_results.append({"n_collocation": nc, **result})
            print(f"  → rel_l2 = {result['relative_l2_error']:.4e}")

        sweep_path = Path(__file__).parent / "outputs" / "sweep_collocation.json"
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2)
        print(f"\n[run] Sweep results → {sweep_path}")

        # Print summary table
        print("\nCollocation sweep summary:")
        print(f"  {'N_r':>6}  {'rel_L2':>10}  {'max_err':>10}  pass?")
        print(f"  {'─'*6}  {'─'*10}  {'─'*10}  {'─'*5}")
        for r in sweep_results:
            print(
                f"  {r['n_collocation']:>6}  "
                f"{r['relative_l2_error']:>10.3e}  "
                f"{r['max_pointwise_error']:>10.3e}  "
                f"{'yes' if r['success_criteria_met'] else 'no'}"
            )
        return

    # ── Standard run ──────────────────────────────────────────────────────
    print("\n" + "═" * 55)
    print("  2D Heat PINN — standard run")
    print("═" * 55)
    n_lbfgs = 0 if args.no_lbfgs else args.n_lbfgs
    lbfgs_tag = f"_lbfgs{n_lbfgs}" if n_lbfgs > 0 else "_nolbfgs"
    run_name = f"standard_nr{args.n_collocation}_adam{args.n_adam}{lbfgs_tag}"
    result = run_single(args, run_name=run_name, device=device)

    if result["success_criteria_met"]:
        print("\nAll success criteria met. Ready for analysis skill.")
    else:
        print("\nSuccess criteria NOT met. Check pinn_run/outputs/verification.json for details.")


if __name__ == "__main__":
    main()
