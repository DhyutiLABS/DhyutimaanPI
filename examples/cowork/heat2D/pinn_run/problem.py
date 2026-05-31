"""
problem.py — 2D Heat Conduction PINN (Steady + Unsteady)
Problem A: -Δu = 2π²sin(πx)sin(πy),  u=0 on ∂Ω
Problem B:  u_t - α(u_xx+u_yy) = 0,   u(x,y,0)=sin(πx)sin(πy), u=0 on ∂Ω×[0,T]
Reference:
  A: u_ref(x,y)   = sin(πx)sin(πy)
  B: u_ref(x,y,t) = sin(πx)sin(πy)*exp(-2π²αt)

PDE residuals use automatic differentiation (torch.autograd.grad) for exact
second-order derivatives — FD Laplacian gave near-zero gradient signal.
"""

import math
import torch
from torch.autograd import grad as _ag

# ── Domain constants ──────────────────────────────────────────────────────────
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 0.1
ALPHA = 1.0          # thermal diffusivity for Problem B
FD_H  = 1e-2         # kept for reference; no longer used in training


# ── Reference solutions ───────────────────────────────────────────────────────

def ref_A(xy: torch.Tensor) -> torch.Tensor:
    """u_ref(x,y) = sin(πx)sin(πy)"""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return torch.sin(math.pi * x) * torch.sin(math.pi * y)


def ref_B(xyt: torch.Tensor) -> torch.Tensor:
    """u_ref(x,y,t) = sin(πx)sin(πy)*exp(-2π²αt)"""
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    return (torch.sin(math.pi * x) * torch.sin(math.pi * y)
            * torch.exp(-2.0 * math.pi**2 * ALPHA * t))


def source_A(xy: torch.Tensor) -> torch.Tensor:
    """f(x,y) = 2π²sin(πx)sin(πy)"""
    x, y = xy[:, 0:1], xy[:, 1:2]
    return 2.0 * math.pi**2 * torch.sin(math.pi * x) * torch.sin(math.pi * y)


# ── Hard-constraint ansatze ────────────────────────────────────────────────────

def apply_hard_A(nn_out: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
    """u = x(1-x)y(1-y)·NN — enforces zero Dirichlet on all sides."""
    x, y = xy[:, 0:1], xy[:, 1:2]
    envelope = x * (1.0 - x) * y * (1.0 - y)
    return envelope * nn_out


def apply_hard_B(nn_out: torch.Tensor, xyt: torch.Tensor) -> torch.Tensor:
    """u = sin(πx)sin(πy) + t·x(1-x)y(1-y)·NN — enforces IC and BCs."""
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    ic_part = torch.sin(math.pi * x) * torch.sin(math.pi * y)
    envelope = t * x * (1.0 - x) * y * (1.0 - y)
    return ic_part + envelope * nn_out


# ── FD-based Laplacian (5-point stencil) ─────────────────────────────────────

def fd_laplacian_A(model, xy: torch.Tensor, h: float = FD_H) -> torch.Tensor:
    """
    Δu ≈ [u(x+h,y) + u(x-h,y) + u(x,y+h) + u(x,y-h) - 4u(x,y)] / h²
    Neighbour points are clamped to [X_MIN+ε, X_MAX-ε] to stay in domain.
    """
    eps = 1e-6
    x, y = xy[:, 0:1], xy[:, 1:2]

    def _u(pt):
        return model(pt)

    xph = torch.cat([torch.clamp(x + h, X_MIN + eps, X_MAX - eps), y], dim=1)
    xmh = torch.cat([torch.clamp(x - h, X_MIN + eps, X_MAX - eps), y], dim=1)
    yph = torch.cat([x, torch.clamp(y + h, Y_MIN + eps, Y_MAX - eps)], dim=1)
    ymh = torch.cat([x, torch.clamp(y - h, Y_MIN + eps, Y_MAX - eps)], dim=1)

    lap = (_u(xph) + _u(xmh) + _u(yph) + _u(ymh) - 4.0 * _u(xy)) / (h * h)
    return lap


def fd_laplacian_B(model, xyt: torch.Tensor, h: float = FD_H) -> torch.Tensor:
    """Laplacian in (x,y) for time-dependent model with inputs (x,y,t)."""
    eps = 1e-6
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]

    def _u(pt):
        return model(pt)

    xph = torch.cat([torch.clamp(x + h, X_MIN + eps, X_MAX - eps), y, t], dim=1)
    xmh = torch.cat([torch.clamp(x - h, X_MIN + eps, X_MAX - eps), y, t], dim=1)
    yph = torch.cat([x, torch.clamp(y + h, Y_MIN + eps, Y_MAX - eps), t], dim=1)
    ymh = torch.cat([x, torch.clamp(y - h, Y_MIN + eps, Y_MAX - eps), t], dim=1)

    lap = (_u(xph) + _u(xmh) + _u(yph) + _u(ymh) - 4.0 * _u(xyt)) / (h * h)
    return lap


def fd_dt_B(model, xyt: torch.Tensor, dt: float = 1e-3) -> torch.Tensor:
    """Forward-difference time derivative: (u(t+dt) - u(t)) / dt."""
    x, y, t = xyt[:, 0:1], xyt[:, 1:2], xyt[:, 2:3]
    t_fwd = torch.clamp(t + dt, T_MIN, T_MAX)
    xyt_fwd = torch.cat([x, y, t_fwd], dim=1)
    return (model(xyt_fwd) - model(xyt)) / dt


# ── AD-based PDE residuals ────────────────────────────────────────────────────
# Uses torch.autograd.grad for exact second-order derivatives.
# Each function internally creates a requires_grad copy of the input so the
# training loop can pass plain tensors and still receive gradients w.r.t.
# model parameters.

def pde_residual_A(model, xy: torch.Tensor) -> torch.Tensor:
    """
    Residual of -Δu - f = 0  (steady Poisson, Problem A).
    Computes u_xx + u_yy via AD; returns shape (N, 1).
    """
    xy_g = xy.detach().requires_grad_(True)
    u    = model(xy_g)                                      # (N, 1)

    # ∂u/∂x and ∂u/∂y in one pass
    grad_u = _ag(u, xy_g,
                 grad_outputs=torch.ones_like(u),
                 create_graph=True)[0]                      # (N, 2)
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]

    # ∂²u/∂x² — retain_graph so the shared graph is still alive for u_yy
    u_xx = _ag(u_x, xy_g,
               grad_outputs=torch.ones_like(u_x),
               create_graph=True,
               retain_graph=True)[0][:, 0:1]
    # ∂²u/∂y²
    u_yy = _ag(u_y, xy_g,
               grad_outputs=torch.ones_like(u_y),
               create_graph=True)[0][:, 1:2]

    lap = u_xx + u_yy
    return -lap - source_A(xy_g)


def pde_residual_B(model, xyt: torch.Tensor) -> torch.Tensor:
    """
    Residual of u_t - α·Δ_{xy}u = 0  (unsteady heat, Problem B).
    Returns shape (N, 1).
    """
    xyt_g = xyt.detach().requires_grad_(True)
    u     = model(xyt_g)                                    # (N, 1)

    # First derivatives: [u_x, u_y, u_t]
    grad_u = _ag(u, xyt_g,
                 grad_outputs=torch.ones_like(u),
                 create_graph=True)[0]                      # (N, 3)
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_t = grad_u[:, 2:3]

    # Second spatial derivatives — retain_graph on first call
    u_xx = _ag(u_x, xyt_g,
               grad_outputs=torch.ones_like(u_x),
               create_graph=True,
               retain_graph=True)[0][:, 0:1]
    u_yy = _ag(u_y, xyt_g,
               grad_outputs=torch.ones_like(u_y),
               create_graph=True)[0][:, 1:2]

    lap = u_xx + u_yy
    return u_t - ALPHA * lap


# ── Boundary / initial losses ─────────────────────────────────────────────────

def bc_loss_A(model, xy_bc: torch.Tensor) -> torch.Tensor:
    """Soft BC: MSE of u on ∂Ω against 0."""
    return (model(xy_bc) ** 2).mean()


def bc_loss_B(model, xyt_bc: torch.Tensor) -> torch.Tensor:
    """Soft BC: MSE of u on ∂Ω×[0,T] against 0."""
    return (model(xyt_bc) ** 2).mean()


def ic_loss_B(model, xy_ic: torch.Tensor) -> torch.Tensor:
    """Soft IC: MSE of u(x,y,0) against sin(πx)sin(πy)."""
    t0 = torch.zeros(xy_ic.shape[0], 1, device=xy_ic.device)
    xyt0 = torch.cat([xy_ic, t0], dim=1)
    u_pred = model(xyt0)
    u_true = ref_B(xyt0)
    return ((u_pred - u_true) ** 2).mean()


# ── Collocation samplers ──────────────────────────────────────────────────────

def sample_interior_A(n: int, device: torch.device) -> torch.Tensor:
    """LHS-like (uniform random) interior points in (0,1)²."""
    pts = torch.rand(n, 2, device=device)
    pts[:, 0] = pts[:, 0] * (X_MAX - X_MIN) + X_MIN
    pts[:, 1] = pts[:, 1] * (Y_MAX - Y_MIN) + Y_MIN
    return pts


def sample_boundary_A(n_per_side: int, device: torch.device) -> torch.Tensor:
    """100 points per side × 4 sides on ∂[0,1]²."""
    t = torch.linspace(0.0, 1.0, n_per_side, device=device)
    sides = [
        torch.stack([torch.zeros_like(t), t], dim=1),   # x=0
        torch.stack([torch.ones_like(t),  t], dim=1),   # x=1
        torch.stack([t, torch.zeros_like(t)], dim=1),   # y=0
        torch.stack([t, torch.ones_like(t)],  dim=1),   # y=1
    ]
    return torch.cat(sides, dim=0)


def sample_interior_B(n: int, device: torch.device) -> torch.Tensor:
    """Random interior points in (0,1)²×(0,0.1]."""
    pts = torch.rand(n, 3, device=device)
    pts[:, 0] = pts[:, 0] * (X_MAX - X_MIN) + X_MIN
    pts[:, 1] = pts[:, 1] * (Y_MAX - Y_MIN) + Y_MIN
    pts[:, 2] = pts[:, 2] * (T_MAX - T_MIN) + T_MIN
    return pts


def sample_boundary_B(n_per_side: int, n_t: int, device: torch.device) -> torch.Tensor:
    """BC points on ∂[0,1]² × uniform time grid."""
    ts = torch.linspace(T_MIN, T_MAX, n_t, device=device)
    sp = torch.linspace(0.0, 1.0, n_per_side, device=device)
    all_pts = []
    for t_val in ts:
        t_col = torch.full((n_per_side,), t_val.item(), device=device)
        all_pts.append(torch.stack([torch.zeros(n_per_side, device=device), sp, t_col], 1))
        all_pts.append(torch.stack([torch.ones(n_per_side,  device=device), sp, t_col], 1))
        all_pts.append(torch.stack([sp, torch.zeros(n_per_side, device=device), t_col], 1))
        all_pts.append(torch.stack([sp, torch.ones(n_per_side,  device=device), t_col], 1))
    return torch.cat(all_pts, dim=0)


def sample_ic_B(n: int, device: torch.device) -> torch.Tensor:
    """IC points: random (x,y) in (0,1)²."""
    xy = torch.rand(n, 2, device=device)
    xy[:, 0] = xy[:, 0] * (X_MAX - X_MIN) + X_MIN
    xy[:, 1] = xy[:, 1] * (Y_MAX - Y_MIN) + Y_MIN
    return xy
