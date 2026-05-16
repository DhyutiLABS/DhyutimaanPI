"""
problem.py — 2D Steady-State Heat Conduction (Laplace equation)

PDE:  ∇²T = T_xx + T_yy = 0   on  Ω = (0,1)²
BCs:  T = 0            on left (x=0), right (x=1), bottom (y=0)
      T = sin(πx)      on top  (y=1)
Ref:  T*(x,y) = sin(πx) · sinh(πy) / sinh(π)

Two residual back-ends:
  • torch_func  — torch.func.hessian + vmap  (preferred; may need CPU on some MPS builds)
  • autograd    — torch.autograd.grad with create_graph=True  (device-agnostic fallback)
"""

import math
import torch
import torch.nn as nn

# ── Domain constants ──────────────────────────────────────────────────────────
X_MIN, X_MAX = 0.0, 1.0
Y_MIN, Y_MAX = 0.0, 1.0
PI = math.pi


# ── Reference solution ────────────────────────────────────────────────────────

def reference_solution(xy: torch.Tensor) -> torch.Tensor:
    """Closed-form: T*(x,y) = sin(πx)·sinh(πy)/sinh(π).

    Args:
        xy: (N, 2) tensor of (x, y) coordinates.
    Returns:
        (N, 1) tensor of analytic temperature values.
    """
    x, y = xy[:, 0], xy[:, 1]
    return (torch.sin(PI * x) * torch.sinh(PI * y) / math.sinh(PI)).unsqueeze(-1)


# ── PDE residual: torch.func back-end ─────────────────────────────────────────

def _scalar_forward(model: nn.Module):
    """Return a function  xy:(2,) → scalar  wrapping the model."""
    def u(xy: torch.Tensor) -> torch.Tensor:
        return model(xy.unsqueeze(0)).squeeze()   # (1,1) → scalar
    return u


def pde_residual_torch_func(model: nn.Module, x_r: torch.Tensor) -> torch.Tensor:
    """Laplace residual ∇²T via torch.func.hessian + vmap.

    Args:
        model: the PINN network.
        x_r:   (N, 2) collocation points (interior).
    Returns:
        (N,) residual values  ∇²T(x_r)  — should be 0 for Laplace.
    """
    u = _scalar_forward(model)

    def laplacian_at(xy: torch.Tensor) -> torch.Tensor:
        H = torch.func.hessian(u)(xy)   # (2, 2) Hessian w.r.t. xy
        return H.diagonal().sum()        # trace = Laplacian

    return torch.func.vmap(laplacian_at)(x_r)   # (N,)


# ── PDE residual: autograd back-end (device-agnostic fallback) ────────────────

def pde_residual_autograd(model: nn.Module, x_r: torch.Tensor) -> torch.Tensor:
    """Laplace residual via torch.autograd.grad with create_graph=True.

    Args:
        model: the PINN network.
        x_r:   (N, 2) collocation points (interior).
    Returns:
        (N,) residual values  ∇²T  — should be 0 for Laplace.
    """
    xr = x_r.clone().requires_grad_(True)
    T = model(xr)                                                    # (N, 1)

    grad_T = torch.autograd.grad(
        T.sum(), xr, create_graph=True
    )[0]                                                             # (N, 2)

    T_xx = torch.autograd.grad(
        grad_T[:, 0].sum(), xr, create_graph=True
    )[0][:, 0]                                                       # (N,)

    T_yy = torch.autograd.grad(
        grad_T[:, 1].sum(), xr, create_graph=True
    )[0][:, 1]                                                       # (N,)

    return T_xx + T_yy                                               # (N,)


# ── Unified dispatcher ────────────────────────────────────────────────────────

def pde_residual(
    model: nn.Module,
    x_r: torch.Tensor,
    use_torch_func: bool = True,
) -> torch.Tensor:
    """Compute Laplace residual, with automatic fallback to autograd."""
    if use_torch_func:
        try:
            return pde_residual_torch_func(model, x_r)
        except Exception as e:
            print(f"[problem] torch.func residual failed ({e}); falling back to autograd.")
    return pde_residual_autograd(model, x_r)


# ── Boundary loss ─────────────────────────────────────────────────────────────

def bc_loss(model: nn.Module, x_bc: torch.Tensor, T_bc: torch.Tensor) -> torch.Tensor:
    """MSE loss on boundary conditions.

    Args:
        model:  PINN network.
        x_bc:   (M, 2) boundary points.
        T_bc:   (M, 1) target temperatures on the boundary.
    Returns:
        scalar MSE.
    """
    return ((model(x_bc) - T_bc) ** 2).mean()


# ── Point samplers ────────────────────────────────────────────────────────────

def sample_interior(n: int, device: torch.device) -> torch.Tensor:
    """Uniform random interior collocation points.

    Returns:
        (n, 2) tensor with x,y ∈ (0,1).
    """
    return torch.rand(n, 2, device=device)


def sample_boundary(n_per_edge: int, device: torch.device):
    """Sample uniformly on all four edges with correct Dirichlet targets.

    Returns:
        x_bc : (4*n_per_edge, 2)
        T_bc : (4*n_per_edge, 1)
    """
    t = torch.linspace(0.0, 1.0, n_per_edge, device=device)

    # Left  x=0 : T=0
    x_left  = torch.stack([torch.zeros_like(t), t], dim=1)
    T_left  = torch.zeros(n_per_edge, 1, device=device)

    # Right x=1 : T=0
    x_right = torch.stack([torch.ones_like(t), t], dim=1)
    T_right = torch.zeros(n_per_edge, 1, device=device)

    # Bottom y=0 : T=0
    x_bot   = torch.stack([t, torch.zeros_like(t)], dim=1)
    T_bot   = torch.zeros(n_per_edge, 1, device=device)

    # Top y=1 : T=sin(πx)
    x_top   = torch.stack([t, torch.ones_like(t)], dim=1)
    T_top   = torch.sin(PI * t).unsqueeze(1)

    x_bc = torch.cat([x_left, x_right, x_bot, x_top], dim=0)   # (4N, 2)
    T_bc = torch.cat([T_left, T_right, T_bot, T_top], dim=0)   # (4N, 1)

    return x_bc, T_bc
