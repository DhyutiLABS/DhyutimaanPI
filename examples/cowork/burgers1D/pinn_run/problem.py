"""
problem.py — 1D Nonlinear Viscous Burgers PINN
PDE: u_t + u*u_x - ν*u_xx = 0,  x∈[-1,1], t∈[0,1]
IC:  u(x,0) = -sin(πx)
BC:  u(-1,t) = u(1,t) = 0

Reference: high-fidelity FD (2nd-order upwind + RK4, N=512, dt=1e-4)
"""

import math
import numpy as np
import torch
import torch.nn as nn

# ── Domain constants ──────────────────────────────────────────────────────────
X_MIN, X_MAX = -1.0, 1.0
T_MIN, T_MAX =  0.0, 1.0
NU_DEFAULT   = 0.01 / math.pi       # ≈ 0.00318  (standard Raissi 2019)

# ── IC / BC functions ─────────────────────────────────────────────────────────

def ic_fn(x: torch.Tensor) -> torch.Tensor:
    """u(x,0) = -sin(πx)"""
    return -torch.sin(math.pi * x)


# ── Hard-constraint ansatz ────────────────────────────────────────────────────

def apply_hard(nn_raw: torch.Tensor, xt: torch.Tensor) -> torch.Tensor:
    """
    u(x,t) = -sin(πx) + t*(1-x²)*NN(x,t)
    Satisfies:
      u(x,0)  = -sin(πx)   exactly (IC)
      u(±1,t) = -sin(±π) + t*0*NN = 0  exactly (BC)
    """
    x = xt[:, 0:1]
    t = xt[:, 1:2]
    ic_part  = -torch.sin(math.pi * x)
    envelope = t * (1.0 - x * x)
    return ic_part + envelope * nn_raw


# ── AD-based PDE residual ─────────────────────────────────────────────────────

def pde_residual(model: nn.Module, xt: torch.Tensor, nu: float) -> torch.Tensor:
    """
    r = u_t + u*u_x - ν*u_xx
    Uses torch.autograd with create_graph=True for higher-order derivatives.
    """
    xt = xt.requires_grad_(True)
    u  = model(xt)                           # (N,1)

    # First-order derivatives
    grads1 = torch.autograd.grad(
        u, xt,
        grad_outputs=torch.ones_like(u),
        create_graph=True, retain_graph=True
    )[0]                                     # (N,2) — [du/dx, du/dt]
    u_x = grads1[:, 0:1]
    u_t = grads1[:, 1:2]

    # Second-order spatial derivative
    u_xx = torch.autograd.grad(
        u_x, xt,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True, retain_graph=True
    )[0][:, 0:1]

    return u_t + u * u_x - nu * u_xx


# ── Soft BC / IC losses ───────────────────────────────────────────────────────

def soft_ic_loss(model: nn.Module, x_ic: torch.Tensor) -> torch.Tensor:
    """MSE of u(x,0) vs -sin(πx)."""
    t0  = torch.zeros(x_ic.shape[0], 1, device=x_ic.device)
    xt0 = torch.cat([x_ic, t0], dim=1)
    u_p = model(xt0)
    u_r = ic_fn(x_ic)
    return ((u_p - u_r) ** 2).mean()


def soft_bc_loss(model: nn.Module, xt_bc: torch.Tensor) -> torch.Tensor:
    """MSE of u at x=±1 vs 0."""
    return (model(xt_bc) ** 2).mean()


# ── Causal weighting (Wang, Sankaran, Perdikaris 2022) ───────────────────────

def causal_pde_loss(pde_res: torch.Tensor, t_vals: torch.Tensor,
                    n_slabs: int = 50, eps: float = 100.0) -> torch.Tensor:
    """
    Sort collocation by t; weight slab k by exp(-ε * Σ_{j<k} L_j).
    Returns scalar weighted PDE loss.
    """
    dt_slab = (T_MAX - T_MIN) / n_slabs
    total   = torch.tensor(0.0, device=pde_res.device)
    cum     = torch.tensor(0.0, device=pde_res.device)
    sq      = pde_res ** 2

    for k in range(n_slabs):
        lo = T_MIN + k * dt_slab
        hi = T_MIN + (k + 1) * dt_slab
        mask = (t_vals >= lo) & (t_vals < hi)
        if mask.sum() == 0:
            continue
        w_k   = torch.exp(-eps * cum)
        total = total + w_k * sq[mask].mean()
        cum   = cum + sq[mask].mean().detach()

    return total


# ── High-fidelity FD reference ────────────────────────────────────────────────

def build_fd_reference(nu: float = NU_DEFAULT,
                       N: int = 512, dt: float = 1e-4,
                       T: float = T_MAX) -> tuple:
    """
    2nd-order upwind FD for convection + central FD for diffusion, RK4 in time.
    Returns:
        x_ref : (N,)   spatial grid
        t_ref : (M,)   saved time steps (every 100 FD steps)
        U_ref : (M, N) solution snapshots
    """
    dx  = (X_MAX - X_MIN) / (N - 1)
    x   = np.linspace(X_MIN, X_MAX, N)
    u   = -np.sin(np.pi * x).copy()       # IC

    n_steps = int(round(T / dt))
    save_every = max(1, n_steps // 100)   # ~100 snapshots

    t_snaps = [0.0]
    U_snaps = [u.copy()]

    def rhs(u_):
        """du/dt = -u*u_x + ν*u_xx  (upwind + central)"""
        # 2nd-order upwind for u*u_x
        # positive u: backward difference; negative u: forward difference
        u_xm  = (3*u_ - 4*np.roll(u_,  1) + np.roll(u_,  2)) / (2*dx)  # backward
        u_xp  = (-3*u_ + 4*np.roll(u_, -1) - np.roll(u_, -2)) / (2*dx)  # forward
        u_x_  = np.where(u_ > 0, u_xm, u_xp)
        conv  = u_ * u_x_
        # central difference for u_xx
        u_xx_ = (np.roll(u_, -1) - 2*u_ + np.roll(u_, 1)) / (dx * dx)
        diffs = nu * u_xx_
        dudt  = -conv + diffs
        # enforce BCs
        dudt[0]  = 0.0
        dudt[-1] = 0.0
        return dudt

    for step in range(1, n_steps + 1):
        # RK4
        k1 = rhs(u)
        k2 = rhs(u + 0.5*dt*k1)
        k3 = rhs(u + 0.5*dt*k2)
        k4 = rhs(u +     dt*k3)
        u  = u + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        u[0]  = 0.0    # hard BC
        u[-1] = 0.0

        if step % save_every == 0:
            t_snaps.append(step * dt)
            U_snaps.append(u.copy())

    return x, np.array(t_snaps), np.array(U_snaps)


# ── Samplers ──────────────────────────────────────────────────────────────────

def sample_collocation(n: int, device: torch.device) -> torch.Tensor:
    """LHS-like random collocation in (x,t) ∈ (-1,1)×(0,1)."""
    x = torch.rand(n, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    t = torch.rand(n, 1, device=device) * (T_MAX - T_MIN) + T_MIN
    return torch.cat([x, t], dim=1)


def sample_ic(n: int, device: torch.device) -> torch.Tensor:
    """IC points: x ∈ (-1,1), t=0."""
    x = torch.rand(n, 1, device=device) * (X_MAX - X_MIN) + X_MIN
    return x                              # shape (n,1); t=0 added in loss fn


def sample_bc(n_t: int, device: torch.device) -> torch.Tensor:
    """BC points: x=±1 × uniform t grid."""
    t  = torch.linspace(T_MIN, T_MAX, n_t, device=device).unsqueeze(1)
    x_l = torch.full((n_t, 1), X_MIN, device=device)
    x_r = torch.full((n_t, 1), X_MAX, device=device)
    left  = torch.cat([x_l, t], dim=1)
    right = torch.cat([x_r, t], dim=1)
    return torch.cat([left, right], dim=0)
