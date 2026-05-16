"""
FD-PINN for 2D Heat Conduction
==============================

Self-contained NumPy implementation of a finite-difference Physics-Informed
Neural Network for the 2D heat (diffusion) equation in steady (Poisson) and
unsteady (parabolic) form.

The FD-stencil approach (a.k.a. ND-PINN / FD-PINN; closely related to CAN-PINN
by Chiu et al. 2022) replaces input-side automatic differentiation by central
finite differences, so we only need a custom first-order autodiff for the MLP
parameters.  This keeps the implementation in pure NumPy.

Variants:
    A1: Soft-baseline   (Problem A)
    A2: Soft, w_bc=100  (Problem A)
    A3: Hard-constraint (Problem A)
    A4: Hard + Fourier  (Problem A)
    B1: Soft-baseline   (Problem B)
    B2: Soft, w_bc=w_ic=100 (Problem B)
    B3: Hard-constraint (Problem B)

Usage:
    python pinn_2d_heat.py --variant A1
    python pinn_2d_heat.py --all
"""
from __future__ import annotations
import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

# -----------------------------------------------------------------------------
# RNG
# -----------------------------------------------------------------------------
RNG = np.random.default_rng(0)


# -----------------------------------------------------------------------------
# MLP with manual reverse-mode backprop (params only; inputs handled by FD)
# -----------------------------------------------------------------------------
class MLP:
    """Tanh-activated MLP with explicit forward + backward over parameters.

    Forward keeps a per-call cache of pre-activations and activations so that a
    subsequent backward(dL_du) call can compute parameter gradients exactly.
    """

    def __init__(self, layers, fourier=None, seed=0):
        self.layers = layers  # e.g. [2, 50, 50, 50, 50, 1]
        self.fourier = fourier  # None or dict(B=array(d_in, n_freq))
        rng = np.random.default_rng(seed)
        self.W = []
        self.b = []
        d_in = layers[0]
        if fourier is not None:
            d_in_eff = 2 * fourier["B"].shape[1]
        else:
            d_in_eff = d_in
        prev = d_in_eff
        for nxt in layers[1:]:
            # Glorot uniform
            limit = np.sqrt(6.0 / (prev + nxt))
            self.W.append(rng.uniform(-limit, limit, size=(prev, nxt)))
            self.b.append(np.zeros(nxt))
            prev = nxt

    def n_params(self):
        return sum(W.size + b.size for W, b in zip(self.W, self.b))

    # --- input embedding ----------------------------------------------------
    def _embed(self, X):
        if self.fourier is None:
            return X
        B = self.fourier["B"]
        proj = X @ B  # (N, n_freq)
        return np.concatenate([np.cos(2 * np.pi * proj), np.sin(2 * np.pi * proj)], axis=1)

    # --- forward ------------------------------------------------------------
    def forward(self, X):
        """Return (u, cache) where u is (N,) and cache supports backward()."""
        H = self._embed(X)
        acts = [H]  # post-activation outputs (with input as first)
        zs = []     # pre-activations
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            Z = H @ W + b
            zs.append(Z)
            if l < len(self.W) - 1:
                H = np.tanh(Z)
            else:
                H = Z  # linear output
            acts.append(H)
        u = acts[-1].squeeze(-1)
        cache = {"acts": acts, "zs": zs}
        return u, cache

    # --- backward over parameters -------------------------------------------
    def backward(self, dL_du, cache):
        """Given dL/du with shape (N,), compute (dW_list, db_list)."""
        acts = cache["acts"]
        zs = cache["zs"]
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        # gradient at output layer: dL/dZ_L
        dZ = dL_du[:, None]  # (N, 1)
        for l in reversed(range(len(self.W))):
            A_prev = acts[l]  # (N, prev)
            dW[l] = A_prev.T @ dZ
            db[l] = dZ.sum(axis=0)
            if l > 0:
                dA_prev = dZ @ self.W[l].T
                # tanh derivative: 1 - tanh(z)^2 = 1 - acts[l]^2
                dZ = dA_prev * (1.0 - acts[l] ** 2)
        return dW, db

    # --- flat parameter view ------------------------------------------------
    def params_flat(self):
        return np.concatenate([W.ravel() for W in self.W] + [b.ravel() for b in self.b])

    def set_params_flat(self, p):
        idx = 0
        for i, W in enumerate(self.W):
            n = W.size
            self.W[i] = p[idx: idx + n].reshape(W.shape)
            idx += n
        for i, b in enumerate(self.b):
            n = b.size
            self.b[i] = p[idx: idx + n].reshape(b.shape)
            idx += n


# -----------------------------------------------------------------------------
# Adam optimizer in pure NumPy
# -----------------------------------------------------------------------------
class Adam:
    def __init__(self, params_shape_list, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1 = beta1
        self.b2 = beta2
        self.eps = eps
        self.m = [np.zeros(s) for s in params_shape_list]
        self.v = [np.zeros(s) for s in params_shape_list]
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            mhat = self.m[i] / (1 - self.b1 ** self.t)
            vhat = self.v[i] / (1 - self.b2 ** self.t)
            params[i] = p - self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return params


# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------
@dataclass
class Config:
    name: str
    problem: str            # "A" steady, "B" unsteady
    n_interior: int = 1024
    n_boundary: int = 128
    n_ic: int = 256
    h: float = 1e-2         # spatial FD step
    dt: float = 1e-3        # temporal FD step (Problem B)
    alpha: float = 1.0      # diffusivity (Problem B)
    T: float = 0.1          # final time (Problem B)
    layers: tuple = (2, 50, 50, 50, 50, 1)   # overridden for B
    n_steps: int = 3000
    lr: float = 1e-3
    w_pde: float = 1.0
    w_bc: float = 1.0
    w_ic: float = 10.0
    hard_constraint: bool = False
    fourier_n_freq: int = 0
    fourier_sigma: float = 1.0
    log_every: int = 200
    seed: int = 0


# -----------------------------------------------------------------------------
# Sources / references
# -----------------------------------------------------------------------------
def f_source_steady(X):
    """f(x,y) = 2 pi^2 sin(pi x) sin(pi y)."""
    x, y = X[:, 0], X[:, 1]
    return 2.0 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)


def u_ref_steady(X):
    x, y = X[:, 0], X[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def u_ref_unsteady(X, alpha=1.0):
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    return np.sin(np.pi * x) * np.sin(np.pi * y) * np.exp(-2.0 * np.pi ** 2 * alpha * t)


def u_ic_unsteady(X):
    x, y = X[:, 0], X[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# -----------------------------------------------------------------------------
# Hard-constraint output transforms
# -----------------------------------------------------------------------------
def hard_envelope_steady(X):
    """B(x,y) = x(1-x)y(1-y); zeros on boundary of unit square."""
    x, y = X[:, 0], X[:, 1]
    return x * (1 - x) * y * (1 - y)


def hard_envelope_unsteady(X):
    """B(x,y,t) = x(1-x)y(1-y) * t; zero on space-boundary AND at t=0."""
    x, y, t = X[:, 0], X[:, 1], X[:, 2]
    return x * (1 - x) * y * (1 - y) * t


def hard_offset_unsteady(X):
    """g(x,y,t) = sin(pi x) sin(pi y) (acts as IC carrier; combined with envelope)."""
    x, y = X[:, 0], X[:, 1]
    return np.sin(np.pi * x) * np.sin(np.pi * y)


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def sample_interior_steady(n, h, rng):
    """Interior collocation points on [h, 1-h]^2 (so stencil stays in domain)."""
    pts = rng.uniform(h, 1 - h, size=(n, 2))
    return pts


def sample_boundary_steady(n, rng):
    """n points on each of 4 edges (returned as a single (4n, 2) array)."""
    s = rng.uniform(0, 1, size=n)
    bot = np.stack([s, np.zeros(n)], axis=1)
    top = np.stack([s, np.ones(n)], axis=1)
    lef = np.stack([np.zeros(n), s], axis=1)
    rig = np.stack([np.ones(n), s], axis=1)
    return np.concatenate([bot, top, lef, rig], axis=0)


def sample_interior_unsteady(n, h, dt, T, rng):
    pts = np.stack([
        rng.uniform(h, 1 - h, size=n),
        rng.uniform(h, 1 - h, size=n),
        rng.uniform(dt, T - dt, size=n),
    ], axis=1)
    return pts


def sample_boundary_unsteady(n, T, rng):
    s = rng.uniform(0, 1, size=n)
    t = rng.uniform(0, T, size=n)
    bot = np.stack([s, np.zeros(n), t], axis=1)
    top = np.stack([s, np.ones(n), t], axis=1)
    lef = np.stack([np.zeros(n), s, t], axis=1)
    rig = np.stack([np.ones(n), s, t], axis=1)
    return np.concatenate([bot, top, lef, rig], axis=0)


def sample_ic_unsteady(n, rng):
    s = rng.uniform(0, 1, size=(n, 2))
    return np.concatenate([s, np.zeros((n, 1))], axis=1)


# -----------------------------------------------------------------------------
# Reference grids for L2 error
# -----------------------------------------------------------------------------
def ref_grid_steady(nx=101):
    g = np.linspace(0, 1, nx)
    X, Y = np.meshgrid(g, g, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    return pts, X, Y


def ref_grid_unsteady(nx=51, nt=11, T=0.1):
    g = np.linspace(0, 1, nx)
    tg = np.linspace(0, T, nt)
    X, Y, T_ = np.meshgrid(g, g, tg, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), T_.ravel()], axis=1)
    return pts, (X, Y, T_)


# -----------------------------------------------------------------------------
# Wrapped network output (handles hard constraint + Fourier features)
# -----------------------------------------------------------------------------
class NetWrapper:
    def __init__(self, mlp, hard_constraint=False, problem="A"):
        self.mlp = mlp
        self.hard = hard_constraint
        self.problem = problem

    def u_at(self, X):
        """Evaluate u and return (u, cache_for_backward, dU_dN)."""
        nn_out, cache = self.mlp.forward(X)
        if not self.hard:
            return nn_out, cache, np.ones_like(nn_out)
        if self.problem == "A":
            B = hard_envelope_steady(X)
            u = B * nn_out
            return u, cache, B
        else:
            # u(x,y,t) = sin(pi x) sin(pi y)        <- carrier (matches IC, vanishes on dOmega)
            #         + t * x(1-x) y(1-y) * NN     <- correction
            # At t=0: u = IC. On dOmega: both terms vanish.
            # Crucially, the carrier is CONSTANT in t -- the NN must learn the decay.
            B = hard_envelope_unsteady(X)            # = x(1-x)y(1-y) t
            g = hard_offset_unsteady(X)              # = sin(pi x) sin(pi y), constant in t
            base = g                                  # constant-in-time IC carrier
            u = base + B * nn_out
            return u, cache, B


# -----------------------------------------------------------------------------
# Steady (Problem A) training step
# -----------------------------------------------------------------------------
def train_steady(cfg: Config, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    fourier = None
    if cfg.fourier_n_freq > 0:
        B = rng.normal(0, cfg.fourier_sigma, size=(2, cfg.fourier_n_freq))
        fourier = {"B": B}
    mlp = MLP(layers=list(cfg.layers), fourier=fourier, seed=cfg.seed)
    net = NetWrapper(mlp, hard_constraint=cfg.hard_constraint, problem="A")

    # one fixed sample (full-batch) to keep things deterministic
    Xc = sample_interior_steady(cfg.n_interior, cfg.h, rng)
    Xb = sample_boundary_steady(cfg.n_boundary // 4, rng)
    Nb = Xb.shape[0]
    f_vals = f_source_steady(Xc)

    # Build stencil neighbor sets for each interior point
    h = cfg.h
    e1 = np.array([h, 0.0])
    e2 = np.array([0.0, h])
    Xc_p1 = Xc + e1
    Xc_m1 = Xc - e1
    Xc_p2 = Xc + e2
    Xc_m2 = Xc - e2

    # Pack everything into one big batch:
    # rows 0..N-1: centre, N..2N-1: +e1, 2N..3N-1: -e1, 3N..4N-1: +e2, 4N..5N-1: -e2
    N = Xc.shape[0]
    Xpde = np.concatenate([Xc, Xc_p1, Xc_m1, Xc_p2, Xc_m2], axis=0)
    if not cfg.hard_constraint:
        Xall = np.concatenate([Xpde, Xb], axis=0)
    else:
        Xall = Xpde

    # Reference grid for verification
    Xref, Xg, Yg = ref_grid_steady(101)
    u_ref = u_ref_steady(Xref)

    # Optimizer
    shapes = [W.shape for W in mlp.W] + [b.shape for b in mlp.b]
    opt = Adam(shapes, lr=cfg.lr)

    log_rows = []
    t0 = time.time()
    for step in range(cfg.n_steps + 1):
        # forward all points
        u_all, cache, B_all = net.u_at(Xall)
        # split
        u_c = u_all[0:N]
        u_p1 = u_all[N:2*N]
        u_m1 = u_all[2*N:3*N]
        u_p2 = u_all[3*N:4*N]
        u_m2 = u_all[4*N:5*N]
        if not cfg.hard_constraint:
            u_b = u_all[5*N:5*N+Nb]
        # FD Laplacian
        lap_x = (u_p1 - 2 * u_c + u_m1) / (h * h)
        lap_y = (u_p2 - 2 * u_c + u_m2) / (h * h)
        # residual r = -lap - f = 0  i.e.  -(lap_x+lap_y) = f  =>  r = -(lap_x+lap_y) - f
        r = -(lap_x + lap_y) - f_vals
        loss_pde = float(np.mean(r * r))
        if not cfg.hard_constraint:
            loss_bc = float(np.mean(u_b * u_b))  # boundary g=0
            loss = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc
        else:
            loss_bc = 0.0
            loss = cfg.w_pde * loss_pde

        # ---- backward ----
        # dL/dr = 2/N r * w_pde
        dr = (2.0 / N) * r * cfg.w_pde
        # r = -(lap_x+lap_y) - f
        d_lap_x = -dr
        d_lap_y = -dr
        # lap_x = (u_p1 - 2 u_c + u_m1) / h^2 ...
        invh2 = 1.0 / (h * h)
        du_c = -2.0 * (d_lap_x + d_lap_y) * invh2
        du_p1 = d_lap_x * invh2
        du_m1 = d_lap_x * invh2
        du_p2 = d_lap_y * invh2
        du_m2 = d_lap_y * invh2

        # accumulate dL/du for all sampled points
        dL_du = np.zeros_like(u_all)
        dL_du[0:N] = du_c
        dL_du[N:2*N] = du_p1
        dL_du[2*N:3*N] = du_m1
        dL_du[3*N:4*N] = du_p2
        dL_du[4*N:5*N] = du_m2
        if not cfg.hard_constraint:
            dL_du[5*N:5*N+Nb] = (2.0 / Nb) * u_b * cfg.w_bc

        # propagate through hard envelope: u = B * NN  =>  dL/dNN = dL/du * B
        # (and base term has no parameter dependence in unsteady case)
        dL_dN = dL_du * B_all

        dW, db = mlp.backward(dL_dN, cache)
        # Adam step
        params = mlp.W + mlp.b
        grads = list(dW) + list(db)
        new_params = opt.step(list(params), grads)
        # write back
        nW = len(mlp.W)
        mlp.W = list(new_params[:nW])
        mlp.b = list(new_params[nW:])

        if step % cfg.log_every == 0:
            # verify
            u_pred_ref, _, _ = net.u_at(Xref)
            err = u_pred_ref - u_ref
            rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(u_ref))
            log_rows.append((step, loss, loss_pde, loss_bc, 0.0, rel_l2))
            print(f"[{cfg.name} step {step:5d}] loss={loss:.4e} pde={loss_pde:.4e} bc={loss_bc:.4e} rel_l2={rel_l2:.4e}")

    wall = time.time() - t0
    # final outputs
    u_pred_ref, _, _ = net.u_at(Xref)
    u_pred_grid = u_pred_ref.reshape(Xg.shape)
    u_ref_grid = u_ref.reshape(Xg.shape)
    err = u_pred_ref - u_ref
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(u_ref))
    max_err = float(np.max(np.abs(err)))
    np.save(os.path.join(out_dir, "prediction.npy"), u_pred_grid)
    np.save(os.path.join(out_dir, "reference.npy"), u_ref_grid)
    with open(os.path.join(out_dir, "training_log.csv"), "w") as fh:
        fh.write("step,loss_total,loss_pde,loss_bc,loss_ic,rel_l2\n")
        for row in log_rows:
            fh.write(",".join(f"{x:.6e}" if not isinstance(x, int) else str(x) for x in row) + "\n")
    with open(os.path.join(out_dir, "verification.json"), "w") as fh:
        json.dump({
            "rel_l2_final": rel_l2,
            "max_err": max_err,
            "n_params": int(mlp.n_params()),
            "wallclock_s": wall,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        }, fh, indent=2)
    print(f"[{cfg.name}] DONE rel_l2={rel_l2:.4e} max_err={max_err:.4e} wall={wall:.1f}s")
    return rel_l2, max_err, wall


# -----------------------------------------------------------------------------
# Unsteady (Problem B) training step
# -----------------------------------------------------------------------------
def train_unsteady(cfg: Config, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    fourier = None
    if cfg.fourier_n_freq > 0:
        B = rng.normal(0, cfg.fourier_sigma, size=(3, cfg.fourier_n_freq))
        fourier = {"B": B}
    layers = list(cfg.layers)
    if layers[0] == 2:
        layers[0] = 3
    mlp = MLP(layers=layers, fourier=fourier, seed=cfg.seed)
    net = NetWrapper(mlp, hard_constraint=cfg.hard_constraint, problem="B")

    h = cfg.h
    dt = cfg.dt
    T = cfg.T
    alpha = cfg.alpha

    Xc = sample_interior_unsteady(cfg.n_interior, h, dt, T, rng)
    Xb = sample_boundary_unsteady(cfg.n_boundary // 4, T, rng)
    X0 = sample_ic_unsteady(cfg.n_ic, rng)
    N = Xc.shape[0]
    Nb = Xb.shape[0]
    N0 = X0.shape[0]
    u0_target = u_ic_unsteady(X0)

    e_x = np.array([h, 0.0, 0.0])
    e_y = np.array([0.0, h, 0.0])
    e_t = np.array([0.0, 0.0, dt])
    Xc_px = Xc + e_x
    Xc_mx = Xc - e_x
    Xc_py = Xc + e_y
    Xc_my = Xc - e_y
    Xc_pt = Xc + e_t
    Xc_mt = Xc - e_t

    # Pack: 0..N-1 centre; +x; -x; +y; -y; +t; -t  (7N) ; then boundary ; then IC
    pieces = [Xc, Xc_px, Xc_mx, Xc_py, Xc_my, Xc_pt, Xc_mt]
    Xpde = np.concatenate(pieces, axis=0)
    extra_count = 0
    if not cfg.hard_constraint:
        Xall = np.concatenate([Xpde, Xb, X0], axis=0)
        extra_count = Nb + N0
    else:
        Xall = Xpde

    Xref, _ = ref_grid_unsteady(51, 11, T=T)
    u_ref = u_ref_unsteady(Xref, alpha=alpha)

    shapes = [W.shape for W in mlp.W] + [b.shape for b in mlp.b]
    opt = Adam(shapes, lr=cfg.lr)

    log_rows = []
    t0 = time.time()
    invh2 = 1.0 / (h * h)
    inv2dt = 1.0 / (2 * dt)
    for step in range(cfg.n_steps + 1):
        u_all, cache, B_all = net.u_at(Xall)
        u_c = u_all[0:N]
        u_px = u_all[N:2*N]
        u_mx = u_all[2*N:3*N]
        u_py = u_all[3*N:4*N]
        u_my = u_all[4*N:5*N]
        u_pt = u_all[5*N:6*N]
        u_mt = u_all[6*N:7*N]
        if not cfg.hard_constraint:
            u_b = u_all[7*N:7*N + Nb]
            u_ic = u_all[7*N + Nb:7*N + Nb + N0]
        # FD derivatives
        u_t = (u_pt - u_mt) * inv2dt
        lap_x = (u_px - 2 * u_c + u_mx) * invh2
        lap_y = (u_py - 2 * u_c + u_my) * invh2
        r = u_t - alpha * (lap_x + lap_y)
        loss_pde = float(np.mean(r * r))
        if not cfg.hard_constraint:
            loss_bc = float(np.mean(u_b * u_b))
            loss_ic = float(np.mean((u_ic - u0_target) ** 2))
            loss = cfg.w_pde * loss_pde + cfg.w_bc * loss_bc + cfg.w_ic * loss_ic
        else:
            loss_bc = 0.0
            loss_ic = 0.0
            loss = cfg.w_pde * loss_pde

        # ---- backward ----
        dr = (2.0 / N) * r * cfg.w_pde
        # r = u_t - alpha (lap_x + lap_y)
        du_t = dr
        d_lap_x = -alpha * dr
        d_lap_y = -alpha * dr
        # u_t = (u_pt - u_mt)/(2 dt)
        du_pt = du_t * inv2dt
        du_mt = -du_t * inv2dt
        # lap_x
        du_c_x = -2 * d_lap_x * invh2
        du_px = d_lap_x * invh2
        du_mx = d_lap_x * invh2
        du_c_y = -2 * d_lap_y * invh2
        du_py = d_lap_y * invh2
        du_my = d_lap_y * invh2
        du_c = du_c_x + du_c_y

        dL_du = np.zeros_like(u_all)
        dL_du[0:N] = du_c
        dL_du[N:2*N] = du_px
        dL_du[2*N:3*N] = du_mx
        dL_du[3*N:4*N] = du_py
        dL_du[4*N:5*N] = du_my
        dL_du[5*N:6*N] = du_pt
        dL_du[6*N:7*N] = du_mt
        if not cfg.hard_constraint:
            dL_du[7*N:7*N + Nb] = (2.0 / Nb) * u_b * cfg.w_bc
            dL_du[7*N + Nb:7*N + Nb + N0] = (2.0 / N0) * (u_ic - u0_target) * cfg.w_ic

        dL_dN = dL_du * B_all

        dW, db = mlp.backward(dL_dN, cache)
        params = mlp.W + mlp.b
        grads = list(dW) + list(db)
        new_params = opt.step(list(params), grads)
        nW = len(mlp.W)
        mlp.W = list(new_params[:nW])
        mlp.b = list(new_params[nW:])

        if step % cfg.log_every == 0:
            u_pred_ref, _, _ = net.u_at(Xref)
            err = u_pred_ref - u_ref
            rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(u_ref))
            log_rows.append((step, loss, loss_pde, loss_bc, loss_ic, rel_l2))
            print(f"[{cfg.name} step {step:5d}] loss={loss:.4e} pde={loss_pde:.4e} bc={loss_bc:.4e} ic={loss_ic:.4e} rel_l2={rel_l2:.4e}")

    wall = time.time() - t0
    u_pred_ref, _, _ = net.u_at(Xref)
    err = u_pred_ref - u_ref
    rel_l2 = float(np.linalg.norm(err) / np.linalg.norm(u_ref))
    max_err = float(np.max(np.abs(err)))
    # also save final-time slice
    u_pred_grid = u_pred_ref.reshape(51, 51, 11)
    u_ref_grid = u_ref.reshape(51, 51, 11)
    np.save(os.path.join(out_dir, "prediction.npy"), u_pred_grid)
    np.save(os.path.join(out_dir, "reference.npy"), u_ref_grid)
    with open(os.path.join(out_dir, "training_log.csv"), "w") as fh:
        fh.write("step,loss_total,loss_pde,loss_bc,loss_ic,rel_l2\n")
        for row in log_rows:
            fh.write(",".join(f"{x:.6e}" if not isinstance(x, int) else str(x) for x in row) + "\n")
    with open(os.path.join(out_dir, "verification.json"), "w") as fh:
        json.dump({
            "rel_l2_final": rel_l2,
            "max_err": max_err,
            "n_params": int(mlp.n_params()),
            "wallclock_s": wall,
            "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.__dict__.items()},
        }, fh, indent=2)
    print(f"[{cfg.name}] DONE rel_l2={rel_l2:.4e} max_err={max_err:.4e} wall={wall:.1f}s")
    return rel_l2, max_err, wall


# -----------------------------------------------------------------------------
# Variant registry
# -----------------------------------------------------------------------------
def get_variant(name: str) -> Config:
    base_steady = dict(
        problem="A", n_interior=1024, n_boundary=128, n_steps=3000,
        layers=(2, 50, 50, 50, 50, 1),
    )
    base_unsteady = dict(
        problem="B", n_interior=768, n_boundary=128, n_ic=256, n_steps=2500,
        layers=(3, 50, 50, 50, 50, 1),
    )
    if name == "A1":
        return Config(name="A1", **base_steady, w_pde=1, w_bc=1, hard_constraint=False)
    if name == "A2":
        return Config(name="A2", **base_steady, w_pde=1, w_bc=100, hard_constraint=False)
    if name == "A3":
        return Config(name="A3", **base_steady, hard_constraint=True)
    if name == "A4":
        return Config(name="A4", **base_steady, hard_constraint=True,
                      fourier_n_freq=32, fourier_sigma=1.0)
    if name == "B1":
        return Config(name="B1", **base_unsteady, w_pde=1, w_bc=1, w_ic=10, hard_constraint=False)
    if name == "B2":
        return Config(name="B2", **base_unsteady, w_pde=1, w_bc=100, w_ic=100, hard_constraint=False)
    if name == "B3":
        return Config(name="B3", **base_unsteady, hard_constraint=True)
    raise ValueError(f"unknown variant {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--outdir", default="runs")
    args = parser.parse_args()
    if args.all:
        variants = ["A1", "A2", "A3", "A4", "B1", "B2", "B3"]
    elif args.variant:
        variants = [args.variant]
    else:
        raise SystemExit("specify --variant XX or --all")
    summary = []
    for v in variants:
        cfg = get_variant(v)
        out_dir = os.path.join(args.outdir, v)
        if cfg.problem == "A":
            r = train_steady(cfg, out_dir)
        else:
            r = train_unsteady(cfg, out_dir)
        summary.append((v, r))
    # save summary
    with open(os.path.join(args.outdir, "summary.json"), "w") as fh:
        json.dump([{"variant": v, "rel_l2": r[0], "max_err": r[1], "wallclock_s": r[2]}
                   for v, r in summary], fh, indent=2)
    print("=" * 60)
    print("SUMMARY")
    for v, r in summary:
        print(f"  {v}: rel_l2={r[0]:.4e}  max_err={r[1]:.4e}  wall={r[2]:.1f}s")


if __name__ == "__main__":
    main()
