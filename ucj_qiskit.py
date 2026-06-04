"""
UCJ (Unitary Cluster Jastrow) variational ansatz
- Optimisation : JAX + scipy L-BFGS-B
- Circuit info : Qiskit (transpile to {cx, rz, h, s, sdg})
- No PennyLane dependency

Gate conventions (Qiskit)
─────────────────────────
_qk_xy        : unphased XX+YY  = RXX(-θ) · RYY(-θ)   (imaginary Givens)
_qk_xy_phased : phased  XX+YY  = RZ(-π/2) · RXX(-θ) · RYY(-θ) · RZ(+π/2)  (real Givens)
CPhaseGate(φ) : controlled-phase  = diag(1,1,1,e^{iφ})   (Jastrow)
"""

from __future__ import annotations

import csv
import json
import os
import pathlib
import time
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import svds
import functools

import jax
import jax.numpy as jnp
from datetime import datetime

# ---------------------------------------------------------------------------
# Qiskit imports
# ---------------------------------------------------------------------------
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZGate, CPhaseGate, XGate
from qiskit.quantum_info import SparsePauliOp, Statevector

try:
    from qiskit.primitives import StatevectorEstimator as _QkEstimator
    _HAS_ESTIMATOR = True
except ImportError:
    _HAS_ESTIMATOR = False

print("[Qiskit]  backend ready (statevector)")

# ---------------------------------------------------------------------------
# JAX setup
# ---------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)
_GPU_DEVICES = jax.devices("cpu")
_CPU_DEVICE  = jax.devices("cpu")[0]
_JAX_DEVICE  = _GPU_DEVICES[0] if _GPU_DEVICES else _CPU_DEVICE
print(f"[JAX]  using device: {_JAX_DEVICE}")


def _to_device(x):
    return jax.device_put(x, _JAX_DEVICE)


# =============================================================================
# CONFIG
# =============================================================================
J1          = 1.0
J2          = 0.0
PBC         = True
ALPHA       = 3
K_MAX       = 1
E_TOL       = 1e-6
SEED        = 23
N_RESTARTS  = 1

LBFGS_MAXITER = 800
LBFGS_MAXFUN  = 50_000

N_COLD_RESTARTS = 1

VARIANTS = ['re', 'im', 'g']

# Qiskit circuit-info target basis
_TARGET_BASIS = ["cx", "rz", "h", "s", "sdg"]

# Jastrow GPU chunking
JASTROW_CHUNKED = False
JASTROW_CHUNK   = 32


# =============================================================================
# TIMER
# =============================================================================
class Timer:
    def __init__(self):
        self._laps   = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = _time.perf_counter()

    def stop(self, name):
        if name not in self._starts:
            raise KeyError(f"Timer '{name}' was never started.")
        self._laps[name] = _time.perf_counter() - self._starts.pop(name)
        return self._laps[name]

    def summary(self, title="Runtime summary"):
        W = 54
        print(f"\n{'='*W}\n  {title}\n{'='*W}")
        total = 0.0
        for name, t in self._laps.items():
            total += t
            m, s = divmod(t, 60)
            print(f"  {name:<36}  {int(m):2d}m {s:05.2f}s")
        m, s = divmod(total, 60)
        print(f"  {'─'*48}")
        print(f"  {'TOTAL':<36}  {int(m):2d}m {s:05.2f}s")
        print(f"{'='*W}\n")
        return self._laps.copy()


# =============================================================================
# DIAGNOSTIC DATA STRUCTURES
# =============================================================================
@dataclass
class RestartRecord:
    restart_idx:      int
    kind:             str
    x0_norm:          float
    x0_jastrow_norm:  float
    x0_givens_norm:   float
    E_final:          float
    variance:         float
    nit:              int
    nfev:             int
    wall_sec:         float
    converged:        bool

    def report(self):
        # ... (header lines unchanged) ...
        for lr in sorted(self.layers, key=lambda l: l.k):
            for r in lr.restarts:
                dE   = abs(r.E_final - self.e_exact)
                conv = " ✓" if r.converged else ""
                print(f"  {lr.k:<4} {r.restart_idx:<10} {r.kind:<14} "
                      f"{r.E_final:>12.8f} {dE:>10.6f} {r.variance:>10.6e} "  # ← variance
                      f"{r.nit:>6} {r.wall_sec:>7.1f}{conv}")


@dataclass
class LayerRecord:
    k:        int
    variant:  str
    restarts: list[RestartRecord] = field(default_factory=list)

    @property
    def best(self) -> RestartRecord | None:
        return min(self.restarts, key=lambda r: r.E_final) if self.restarts else None

    @property
    def cold_restarts(self):
        return [r for r in self.restarts if r.kind == "cold"]


@dataclass
class DiagnosticTracker:
    e_exact:  float
    n:        int
    variant:  str
    e_tol:    float = 1e-5

    layers:   list[LayerRecord] = field(default_factory=list)

    def _get_layer(self, k: int) -> LayerRecord:
        for lr in self.layers:
            if lr.k == k:
                return lr
        lr = LayerRecord(k=k, variant=self.variant)
        self.layers.append(lr)
        return lr

    def log_restart(self, k: int, record: RestartRecord):
        self._get_layer(k).restarts.append(record)

    def report(self):
        W    = 72
        sep  = "=" * W
        thin = "-" * W

        print(f"\n{sep}")
        print(f"  DIAGNOSTIC  |  {self.variant}-uCJ  N={self.n}")
        print(f"  E_exact = {self.e_exact:.8f}    e_tol = {self.e_tol:.2e}")
        print(sep)

        print(f"\n  [1]  PER-LAYER RESULTS")
        print(thin)
        hdr = (f"  {'k':<4} {'restart':<10} {'kind':<14} "
               f"{'E_final':>12} {'|ΔE|':>10} {'fidelity':>10} "
               f"{'nit':>6} {'sec':>7}")
        print(hdr)
        print("  " + "-" * (W - 2))

        for lr in sorted(self.layers, key=lambda l: l.k):
            for r in lr.restarts:
                dE   = abs(r.E_final - self.e_exact)
                conv = " ✓" if r.converged else ""
                print(f"  {lr.k:<4} {r.restart_idx:<10} {r.kind:<14} "
                      f"{r.E_final:>12.8f} {dE:>10.6f}  "
                      f"{r.nit:>6} {r.wall_sec:>7.1f}{conv}")

        print(f"\n  [2]  INIT VECTOR DECOMPOSITION  (norms by block)")
        print(thin)
        print(f"  {'k':<4} {'restart':<10} {'kind':<14} "
              f"{'‖x0‖':>10} {'‖Jastrow‖':>12} {'‖Givens‖':>12}")
        print("  " + "-" * (W - 2))
        for lr in sorted(self.layers, key=lambda l: l.k):
            for r in lr.restarts:
                print(f"  {lr.k:<4} {r.restart_idx:<10} {r.kind:<14} "
                      f"{r.x0_norm:>10.5f} {r.x0_jastrow_norm:>12.5f} "
                      f"{r.x0_givens_norm:>12.5f}")


# =============================================================================
# EXACT DIAGONALISATION
# =============================================================================
def build_basis(n, n_up):
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up],
                    dtype=np.int64)


def build_hamiltonian_matvec(n, n_up, nn_edges, nnn_edges, j1=J1, j2=J2):
    """
    Returns (matvec, basis, idx_map) without ever building the dense matrix.
    Stores only COO triplets; applies H via a single scatter-add per call.
    Memory: O(nnz)  vs  O(dim^2) before.
    """
    basis   = build_basis(n, n_up)          # still need basis list
    idx_map = {int(b): i for i, b in enumerate(basis)}
    dim     = len(basis)

    rows_list, cols_list, vals_list = [], [], []

    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = 0.5 if (bits >> si) & 1 else -0.5
                zj = 0.5 if (bits >> sj) & 1 else -0.5
                rows_list.append(row); cols_list.append(row)
                vals_list.append(j * zi * zj)
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl  = bits ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(int(fl), -1)
                    if col >= 0:
                        rows_list.append(row); cols_list.append(col)
                        vals_list.append(0.5 * j)

    rows_np = np.array(rows_list, dtype=np.int32)
    cols_np = np.array(cols_list, dtype=np.int32)
    vals_np = np.array(vals_list, dtype=np.float64)

    def matvec(v):
        out = np.zeros(dim, dtype=v.dtype)
        np.add.at(out, rows_np, vals_np * v[cols_np])
        return out

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    return op, basis, idx_map

def get_ground_state(n, n_up, nn_edges, nnn_edges, j1=J1, j2=J2):
    op, basis, idx_map = build_hamiltonian_matvec(
        n, n_up, nn_edges, nnn_edges, j1, j2)
    evals, evecs = eigsh(op, k=1, which='SA', tol=1e-10, maxiter=10_000)
    e_exact      = float(evals[0])
    psi_exact_np = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    return e_exact, psi_exact_np, basis, idx_map


def get_n_up(n):
    return (n + 1) // 2 if n % 2 == 1 else n // 2


# =============================================================================
# JAX HAMILTONIAN
# =============================================================================
def build_jax_hamiltonian(n, n_up, nn_edges, nnn_edges, j1=J1, j2=J2):
    basis_list = [b for b in range(1 << n) if bin(b).count('1') == n_up]
    idx_map    = {b: i for i, b in enumerate(basis_list)}
    rows, cols, vals = [], [], []

    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for i, js in edges:
            for row, bits in enumerate(basis_list):
                zi = 0.5 if (bits >> i) & 1 else -0.5
                zj = 0.5 if (bits >> js) & 1 else -0.5
                rows.append(row); cols.append(row); vals.append(j * zi * zj)
                if ((bits >> i) & 1) != ((bits >> js) & 1):
                    fl = bits ^ (1 << i) ^ (1 << js)
                    if fl in idx_map:
                        rows.append(row)
                        cols.append(idx_map[fl])
                        vals.append(0.5 * j)

    h_rows = _to_device(jnp.array(rows, dtype=jnp.int32))
    h_cols = _to_device(jnp.array(cols, dtype=jnp.int32))
    h_vals = _to_device(jnp.array(vals, dtype=jnp.float64))
    return h_rows, h_cols, h_vals


def make_apply_H(h_rows, h_cols, h_vals, dim):
    @functools.partial(jax.jit, donate_argnums=())
    def apply_H(psi):
        return (jnp.zeros(dim, dtype=psi.dtype)
                .at[h_rows].add(h_vals * psi[h_cols]))
    return apply_H


def neel_state(n, n_up, basis, idx_map):
    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi = jnp.zeros(len(basis), dtype=jnp.complex128)
    psi = psi.at[idx_map[neel_bits]].set(1.0)
    return _to_device(psi)


# =============================================================================
# JASTROW
# =============================================================================
def build_jastrow_indices(n, basis):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_i_np = np.array([p[0] for p in pairs], dtype=np.int32)
    pair_j_np = np.array([p[1] for p in pairs], dtype=np.int32)
    basis_bits_gpu = _to_device(jnp.array(basis, dtype=jnp.int32))
    return pair_i_np, pair_j_np, basis_bits_gpu


def _make_jastrow_phase_fn(pair_i_np, pair_j_np, basis_bits_gpu):
    """
    Computes  phase[b] = Σ_k  theta_k * occ(b,i_k) * occ(b,j_k)
    without ever materialising the [n_pair × dim] occupation matrix.

    Uses lax.scan over pairs: O(n_pair × dim) time, O(dim) peak memory.
    """
    pi_gpu = _to_device(jnp.array(pair_i_np, dtype=jnp.int32))
    pj_gpu = _to_device(jnp.array(pair_j_np, dtype=jnp.int32))

    @jax.jit
    def jastrow_phase(theta_J):
        def accumulate(phase, k_args):
            theta_k, i_k, j_k = k_args
            bi = ((basis_bits_gpu >> i_k) & 1).astype(jnp.float64)
            bj = ((basis_bits_gpu >> j_k) & 1).astype(jnp.float64)
            return phase + theta_k * bi * bj, None

        phase, _ = jax.lax.scan(
            accumulate,
            jnp.zeros(basis_bits_gpu.shape[0], dtype=jnp.float64),
            (theta_J, pi_gpu, pj_gpu),
        )
        return phase

    return jastrow_phase


def apply_jastrow(psi, theta_J, jastrow_phase_fn):
    return psi * jnp.exp(1j * jastrow_phase_fn(theta_J))


# =============================================================================
# GIVENS PAIRS
# =============================================================================
def build_givens_pairs_padded(n, basis, idx_map):
    srcs_ragged, dsts_ragged = [], []
    for i in range(n):
        for j in range(i + 1, n):
            srcs, dsts = [], []
            for row, bits in enumerate(basis):
                if ((bits >> i) & 1) and not ((bits >> j) & 1):
                    flipped = bits ^ (1 << i) ^ (1 << j)
                    if flipped in idx_map:
                        srcs.append(row)
                        dsts.append(idx_map[flipped])
            srcs_ragged.append(np.array(srcs, dtype=np.int32))
            dsts_ragged.append(np.array(dsts, dtype=np.int32))

    counts  = np.array([len(s) for s in srcs_ragged], dtype=np.int32)
    row_ptr = np.zeros(len(counts) + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts)

    srcs_cat = (np.concatenate(srcs_ragged) if srcs_ragged
                else np.array([], dtype=np.int32))
    dsts_cat = (np.concatenate(dsts_ragged) if dsts_ragged
                else np.array([], dtype=np.int32))

    srcs_flat_gpu = _to_device(jnp.array(srcs_cat, dtype=jnp.int32))
    dsts_flat_gpu = _to_device(jnp.array(dsts_cat, dtype=jnp.int32))

    nnz    = len(srcs_cat)
    n_pair = n * (n - 1) // 2
    print(f"[GivensPairs N={n}]  n_pair={n_pair}  total_nnz={nnz}  "
          f"GPU mem≈{nnz*2*4/1e6:.1f} MB  device={_JAX_DEVICE}")
    return srcs_flat_gpu, dsts_flat_gpu, row_ptr


# =============================================================================
# GIVENS SCAN
# =============================================================================
def _givens_scan_csr(psi, thetas, srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
                     imag=False):
    n_pair = row_ptr_np.shape[0] - 1
    for k in range(n_pair):
        start = int(row_ptr_np[k])
        end   = int(row_ptr_np[k + 1])
        if start == end:
            continue
        srcs_k = srcs_flat_gpu[start:end]
        dsts_k = dsts_flat_gpu[start:end]
        c = jnp.cos(thetas[k])
        s = jnp.sin(thetas[k])
        p_s, p_d = psi[srcs_k], psi[dsts_k]
        if imag:
            new_s =  c * p_s - 1j * s * p_d
            new_d = -1j * s * p_s + c * p_d
        else:
            new_s = c * p_s - s * p_d
            new_d = s * p_s + c * p_d
        psi = psi.at[srcs_k].set(new_s).at[dsts_k].set(new_d)
    return psi


# =============================================================================
# ANSATZ STATE BUILDERS
# =============================================================================
def ucj_state_re(theta, k_layers, psi0, n_pair,
                 srcs_flat_gpu, dsts_flat_gpu, row_ptr_np, jastrow_phase_fn):
    psi = psi0
    for layer in range(k_layers):
        off = layer * 2 * n_pair
        psi = apply_jastrow(psi, theta[off:off+n_pair], jastrow_phase_fn)
        psi = _givens_scan_csr(psi, theta[off+n_pair:off+2*n_pair],
                               srcs_flat_gpu, dsts_flat_gpu, row_ptr_np)
    return psi


def ucj_state_im(theta, k_layers, psi0, n_pair,
                 srcs_flat_gpu, dsts_flat_gpu, row_ptr_np, jastrow_phase_fn):
    psi = psi0
    for layer in range(k_layers):
        off = layer * 2 * n_pair
        psi = apply_jastrow(psi, theta[off:off+n_pair], jastrow_phase_fn)
        psi = _givens_scan_csr(psi, theta[off+n_pair:off+2*n_pair],
                               srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
                               imag=True)
    return psi


def ucj_state_g(theta, k_layers, psi0, n_pair,
                srcs_flat_gpu, dsts_flat_gpu, row_ptr_np, jastrow_phase_fn):
    psi = psi0
    for layer in range(k_layers):
        off = layer * 3 * n_pair
        psi = apply_jastrow(psi, theta[off:off+n_pair], jastrow_phase_fn)
        psi = _givens_scan_csr(psi, theta[off+n_pair:off+2*n_pair],
                               srcs_flat_gpu, dsts_flat_gpu, row_ptr_np)
        psi = _givens_scan_csr(psi, theta[off+2*n_pair:off+3*n_pair],
                               srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
                               imag=True)
    return psi


def _energy(psi, apply_H):
    norm = jnp.dot(jnp.conj(psi), psi)
    return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)


def make_energy_grad(variant, n, k_layers, psi0,
                     srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
                     jastrow_phase_fn, apply_H):
    n_pair = n * (n - 1) // 2
    _state_fns = {
        're': lambda th: ucj_state_re(th, k_layers, psi0, n_pair,
                                      srcs_flat_gpu, dsts_flat_gpu,
                                      row_ptr_np, jastrow_phase_fn),
        'im': lambda th: ucj_state_im(th, k_layers, psi0, n_pair,
                                      srcs_flat_gpu, dsts_flat_gpu,
                                      row_ptr_np, jastrow_phase_fn),
        'g':  lambda th: ucj_state_g(th, k_layers, psi0, n_pair,
                                     srcs_flat_gpu, dsts_flat_gpu,
                                     row_ptr_np, jastrow_phase_fn),
    }
    state_fn = _state_fns[variant]

    def efn(theta_gpu):
        return _energy(state_fn(theta_gpu), apply_H)

    jit_val_grad = jax.jit(jax.value_and_grad(efn, holomorphic=False))
    jit_efn      = jax.jit(efn)
    return jit_val_grad, jit_efn


def energy_variance(theta, variant, k_layers, psi0, apply_H,
                    srcs_pad, dsts_pad, mask_pad, jastrow_phase_fn, n):
    """
    Var[H] = <H^2> - <H>^2  (zero iff psi is an eigenstate).
    Replaces fidelity as a convergence diagnostic.
    Cost: two H applications per call instead of one.
    """
    n_pair    = n * (n - 1) // 2
    theta_gpu = _to_device(jnp.array(theta, dtype=jnp.float64))

    _state_fns = {
        're': lambda th: ucj_state_re(th, k_layers, psi0, n_pair,
                                      srcs_pad, dsts_pad, mask_pad,
                                      jastrow_phase_fn),
        'im': lambda th: ucj_state_im(th, k_layers, psi0, n_pair,
                                      srcs_pad, dsts_pad, mask_pad,
                                      jastrow_phase_fn),
        'g':  lambda th: ucj_state_g(th, k_layers, psi0, n_pair,
                                     srcs_pad, dsts_pad, mask_pad,
                                     jastrow_phase_fn),
    }
    psi  = _state_fns[variant](theta_gpu)
    norm = jnp.dot(jnp.conj(psi), psi)
    Hpsi = apply_H(psi)
    E    = jnp.real(jnp.dot(jnp.conj(psi), Hpsi) / norm)
    H2   = jnp.real(jnp.dot(jnp.conj(Hpsi), Hpsi) / norm)
    return float(H2 - E ** 2)


# =============================================================================
# COLD INITIALISATION
# =============================================================================
def cold_start(variant, n, k_layers, seed=SEED, noise_scale=0.05):
    """Pure random initialisation. Returns flat param vector."""
    n_pair = n * (n - 1) // 2
    rng    = np.random.default_rng(seed)
    stride = 3 * n_pair if variant == 'g' else 2 * n_pair
    return noise_scale * rng.standard_normal(k_layers * stride)


# =============================================================================
# INSTRUMENTED L-BFGS
# =============================================================================
def _optimise_layer_instrumented(
    tracker, variant, n, k, x0, x0_kind, restart_idx,
    e_exact, e_tol, val_grad_fn, variance_fn, n_pair,   # variance_fn replaces fidelity_fn
    lbfgs_maxiter=LBFGS_MAXITER, lbfgs_maxfun=LBFGS_MAXFUN,
):
    x0_jJ      = x0[:n_pair]
    x0_gK      = x0[n_pair:]
    x0_norm    = float(np.linalg.norm(x0))
    x0_jJ_norm = float(np.linalg.norm(x0_jJ))
    x0_gK_norm = float(np.linalg.norm(x0_gK))

    x0_gpu = _to_device(jnp.array(x0, dtype=jnp.float64))
    val_grad_fn(x0_gpu)  # warm-up JIT

    def scipy_fn(x_np):
        x_gpu = _to_device(jnp.array(x_np, dtype=jnp.float64))
        E, g  = val_grad_fn(x_gpu)
        return float(E), np.array(g, dtype=np.float64)

    t0 = time.perf_counter()
    result = scipy_minimize(
        scipy_fn, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": lbfgs_maxiter, "maxfun": lbfgs_maxfun,
                 "ftol": 1e-14, "gtol": 1e-8})
    wall = time.perf_counter() - t0

    opt_x = np.array(result.x)
    opt_E = float(result.fun)
    var   = variance_fn(opt_x)          # replaces fidelity call

    record = RestartRecord(
        restart_idx=restart_idx,
        kind=x0_kind,
        x0_norm=x0_norm,
        x0_jastrow_norm=x0_jJ_norm,
        x0_givens_norm=x0_gK_norm,
        E_final=opt_E,
        variance=var,                   # replaces fidelity field
        nit=result.nit,
        nfev=result.nfev,
        wall_sec=wall,
        converged=abs(opt_E - e_exact) < e_tol,
    )
    tracker.log_restart(k, record)

    print(f"  [{x0_kind:>14} restart={restart_idx}  k={k}]  "
          f"E={opt_E:.8f}  |ΔE|={abs(opt_E-e_exact):.4e}  "
          f"Var={var:.4e}  nit={result.nit}  nfev={result.nfev}  "
          f"t={wall:.1f}s")

    return opt_x, opt_E, record


def _cold_baseline(
    tracker, k, n_pair, n_restarts, e_exact, e_tol,
    val_grad_fn, variance_fn,           # variance_fn replaces fidelity_fn
    noise_scale=0.05, seed=9999,
):
    variant = tracker.variant
    n       = tracker.n
    rng     = np.random.default_rng(seed + k * 1000)
    stride  = 3 * n_pair if variant == "g" else 2 * n_pair

    print(f"\n  [cold baseline  k={k}  n_restarts={n_restarts}]")
    for i in range(n_restarts):
        x0_cold = noise_scale * rng.standard_normal(k * stride)
        _optimise_layer_instrumented(
            tracker=tracker, variant=variant, n=n, k=k,
            x0=x0_cold, x0_kind="cold", restart_idx=i,
            e_exact=e_exact, e_tol=e_tol,
            val_grad_fn=val_grad_fn, variance_fn=variance_fn,
            n_pair=n_pair)


# =============================================================================
# ADAPTIVE LAYER SEARCH
# =============================================================================
def adaptive_ucj(variant, n, k_max, e_tol, e_exact, psi_neel,
                 srcs_pad, dsts_pad, mask_pad, jastrow_phase_fn, apply_H,
                 tracker: DiagnosticTracker,
                 n_restarts=N_RESTARTS, n_cold_restarts=N_COLD_RESTARTS,
                 basis=None, lattice_name: str = "unknown",
                 j1: float = J1, j2: float = J2):

    best        = dict(E=np.inf, variance=np.inf, params=None, k=1)
    prev_params = None
    n_pair      = n * (n - 1) // 2
    history     = []

    for k in range(1, k_max + 1):
        layer_best = dict(E=np.inf, variance=np.inf, params=None)

        val_grad_fn, _ = make_energy_grad(
            variant, n, k, psi_neel,
            srcs_pad, dsts_pad, mask_pad,
            jastrow_phase_fn, apply_H)

        def var_fn(x):
            return energy_variance(
                x, variant, k, psi_neel, apply_H,
                srcs_pad, dsts_pad, mask_pad,
                jastrow_phase_fn, n)

        for restart in range(n_restarts):
            rng = np.random.default_rng(SEED + k * 100 + restart)
            if prev_params is None:
                x0      = cold_start(variant, n, k, seed=SEED + restart)
                x0_kind = "cold"
            else:
                noise = lambda m: 0.05 * rng.standard_normal(m)
                if variant == 'g':
                    new_layer = np.concatenate([noise(n_pair)] * 3)
                else:
                    new_layer = np.concatenate([noise(n_pair)] * 2)
                x0      = np.concatenate([prev_params, new_layer])
                x0_kind = "layer_append"

            opt_x, opt_E, record = _optimise_layer_instrumented(
                tracker=tracker, variant=variant, n=n, k=k,
                x0=x0, x0_kind=x0_kind,
                restart_idx=restart, e_exact=e_exact, e_tol=e_tol,
                val_grad_fn=val_grad_fn, variance_fn=var_fn,
                n_pair=n_pair)

            if opt_E < layer_best['E']:
                layer_best.update(E=opt_E, variance=record.variance,
                                  params=opt_x)
            if record.converged:
                break

        if n_cold_restarts > 0:
            _cold_baseline(
                tracker=tracker, k=k, n_pair=n_pair,
                n_restarts=n_cold_restarts,
                e_exact=e_exact, e_tol=e_tol,
                val_grad_fn=val_grad_fn, variance_fn=var_fn)

        prev_params = layer_best['params']
        if layer_best['E'] < best['E']:
            best.update(**layer_best, k=k)

        delta     = abs(layer_best['E'] - e_exact)
        theta_gpu = _to_device(jnp.array(layer_best['params'], dtype=jnp.float64))

        _state_fns = {
            're': lambda th: ucj_state_re(th, k, psi_neel, n_pair,
                                          srcs_pad, dsts_pad, mask_pad,
                                          jastrow_phase_fn),
            'im': lambda th: ucj_state_im(th, k, psi_neel, n_pair,
                                          srcs_pad, dsts_pad, mask_pad,
                                          jastrow_phase_fn),
            'g':  lambda th: ucj_state_g(th, k, psi_neel, n_pair,
                                         srcs_pad, dsts_pad, mask_pad,
                                         jastrow_phase_fn),
        }

        psi_layer  = np.array(_state_fns[variant](theta_gpu), dtype=np.complex128)
        psi_layer /= np.linalg.norm(psi_layer)

        spec = schmidt_spectrum(
            psi_layer, np.array(list(basis), dtype=np.int64), n)
        print_schmidt(spec, label=f"{variant}-uCJ layer k={k}")

        dE_layer  = (history[-1]["E"] - layer_best["E"]) if history else None
        dVar_layer = (history[-1]["variance"] - layer_best["variance"]) if history else None
        improvement = (f"  ΔE_layer={dE_layer:+.4e}  ΔVar_layer={dVar_layer:+.4e}"
                       if dE_layer is not None else "")
        print(f"[{variant}-uCJ k={k}]  "
              f"E={layer_best['E']:.8f}  |ΔE_exact|={delta:.4e}  "
              f"Var={layer_best['variance']:.4e}{improvement}")

        gate_counts_k, depth_k = circuit_info_qiskit(
            n, k, variant, layer_best["params"])
        write_circuit_summary(
            variant=variant, lattice_name=lattice_name, n=n,
            j1=j1, j2=j2, k=k, gate_counts=gate_counts_k, depth=depth_k)

        history.append({
            "k":            k,
            "E":            layer_best["E"],
            "deltaE":       delta,
            "variance":     layer_best["variance"],
            "dE_layer":     dE_layer,
            "dVar_layer":   dVar_layer,
            "S_vN":         spec["entropy_vn"],
            "S2":           spec["entropy_renyi2"],
            "schmidt_gap":  spec["schmidt_gap"],
            "schmidt_values": spec["schmidt_values"].tolist(),
        })

    return best, history


# =============================================================================
# QISKIT GATE HELPERS
# =============================================================================
def _qk_xy(qc: QuantumCircuit, theta: float, q0: int, q1: int) -> None:
    """Unphased XX+YY rotation (imaginary Givens): RXX(-θ) · RYY(-θ)."""
    qc.append(RXXGate(-theta), [q0, q1])
    qc.append(RYYGate(-theta), [q0, q1])


def _qk_xy_phased(qc: QuantumCircuit, theta: float, q0: int, q1: int) -> None:
    """Phased XX+YY rotation (real Givens): RZ(-π/2) · RXX(-θ) · RYY(-θ) · RZ(+π/2)."""
    qc.append(RZGate(-np.pi / 2), [q1])
    qc.append(RXXGate(-theta), [q0, q1])
    qc.append(RYYGate(-theta), [q0, q1])
    qc.append(RZGate(np.pi / 2), [q1])


# =============================================================================
# QISKIT UCJ CIRCUIT BUILDER
# =============================================================================
def build_ucj_qiskit(
    n: int,
    k_layers: int,
    variant: str,
    params: np.ndarray,
    pairs=None,
) -> QuantumCircuit:
    """
    Build the UCJ ansatz as a concrete Qiskit QuantumCircuit.

    Parameters
    ----------
    n        : number of qubits
    k_layers : number of UCJ layers
    variant  : 're' – real (phased) Givens
               'im' – imaginary Givens only
               'g'  – general (phased + imaginary)
    params   : flat parameter array
    pairs    : list of (i,j) pairs; defaults to all upper-triangle pairs
    """
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pair = len(pairs)
    stride = 3 * n_pair if variant == "g" else 2 * n_pair

    qc = QuantumCircuit(n)

    # Initial state: X on even qubits (Néel state)
    for i in range(n):
        if i % 2 == 0:
            qc.x(i)

    for l in range(k_layers):
        off  = l * stride
        tJ   = params[off          : off + n_pair]
        tK_r = params[off + n_pair : off + 2 * n_pair]
        tK_i = (params[off + 2 * n_pair : off + 3 * n_pair]
                if variant == "g" else None)

        # Jastrow (controlled-phase) layer
        for k, (i, j) in enumerate(pairs):
            qc.append(CPhaseGate(float(tJ[k])), [i, j])

        # Givens rotation layer
        if variant == "im":
            for k, (i, j) in enumerate(pairs):
                _qk_xy(qc, float(tK_r[k]), j, i)
        else:
            for k, (i, j) in enumerate(pairs):
                _qk_xy_phased(qc, float(tK_r[k]), j, i)
            if variant == "g":
                for k, (i, j) in enumerate(pairs):
                    _qk_xy(qc, float(tK_i[k]), j, i)

    return qc


# =============================================================================
# QISKIT HEISENBERG HAMILTONIAN
# =============================================================================
def build_heisenberg_qiskit(
    n: int,
    nn_edges: list,
    nnn_edges: list,
    j1: float = J1,
    j2: float = J2,
) -> SparsePauliOp:
    """J1-J2 Heisenberg Hamiltonian as a Qiskit SparsePauliOp."""
    pauli_terms = []

    def pauli_str(op_char: str, si: int, sj: int) -> str:
        # Qiskit strings are indexed right-to-left (qubit 0 = rightmost)
        chars = ["I"] * n
        chars[n - 1 - si] = op_char
        chars[n - 1 - sj] = op_char
        return "".join(chars)

    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for op in "XYZ":
                pauli_terms.append((pauli_str(op, si, sj), j / 4.0))

    labels = [t[0] for t in pauli_terms]
    coeffs = [t[1] for t in pauli_terms]
    return SparsePauliOp(labels, coeffs=coeffs).simplify()


# =============================================================================
# QISKIT NOISELESS ENERGY  (statevector)
# =============================================================================
def energy_noiseless_qiskit(
    n: int,
    k_layers: int,
    variant: str,
    params: np.ndarray,
    nn_edges: list,
    nnn_edges: list,
    j1: float = J1,
    j2: float = J2,
    pairs=None,
) -> float:
    """Exact (statevector) expectation value of the Heisenberg Hamiltonian."""
    qc  = build_ucj_qiskit(n, k_layers, variant, params, pairs)
    ham = build_heisenberg_qiskit(n, nn_edges, nnn_edges, j1, j2)

    if _HAS_ESTIMATOR:
        try:
            estimator = _QkEstimator()
            job    = estimator.run([(qc, ham)])
            result = job.result()
            return float(result[0].data.evs)
        except Exception:
            pass

    # Fallback: direct statevector contraction
    sv  = Statevector(qc)
    exp = sv.expectation_value(ham)
    return float(exp.real)


# =============================================================================
# QISKIT CIRCUIT INFO  (transpile to target basis, count + depth)
# =============================================================================
def circuit_info_qiskit(
    n: int,
    k_layers: int,
    variant: str,
    params: np.ndarray,
    pairs=None,
    basis_gates: list[str] | None = None,
    optimization_level: int = 3,
) -> tuple[dict, int]:
    """
    Transpile the UCJ circuit to a hardware-friendly basis and report gate counts.

    Parameters
    ----------
    basis_gates        : target gate set; defaults to _TARGET_BASIS
                         = ["cx", "rz", "h", "s", "sdg"]
    optimization_level : Qiskit transpiler level (0-3); default 3

    Returns
    -------
    gate_counts : dict[str, int]
    depth       : int
    """
    if basis_gates is None:
        basis_gates = _TARGET_BASIS

    qc = build_ucj_qiskit(n, k_layers, variant, params, pairs)

    try:
        tqc = transpile(qc, basis_gates=basis_gates,
                        optimization_level=optimization_level)
    except Exception as exc:
        warnings.warn(f"[circuit_info_qiskit] transpile failed: {exc}; "
                      "using undecomposed circuit.")
        tqc = qc

    gate_counts: dict[str, int] = dict(tqc.count_ops())
    depth = tqc.depth()
    total = sum(gate_counts.values())

    print(f"\n[Circuit info: {variant}-uCJ  N={n}  k={k_layers}]")
    print(f"  depth={depth}  total_gates={total}")
    print(f"  Gate counts (basis: {basis_gates}):")
    for gate, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
        print(f"    {gate:<30} {count}")

    return gate_counts, depth


# =============================================================================
# PRINT UCJ OPERATORS  (Qiskit version)
# =============================================================================
def print_ucj_operators(n: int, k_layers: int, variant: str,
                         params: np.ndarray, pairs=None):
    """Print the raw (pre-transpile) gate list of the UCJ circuit."""
    qc = build_ucj_qiskit(n, k_layers, variant, params, pairs)
    print(f"\n[UCJ operators: {variant}-uCJ  N={n}  k={k_layers}]")
    print(f"  {'#':<5} {'Gate':<30} {'Qubits':<15} {'Params'}")
    print(f"  {'─'*70}")
    for i, inst in enumerate(qc.data):
        op      = inst.operation
        qubits  = [qc.find_bit(q).index for q in inst.qubits]
        p_str   = (f"{[round(float(p), 4) for p in op.params]}"
                   if op.params else "—")
        print(f"  {i:<5} {op.name:<30} {str(qubits):<15} {p_str}")
    print(f"\n  Total operators: {len(qc.data)}")


# =============================================================================
# STATE OVERLAP  (JAX, renamed from state_overlap_pl)
# =============================================================================
def state_overlap(params: np.ndarray, variant: str, k_layers: int,
                  psi_neel, psi_exact_np: np.ndarray,
                  srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
                  jastrow_phase_fn, n: int) -> dict:
    n_pair     = n * (n - 1) // 2
    theta_gpu  = _to_device(jnp.array(params, dtype=jnp.float64))
    psi_ex_gpu = _to_device(jnp.array(psi_exact_np, dtype=jnp.complex128))

    _state_fns = {
        're': lambda th: ucj_state_re(th, k_layers, psi_neel, n_pair,
                                      srcs_flat_gpu, dsts_flat_gpu,
                                      row_ptr_np, jastrow_phase_fn),
        'im': lambda th: ucj_state_im(th, k_layers, psi_neel, n_pair,
                                      srcs_flat_gpu, dsts_flat_gpu,
                                      row_ptr_np, jastrow_phase_fn),
        'g':  lambda th: ucj_state_g(th, k_layers, psi_neel, n_pair,
                                     srcs_flat_gpu, dsts_flat_gpu,
                                     row_ptr_np, jastrow_phase_fn),
    }
    psi_ucj      = _state_fns[variant](theta_gpu)
    psi_ucj_norm = psi_ucj / jnp.sqrt(jnp.dot(jnp.conj(psi_ucj), psi_ucj))
    overlap      = float(jnp.abs(jnp.dot(jnp.conj(psi_ex_gpu), psi_ucj_norm)) ** 2)
    print(f"\n[State overlap]  overlap={overlap:.8f}")
    return dict(overlap=overlap)


# =============================================================================
# RDM FROBENIUS NORMS
# =============================================================================
def rdm_norms_at_convergence(params, variant, k_layers, psi0,
                              srcs_pad, dsts_pad, mask_pad,
                              jastrow_phase_fn, apply_H, n, basis, bindex):
    n_pair    = n * (n - 1) // 2
    theta_gpu = _to_device(jnp.array(params, dtype=jnp.float64))
    basis_np  = list(basis)

    _state_fns = {
        're': lambda th: ucj_state_re(th, k_layers, psi0, n_pair,
                                      srcs_pad, dsts_pad, mask_pad, jastrow_phase_fn),
        'im': lambda th: ucj_state_im(th, k_layers, psi0, n_pair,
                                      srcs_pad, dsts_pad, mask_pad, jastrow_phase_fn),
        'g':  lambda th: ucj_state_g(th, k_layers, psi0, n_pair,
                                     srcs_pad, dsts_pad, mask_pad, jastrow_phase_fn),
    }
    psi_gpu = _state_fns[variant](theta_gpu)
    norm    = jnp.sqrt(jnp.dot(jnp.conj(psi_gpu), psi_gpu))
    psi_gpu = psi_gpu / norm

    basis_gpu    = _to_device(jnp.array(basis_np, dtype=jnp.int32))
    shifts       = _to_device(jnp.arange(n, dtype=jnp.int32))
    occ_gpu      = ((basis_gpu[:, None] >> shifts[None, :]) & 1).astype(jnp.float64)
    probs_gpu    = jnp.abs(psi_gpu) ** 2

    # Diagonal
    rho_diag = jnp.einsum('b,bi->i', probs_gpu, occ_gpu)  # [n]

    # Hoist sorted basis lookup out of the per-pair loop
    sorted_order = jnp.argsort(basis_gpu)
    sorted_basis = basis_gpu[sorted_order]   # sorted values
    sorted_idx   = sorted_order              # original positions

    # Pre-compute site index array once
    site_indices = jnp.arange(n, dtype=jnp.int32)  # [n]

    @jax.jit
    def rho_offdiag_ij(i, j):
        # States where site i occupied, site j empty
        mask_src = (((basis_gpu >> i) & 1) & ~((basis_gpu >> j) & 1)).astype(bool)

        # Flip both bits to get the connected basis state
        flip    = jnp.int32((1 << i) | (1 << j))
        flipped = basis_gpu ^ flip  # [dim]

        # Lookup flipped state index via sorted searchsorted
        pos     = jnp.searchsorted(sorted_basis, flipped)
        pos     = jnp.clip(pos, 0, len(sorted_basis) - 1)
        col_idx = jnp.where(sorted_basis[pos] == flipped, sorted_idx[pos], -1)

        # Jordan-Wigner sign: count occupied sites strictly between i and j
        # FIX: replace dynamic slice with boolean mask — JIT-safe
        mid_mask = (site_indices > i) & (site_indices < j)          # [n] bool
        jw_bits  = jnp.where(mid_mask[None, :], occ_gpu, 0.0)      # [dim, n]
        jw_sign  = (-1.0) ** jw_bits.sum(axis=-1)                   # [dim]

        valid  = (col_idx >= 0) & mask_src
        rho_ij = jnp.sum(
            jnp.where(valid,
                      jnp.conj(psi_gpu[col_idx]) * psi_gpu * jw_sign,
                      0.0 + 0.0j)
        )
        return rho_ij

    rho = jnp.diag(rho_diag).astype(jnp.complex128)
    for i in range(n):
        for j in range(i + 1, n):
            rij = rho_offdiag_ij(i, j)
            rho = rho.at[i, j].set(rij)
            rho = rho.at[j, i].set(jnp.conj(rij))

    rho_np       = np.array(rho)
    mask_offdiag = ~np.eye(n, dtype=bool)
    re_frob      = np.linalg.norm(np.real(rho_np)[mask_offdiag])
    im_frob      = np.linalg.norm(np.imag(rho_np)[mask_offdiag])
    ratio        = im_frob / (re_frob + 1e-12)

    print(f"\n[RDM @ convergence  {variant}-uCJ  N={n}  k={k_layers}]")
    print(f"  ‖Re(ρ_ij)‖_F = {re_frob:.6f}")
    print(f"  ‖Im(ρ_ij)‖_F = {im_frob:.6f}")
    print(f"  Im/Re ratio  = {ratio:.6f}")
    return dict(re_frob=re_frob, im_frob=im_frob, ratio=ratio, rho=rho_np)


# =============================================================================
# SHARED QUANTUM STRUCTURES
# =============================================================================
def _build_quantum_structures(lattice, j1=J1, j2=J2):
    n         = lattice.n_sites
    nn_edges  = lattice.nn_edges
    nnn_edges = lattice.nnn_edges
    n_up      = get_n_up(n)

    # Matrix-free exact energy (no eigenvector stored)
    e_exact, psi_exact_np, basis, bindex = get_ground_state(
        n, n_up, nn_edges, nnn_edges, j1=j1, j2=j2)

    # JAX Hamiltonian — COO only
    h_rows, h_cols, h_vals = build_jax_hamiltonian(
        n, n_up, nn_edges, nnn_edges, j1=j1, j2=j2)
    apply_H = make_apply_H(h_rows, h_cols, h_vals, len(basis))
    del h_rows, h_cols, h_vals

    # Givens pairs — padded for lax.scan
    srcs_pad, dsts_pad, mask_pad = build_givens_pairs_padded(
        n, list(basis), bindex)

    # Jastrow — scan-based, no occ matrix
    pair_i_np, pair_j_np, basis_bits_gpu = build_jastrow_indices(n, basis)
    jastrow_phase_fn = _make_jastrow_phase_fn(
        pair_i_np, pair_j_np, basis_bits_gpu)

    psi_neel = neel_state(n, n_up, list(basis), bindex)

    return dict(
        n=n, n_up=n_up, nn_edges=nn_edges, nnn_edges=nnn_edges,
        basis=basis, bindex=bindex,
        e_exact=e_exact,
        psi_exact_np = psi_exact_np, #psi_exact_np / psi_exact_gpu intentionally absent
        apply_H=apply_H,
        srcs_pad=srcs_pad, dsts_pad=dsts_pad, mask_pad=mask_pad,
        jastrow_phase_fn=jastrow_phase_fn,
        psi_neel=psi_neel,
    )


# =============================================================================
# RUN  (top-level entry for a single lattice)
# =============================================================================
def run(
    lattice,
    variants: list[str] = VARIANTS,
    j1: float = J1,
    j2: float = J2,
    k_max: int = K_MAX,
    e_tol: float = E_TOL,
    n_restarts: int = N_RESTARTS,
    n_cold_restarts: int = N_COLD_RESTARTS,
    timer: Timer | None = None,
) -> dict:
    """
    Run the full UCJ optimisation pipeline for one lattice + (j1, j2).

    Steps per variant
    -----------------
    1.  Build shared quantum structures (exact diag, JAX Hamiltonian, Givens/Jastrow tables)
    2.  Adaptive layer search (adaptive_ucj) — L-BFGS cold + append restarts
    3.  Post-processing: state overlap, RDM Frobenius norms, EE profile
    4.  Qiskit noiseless energy cross-check
    5.  Write history CSV, summary .txt/.json, entanglement .txt/.json

    Returns
    -------
    results : dict keyed by variant, each holding
        tracker, jax_best, history, E_pl, overlap, rdm_conv
    """
    if timer is None:
        timer = Timer()

    lattice_name = lattice.name
    n            = lattice.n_sites
    print(f"\n{'='*70}")
    print(f"  RUN  |  {lattice_name}  N={n}  J1={j1}  J2={j2}")
    print(f"{'='*70}")

    # ── 1. Build shared structures ────────────────────────────────────────────
    timer.start("build_structures")
    qs = _build_quantum_structures(lattice, j1=j1, j2=j2)
    timer.stop("build_structures")

    e_exact      = qs["e_exact"]
    psi_exact_np = qs["psi_exact_np"]
    print(f"  E_exact = {e_exact:.10f}  (E/site = {e_exact/n:.10f})")

    # Exact EE spectrum (mid-chain) for comparison
    spec_exact = schmidt_spectrum(
        psi_exact_np,
        np.array(list(qs["basis"]), dtype=np.int64),
        n,
    )
    s_vn_exact = spec_exact["entropy_vn"]
    s2_exact   = spec_exact["entropy_renyi2"]

    results: dict = {}

    for variant in variants:
        print(f"\n{'─'*60}")
        print(f"  VARIANT: {variant}-uCJ  N={n}")
        print(f"{'─'*60}")

        tracker = DiagnosticTracker(
            e_exact=e_exact, n=n, variant=variant, e_tol=e_tol)

        # ── 2. Adaptive layer search ──────────────────────────────────────────
        timer.start(f"{variant}_adaptive")
        jax_best, history = adaptive_ucj(
            variant=variant,
            n=n,
            k_max=k_max,
            e_tol=e_tol,
            e_exact=e_exact,
            psi_neel=qs["psi_neel"],
            srcs_pad=qs["srcs_pad"],
            dsts_pad=qs["dsts_pad"],
            mask_pad=qs["mask_pad"],
            jastrow_phase_fn=qs["jastrow_phase_fn"],
            apply_H=qs["apply_H"],
            tracker=tracker,
            n_restarts=n_restarts,
            n_cold_restarts=n_cold_restarts,
            basis=qs["basis"],
            lattice_name=lattice_name,
            j1=j1,
            j2=j2,
        )
        timer.stop(f"{variant}_adaptive")

        best_params = jax_best["params"]
        best_k      = jax_best["k"]

        # ── 3a. State overlap ─────────────────────────────────────────────────
        timer.start(f"{variant}_overlap")
        overlap_res = state_overlap(
            params=best_params,
            variant=variant,
            k_layers=best_k,
            psi_neel=qs["psi_neel"],
            psi_exact_np=psi_exact_np,
            srcs_flat_gpu=qs["srcs_pad"],
            dsts_flat_gpu=qs["dsts_pad"],
            row_ptr_np=qs["mask_pad"],
            jastrow_phase_fn=qs["jastrow_phase_fn"],
            n=n,
        )
        timer.stop(f"{variant}_overlap")

        # ── 3b. RDM Frobenius norms ───────────────────────────────────────────
        timer.start(f"{variant}_rdm")
        rdm_conv = rdm_norms_at_convergence(
            params=best_params,
            variant=variant,
            k_layers=best_k,
            psi0=qs["psi_neel"],
            srcs_pad=qs["srcs_pad"],
            dsts_pad=qs["dsts_pad"],
            mask_pad=qs["mask_pad"],
            jastrow_phase_fn=qs["jastrow_phase_fn"],
            apply_H=qs["apply_H"],
            n=n,
            basis=qs["basis"],
            bindex=qs["bindex"],
        )
        timer.stop(f"{variant}_rdm")

        # ── 3c. Entanglement entropy profile ─────────────────────────────────
        timer.start(f"{variant}_ee")
        n_pair    = n * (n - 1) // 2
        theta_gpu = _to_device(jnp.array(best_params, dtype=jnp.float64))
        _sfns = {
            're': lambda th: ucj_state_re(th, best_k, qs["psi_neel"], n_pair,
                                          qs["srcs_pad"], qs["dsts_pad"],
                                          qs["mask_pad"], qs["jastrow_phase_fn"]),
            'im': lambda th: ucj_state_im(th, best_k, qs["psi_neel"], n_pair,
                                          qs["srcs_pad"], qs["dsts_pad"],
                                          qs["mask_pad"], qs["jastrow_phase_fn"]),
            'g':  lambda th: ucj_state_g(th, best_k, qs["psi_neel"], n_pair,
                                         qs["srcs_pad"], qs["dsts_pad"],
                                          qs["mask_pad"], qs["jastrow_phase_fn"]),
        }
        psi_best = np.array(_sfns[variant](theta_gpu), dtype=np.complex128)
        psi_best /= np.linalg.norm(psi_best)

        basis_arr     = np.array(list(qs["basis"]), dtype=np.int64)
        ee_rows       = compare_schmidt(psi_best, psi_exact_np, basis_arr, n)
        mid_spec_ucj  = schmidt_spectrum(psi_best,      basis_arr, n)
        mid_spec_exact = schmidt_spectrum(psi_exact_np, basis_arr, n)
        timer.stop(f"{variant}_ee")

        # ── 4. Qiskit noiseless energy ────────────────────────────────────────
        try:
            E_pl = energy_noiseless_qiskit(
                n=n, k_layers=best_k, variant=variant,
                params=best_params,
                nn_edges=lattice.nn_edges, nnn_edges=lattice.nnn_edges,
                j1=j1, j2=j2,
            )
        except Exception as exc:
            warnings.warn(f"[Qiskit energy] failed: {exc}")
            E_pl = float('nan')

        print(f"[{variant}-uCJ]  E_qiskit={E_pl:.10f}  "
              f"|ΔE_qiskit|={abs(E_pl - e_exact):.4e}")
        

        # ── 5. Write outputs ──────────────────────────────────────────────────
        write_history_csv(
            history=history,
            variant=variant,
            lattice_name=lattice_name,
            n=n, j1=j1, j2=j2,
            s_vn_exact=s_vn_exact,
            s2_exact=s2_exact,
        )
        write_entanglement_file(
            lat=lattice, j1=j1, j2=j2, e_exact=e_exact,
            variant=variant, jax_best=jax_best,
            ee_rows=ee_rows, mid_spec_ucj=mid_spec_ucj,
            mid_spec_exact=mid_spec_exact,
        )

        tracker.report()

        results[variant] = dict(
            tracker=tracker,
            jax_best=jax_best,
            history=history,
            E_pl=E_pl,
            overlap=overlap_res,
            rdm_conv=rdm_conv,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    write_summary_file(
        lat=lattice, j1=j1, j2=j2, e_exact=e_exact,
        results=results, variants=variants)

    return results


# =============================================================================
# SCHMIDT SPECTRUM
# =============================================================================
def _split_basis(basis_np: np.ndarray, n: int, cut: int):
    mask_left  = (1 << cut) - 1
    left_bits  = basis_np & mask_left
    right_bits = basis_np >> cut
    return left_bits, right_bits, left_bits.astype(int), right_bits.astype(int)


def schmidt_spectrum(
    psi: np.ndarray,
    basis: np.ndarray,
    n: int,
    cut: int | None = None,
    n_sv: int = 32,
    zero_thresh: float = 1e-14,
) -> dict:
    if cut is None:
        cut = n // 2

    basis_np = np.asarray(basis, dtype=np.int64)
    psi_np   = np.asarray(psi,   dtype=np.complex128)

    mask_left = np.int64((1 << cut) - 1)
    left_idx  = (basis_np & mask_left).astype(np.int32)
    right_idx = (basis_np >> cut).astype(np.int32)

    nz       = np.abs(psi_np) > zero_thresh
    psi_nz   = psi_np[nz]
    left_nz  = left_idx[nz]
    right_nz = right_idx[nz]

    dim_l  = 1 << cut
    dim_r  = 1 << (n - cut)
    min_dim = min(dim_l, dim_r)

    Psi_sp = coo_matrix(
        (psi_nz, (left_nz, right_nz)),
        shape=(dim_l, dim_r),
        dtype=np.complex128,
    ).tocsr()

    # svds requires k strictly < min(shape) AND k <= nnz.
    # If the matrix is small or nearly full-rank, dense SVD is cheaper anyway.
    n_nonzero_rows = int((np.diff(Psi_sp.indptr) > 0).sum())
    k = min(n_sv, n_nonzero_rows, min_dim - 1)

    if k <= 0:
        # Degenerate: only one nonzero row/col, entropy is zero
        sv = np.array([1.0])
    elif min_dim <= 2 or k >= min_dim - 1:
        # Dense fallback for small matrices
        sv = np.linalg.svd(Psi_sp.toarray(), compute_uv=False, full_matrices=False)
    else:
        try:
            sv = svds(Psi_sp, k=k, which='LM', return_singular_vectors=False)
            sv = np.sort(sv)[::-1]
        except Exception:
            # svds can still fail on pathological sparsity patterns; dense is safe
            sv = np.linalg.svd(Psi_sp.toarray(), compute_uv=False, full_matrices=False)

    sv   = sv[sv > zero_thresh]
    if len(sv) == 0:
        sv = np.array([1.0])

    lam2  = sv ** 2
    lam2 /= lam2.sum()

    s_vn     = float(-np.sum(lam2 * np.log(lam2 + 1e-300)))
    s_renyi2 = float(-np.log(np.sum(lam2 ** 2) + 1e-300))
    gap      = float(lam2[0] - lam2[1]) if len(lam2) > 1 else float(lam2[0])

    return dict(
        schmidt_values=lam2,
        entropy_vn=s_vn,
        entropy_renyi2=s_renyi2,
        schmidt_gap=gap,
        n_schmidt=len(lam2),
        cut=cut,
    )

def schmidt_profile(psi, basis, n, cuts=None):
    if cuts is None:
        cuts = list(range(1, n))
    return [schmidt_spectrum(psi, basis, n, cut=c) for c in cuts]


def print_schmidt(spec: dict, label: str = ""):
    lam2 = spec['schmidt_values']
    top  = min(8, len(lam2))
    print(f"\n[Schmidt  cut={spec['cut']}  {label}]")
    print(f"  S_vN        = {spec['entropy_vn']:.8f}")
    print(f"  S_Renyi2    = {spec['entropy_renyi2']:.8f}")
    print(f"  Schmidt gap = {spec['schmidt_gap']:.8f}")
    print(f"  n_Schmidt   = {spec['n_schmidt']}")
    print(f"  top-{top} λ²  : "
          + "  ".join(f"{v:.6f}" for v in lam2[:top]))


def compare_schmidt(psi_ucj, psi_exact, basis, n, cuts=None,
                    label_ucj="uCJ", label_exact="exact"):
    if cuts is None:
        cuts = list(range(1, n))

    rows = []
    print(f"\n{'─'*66}")
    print(f"  Entanglement entropy profile  [{label_ucj}] vs [{label_exact}]")
    print(f"  {'cut':>4}  {'S_vN (UCJ)':>12}  {'S_vN (exact)':>13}  "
          f"{'ΔS':>10}  {'S2 (UCJ)':>10}  {'S2 (exact)':>11}")
    print(f"  {'─'*62}")
    for c in cuts:
        sp_ucj   = schmidt_spectrum(psi_ucj,   basis, n, cut=c)
        sp_exact = schmidt_spectrum(psi_exact, basis, n, cut=c)
        dS = sp_ucj['entropy_vn'] - sp_exact['entropy_vn']
        print(f"  {c:>4}  {sp_ucj['entropy_vn']:>12.6f}  "
              f"{sp_exact['entropy_vn']:>13.6f}  {dS:>+10.6f}  "
              f"{sp_ucj['entropy_renyi2']:>10.6f}  "
              f"{sp_exact['entropy_renyi2']:>11.6f}")
        rows.append(dict(
            cut=c,
            S_ucj=sp_ucj['entropy_vn'],    S_exact=sp_exact['entropy_vn'],
            dS=dS,
            S2_ucj=sp_ucj['entropy_renyi2'], S2_exact=sp_exact['entropy_renyi2'],
            dS2=sp_ucj['entropy_renyi2'] - sp_exact['entropy_renyi2'],
            gap_ucj=sp_ucj['schmidt_gap'],   gap_exact=sp_exact['schmidt_gap'],
            n_schmidt_ucj=sp_ucj['n_schmidt'],
        ))
    print(f"  {'─'*62}")
    return rows


# =============================================================================
# FILE WRITERS
# =============================================================================
def write_summary_file(lat, j1, j2, e_exact, results, variants,
                       out_dir="summaries"):
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lat_tag = lat.name.replace(" ", "_").replace("/", "-")
    j2_tag  = f"{j2:.3f}".replace(".", "p")
    fname   = out_path / f"{lat_tag}_J2_{j2_tag}.txt"

    W    = 80
    sep  = "=" * W
    thin = "-" * W

    def _best_by_kind(tracker, kind):
        candidates = [r for lr in tracker.layers for r in lr.restarts
                      if r.kind == kind]
        return min(candidates, key=lambda r: r.E_final) if candidates else None

    lines = [sep,
             f"  UCJ COLD-START SUMMARY",
             f"  Lattice : {lat.name}   N={lat.n_sites}",
             f"  J1={j1:.4f}   J2={j2:.4f}",
             f"  E_exact = {e_exact:.10f}   E/site = {e_exact / lat.n_sites:.10f}",
             sep]

    for variant in variants:
        if variant not in results:
            continue
        res      = results[variant]
        tracker  = res["tracker"]
        rdm_conv = res["rdm_conv"]
        jax_best = res["jax_best"]
        overlap  = res["overlap"]["overlap"]

        cold_rec   = _best_by_kind(tracker, "cold")
        append_rec = _best_by_kind(tracker, "layer_append")

        lines += ["", f"  VARIANT : {variant}-uCJ   k_opt={jax_best['k']}", thin,
                  "  [ COLD START – best across all layers / restarts ]"]
        if cold_rec:
            dE_cold = abs(cold_rec.E_final - e_exact)
            lines += [f"    E_final   = {cold_rec.E_final:.10f}",
                      f"    |ΔE|      = {dE_cold:.6e}",
                      f"    variance  = {cold_rec.variance:.6e}",   # ← was fidelity
                      f"    nit / nfev= {cold_rec.nit} / {cold_rec.nfev}",
                      f"    wall_sec  = {cold_rec.wall_sec:.2f} s",
                      f"    converged = {cold_rec.converged}"]
        else:
            lines.append("    (no cold-start restarts recorded)")

        lines += ["", "  [ LAYER APPEND – best across all layers / restarts ]"]
        if append_rec:
            dE_app = abs(append_rec.E_final - e_exact)
            lines += [f"    E_final   = {append_rec.E_final:.10f}",
                      f"    |ΔE|      = {dE_app:.6e}",
                      f"    variance  = {append_rec.variance:.8f}",
                      f"    nit / nfev= {append_rec.nit} / {append_rec.nfev}",
                      f"    wall_sec  = {append_rec.wall_sec:.2f} s",
                      f"    converged = {append_rec.converged}"]
        else:
            lines.append("    (no layer-append restarts recorded)")

        lines += ["",
                  "  [ FROBENIUS NORMS @ CONVERGENCE ]",
                  f"    ‖Re(ρ_ij)‖_F = {rdm_conv['re_frob']:.8f}",
                  f"    ‖Im(ρ_ij)‖_F = {rdm_conv['im_frob']:.8f}",
                  f"    Im / Re ratio = {rdm_conv['ratio']:.8f}",
                  "",
                  "  [ OVERALL BEST (JAX L-BFGS) ]",
                  f"    E_best      = {jax_best['E']:.10f}",
                  f"    |ΔE|        = {abs(jax_best['E'] - e_exact):.6e}",
                  f"    overlap     = {overlap:.8f}",
                  f"    E_qiskit    = {res['E_pl']:.10f}",
                  thin]

    lines += ["", sep, f"  END OF SUMMARY  –  {lat.name}  J2={j2:.4f}", sep]
    text = "\n".join(lines) + "\n"
    fname.write_text(text, encoding="utf-8")
    print(f"[summary]  written → {fname}")

    # JSON sidecar
    json_data = {"lattice": lat.name, "n_sites": lat.n_sites,
                 "j1": j1, "j2": j2, "e_exact": e_exact, "variants": {}}
    for variant in variants:
        if variant not in results:
            continue
        res        = results[variant]
        tracker    = res["tracker"]
        rdm_conv   = res["rdm_conv"]
        jax_best   = res["jax_best"]
        cold_rec   = _best_by_kind(tracker, "cold")
        append_rec = _best_by_kind(tracker, "layer_append")

        def _rec(r):
            return (None if r is None else {
                "E_final": r.E_final, "abs_dE": abs(r.E_final - e_exact),
                 "nit": r.nit, "nfev": r.nfev,
                "wall_sec": r.wall_sec, "converged": r.converged})

        json_data["variants"][variant] = {
            "k_opt": jax_best["k"],
            "cold_best": _rec(cold_rec),
            "append_best": _rec(append_rec),
            "rdm_conv": {"re_frob": rdm_conv["re_frob"],
                         "im_frob": rdm_conv["im_frob"],
                         "ratio":   rdm_conv["ratio"]},
            "jax_overall": {
                "E": jax_best["E"], "abs_dE": abs(jax_best["E"] - e_exact),
                "overlap":  res["overlap"]["overlap"],
                "E_qiskit": res["E_pl"]},
        }

    json_fname = fname.with_suffix(".json")
    json_fname.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[summary]  JSON  → {json_fname}")
    return fname


def write_history_csv(history, variant, lattice_name, n, j1, j2,
                      s_vn_exact=float('nan'), s2_exact=float('nan'),
                      out_dir="layer_summaries"):
    lat_tag  = lattice_name.replace(" ", "_").replace("/", "-")
    j1_tag   = f"{j1:.3f}".replace(".", "p")
    j2_tag   = f"{j2:.3f}".replace(".", "p")
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = out_path / f"history_{variant}_{lat_tag}_N{n}_J1{j1_tag}_J2{j2_tag}.csv"

    fieldnames = ["k", "E", "deltaE", "fid",
                  "dE_layer", "dF_layer",
                  "dSvN_layer", "dS2_layer",
                  "S_vN", "S2",
                  "S_vN_exact", "S2_exact",
                  "dSvN_exact", "dS2_exact",
                  "schmidt_gap", "schmidt_values", "variance", "dVar_layer"]

    enriched = []
    for row in history:
        r = dict(row)
        r["S_vN_exact"] = s_vn_exact
        r["S2_exact"]   = s2_exact
        r["dSvN_exact"] = r["S_vN"] - s_vn_exact
        r["dS2_exact"]  = r["S2"]   - s2_exact
        enriched.append(r)

    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"  [history] wrote {len(history)} rows → {fpath}")
    return fpath


def write_entanglement_file(lat, j1, j2, e_exact, variant, jax_best,
                             ee_rows, mid_spec_ucj, mid_spec_exact,
                             out_dir="entanglement_info"):
    lat_tag  = lat.name.replace(" ", "_").replace("/", "-")
    j2_tag   = f"{j2:.3f}".replace(".", "p")
    out_path = pathlib.Path(out_dir) / lat_tag / f"J2_{j2_tag}"
    out_path.mkdir(parents=True, exist_ok=True)

    txt_file  = out_path / f"{variant}-uCJ.txt"
    json_file = out_path / f"{variant}-uCJ.json"

    n        = lat.n_sites
    k        = jax_best['k']
    W        = 76
    sep      = "=" * W
    thin     = "-" * W
    lam2_ucj = mid_spec_ucj['schmidt_values']
    lam2_ex  = mid_spec_exact['schmidt_values']

    def _triplet_var(lam2):
        return float(np.var(lam2[1:4])) if len(lam2) >= 4 else float('nan')

    def _cft_fit(rows):
        if "chain" not in lat.name.lower() or len(rows) < 4:
            return None
        cuts  = np.array([r['cut'] for r in rows], dtype=float)
        s_ucj = np.array([r['S_ucj'] for r in rows])
        s_ex  = np.array([r['S_exact'] for r in rows])
        chord = (n / np.pi) * np.sin(np.pi * cuts / n)
        lc    = np.log(chord)
        A     = np.column_stack([lc, np.ones_like(lc)])

        def _fit(s):
            coef, *_ = np.linalg.lstsq(A, s, rcond=None)
            pred = coef[0] * lc + coef[1]
            r2   = float(1 - np.sum((s - pred)**2) /
                             np.sum((s - np.mean(s))**2))
            return float(coef[0] * 3), float(coef[1]), r2

        c_u, b_u, r2_u = _fit(s_ucj)
        c_e, b_e, r2_e = _fit(s_ex)
        return dict(c_ucj=c_u, c_exact=c_e, const_ucj=b_u, const_exact=b_e,
                    r2_ucj=r2_u, r2_exact=r2_e)

    lines = [
        sep,
        f"  ENTANGLEMENT SUMMARY  –  {variant}-uCJ  (best, k={k})",
        f"  Lattice  : {lat.name}   N={n}",
        f"  J1={j1:.4f}   J2={j2:.4f}",
        f"  E_exact  = {e_exact:.10f}   E/site = {e_exact/n:.10f}",
        f"  E_best   = {jax_best['E']:.10f}   |ΔE| = {abs(jax_best['E']-e_exact):.4e}",
        sep, "",
        "  [ EE PROFILE  (von Neumann & Rényi-2) ]",
        thin,
        (f"  {'cut':>4}  {'S_vN (UCJ)':>12}  {'S_vN (exact)':>13}  "
         f"{'ΔS':>10}  {'S2 (UCJ)':>10}  {'S2 (exact)':>11}  "
         f"{'ΔS2':>10}  {'gap(UCJ)':>10}  {'χ':>5}"),
        "  " + thin,
    ]
    for row in ee_rows:
        lines.append(
            f"  {row['cut']:>4}  {row['S_ucj']:>12.8f}  {row['S_exact']:>13.8f}  "
            f"{row['dS']:>+10.6f}  {row['S2_ucj']:>10.8f}  {row['S2_exact']:>11.8f}  "
            f"{row['dS2']:>+10.6f}  {row['gap_ucj']:>10.8f}  {row['n_schmidt_ucj']:>5}")

    lines += ["  " + thin, "",
              f"  [ MID-CHAIN SCHMIDT SPECTRUM  (cut={mid_spec_ucj['cut']}) ]",
              thin,
              f"  {'metric':<28}  {'UCJ':>14}  {'exact':>14}  {'diff':>14}",
              "  " + thin,
              (f"  {'S_vN':<28}  {mid_spec_ucj['entropy_vn']:>14.8f}  "
               f"{mid_spec_exact['entropy_vn']:>14.8f}  "
               f"{mid_spec_ucj['entropy_vn']-mid_spec_exact['entropy_vn']:>+14.8f}"),
              (f"  {'S_Renyi2':<28}  {mid_spec_ucj['entropy_renyi2']:>14.8f}  "
               f"{mid_spec_exact['entropy_renyi2']:>14.8f}  "
               f"{mid_spec_ucj['entropy_renyi2']-mid_spec_exact['entropy_renyi2']:>+14.8f}"),
              (f"  {'Schmidt gap':<28}  {mid_spec_ucj['schmidt_gap']:>14.8f}  "
               f"{mid_spec_exact['schmidt_gap']:>14.8f}  "
               f"{mid_spec_ucj['schmidt_gap']-mid_spec_exact['schmidt_gap']:>+14.8f}"),
              (f"  {'n_Schmidt':<28}  {mid_spec_ucj['n_schmidt']:>14d}  "
               f"{mid_spec_exact['n_schmidt']:>14d}"),
              (f"  {'triplet var (λ²₁₋₃)':<28}  {_triplet_var(lam2_ucj):>14.6e}  "
               f"{_triplet_var(lam2_ex):>14.6e}"),
              "  " + thin, "",
              f"  [ TOP-{min(12,len(lam2_ucj))} SCHMIDT VALUES λ²  (cut={mid_spec_ucj['cut']}) ]",
              thin,
              f"  {'rank':>5}  {'λ² (UCJ)':>14}  {'λ² (exact)':>14}  {'diff':>14}",
              "  " + thin]

    for i in range(min(12, len(lam2_ucj))):
        u = float(lam2_ucj[i]) if i < len(lam2_ucj) else float('nan')
        e = float(lam2_ex[i])  if i < len(lam2_ex)  else float('nan')
        lines.append(f"  {i:>5}  {u:>14.8f}  {e:>14.8f}  {u-e:>+14.8f}")
    lines += ["  " + thin, ""]

    cft = _cft_fit(ee_rows)
    if cft:
        lines += [
            "  [ CFT AREA-LAW FIT  S ~ (c/3) ln[chord(ℓ)] + const ]",
            thin,
            f"  {'':28}  {'UCJ':>14}  {'exact':>14}",
            "  " + thin,
            f"  {'central charge c':<28}  {cft['c_ucj']:>14.6f}  {cft['c_exact']:>14.6f}",
            f"  {'const':<28}  {cft['const_ucj']:>14.6f}  {cft['const_exact']:>14.6f}",
            f"  {'R²':<28}  {cft['r2_ucj']:>14.6f}  {cft['r2_exact']:>14.6f}",
            thin, ""]

    lines += [sep, f"  END  –  {lat.name}  {variant}-uCJ  J2={j2:.4f}", sep]
    txt_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[entanglement]  txt  → {txt_file}")

    json_data = dict(
        lattice=lat.name, n_sites=n, j1=j1, j2=j2,
        variant=variant, k_opt=k,
        e_exact=e_exact, e_ucj=jax_best['E'],
        abs_dE=abs(jax_best['E'] - e_exact),
        ee_profile=ee_rows,
        mid_schmidt_ucj=dict(
            cut=mid_spec_ucj['cut'],
            entropy_vn=mid_spec_ucj['entropy_vn'],
            entropy_renyi2=mid_spec_ucj['entropy_renyi2'],
            schmidt_gap=mid_spec_ucj['schmidt_gap'],
            n_schmidt=mid_spec_ucj['n_schmidt'],
            triplet_var=_triplet_var(lam2_ucj),
            top12_lam2=lam2_ucj[:12].tolist(),
        ),
        mid_schmidt_exact=dict(
            cut=mid_spec_exact['cut'],
            entropy_vn=mid_spec_exact['entropy_vn'],
            entropy_renyi2=mid_spec_exact['entropy_renyi2'],
            schmidt_gap=mid_spec_exact['schmidt_gap'],
            n_schmidt=mid_spec_exact['n_schmidt'],
            triplet_var=_triplet_var(lam2_ex),
            top12_lam2=lam2_ex[:12].tolist(),
        ),
        cft_fit=cft,
    )
    json_file.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[entanglement]  json → {json_file}")
    return txt_file


def write_circuit_summary(variant, lattice_name, n, j1, j2, k,
                           gate_counts, depth, out_dir="circuit_summaries"):
    lat_tag  = lattice_name.replace(" ", "_").replace("/", "-")
    j1_tag   = f"{j1:.3f}".replace(".", "p")
    j2_tag   = f"{j2:.3f}".replace(".", "p")
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = out_path / f"circuit_{variant}_{lat_tag}_N{n}_J1{j1_tag}_J2{j2_tag}.txt"

    total = sum(gate_counts.values())
    W     = 60
    sep   = "=" * W
    thin  = "-" * W

    if not fpath.exists():
        header_lines = [
            sep,
            f"  CIRCUIT SUMMARY  –  {variant}-uCJ",
            f"  Lattice  : {lattice_name}   N={n}",
            f"  J1={j1:.4f}   J2={j2:.4f}",
            f"  Gate set : {_TARGET_BASIS}",
            sep, "",
        ]
        fpath.write_text("\n".join(header_lines) + "\n", encoding="utf-8")

    block_lines = [
        f"  [ k = {k} ]",
        thin,
        f"  {'depth':<28}  {depth:>8}",
        f"  {'total gates':<28}  {total:>8}",
        thin,
    ]
    for gate, count in sorted(gate_counts.items(), key=lambda x: (-x[1], x[0])):
        block_lines.append(f"  {gate:<28}  {count:>8}")
    block_lines += [thin, ""]

    with open(fpath, "a", encoding="utf-8") as f:
        f.write("\n".join(block_lines) + "\n")

    print(f"  [circuit_summary]  k={k}  → {fpath}")
    return fpath


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == '__main__':
    #from lattices import make_lattice   # user-supplied lattice module

    n_list  = [6]
    J2_list = [0.0]

    for n in n_list:
        timer = Timer()
        run(make_lattice('chain', L=n), variants=['re'],
            j1=1.0, j2=0.0, timer=timer)
        timer.summary(f"chain N={n}")
