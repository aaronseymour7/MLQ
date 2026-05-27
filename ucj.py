from __future__ import annotations

import os
import time
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize as scipy_minimize

import jax
import jax.numpy as jnp
import optax
import csv
from datetime import datetime

try:
    import pennylane as qml
    print(f"[PennyLane]  version {qml.version()}")
except ImportError:
    raise ImportError("pip install pennylane pennylane-lightning[gpu]")
from lattices import BaseLattice, make_lattice


# ---------------------------------------------------------------------------
# JAX setup
# ---------------------------------------------------------------------------
jax.config.update("jax_enable_x64", True)
_GPU_DEVICES = jax.devices("gpu")
_CPU_DEVICE  = jax.devices("cpu")[0]
_JAX_DEVICE  = _GPU_DEVICES[0] if _GPU_DEVICES else _CPU_DEVICE
print(f"[JAX]  using device: {_JAX_DEVICE}")

def _to_device(x):
    return jax.device_put(x, _JAX_DEVICE)

# ---------------------------------------------------------------------------
# PennyLane device
# ---------------------------------------------------------------------------
_BACKEND_PREF = os.environ.get("PENNYLANE_BACKEND", "lightning.qubit")

def _make_device(n_wires: int):
    bond_dim = int(os.environ.get("MPS_BOND_DIM", "64"))
    for backend in [_BACKEND_PREF, "lightning.qubit", "default.qubit"]:
        try:
            if backend == "lightning.tensor":
                dev = qml.device(backend, wires=n_wires,
                                 method="mps", max_bond_dim=bond_dim)
            else:
                dev = qml.device(backend, wires=n_wires)
            print(f"[PennyLane device]  N={n_wires}  backend={backend}")
            return dev, backend
        except Exception as exc:
            print(f"[PennyLane]  {backend} unavailable ({exc}), trying next…")
    raise RuntimeError("No PennyLane backend could be initialised.")


# =============================================================================
# CONFIG
# =============================================================================
J1          = 1.0
J2          = 0.0
PBC         = True
ALPHA       = 3
K_MAX       = 4
E_TOL       = 1e-6
SEED        = 23
N_RESTARTS  = 1

LBFGS_MAXITER = 800
LBFGS_MAXFUN  = 50_000

N_COLD_RESTARTS = 1

VARIANTS = ['re', 'im', 'g']

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
    """Stores the outcome of a single L-BFGS restart."""
    restart_idx:      int
    kind:             str        # "cold" | "layer_append"
    x0_norm:          float
    x0_jastrow_norm:  float
    x0_givens_norm:   float
    E_final:          float
    fidelity:         float
    nit:              int
    nfev:             int
    wall_sec:         float
    converged:        bool


@dataclass
class LayerRecord:
    """Collects all restarts for a single k-layer."""
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
    """
    Central diagnostic object. One instance per variant per run().
    """
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

    # ------------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------------
    def report(self):
        W    = 72
        sep  = "=" * W
        thin = "-" * W

        print(f"\n{sep}")
        print(f"  DIAGNOSTIC  |  {self.variant}-uCJ  N={self.n}")
        print(f"  E_exact = {self.e_exact:.8f}    e_tol = {self.e_tol:.2e}")
        print(sep)

        # ── Per-layer breakdown ───────────────────────────────────────
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
                      f"{r.E_final:>12.8f} {dE:>10.6f} {r.fidelity:>10.6f} "
                      f"{r.nit:>6} {r.wall_sec:>7.1f}{conv}")

        # ── x0 decomposition ─────────────────────────────────────────
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
# EXACT DIAGONALISATION  (CPU, arbitrary lattice edges)
# =============================================================================
def build_basis(n, n_up):
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up],
                    dtype=np.int64)


def build_hamiltonian(n, n_up, nn_edges, nnn_edges, j1=J1, j2=J2):
    basis   = build_basis(n, n_up)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    H       = lil_matrix((len(basis), len(basis)), dtype=np.float64)

    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = 0.5 if (bits >> si) & 1 else -0.5
                zj = 0.5 if (bits >> sj) & 1 else -0.5
                H[row, row] += j * zi * zj
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl  = bits ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(int(fl), -1)
                    if col >= 0:
                        H[row, col] += 0.5 * j

    return csr_matrix(H), basis, idx_map


def get_n_up(n):
    return (n + 1) // 2 if n % 2 == 1 else n // 2


# =============================================================================
# JAX HAMILTONIAN  (GPU, arbitrary lattice edges)
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
    @jax.jit
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
    pi_gpu = _to_device(jnp.array(pair_i_np, dtype=jnp.int32))
    pj_gpu = _to_device(jnp.array(pair_j_np, dtype=jnp.int32))

    if not JASTROW_CHUNKED:
        def _occ_product(i, j):
            bi = (basis_bits_gpu >> i.astype(jnp.int32)) & 1
            bj = (basis_bits_gpu >> j.astype(jnp.int32)) & 1
            return (bi * bj).astype(jnp.float64)

        _occ_mat_fn = jax.vmap(_occ_product)

        @jax.jit
        def jastrow_phase(theta_J):
            return jnp.dot(theta_J, _occ_mat_fn(pi_gpu, pj_gpu))
    else:
        n_pair = pair_i_np.shape[0]
        chunk  = JASTROW_CHUNK

        @jax.jit
        def jastrow_phase(theta_J):
            phase = jnp.zeros(basis_bits_gpu.shape[0], dtype=jnp.float64)
            for start in range(0, n_pair, chunk):
                end  = min(start + chunk, n_pair)
                pi_c = pi_gpu[start:end]
                pj_c = pj_gpu[start:end]
                th_c = theta_J[start:end]
                def _occ_c(i, j):
                    bi = (basis_bits_gpu >> i.astype(jnp.int32)) & 1
                    bj = (basis_bits_gpu >> j.astype(jnp.int32)) & 1
                    return (bi * bj).astype(jnp.float64)
                phase = phase + jnp.dot(th_c, jax.vmap(_occ_c)(pi_c, pj_c))
            return phase

    return jastrow_phase


def apply_jastrow(psi, theta_J, jastrow_phase_fn):
    return psi * jnp.exp(1j * jastrow_phase_fn(theta_J))


# =============================================================================
# GIVENS PAIRS
# =============================================================================
def build_givens_pairs(n, basis, idx_map):
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


def fidelity(theta, variant, k_layers, psi0, psi_exact_gpu,
             srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
             jastrow_phase_fn, n):
    n_pair    = n * (n - 1) // 2
    theta_gpu = _to_device(jnp.array(theta, dtype=jnp.float64))
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
    psi = _state_fns[variant](theta_gpu)
    return float(jnp.abs(jnp.dot(jnp.conj(psi_exact_gpu), psi)) ** 2)


# =============================================================================
# COLD INITIALISATION  (replaces warm_start)
# =============================================================================
def cold_start(variant, n, k_layers, seed=SEED, noise_scale=0.05):
    """
    Pure random initialisation — no VMC / RBM warm start.
    Returns a flat parameter vector of the correct size for the given
    variant and number of layers.
    """
    n_pair = n * (n - 1) // 2
    rng    = np.random.default_rng(seed)
    stride = 3 * n_pair if variant == 'g' else 2 * n_pair
    return noise_scale * rng.standard_normal(k_layers * stride)


# =============================================================================
# INSTRUMENTED L-BFGS  (diagnostic-aware optimiser)
# =============================================================================
def _optimise_layer_instrumented(
    tracker:      DiagnosticTracker,
    variant:      str,
    n:            int,
    k:            int,
    x0:           np.ndarray,
    x0_kind:      str,
    restart_idx:  int,
    e_exact:      float,
    e_tol:        float,
    val_grad_fn,
    fidelity_fn,
    n_pair:       int,
    lbfgs_maxiter: int = LBFGS_MAXITER,
    lbfgs_maxfun:  int = LBFGS_MAXFUN,
) -> tuple[np.ndarray, float, RestartRecord]:
    x0_jJ      = x0[:n_pair]
    x0_gK      = x0[n_pair:]
    x0_norm    = float(np.linalg.norm(x0))
    x0_jJ_norm = float(np.linalg.norm(x0_jJ))
    x0_gK_norm = float(np.linalg.norm(x0_gK))

    x0_gpu = _to_device(jnp.array(x0, dtype=jnp.float64))
    val_grad_fn(x0_gpu)

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
    fid   = fidelity_fn(opt_x)

    record = RestartRecord(
        restart_idx=restart_idx,
        kind=x0_kind,
        x0_norm=x0_norm,
        x0_jastrow_norm=x0_jJ_norm,
        x0_givens_norm=x0_gK_norm,
        E_final=opt_E,
        fidelity=fid,
        nit=result.nit,
        nfev=result.nfev,
        wall_sec=wall,
        converged=abs(opt_E - e_exact) < e_tol,
    )
    tracker.log_restart(k, record)

    print(f"  [{x0_kind:>14} restart={restart_idx}  k={k}]  "
          f"E={opt_E:.8f}  |ΔE|={abs(opt_E-e_exact):.4e}  "
          f"|<ψ|exact>|²={fid:.6f}  nit={result.nit}  nfev={result.nfev}  "
          f"t={wall:.1f}s")

    return opt_x, opt_E, record


def _cold_baseline(
    tracker:      DiagnosticTracker,
    k:            int,
    n_pair:       int,
    n_restarts:   int,
    e_exact:      float,
    e_tol:        float,
    val_grad_fn,
    fidelity_fn,
    noise_scale:  float = 0.05,
    seed:         int   = 9999,
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
            val_grad_fn=val_grad_fn, fidelity_fn=fidelity_fn,
            n_pair=n_pair)


# =============================================================================
# ADAPTIVE LAYER SEARCH  (instrumented, cold init only)
# =============================================================================
def adaptive_ucj(variant, n, k_max, e_tol, e_exact, psi_neel, psi_exact_gpu,
                 srcs_flat_gpu, dsts_flat_gpu, row_ptr_np, jastrow_phase_fn, apply_H,
                 tracker: DiagnosticTracker,
                 n_restarts=N_RESTARTS, n_cold_restarts=N_COLD_RESTARTS, basis = None):

    best = dict(E=np.inf, fid=0., params=None, k=1)
    prev_params = None
    n_pair = n * (n - 1) // 2
    history = []

    for k in range(1, k_max + 1):
        layer_best = dict(E=np.inf, params=None, fid=0.)
        val_grad_fn, _ = make_energy_grad(
            variant, n, k, psi_neel, srcs_flat_gpu, dsts_flat_gpu,
            row_ptr_np, jastrow_phase_fn, apply_H)

        def fid_fn(x):
            return fidelity(x, variant, k, psi_neel, psi_exact_gpu,
                            srcs_flat_gpu, dsts_flat_gpu, row_ptr_np, jastrow_phase_fn, n)

        for restart in range(n_restarts):
            rng = np.random.default_rng(SEED + k * 100 + restart)
            if prev_params is None:
                # First layer: pure cold start
                x0 = cold_start(variant, n, k, seed=SEED + restart)
                x0_kind = "cold"
            else:
                # Subsequent layers: append a cold new layer to the frozen best params
                noise = lambda m: 0.05 * rng.standard_normal(m)
                if variant == 'g':
                    new_layer = np.concatenate([noise(n_pair), noise(n_pair), noise(n_pair)])
                else:
                    new_layer = np.concatenate([noise(n_pair), noise(n_pair)])
                x0 = np.concatenate([prev_params, new_layer])
                x0_kind = "layer_append"

            opt_x, opt_E, record = _optimise_layer_instrumented(
                tracker=tracker, variant=variant, n=n, k=k, x0=x0, x0_kind=x0_kind,
                restart_idx=restart, e_exact=e_exact, e_tol=e_tol,
                val_grad_fn=val_grad_fn, fidelity_fn=fid_fn, n_pair=n_pair)

            if opt_E < layer_best['E']:
                layer_best.update(E=opt_E, params=opt_x, fid=record.fidelity)
            if record.converged:
                break

        if n_cold_restarts > 0:
            _cold_baseline(
                tracker=tracker, k=k, n_pair=n_pair, n_restarts=n_cold_restarts,
                e_exact=e_exact, e_tol=e_tol, val_grad_fn=val_grad_fn, fidelity_fn=fid_fn)

        prev_params = layer_best['params']
        if layer_best['E'] < best['E']:
            best.update(**layer_best, k=k)

        delta = abs(layer_best['E'] - e_exact)
        # ── EE spectrum at current layer ─────────────────────────────
        n_pair = n * (n - 1) // 2
        theta_gpu = _to_device(jnp.array(layer_best['params'], dtype=jnp.float64))

        _state_fns = {
            're': lambda th: ucj_state_re(
                th, k, psi_neel, n_pair,
                srcs_flat_gpu, dsts_flat_gpu,
                row_ptr_np, jastrow_phase_fn
            ),
            'im': lambda th: ucj_state_im(
                th, k, psi_neel, n_pair,
                srcs_flat_gpu, dsts_flat_gpu,
                row_ptr_np, jastrow_phase_fn
            ),
            'g': lambda th: ucj_state_g(
                th, k, psi_neel, n_pair,
                srcs_flat_gpu, dsts_flat_gpu,
                row_ptr_np, jastrow_phase_fn
            ),
        }

        psi_layer = np.array(
            _state_fns[variant](theta_gpu),
            dtype=np.complex128
        )
        psi_layer /= np.linalg.norm(psi_layer)

        spec = schmidt_spectrum(
            psi_layer,
            np.array(list(basis), dtype=np.int64),
            n,
        )

        print_schmidt(
            spec,
            label=f"{variant}-uCJ layer k={k}"
        )
        dE_layer = (history[-1]["E"] - layer_best["E"]) if history else None
        dF_layer = (layer_best["fid"] - history[-1]["fid"]) if history else None

        improvement = ""
        if dE_layer is not None:
            improvement = f"  ΔE_layer={dE_layer:+.4e}  ΔFid_layer={dF_layer:+.4e}"
        print(
            f"[{variant}-uCJ k={k}] "
            f"E={layer_best['E']:.8f}  |ΔE_exact|={delta:.4e}  "
            f"Fid={layer_best['fid']:.6f}"
            f"{improvement}"
        )

        history.append({
            "k": k,
            "E": layer_best["E"],
            "deltaE": delta,
            "fid": layer_best["fid"],
            "dE_layer": dE_layer,
            "dF_layer": dF_layer,
            "S_vN": spec["entropy_vn"],
            "S2": spec["entropy_renyi2"],
            "schmidt_gap": spec["schmidt_gap"],
            "schmidt_values": spec["schmidt_values"].tolist(),
        })

        if delta < e_tol:
            print(f"  ✓ Converged at k={k}.")
            break

    return best, history


# =============================================================================
# PENNYLANE CIRCUIT BUILDER
# =============================================================================
def _pl_xy(theta: float, q0: int, q1: int):
    """Unphased XX+YY rotation (imag Givens)."""
    qml.IsingXX(-theta, wires=[q0, q1])
    qml.IsingYY(-theta, wires=[q0, q1])


def _pl_xy_phased(theta: float, q0: int, q1: int):
    """Phased XX+YY rotation (real Givens)."""
    qml.RZ(-np.pi / 2, wires=q1)
    qml.IsingXX(-theta, wires=[q0, q1])
    qml.IsingYY(-theta, wires=[q0, q1])
    qml.RZ(np.pi / 2, wires=q1)


def build_ucj_pennylane(n: int, k_layers: int, variant: str, pairs=None):
    """Returns ucj_circuit(params) as a PennyLane-compatible callable."""
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pair = len(pairs)
    stride = 3 * n_pair if variant == 'g' else 2 * n_pair

    def ucj_circuit(params):
        for i in range(n):
            if i % 2 == 0:
                qml.PauliX(wires=i)

        for l in range(k_layers):
            off  = l * stride
            tJ   = params[off          : off + n_pair]
            tK_r = params[off + n_pair : off + 2 * n_pair]
            tK_i = (params[off + 2 * n_pair : off + 3 * n_pair]
                    if variant == 'g' else None)

            for k, (i, j) in enumerate(pairs):
                qml.ControlledPhaseShift(float(tJ[k]), wires=[i, j])

            if variant == 'im':
                for k, (i, j) in enumerate(pairs):
                    _pl_xy(float(tK_r[k]), j, i)
            else:
                for k, (i, j) in enumerate(pairs):
                    _pl_xy_phased(float(tK_r[k]), j, i)
                if variant == 'g':
                    for k, (i, j) in enumerate(pairs):
                        _pl_xy(float(tK_i[k]), j, i)

    return ucj_circuit


# =============================================================================
# PENNYLANE HAMILTONIAN  (arbitrary lattice edges)
# =============================================================================
def build_heisenberg_pennylane(n: int, nn_edges: list, nnn_edges: list,
                                j1: float = J1, j2: float = J2):
    coeffs, obs = [], []
    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for Qop in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeffs.append(j / 4.0)
                obs.append(Qop(wires=si) @ Qop(wires=sj))

    return coeffs, obs


# =============================================================================
# NOISELESS PENNYLANE ENERGY
# =============================================================================
def energy_noiseless_pl(n: int, k_layers: int, variant: str,
                         params: np.ndarray,
                         nn_edges: list, nnn_edges: list,
                         j1: float = J1, j2: float = J2,
                         pairs=None) -> float:
    bond_dim = int(os.environ.get("MPS_BOND_DIM", "64"))
    try:
        dev = qml.device("lightning.qubit", wires=n,
                         method="mps", max_bond_dim=bond_dim)
    except Exception:
        dev = qml.device("default.qubit", wires=n)

    coeffs, obs = build_heisenberg_pennylane(n, nn_edges, nnn_edges, j1, j2)
    circuit     = build_ucj_pennylane(n, k_layers, variant, pairs)

    @qml.qnode(dev, diff_method="best")
    def qnode(p):
        circuit(p)
        return qml.expval(qml.dot(coeffs, obs))

    return float(qnode(params))


# =============================================================================
# STATE OVERLAP
# =============================================================================
def state_overlap_pl(params: np.ndarray, variant: str, k_layers: int,
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


def rdm_norms_at_convergence(params: np.ndarray, variant: str, k_layers: int,
                              psi_neel, srcs_flat_gpu, dsts_flat_gpu,
                              row_ptr_np, jastrow_phase_fn, n: int,
                              basis, bindex) -> dict:
    n_pair    = n * (n - 1) // 2
    theta_gpu = _to_device(jnp.array(params, dtype=jnp.float64))

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
    psi      = np.array(_state_fns[variant](theta_gpu), dtype=np.complex128)
    psi     /= np.linalg.norm(psi)
    basis_np = list(basis)
    idx_map  = bindex

    basis_arr = np.array(
        [[(1 if (b >> s) & 1 else -1) for s in range(n)] for b in basis_np],
        dtype=np.float64)
    occ  = (basis_arr + 1) / 2
    probs = np.abs(psi) ** 2

    rho = np.zeros((n, n), dtype=np.complex128)

    n_mean = (probs[:, None] * occ).sum(0)
    for i in range(n):
        rho[i, i] = n_mean[i]

    for i in range(n):
        for j in range(i + 1, n):
            mask = (basis_arr[:, i] == 1) & (basis_arr[:, j] == -1)
            if not mask.any():
                continue
            sigma_v      = basis_arr[mask]
            flipped_bits = np.array(basis_np)[mask]
            new_bits     = (flipped_bits ^ (1 << i) ^ (1 << j)).astype(np.int64)

            col_indices = np.array([idx_map.get(int(b), -1) for b in new_bits])
            valid       = col_indices >= 0
            if not valid.any():
                continue

            rows_v  = np.where(mask)[0][valid]
            rows_f  = col_indices[valid]
            jw_signs = np.array([
                (-1) ** int(((sigma_v[k, i+1:j] + 1) / 2).sum())
                for k in range(sigma_v.shape[0])
            ])[valid]

            rho_ij       = (np.conj(psi[rows_f]) * psi[rows_v] * jw_signs).sum()
            rho[i, j]    = rho_ij
            rho[j, i]    = np.conj(rho_ij)

    mask_offdiag       = ~np.eye(n, dtype=bool)
    re_frob = np.linalg.norm(np.real(rho)[mask_offdiag])
    im_frob = np.linalg.norm(np.imag(rho)[mask_offdiag])
    ratio   = im_frob / (re_frob + 1e-12)

    print(f"\n[RDM @ convergence  {variant}-uCJ  N={n}  k={k_layers}]")
    print(f"  ‖Re(ρ_ij)‖_F = {re_frob:.6f}")
    print(f"  ‖Im(ρ_ij)‖_F = {im_frob:.6f}")
    print(f"  Im/Re ratio  = {ratio:.6f}"
          + ("  ← Im negligible" if ratio < 0.1 else
             "  ← Im moderate"   if ratio < 0.5 else
             "  ← Im significant"))

    return dict(re_frob=re_frob, im_frob=im_frob, ratio=ratio, rho=rho)


# =============================================================================
# CIRCUIT INFO
# =============================================================================
def circuit_info_pl(n: int, k_layers: int, variant: str,
                    params: np.ndarray, pairs=None):
    circuit = build_ucj_pennylane(n, k_layers, variant, pairs)
    tape    = qml.tape.make_qscript(circuit)(params)

    gate_counts = {}
    for op in tape.operations:
        gate_counts[op.name] = gate_counts.get(op.name, 0) + 1

    print(f"\n[Circuit info: {variant}-uCJ  N={n}  k={k_layers}]")
    print(f"  depth={len(tape.operations)}  "
          f"total_gates={sum(gate_counts.values())}")
    for gate, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
        print(f"    {gate:<30} {count}")
    return gate_counts


def print_ucj_operators(n: int, k_layers: int, variant: str,
                        params: np.ndarray, pairs=None):
    circuit = build_ucj_pennylane(n, k_layers, variant, pairs)
    tape    = qml.tape.make_qscript(circuit)(params)
    print(f"\n[UCJ operators: {variant}-uCJ  N={n}  k={k_layers}]")
    print(f"  {'#':<5} {'Gate':<30} {'Wires':<15} {'Params'}")
    print(f"  {'─'*70}")
    for i, op in enumerate(tape.operations):
        param_str = (f"{[round(float(p), 4) for p in op.parameters]}"
                     if op.parameters else "—")
        print(f"  {i:<5} {op.name:<30} {str(list(op.wires)):<15} {param_str}")
    print(f"\n  Total operators: {len(tape.operations)}")


# =============================================================================
# SHARED QUANTUM STRUCTURES
# =============================================================================
def _build_quantum_structures(lattice: BaseLattice,
                               j1: float = J1, j2: float = J2) -> dict:
    n          = lattice.n_sites
    nn_edges   = lattice.nn_edges
    nnn_edges  = lattice.nnn_edges
    n_up       = get_n_up(n)

    H_sp, basis, bindex = build_hamiltonian(n, n_up, nn_edges, nnn_edges,
                                             j1=j1, j2=j2)
    evals, evecs = eigsh(H_sp, k=1, which='SA')
    e_exact      = float(evals[0])
    psi_exact_np = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    del H_sp, evals, evecs

    h_rows, h_cols, h_vals = build_jax_hamiltonian(n, n_up, nn_edges, nnn_edges,
                                                    j1=j1, j2=j2)
    apply_H = make_apply_H(h_rows, h_cols, h_vals, len(basis))
    del h_rows, h_cols, h_vals

    srcs_flat_gpu, dsts_flat_gpu, row_ptr_np = build_givens_pairs(
        n, list(basis), bindex)

    pair_i_np, pair_j_np, basis_bits_gpu = build_jastrow_indices(n, basis)
    jastrow_phase_fn = _make_jastrow_phase_fn(pair_i_np, pair_j_np, basis_bits_gpu)

    psi_neel      = neel_state(n, n_up, list(basis), bindex)
    psi_exact_gpu = _to_device(jnp.array(psi_exact_np, dtype=jnp.complex128))

    return dict(
        n=n, n_up=n_up, nn_edges=nn_edges, nnn_edges=nnn_edges,
        basis=basis, bindex=bindex,
        e_exact=e_exact, psi_exact_np=psi_exact_np, psi_exact_gpu=psi_exact_gpu,
        apply_H=apply_H,
        srcs_flat_gpu=srcs_flat_gpu, dsts_flat_gpu=dsts_flat_gpu,
        row_ptr_np=row_ptr_np, jastrow_phase_fn=jastrow_phase_fn,
        psi_neel=psi_neel,
    )


# =============================================================================
# SUMMARY PRINTER
# =============================================================================
def _print_summary(n, e_exact, results, variants,
                   lattice_name: str = "", j1: float = J1, j2: float = J2):
    col  = 14
    W    = max(80, 28 + col * len(variants) + 2)
    rfmt = f"  {{:<28}}" + f"{{:>{col}}}" * len(variants)
    print(f"\n{'='*W}")
    print(f"  SUMMARY  {lattice_name}  N={n}  J1={j1}  J2={j2}  "
          f"E_exact={e_exact:.6f}  JAX={_JAX_DEVICE}")
    print(f"{'='*W}")
    print(rfmt.format("metric", *[f"{v}-uCJ" for v in variants]))
    print("  " + "─" * (28 + col * len(variants)))

    def _row(label, vals):
        print(rfmt.format(label[:28],
                          *[f"{vals.get(v, float('nan')):.6f}" for v in variants]))

    print(rfmt.format("E_exact", *[f"{e_exact:.6f}"] * len(variants)))
    _row("E_theory (JAX)",   {v: results[v]['jax_best']['E']     for v in variants})
    _row("E_pl (noiseless)", {v: results[v]['E_pl']              for v in variants})
    _row("overlap",          {v: results[v]['overlap']['overlap'] for v in variants})
    _row("‖Re(ρ)‖_F @ conv",  {v: results[v]['rdm_conv']['re_frob'] for v in variants})
    _row("‖Im(ρ)‖_F @ conv",  {v: results[v]['rdm_conv']['im_frob'] for v in variants})
    _row("Im/Re ratio @ conv", {v: results[v]['rdm_conv']['ratio']   for v in variants})
    print(f"{'='*W}\n")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def run(lattice: BaseLattice,
        variants=VARIANTS, k_max: int = K_MAX,
        e_tol: float = E_TOL, timer: Timer | None = None,
        j1: float = J1, j2: float = J2):

    n = lattice.n_sites
    W = 70
    print("\n" + "=" * W)
    print(f"  run  |  lattice={lattice.name}  N={n}  J1={j1}  J2={j2}  "
          f"variants={variants}  JAX={_JAX_DEVICE}")
    print(f"       |  PL_backend={_BACKEND_PREF}")
    print("=" * W)

    qs = _build_quantum_structures(lattice, j1=j1, j2=j2)
    n                = qs['n']
    nn_edges         = qs['nn_edges']
    nnn_edges        = qs['nnn_edges']
    e_exact          = qs['e_exact']
    psi_exact_np     = qs['psi_exact_np']
    psi_exact_gpu    = qs['psi_exact_gpu']
    apply_H          = qs['apply_H']
    srcs_flat_gpu    = qs['srcs_flat_gpu']
    dsts_flat_gpu    = qs['dsts_flat_gpu']
    row_ptr_np       = qs['row_ptr_np']
    jastrow_phase_fn = qs['jastrow_phase_fn']
    psi_neel         = qs['psi_neel']
    basis            = qs['basis']
    bindex           = qs['bindex']

    print(f"\n[Lanczos]  {lattice.name}  N={n}  J1={j1}  J2={j2}  "
          f"E_exact={e_exact:.8f}  E/site={e_exact/n:.8f}")

    results = {}
    for variant in variants:
        print(f"\n{'─'*W}")
        print(f"  JAX L-BFGS  |  {variant}-uCJ  {lattice.name}  N={n}  J1={j1}  J2={j2}")
        print(f"{'─'*W}")

        tracker = DiagnosticTracker(
            e_exact=e_exact, n=n, variant=variant, e_tol=e_tol)

        label = f"jax_lbfgs {variant} {lattice.name} J2={j2}"
        if timer: timer.start(label)
        jax_best, history = adaptive_ucj(
            variant, n, k_max, e_tol, e_exact,
            psi_neel, psi_exact_gpu,
            srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
            jastrow_phase_fn, apply_H,
            tracker=tracker, basis = basis)
        if timer: timer.stop(label)

        tracker.report()

        params = np.array(jax_best['params'])

        label = f"pl_eval {variant} {lattice.name} J2={j2}"
        if timer: timer.start(label)
        E_pl = energy_noiseless_pl(n, jax_best['k'], variant, params,
                                   nn_edges, nnn_edges, j1, j2)
        if timer: timer.stop(label)

        print(f"\n[{variant}-uCJ]  E_theory={jax_best['E']:.8f}"
              f"  E_pl={E_pl:.8f}  k_opt={jax_best['k']}")

        circuit_info_pl(n, jax_best['k'], variant, params)

        ov = state_overlap_pl(
            params, variant, jax_best['k'],
            psi_neel, psi_exact_np,
            srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
            jastrow_phase_fn, n)
        ov['E_pl'] = E_pl

        rdm_conv = rdm_norms_at_convergence(
            params, variant, jax_best['k'],
            psi_neel, srcs_flat_gpu, dsts_flat_gpu, row_ptr_np,
            jastrow_phase_fn, n, basis, bindex)
        
        results[variant] = dict(jax_best=jax_best, params=params,
                                E_pl=E_pl, overlap=ov,
                                rdm_conv=rdm_conv, tracker=tracker)
        # ── Schmidt spectrum ──────────────────────────────────────────────────
        # Reconstruct the normalised UCJ state vector
        n_pair    = n * (n - 1) // 2
        theta_gpu = _to_device(jnp.array(params, dtype=jnp.float64))
        _state_fns = {
            're': lambda th: ucj_state_re(th, jax_best['k'], psi_neel, n_pair,
                                          srcs_flat_gpu, dsts_flat_gpu,
                                          row_ptr_np, jastrow_phase_fn),
            'im': lambda th: ucj_state_im(th, jax_best['k'], psi_neel, n_pair,
                                          srcs_flat_gpu, dsts_flat_gpu,
                                          row_ptr_np, jastrow_phase_fn),
            'g':  lambda th: ucj_state_g(th, jax_best['k'], psi_neel, n_pair,
                                          srcs_flat_gpu, dsts_flat_gpu,
                                          row_ptr_np, jastrow_phase_fn),
        }
        psi_ucj_np = np.array(_state_fns[variant](theta_gpu), dtype=np.complex128)
        psi_ucj_np /= np.linalg.norm(psi_ucj_np)

        ee_rows = compare_schmidt(
            psi_ucj_np, psi_exact_np,
            np.array(list(basis), dtype=np.int64), n,
            label_ucj=f"{variant}-uCJ k={jax_best['k']}",
        )

        # mid-chain spectrum for the summary
        mid_spec_ucj   = schmidt_spectrum(psi_ucj_np,  np.array(list(basis)), n)
        mid_spec_exact = schmidt_spectrum(psi_exact_np, np.array(list(basis)), n)
        write_entanglement_file(
            lat=lattice, j1=j1, j2=j2,
            e_exact=e_exact, variant=variant,
            jax_best=jax_best, ee_rows=ee_rows,
            mid_spec_ucj=mid_spec_ucj,
            mid_spec_exact=mid_spec_exact,
        )
        print_schmidt(mid_spec_ucj,   label=f"{variant}-uCJ")
        print_schmidt(mid_spec_exact, label="exact ED")

        results[variant]['ee_profile']    = ee_rows
        results[variant]['mid_spec_ucj']  = mid_spec_ucj
        results[variant]['mid_spec_exact'] = mid_spec_exact
        write_history_csv(history, variant, lattice.name, n, j1, j2)

    _print_summary(n, e_exact, results, variants,
                   lattice_name=lattice.name, j1=j1, j2=j2)
    write_summary_file(lattice, j1, j2, e_exact, results, variants)
    return results



# =============================================================================
# SUMMARY FILE WRITER
# =============================================================================
import json
import pathlib

def write_summary_file(
    lat: "BaseLattice",
    j1: float,
    j2: float,
    e_exact: float,
    results: dict,
    variants: list[str],
    out_dir: str = "summaries",
) -> pathlib.Path:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    lat_tag = lat.name.replace(" ", "_").replace("/", "-")
    j2_tag  = f"{j2:.3f}".replace(".", "p")
    fname   = out_path / f"{lat_tag}_J2_{j2_tag}.txt"

    W    = 80
    sep  = "=" * W
    thin = "-" * W

    def _best_by_kind(tracker, kind: str):
        candidates = [
            r
            for lr in tracker.layers
            for r in lr.restarts
            if r.kind == kind
        ]
        return min(candidates, key=lambda r: r.E_final) if candidates else None

    lines = []
    lines.append(sep)
    lines.append(f"  UCJ COLD-START SUMMARY")
    lines.append(f"  Lattice : {lat.name}   N={lat.n_sites}")
    lines.append(f"  J1={j1:.4f}   J2={j2:.4f}")
    lines.append(f"  E_exact = {e_exact:.10f}   E/site = {e_exact / lat.n_sites:.10f}")
    lines.append(sep)

    for variant in variants:
        if variant not in results:
            continue

        res      = results[variant]
        tracker  = res["tracker"]
        rdm_conv = res["rdm_conv"]
        jax_best = res["jax_best"]
        overlap  = res["overlap"]["overlap"]

        cold_rec        = _best_by_kind(tracker, "cold")
        append_rec      = _best_by_kind(tracker, "layer_append")

        lines.append("")
        lines.append(f"  VARIANT : {variant}-uCJ   k_opt={jax_best['k']}")
        lines.append(thin)

        # -- cold start --
        lines.append("  [ COLD START – best across all layers / restarts ]")
        if cold_rec is not None:
            dE_cold = abs(cold_rec.E_final - e_exact)
            lines.append(f"    E_final   = {cold_rec.E_final:.10f}")
            lines.append(f"    |ΔE|      = {dE_cold:.6e}")
            lines.append(f"    fidelity  = {cold_rec.fidelity:.8f}")
            lines.append(f"    nit / nfev= {cold_rec.nit} / {cold_rec.nfev}")
            lines.append(f"    wall_sec  = {cold_rec.wall_sec:.2f} s")
            lines.append(f"    converged = {cold_rec.converged}")
        else:
            lines.append("    (no cold-start restarts recorded)")

        lines.append("")

        # -- layer append --
        lines.append("  [ LAYER APPEND – best across all layers / restarts ]")
        if append_rec is not None:
            dE_app = abs(append_rec.E_final - e_exact)
            lines.append(f"    E_final   = {append_rec.E_final:.10f}")
            lines.append(f"    |ΔE|      = {dE_app:.6e}")
            lines.append(f"    fidelity  = {append_rec.fidelity:.8f}")
            lines.append(f"    nit / nfev= {append_rec.nit} / {append_rec.nfev}")
            lines.append(f"    wall_sec  = {append_rec.wall_sec:.2f} s")
            lines.append(f"    converged = {append_rec.converged}")
        else:
            lines.append("    (no layer-append restarts recorded)")

        lines.append("")

        # -- frobenius norms at convergence --
        lines.append("  [ FROBENIUS NORMS @ CONVERGENCE  (best preparation) ]")
        lines.append(f"    ‖Re(ρ_ij)‖_F = {rdm_conv['re_frob']:.8f}")
        lines.append(f"    ‖Im(ρ_ij)‖_F = {rdm_conv['im_frob']:.8f}")
        lines.append(f"    Im / Re ratio = {rdm_conv['ratio']:.8f}")
        lines.append("")

        # -- overall best --
        lines.append("  [ OVERALL BEST (JAX L-BFGS) ]")
        lines.append(f"    E_best      = {jax_best['E']:.10f}")
        lines.append(f"    |ΔE|        = {abs(jax_best['E'] - e_exact):.6e}")
        lines.append(f"    fidelity    = {jax_best['fid']:.8f}")
        lines.append(f"    overlap     = {overlap:.8f}")
        lines.append(f"    E_pl        = {res['E_pl']:.10f}")
        lines.append(thin)

    lines.append("")
    lines.append(sep)
    lines.append(f"  END OF SUMMARY  –  {lat.name}  J2={j2:.4f}")
    lines.append(sep)

    text = "\n".join(lines) + "\n"
    fname.write_text(text, encoding="utf-8")
    print(f"[summary]  written → {fname}")

    # JSON sidecar
    json_data = {
        "lattice": lat.name,
        "n_sites": lat.n_sites,
        "j1": j1,
        "j2": j2,
        "e_exact": e_exact,
        "variants": {},
    }
    for variant in variants:
        if variant not in results:
            continue
        res      = results[variant]
        tracker  = res["tracker"]
        rdm_conv = res["rdm_conv"]
        jax_best = res["jax_best"]
        cold_rec   = _best_by_kind(tracker, "cold")
        append_rec = _best_by_kind(tracker, "layer_append")

        def _rec_to_dict(r):
            if r is None:
                return None
            return {
                "E_final":   r.E_final,
                "abs_dE":    abs(r.E_final - e_exact),
                "fidelity":  r.fidelity,
                "nit":       r.nit,
                "nfev":      r.nfev,
                "wall_sec":  r.wall_sec,
                "converged": r.converged,
            }

        json_data["variants"][variant] = {
            "k_opt":        jax_best["k"],
            "cold_best":    _rec_to_dict(cold_rec),
            "append_best":  _rec_to_dict(append_rec),
            "rdm_conv": {
                "re_frob": rdm_conv["re_frob"],
                "im_frob": rdm_conv["im_frob"],
                "ratio":   rdm_conv["ratio"],
            },
            "jax_overall": {
                "E":        jax_best["E"],
                "abs_dE":   abs(jax_best["E"] - e_exact),
                "fidelity": jax_best["fid"],
                "overlap":  res["overlap"]["overlap"],
                "E_pl":     res["E_pl"],
            },
        }

    json_fname = fname.with_suffix(".json")
    json_fname.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[summary]  JSON  → {json_fname}")

    return fname


def write_history_csv(
    history: list[dict],
    variant: str,
    lattice_name: str,
    n: int,
    j1: float,
    j2: float,
    out_dir: str = "layer_summaries",
) -> pathlib.Path:
    lat_tag = lattice_name.replace(" ", "_").replace("/", "-")
    j1_tag  = f"{j1:.3f}".replace(".", "p")
    j2_tag  = f"{j2:.3f}".replace(".", "p")

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = out_path / f"history_{variant}_{lat_tag}_N{n}_J1{j1_tag}_J2{j2_tag}.csv"

    fieldnames = [
    "k",
    "E",
    "deltaE",
    "fid",
    "dE_layer",
    "dF_layer",
    "S_vN",
    "S2",
    "schmidt_gap",
    "schmidt_values",
    ]
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    print(f"  [history] wrote {len(history)} rows → {fpath}")
    return fpath

# =============================================================================
# SCHMIDT SPECTRUM  (bipartite, fixed n_up sector)
# =============================================================================
import itertools

def _split_basis(basis_np: np.ndarray, n: int, cut: int):
    """
    Map each basis state (integer bit-string) to a (left, right) pair of
    indices for the bipartition  [0..cut-1 | cut..n-1].

    Returns
    -------
    left_bits  : np.ndarray[int]   occupation patterns on sites 0..cut-1
    right_bits : np.ndarray[int]   occupation patterns on sites cut..n-1
    left_idx   : np.ndarray[int]   dense row index (0..2^cut - 1)
    right_idx  : np.ndarray[int]   dense col index (0..2^(n-cut) - 1)
    """
    mask_left  = (1 << cut) - 1
    left_bits  = basis_np & mask_left
    right_bits = basis_np >> cut
    return left_bits, right_bits, left_bits.astype(int), right_bits.astype(int)


def schmidt_spectrum(
    psi: np.ndarray,          # complex128, norm=1, length=dim(n_up sector)
    basis: np.ndarray,        # int64, computational-basis integers in sector
    n: int,
    cut: int | None = None,   # default: n//2
    zero_thresh: float = 1e-14,
) -> dict:
    """
    Compute the Schmidt decomposition of |psi> across a left/right bipartition.

    Parameters
    ----------
    psi     : state vector in the fixed-n_up basis (already normalised)
    basis   : integer bit-strings corresponding to each component of psi
    n       : total number of sites
    cut     : bond position; sites 0..cut-1 are "left"
    zero_thresh : singular values below this are discarded

    Returns
    -------
    dict with keys:
        schmidt_values  : np.ndarray  (λ_i, already squared = eigenvalues of ρ_L)
        entropy_vn      : float       von Neumann entropy S = -Σ λ² ln λ²
        entropy_renyi2  : float       Rényi-2   S2 = -ln Σ λ⁴
        schmidt_gap     : float       λ²_0 - λ²_1  (gap in reduced spectrum)
        n_schmidt       : int         number of non-negligible Schmidt values
        cut             : int
    """
    if cut is None:
        cut = n // 2

    basis_np = np.asarray(basis, dtype=np.int64)
    psi_np   = np.asarray(psi,   dtype=np.complex128)

    lb, rb, li, ri = _split_basis(basis_np, n, cut)

    # Build dense bipartite matrix  Ψ[left, right]
    dim_l = 1 << cut
    dim_r = 1 << (n - cut)
    Psi   = np.zeros((dim_l, dim_r), dtype=np.complex128)
    Psi[li, ri] = psi_np

    # SVD  (economy)
    sv = np.linalg.svd(Psi, compute_uv=False, full_matrices=False)
    sv = sv[sv > zero_thresh]

    lam2        = sv ** 2          # eigenvalues of reduced density matrix
    lam2       /= lam2.sum()       # re-normalise to cure floating-point drift

    s_vn        = float(-np.sum(lam2 * np.log(lam2 + 1e-300)))
    s_renyi2    = float(-np.log(np.sum(lam2 ** 2) + 1e-300))
    gap         = float(lam2[0] - lam2[1]) if len(lam2) > 1 else float(lam2[0])

    return dict(
        schmidt_values=lam2,
        entropy_vn=s_vn,
        entropy_renyi2=s_renyi2,
        schmidt_gap=gap,
        n_schmidt=len(lam2),
        cut=cut,
    )


def schmidt_profile(
    psi: np.ndarray,
    basis: np.ndarray,
    n: int,
    cuts: list[int] | None = None,
) -> list[dict]:
    """
    Compute Schmidt spectra across every bond (or a chosen subset).
    Useful for spotting whether EE peaks at the centre or is uniform.
    """
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


def compare_schmidt(
    psi_ucj:    np.ndarray,
    psi_exact:  np.ndarray,
    basis:      np.ndarray,
    n:          int,
    cuts:       list[int] | None = None,
    label_ucj:  str = "uCJ",
    label_exact: str = "exact",
) -> list[dict]:
    """
    Compare full EE profiles of the UCJ ansatz vs the exact ground state.
    Returns a list of dicts, one per cut, each with keys:
        cut, S_ucj, S_exact, dS, S2_ucj, S2_exact, dS2
    """
    if cuts is None:
        cuts = list(range(1, n))

    rows = []
    print(f"\n{'─'*66}")
    print(f"  Entanglement entropy profile  "
          f"[{label_ucj}] vs [{label_exact}]")
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
            S_ucj=sp_ucj['entropy_vn'],   S_exact=sp_exact['entropy_vn'],
            dS=dS,
            S2_ucj=sp_ucj['entropy_renyi2'], S2_exact=sp_exact['entropy_renyi2'],
            dS2=sp_ucj['entropy_renyi2'] - sp_exact['entropy_renyi2'],
            gap_ucj=sp_ucj['schmidt_gap'],   gap_exact=sp_exact['schmidt_gap'],
            n_schmidt_ucj=sp_ucj['n_schmidt'],
        ))
    print(f"  {'─'*62}")
    return rows


def write_entanglement_file(
    lat:            "BaseLattice",
    j1:             float,
    j2:             float,
    e_exact:        float,
    variant:        str,
    jax_best:       dict,
    ee_rows:        list[dict],
    mid_spec_ucj:   dict,
    mid_spec_exact: dict,
    out_dir:        str = "entanglement_info",
) -> pathlib.Path:
    """
    One .txt + .json per (variant, lattice, J2), using the best params only.

    Directory layout:
        entanglement_info/
            chain_L=8/
                J2_0p000/
                    re-uCJ.txt   re-uCJ.json
                    im-uCJ.txt   im-uCJ.json
                    g-uCJ.txt    g-uCJ.json
    """
    lat_tag = lat.name.replace(" ", "_").replace("/", "-")
    j2_tag  = f"{j2:.3f}".replace(".", "p")

    out_path = pathlib.Path(out_dir) / lat_tag / f"J2_{j2_tag}"
    out_path.mkdir(parents=True, exist_ok=True)

    txt_file  = out_path / f"{variant}-uCJ.txt"
    json_file = out_path / f"{variant}-uCJ.json"

    n         = lat.n_sites
    k         = jax_best['k']
    W         = 76
    sep       = "=" * W
    thin      = "-" * W
    lam2_ucj  = mid_spec_ucj['schmidt_values']
    lam2_ex   = mid_spec_exact['schmidt_values']

    def _triplet_var(lam2):
        return float(np.var(lam2[1:4])) if len(lam2) >= 4 else float('nan')

    def _cft_fit(rows):
        if "chain" not in lat.name.lower() or len(rows) < 4:
            return None
        cuts  = np.array([r['cut']    for r in rows], dtype=float)
        s_ucj = np.array([r['S_ucj']  for r in rows])
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

    # ── text ──────────────────────────────────────────────────────────────────
    lines = [
        sep,
        f"  ENTANGLEMENT SUMMARY  –  {variant}-uCJ  (best, k={k})",
        f"  Lattice  : {lat.name}   N={n}",
        f"  J1={j1:.4f}   J2={j2:.4f}",
        f"  E_exact  = {e_exact:.10f}   E/site = {e_exact/n:.10f}",
        f"  E_best   = {jax_best['E']:.10f}   |ΔE| = {abs(jax_best['E']-e_exact):.4e}",
        f"  fidelity = {jax_best['fid']:.8f}   k_opt = {k}",
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
            f"{row['dS2']:>+10.6f}  {row['gap_ucj']:>10.8f}  {row['n_schmidt_ucj']:>5}"
        )
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
        "  " + thin,
    ]
    for i in range(min(12, len(lam2_ucj))):
        u = float(lam2_ucj[i]) if i < len(lam2_ucj) else float('nan')
        e = float(lam2_ex[i])  if i < len(lam2_ex)  else float('nan')
        lines.append(
            f"  {i:>5}  {u:>14.8f}  {e:>14.8f}  {u-e:>+14.8f}"
        )
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
            thin, "",
        ]

    lines += [sep, f"  END  –  {lat.name}  {variant}-uCJ  J2={j2:.4f}", sep]

    txt_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[entanglement]  txt  → {txt_file}")

    # ── JSON ──────────────────────────────────────────────────────────────────
    json_data = dict(
        lattice=lat.name, n_sites=n, j1=j1, j2=j2,
        variant=variant, k_opt=k,
        e_exact=e_exact, e_ucj=jax_best['E'],
        abs_dE=abs(jax_best['E'] - e_exact),
        fidelity=jax_best['fid'],
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


# =============================================================================
# ENTRY
# =============================================================================
if __name__ == '__main__':
    J2_list = [0.0, 0.25, 0.50, 0.75, 1.00]

    for j2 in J2_list:

        timer = Timer()

        run(make_lattice('chain', L=16), variants=['re', 'im', 'g'], j1=1.0, j2=j2, timer=timer)
        run(make_lattice('square', L=16), variants=['re', 'im', 'g'], j1=1.0, j2=j2, timer=timer)
        run(make_lattice('triangular', L=16), variants=['re', 'im', 'g'], j1=1.0, j2=j2, timer=timer)
        run(make_lattice('honeycomb', L=18), variants=['re', 'im', 'g'], j1=1.0, j2=j2, timer=timer)
        run(make_lattice('kagome', L=12), variants=['re', 'im', 'g'], j1=1.0, j2=j2, timer=timer)

        timer.summary("Multi-lattice sweep")
