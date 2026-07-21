# @UCJ.py
"""
UCJ (Unitary Cluster Jastrow) — optimised circuit builder.

Entry point:  build_ucj(n, lattice, j1, j2, ...)
Returns:      (transpiled QuantumCircuit, gate_counts, depth)

Gate conventions:
  _qk_xy        : imaginary Givens  = RXX(-θ) · RYY(-θ)
  _qk_xy_phased : real Givens       = RZ(-π/2) · RXX(-θ) · RYY(-θ) · RZ(+π/2)
  CPhaseGate(φ) : Jastrow           = diag(1,1,1,e^{iφ})
"""

from __future__ import annotations
import warnings
import functools
import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse.linalg import eigsh, LinearOperator

import jax
import jax.numpy as jnp

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZGate, CPhaseGate
from lattices import BaseLattice, make_lattice

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
J1              = 1.0
J2              = 0.0
VARIANT         = "re"        # 're' | 'im' | 'g'
K_LAYERS        = 1
N_RESTARTS      = 1
NOISE_SCALE     = 0.05
SEED            = 23

LBFGS_MAXITER   = 800
LBFGS_MAXFUN    = 50_000
LBFGS_FTOL      = 1e-14
LBFGS_GTOL      = 1e-8

TARGET_BASIS    = ["cx", "rz", "h", "s", "sdg"]
OPT_LEVEL       = 3           # Qiskit transpiler optimisation level (0-3)

# =============================================================================
# JAX SETUP
# =============================================================================
jax.config.update("jax_enable_x64", True)
_DEVICE = jax.devices("cpu")[0]

def _to_device(x):
    return jax.device_put(x, _DEVICE)


# =============================================================================
# EXACT DIAGONALISATION
# =============================================================================
def _get_n_up(n: int) -> int:
    return n // 2


def _build_basis(n: int, n_up: int) -> np.ndarray:
    return np.array([b for b in range(1 << n) if bin(b).count("1") == n_up],
                    dtype=np.int64)


def _build_hamiltonian_op(n, n_up, nn_edges, nnn_edges, j1, j2):
    basis   = _build_basis(n, n_up)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    dim     = len(basis)
    rows, cols, vals = [], [], []

    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = 0.5 if (bits >> si) & 1 else -0.5
                zj = 0.5 if (bits >> sj) & 1 else -0.5
                rows.append(row); cols.append(row)
                vals.append(j * zi * zj)
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl  = bits ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(int(fl), -1)
                    if col >= 0:
                        rows.append(row); cols.append(col)
                        vals.append(0.5 * j)

    rows_np = np.array(rows, dtype=np.int32)
    cols_np = np.array(cols, dtype=np.int32)
    vals_np = np.array(vals, dtype=np.float64)

    def matvec(v):
        out = np.zeros(dim, dtype=v.dtype)
        np.add.at(out, rows_np, vals_np * v[cols_np])
        return out

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    return op, basis, idx_map


def get_ground_state(n, n_up, nn_edges, nnn_edges, j1, j2):
    op, basis, idx_map = _build_hamiltonian_op(
        n, n_up, nn_edges, nnn_edges, j1, j2)
    evals, evecs = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=10_000)
    e_exact      = float(evals[0])
    psi_exact    = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    return e_exact, psi_exact, basis, idx_map


# =============================================================================
# JAX HAMILTONIAN
# =============================================================================
def build_jax_hamiltonian(n, n_up, nn_edges, nnn_edges, j1, j2, basis, idx_map):
    rows, cols, vals = [], [], []
    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = 0.5 if (bits >> si) & 1 else -0.5
                zj = 0.5 if (bits >> sj) & 1 else -0.5
                rows.append(row); cols.append(row); vals.append(j * zi * zj)
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl = bits ^ (1 << si) ^ (1 << sj)
                    if fl in idx_map:
                        rows.append(row)
                        cols.append(idx_map[fl])
                        vals.append(0.5 * j)

    h_rows = _to_device(jnp.array(rows, dtype=jnp.int32))
    h_cols = _to_device(jnp.array(cols, dtype=jnp.int32))
    h_vals = _to_device(jnp.array(vals, dtype=jnp.float64))

    dim = len(basis)

    @functools.partial(jax.jit)
    def apply_H(psi):
        return (jnp.zeros(dim, dtype=psi.dtype)
                .at[h_rows].add(h_vals * psi[h_cols]))

    return apply_H


# =============================================================================
# JASTROW
# =============================================================================
def build_jastrow_fn(n, basis):
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pi    = _to_device(jnp.array([p[0] for p in pairs], dtype=jnp.int32))
    pj    = _to_device(jnp.array([p[1] for p in pairs], dtype=jnp.int32))
    bits  = _to_device(jnp.array(basis, dtype=jnp.int32))

    @jax.jit
    def jastrow_phase(theta_J):
        def acc(phase, args):
            tk, i, j = args
            bi = ((bits >> i) & 1).astype(jnp.float64)
            bj = ((bits >> j) & 1).astype(jnp.float64)
            return phase + tk * bi * bj, None
        phase, _ = jax.lax.scan(
            acc,
            jnp.zeros(bits.shape[0], dtype=jnp.float64),
            (theta_J, pi, pj),
        )
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
                    fl = bits ^ (1 << i) ^ (1 << j)
                    if fl in idx_map:
                        srcs.append(row)
                        dsts.append(idx_map[fl])
            srcs_ragged.append(np.array(srcs, dtype=np.int32))
            dsts_ragged.append(np.array(dsts, dtype=np.int32))

    counts  = np.array([len(s) for s in srcs_ragged], dtype=np.int32)
    row_ptr = np.zeros(len(counts) + 1, dtype=np.int32)
    row_ptr[1:] = np.cumsum(counts)

    srcs_cat = np.concatenate(srcs_ragged) if srcs_ragged else np.array([], dtype=np.int32)
    dsts_cat = np.concatenate(dsts_ragged) if dsts_ragged else np.array([], dtype=np.int32)

    return (_to_device(jnp.array(srcs_cat, dtype=jnp.int32)),
            _to_device(jnp.array(dsts_cat, dtype=jnp.int32)),
            row_ptr)


# =============================================================================
# ANSATZ STATE
# =============================================================================
def _givens_scan(psi, thetas, srcs, dsts, row_ptr, imag=False):
    for k in range(row_ptr.shape[0] - 1):
        s, e = int(row_ptr[k]), int(row_ptr[k + 1])
        if s == e:
            continue
        c, ss = jnp.cos(thetas[k]), jnp.sin(thetas[k])
        ps, pd = psi[srcs[s:e]], psi[dsts[s:e]]
        if imag:
            ns, nd = c * ps - 1j * ss * pd, -1j * ss * ps + c * pd
        else:
            ns, nd = c * ps - ss * pd, ss * ps + c * pd
        psi = psi.at[srcs[s:e]].set(ns).at[dsts[s:e]].set(nd)
    return psi


def _ucj_state(theta, variant, k_layers, psi0, n_pair, srcs, dsts, row_ptr, jastrow_fn):
    psi    = psi0
    stride = 3 * n_pair if variant == "g" else 2 * n_pair
    for l in range(k_layers):
        off = l * stride
        psi = apply_jastrow(psi, theta[off:off + n_pair], jastrow_fn)
        psi = _givens_scan(psi, theta[off + n_pair:off + 2 * n_pair],
                           srcs, dsts, row_ptr, imag=(variant == "im"))
        if variant == "g":
            psi = _givens_scan(psi, theta[off + 2 * n_pair:off + 3 * n_pair],
                               srcs, dsts, row_ptr, imag=True)
    return psi


def _energy(psi, apply_H):
    norm = jnp.dot(jnp.conj(psi), psi)
    return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)


# =============================================================================
# OPTIMISATION
# =============================================================================
def _make_val_grad(variant, k_layers, psi0, n_pair, srcs, dsts, row_ptr,
                   jastrow_fn, apply_H):
    def efn(theta):
        psi = _ucj_state(theta, variant, k_layers, psi0, n_pair,
                         srcs, dsts, row_ptr, jastrow_fn)
        return _energy(psi, apply_H)

    return jax.jit(jax.value_and_grad(efn))


def _optimise(val_grad_fn, x0):
    x0_gpu = _to_device(jnp.array(x0, dtype=jnp.float64))
    val_grad_fn(x0_gpu)  # warm-up JIT

    def scipy_fn(x_np):
        x_gpu = _to_device(jnp.array(x_np, dtype=jnp.float64))
        E, g  = val_grad_fn(x_gpu)
        return float(E), np.array(g, dtype=np.float64)

    result = scipy_minimize(
        scipy_fn, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER, "maxfun": LBFGS_MAXFUN,
                 "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL})
    return np.array(result.x), float(result.fun), result


def _run_optimisation(variant, k_layers, n, n_pair, psi_neel,
                      srcs, dsts, row_ptr, jastrow_fn, apply_H, e_exact):
    val_grad_fn = _make_val_grad(variant, k_layers, psi_neel, n_pair,
                                 srcs, dsts, row_ptr, jastrow_fn, apply_H)
    stride = 3 * n_pair if variant == "g" else 2 * n_pair

    best_params, best_E = None, np.inf
    rng = np.random.default_rng(SEED)

    for restart in range(N_RESTARTS):
        x0 = NOISE_SCALE * rng.standard_normal(k_layers * stride)
        opt_x, opt_E, result = _optimise(val_grad_fn, x0)
        print(f"  [restart {restart}]  E={opt_E:.8f}  "
              f"|ΔE|={abs(opt_E - e_exact):.4e}  "
              f"nit={result.nit}  nfev={result.nfev}")
        if opt_E < best_E:
            best_E, best_params = opt_E, opt_x

    return best_params, best_E


# =============================================================================
# QISKIT GATE HELPERS
# =============================================================================
def _qk_xy(qc, theta, q0, q1):
    qc.append(RXXGate(-theta), [q0, q1])
    qc.append(RYYGate(-theta), [q0, q1])


def _qk_xy_phased(qc, theta, q0, q1):
    qc.append(RZGate(-np.pi / 2), [q1])
    qc.append(RXXGate(-theta), [q0, q1])
    qc.append(RYYGate(-theta), [q0, q1])
    qc.append(RZGate(np.pi / 2), [q1])


# =============================================================================
# CIRCUIT BUILDER
# =============================================================================
def _build_circuit(n, k_layers, variant, params, pairs):
    n_pair = len(pairs)
    stride = 3 * n_pair if variant == "g" else 2 * n_pair

    qc = QuantumCircuit(n)
    for i in range(0, n, 2):
        qc.x(i)

    for l in range(k_layers):
        off  = l * stride
        tJ   = params[off           : off + n_pair]
        tK_r = params[off + n_pair  : off + 2 * n_pair]
        tK_i = params[off + 2*n_pair: off + 3 * n_pair] if variant == "g" else None

        for k, (i, j) in enumerate(pairs):
            qc.append(CPhaseGate(float(tJ[k])), [i, j])

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
# ENTRY POINT
# =============================================================================
def build_ucj(
    lattice,
    j1: float = J1,
    j2: float = J2,
    *,
    variant: str       = VARIANT,
    k_layers: int      = K_LAYERS,
    pairs: list | None = None,
    basis_gates: list | None = None,
) -> tuple[QuantumCircuit, dict[str, int], int]:
    """
    Optimise UCJ parameters then build + transpile the circuit.

    Parameters
    ----------
    lattice     : object with .nn_edges and .nnn_edges
    j1, j2      : Heisenberg couplings
    variant     : 're' | 'im' | 'g'
    k_layers    : number of UCJ layers
    pairs       : (i,j) qubit pairs; defaults to all upper-triangle pairs
    basis_gates : transpile target; defaults to TARGET_BASIS

    Returns
    -------
    tqc         : transpiled QuantumCircuit with optimised parameters
    gate_counts : dict[str, int]
    depth       : int
    """
    n = lattice.n_sites
    
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if basis_gates is None:
        basis_gates = TARGET_BASIS

    n_pair = len(pairs)
    n_up   = _get_n_up(n)

    # ── exact diagonalisation ─────────────────────────────────────────────────
    print(f"\n[build_ucj]  N={n}  variant={variant}  k={k_layers}  "
          f"J1={j1}  J2={j2}")
    print("  Running exact diagonalisation...")
    e_exact, _, basis, idx_map = get_ground_state(
        n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2)
    print(f"  E_exact = {e_exact:.10f}  (E/site = {e_exact/n:.10f})")

    # ── JAX structures ────────────────────────────────────────────────────────
    apply_H     = build_jax_hamiltonian(
        n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2, basis, idx_map)
    jastrow_fn  = build_jastrow_fn(n, basis)
    srcs, dsts, row_ptr = build_givens_pairs(n, basis, idx_map)

    # Néel initial state
    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi_neel  = _to_device(
        jnp.zeros(len(basis), dtype=jnp.complex128)
        .at[idx_map[neel_bits]].set(1.0)
    )

    # ── optimisation ─────────────────────────────────────────────────────────
    print(f"  Optimising ({N_RESTARTS} restart(s))...")
    best_params, best_E = _run_optimisation(
        variant, k_layers, n, n_pair, psi_neel,
        srcs, dsts, row_ptr, jastrow_fn, apply_H, e_exact)
    print(f"  Best E = {best_E:.10f}  |ΔE| = {abs(best_E - e_exact):.4e}")

    # ── build + transpile circuit ─────────────────────────────────────────────
    qc = _build_circuit(n, k_layers, variant, best_params, pairs)
    try:
        tqc = transpile(qc, basis_gates=basis_gates,
                        optimization_level=OPT_LEVEL)
    except Exception as exc:
        warnings.warn(f"transpile failed ({exc}); returning undecomposed circuit")
        tqc = qc

    gate_counts = dict(tqc.count_ops())
    depth       = tqc.depth()

    print(f"\n  [Circuit]  depth={depth}  total_gates={sum(gate_counts.values())}")
    for gate, count in sorted(gate_counts.items(), key=lambda x: -x[1]):
        print(f"    {gate:<20} {count}")

    return tqc, gate_counts, depth

'''
# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    lattice = make_lattice('kagome', L=n)

    tqc, counts, depth = build_ucj(lattice, j1=1.0, j2=0.0,
                                   variant="re", k_layers=1)
'''
