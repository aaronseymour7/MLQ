# @UCJ.py
"""
UCJ (Unitary Cluster Jastrow) — optimised circuit builder.

Entry point:  build_ucj(lattice, j1, j2, ...)
Returns:      (transpiled QuantumCircuit, gate_counts, depth)

Gate conventions:
  _qk_xy        : imaginary Givens  = RXX(-θ) · RYY(-θ)
  _qk_xy_phased : real Givens       = RZ(-π/2) · RXX(-θ) · RYY(-θ) · RZ(+π/2)
  CPhaseGate(φ) : Jastrow           = diag(1,1,1,e^{iφ})

Parallelism:
  Jastrow and Givens gates are applied in parallel *rounds* derived from a
  greedy edge-coloring of the pair graph.  Within each round all gates act on
  disjoint qubit pairs and can be executed simultaneously.  For a 2-D square
  lattice with nearest-neighbour pairs this reduces the two-qubit gate depth
  from O(N²) (all-to-all sequential) to O(lattice_diameter) ≈ O(√N).
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
# EDGE COLORING  (new)
# =============================================================================
def color_edges(edges: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    """
    Greedy edge coloring of a pair list.

    Returns a list of *rounds*, where each round is a list of (i, j) pairs
    that share no endpoint — i.e. they can be executed in parallel.

    The number of rounds equals the chromatic index of the pair graph, which
    for a d-regular graph is at most d+1 (Vizing's theorem).  For NN edges on
    a 2-D square lattice (d=4) this gives ≤ 5 rounds; in practice 4 suffice.

    Parameters
    ----------
    edges : list of (i, j) pairs (i < j assumed, but not required)

    Returns
    -------
    rounds : list of lists of (i, j) pairs
    """
    edge_color: dict[tuple[int, int], int] = {}
    for e in edges:
        e = (min(e), max(e))
        # colours already used by edges incident to either endpoint
        used: set[int] = set()
        for f, c in edge_color.items():
            if f[0] in e or f[1] in e:
                used.add(c)
        color = 0
        while color in used:
            color += 1
        edge_color[e] = color

    rounds: dict[int, list[tuple[int, int]]] = {}
    for e, c in edge_color.items():
        rounds.setdefault(c, []).append(e)

    return [rounds[c] for c in sorted(rounds)]


def _pairs_to_index(pairs: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    """Map each (i,j) pair to its position in the flat pairs list."""
    return {tuple(sorted((a, b))): k for k, (a, b) in enumerate(pairs)}


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
def build_jastrow_fn(n, basis, pairs):
    """
    Build a Jastrow phase function restricted to the given pair list.

    Previously this used all N(N-1)/2 pairs.  Now it is driven by `pairs`,
    which should be the same list used for the Givens rotations (e.g. NN edges)
    so that the parameter count and the circuit topology are consistent.
    """
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
def build_givens_pairs(n, basis, idx_map, pairs):
    """
    Build scatter indices for Givens rotations over the given pair list.

    `pairs` controls which (i,j) Givens rotations are included.  Pass
    lattice.nn_edges for a local ansatz; pass the all-to-all list only if
    you specifically need the full UCJ expressiveness.
    """
    srcs_ragged, dsts_ragged = [], []
    for i, j in pairs:
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
# CIRCUIT BUILDER  (parallelised)
# =============================================================================
def _build_circuit(
    n: int,
    k_layers: int,
    variant: str,
    params: np.ndarray,
    pairs: list[tuple[int, int]],
    rounds: list[list[tuple[int, int]]],
) -> QuantumCircuit:
    """
    Build the UCJ circuit with parallel gate rounds.

    The key change vs. the original:
      - `rounds` is a list of *parallel groups* of (i,j) pairs produced by
        edge-coloring.  Within each round all pairs are disjoint, so Qiskit
        can schedule them simultaneously.
      - A barrier is inserted between the Jastrow sub-layer, the real-Givens
        sub-layer, and (for variant 'g') the imaginary-Givens sub-layer so
        the transpiler cannot merge gates across sub-layers while still
        parallelising within each sub-layer.
      - The flat `pairs` list is used only for parameter indexing; circuit
        gates are emitted round-by-round.

    Parameters
    ----------
    n        : number of qubits
    k_layers : UCJ layers
    variant  : 're' | 'im' | 'g'
    params   : flat optimised parameter array
    pairs    : canonical flat pair list (same order used during optimisation)
    rounds   : edge-coloring of `pairs` — list of disjoint parallel groups
    """
    n_pair  = len(pairs)
    stride  = 3 * n_pair if variant == "g" else 2 * n_pair
    pair_idx = _pairs_to_index(pairs)   # (i,j) → position in flat param vector

    qc = QuantumCircuit(n)

    # initialise Néel state: X on every even qubit
    for i in range(0, n, 2):
        qc.x(i)

    for l in range(k_layers):
        off  = l * stride
        tJ   = params[off           : off + n_pair]
        tK_r = params[off + n_pair  : off + 2 * n_pair]
        tK_i = (params[off + 2 * n_pair : off + 3 * n_pair]
                if variant == "g" else None)

        # ── Jastrow sub-layer ─────────────────────────────────────────────
        # Each round contains disjoint pairs → parallel CPhase gates.
        for round_pairs in rounds:
            for (i, j) in round_pairs:
                k = pair_idx[(min(i, j), max(i, j))]
                qc.append(CPhaseGate(float(tJ[k])), [i, j])
            qc.barrier()   # separate rounds so transpiler respects ordering

        # ── real-Givens sub-layer ─────────────────────────────────────────
        for round_pairs in rounds:
            for (i, j) in round_pairs:
                k = pair_idx[(min(i, j), max(i, j))]
                if variant == "im":
                    _qk_xy(qc, float(tK_r[k]), j, i)
                else:
                    _qk_xy_phased(qc, float(tK_r[k]), j, i)
            qc.barrier()

        # ── imaginary-Givens sub-layer (variant 'g' only) ─────────────────
        if variant == "g":
            for round_pairs in rounds:
                for (i, j) in round_pairs:
                    k = pair_idx[(min(i, j), max(i, j))]
                    _qk_xy(qc, float(tK_i[k]), j, i)
                qc.barrier()

    return qc


# =============================================================================
# ENTRY POINT
# =============================================================================
def build_ucj(
    lattice,
    j1: float = J1,
    j2: float = J2,
    *,
    variant: str        = VARIANT,
    k_layers: int       = K_LAYERS,
    pairs: list | None  = None,
    basis_gates: list | None = None,
) -> tuple[QuantumCircuit, dict[str, int], int]:
    """
    Optimise UCJ parameters then build + transpile the parallelised circuit.

    Parameters
    ----------
    lattice     : object with .n_sites, .nn_edges, .nnn_edges
    j1, j2      : Heisenberg couplings
    variant     : 're' | 'im' | 'g'
    k_layers    : number of UCJ layers
    pairs       : (i,j) qubit pairs for Jastrow + Givens.
                  Defaults to lattice.nn_edges (nearest-neighbour only).
                  Pass all upper-triangle pairs only if you need full UCJ.
    basis_gates : transpile target; defaults to TARGET_BASIS

    Returns
    -------
    tqc         : transpiled QuantumCircuit with optimised parameters
    gate_counts : dict[str, int]
    depth       : int

    Notes
    -----
    The pairs list is edge-colored to derive parallel rounds.  For a 2-D
    square lattice with NN pairs this yields 4 rounds (horizontal even/odd +
    vertical even/odd), giving two-qubit gate depth O(k_layers * rounds)
    rather than O(k_layers * N²).
    """
    n = lattice.n_sites

    # ── default pairs: nearest-neighbour only ────────────────────────────────
    if pairs is None:
        pairs = [(min(i, j), max(i, j)) for (i, j) in lattice.nn_edges]
        pairs = list(dict.fromkeys(pairs))   # deduplicate, preserve order
    if basis_gates is None:
        basis_gates = TARGET_BASIS

    n_pair = len(pairs)
    n_up   = _get_n_up(n)

    # ── edge-color the pairs into parallel rounds ─────────────────────────────
    rounds = color_edges(pairs)
    print(f"\n[build_ucj]  N={n}  variant={variant}  k={k_layers}  "
          f"J1={j1}  J2={j2}")
    print(f"  Pairs: {n_pair} ({len(rounds)} parallel rounds after edge-coloring)")

    # ── exact diagonalisation ─────────────────────────────────────────────────
    print("  Running exact diagonalisation...")
    e_exact, _, basis, idx_map = get_ground_state(
        n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2)
    print(f"  E_exact = {e_exact:.10f}  (E/site = {e_exact/n:.10f})")

    # ── JAX structures ────────────────────────────────────────────────────────
    apply_H    = build_jax_hamiltonian(
        n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2, basis, idx_map)
    jastrow_fn = build_jastrow_fn(n, basis, pairs)   # restricted to `pairs`
    srcs, dsts, row_ptr = build_givens_pairs(n, basis, idx_map, pairs)

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
    qc = _build_circuit(n, k_layers, variant, best_params, pairs, rounds)
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


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":
    lattice = make_lattice("square", L=8)   # 4×4 = 16 sites

    # ── nearest-neighbour Re-UCJ (default, shallow) ───────────────────────────
    tqc, counts, depth = build_ucj(lattice, j1=1.0, j2=0.0,
                                   variant="re", k_layers=1)

    # ── optional: pass explicit pairs to use NNN or all-to-all ───────────────
    # nn_nnn = [(min(i,j), max(i,j))
    #           for (i,j) in lattice.nn_edges + lattice.nnn_edges]
    # nn_nnn = list(dict.fromkeys(nn_nnn))
    # tqc2, counts2, depth2 = build_ucj(lattice, pairs=nn_nnn,
    #                                    variant="re", k_layers=1)
