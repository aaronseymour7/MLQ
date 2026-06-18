"""
ucj_connectivity_sweep.py
==========================
Standalone UCJ-only study: sweep ansatz CONNECTIVITY at a single fixed
lattice point, with no DMRG / TeNPy / mps-to-circuit dependency at all.

Connectivity tiers (Givens + Jastrow pair list, independent of which
edges actually carry Heisenberg coupling):
    nn        : nearest-neighbor lattice edges only   (lattice.nn_edges)
    nn+nnn    : nearest- + next-nearest-neighbor edges (nn_edges ∪ nnn_edges)
    all-pairs : every (i, j), i < j                    (the UCJ.py default)


Dependencies
------------
    pip install qiskit scipy numpy jax
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize as scipy_minimize
from scipy.sparse.linalg import eigsh, LinearOperator

import jax
import jax.numpy as jnp

from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import RXXGate, RYYGate, RZGate, CPhaseGate
from qiskit.quantum_info import Statevector

from lattices import BaseLattice, make_lattice   # your geometry module

# =============================================================================
# GLOBAL CONFIG
# =============================================================================
J1             = 1.0
J2             = 0.0
UCJ_VARIANT    = "re"   # 're' | 'im' | 'g'
UCJ_K_LAYERS   = 1
UCJ_N_RESTARTS = 1
UCJ_NOISE      = 0.05
UCJ_SEED       = 23
LBFGS_MAXITER  = 1000
LBFGS_MAXFUN   = 50_000
LBFGS_FTOL     = 1e-14
LBFGS_GTOL     = 1e-8

BASIS_GATES        = ["cx", "rz", "h", "s", "sdg"]
NON_CLIFFORD_GATES = {"rz"}
TWO_QUBIT_GATES    = {"cx"}
RZ_SYNTHESIS_EPS   = 1e-3   # Ross–Selinger synthesis precision

CONNECTIVITY_TIERS = ["nn", "nn+nnn", "all-pairs"]

jax.config.update("jax_enable_x64", True)
_CPU = jax.devices("cpu")[0]
def _jput(x): return jax.device_put(x, _CPU)


# =============================================================================
# RESOURCE ACCOUNTING
# =============================================================================
@dataclass
class ResourceReport:
    label: str
    num_qubits: int
    depth: int
    non_clifford_depth: int
    gate_counts: dict
    cx_count: int
    non_clifford_count: int
    t_count_estimate: int
    rz_synthesis_eps: float

    def prefixed(self) -> dict:
        base = dict(
            num_qubits         = self.num_qubits,
            depth              = self.depth,
            non_clifford_depth = self.non_clifford_depth,
            cx_count           = self.cx_count,
            non_clifford_count = self.non_clifford_count,
            t_count_estimate   = self.t_count_estimate,
            rz_synthesis_eps   = self.rz_synthesis_eps,
        )
        base.update({f"n_{g}": c for g, c in self.gate_counts.items()})
        return {f"{self.label}_{k}": v for k, v in base.items()}

    def print_summary(self) -> None:
        print(f"\n[resources:{self.label}]  qubits={self.num_qubits}  "
              f"depth={self.depth}  non_clifford_depth={self.non_clifford_depth}")
        for gate, count in self.gate_counts.items():
            tag = "  ← non-Clifford" if gate in NON_CLIFFORD_GATES else ""
            print(f"  {gate:<6} {count}{tag}")
        print(f"  CX count   : {self.cx_count}")
        print(f"  RZ count   : {self.non_clifford_count}")
        print(f"  T-count est (ε={self.rz_synthesis_eps:g}) : {self.t_count_estimate}")


def _t_per_rz(eps: float = RZ_SYNTHESIS_EPS) -> float:
    """Ross–Selinger upper bound: ~3·log₂(1/ε) T gates per arbitrary RZ."""
    if eps <= 0:
        raise ValueError("eps must be positive")
    return 3.0 * math.log2(1.0 / eps)


def resource_report(
    qc: QuantumCircuit,
    label: str,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    opt_level: int = 3,
) -> ResourceReport:
    qc_t   = transpile(qc, basis_gates=BASIS_GATES, optimization_level=opt_level)
    counts = dict(qc_t.count_ops())

    cx_nc  = sum(counts.get(g, 0) for g in TWO_QUBIT_GATES)
    rz_nc  = sum(counts.get(g, 0) for g in NON_CLIFFORD_GATES)
    depth  = qc_t.depth()
    nc_dep = qc_t.depth(
        filter_function=lambda i: i.operation.name in NON_CLIFFORD_GATES)
    t_est  = int(round(rz_nc * _t_per_rz(rz_eps)))

    rep = ResourceReport(
        label=label,
        num_qubits=qc_t.num_qubits,
        depth=depth,
        non_clifford_depth=nc_dep,
        gate_counts=counts,
        cx_count=cx_nc,
        non_clifford_count=rz_nc,
        t_count_estimate=t_est,
        rz_synthesis_eps=rz_eps,
    )
    rep.print_summary()
    return rep


# =============================================================================
# STATE FIDELITY  (post-circuit, against exact ground state)
# =============================================================================
def state_fidelity_vs_exact(psi_sec: np.ndarray, psi_exact: np.ndarray) -> float:
    """
    |⟨ψ_exact|ψ_circuit⟩|² for two normalised vectors in the same Sz=0
    basis ordering. Strictly stronger than |ΔE| alone — confirms the
    circuit *is* the ground state rather than merely iso-energetic.
    """
    overlap = np.vdot(psi_exact, psi_sec)
    return min(float(np.abs(overlap) ** 2), 1.0)


# =============================================================================
# HAMILTONIAN  (sparse, Sz=0 sector) + EXACT DIAGONALISATION
# =============================================================================
def _build_basis(n: int) -> np.ndarray:
    n_up = n // 2
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up],
                    dtype=np.int64)


def exact_ground_state(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Lanczos ground state in the Sz=0 sector. Returns (e_exact, psi_exact, basis, idx_map)."""
    n       = lattice.n_sites
    basis   = _build_basis(n)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    dim     = len(basis)

    rows, cols, vals = [], [], []
    edge_sets = [(lattice.nn_edges, j1)]
    if j2 and lattice.nnn_edges:
        edge_sets.append((lattice.nnn_edges, j2))

    for edges, j in edge_sets:
        for si, sj in edges:
            for row, bits_ in enumerate(basis):
                zi = 0.5 if (bits_ >> si) & 1 else -0.5
                zj = 0.5 if (bits_ >> sj) & 1 else -0.5
                rows.append(row); cols.append(row); vals.append(j * zi * zj)
                if ((bits_ >> si) & 1) != ((bits_ >> sj) & 1):
                    fl  = int(bits_) ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(fl, -1)
                    if col >= 0:
                        rows.append(row); cols.append(col); vals.append(0.5 * j)

    r_np = np.array(rows, dtype=np.int32)
    c_np = np.array(cols, dtype=np.int32)
    v_np = np.array(vals, dtype=np.float64)

    def matvec(v):
        out = np.zeros(dim, dtype=v.dtype)
        np.add.at(out, r_np, v_np * v[c_np])
        return out

    op           = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    evals, evecs = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=10_000)
    e_exact      = float(evals[0])
    psi_exact    = evecs[:, 0] / np.linalg.norm(evecs[:, 0])

    print(f"[ED]  E_exact={e_exact:.10f}  E/site={e_exact/n:.10f}")
    return e_exact, psi_exact, basis, idx_map


# =============================================================================
# CONNECTIVITY TIERS  (this is the whole point of the script)
# =============================================================================
def _nn_pairs(lattice: BaseLattice) -> list[tuple[int, int]]:
    """Nearest-neighbor lattice edges only."""
    return list(dict.fromkeys(
        (min(i, j), max(i, j)) for i, j in lattice.nn_edges))


def _nn_nnn_pairs(lattice: BaseLattice) -> list[tuple[int, int]]:
    """Nearest- + next-nearest-neighbor lattice edges."""
    edges = list(lattice.nn_edges) + list(lattice.nnn_edges or [])
    return list(dict.fromkeys((min(i, j), max(i, j)) for i, j in edges))


def _all_pairs(lattice: BaseLattice) -> list[tuple[int, int]]:
    """Every (i, j), i < j — the UCJ.py default, ignores lattice geometry."""
    n = lattice.n_sites
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


CONNECTIVITY_BUILDERS = {
    "nn":        _nn_pairs,
    "nn+nnn":    _nn_nnn_pairs,
    "all-pairs": _all_pairs,
}


def get_pairs(lattice: BaseLattice, tier: str) -> list[tuple[int, int]]:
    if tier not in CONNECTIVITY_BUILDERS:
        raise ValueError(f"unknown connectivity tier {tier!r}; "
                         f"choose from {list(CONNECTIVITY_BUILDERS)}")
    return CONNECTIVITY_BUILDERS[tier](lattice)


# =============================================================================
# UCJ  –  JAX Hamiltonian (for gradient-based optimisation)
# =============================================================================
def _jax_hamiltonian(lattice, j1, j2, basis, idx_map):
    rows, cols, vals = [], [], []
    edge_sets = [(lattice.nn_edges, j1)]
    if j2 and lattice.nnn_edges:
        edge_sets.append((lattice.nnn_edges, j2))
    for edges, j in edge_sets:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = 0.5 if (bits >> si) & 1 else -0.5
                zj = 0.5 if (bits >> sj) & 1 else -0.5
                rows.append(row); cols.append(row); vals.append(j * zi * zj)
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl = int(bits) ^ (1 << si) ^ (1 << sj)
                    if fl in idx_map:
                        rows.append(row); cols.append(idx_map[fl])
                        vals.append(0.5 * j)

    h_r = _jput(jnp.array(rows, jnp.int32))
    h_c = _jput(jnp.array(cols, jnp.int32))
    h_v = _jput(jnp.array(vals, jnp.float64))
    dim = len(basis)

    @jax.jit
    def apply_H(psi):
        return jnp.zeros(dim, dtype=psi.dtype).at[h_r].add(h_v * psi[h_c])

    return apply_H


# =============================================================================
# UCJ  –  Jastrow + Givens ansatz in JAX
# =============================================================================
def _jastrow_fn(basis, pairs):
    pi   = _jput(jnp.array([p[0] for p in pairs], jnp.int32))
    pj   = _jput(jnp.array([p[1] for p in pairs], jnp.int32))
    bits = _jput(jnp.array(basis, jnp.int32))

    @jax.jit
    def phase(theta_J):
        def acc(ph, args):
            tk, i, j = args
            bi = ((bits >> i) & 1).astype(jnp.float64)
            bj = ((bits >> j) & 1).astype(jnp.float64)
            return ph + tk * bi * bj, None
        ph, _ = jax.lax.scan(
            acc, jnp.zeros(bits.shape[0], jnp.float64), (theta_J, pi, pj))
        return ph

    return phase


def _givens_scatter(basis, idx_map, pairs):
    srcs_r, dsts_r = [], []
    for i, j in pairs:
        s, d = [], []
        for row, bits in enumerate(basis):
            if ((bits >> i) & 1) and not ((bits >> j) & 1):
                fl = int(bits) ^ (1 << i) ^ (1 << j)
                if fl in idx_map:
                    s.append(row); d.append(idx_map[fl])
        srcs_r.append(np.array(s, np.int32))
        dsts_r.append(np.array(d, np.int32))

    counts  = np.array([len(s) for s in srcs_r], np.int32)
    row_ptr = np.zeros(len(counts) + 1, np.int32)
    row_ptr[1:] = np.cumsum(counts)

    srcs = np.concatenate(srcs_r) if srcs_r else np.array([], np.int32)
    dsts = np.concatenate(dsts_r) if dsts_r else np.array([], np.int32)
    return _jput(jnp.array(srcs, jnp.int32)), \
           _jput(jnp.array(dsts, jnp.int32)), row_ptr


def _givens_scan(psi, thetas, srcs, dsts, row_ptr, imag=False):
    for k in range(row_ptr.shape[0] - 1):
        s, e = int(row_ptr[k]), int(row_ptr[k + 1])
        if s == e:
            continue
        c, ss = jnp.cos(thetas[k]), jnp.sin(thetas[k])
        ps, pd = psi[srcs[s:e]], psi[dsts[s:e]]
        if imag:
            ns, nd = c*ps - 1j*ss*pd, -1j*ss*ps + c*pd
        else:
            ns, nd = c*ps - ss*pd, ss*ps + c*pd
        psi = psi.at[srcs[s:e]].set(ns).at[dsts[s:e]].set(nd)
    return psi


def _ucj_state(theta, variant, k_layers, psi0,
               n_pair, srcs, dsts, row_ptr, jastrow_phase):
    psi    = psi0
    stride = 3*n_pair if variant == "g" else 2*n_pair
    for l in range(k_layers):
        off = l * stride
        psi = psi * jnp.exp(1j * jastrow_phase(theta[off:off+n_pair]))
        psi = _givens_scan(psi, theta[off+n_pair:off+2*n_pair],
                           srcs, dsts, row_ptr, imag=(variant == "im"))
        if variant == "g":
            psi = _givens_scan(psi, theta[off+2*n_pair:off+3*n_pair],
                               srcs, dsts, row_ptr, imag=True)
    return psi


# =============================================================================
# UCJ  –  L-BFGS-B optimisation
# =============================================================================
def _optimise_ucj(
    lattice, j1, j2, basis, idx_map, pairs,
    variant=UCJ_VARIANT, k_layers=UCJ_K_LAYERS, e_exact=None,
) -> tuple[np.ndarray, float]:
    n      = lattice.n_sites
    n_pair = len(pairs)
    stride = 3*n_pair if variant == "g" else 2*n_pair

    apply_H      = _jax_hamiltonian(lattice, j1, j2, basis, idx_map)
    jastrow_ph   = _jastrow_fn(basis, pairs)
    srcs, dsts, row_ptr = _givens_scatter(basis, idx_map, pairs)

    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi0 = _jput(jnp.zeros(len(basis), jnp.complex128)
                      .at[idx_map[neel_bits]].set(1.0))

    def energy_fn(theta):
        psi  = _ucj_state(theta, variant, k_layers, psi0,
                          n_pair, srcs, dsts, row_ptr, jastrow_ph)
        norm = jnp.dot(jnp.conj(psi), psi)
        return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)

    val_grad = jax.jit(jax.value_and_grad(energy_fn))

    rng = np.random.default_rng(UCJ_SEED)
    best_params, best_E = None, np.inf

    for restart in range(UCJ_N_RESTARTS):
        x0 = UCJ_NOISE * rng.standard_normal(k_layers * stride)
        val_grad(_jput(jnp.array(x0, jnp.float64)))   # JIT warm-up

        def scipy_fn(x_np):
            xj    = _jput(jnp.array(x_np, jnp.float64))
            E, g  = val_grad(xj)
            return float(E), np.array(g, np.float64)

        res = scipy_minimize(
            scipy_fn, x0, jac=True, method="L-BFGS-B",
            options={"maxiter": LBFGS_MAXITER, "maxfun": LBFGS_MAXFUN,
                     "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL})
        opt_x, opt_E = np.array(res.x), float(res.fun)
        ref = f"  |ΔE|={abs(opt_E-e_exact):.4e}" if e_exact is not None else ""
        print(f"  [restart {restart}]  E={opt_E:.10f}{ref}  nit={res.nit}")
        if opt_E < best_E:
            best_E, best_params = opt_E, opt_x

    return best_params, best_E


# =============================================================================
# UCJ  –  Qiskit circuit builder  (sequential, no rounds/barriers)
# =============================================================================
def _qk_xy(qc, theta, q0, q1):
    """Imaginary Givens: RXX(-θ) · RYY(-θ)"""
    qc.append(RXXGate(-theta), [q0, q1])
    qc.append(RYYGate(-theta), [q0, q1])


def _qk_xy_phased(qc, theta, q0, q1):
    """Real Givens: RZ(-π/2) · RXX(-θ) · RYY(-θ) · RZ(+π/2)"""
    qc.append(RZGate(-math.pi / 2), [q1])
    qc.append(RXXGate(-theta), [q0, q1])
    qc.append(RYYGate(-theta), [q0, q1])
    qc.append(RZGate(math.pi / 2), [q1])


def _build_ucj_circuit(n, k_layers, variant, params, pairs):
    n_pair = len(pairs)
    stride = 3*n_pair if variant == "g" else 2*n_pair

    qc = QuantumCircuit(n)
    for i in range(0, n, 2):    # Néel initial state
        qc.x(i)

    for l in range(k_layers):
        off  = l * stride
        tJ   = params[off          : off+n_pair]
        tK_r = params[off+n_pair   : off+2*n_pair]
        tK_i = params[off+2*n_pair : off+3*n_pair] if variant == "g" else None

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
# RUN ONE CONNECTIVITY TIER
# =============================================================================
def run_ucj_tier(
    lattice: BaseLattice,
    tier: str,
    j1: float = J1,
    j2: float = J2,
    e_exact: float | None = None,
    psi_exact: np.ndarray | None = None,
    basis: np.ndarray | None = None,
    idx_map: dict | None = None,
    variant: str = UCJ_VARIANT,
    k_layers: int = UCJ_K_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
) -> dict:
    """
    Optimise + build + transpile UCJ for a single connectivity tier,
    and report resources + |ΔE| + fidelity-vs-exact.

    Returns a flat dict prefixed with the tier name (e.g. 'nn_cx_count').
    """
    n = lattice.n_sites
    pairs = get_pairs(lattice, tier)
    print(f"\n[UCJ:{tier}]  N={n}  variant={variant}  k={k_layers}  pairs={len(pairs)}")

    best_params, best_E = _optimise_ucj(
        lattice, j1, j2, basis, idx_map, pairs,
        variant=variant, k_layers=k_layers, e_exact=e_exact)

    de = abs(best_E - e_exact) if e_exact is not None else float('nan')
    print(f"[UCJ:{tier}]  best_E={best_E:.10f}  |ΔE|={de:.4e}")

    qc  = _build_ucj_circuit(n, k_layers, variant, best_params, pairs)
    rep = resource_report(qc, label=tier, rz_eps=rz_eps, opt_level=3)

    fid_info = {}
    if psi_exact is not None:
        sv      = Statevector.from_instruction(qc).data
        psi_sec = sv[basis]
        norm2   = float(np.vdot(psi_sec, psi_sec).real)
        if norm2 < 1e-12:
            raise RuntimeError(
                f"[{tier}] circuit statevector has ~0 amplitude in Sz=0 sector.")
        psi_sec /= math.sqrt(norm2)
        fid = state_fidelity_vs_exact(psi_sec, psi_exact)
        fid_info = dict(
            sector_norm2        = norm2,
            leakage              = 1.0 - norm2,
            fidelity_vs_exact    = fid,
            infidelity_vs_exact  = 1.0 - fid,
        )
        print(f"[verify:{tier}]  fidelity={fid:.10f}  leakage={fid_info['leakage']:.2e}")

    info = dict(
        connectivity = tier,
        n_pairs      = len(pairs),
        energy       = best_E,
        abs_error    = de,
        **fid_info,
    )
    base = dict(
        num_qubits         = rep.num_qubits,
        depth              = rep.depth,
        non_clifford_depth = rep.non_clifford_depth,
        cx_count           = rep.cx_count,
        non_clifford_count = rep.non_clifford_count,
        t_count_estimate   = rep.t_count_estimate,
    )
    info.update(base)
    return {f"{tier}_{k}": v for k, v in info.items()}


# =============================================================================
# CONNECTIVITY SWEEP  (the main entry point)
# =============================================================================
def sweep_connectivity(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
    variant: str = UCJ_VARIANT,
    k_layers: int = UCJ_K_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    tiers: list[str] | None = None,
) -> dict:
    """
    Run UCJ once per connectivity tier (NN / NN+NNN / all-pairs) at a
    single fixed lattice point, and print a side-by-side resource/
    accuracy comparison table. No DMRG, no CSV — just the table.

    Parameters
    ----------
    lattice  : geometry object (.n_sites, .nn_edges, .nnn_edges, .name)
    j1, j2   : Heisenberg couplings (fixed across all tiers)
    variant  : 're' | 'im' | 'g'
    k_layers : UCJ depth layers (fixed across all tiers)
    rz_eps   : synthesis precision for T-count estimate
    tiers    : subset of CONNECTIVITY_TIERS to run; defaults to all three

    Returns
    -------
    info : flat dict with {tier}_* prefixed keys for every tier run
    """
    tiers = tiers or CONNECTIVITY_TIERS
    n = lattice.n_sites

    print(f"\n{'='*60}")
    print(f"[sweep_connectivity]  lattice={lattice.name}  N={n}  "
          f"J1={j1}  J2={j2}  k_layers={k_layers}  variant={variant}")
    print(f"{'='*60}")

    e_exact, psi_exact, basis, idx_map = exact_ground_state(lattice, j1, j2)
    print(f"[H]  Sz=0 sector dim={len(basis)}")

    info: dict = dict(
        lattice    = lattice.name,
        n_sites    = n,
        j1         = j1,
        j2         = j2,
        j2_over_j1 = (j2 / j1) if j1 else float('nan'),
        k_layers   = k_layers,
        variant    = variant,
        ed_energy  = e_exact,
    )

    tier_results = {}
    for tier in tiers:
        tier_info = run_ucj_tier(
            lattice, tier, j1=j1, j2=j2,
            e_exact=e_exact, psi_exact=psi_exact,
            basis=basis, idx_map=idx_map,
            variant=variant, k_layers=k_layers, rz_eps=rz_eps)
        tier_results[tier] = tier_info
        info.update(tier_info)

    _print_connectivity_table(lattice, info, tiers)
    return info


def _print_connectivity_table(lattice: BaseLattice, info: dict, tiers: list[str]) -> None:
    n = info['n_sites']
    width = 14
    print(f"\n{'─'*(28 + width*len(tiers) + 2*len(tiers))}")
    print(f"  UCJ CONNECTIVITY SWEEP  (lattice={lattice.name}  N={n}  "
          f"J2/J1={info['j2_over_j1']:.3f}  k_layers={info['k_layers']})")
    print(f"{'─'*(28 + width*len(tiers) + 2*len(tiers))}")
    header = f"  {'Metric':<28}" + "".join(f"{t:>{width}}  " for t in tiers)
    print(header)
    print(f"  {'─'*28}" + "".join(f"{'─'*width}  " for _ in tiers))

    metrics = [
        ("n_pairs",     "n_pairs"),
        ("CX count",    "cx_count"),
        ("RZ count",    "non_clifford_count"),
        ("T-count est", "t_count_estimate"),
        ("depth",       "depth"),
        ("RZ depth",    "non_clifford_depth"),
        ("|ΔE|",        "abs_error"),
        ("fidelity",    "fidelity_vs_exact"),
    ]
    for label, key in metrics:
        row = f"  {label:<28}"
        for t in tiers:
            v = info.get(f"{t}_{key}", float('nan'))
            v_s = f"{v:.3e}" if isinstance(v, float) else str(v)
            row += f"{v_s:>{width}}  "
        print(row)
    print(f"{'─'*(28 + width*len(tiers) + 2*len(tiers))}")


# =============================================================================
# DEMO
# =============================================================================
if __name__ == "__main__":

    def chain(L):
        return make_lattice("square", L=L)

    # single fixed lattice point: connectivity is the only thing varied
    sweep_connectivity(
        chain(8), j1=J1, j2=0.0,
        variant="re", k_layers=1,
    )
