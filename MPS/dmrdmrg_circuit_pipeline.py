
"""
Heisenberg MPS Pipeline + Timing Harness
=========================================
Merged single-file version. Harness implementations supersede pipeline
implementations wherever the two overlapped (MPO construction now uses
incremental compression; DMRG timing is per-sweep on a single DMRG2 object).

No CLI entry point. Import or run interactively.

Dependencies
------------
    pip install quimb qiskit scipy numpy pandas qiskit-quimb
    pip install git+https://github.com/qiskit-community/mps-to-circuit.git
"""

from __future__ import annotations

import math
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator

from qiskit import transpile
from qiskit.quantum_info import Statevector
from qiskit.exceptions import QiskitError
from qiskit_quimb import quimb_circuit

try:
    import quimb as qu
    import quimb.tensor as qtn
    print(f"[quimb]   {qu.__version__}")
except ImportError:
    raise ImportError("pip install quimb")

try:
    from mps_to_circuit import mps_to_circuit
except ImportError:
    raise ImportError(
        "pip install git+https://github.com/qiskit-community/mps-to-circuit.git"
    )

try:
    from qiskit import QuantumCircuit
    import qiskit
    print(f"[Qiskit]  {qiskit.__version__}")
except ImportError:
    raise ImportError("pip install qiskit")


# =============================================================================
# CONFIG
# =============================================================================
J1              = 1.0
J2              = 0.0
DMRG_BOND_DIMS  = [10, 20, 40, 80, 100]
DMRG_CUTOFF     = 1e-8
DMRG_TOL        = 1e-8
DMRG_MAX_SWEEPS = 40

CIRCUIT_MPS_MAX_BOND = 200

BASIS_GATES        = ["cx", "rz", "h", "s", "sdg"]
NON_CLIFFORD_GATES = {"rz"}
TWO_QUBIT_GATES    = {"cx"}

RZ_SYNTHESIS_EPS = 1e-3

DEFAULT_N_TRIALS = 100

EXACT_DIAG_MAX_N = 4

# Harness-specific defaults (edit before running run_timing_harness())
CHAIN_SIZES            = [20, 50, 80]
COMPRESS_EVERY_N_EDGES = 4
N_LAYERS               = 3
DMRG_TOL_CANDIDATES    = [1e-6, 1e-7, 1e-8, 1e-9, DMRG_TOL]


# =============================================================================
# CHAIN TOPOLOGY
# =============================================================================
@dataclass
class Chain:
    """Minimal 1D open chain descriptor."""
    n_sites: int

    @property
    def name(self) -> str:
        return f"chain{self.n_sites}"

    @property
    def nn_edges(self) -> list[tuple[int, int]]:
        return [(i, i + 1) for i in range(self.n_sites - 1)]

    @property
    def nnn_edges(self) -> list[tuple[int, int]]:
        return [(i, i + 2) for i in range(self.n_sites - 2)]


def make_chain(L: int) -> Chain:
    return Chain(n_sites=L)


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

    def as_row(self) -> dict:
        row = dict(
            label=self.label,
            num_qubits=self.num_qubits,
            depth=self.depth,
            non_clifford_depth=self.non_clifford_depth,
            cx_count=self.cx_count,
            non_clifford_count=self.non_clifford_count,
            t_count_estimate=self.t_count_estimate,
            rz_synthesis_eps=self.rz_synthesis_eps,
        )
        row.update({f"n_{g}": c for g, c in self.gate_counts.items()})
        return row

    def print_summary(self) -> None:
        print(f"\n[resources:{self.label}]  qubits={self.num_qubits}  "
              f"depth={self.depth}  non_clifford_depth={self.non_clifford_depth}")
        for gate, count in self.gate_counts.items():
            tag = "  (non-Clifford)" if gate in NON_CLIFFORD_GATES else ""
            print(f"  {gate:<5} {count}{tag}")
        print(f"  -> CX (2-qubit Clifford) count : {self.cx_count}")
        print(f"  -> RZ (non-Clifford)     count : {self.non_clifford_count}")
        print(f"  -> estimated T-count @ eps={self.rz_synthesis_eps:g} : "
              f"{self.t_count_estimate}")


def estimate_t_count_per_rz(eps: float = RZ_SYNTHESIS_EPS) -> float:
    """
    Estimated T gates per arbitrary-angle RZ rotation via the Ross-Selinger
    bound: T ~ 3*log2(1/eps). Order-of-magnitude only, not an exact synthesis.
    """
    if eps <= 0:
        raise ValueError("eps must be positive")
    return 3.0 * math.log2(1.0 / eps)


def circuit_resource_report(
    qc: QuantumCircuit,
    label: str,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    verbose: bool = True,
) -> ResourceReport:
    """Transpile qc to BASIS_GATES and extract EFT-relevant resource counts."""
    qc_t   = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    counts = dict(qc_t.count_ops())

    cx_count           = sum(counts.get(g, 0) for g in TWO_QUBIT_GATES)
    non_clifford_count = sum(counts.get(g, 0) for g in NON_CLIFFORD_GATES)

    depth = qc_t.depth()
    non_clifford_depth = qc_t.depth(
        filter_function=lambda instr: instr.operation.name in NON_CLIFFORD_GATES
    )

    t_per_rz       = estimate_t_count_per_rz(rz_eps)
    t_count_estimate = int(round(non_clifford_count * t_per_rz))

    report = ResourceReport(
        label=label,
        num_qubits=qc_t.num_qubits,
        depth=depth,
        non_clifford_depth=non_clifford_depth,
        gate_counts=counts,
        cx_count=cx_count,
        non_clifford_count=non_clifford_count,
        t_count_estimate=t_count_estimate,
        rz_synthesis_eps=rz_eps,
    )
    if verbose:
        report.print_summary()
    return report


# =============================================================================
# HAMILTONIAN  (sparse, fixed-Sz sector) -- ED reference
# =============================================================================
def build_basis(n: int) -> np.ndarray:
    """
    Sz=0 sector basis. bit=0 <-> up (Sz=+1/2), bit=1 <-> down (Sz=-1/2),
    matching quimb's qu.up()=[1,0]=|0> and Qiskit qubit ordering (site i
    <-> qubit i) with no remapping needed.
    """
    n_down = n - n // 2
    return np.array(
        [b for b in range(1 << n) if bin(b).count('1') == n_down],
        dtype=np.int64,
    )


def build_hamiltonian(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
) -> tuple[csr_matrix, np.ndarray]:
    """Return (H_sparse, basis) in the n//2-up sector."""
    n       = chain.n_sites
    basis   = build_basis(n)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    H       = lil_matrix((len(basis), len(basis)), dtype=np.float64)

    edge_sets = [(chain.nn_edges, j1)]
    if j2 and chain.nnn_edges:
        edge_sets.append((chain.nnn_edges, j2))

    for edges, j in edge_sets:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = -0.5 if (bits >> si) & 1 else 0.5
                zj = -0.5 if (bits >> sj) & 1 else 0.5
                H[row, row] += j * zi * zj
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl  = bits ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(int(fl), -1)
                    if col >= 0:
                        H[row, col] += 0.5 * j

    return csr_matrix(H), basis


def exact_ground_state(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """Lanczos ground state in the Sz=0 sector via scipy eigsh."""
    n       = chain.n_sites
    basis   = build_basis(n)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    dim     = len(basis)

    rows, cols, vals = [], [], []
    edge_sets = [(chain.nn_edges, j1)]
    if j2 and chain.nnn_edges:
        edge_sets.append((chain.nnn_edges, j2))

    for edges, j in edge_sets:
        for si, sj in edges:
            for row, bits_ in enumerate(basis):
                zi = -0.5 if (bits_ >> si) & 1 else 0.5
                zj = -0.5 if (bits_ >> sj) & 1 else 0.5
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
# QUIMB MPO  -- explicit edge-list Heisenberg Hamiltonian
# =============================================================================
def _two_site_mpo(n: int, i: int, j: int, op_i, op_j, coeff: float = 1.0):
    """One operator-string term coeff*op_i(i) ⊗ op_j(j), identity elsewhere."""
    I2 = qu.eye(2)
    ops    = [I2] * n
    ops[i] = coeff * op_i
    ops[j] = op_j
    return qtn.MPO_product_operator(ops)


def _heisenberg_term_mpo(n: int, i: int, j: int, coupling: float):
    """S_i · S_j as a sum of three bond-dim-1 MPOs."""
    Sz, Sp, Sm = qu.spin_operator('z'), qu.spin_operator('+'), qu.spin_operator('-')
    m  = _two_site_mpo(n, i, j, Sz, Sz, coupling)
    m += _two_site_mpo(n, i, j, Sp, Sm, coupling / 2.0)
    m += _two_site_mpo(n, i, j, Sm, Sp, coupling / 2.0)
    return m


def build_heisenberg_mpo(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
):
    """
    Build the full J1-J2 Heisenberg MPO using incremental compression
    (harness version). Compressing every `compress_every_n_edges` additions
    keeps the running bond dimension small throughout, avoiding the bond-dim
    balloon that the old "accumulate everything, compress once" approach
    suffered at large N.
    """
    n = chain.n_sites
    edge_sets = [(chain.nn_edges, j1)]
    if j2 and chain.nnn_edges:
        edge_sets.append((chain.nnn_edges, j2))

    all_edges = [
        (i, jx, j)
        for edges, j in edge_sets
        for (i, jx) in edges
    ]

    H_mpo = None
    for edge_idx, (i, jx, j) in enumerate(all_edges, start=1):
        term  = _heisenberg_term_mpo(n, i, jx, j)
        H_mpo = term if H_mpo is None else H_mpo + term

        if edge_idx % compress_every_n_edges == 0 or edge_idx == len(all_edges):
            H_mpo.compress(cutoff=1e-12)

    return H_mpo


# =============================================================================
# DMRG  (per-sweep timing version, faithful to a single DMRG2.solve() call)
# =============================================================================
def run_dmrg(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
) -> tuple[float, object]:
    """
    Run DMRG on the chain topology defined by *chain* using quimb's DMRG2.

    Uses a Neel initial state whose Sz is conserved by the Heisenberg
    Hamiltonian throughout the optimization. MPO is built with incremental
    compression (harness version).

    Returns
    -------
    E   : DMRG ground-state energy (real float)
    psi : converged quimb MatrixProductState
    """
    n     = chain.n_sites
    H_mpo = build_heisenberg_mpo(chain, j1, j2, compress_every_n_edges)
    psi0  = qtn.MPS_neel_state(n)

    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dims, cutoffs=DMRG_CUTOFF, p0=psi0)
    dmrg.solve(tol=DMRG_TOL, max_sweeps=DMRG_MAX_SWEEPS, verbosity=0)

    E   = float(np.real(dmrg.energy))
    psi = dmrg.state
    return E, psi


def mps_diagnostics(psi) -> dict:
    """Bond-dimension and entanglement diagnostics for the converged MPS."""
    n         = psi.L
    chi       = [psi.bond_size(i, i + 1) for i in range(n - 1)]
    half      = max(0, n // 2 - 1)
    entropies = [psi.entropy(i + 1) for i in range(n - 1)] if n > 1 else []
    return dict(
        max_chi=int(max(chi)) if chi else 0,
        mean_chi=float(np.mean(chi)) if chi else 0.0,
        half_chain_entropy=float(entropies[half]) if entropies else float('nan'),
        max_entropy=float(np.max(entropies)) if entropies else float('nan'),
    )


# =============================================================================
# MPS -> CIRCUIT
# =============================================================================
def _mps_to_lpr_arrays(psi) -> list[np.ndarray]:
    """
    Pull site tensors as a list of (left-bond, physical, right-bond) numpy
    arrays, which is the shape mps_to_circuit expects. Boundary tensors get
    a size-1 dummy bond on the missing side.
    """
    n      = psi.L
    arrays = []
    for i in range(n):
        t    = psi[i]
        phys = next(ix for ix in t.inds if ix.startswith('k'))
        if i == 0:
            right = psi.bond(0, 1)
            arr   = t.transpose(phys, right).data
            arr   = arr.reshape(1, arr.shape[0], arr.shape[1])
        elif i == n - 1:
            left = psi.bond(n - 2, n - 1)
            arr  = t.transpose(left, phys).data
            arr  = arr.reshape(arr.shape[0], arr.shape[1], 1)
        else:
            left  = psi.bond(i - 1, i)
            right = psi.bond(i, i + 1)
            arr   = t.transpose(left, right, phys).data
            arr   = arr.transpose(0, 2, 1)
        arrays.append(np.asarray(arr))
    return arrays


def mps_to_exact_circuit(psi_mps, n: int) -> QuantumCircuit:
    """Build the exact MPS-isometry circuit (Lin, PRX Quantum 2, 010342 (2021))."""
    mps_arrays = _mps_to_lpr_arrays(psi_mps)
    return mps_to_circuit(mps_arrays, method="exact", shape="lpr")


def mps_to_approx_circuit(psi_mps, n: int, n_layers: int) -> QuantumCircuit:
    """
    Build a brick-wall approximate circuit for the MPS using a fixed number
    of two-qubit layers.

    NOTE ON STOCHASTICITY: the underlying mps_to_circuit(method="approximate")
    routine is randomly initialized; two calls with identical arguments can
    return circuits with different gate angles and gate counts. This is the
    only source of run-to-run variance after DMRG/ED have converged. See
    sweep_n_layers() / repeat_approx_circuit_trials() for statistics.
    """
    mps_arrays = _mps_to_lpr_arrays(psi_mps)
    return mps_to_circuit(
        mps_arrays, method="approximate", shape="lpr", num_layers=n_layers
    )


# =============================================================================
# CIRCUITMPS -- tensor-network fidelity against the DMRG MPS
# =============================================================================
def circuit_to_mps(
    qc: QuantumCircuit,
    max_bond: int = CIRCUIT_MPS_MAX_BOND,
    cutoff: float = 1e-12,
) -> qtn.CircuitMPS:
    """Convert a Qiskit circuit to a quimb CircuitMPS (no 2^N statevector)."""
    qc_t     = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    circ_mps = quimb_circuit(
        qc_t,
        quimb_circuit_class=qtn.CircuitMPS,
        max_bond=max_bond,
        cutoff=cutoff,
    )
    return circ_mps


def mps_fidelity(
    psi_dmrg: qtn.MatrixProductState,
    psi_circuit: qtn.MatrixProductState,
) -> float:
    """Compute |<psi_dmrg | psi_circuit>|^2 as an MPS overlap, O(N chi^3)."""
    overlap = psi_dmrg.H @ psi_circuit
    return float(abs(overlap) ** 2)


def mps_expec(
    psi_ket: qtn.MatrixProductState,
    mpo: qtn.MatrixProductOperator,
) -> complex:
    """
    <psi_ket | mpo | psi_ket> with explicit bra reindexing.

    Needed because mpo.lower_ind_id ('b{}') differs from
    psi_ket.site_ind_id ('k{}') for MPOs built via build_heisenberg_mpo().
    """
    bra = psi_ket.H
    bra = bra.reindex({
        psi_ket.site_ind_id.format(i): mpo.lower_ind_id.format(i)
        for i in range(psi_ket.L)
    })
    ket = psi_ket
    if mpo.upper_ind_id != psi_ket.site_ind_id:
        ket = psi_ket.reindex({
            psi_ket.site_ind_id.format(i): mpo.upper_ind_id.format(i)
            for i in range(psi_ket.L)
        })
    return complex((bra | mpo | ket) ^ all)


def verify_energy_mps(
    circ_mps: qtn.CircuitMPS,
    H_mpo: qtn.MatrixProductOperator,
    psi_dmrg: qtn.MatrixProductState,
    dmrg_energy: float,
) -> dict:
    """MPS-based energy and fidelity verification (scales to large N)."""
    psi_circ = circ_mps.psi
    norm     = float(abs(psi_circ.norm()))
    energy   = float(np.real(mps_expec(psi_circ, H_mpo)))

    abs_error      = abs(energy - dmrg_energy)
    energy_err_pct = abs_error / abs(dmrg_energy) * 100
    fid            = mps_fidelity(psi_dmrg, psi_circ)

    return dict(
        circuit_energy   =energy,
        dmrg_energy      =dmrg_energy,
        abs_error        =abs_error,
        energy_err_pct   =energy_err_pct,
        fidelity         =fid,
        circuit_mps_norm =norm,
    )


# =============================================================================
# ENERGY CLOSURE CHECK (statevector-based, for small-N cross-validation only)
# =============================================================================
def verify_energy(
    qc: QuantumCircuit,
    H: csr_matrix,
    basis: np.ndarray,
    dmrg_energy: float,
    ref_sector: np.ndarray | None = None,
) -> dict:
    """Statevector-based energy and fidelity check. Only practical for small N."""
    sv           = Statevector.from_instruction(qc).data
    psi_sector   = sv[basis]
    sector_norm2 = float(np.vdot(psi_sector, psi_sector).real)
    leak_norm2   = 1.0 - sector_norm2

    if sector_norm2 < 1e-12:
        raise RuntimeError("~zero amplitude in target Sz sector")

    psi_sector     = np.atleast_1d(np.asarray(sv[basis], dtype=complex))
    psi_sector    /= np.sqrt(sector_norm2)
    energy         = np.vdot(psi_sector, H @ psi_sector).real
    energy_err_pct = abs(energy - dmrg_energy) / abs(dmrg_energy) * 100
    fidelity       = (
        float(abs(np.vdot(ref_sector, psi_sector)) ** 2)
        if ref_sector is not None else None
    )

    return dict(
        circuit_energy =energy,
        dmrg_energy    =dmrg_energy,
        abs_error      =abs(energy - dmrg_energy),
        energy_err_pct =energy_err_pct,
        sector_norm2   =sector_norm2,
        leakage_norm2  =leak_norm2,
        fidelity       =fidelity,
    )


# =============================================================================
# SHARED-STATE CONTAINER
# =============================================================================
@dataclass
class GroundStateContext:
    """
    Everything expensive and deterministic for a given
    (chain, j1, j2, bond_dims). Build once, reuse across many approx-circuit
    trials via sweep_n_layers() / repeat_approx_circuit_trials().
    """
    chain: Chain
    j1: float
    j2: float
    bond_dims: list[int]
    n: int
    H: csr_matrix
    H_mpo: object
    basis: np.ndarray
    E_dmrg: float
    psi_mps: object
    diag: dict
    exact_qc: QuantumCircuit
    exact_resources: ResourceReport
    exact_circ_mps: qtn.CircuitMPS
    exact_verification_mps: dict
    exact_verification: Optional[dict] = None
    e_exact: Optional[float] = None
    psi_exact: Optional[np.ndarray] = None


def build_ground_state_context(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
) -> GroundStateContext:
    n      = chain.n_sites
    use_ed = n <= EXACT_DIAG_MAX_N
    print(f"\n[context]  chain={chain.name}  N={n}  J1={j1}  J2={j2}  "
          f"bond_dims={bond_dims}  "
          f"ed_fidelity={'on' if use_ed else 'off (N>%d)' % EXACT_DIAG_MAX_N}")

    H_mpo = build_heisenberg_mpo(chain, j1, j2, compress_every_n_edges)

    if use_ed:
        H, basis     = build_hamiltonian(chain, j1, j2)
        print(f"[H]  sector dim={len(basis)}")
        e_exact, psi_exact, basis, _idx_map = exact_ground_state(chain, j1, j2)
    else:
        H, basis, e_exact, psi_exact = None, None, None, None

    E_dmrg, psi_mps = run_dmrg(chain, j1, j2, bond_dims, compress_every_n_edges)
    diag = mps_diagnostics(psi_mps)
    print(f"[DMRG]  E={E_dmrg:.10f}  E/site={E_dmrg/n:.10f}  max_chi={diag['max_chi']}")

    exact_qc        = mps_to_exact_circuit(psi_mps, n)
    exact_resources = circuit_resource_report(exact_qc, label="exact", rz_eps=rz_eps)

    exact_verification = (
        verify_energy(exact_qc, H, basis, E_dmrg, psi_exact) if use_ed else None
    )

    exact_circ_mps        = circuit_to_mps(exact_qc, max_bond=circuit_mps_max_bond)
    exact_verification_mps = verify_energy_mps(
        exact_circ_mps, H_mpo, psi_mps, E_dmrg
    )
    print(f"[exact-MPS]  fidelity={exact_verification_mps['fidelity']:.10f}  "
          f"energy={exact_verification_mps['circuit_energy']:.10f}  "
          f"norm={exact_verification_mps['circuit_mps_norm']:.10f}")

    return GroundStateContext(
        chain=chain, j1=j1, j2=j2, bond_dims=bond_dims, n=n,
        H=H, H_mpo=H_mpo, basis=basis,
        e_exact=e_exact, psi_exact=psi_exact,
        E_dmrg=E_dmrg, psi_mps=psi_mps, diag=diag,
        exact_qc=exact_qc,
        exact_resources=exact_resources,
        exact_verification=exact_verification,
        exact_circ_mps=exact_circ_mps,
        exact_verification_mps=exact_verification_mps,
    )


# =============================================================================
# APPROXIMATE CIRCUIT TRIALS
# =============================================================================
def _approx_trial(
    ctx: GroundStateContext,
    n_layers: int,
    rz_eps: float,
    verbose: bool,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    max_retries: int = 5,
) -> dict:
    """
    One stochastic re-fit of the approximate circuit + its verification.

    Retries on QiskitError (qiskit-terra#4159: TwoQubitWeylDecomposition can
    fail on degenerate random inits). Each retry is an independent random draw.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            approx_qc        = mps_to_approx_circuit(ctx.psi_mps, ctx.n, n_layers)
            approx_resources = circuit_resource_report(
                approx_qc, label="approximate", rz_eps=rz_eps, verbose=verbose
            )

            approx_verification = None
            if ctx.n < EXACT_DIAG_MAX_N:
                approx_verification = verify_energy(
                    approx_qc, ctx.H, ctx.basis, ctx.E_dmrg, ctx.psi_exact
                )

            approx_circ_mps        = circuit_to_mps(approx_qc, max_bond=circuit_mps_max_bond)
            approx_verification_mps = verify_energy_mps(
                approx_circ_mps, ctx.H_mpo, ctx.psi_mps, ctx.E_dmrg
            )

            row = {}
            row.update({f"approx_{k}": v for k, v in approx_resources.as_row().items()})
            if approx_verification is not None:
                row.update({f"approx_{k}": v for k, v in approx_verification.items()})
            row.update({f"approx_{k}_mps": v for k, v in approx_verification_mps.items()})
            row["n_retries"] = attempt
            return row

        except QiskitError as e:
            last_err = e
            if verbose:
                print(f"  [warn] synthesis failure on attempt "
                      f"{attempt+1}/{max_retries} (n_layers={n_layers}); "
                      f"re-drawing stochastic fit -- {e}")

    raise RuntimeError(
        f"_approx_trial: {max_retries} consecutive synthesis failures at "
        f"n_layers={n_layers}."
    ) from last_err


def repeat_approx_circuit_trials(
    ctx: GroundStateContext,
    n_layers: int,
    n_trials: int = DEFAULT_N_TRIALS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    verbose: bool = False,
) -> pd.DataFrame:
    """Re-fit the approximate circuit n_trials times; return one row per trial."""
    rows = []
    for t in range(n_trials):
        if verbose:
            print(f"\n[trial {t+1}/{n_trials}]  n_layers={n_layers}")
        row           = _approx_trial(
            ctx, n_layers, rz_eps,
            verbose=verbose,
            circuit_mps_max_bond=circuit_mps_max_bond,
        )
        row["trial"]    = t
        row["n_layers"] = n_layers
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SINGLE-POINT ENTRY POINT
# =============================================================================
def run(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    n_layers: int = 3,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
) -> tuple[QuantumCircuit, QuantumCircuit, csr_matrix, np.ndarray, dict, dict]:
    """
    Run DMRG on *chain*, compile exact and approximate circuits, verify
    both energies and fidelities (statevector + MPS).

    Returns
    -------
    exact_qc, approx_qc, H, basis, exact_info, approx_info
    """
    n  = chain.n_sites
    ed = n < EXACT_DIAG_MAX_N
    print(f"\n[run]  chain={chain.name}  N={n}  J1={j1}  J2={j2}  "
          f"bond_dims={bond_dims}  n_layers={n_layers}")

    ctx = build_ground_state_context(
        chain, j1, j2, bond_dims, rz_eps,
        circuit_mps_max_bond=circuit_mps_max_bond,
        compress_every_n_edges=compress_every_n_edges,
    )

    approx_qc        = mps_to_approx_circuit(ctx.psi_mps, ctx.n, n_layers)
    approx_resources = circuit_resource_report(approx_qc, label="approximate", rz_eps=rz_eps)

    approx_verification = None
    if ed:
        approx_verification = verify_energy(
            approx_qc, ctx.H, ctx.basis, ctx.E_dmrg, ctx.psi_exact
        )

    approx_circ_mps        = circuit_to_mps(approx_qc, max_bond=circuit_mps_max_bond)
    approx_verification_mps = verify_energy_mps(
        approx_circ_mps, ctx.H_mpo, ctx.psi_mps, ctx.E_dmrg
    )

    base = dict(
        lattice    =chain.name,
        n_sites    =n,
        j1         =j1,
        j2         =j2,
        j2_over_j1 =(j2 / j1) if j1 else float("nan"),
        n_layers   =n_layers,
        dmrg_energy=ctx.E_dmrg,
        **ctx.diag,
    )

    exact_info = {
        **base,
        **{f"exact_{k}": v for k, v in ctx.exact_resources.as_row().items()},
        **{f"exact_{k}_mps": v for k, v in ctx.exact_verification_mps.items()},
    }
    approx_info = {
        **base,
        **{f"approx_{k}": v for k, v in approx_resources.as_row().items()},
        **{f"approx_{k}_mps": v for k, v in approx_verification_mps.items()},
    }
    if ed:
        exact_info.update(
            {f"exact_{k}": v for k, v in ctx.exact_verification.items()}
        )
        approx_info.update(
            {f"approx_{k}": v for k, v in approx_verification.items()}
        )

    return ctx.exact_qc, approx_qc, ctx.H, ctx.basis, exact_info, approx_info


# =============================================================================
# N_LAYERS SWEEP
# =============================================================================
SUMMARY_METRICS = [
    "approx_fidelity",
    "approx_energy_err_pct",
    "approx_abs_error",
    "approx_leakage_norm2",
    "approx_fidelity_mps",
    "approx_energy_err_pct_mps",
    "approx_abs_error_mps",
    "approx_circuit_mps_norm_mps",
    "approx_cx_count",
    "approx_depth",
    "approx_non_clifford_count",
    "approx_non_clifford_depth",
    "approx_t_count_estimate",
]


def _summarize(df_trials: pd.DataFrame, metrics: list[str]) -> dict:
    """Mean/std/min/max/median for each metric column, ignoring NaNs."""
    out = {}
    for m in metrics:
        if m not in df_trials.columns:
            continue
        vals = pd.to_numeric(df_trials[m], errors="coerce").dropna()
        if len(vals) == 0:
            out.update({f"{m}_{s}": float("nan")
                        for s in ("mean", "std", "min", "max", "median")})
            continue
        out[f"{m}_mean"]   = float(vals.mean())
        out[f"{m}_std"]    = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{m}_min"]    = float(vals.min())
        out[f"{m}_max"]    = float(vals.max())
        out[f"{m}_median"] = float(vals.median())
    return out


def sweep_n_layers(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    layers: list[int] | None = None,
    n_trials: int = DEFAULT_N_TRIALS,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    verbose_trials: bool = False,
    return_all_trials: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep over n_layers values. DMRG/ED solved ONCE; approximate circuit
    re-fit n_trials times per layer. Returns a summary DataFrame (and
    optionally all raw trial rows).
    """
    if layers is None:
        layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"\n[sweep_n_layers]  layers={layers}  n_trials={n_trials}  "
          f"(DMRG/ED solved once, approx-circuit re-fit {n_trials}x per layer)")

    ctx = build_ground_state_context(
        chain, j1, j2, bond_dims, rz_eps,
        circuit_mps_max_bond=circuit_mps_max_bond,
        compress_every_n_edges=compress_every_n_edges,
    )

    summary_rows   = []
    all_trial_dfs  = []
    for n_lay in layers:
        print(f"\n[sweep_n_layers]  n_layers={n_lay}  "
              f"running {n_trials} independent approx-circuit fits...")
        df_trials = repeat_approx_circuit_trials(
            ctx, n_lay, n_trials=n_trials, rz_eps=rz_eps,
            circuit_mps_max_bond=circuit_mps_max_bond,
            verbose=verbose_trials,
        )
        all_trial_dfs.append(df_trials)

        summary = dict(
            n_layers   =n_lay,
            n_trials   =n_trials,
            lattice    =chain.name,
            n_sites    =ctx.n,
            j1=j1, j2=j2,
            j2_over_j1 =(j2 / j1) if j1 else float("nan"),
            dmrg_energy=ctx.E_dmrg,
            **ctx.diag,
            exact_fidelity_mps      =ctx.exact_verification_mps.get("fidelity"),
            exact_energy_err_pct_mps=ctx.exact_verification_mps.get("energy_err_pct"),
            exact_cx_count          =ctx.exact_resources.cx_count,
            exact_depth             =ctx.exact_resources.depth,
        )
        if ctx.n < EXACT_DIAG_MAX_N:
            summary.update({
                "exact_energy_err_pct": ctx.exact_verification.get("energy_err_pct"),
                "exact_fidelity"      : ctx.exact_verification.get("fidelity"),
            })
        summary.update(_summarize(df_trials, SUMMARY_METRICS))
        summary_rows.append(summary)

        fmean = summary.get("approx_fidelity_mps_mean", float("nan"))
        fstd  = summary.get("approx_fidelity_mps_std",  float("nan"))
        emean = summary.get("approx_energy_err_pct_mps_mean", float("nan"))
        estd  = summary.get("approx_energy_err_pct_mps_std",  float("nan"))
        print(f"  -> MPS fidelity = {fmean:.6f} +/- {fstd:.6f}   "
              f"MPS energy_err% = {emean:.4f} +/- {estd:.4f}   "
              f"(n_trials={n_trials})")

    df_summary = pd.DataFrame(summary_rows)
    print("\n[sweep_n_layers]  summary (mean over n_trials independent fits)")
    print(df_summary.to_string(index=False))

    if return_all_trials:
        return df_summary, pd.concat(all_trial_dfs, ignore_index=True)
    return df_summary


# =============================================================================
# TIMING HARNESS
# =============================================================================
class StageTimer:
    """Collects (stage_name, elapsed_seconds) in call order."""

    def __init__(self):
        self.records = []

    @contextmanager
    def time(self, stage_name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed      = time.perf_counter() - t0
            self.records.append((stage_name, elapsed))
            running_total = sum(t for _, t in self.records)
            print(f"    [{stage_name:<32s}] {elapsed:8.3f}s   "
                  f"(cumulative: {running_total:8.3f}s)")

    def total(self):
        return sum(t for _, t in self.records)


def _harness_one_size(
    n_sites: int,
    bond_dims: list[int],
    compress_every_n_edges: int,
    n_layers: int,
    dmrg_tol_candidates: list[float],
) -> StageTimer:
    """
    Time all pipeline stages for a single chain size.

    Stages timed:
      1. MPO construction with incremental compression.
      2. DMRG, per real sweep on a single DMRG2 object (faithful to one
         DMRG2.solve() call, with sweep-level visibility).
      3. mps_diagnostics().
      4. mps_to_approx_circuit() at the given n_layers.
      5. circuit_resource_report() on the approx circuit.

    After DMRG, a retroactive DMRG_TOL sensitivity report is printed for
    each candidate tolerance at zero additional DMRG cost.
    """
    print(f"\n{'='*70}")
    print(f"  N = {n_sites}   bond_dims = {bond_dims}")
    print(f"{'='*70}")

    timer = StageTimer()
    chain = make_chain(n_sites)

    use_ed = n_sites <= EXACT_DIAG_MAX_N
    print(f"  EXACT_DIAG_MAX_N={EXACT_DIAG_MAX_N}  -> "
          f"ED {'WILL run (small N)' if use_ed else 'will be skipped'} at this size")

    # ---- 1. MPO construction with incremental compression ----------------
    H_mpo = None
    try:
        edge_sets = [(chain.nn_edges, J1)]
        if J2 and chain.nnn_edges:
            edge_sets.append((chain.nnn_edges, J2))
        all_edges = [
            (i, jx, j)
            for edges, j in edge_sets
            for (i, jx) in edges
        ]

        with timer.time("mpo_build_incremental_total"):
            compress_round = 0
            for edge_idx, (i, jx, j) in enumerate(all_edges, start=1):
                term  = _heisenberg_term_mpo(n_sites, i, jx, j)
                H_mpo = term if H_mpo is None else H_mpo + term
                if edge_idx % compress_every_n_edges == 0 or edge_idx == len(all_edges):
                    compress_round += 1
                    H_mpo.compress(cutoff=1e-12)

        try:
            max_bond_final = max(
                H_mpo.bond_size(k, k + 1) for k in range(n_sites - 1)
            )
            print(f"    -> MPO max bond dim AFTER incremental build+compress "
                  f"({compress_round} compress rounds, every "
                  f"{compress_every_n_edges} edges): {max_bond_final}")
        except Exception:
            pass

    except Exception:
        print("  [build_heisenberg_mpo (incremental) FAILED]")
        traceback.print_exc()

    # ---- 2. DMRG, per-sweep on a single DMRG2 object --------------------
    psi_mps   = None
    E_dmrg    = None
    sweep_log = []   # (sweep_num, elapsed, energy, cumulative) for tol sensitivity
    try:
        if H_mpo is None:
            raise RuntimeError("H_mpo was not built -- skipping DMRG")

        psi0 = qtn.MPS_neel_state(n_sites)
        dmrg = qtn.DMRG2(
            H_mpo, bond_dims=bond_dims, cutoffs=DMRG_CUTOFF, p0=psi0
        )

        prev_energy = None
        converged   = False
        sweep_num   = 0
        cumulative  = 0.0

        while sweep_num < DMRG_MAX_SWEEPS and not converged:
            sweep_num   += 1
            stage_name   = f"run_dmrg_sweep_{sweep_num:02d}"
            t0           = time.perf_counter()
            with timer.time(stage_name):
                converged = dmrg.solve(tol=DMRG_TOL, max_sweeps=1, verbosity=0)
            sweep_elapsed = time.perf_counter() - t0
            cumulative   += sweep_elapsed

            energy = float(np.real(dmrg.energies[-1])) if dmrg.energies else float("nan")
            sweep_log.append((sweep_num, sweep_elapsed, energy, cumulative))

            current_chi = bond_dims[min(sweep_num - 1, len(bond_dims) - 1)]
            delta = "" if prev_energy is None else f"  dE={energy - prev_energy:+.2e}"
            print(f"    -> sweep={sweep_num:<3d} chi_target={current_chi:<4d} "
                  f"E={energy:.10f}{delta}  converged={converged}")
            prev_energy = energy
            E_dmrg      = energy

        psi_mps = dmrg.state
        print(f"    -> DMRG finished after {sweep_num} real sweep(s), "
              f"converged={converged}  (actual DMRG_TOL={DMRG_TOL})")

        # Retroactive DMRG_TOL sensitivity report
        if len(sweep_log) >= 2:
            print(f"    -- DMRG_TOL sensitivity (retroactive, real run stopped "
                  f"at sweep {sweep_num}, {cumulative:.1f}s) --")
            for cand_tol in dmrg_tol_candidates:
                stop_at = None
                for k in range(1, len(sweep_log)):
                    _, _, e_curr, _ = sweep_log[k]
                    _, _, e_prev, _ = sweep_log[k - 1]
                    if abs(e_curr - e_prev) < cand_tol:
                        stop_at = k + 1
                        break
                if stop_at is None:
                    print(f"       tol={cand_tol:<10g} -> never reached within "
                          f"the {len(sweep_log)} sweeps actually run")
                else:
                    stop_cum_time = sweep_log[stop_at - 1][3]
                    saved = cumulative - stop_cum_time
                    print(f"       tol={cand_tol:<10g} -> would stop at sweep "
                          f"{stop_at:<3d} ({stop_cum_time:6.1f}s)   "
                          f"saved={saved:7.1f}s vs actual run")

    except Exception:
        print("  [run_dmrg (per-sweep) FAILED]")
        traceback.print_exc()

    # ---- 3. MPS diagnostics ----------------------------------------------
    if psi_mps is not None:
        try:
            with timer.time("mps_diagnostics"):
                diag = mps_diagnostics(psi_mps)
            print(f"    -> max_chi={diag['max_chi']}  "
                  f"half_chain_entropy={diag['half_chain_entropy']:.4f}")
        except Exception:
            print("  [mps_diagnostics FAILED]")
            traceback.print_exc()

    # ---- 4. Approximate (brick-wall) circuit synthesis -------------------
    approx_qc = None
    if psi_mps is not None:
        max_retries = 5
        for attempt in range(max_retries):
            try:
                with timer.time(f"mps_to_approx_circuit_nlayers_{n_layers}"):
                    approx_qc = mps_to_approx_circuit(psi_mps, n_sites, n_layers)
                print(f"    -> approx circuit (n_layers={n_layers}): "
                      f"{approx_qc.num_qubits} qubits, "
                      f"{sum(approx_qc.count_ops().values())} raw gates"
                      + (f"  (succeeded on retry {attempt})" if attempt else ""))
                break
            except QiskitError as e:
                print(f"  [warn] synthesis failure on attempt "
                      f"{attempt+1}/{max_retries} (n_layers={n_layers}); "
                      f"re-drawing stochastic fit -- {e}")
        else:
            print(f"  [mps_to_approx_circuit FAILED] {max_retries} consecutive "
                  f"synthesis failures at n_layers={n_layers}")

    # ---- 5. Resource report on the approx circuit -----------------------
    if approx_qc is not None:
        try:
            with timer.time("circuit_resource_report_approx"):
                report = circuit_resource_report(approx_qc, label="approx", verbose=False)
            print(f"    -> depth={report.depth}  cx={report.cx_count}  "
                  f"rz={report.non_clifford_count}")
        except Exception:
            print("  [circuit_resource_report FAILED]")
            traceback.print_exc()

    print(f"\n  TOTAL for N={n_sites}: {timer.total():.3f}s")
    return timer


def run_timing_harness(
    chain_sizes: list[int] = CHAIN_SIZES,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    n_layers: int = N_LAYERS,
    dmrg_tol_candidates: list[float] = DMRG_TOL_CANDIDATES,
) -> dict[int, StageTimer]:
    """
    Run the full timing harness over all chain sizes and print a cross-size
    summary table. Returns a dict mapping n_sites -> StageTimer.

    Edit the module-level constants (CHAIN_SIZES, COMPRESS_EVERY_N_EDGES,
    N_LAYERS, DMRG_TOL_CANDIDATES) or pass overrides here before calling.
    """
    all_timers = {}
    for n_sites in chain_sizes:
        all_timers[n_sites] = _harness_one_size(
            n_sites, bond_dims, compress_every_n_edges,
            n_layers, dmrg_tol_candidates,
        )

    # Cross-size summary table
    print(f"\n{'='*70}")
    print("  SUMMARY (seconds per stage, by chain size)")
    print(f"{'='*70}")

    stage_names = []
    for timer in all_timers.values():
        for name, _ in timer.records:
            if name not in stage_names:
                stage_names.append(name)

    header = f"{'stage':<32s}" + "".join(f"{f'N={n}':>12s}" for n in chain_sizes)
    print(header)
    print("-" * len(header))
    for stage in stage_names:
        row = f"{stage:<32s}"
        for n_sites in chain_sizes:
            d   = dict(all_timers[n_sites].records)
            val = d.get(stage)
            row += f"{val:>12.3f}" if val is not None else f"{'--':>12s}"
        print(row)
    print("-" * len(header))
    totals_row = f"{'TOTAL':<32s}"
    for n_sites in chain_sizes:
        totals_row += f"{all_timers[n_sites].total():>12.3f}"
    print(totals_row)

    return all_timers
