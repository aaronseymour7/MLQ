"""
DMRG -> MPS -> Circuit pipeline, with resource accounting, fidelity
verification (statevector- and MPS-based), a research-grade n_layers sweep,
and an optional per-stage timing harness for scaling studies.

Returns
-------
  run() -> (exact_qc, approx_qc, H_sparse, basis, exact_info, approx_info)

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

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.exceptions import QiskitError
from qiskit_quimb import quimb_circuit

# -- quimb ---------------------------------------------------------------
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
    import qiskit
    print(f"[Qiskit]  {qiskit.__version__}")
except ImportError:
    raise ImportError("pip install qiskit")


# =============================================================================
# CONFIG
# =============================================================================
J1             = 1.0
J2             = 0.0
DMRG_BOND_DIMS = [10, 20, 40, 80, 100]   # ramped bond-dim schedule for DMRG2
DMRG_CUTOFF    = 1e-6
DMRG_TOL       = 1e-6
DMRG_MAX_SWEEPS = 40

# Default max bond dim for CircuitMPS contraction. Larger values give a more
# faithful MPS representation of the circuit output at higher cost; for the
# exact isometry circuit the true bond dim is bounded by DMRG_BOND_DIMS[-1],
# so matching that ceiling is sufficient. For approximate (brick-wall)
# circuits at low n_layers the bond dim is much smaller, and the contraction
# will automatically stay well below this ceiling.
CIRCUIT_MPS_MAX_BOND = 200

# Gate set used for resource accounting. CX/H/S/Sdg are Clifford; RZ is the
# only non-Clifford gate, so its count/depth is exactly the quantity that
# determines Clifford+T synthesis cost on a fault-tolerant device.
BASIS_GATES        = ["cx", "rz", "h", "s", "sdg"]
NON_CLIFFORD_GATES = {"rz"}
TWO_QUBIT_GATES    = {"cx"}

# Default synthesis precision used for the T-count estimate (per RZ gate).
RZ_SYNTHESIS_EPS = 1e-3

# Default number of independent re-fits of the approximate circuit per
# n_layers value, used by sweep_n_layers() for research-grade statistics.
DEFAULT_N_TRIALS = 100

EXACT_DIAG_MAX_N = 4

# Default settings for the timing harness (time_chain_sizes). Override by
# passing explicit arguments -- these are not read from argv/env anywhere.
TIMING_CHAIN_SIZES            = [20, 50, 80]
TIMING_COMPRESS_EVERY_N_EDGES = 4   # compress the running MPO after every N
                                     # edges added (1 = most aggressive)
TIMING_N_LAYERS                = 3  # n_layers for the approx circuit being timed
TIMING_DMRG_TOL_CANDIDATES     = [1e-6, 1e-7, 1e-8, 1e-9, DMRG_TOL]


# =============================================================================
# CHAIN TOPOLOGY  (replaces lattices.BaseLattice -- 1D open chain only)
# =============================================================================
@dataclass
class Chain:
    """Minimal stand-in for the old BaseLattice, 1D open chain only."""
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


def _edge_sets(chain: Chain, j1: float, j2: float) -> list[tuple[list[tuple[int, int]], float]]:
    """Shared helper: (edges, coupling) pairs for J1 (nn) and, if nonzero, J2
    (nnn). Used by every Hamiltonian/MPO builder below so the topology logic
    lives in exactly one place."""
    edge_sets = [(chain.nn_edges, j1)]
    if j2 and chain.nnn_edges:
        edge_sets.append((chain.nnn_edges, j2))
    return edge_sets


# =============================================================================
# RESOURCE ACCOUNTING  (early-fault-tolerant oriented)
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
    Estimated number of T gates needed to synthesize a single arbitrary-angle
    RZ rotation to within precision eps in a Clifford+T gate set, using the
    asymptotically-optimal Ross-Selinger bound T ~ 3*log2(1/eps) + O(loglog).
    This is a standard order-of-magnitude estimate used in early-FT resource
    papers -- it is NOT an exact circuit synthesis, just a way to translate
    "number of arbitrary rotations" into "number of T gates" for a poster.
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
    qc_t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    counts = dict(qc_t.count_ops())

    cx_count = sum(counts.get(g, 0) for g in TWO_QUBIT_GATES)
    non_clifford_count = sum(counts.get(g, 0) for g in NON_CLIFFORD_GATES)

    depth = qc_t.depth()
    non_clifford_depth = qc_t.depth(
        filter_function=lambda instr: instr.operation.name in NON_CLIFFORD_GATES
    )

    t_per_rz = estimate_t_count_per_rz(rz_eps)
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
# HAMILTONIAN  (sparse, fixed-Sz sector) -- ED reference, used for verification
# =============================================================================
def build_basis(n: int) -> np.ndarray:
    """
    Sz=0 sector basis. Bit convention: bit=0 <-> up (Sz=+1/2),
    bit=1 <-> down (Sz=-1/2) -- matches quimb's qu.up()=[1,0]=|0>,
    qu.down()=[0,1]=|1>, so this lines up directly with Qiskit qubit
    ordering (site i <-> qubit i) with no remapping needed.
    """
    n_up = n // 2
    n_down = n - n_up
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_down],
                    dtype=np.int64)


def _build_sector_coo(chain: Chain, j1: float, j2: float):
    """
    Shared core for build_hamiltonian() and exact_ground_state(): builds the
    Sz=0 sector basis and the (rows, cols, vals) COO entries of H in that
    sector. Both callers previously duplicated this double loop; consolidated
    here so there is exactly one place that encodes the bit convention.
    """
    n = chain.n_sites
    basis = build_basis(n)
    idx_map = {int(b): i for i, b in enumerate(basis)}

    rows, cols, vals = [], [], []
    for edges, j in _edge_sets(chain, j1, j2):
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = -0.5 if (bits >> si) & 1 else 0.5
                zj = -0.5 if (bits >> sj) & 1 else 0.5
                rows.append(row); cols.append(row); vals.append(j * zi * zj)
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl = int(bits) ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(fl, -1)
                    if col >= 0:
                        rows.append(row); cols.append(col); vals.append(0.5 * j)

    return basis, idx_map, rows, cols, vals


def build_hamiltonian(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Return (H_sparse, basis) in the n//2-up sector.

    Edge topology comes from chain.nn_edges (J1) and chain.nnn_edges (J2).

    Bit convention: bit `i` of a basis integer is the Sz state of site i
    (bit=0 -> up/+0.5, bit=1 -> down/-0.5), matching the qubit ordering
    produced by mps_to_circuit from quimb MPS arrays (qubit i <-> site i,
    quimb |0>=up, little-endian Qiskit statevector indexing). This is what
    lets verify_energy() compare a circuit statevector directly against
    this Hamiltonian without any index remapping.
    """
    basis, _, rows, cols, vals = _build_sector_coo(chain, j1, j2)
    dim = len(basis)
    H = lil_matrix((dim, dim), dtype=np.float64)
    for r, c, v in zip(rows, cols, vals):
        H[r, c] += v
    return csr_matrix(H), basis


def exact_ground_state(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """
    Lanczos ground state in the Sz=0 sector via scipy eigsh.

    Returns (e_exact, psi_exact, basis, idx_map).
    Used as the universal reference energy and for fidelity checks.
    """
    n = chain.n_sites
    basis, idx_map, rows, cols, vals = _build_sector_coo(chain, j1, j2)
    dim = len(basis)

    r_np = np.array(rows, dtype=np.int32)
    c_np = np.array(cols, dtype=np.int32)
    v_np = np.array(vals, dtype=np.float64)

    def matvec(v):
        out = np.zeros(dim, dtype=v.dtype)
        np.add.at(out, r_np, v_np * v[c_np])
        return out

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    evals, evecs = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=10_000)
    e_exact = float(evals[0])
    psi_exact = evecs[:, 0] / np.linalg.norm(evecs[:, 0])

    print(f"[ED]  E_exact={e_exact:.10f}  E/site={e_exact/n:.10f}")
    return e_exact, psi_exact, basis, idx_map


# =============================================================================
# QUIMB MPO  -- explicit edge-list Heisenberg Hamiltonian
# =============================================================================
def _two_site_mpo(n: int, i: int, j: int, op_i, op_j, coeff: float = 1.0):
    """One operator-string term coeff*op_i(i) (x) op_j(j), identity elsewhere,
    as a bond-dim-1 MPO. Works for ANY i,j (not just nearest neighbours)."""
    I2 = qu.eye(2)
    ops = [I2] * n
    ops[i] = coeff * op_i
    ops[j] = op_j
    return qtn.MPO_product_operator(ops)


def _heisenberg_term_mpo(n: int, i: int, j: int, coupling: float):
    """S_i . S_j = Sz_i Sz_j + 1/2(S+_i S-_j + S-_i S+_j), as a sum of three
    bond-dim-1 MPOs (sum has bond dim 3 before compression)."""
    Sz, Sp, Sm = qu.spin_operator('z'), qu.spin_operator('+'), qu.spin_operator('-')
    m = _two_site_mpo(n, i, j, Sz, Sz, coupling)
    m += _two_site_mpo(n, i, j, Sp, Sm, coupling / 2.0)
    m += _two_site_mpo(n, i, j, Sm, Sp, coupling / 2.0)
    return m


def build_heisenberg_mpo(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    compress_every_n_edges: Optional[int] = None,
    cutoff: float = 1e-12,
    verbose: bool = False,
):
    """
    Build the full J1-J2 Heisenberg MPO by summing one _heisenberg_term_mpo
    per edge, then compressing.

    By default (compress_every_n_edges=None) this compresses ONCE at the end,
    matching the original simple behaviour. Passing a small integer (e.g. 4)
    instead compresses the running MPO every N edges, which keeps the
    intermediate bond dimension from ballooning on long chains -- at N=50 the
    "compress once at the end" approach let bond dim grow to 147 before a
    single compress() call that alone cost ~38s, whereas periodic compression
    keeps every "+term" addition cheap throughout the build. verbose=True
    prints the resulting max bond dimension and number of compression rounds
    performed, which is useful when tuning compress_every_n_edges for a given
    chain length.
    """
    n = chain.n_sites
    all_edges = [(i, jx, j) for edges, j in _edge_sets(chain, j1, j2) for (i, jx) in edges]

    H_mpo = None
    compress_round = 0
    step = compress_every_n_edges or len(all_edges) or 1
    for edge_idx, (i, jx, j) in enumerate(all_edges, start=1):
        term = _heisenberg_term_mpo(n, i, jx, j)
        H_mpo = term if H_mpo is None else H_mpo + term
        if edge_idx % step == 0 or edge_idx == len(all_edges):
            compress_round += 1
            H_mpo.compress(cutoff=cutoff)

    if verbose and H_mpo is not None and n > 1:
        max_bond = max(H_mpo.bond_size(k, k + 1) for k in range(n - 1))
        print(f"    -> MPO max bond dim after build+compress "
              f"({compress_round} compress round(s), every {step} edges): {max_bond}")

    return H_mpo


# =============================================================================
# DMRG
# =============================================================================
def run_dmrg(
    chain: Chain,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    H_mpo=None,
):
    """
    Run DMRG on the chain topology defined by *chain* using quimb's DMRG2.

    quimb has no Sz-conservation sectors built into its MPO-DMRG (unlike
    TeNPy's conserve='Sz'), so we start from a Neel initial state, whose
    Sz is exactly conserved by the Heisenberg Hamiltonian throughout the
    optimization -- DMRG2 will not leave that sector even though it isn't
    explicitly enforced. Verified against ED in build_ground_state_context.

    H_mpo can be passed in to reuse an already-built MPO (e.g. from the
    timing harness, which builds it incrementally); if omitted it is built
    fresh via build_heisenberg_mpo().

    Returns
    -------
    E   : DMRG ground-state energy (real float)
    psi : converged quimb MatrixProductState
    """
    n = chain.n_sites
    if H_mpo is None:
        H_mpo = build_heisenberg_mpo(chain, j1, j2)

    psi0 = qtn.MPS_neel_state(n)

    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dims, cutoffs=DMRG_CUTOFF, p0=psi0)
    dmrg.solve(tol=DMRG_TOL, max_sweeps=DMRG_MAX_SWEEPS, verbosity=0)

    E = float(np.real(dmrg.energy))
    psi = dmrg.state
    return E, psi


def mps_diagnostics(psi) -> dict:
    """Bond-dimension and entanglement diagnostics for the converged MPS.

    max_chi / mean_chi describe the compression cost of the state; the
    half-chain entanglement entropy is the standard order parameter for
    tracking how close the chain is to a critical point (entropy grows
    with system size at criticality, saturates away from it).
    """
    n = psi.L
    chi = [psi.bond_size(i, i + 1) for i in range(n - 1)]
    half = max(0, n // 2 - 1)
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
    Pull this quimb MatrixProductState's site tensors out as a list of
    'lpr' (left-bond, physical, right-bond) shaped numpy arrays, which is
    what mps_to_circuit expects. Boundary tensors get a size-1 dummy bond
    on the missing side (mps_to_circuit squeezes these internally).

    Left/right bonds are resolved via psi.bond(i, j) rather than by
    assuming an index order, since that's the documented, robust way to
    identify which leg connects to which neighbour in quimb.
    """
    n = psi.L
    arrays = []
    for i in range(n):
        t = psi[i]
        phys = next(ix for ix in t.inds if ix.startswith('k'))
        if i == 0:
            right = psi.bond(0, 1)
            arr = t.transpose(phys, right).data
            arr = arr.reshape(1, arr.shape[0], arr.shape[1])
        elif i == n - 1:
            left = psi.bond(n - 2, n - 1)
            arr = t.transpose(left, phys).data
            arr = arr.reshape(arr.shape[0], arr.shape[1], 1)
        else:
            left = psi.bond(i - 1, i)
            right = psi.bond(i, i + 1)
            arr = t.transpose(left, right, phys).data
            arr = arr.transpose(0, 2, 1)
        arrays.append(np.asarray(arr))
    return arrays


def mps_to_exact_circuit(psi_mps, n: int) -> QuantumCircuit:
    """
    Build the EXACT MPS-isometry circuit (Lin, PRX Quantum 2, 010342 (2021)).

    We deliberately do not offer the brick-wall "approximate" method here:
    for an early-fault-tolerant resource study we want the true cost of
    preparing the state DMRG actually found, not a depth-truncated
    heuristic. mps_to_circuit expects tensors in 'lpr' shape
    (left-bond, physical, right-bond).

    Note: this circuit's depth grows very fast with n (e.g. depth in the
    millions at n=50) -- the timing harness intentionally skips this stage
    and times only the approximate circuit instead.
    """
    mps_arrays = _mps_to_lpr_arrays(psi_mps)
    qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")
    return qc


def mps_to_approx_circuit(psi_mps, n: int, n_layers: int) -> QuantumCircuit:
    """
    Build a brick-wall approximate circuit for the MPS using a fixed number
    of two-qubit layers. Fewer layers = shallower circuit but lower fidelity.
    mps_to_circuit expects tensors in 'lpr' shape (left-bond, physical, right-bond).

    NOTE ON STOCHASTICITY: the underlying mps_to_circuit(method="approximate")
    routine fits the brick-wall unitaries via a local/iterative optimization
    that is randomly initialized internally (and does not expose a seed
    argument in the public API at the time of writing). Two calls with
    identical arguments can therefore return circuits with different gate
    angles, different fidelity to the target MPS, and -- after transpilation
    -- even different gate counts (since transpile(optimization_level=0)
    on a different set of angles can synthesize a different RZ/CX layout).
    This is the *only* source of run-to-run variance in this pipeline once
    DMRG/ED have converged, which is why repeated trials should re-call only
    this function (see sweep_n_layers / repeat_approx_circuit_trials below)
    rather than re-running the whole DMRG+ED pipeline.
    """
    mps_arrays = _mps_to_lpr_arrays(psi_mps)
    qc = mps_to_circuit(mps_arrays, method="approximate", shape="lpr", num_layers=n_layers)
    return qc


def _fit_approx_circuit_with_retries(
    psi_mps,
    n: int,
    n_layers: int,
    max_retries: int = 5,
    verbose: bool = False,
) -> QuantumCircuit:
    """
    Shared retry wrapper around mps_to_approx_circuit(): the brick-wall
    optimizer is randomly initialized and occasionally converges to a
    two-qubit block that Qiskit's TwoQubitWeylDecomposition fails to
    diagonalize (QiskitError, a known upstream edge case --
    qiskit-terra#4159). Since the fit is already stochastic, the simplest
    fix is to throw the bad draw away and re-fit with an independent random
    initialization. Used by both _approx_trial() (pipeline) and the timing
    harness, which previously duplicated this retry loop.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            return mps_to_approx_circuit(psi_mps, n, n_layers)
        except QiskitError as e:
            last_err = e
            if verbose:
                print(f"  [warn] synthesis failure on attempt "
                      f"{attempt + 1}/{max_retries} (n_layers={n_layers}); "
                      f"re-drawing stochastic fit -- {e}")
    raise RuntimeError(
        f"{max_retries} consecutive synthesis failures at n_layers={n_layers}. "
        f"This many failures in a row suggests something structural (e.g. "
        f"n_layers too small/large for n, or a version mismatch), not just "
        f"bad luck in the random init."
    ) from last_err


# =============================================================================
# CIRCUITMPS -- tensor-network fidelity against the DMRG MPS
# =============================================================================
def circuit_to_mps(
    qc: QuantumCircuit,
    max_bond: int = CIRCUIT_MPS_MAX_BOND,
    cutoff: float = 1e-12,
) -> qtn.CircuitMPS:
    """
    Convert a Qiskit circuit to a quimb CircuitMPS object.

    The circuit is applied gate-by-gate in tensor-network space; no 2^N
    statevector is materialised. max_bond truncates the MPS bond dimension
    after each two-qubit gate, introducing a controllable approximation:
    for the exact isometry circuit this should be set >= the DMRG bond dim;
    for approximate (brick-wall) circuits a smaller value is fine since their
    intrinsic bond dim is bounded by 2^(n_layers).

    The returned CircuitMPS has site indices k0..k(N-1) matching the DMRG
    MPS convention, so mps_fidelity() needs no index remapping.
    """
    qc_t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
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
    """
    Compute |<psi_dmrg | psi_circuit>|^2 as an MPS overlap.

    Both MPS objects must share the same site-index naming (k0..kN-1),
    which is guaranteed when psi_dmrg comes from quimb DMRG2 and
    psi_circuit comes from CircuitMPS.psi. The contraction is O(N chi^3)
    and avoids exponential statevector construction entirely.
    """
    overlap = psi_dmrg.H @ psi_circuit
    return float(abs(overlap) ** 2)


def mps_expec(
    psi_ket: qtn.MatrixProductState,
    mpo: qtn.MatrixProductOperator,
) -> complex:
    """
    <psi_ket | mpo | psi_ket>, computed with an explicit bra reindex.

    Needed because mpo.lower_ind_id ('b{}') differs from
    psi_ket.site_ind_id ('k{}') for MPOs built via build_heisenberg_mpo()
    (MPO_product_operator + sum + compress leaves the default upper/lower
    ind id pairing of 'k{}'/'b{}' rather than both matching the state's
    site_ind_id). Neither psi_circ.H.expec(mpo) nor psi_circ.expec(mpo)
    correctly renames the bra's physical legs to 'b{}' in this quimb
    version -- both leave all N bra legs open instead of contracting
    them against the MPO's lower legs, returning an unreduced Tensor of
    shape (2,)*N instead of a scalar. This function does the bra (and,
    defensively, ket) reindexing explicitly so the contraction is
    guaranteed to close every physical index regardless of quimb's
    .expec()/.H auto-detection behaviour.
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
    psi_circ = circ_mps.psi

    norm = float(abs(psi_circ.norm()))
    energy = float(np.real(mps_expec(psi_circ, H_mpo)))

    abs_error = abs(energy - dmrg_energy)
    energy_err_pct = abs_error / abs(dmrg_energy) * 100

    fid = mps_fidelity(psi_dmrg, psi_circ)

    return dict(
        circuit_energy=energy,
        dmrg_energy=dmrg_energy,
        abs_error=abs_error,
        energy_err_pct=energy_err_pct,
        fidelity=fid,
        circuit_mps_norm=norm,
    )


# =============================================================================
# ENERGY CLOSURE CHECK (statevector-based, kept for small-N cross-validation)
# =============================================================================
def verify_energy(
    qc: QuantumCircuit,
    H: csr_matrix,
    basis: np.ndarray,
    dmrg_energy: float,
    ref_sector: np.ndarray | None = None,   # exact ground state in sector
) -> dict:
    sv = Statevector.from_instruction(qc).data
    psi_sector = np.asarray(sv[basis], dtype=complex)
    psi_sector = np.atleast_1d(psi_sector)

    sector_norm2 = float(np.vdot(psi_sector, psi_sector).real)
    leak_norm2 = 1.0 - sector_norm2

    if sector_norm2 < 1e-12:
        raise RuntimeError("~zero amplitude in target Sz sector")

    psi_sector = psi_sector / np.sqrt(sector_norm2)

    energy = np.vdot(psi_sector, H @ psi_sector).real
    energy_err_pct = abs(energy - dmrg_energy) / abs(dmrg_energy) * 100

    # fidelity vs the converged ground state (ED), not vs exact circuit
    fidelity = float(abs(np.vdot(ref_sector, psi_sector)) ** 2) if ref_sector is not None else None

    return dict(
        circuit_energy=energy,
        dmrg_energy=dmrg_energy,
        abs_error=abs(energy - dmrg_energy),
        energy_err_pct=energy_err_pct,
        sector_norm2=sector_norm2,
        leakage_norm2=leak_norm2,
        fidelity=fidelity,   # |<psi_exact|psi_circuit>|^2
    )


# =============================================================================
# SHARED-STATE CONTAINER  (one DMRG/ED solve, reused across many approx fits)
# =============================================================================
@dataclass
class GroundStateContext:
    """
    Everything that is expensive and DETERMINISTIC for a given
    (chain, j1, j2, bond_dims): the Hamiltonian, the ED reference state,
    the converged DMRG MPS, and the exact isometry circuit + its resources.

    Building this once and reusing it across many approximate-circuit
    trials is what makes sweep_n_layers() cheap: only the stochastic
    brick-wall fit (mps_to_approx_circuit) is repeated per trial.

    H_mpo                   : quimb MPO -- needed by verify_energy_mps()
    exact_circ_mps           : CircuitMPS for the exact isometry circuit
    exact_verification_mps   : result of verify_energy_mps() for the
                                exact circuit (MPS-based fidelity)
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
) -> GroundStateContext:
    n = chain.n_sites
    use_ed = n <= EXACT_DIAG_MAX_N
    print(f"\n[context]  chain={chain.name}  N={n}  J1={j1}  J2={j2}  "
          f"bond_dims={bond_dims}  ed_fidelity={'on' if use_ed else 'off (N>%d)' % EXACT_DIAG_MAX_N}")

    H_mpo = build_heisenberg_mpo(chain, j1, j2)

    if use_ed:
        H, basis = build_hamiltonian(chain, j1, j2)
        print(f"[H]  sector dim={len(basis)}")
        e_exact, psi_exact, basis, idx_map = exact_ground_state(chain, j1, j2)
    else:
        H, basis, e_exact, psi_exact = None, None, None, None

    E_dmrg, psi_mps = run_dmrg(chain, j1, j2, bond_dims, H_mpo=H_mpo)
    diag = mps_diagnostics(psi_mps)
    print(f"[DMRG]  E={E_dmrg:.10f}  E/site={E_dmrg/n:.10f}  max_chi={diag['max_chi']}")

    exact_qc = mps_to_exact_circuit(psi_mps, n)
    exact_resources = circuit_resource_report(exact_qc, label="exact", rz_eps=rz_eps)

    exact_verification = (
        verify_energy(exact_qc, H, basis, E_dmrg, psi_exact) if use_ed else None
    )

    exact_circ_mps = circuit_to_mps(exact_qc, max_bond=circuit_mps_max_bond)
    exact_verification_mps = verify_energy_mps(exact_circ_mps, H_mpo, psi_mps, E_dmrg)
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

    Runs BOTH verify_energy_mps() (MPS-based, always) and verify_energy()
    (statevector-based, only when n < EXACT_DIAG_MAX_N -- kept for
    sector-leakage diagnostics). The MPS fidelity is computed against
    ctx.psi_mps (the DMRG MPS), which is the natural target: the circuit
    should reproduce the DMRG output, not the ED state (they differ by at
    most the DMRG truncation error, already tracked via dmrg_energy vs
    e_exact).
    """
    approx_qc = _fit_approx_circuit_with_retries(
        ctx.psi_mps, ctx.n, n_layers, max_retries=max_retries, verbose=verbose
    )
    approx_resources = circuit_resource_report(
        approx_qc, label="approximate", rz_eps=rz_eps, verbose=verbose
    )

    row = {}
    row.update({f"approx_{k}": v for k, v in approx_resources.as_row().items()})

    if ctx.n < EXACT_DIAG_MAX_N:
        approx_verification = verify_energy(
            approx_qc, ctx.H, ctx.basis, ctx.E_dmrg, ctx.psi_exact
        )
        row.update({f"approx_{k}": v for k, v in approx_verification.items()})

    approx_circ_mps = circuit_to_mps(approx_qc, max_bond=circuit_mps_max_bond)
    approx_verification_mps = verify_energy_mps(
        approx_circ_mps, ctx.H_mpo, ctx.psi_mps, ctx.E_dmrg
    )
    # MPS-based metrics stored with _mps suffix to distinguish from the
    # statevector-based ones (they use different reference states: psi_mps
    # for MPS-overlap fidelity, psi_exact for sv fidelity).
    row.update({f"approx_{k}_mps": v for k, v in approx_verification_mps.items()})

    return row


def repeat_approx_circuit_trials(
    ctx: GroundStateContext,
    n_layers: int,
    n_trials: int = DEFAULT_N_TRIALS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Re-fit the approximate circuit to the SAME converged MPS n_trials times
    and return one row per trial. DMRG/ED are not touched here -- ctx
    already holds their results.

    Each row contains both statevector-based metrics (approx_fidelity,
    approx_energy_err_pct, ...) and MPS-based metrics (approx_fidelity_mps,
    approx_energy_err_pct_mps, ...).
    """
    rows = []
    for t in range(n_trials):
        if verbose:
            print(f"\n[trial {t + 1}/{n_trials}]  n_layers={n_layers}")
        row = _approx_trial(
            ctx, n_layers, rz_eps,
            verbose=verbose,
            circuit_mps_max_bond=circuit_mps_max_bond,
        )
        row["trial"] = t
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
) -> tuple[QuantumCircuit, QuantumCircuit, csr_matrix, np.ndarray, dict, dict]:
    """
    Run DMRG on *chain*, compile exact MPS circuit AND approximate, verify
    both energies and fidelities (statevector + MPS), and return everything
    needed for poster plots.

    Returns
    -------
    exact_qc    : Qiskit circuit (exact MPS isometry)
    approx_qc   : Qiskit circuit (brick-wall approx, n_layers)
    H           : sparse Hamiltonian in the n//2-up sector
    basis       : integer basis states for that sector
    exact_info  : dict -- DMRG diagnostics + exact circuit resources + both
                  verification methods (sv-based and MPS-based)
    approx_info : dict -- same base fields + approx circuit resources + both
                  verification methods
    """
    n = chain.n_sites
    ed = n < EXACT_DIAG_MAX_N
    print(f"\n[run]  chain={chain.name}  N={n}  J1={j1}  J2={j2}  "
          f"bond_dims={bond_dims}  n_layers={n_layers}")

    ctx = build_ground_state_context(
        chain, j1, j2, bond_dims, rz_eps,
        circuit_mps_max_bond=circuit_mps_max_bond,
    )

    approx_qc = _fit_approx_circuit_with_retries(ctx.psi_mps, ctx.n, n_layers)
    approx_resources = circuit_resource_report(approx_qc, label="approximate", rz_eps=rz_eps)
    approx_verification = (
        verify_energy(approx_qc, ctx.H, ctx.basis, ctx.E_dmrg, ctx.psi_exact) if ed else None
    )

    approx_circ_mps = circuit_to_mps(approx_qc, max_bond=circuit_mps_max_bond)
    approx_verification_mps = verify_energy_mps(
        approx_circ_mps, ctx.H_mpo, ctx.psi_mps, ctx.E_dmrg
    )

    base = dict(
        lattice=chain.name,
        n_sites=n,
        j1=j1,
        j2=j2,
        j2_over_j1=(j2 / j1) if j1 else float("nan"),
        n_layers=n_layers,
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
        exact_info.update({f"exact_{k}": v for k, v in ctx.exact_verification.items()})
        approx_info.update({f"approx_{k}": v for k, v in approx_verification.items()})

    return ctx.exact_qc, approx_qc, ctx.H, ctx.basis, exact_info, approx_info


# =============================================================================
# N_LAYERS SWEEP  (research-grade: n_trials independent approx-circuit fits
# per n_layers value, full summary statistics)
# =============================================================================
SUMMARY_METRICS = [
    # statevector-based (vs ED ground state)
    "approx_fidelity",
    "approx_energy_err_pct",
    "approx_abs_error",
    "approx_leakage_norm2",
    # MPS-based (vs DMRG MPS) -- preferred for large N
    "approx_fidelity_mps",
    "approx_energy_err_pct_mps",
    "approx_abs_error_mps",
    "approx_circuit_mps_norm_mps",
    # resource counts (same regardless of verification method)
    "approx_cx_count",
    "approx_depth",
    "approx_non_clifford_count",
    "approx_non_clifford_depth",
    "approx_t_count_estimate",
]


def _summarize(df_trials: pd.DataFrame, metrics: list[str]) -> dict:
    """Mean/std/min/max/median for each metric column, ignoring NaNs/None."""
    out = {}
    for m in metrics:
        if m not in df_trials.columns:
            continue
        vals = pd.to_numeric(df_trials[m], errors="coerce").dropna()
        if len(vals) == 0:
            out[f"{m}_mean"] = float("nan")
            out[f"{m}_std"] = float("nan")
            out[f"{m}_min"] = float("nan")
            out[f"{m}_max"] = float("nan")
            out[f"{m}_median"] = float("nan")
            continue
        out[f"{m}_mean"] = float(vals.mean())
        out[f"{m}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{m}_min"] = float(vals.min())
        out[f"{m}_max"] = float(vals.max())
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
    verbose_trials: bool = False,
    return_all_trials: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep over n_layers values. DMRG/ED are solved ONCE for this
    (chain, j1, j2, bond_dims) configuration via build_ground_state_context();
    for each n_layers value, the approximate circuit is independently re-fit
    n_trials times and summarized with mean/std/min/max/median.

    Both statevector-based and MPS-based fidelity/energy metrics are included
    in the summary. The MPS-based fidelity column (approx_fidelity_mps_mean)
    is the one that scales to large N; the statevector-based one
    (approx_fidelity_mean) agrees with it for small N and is kept as a
    cross-check.
    """
    if layers is None:
        layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"\n[sweep_n_layers]  layers={layers}  n_trials={n_trials}  "
          f"(DMRG/ED solved once, approx-circuit re-fit {n_trials}x per layer)")

    ctx = build_ground_state_context(
        chain, j1, j2, bond_dims, rz_eps,
        circuit_mps_max_bond=circuit_mps_max_bond,
    )

    summary_rows = []
    all_trial_dfs = []
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
            n_layers=n_lay,
            n_trials=n_trials,
            lattice=chain.name,
            n_sites=ctx.n,
            j1=j1, j2=j2,
            j2_over_j1=(j2 / j1) if j1 else float("nan"),
            dmrg_energy=ctx.E_dmrg,
            **ctx.diag,
            exact_fidelity_mps=ctx.exact_verification_mps.get("fidelity"),
            exact_energy_err_pct_mps=ctx.exact_verification_mps.get("energy_err_pct"),
            exact_cx_count=ctx.exact_resources.cx_count,
            exact_depth=ctx.exact_resources.depth,
        )
        if ctx.n < EXACT_DIAG_MAX_N:
            summary.update({
                "exact_energy_err_pct": ctx.exact_verification.get("energy_err_pct"),
                "exact_fidelity": ctx.exact_verification.get("fidelity"),
            })
        summary.update(_summarize(df_trials, SUMMARY_METRICS))
        summary_rows.append(summary)

        fmean = summary.get("approx_fidelity_mps_mean", float("nan"))
        fstd = summary.get("approx_fidelity_mps_std", float("nan"))
        emean = summary.get("approx_energy_err_pct_mps_mean", float("nan"))
        estd = summary.get("approx_energy_err_pct_mps_std", float("nan"))
        print(f"  -> MPS fidelity = {fmean:.6f} +/- {fstd:.6f}   "
              f"MPS energy_err% = {emean:.4f} +/- {estd:.4f}   "
              f"(n_trials={n_trials})")

    df_summary = pd.DataFrame(summary_rows)
    print("\n[sweep_n_layers]  summary (mean over n_trials independent fits)")
    print(df_summary.to_string(index=False))

    if return_all_trials:
        df_all = pd.concat(all_trial_dfs, ignore_index=True)
        return df_summary, df_all
    return df_summary


# =============================================================================
# TIMING HARNESS  (per-stage wall-clock timing for scaling studies)
#
# Reuses the pipeline's own build_heisenberg_mpo() (with incremental
# compression), run_dmrg() (per-sweep, on a single persistent DMRG2 object),
# mps_diagnostics(), _fit_approx_circuit_with_retries(), and
# circuit_resource_report() -- no logic is re-derived here, only timed.
#
# Per chain size, this times:
#   1. build_heisenberg_mpo()'s edge-accumulation loop, compressing
#      incrementally every compress_every_n_edges edges (rather than once at
#      the end -- see build_heisenberg_mpo's docstring for why that matters
#      at large N).
#   2. DMRG, timed PER SWEEP. A single qtn.DMRG2 object is solved repeatedly
#      with max_sweeps=1; since solve() never resets its internal state
#      (self.energies, the bond_dims iterator, the cutoffs iterator), this
#      is exactly equivalent to one solve(max_sweeps=N) call, just with a
#      timer between sweeps. (An earlier version of this harness created a
#      fresh DMRG2 object per bond-dim "rung", which overcounted cost ~3x
#      by forcing every rung through redundant cold-start sweeps.)
#   3. mps_diagnostics() -- the entropy/bond-dim scan.
#   4. The approximate (brick-wall) circuit at a given n_layers. The exact
#      isometry circuit is intentionally never built here: its cost grows
#      extremely fast with N (e.g. depth in the millions at N=50) and the
#      harness only needs the approx-circuit cost.
#   5. circuit_resource_report() on that approx circuit.
#
# DMRG_TOL note: quimb's DMRG._check_convergence(tol) is a sweep-to-sweep
# stagnation check (abs(E[-2] - E[-1]) < tol), not a distance-to-truth check
# -- there is no ED reference at these sizes (N > EXACT_DIAG_MAX_N) to
# compare against. After each real run, this harness also reports, for each
# candidate tolerance in dmrg_tol_candidates, which sweep number you would
# have stopped at and the wall-clock time that implies -- computed
# retroactively from the energy history already collected, at zero extra
# DMRG cost.
#
# Each stage is wrapped in its own try/except so one slow/failing stage
# doesn't block timings for the stages before it.
# =============================================================================
class StageTimer:
    """Collects (stage_name, elapsed_seconds) in call order, per chain size."""

    def __init__(self):
        self.records = []

    @contextmanager
    def time(self, stage_name):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - t0
            self.records.append((stage_name, elapsed))
            running_total = sum(t for _, t in self.records)
            print(f"    [{stage_name:<32s}] {elapsed:8.3f}s   "
                  f"(cumulative: {running_total:8.3f}s)")

    def total(self):
        return sum(t for _, t in self.records)


def _time_mpo_build(chain, j1, j2, compress_every_n_edges, timer: StageTimer):
    """Stage 1: incremental MPO build, reusing build_heisenberg_mpo()."""
    try:
        with timer.time("mpo_build_incremental_total"):
            H_mpo = build_heisenberg_mpo(
                chain, j1, j2,
                compress_every_n_edges=compress_every_n_edges,
                verbose=True,
            )
        return H_mpo
    except Exception:
        print("  [build_heisenberg_mpo (incremental) FAILED]")
        traceback.print_exc()
        return None


def _time_dmrg_per_sweep(chain, j1, j2, bond_dims, H_mpo, dmrg_tol_candidates, timer: StageTimer):
    """
    Stage 2: DMRG, timed per real sweep on a single persistent DMRG2 object.
    Returns the converged psi_mps (or None on failure).
    """
    if H_mpo is None:
        print("  [run_dmrg (per-sweep) SKIPPED] H_mpo was not built successfully")
        return None

    psi_mps = None
    sweep_log = []  # (sweep_num, elapsed_seconds, energy, cumulative_seconds)
    try:
        n_sites = chain.n_sites
        psi0 = qtn.MPS_neel_state(n_sites)
        dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dims, cutoffs=DMRG_CUTOFF, p0=psi0)

        prev_energy = None
        converged = False
        sweep_num = 0
        cumulative = 0.0
        while sweep_num < DMRG_MAX_SWEEPS and not converged:
            sweep_num += 1
            stage_name = f"run_dmrg_sweep_{sweep_num:02d}"
            t0 = time.perf_counter()
            with timer.time(stage_name):
                converged = dmrg.solve(tol=DMRG_TOL, max_sweeps=1, verbosity=0)
            sweep_elapsed = time.perf_counter() - t0
            cumulative += sweep_elapsed

            energy = float(np.real(dmrg.energies[-1])) if dmrg.energies else float("nan")
            sweep_log.append((sweep_num, sweep_elapsed, energy, cumulative))
            # current_chi is inferred from the documented bond_dims schedule
            # (sweep k -> bond_dims[k-1], then bond_dims[-1] forever after),
            # matching DMRG._set_bond_dim_seq()'s
            # itertools.chain(bds, itertools.repeat(bds[-1])).
            current_chi = bond_dims[min(sweep_num - 1, len(bond_dims) - 1)]
            delta = "" if prev_energy is None else f"  dE={energy - prev_energy:+.2e}"
            print(f"    -> sweep={sweep_num:<3d} chi_target={current_chi:<4d} "
                  f"E={energy:.10f}{delta}  converged={converged}")
            prev_energy = energy

        psi_mps = dmrg.state
        print(f"    -> DMRG finished after {sweep_num} real sweep(s), "
              f"converged={converged}  (actual DMRG_TOL={DMRG_TOL})")

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

    return psi_mps


def time_one_chain_size(
    n_sites: int,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    compress_every_n_edges: int = TIMING_COMPRESS_EVERY_N_EDGES,
    n_layers: int = TIMING_N_LAYERS,
    dmrg_tol_candidates: list[float] = TIMING_DMRG_TOL_CANDIDATES,
    j1: float = J1,
    j2: float = J2,
) -> StageTimer:
    """
    Time every pipeline stage for a single chain size, reusing the pipeline's
    own build_heisenberg_mpo / run_dmrg / mps_diagnostics /
    _fit_approx_circuit_with_retries / circuit_resource_report functions.
    """
    print(f"\n{'='*70}")
    print(f"  N = {n_sites}   bond_dims = {bond_dims}")
    print(f"{'='*70}")

    timer = StageTimer()
    chain = make_chain(n_sites)

    use_ed = n_sites <= EXACT_DIAG_MAX_N
    print(f"  EXACT_DIAG_MAX_N={EXACT_DIAG_MAX_N}  -> "
          f"ED {'WILL run (small N)' if use_ed else 'will be skipped'} at this size")

    # Stage 1: incremental MPO build.
    H_mpo = _time_mpo_build(chain, j1, j2, compress_every_n_edges, timer)

    # Stage 2: DMRG, per-sweep timing on a single persistent object.
    psi_mps = _time_dmrg_per_sweep(chain, j1, j2, bond_dims, H_mpo, dmrg_tol_candidates, timer)

    # Stage 3: MPS diagnostics.
    if psi_mps is not None:
        try:
            with timer.time("mps_diagnostics"):
                diag = mps_diagnostics(psi_mps)
            print(f"    -> max_chi={diag['max_chi']}  "
                  f"half_chain_entropy={diag['half_chain_entropy']:.4f}")
        except Exception:
            print("  [mps_diagnostics FAILED]")
            traceback.print_exc()

    # Stage 4: approximate (brick-wall) circuit at n_layers, with the same
    # stochastic retry behavior as the pipeline's _approx_trial().
    approx_qc = None
    if psi_mps is not None:
        try:
            with timer.time(f"mps_to_approx_circuit_nlayers_{n_layers}"):
                approx_qc = _fit_approx_circuit_with_retries(
                    psi_mps, n_sites, n_layers, verbose=True
                )
            print(f"    -> approx circuit (n_layers={n_layers}): "
                  f"{approx_qc.num_qubits} qubits, "
                  f"{sum(approx_qc.count_ops().values())} raw gates")
        except Exception:
            print(f"  [mps_to_approx_circuit FAILED] at n_layers={n_layers}")
            traceback.print_exc()

    # Stage 5: resource report on the approx circuit.
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


def time_chain_sizes(
    chain_sizes: list[int] = TIMING_CHAIN_SIZES,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    compress_every_n_edges: int = TIMING_COMPRESS_EVERY_N_EDGES,
    n_layers: int = TIMING_N_LAYERS,
    dmrg_tol_candidates: list[float] = TIMING_DMRG_TOL_CANDIDATES,
    j1: float = J1,
    j2: float = J2,
) -> dict[int, StageTimer]:
    """
    Run time_one_chain_size() for each size in chain_sizes and print a
    final cross-size summary table (seconds per stage, by chain size).

    Returns a dict {n_sites: StageTimer} for further inspection/plotting.
    """
    all_timers: dict[int, StageTimer] = {}
    for n_sites in chain_sizes:
        all_timers[n_sites] = time_one_chain_size(
            n_sites, bond_dims, compress_every_n_edges, n_layers, dmrg_tol_candidates,
            j1=j1, j2=j2,
        )

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
            d = dict(all_timers[n_sites].records)
            val = d.get(stage)
            row += f"{val:>12.3f}" if val is not None else f"{'--':>12s}"
        print(row)
    print("-" * len(header))
    totals_row = f"{'TOTAL':<32s}"
    for n_sites in chain_sizes:
        totals_row += f"{all_timers[n_sites].total():>12.3f}"
    print(totals_row)

    return all_timers








import time

chain = make_chain(20)   # your chain size
n_layers = 3              # your brick-wall depth

# One-time DMRG solve (expensive part, reuse ctx across trials/n_layers)
ctx = build_ground_state_context(chain)

t0 = time.perf_counter()
row = _approx_trial(ctx, n_layers=n_layers, rz_eps=1e-3, verbose=True)
elapsed = time.perf_counter() - t0

print(f"\nwall time: {elapsed:.3f}s")
print(f"cx_count: {row['approx_cx_count']}")
print(f"depth: {row['approx_depth']}")
print(f"non_clifford_count (rz): {row['approx_non_clifford_count']}")
print(f"t_count_estimate: {row['approx_t_count_estimate']}")
print(f"circuit energy (CircuitMPS): {row['approx_circuit_energy_mps']}")
print(f"dmrg energy: {row['approx_dmrg_energy_mps']}")
print(f"energy_err_pct (MPS): {row['approx_energy_err_pct_mps']}")
print(f"fidelity |<dmrg|circuit>|^2 (MPS overlap): {row['approx_fidelity_mps']}")
