"""
Returns
-------
  run() -> (exact_qc, H_sparse, basis, info)

Dependencies
------------
    pip install physics-tenpy qiskit scipy numpy
    pip install git+https://github.com/qiskit-community/mps-to-circuit.git
"""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field


import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from qiskit import transpile
from qiskit.quantum_info import Statevector

# -- TeNPy --------------------------------------------------------------------
try:
    import tenpy
    from tenpy.networks.site import SpinHalfSite
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    print(f"[TeNPy]   {tenpy.__version__}")
except ImportError:
    raise ImportError("pip install physics-tenpy")

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

from lattices import BaseLattice

# =============================================================================
# CONFIG
# =============================================================================
J1            = 1.0
J2            = 0.0
DMRG_CHI_MAX  = 256
DMRG_MIXER    = True

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
# HAMILTONIAN  (sparse, fixed-Sz sector)
# Used for energy checks and basis construction; topology from BaseLattice.
# =============================================================================
def build_basis(n: int) -> np.ndarray:
    n_up = n // 2
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up],
                    dtype=np.int64)


def build_hamiltonian(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
) -> tuple[csr_matrix, np.ndarray]:
    """
    Return (H_sparse, basis) in the n//2-up sector.

    Edge topology is taken from lattice.nn_edges (J1) and
    lattice.nnn_edges (J2); no assumption about 1D chain structure.

    Bit convention: bit `i` of a basis integer is the Sz state of site i,
    matching the qubit ordering produced by mps_to_circuit (qubit i <->
    site i, with little-endian Qiskit statevector indexing). This is what
    lets verify_energy() compare a circuit statevector directly against
    this Hamiltonian without any index remapping.
    """
    n       = lattice.n_sites
    basis   = build_basis(n)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    H       = lil_matrix((len(basis), len(basis)), dtype=np.float64)

    edge_sets = [(lattice.nn_edges, j1)]
    if j2 and lattice.nnn_edges:
        edge_sets.append((lattice.nnn_edges, j2))

    for edges, j in edge_sets:
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

    return csr_matrix(H), basis
def _build_basis(n: int) -> np.ndarray:
    n_up = n // 2
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up],
                    dtype=np.int64) 
def exact_ground_state(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
) -> tuple[float, np.ndarray, np.ndarray, dict]:
    """
    Lanczos ground state in the Sz=0 sector via scipy eigsh.

    Returns (e_exact, psi_exact, basis, idx_map).
    Used by UCJ to compute |ΔE| and as the universal reference energy.
    """
    n       = lattice.n_sites
    n_up    = n // 2
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

    op             = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    evals, evecs   = eigsh(op, k=1, which="SA", tol=1e-10, maxiter=10_000)
    e_exact        = float(evals[0])
    psi_exact      = evecs[:, 0] / np.linalg.norm(evecs[:, 0])

    print(f"[ED]  E_exact={e_exact:.10f}  E/site={e_exact/n:.10f}")
    return e_exact, psi_exact, basis, idx_map
# =============================================================================
# TENPY MODEL  -- topology-aware
# =============================================================================
class HeisenbergLattice(CouplingMPOModel):
    """
    Heisenberg J1-J2 model on an arbitrary geometry.

    model_params must contain:
        tenpy_lat  : the raw TeNPy lattice object  (from BaseLattice.tenpy_lat)
        nn_edges   : list[tuple[int,int]]  -- nearest-neighbour pairs
        nnn_edges  : list[tuple[int,int]]  -- next-nearest-neighbour pairs
        J1, J2     : float coupling constants
        conserve   : 'Sz' (default)
        bc_MPS     : 'finite' (default)

    Couplings are added site-by-site from the explicit edge lists, so
    any lattice geometry is handled correctly without relying on dx offsets.
    """

    def init_sites(self, model_params):
        return SpinHalfSite(conserve=model_params.get('conserve', 'Sz'))

    def init_lattice(self, model_params):
        # Reuse the pre-built TeNPy lattice so MPS ordering is consistent
        # with the one embedded in BaseLattice.
        return model_params['tenpy_lat']

    def init_terms(self, model_params):
        j1        = model_params.get('J1', 1.0)
        j2        = model_params.get('J2', 0.0)
        nn_edges  = model_params['nn_edges']
        nnn_edges = model_params.get('nnn_edges', [])

        for i, j in nn_edges:
            self._add_heisenberg(i, j, j1)

        if j2:
            for i, j in nnn_edges:
                self._add_heisenberg(i, j, j2)

    # ------------------------------------------------------------------
    def _add_heisenberg(self, i: int, j: int, coupling: float) -> None:
        """
        Add S_i . S_j = Sz_i Sz_j + 1/2(S+_i S-_j + S-_i S+_j).

        add_coupling_term(strength, i, op_i, j, op_j) accepts flat MPS
        site indices directly -- no lattice-coordinate conversion needed.
        This is the correct API when i, j come from our flat edge lists.
        add_local_term / add_coupling both go through lat2mps_idx which
        expects (x, y, ..., u) tuples and raises IndexError on plain ints.
        """
        # add_coupling_term requires i < j
        if i > j:
            i, j = j, i
        self.add_coupling_term(coupling,       i, j, 'Sz', 'Sz')
        self.add_coupling_term(coupling / 2.0, i, j, 'Sp', 'Sm')
        self.add_coupling_term(coupling / 2.0, i, j, 'Sm', 'Sp')


# =============================================================================
# DMRG
# =============================================================================
def run_dmrg(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
    chi_max: int = DMRG_CHI_MAX,
) -> tuple[float, MPS]:
    """
    Run DMRG on the lattice topology defined by *lattice*.

    Parameters
    ----------
    lattice : BaseLattice
        Provides n_sites, nn_edges, nnn_edges, and the raw TeNPy lattice
        object (.tenpy_lat) so that both the Hamiltonian and the MPS bond
        structure reflect the true geometry.
    j1, j2  : Heisenberg couplings
    chi_max : maximum MPS bond dimension

    Returns
    -------
    E     : DMRG ground-state energy
    psi   : converged MPS
    """
    n          = lattice.n_sites
    tenpy_lat  = lattice.tenpy_lat   # raw TeNPy object with MPS ordering

    model_params = dict(
        tenpy_lat  = tenpy_lat,
        nn_edges   = lattice.nn_edges,
        nnn_edges  = lattice.nnn_edges,
        J1         = j1,
        J2         = j2,
        conserve   = 'Sz',
        bc_MPS     = 'finite',
    )
    model = HeisenbergLattice(model_params)

    # Neel initial state along the MPS chain
    initial_state = ['up' if i % 2 == 0 else 'down' for i in range(n)]
    psi = MPS.from_product_state(
        model.lat.mps_sites(), initial_state, bc='finite')

    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'N_sweeps_check': 1,
        'mixer': DMRG_MIXER,
        'mixer_params': {'amplitude': 1e-4, 'decay': 1.2, 'disable_after': 50},
    })
    E, psi = eng.run()
    #print(f"[DMRG]  lattice={lattice.name}  E={E:.10f}  "
    #      f"E/site={E/n:.10f}  chi_max={max(psi.chi)}")
    return E, psi


def mps_diagnostics(psi: MPS) -> dict:
    """Bond-dimension and entanglement diagnostics for the converged MPS.

    max_chi / mean_chi describe the compression cost of the state; the
    half-chain entanglement entropy is the standard order parameter for
    tracking how close the chain is to a critical point (entropy grows
    with system size at criticality, saturates away from it).
    """
    chi = list(psi.chi)
    ee  = psi.entanglement_entropy()
    n   = psi.L
    half = max(0, n // 2 - 1)
    return dict(
        max_chi=int(max(chi)) if chi else 0,
        mean_chi=float(np.mean(chi)) if chi else 0.0,
        half_chain_entropy=float(ee[half]) if len(ee) else float('nan'),
        max_entropy=float(np.max(ee)) if len(ee) else float('nan'),
    )


# =============================================================================
# MPS -> CIRCUIT  (exact isometry method only)
# =============================================================================
def mps_to_exact_circuit(psi_mps: MPS, n: int) -> QuantumCircuit:
    """
    Build the EXACT MPS-isometry circuit (Lin, PRX Quantum 2, 010342 (2021)).

    We deliberately do not offer the brick-wall "approximate" method here:
    for an early-fault-tolerant resource study we want the true cost of
    preparing the state DMRG actually found, not a depth-truncated
    heuristic. mps_to_circuit expects tensors in 'lpr' shape
    (left-bond, physical, right-bond).
    """
    mps_arrays = [psi_mps.get_B(i).to_ndarray() for i in range(n)]
    qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")
    return qc


def mps_to_approx_circuit(psi_mps: MPS, n: int, n_layers: int) -> QuantumCircuit:
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
    mps_arrays = [psi_mps.get_B(i).to_ndarray() for i in range(n)]
    qc = mps_to_circuit(mps_arrays, method="approximate", shape="lpr", num_layers=n_layers)
    return qc


# =============================================================================
# ENERGY CLOSURE CHECK
# =============================================================================
def verify_energy(
    qc: QuantumCircuit,
    H: csr_matrix,
    basis: np.ndarray,
    dmrg_energy: float,
    ref_sector: np.ndarray | None = None,   # MPS-contracted ground state in sector
) -> dict:
    sv = Statevector.from_instruction(qc).data
    psi_sector = sv[basis]
    sector_norm2 = float(np.vdot(psi_sector, psi_sector).real)
    leak_norm2 = 1.0 - sector_norm2

    if sector_norm2 < 1e-12:
        raise RuntimeError("~zero amplitude in target Sz sector")

    psi_sector = psi_sector / np.sqrt(sector_norm2)
    energy = complex(psi_sector.conj() @ (H @ psi_sector)).real
    energy_err_pct = abs(energy - dmrg_energy) / abs(dmrg_energy) * 100

    # fidelity vs DMRG MPS (true ground state), not vs exact circuit
    fidelity = float(abs(np.vdot(ref_sector, psi_sector))**2) if ref_sector is not None else None

    #print(
    #    f"\n[verify]  E={energy:.10f}  E_DMRG={dmrg_energy:.10f}  "
    #    f"err%={energy_err_pct:.4f}  leakage={leak_norm2:.2e}"
    #    + (f"  fidelity_vs_mps={fidelity:.6f}" if fidelity is not None else "")
    #)
    return dict(
        circuit_energy    = energy,
        dmrg_energy       = dmrg_energy,
        abs_error         = abs(energy - dmrg_energy),
        energy_err_pct    = energy_err_pct,
        sector_norm2      = sector_norm2,
        leakage_norm2     = leak_norm2,
        fidelity          = fidelity,   # |<ψ_mps|ψ_circuit>|² — vs true ground state
    )


# =============================================================================
# SHARED-STATE CONTAINER  (one DMRG/ED solve, reused across many approx fits)
# =============================================================================
@dataclass
class GroundStateContext:
    """
    Everything that is expensive and DETERMINISTIC for a given
    (lattice, j1, j2, chi_max): the Hamiltonian, the ED reference state,
    the converged DMRG MPS, and the exact isometry circuit + its resources.

    Building this once and reusing it across many approximate-circuit
    trials is what makes sweep_n_layers() cheap: only the stochastic
    brick-wall fit (mps_to_approx_circuit) is repeated per trial.
    """
    lattice: BaseLattice
    j1: float
    j2: float
    chi_max: int
    n: int
    H: csr_matrix
    basis: np.ndarray
    e_exact: float
    psi_exact: np.ndarray
    E_dmrg: float
    psi_mps: MPS
    diag: dict
    exact_qc: QuantumCircuit
    exact_resources: ResourceReport
    exact_verification: dict


def build_ground_state_context(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
    chi_max: int = DMRG_CHI_MAX,
    rz_eps: float = RZ_SYNTHESIS_EPS,
) -> GroundStateContext:
    """
    Run the expensive, deterministic part of the pipeline EXACTLY ONCE:
    build H, solve ED, run DMRG, compile + verify the exact circuit.

    Call this once per (lattice, j1, j2, chi_max) configuration, then pass
    the returned context into repeat_approx_circuit_trials() / run_from_context()
    as many times as you like without re-solving DMRG/ED.
    """
    n = lattice.n_sites
    print(f"\n[context]  lattice={lattice.name}  N={n}  J1={j1}  J2={j2}  "
          f"chi_max={chi_max}")

    H, basis = build_hamiltonian(lattice, j1, j2)
    print(f"[H]  sector dim={len(basis)}")

    e_exact, psi_exact, basis, idx_map = exact_ground_state(lattice, j1, j2)

    E_dmrg, psi_mps = run_dmrg(lattice, j1, j2, chi_max)
    diag = mps_diagnostics(psi_mps)

    exact_qc = mps_to_exact_circuit(psi_mps, n)
    exact_resources = circuit_resource_report(exact_qc, label="exact", rz_eps=rz_eps)
    exact_verification = verify_energy(exact_qc, H, basis, E_dmrg, psi_exact)

    return GroundStateContext(
        lattice=lattice, j1=j1, j2=j2, chi_max=chi_max, n=n,
        H=H, basis=basis,
        e_exact=e_exact, psi_exact=psi_exact,
        E_dmrg=E_dmrg, psi_mps=psi_mps, diag=diag,
        exact_qc=exact_qc,
        exact_resources=exact_resources,
        exact_verification=exact_verification,
    )


from qiskit.exceptions import QiskitError

def _approx_trial(
    ctx: GroundStateContext,
    n_layers: int,
    rz_eps: float,
    verbose: bool,
    max_retries: int = 5,
) -> dict:
    """
    One stochastic re-fit of the approximate circuit + its verification.

    mps_to_approx_circuit's brick-wall optimizer is randomly initialized
    (see its docstring), and occasionally converges to a two-qubit block
    that Qiskit's TwoQubitWeylDecomposition fails to diagonalize
    (QiskitError, a known upstream edge case -- qiskit-terra#4159). Since
    the fit is already stochastic, the simplest fix is to just throw the
    bad draw away and re-fit: re-running mps_to_approx_circuit gives an
    independent random initialization, which either won't hit the
    degenerate case or will draw a different one. We don't touch the
    circuit's gates/matrices ourselves -- we only decide whether to keep
    or discard a given stochastic draw.
    """
    last_err = None
    for attempt in range(max_retries):
        try:
            approx_qc = mps_to_approx_circuit(ctx.psi_mps, ctx.n, n_layers)
            approx_resources = circuit_resource_report(
                approx_qc, label="approximate", rz_eps=rz_eps, verbose=verbose
            )
            approx_verification = verify_energy(
                approx_qc, ctx.H, ctx.basis, ctx.E_dmrg, ctx.psi_exact
            )
            row = {}
            row.update({f"approx_{k}": v for k, v in approx_resources.as_row().items()})
            row.update({f"approx_{k}": v for k, v in approx_verification.items()})
            row["n_retries"] = attempt
            return row
        except QiskitError as e:
            last_err = e
            if verbose:
                print(f"  [warn] transpile/synthesis failure on attempt "
                      f"{attempt+1}/{max_retries} (n_layers={n_layers}); "
                      f"re-drawing stochastic fit -- {e}")

    raise RuntimeError(
        f"_approx_trial: {max_retries} consecutive synthesis failures at "
        f"n_layers={n_layers}. This many failures in a row suggests "
        f"something structural (e.g. n_layers too small/large for n, or a "
        f"version mismatch), not just bad luck in the random init."
    ) from last_err


def repeat_approx_circuit_trials(
    ctx: GroundStateContext,
    n_layers: int,
    n_trials: int = DEFAULT_N_TRIALS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Re-fit the approximate circuit to the SAME converged MPS n_trials times
    and return one row per trial. DMRG/ED are not touched here -- ctx
    already holds their results.
    """
    rows = []
    for t in range(n_trials):
        if verbose:
            print(f"\n[trial {t+1}/{n_trials}]  n_layers={n_layers}")
        row = _approx_trial(ctx, n_layers, rz_eps, verbose=verbose)
        row["trial"] = t
        row["n_layers"] = n_layers
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# SINGLE-POINT ENTRY POINT
# =============================================================================
def run(lattice: BaseLattice,
j1: float = J1,
j2: float = J2,
n_layers: int = 3,
chi_max: int = DMRG_CHI_MAX,
rz_eps: float = RZ_SYNTHESIS_EPS,) -> tuple[QuantumCircuit, QuantumCircuit, csr_matrix, np.ndarray, dict, dict]:
    """
    Run DMRG on *lattice*, compile exact MPS circuit AND approximate, verify
    both energies, and return everything needed for poster plots.

    This is a thin convenience wrapper around build_ground_state_context()
    + a single approximate-circuit trial. For repeated/averaged approximate
    circuits at fixed n_layers, prefer build_ground_state_context() once
    followed by repeat_approx_circuit_trials() -- that avoids re-running
    DMRG/ED on every call.

    Returns
    -------
    exact_qc    : Qiskit circuit (exact MPS isometry)
    approx_qc   : Qiskit circuit (brick-wall approx, n_layers)
    H           : sparse Hamiltonian in the n//2-up sector
    basis       : integer basis states for that sector
    exact_info  : dict — DMRG diagnostics + exact circuit resources + energy check
    approx_info : dict — same base fields + approx circuit resources + energy check
    """
    n = lattice.n_sites
    print(f"\n[run]  lattice={lattice.name}  N={n}  J1={j1}  J2={j2}  "
          f"chi_max={chi_max}  n_layers={n_layers}")

    ctx = build_ground_state_context(lattice, j1, j2, chi_max, rz_eps)

    approx_qc = mps_to_approx_circuit(ctx.psi_mps, ctx.n, n_layers)
    approx_resources = circuit_resource_report(approx_qc, label="approximate", rz_eps=rz_eps)
    approx_verification = verify_energy(approx_qc, ctx.H, ctx.basis, ctx.E_dmrg, ctx.psi_exact)

    # ── shared base fields ────────────────────────────────────────────────────
    base = dict(
            lattice        = lattice.name,
            n_sites        = n,
            j1             = j1,
            j2             = j2,
            j2_over_j1     = (j2 / j1) if j1 else float("nan"),
            chi_max_setting = chi_max,
            n_layers       = n_layers,
            dmrg_energy    = ctx.E_dmrg,
            **ctx.diag,
            )

    # ── per-circuit dicts ─────────────────────────────────────────────────────
    exact_info = {
        **base,
        **{f"exact_{k}": v for k, v in ctx.exact_resources.as_row().items()},
        **{f"exact_{k}": v for k, v in ctx.exact_verification.items()},
    }
    approx_info = {
        **base,
        **{f"approx_{k}": v for k, v in approx_resources.as_row().items()},
        **{f"approx_{k}": v for k, v in approx_verification.items()},
    }

    return ctx.exact_qc, approx_qc, ctx.H, ctx.basis, exact_info, approx_info


# =============================================================================
# N_LAYERS SWEEP  (research-grade: n_trials independent approx-circuit fits
# per n_layers value, full summary statistics)
# =============================================================================

# Metrics that get full (mean/std/min/max/median) summary statistics in
# sweep_n_layers(). Add/remove keys here to change what gets summarized;
# they must match the column names produced by _approx_trial()'s row dict.
SUMMARY_METRICS = [
    "approx_fidelity",
    "approx_energy_err_pct",
    "approx_abs_error",
    "approx_cx_count",
    "approx_depth",
    "approx_non_clifford_count",
    "approx_non_clifford_depth",
    "approx_t_count_estimate",
    "approx_leakage_norm2",
]


def _summarize(df_trials: pd.DataFrame, metrics: list[str]) -> dict:
    """Mean/std/min/max/median for each metric column, ignoring NaNs/None."""
    out = {}
    for m in metrics:
        if m not in df_trials.columns:
            continue
        vals = pd.to_numeric(df_trials[m], errors="coerce").dropna()
        if len(vals) == 0:
            out[f"{m}_mean"]   = float("nan")
            out[f"{m}_std"]    = float("nan")
            out[f"{m}_min"]    = float("nan")
            out[f"{m}_max"]    = float("nan")
            out[f"{m}_median"] = float("nan")
            continue
        out[f"{m}_mean"]   = float(vals.mean())
        out[f"{m}_std"]    = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        out[f"{m}_min"]    = float(vals.min())
        out[f"{m}_max"]    = float(vals.max())
        out[f"{m}_median"] = float(vals.median())
    return out


def sweep_n_layers(
            lattice: BaseLattice,
            j1: float = J1,
            j2: float = J2,
            chi_max: int = DMRG_CHI_MAX,
            rz_eps: float = RZ_SYNTHESIS_EPS,
            layers: list[int] | None = None,
            n_trials: int = DEFAULT_N_TRIALS,
            verbose_trials: bool = False,
            return_all_trials: bool = False,
            ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sweep over n_layers values. DMRG/ED are solved ONCE for this
    (lattice, j1, j2, chi_max) configuration via build_ground_state_context();
    for each n_layers value, the approximate circuit is independently re-fit
    n_trials times (this is the only stochastic step) and summarized with
    mean/std/min/max/median.

    Parameters
    ----------
    layers             : list of n_layers values to sweep (default 1..10-ish)
    n_trials           : number of independent approx-circuit re-fits per
                          n_layers value (default DEFAULT_N_TRIALS = 20)
    verbose_trials      : if True, print full resource/verify output for
                          every single trial (very chatty for n_trials=20+)
    return_all_trials   : if True, also return the raw per-trial DataFrame
                          (one row per (n_layers, trial)) for custom analysis
                          or violin/box plots, in addition to the summary.

    Returns
    -------
    df_summary  : pd.DataFrame, one row per n_layers value, with
                  <metric>_mean / _std / _min / _max / _median columns
                  for every metric in SUMMARY_METRICS, plus n_trials and
                  the exact-circuit reference values (constant across rows
                  since DMRG/ED/exact-circuit were solved once).
    df_all      : (only if return_all_trials=True) pd.DataFrame with one
                  row per individual trial, columns = n_layers, trial, and
                  all raw approx_* metrics. Useful for box/violin plots
                  showing the full distribution, not just summary stats.
    """
    if layers is None:
        layers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"\n[sweep_n_layers]  layers={layers}  n_trials={n_trials}  "
          f"(DMRG/ED solved once, approx-circuit re-fit {n_trials}x per layer)")

    ctx = build_ground_state_context(lattice, j1, j2, chi_max, rz_eps)

    summary_rows = []
    all_trial_dfs = []
    for n_lay in layers:
        print(f"\n[sweep_n_layers]  n_layers={n_lay}  "
              f"running {n_trials} independent approx-circuit fits...")
        df_trials = repeat_approx_circuit_trials(
            ctx, n_lay, n_trials=n_trials, rz_eps=rz_eps, verbose=verbose_trials
        )
        all_trial_dfs.append(df_trials)

        summary = dict(
            n_layers=n_lay,
            n_trials=n_trials,
            lattice=lattice.name,
            n_sites=ctx.n,
            j1=j1, j2=j2,
            j2_over_j1=(j2 / j1) if j1 else float("nan"),
            chi_max_setting=chi_max,
            dmrg_energy=ctx.E_dmrg,
            **ctx.diag,
            exact_fidelity=ctx.exact_verification.get("fidelity"),
            exact_energy_err_pct=ctx.exact_verification.get("energy_err_pct"),
            exact_cx_count=ctx.exact_resources.cx_count,
            exact_depth=ctx.exact_resources.depth,
        )
        summary.update(_summarize(df_trials, SUMMARY_METRICS))
        summary_rows.append(summary)

        # quick console readout per layer count
        fmean = summary.get("approx_fidelity_mean", float("nan"))
        fstd  = summary.get("approx_fidelity_std", float("nan"))
        emean = summary.get("approx_energy_err_pct_mean", float("nan"))
        estd  = summary.get("approx_energy_err_pct_std", float("nan"))
        print(f"  -> fidelity = {fmean:.6f} ± {fstd:.6f}   "
              f"energy_err% = {emean:.4f} ± {estd:.4f}   "
              f"(n_trials={n_trials})")

    df_summary = pd.DataFrame(summary_rows)
    print("\n[sweep_n_layers]  summary (mean over n_trials independent fits)")
    print(df_summary.to_string(index=False))

    if return_all_trials:
        df_all = pd.concat(all_trial_dfs, ignore_index=True)
        return df_summary, df_all
    return df_summary


if __name__ == '__main__':
    from lattices import make_lattice

    def chain(L):
        return make_lattice('chain', L=L)
    lattice = chain(8)

    # --- single point, sanity check -----------------------------------------
    exact_qc, approx_qc, H, basis, exact_info, approx_info = run(
        chain(8), j1=J1, j2=0.0, chi_max=DMRG_CHI_MAX
    )

    # --- research-grade sweep: 20 independent approx-circuit fits per
    #     n_layers value, DMRG/ED solved once ---------------------------------
    df_summary, df_all = sweep_n_layers(
        lattice,
        layers=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        n_trials=DEFAULT_N_TRIALS,
        return_all_trials=True,
    )

    # save both the summary (for the poster table/plot) and the raw
    # per-trial data (for error bars / box-and-whisker if you want them)
    df_summary.to_csv("sweep_n_layers_summary.csv", index=False)
    df_all.to_csv("sweep_n_layers_all_trials.csv", index=False)

    # error-bar plot: mean fidelity vs n_layers with std as error bars
    ax = df_summary.plot(
        x="n_layers", y="approx_fidelity_mean",
        yerr="approx_fidelity_std", marker="o", capsize=4,
        title="Approx-circuit fidelity vs n_layers (mean ± std, n=%d trials)" % DEFAULT_N_TRIALS,
    )

    df_summary.plot(
        x="n_layers", y="approx_energy_err_pct_mean",
        yerr="approx_energy_err_pct_std", marker=".",
        title="Energy error %% vs n_layers (mean ± std)",
    )

    # cost vs quality, mean over trials
    df_summary.plot(
        x="approx_fidelity_mean", y="approx_cx_count_mean", marker="o",
        title="Resource cost vs quality (means over %d trials)" % DEFAULT_N_TRIALS,
    )
