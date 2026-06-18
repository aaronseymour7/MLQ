"""
====================
DMRG ground state via TeNPy -> Qiskit circuit + Hamiltonian + basis,
with resource accounting aimed at EARLY FAULT-TOLERANT hardware.

The TeNPy model is driven entirely by a BaseLattice object so that
arbitrary geometries (chain, square, triangular, honeycomb, kagome, ...)
are handled correctly. The coupling topology comes from
  lattice.nn_edges   -> J1 terms
  lattice.nnn_edges  -> J2 terms
and the TeNPy lattice object (lattice.tenpy_lat) is used directly so
that the MPS bond structure respects the canonical MPS ordering that
TeNPy chose when it built the lattice.

This version:
  * drops the brick-wall "approximate" MPS-to-circuit method entirely --
    we only ever compile the EXACT MPS isometry circuit. For an EFT
    poster the interesting question is "what does it cost to prepare
    the state exactly", not "how good is a depth-truncated heuristic".
  * reports resources in EFT-relevant terms: Clifford (CX, H, S, Sdg)
    vs. non-Clifford (RZ) counts/depth, plus an estimated T-count from
    the RZ count via the standard Clifford+T synthesis bound, since RZ
    angles are exactly the gates that need to be synthesized out of
    Clifford+T on a fault-tolerant device.
  * closes the loop: samples the statevector of the compiled circuit
    and computes <H> directly, compared against the DMRG energy, so
    "we compiled a state" becomes "we prepared a state and verified it
    reproduces the target energy".
  * provides sweep_j2_ratio() and sweep_system_size() to generate the
    two headline poster plots: resource cost / entanglement vs J2/J1
    (crossing the gapless -> dimerized transition), and resource cost /
    energy-error vs system size at fixed bond dimension.

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
    print(f"[DMRG]  lattice={lattice.name}  E={E:.10f}  "
          f"E/site={E/n:.10f}  chi_max={max(psi.chi)}")
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
    """
    mps_arrays = [psi_mps.get_B(i).to_ndarray() for i in range(n)]
    qc = mps_to_circuit(mps_arrays, method="approximate", shape="lpr", num_layers=n_layers)
    return qc

def mps_to_statevector(psi_mps: MPS, n: int, basis: np.ndarray) -> np.ndarray:
    """
    Contract the DMRG MPS into a full statevector (in the fixed-Sz sector).
    This is the true reference state — not the exact circuit, which has its
    own compilation error from the isometry decomposition.
    """
    full_sv = np.zeros(2**n, dtype=complex)
    # TeNPy: get the dense statevector directly
    psi_dense = psi_mps.get_theta(0, n)             # MPS contraction
    # reshape to 2**n vector (TeNPy returns a LegCharge tensor)
    full_sv = psi_mps.to_dense()                     # or equivalent for your TeNPy version
    # project to Sz sector (same convention as build_hamiltonian)
    psi_sector = full_sv[basis]
    norm = np.linalg.norm(psi_sector)
    return psi_sector / norm
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

    print(
        f"\n[verify]  E={energy:.10f}  E_DMRG={dmrg_energy:.10f}  "
        f"err%={energy_err_pct:.4f}  leakage={leak_norm2:.2e}"
        + (f"  fidelity_vs_mps={fidelity:.6f}" if fidelity is not None else "")
    )
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
# SINGLE-POINT ENTRY POINT
# =============================================================================
def run(lattice: BaseLattice,
j1: float = J1,
j2: float = J2,
n_layers: int = 3,
# fixed: was missing comma
chi_max: int = DMRG_CHI_MAX,
rz_eps: float = RZ_SYNTHESIS_EPS,) -> tuple[QuantumCircuit, QuantumCircuit, csr_matrix, np.ndarray, dict, dict]:
    """
    Run DMRG on *lattice*, compile exact MPS circuit AND approximate, verify
    both energies, and return everything needed for poster plots.

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

    H, basis = build_hamiltonian(lattice, j1, j2)
    print(f"[H]  sector dim={len(basis)}")

    E_dmrg, psi_mps = run_dmrg(lattice, j1, j2, chi_max)
    diag = mps_diagnostics(psi_mps)

    exact_qc  = mps_to_exact_circuit(psi_mps, n)
    approx_qc = mps_to_approx_circuit(psi_mps, n, n_layers)
    
    exact_resources  = circuit_resource_report(exact_qc,  label="exact",       rz_eps=rz_eps)
    approx_resources = circuit_resource_report(approx_qc, label="approximate", rz_eps=rz_eps)


    # Contract MPS → sector statevector (true reference, not the compiled circuit)
    mps_ref_sector = mps_to_statevector(psi_mps, n, basis)

    exact_verification  = verify_energy(exact_qc,  H, basis, E_dmrg, ref_sector=mps_ref_sector)
    approx_verification = verify_energy(approx_qc, H, basis, E_dmrg, ref_sector=mps_ref_sector)


    # ── shared base fields ────────────────────────────────────────────────────
    base = dict(
            lattice        = lattice.name,
            n_sites        = n,
            j1             = j1,
            j2             = j2,
            j2_over_j1     = (j2 / j1) if j1 else float("nan"),
            chi_max_setting = chi_max,
            n_layers       = n_layers,
            dmrg_energy    = E_dmrg,
            **diag,
            )

    # ── per-circuit dicts ─────────────────────────────────────────────────────
    # If as_row() doesn't accept a prefix, replace with:
    # **{f"exact_{k}": v for k, v in exact_resources.as_row().items()}
    exact_info = {
        **base,
        **{f"exact_{k}": v for k, v in exact_resources.as_row().items()},
        **{f"exact_{k}": v for k, v in exact_verification.items()},
    }
    approx_info = {
        **base,
        **{f"approx_{k}": v for k, v in approx_resources.as_row().items()},
        **{f"approx_{k}": v for k, v in approx_verification.items()},
    }

    return exact_qc, approx_qc, H, basis, exact_info, approx_info


# =============================================================================
# N_LAYERS SWEEP
# =============================================================================
def sweep_n_layers(
            lattice: BaseLattice,
            j1: float = J1,
            j2: float = J2,
            chi_max: int = DMRG_CHI_MAX,
            rz_eps: float = RZ_SYNTHESIS_EPS,
            layers: list[int] | None = None,
            ) -> pd.DataFrame:
    """
    Sweep over n_layers values and collect approx-circuit metrics.

    DMRG runs once (inside run() with n_layers=layers[0]); the MPS is reused
    implicitly because each run() call shares the same lattice/j1/j2/chi_max.
    For large systems consider refactoring to run DMRG once and pass psi_mps in.

    Returns
    -------
    pd.DataFrame  with columns:
        n_layers, fidelity, energy_error_pct, approx_cx_count, approx_depth, …
    """
    if layers is None:
        layers = [1, 2, 3, 4, 5, 6, 8, 10]

    rows = []
    for n_lay in layers:
        print(f"\n[sweep_n_layers]  n_layers={n_lay}")
        _, _, H, basis, exact_info, approx_info = run(
            lattice,
            j1=j1,
            j2=j2,
            n_layers=n_lay,
            chi_max=chi_max,
            rz_eps=rz_eps,
            )
        rows.append({
            "n_layers":         n_lay,
            # energy metrics
            "exact_fidelity":   exact_info.get("exact_fidelity"),
            "approx_fidelity":  approx_info.get("approx_fidelity"),
            "exact_energy_err": exact_info.get("exact_energy_error_pct"),
            "approx_energy_err":approx_info.get("approx_energy_error_pct"),
            # circuit cost
            "approx_cx":        approx_info.get("approx_cx_count"),
            "approx_depth":     approx_info.get("approx_depth"),
            "exact_cx":         exact_info.get("exact_cx_count"),
            "exact_depth":      exact_info.get("exact_depth"),
            })

    df = pd.DataFrame(rows)
    print("\n[sweep_n_layers]  done")
    print(df.to_string(index=False))
    return df


# =============================================================================
# SWEEPS FOR THE POSTER
# =============================================================================
def sweep_j2_ratio(
            make_lattice_fn,
            L: int,
            ratios: list[float],
            j1: float = J1,
            chi_max: int = DMRG_CHI_MAX,
            rz_eps: float = RZ_SYNTHESIS_EPS,
            ) -> list[dict]:
    """
    Fixed system size L, sweep J2/J1 across the gapless -> dimerized
    transition (critical point near J2/J1 ~ 0.241; exactly solvable
    dimer point at J2/J1 = 0.5, the Majumdar-Ghosh point -- useful as a
    sanity check since the exact ground state is known there).

    Returns a list of per-point `info` dicts (see run()), ready to dump
    to CSV / a dataframe for plotting resource cost & entanglement vs.
    J2/J1.
    """
    rows = []
    for ratio in ratios:
        lattice = make_lattice_fn(L)
        j2 = ratio * j1
        try:
            exact_qc, approx_qc, H, basis, exact_info, approx_info = run(chain(8), j1=J1, j2=0.0, chi_max=DMRG_CHI_MAX)
            rows.append(exact_info)
            rows.append(approx_info)
        except Exception as exc:  # keep the sweep alive if one point fails
            print(f"[sweep_j2_ratio] FAILED at J2/J1={ratio}: {exc}")
            rows.append(dict(lattice=lattice.name, n_sites=L, j1=j1, j2=j2,
                              j2_over_j1=ratio, error=str(exc)))
    return rows


def sweep_system_size(
            make_lattice_fn,
            sizes: list[int],
            j1: float = J1,
            j2: float = J2,
            chi_max: int = DMRG_CHI_MAX,
            rz_eps: float = RZ_SYNTHESIS_EPS,
            ) -> list[dict]:
    """
    Fixed J2/J1, sweep system size at a fixed chi_max truncation. This is
    the scaling plot: resource cost (qubits, CX, RZ/T-count) and DMRG
    energy-per-site vs. N, at a bond dimension cap that does NOT grow with
    N -- showing where a fixed-chi_max budget starts to bite.
    """
    rows = []
    for n in sizes:
        lattice = make_lattice_fn(n)
        try:
            exact_qc, approx_qc, H, basis, exact_info, approx_info = run(chain(8), j1=J1, j2=0.0, chi_max=DMRG_CHI_MAX)
            info["energy_per_site"] = info["dmrg_energy"] / n
            rows.append(exact_info)
            rows.append(approx_info)
        except Exception as exc:
            print(f"[sweep_system_size] FAILED at N={n}: {exc}")
            rows.append(dict(lattice=lattice.name, n_sites=n, j1=j1, j2=j2,
                              chi_max_setting=chi_max, error=str(exc)))
    return rows


def save_results_csv(rows: list[dict], path: str) -> None:
    """Flatten a list of per-point info dicts (possibly with different keys,
    e.g. failed points) to a single CSV for plotting outside this script."""
    if not rows:
        print(f"[save_results_csv] nothing to write for {path}")
        return
    fieldnames: list[str] = []
    for row in rows:
        for k in row.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"[save_results_csv] wrote {len(rows)} rows -> {path}")


if __name__ == '__main__':
    from lattices import make_lattice

    def chain(L):
        return make_lattice('chain', L=L)
    lattice = chain(8)
    # --- single point, sanity check -----------------------------------------
    exact_qc, approx_qc, H, basis, exact_info, approx_info = run(chain(8), j1=J1, j2=0.0, chi_max=DMRG_CHI_MAX)

    # --- poster plot 1: resource cost & entanglement vs J2/J1 ---------------
    df = sweep_n_layers(lattice, layers=[1, 2, 3, 4, 6, 8])
    df.plot("n_layers", ["approx_fidelity", "approx_energy_err"])
    df.plot("approx_cx", "approx_fidelity", marker="o")   # cost vs quality
