# @title
"""
dmrg_mps_circuit.py
====================
DMRG ground state via TeNPy → Qiskit circuit + Hamiltonian + basis.

The TeNPy model is now driven entirely by a BaseLattice object so that
arbitrary geometries (chain, square, triangular, honeycomb, kagome, …)
are handled correctly.  The coupling topology comes from
  lattice.nn_edges   → J1 terms
  lattice.nnn_edges  → J2 terms
and the TeNPy lattice object (lattice.tenpy_lat) is used directly so
that the MPS bond structure respects the canonical MPS ordering that
TeNPy chose when it built the lattice.

Returns
-------
  run() → (exact_qc, approx_qc, H_sparse, basis)

Dependencies
------------
    pip install physics-tenpy qiskit scipy numpy
    pip install git+https://github.com/qiskit-community/mps-to-circuit.git
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from qiskit import transpile

# ── TeNPy ────────────────────────────────────────────────────────────────────
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
J1           = 1.0
J2           = 0.0
DMRG_CHI_MAX = 256
DMRG_MIXER   = True
BASIS_GATES  = ["cx", "rz", "h", "s", "sdg"]

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


# =============================================================================
# TENPY MODEL  —  topology-aware
# =============================================================================
class HeisenbergLattice(CouplingMPOModel):
    """
    Heisenberg J1-J2 model on an arbitrary geometry.

    model_params must contain:
        tenpy_lat  : the raw TeNPy lattice object  (from BaseLattice.tenpy_lat)
        nn_edges   : list[tuple[int,int]]  — nearest-neighbour pairs
        nnn_edges  : list[tuple[int,int]]  — next-nearest-neighbour pairs
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
        Add S_i · S_j = Sz_i Sz_j + ½(S+_i S-_j + S-_i S+_j).

        add_coupling_term(strength, i, op_i, j, op_j) accepts flat MPS
        site indices directly — no lattice-coordinate conversion needed.
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

    # Néel initial state along the MPS chain
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


# =============================================================================
# MPS → CIRCUITS
# =============================================================================
def mps_to_circuits(
    psi_mps: MPS,
    n: int,
    n_approx_layers: int = 10,
) -> tuple[QuantumCircuit, QuantumCircuit]:
    """
    Return (exact_qc, approx_qc) transpiled to BASIS_GATES.
    mps_to_circuit expects tensors in 'lpr' shape (left-bond, physical, right-bond).
    """
    mps_arrays = [psi_mps.get_B(i).to_ndarray() for i in range(n)]

    exact_qc  = mps_to_circuit(mps_arrays, method="exact",       shape="lpr")
    approx_qc = mps_to_circuit(mps_arrays, method="approximate", shape="lpr",
                                num_layers=n_approx_layers)

    exact_qc  = _transpile(exact_qc,  "exact")
    approx_qc = _transpile(approx_qc, "approximate")
    return exact_qc, approx_qc


def _transpile(qc: QuantumCircuit, label: str) -> QuantumCircuit:
    qc_t = transpile(qc, basis_gates=BASIS_GATES, optimization_level=0)
    print(f"\n[circuit:{label}]  qubits={qc_t.num_qubits}  "
          f"depth={qc_t.depth()}  gates={qc_t.size()}")
    for gate, count in qc_t.count_ops().items():
        print(f"  {gate:<5} {count}")
    return qc_t


# =============================================================================
# ENTRY POINT
# =============================================================================
def run(
    lattice: BaseLattice,
    j1: float = J1,
    j2: float = J2,
    chi_max: int = DMRG_CHI_MAX,
    n_approx_layers: int = 10,
) -> tuple[QuantumCircuit, QuantumCircuit, csr_matrix, np.ndarray]:
    """
    Run DMRG on *lattice* and return (exact_qc, approx_qc, H_sparse, basis).

    Parameters
    ----------
    lattice         : BaseLattice  — geometry, edges, and TeNPy object
    j1, j2          : Heisenberg couplings
    chi_max         : DMRG bond dimension cap
    n_approx_layers : brick-wall layers for the approximate MPS circuit

    Returns
    -------
    exact_qc   : Qiskit circuit (exact MPS isometry)
    approx_qc  : Qiskit circuit (approximate, n_approx_layers brick-wall layers)
    H          : sparse Hamiltonian in the n//2-up sector
    basis      : integer basis states for that sector
    """
    n = lattice.n_sites
    print(f"\n[run]  lattice={lattice.name}  N={n}  J1={j1}  J2={j2}  chi_max={chi_max}")

    H, basis = build_hamiltonian(lattice, j1, j2)
    print(f"[H]  sector dim={len(basis)}")

    _, psi_mps = run_dmrg(lattice, j1, j2, chi_max)
    exact_qc, approx_qc = mps_to_circuits(psi_mps, n, n_approx_layers)

    return exact_qc, approx_qc, H, basis

'''
if __name__ == '__main__':
    from lattices import make_lattice
    lattice = make_lattice('chain', L=8)
    exact_qc, approx_qc, H, basis = run(lattice, j1=J1, j2=J2, chi_max=DMRG_CHI_MAX)
'''
