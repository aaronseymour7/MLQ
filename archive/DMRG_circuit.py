"""
dmrg_mps_circuit.py
====================
DMRG ground state via TeNPy → Qiskit circuit + Hamiltonian + basis.

Returns
-------
  run() → (exact_qc, approx_qc, H_sparse, basis)

Dependencies
------------
    pip install tenpy qiskit scipy numpy
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
    from tenpy.models.lattice import Chain
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
# =============================================================================
def build_basis(n: int) -> np.ndarray:
    n_up = n // 2
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up],
                    dtype=np.int64)


def build_hamiltonian(n: int, j1: float = J1, j2: float = J2,
                      pbc: bool = True) -> tuple[csr_matrix, np.ndarray]:
    """Return (H_sparse, basis) in the n//2-up sector."""
    basis   = build_basis(n)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    H       = lil_matrix((len(basis), len(basis)), dtype=np.float64)

    nn  = [(i, (i + 1) % n) for i in range(n if pbc else n - 1)]
    nnn = [(i, (i + 2) % n) for i in range(n if pbc else n - 2)]

    for edges, j in [(nn, j1), (nnn, j2)]:
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
# TENPY MODEL
# =============================================================================
class HeisenbergChainJ1J2(CouplingMPOModel):
    def init_sites(self, model_params):
        return SpinHalfSite(conserve=model_params.get('conserve', 'Sz'))

    def init_lattice(self, model_params):
        site = self.init_sites(model_params)
        return Chain(model_params['L'], site,
                     bc_MPS=model_params.get('bc_MPS', 'finite'), bc='open')

    def init_terms(self, model_params):
        j1 = model_params.get('J1', 1.0)
        j2 = model_params.get('J2', 0.0)
        self.add_coupling(j1 / 2, 0, 'Sp', 0, 'Sm', dx=1, op_string='JW')
        self.add_coupling(j1 / 2, 0, 'Sm', 0, 'Sp', dx=1, op_string='JW')
        self.add_coupling(j1,     0, 'Sz', 0, 'Sz', dx=1)
        if j2:
            self.add_coupling(j2 / 2, 0, 'Sp', 0, 'Sm', dx=2, op_string='JW')
            self.add_coupling(j2 / 2, 0, 'Sm', 0, 'Sp', dx=2, op_string='JW')
            self.add_coupling(j2,     0, 'Sz', 0, 'Sz', dx=2)


# =============================================================================
# DMRG
# =============================================================================
def run_dmrg(n: int, j1: float = J1, j2: float = J2,
             chi_max: int = DMRG_CHI_MAX) -> tuple[float, MPS]:
    model_params = dict(L=n, J1=j1, J2=j2, conserve='Sz', bc_MPS='finite')
    model = HeisenbergChainJ1J2(model_params)

    initial_state = ['up' if i % 2 == 0 else 'down' for i in range(n)]
    psi = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc='finite')

    eng = dmrg.TwoSiteDMRGEngine(psi, model, {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'N_sweeps_check': 1,
        'mixer': DMRG_MIXER,
        'mixer_params': {'amplitude': 1e-4, 'decay': 1.2, 'disable_after': 50},
    })
    E, psi = eng.run()
    print(f"[DMRG]  E={E:.10f}  E/site={E/n:.10f}  chi_max={max(psi.chi)}")
    return E, psi


# =============================================================================
# MPS → CIRCUITS
# =============================================================================
def mps_to_circuits(psi_mps: MPS, n: int,
                    n_approx_layers: int = 10) -> tuple[QuantumCircuit, QuantumCircuit]:
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
def run(n: int, j1: float = J1, j2: float = J2,
        chi_max: int = DMRG_CHI_MAX, pbc: bool = True,
        n_approx_layers: int = 10,
        ) -> tuple[QuantumCircuit, QuantumCircuit, csr_matrix, np.ndarray]:
    """
    Run DMRG and return (exact_qc, approx_qc, H_sparse, basis).

    Parameters
    ----------
    n               : number of sites
    j1, j2          : Heisenberg couplings
    chi_max         : DMRG bond dimension
    pbc             : use PBC for the Hamiltonian (DMRG always uses OBC)
    n_approx_layers : brick-wall layers for the approximate circuit

    Returns
    -------
    exact_qc   : Qiskit circuit (exact MPS isometry)
    approx_qc  : Qiskit circuit (approximate, n_approx_layers brick-wall layers)
    H          : sparse Hamiltonian in the n//2-up sector
    basis      : integer basis states for that sector
    """
    print(f"\n[run]  N={n}  J1={j1}  J2={j2}  chi_max={chi_max}  pbc={pbc}")

    H, basis = build_hamiltonian(n, j1, j2, pbc=pbc)
    print(f"[H]  sector dim={len(basis)}")

    _, psi_mps = run_dmrg(n, j1, j2, chi_max)
    exact_qc, approx_qc = mps_to_circuits(psi_mps, n, n_approx_layers)

    return exact_qc, approx_qc, H, basis


if __name__ == '__main__':
    exact_qc, approx_qc, H, basis = run(n=8, j1=J1, j2=J2, chi_max=DMRG_CHI_MAX)
