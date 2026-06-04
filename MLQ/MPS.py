"""
dmrg_mps_circuit.py
====================
DMRG baseline using TeNPy, then converts the ground-state MPS to a quantum
circuit (Qiskit) and evaluates it with PennyLane — keeping the same Hamiltonian,
gate basis, and diagnostic output format as the UCJ script.

Pipeline
--------
  1.  Build J1-J2 Heisenberg Hamiltonian  →  run TeNPy DMRG
  2.  Extract ground-state MPS from TeNPy
  3.  Convert MPS → Qiskit QuantumCircuit  (isometric decomposition)
  4.  Transpile to target gate set {CNOT, RZ, S, H, Adjoint(S)}
  5.  Import circuit into PennyLane and evaluate energy + circuit metrics
  6.  Compare with exact ED  (same scipy sparse solver used in UCJ script)
  7.  Compute Schmidt spectrum / EE profile, write same CSV / TXT / JSON outputs

Dependencies
------------
    pip install tenpy qiskit qiskit-aer pennylane scipy numpy
"""

from __future__ import annotations

import json
import pathlib
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from qiskit import transpile
# ── TeNPy ────────────────────────────────────────────────────────────────────
try:
    import tenpy
    import tenpy.linalg.np_conserved as npc
    from tenpy.networks.site import SpinHalfSite
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    print(f"[TeNPy]  version {tenpy.__version__}")
except ImportError:
    raise ImportError("pip install physics-tenpy")

try:
    from mps_to_circuit import mps_to_circuit
except ImportError:
    raise ImportError("pip install git+https://github.com/qiskit-community/mps-to-circuit.git")
    
# ── Qiskit ───────────────────────────────────────────────────────────────────
try:
    from qiskit import QuantumCircuit
    from qiskit.synthesis.one_qubit import OneQubitEulerDecomposer
    from qiskit.quantum_info import Statevector
    from qiskit.synthesis import TwoQubitBasisDecomposer
    from qiskit.circuit.library import CXGate
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit.transpiler import Target, InstructionProperties
    from qiskit.circuit.library import CXGate, RZGate, SGate, HGate, SdgGate
    from qiskit.circuit import Parameter
    import qiskit
    print(f"[Qiskit]  version {qiskit.__version__}")
except ImportError:
    raise ImportError("pip install qiskit")

# ── PennyLane ────────────────────────────────────────────────────────────────
try:
    import pennylane as qml
    print(f"[PennyLane]  version {qml.version()}")
except ImportError:
    raise ImportError("pip install pennylane")

# =============================================================================
# CONFIG  (mirrors UCJ script)
# =============================================================================
J1   = 1.0
J2   = 0.0
PBC  = True        # PennyLane/ED uses PBC; TeNPy uses OBC by default (see below)

DMRG_CHI_MAX  = 256   # max MPS bond dimension for DMRG
DMRG_SWEEPS   = 10
DMRG_MIXER    = True

TARGET_GATES  = {"CNOT", "RZ", "S", "Hadamard", "Adjoint(S)"}  # same as UCJ

# =============================================================================
# TIMER  (identical to UCJ)
# =============================================================================
class Timer:
    def __init__(self):
        self._laps   = {}
        self._starts = {}

    def start(self, name: str):
        self._starts[name] = time.perf_counter()

    def stop(self, name: str) -> float:
        if name not in self._starts:
            raise KeyError(f"Timer '{name}' was never started.")
        self._laps[name] = time.perf_counter() - self._starts.pop(name)
        return self._laps[name]

    def summary(self, title: str = "Runtime summary"):
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
# EXACT DIAGONALISATION  (same as UCJ — for direct comparison)
# =============================================================================
def get_n_up(n: int) -> int:
    return (n + 1) // 2 if n % 2 == 1 else n // 2


def build_basis(n: int, n_up: int) -> np.ndarray:
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


def exact_ground_state(n: int, nn_edges, nnn_edges,
                       j1=J1, j2=J2) -> tuple[float, np.ndarray, np.ndarray, dict]:
    n_up = get_n_up(n)
    H_sp, basis, bindex = build_hamiltonian(n, n_up, nn_edges, nnn_edges, j1, j2)
    evals, evecs = eigsh(H_sp, k=1, which='SA')
    e_exact      = float(evals[0])
    psi_exact    = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    return e_exact, psi_exact, basis, bindex


# =============================================================================
# CHAIN EDGES  (PBC or OBC — mirrors UCJ lattice helper)
# =============================================================================
def chain_edges(n: int, pbc: bool = True):
    nn  = [(i, (i + 1) % n) for i in range(n if pbc else n - 1)]
    nnn = [(i, (i + 2) % n) for i in range(n if pbc else n - 2)]
    return nn, nnn


# =============================================================================
# TENPY HEISENBERG MODEL
# =============================================================================
class HeisenbergChainJ1J2(CouplingMPOModel):
    """J1-J2 Heisenberg chain in TeNPy."""

    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Sz')
        return SpinHalfSite(conserve=conserve)

    def init_lattice(self, model_params):
        L        = model_params['L']
        site     = self.init_sites(model_params)
        bc_MPS   = model_params.get('bc_MPS', 'finite')
        return Chain(L, site, bc_MPS=bc_MPS, bc='periodic' if PBC else 'open')

    def init_terms(self, model_params):
        J1 = model_params.get('J1', 1.0)
        J2 = model_params.get('J2', 0.0)
        self.add_coupling(J1 / 2, 0, 'Sp', 0, 'Sm', dx=1, op_string='JW')
        self.add_coupling(J1 / 2, 0, 'Sm', 0, 'Sp', dx=1, op_string='JW')
        self.add_coupling(J1,     0, 'Sz', 0, 'Sz', dx=1)
        if J2 != 0.0:
            self.add_coupling(J2 / 2, 0, 'Sp', 0, 'Sm', dx=2, op_string='JW')
            self.add_coupling(J2 / 2, 0, 'Sm', 0, 'Sp', dx=2, op_string='JW')
            self.add_coupling(J2,     0, 'Sz', 0, 'Sz', dx=2)


def run_dmrg(n: int, j1: float = J1, j2: float = J2,
             chi_max: int = DMRG_CHI_MAX,
             n_sweeps: int = DMRG_SWEEPS) -> tuple[float, MPS]:
    """
    Run DMRG and return (E_dmrg, psi_mps).
    Uses finite-chain OBC for DMRG (most stable), then we handle the
    PBC energy comparison separately via ED.
    """
    bc_MPS = 'finite'
    model_params = dict(L=n, J1=j1, J2=j2, conserve='Sz', bc_MPS=bc_MPS)
    model = HeisenbergChainJ1J2(model_params)

    # Initial MPS: alternating up/down (Néel)
    initial_state = ['up' if i % 2 == 0 else 'down' for i in range(n)]
    psi_mps = MPS.from_product_state(model.lat.mps_sites(), initial_state, bc='finite')

    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'N_sweeps_check': 1,
        'mixer': DMRG_MIXER,
        'mixer_params': {'amplitude': 1e-4, 'decay': 1.2, 'disable_after': 50},
    }

    eng = dmrg.TwoSiteDMRGEngine(psi_mps, model, dmrg_params)
    E_dmrg, psi_mps = eng.run()
    print(f"[DMRG]  E={E_dmrg:.10f}  E/site={E_dmrg/n:.10f}  "
          f"chi_max={max(psi_mps.chi)}  sweeps={n_sweeps}")
    return E_dmrg, psi_mps


# =============================================================================
# MPS → STATEVECTOR  (TeNPy → numpy)
# =============================================================================
def mps_to_statevector(psi_mps: MPS, n: int) -> np.ndarray:
    theta = psi_mps.get_theta(0, n)          # shape: vL i0 i1 ... i_{n-1} vR
    arr = theta.to_ndarray()                  # shape (1, 2, 2, ..., 2, 1)
    arr = arr.reshape([2] * n)
    sv = arr.flatten().astype(np.complex128)
    sv /= np.linalg.norm(sv)
    return sv


def mps_sv_to_sector(sv_full: np.ndarray, n: int,
                     basis: np.ndarray) -> np.ndarray:
    """
    Project full statevector onto the fixed-n_up sector used by ED.
    Returns a vector of length len(basis).
    """
    psi_sector = sv_full[basis]
    norm = np.linalg.norm(psi_sector)
    if norm < 1e-12:
        warnings.warn("MPS projection onto Sz sector has near-zero norm!")
    return psi_sector / (norm + 1e-300)


# =============================================================================
# STATEVECTOR → QISKIT CIRCUIT  (isometric / column-by-column decomposition)
# =============================================================================
def statevector_to_circuit(sv: np.ndarray, n_qubits: int) -> QuantumCircuit:
    from qiskit.circuit.library import StatePreparation
    sv_qiskit = _reorder_sv_for_qiskit(sv, n_qubits)
    qc = QuantumCircuit(n_qubits)
    qc.append(StatePreparation(sv_qiskit), range(n_qubits))
    qc_decomposed = qc.decompose(reps=6)
    return qc_decomposed


def _reorder_sv_for_qiskit(sv: np.ndarray, n: int) -> np.ndarray:
    """
    Reverse bit ordering: index k → reversed binary of k.
    This converts from big-endian (site 0 = MSB) to little-endian (qubit 0 = LSB).
    """
    out = np.zeros_like(sv)
    for i in range(len(sv)):
        j = int(f"{i:0{n}b}"[::-1], 2)
        out[j] = sv[i]
    return out


# =============================================================================
# TRANSPILE TO TARGET GATE SET  (same as UCJ: CNOT, RZ, S, H, Adjoint(S))
# =============================================================================
def transpile_to_target(qc: QuantumCircuit) -> QuantumCircuit:
    """
    Transpile the MPS circuit to {CX, RZ, S, H, Sdg} — matching UCJ's
    TARGET_GATES = {CNOT, RZ, S, Hadamard, Adjoint(S)}.
    """
    basis_gates = ['cx', 'rz', 's', 'h', 'sdg']
    pm = generate_preset_pass_manager(
        optimization_level=3,
        basis_gates=basis_gates,
        seed_transpiler=42,
    )
    qc_t = pm.run(qc)
    return qc_t


def gate_counts_from_qiskit(qc: QuantumCircuit) -> tuple[dict, int]:
    """Return (gate_counts_dict, depth) in the UCJ naming convention."""
    name_map = {
        'cx':  'CNOT',
        'rz':  'RZ',
        's':   'S',
        'h':   'Hadamard',
        'sdg': 'Adjoint(S)',
        'x':   'PauliX',
    }
    counts = {}
    for gate, n_gates in qc.count_ops().items():
        key = name_map.get(gate, gate)
        counts[key] = counts.get(key, 0) + n_gates
    return counts, qc.depth()


# =============================================================================
# PENNYLANE CIRCUIT FROM QISKIT  (noiseless energy evaluation)
# =============================================================================
def qiskit_to_pennylane_energy(
    qc_transpiled: QuantumCircuit,
    n: int,
    nn_edges: list,
    nnn_edges: list,
    j1: float = J1,
    j2: float = J2,
) -> float:
    """
    Evaluate energy from the Qiskit circuit's statevector directly,
    without requiring the pennylane-qiskit plugin.
    """
    from qiskit.quantum_info import Statevector
    sv = Statevector(qc_transpiled).data   # shape (2^n,), little-endian

    # Reverse bit ordering back to big-endian (site 0 = MSB) to match ED basis
    sv_be = _reorder_sv_for_qiskit(sv, n)  # same permutation is its own inverse

    dev = qml.device("default.qubit", wires=n)
    coeffs, obs = _build_heisenberg_pennylane(n, nn_edges, nnn_edges, j1, j2)

    @qml.qnode(dev)
    def energy_qnode():
        qml.StatePrep(sv_be, wires=range(n))
        return qml.expval(qml.dot(coeffs, obs))

    return float(energy_qnode())


def _build_heisenberg_pennylane(n, nn_edges, nnn_edges, j1=J1, j2=J2):
    """Identical to build_heisenberg_pennylane() in the UCJ script."""
    coeffs, obs = [], []
    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for Qop in [qml.PauliX, qml.PauliY, qml.PauliZ]:
                coeffs.append(j / 4.0)
                obs.append(Qop(wires=si) @ Qop(wires=sj))
    return coeffs, obs


# =============================================================================
# FIDELITY  (overlap with ED ground state)
# =============================================================================
def compute_fidelity(sv_mps: np.ndarray, psi_exact: np.ndarray,
                     basis: np.ndarray) -> float:
    """
    |<ψ_DMRG | ψ_exact>|²  in the fixed-n_up sector.
    sv_mps is the full 2^n vector; psi_exact lives in the sector basis.
    """
    psi_mps_sector = sv_mps[basis]
    norm = np.linalg.norm(psi_mps_sector)
    if norm < 1e-10:
        return 0.0
    psi_mps_sector /= norm
    return float(np.abs(np.dot(np.conj(psi_exact), psi_mps_sector)) ** 2)


# =============================================================================
# SCHMIDT SPECTRUM  (same implementation as UCJ)
# =============================================================================
def _split_basis(basis_np, n, cut):
    mask_left  = (1 << cut) - 1
    left_bits  = basis_np & mask_left
    right_bits = basis_np >> cut
    return left_bits, right_bits, left_bits.astype(int), right_bits.astype(int)


def schmidt_spectrum(psi, basis, n, cut=None, zero_thresh=1e-14):
    if cut is None:
        cut = n // 2
    basis_np = np.asarray(basis, dtype=np.int64)
    psi_np   = np.asarray(psi,   dtype=np.complex128)
    lb, rb, li, ri = _split_basis(basis_np, n, cut)
    dim_l = 1 << cut
    dim_r = 1 << (n - cut)
    Psi = np.zeros((dim_l, dim_r), dtype=np.complex128)
    Psi[li, ri] = psi_np
    sv    = np.linalg.svd(Psi, compute_uv=False, full_matrices=False)
    sv    = sv[sv > zero_thresh]
    lam2  = sv ** 2
    lam2 /= lam2.sum()
    s_vn     = float(-np.sum(lam2 * np.log(lam2 + 1e-300)))
    s_renyi2 = float(-np.log(np.sum(lam2 ** 2) + 1e-300))
    gap      = float(lam2[0] - lam2[1]) if len(lam2) > 1 else float(lam2[0])
    return dict(schmidt_values=lam2, entropy_vn=s_vn, entropy_renyi2=s_renyi2,
                schmidt_gap=gap, n_schmidt=len(lam2), cut=cut)


def print_schmidt(spec, label=""):
    lam2 = spec['schmidt_values']
    top  = min(8, len(lam2))
    print(f"\n[Schmidt  cut={spec['cut']}  {label}]")
    print(f"  S_vN        = {spec['entropy_vn']:.8f}")
    print(f"  S_Renyi2    = {spec['entropy_renyi2']:.8f}")
    print(f"  Schmidt gap = {spec['schmidt_gap']:.8f}")
    print(f"  n_Schmidt   = {spec['n_schmidt']}")
    print(f"  top-{top} λ²  : " + "  ".join(f"{v:.6f}" for v in lam2[:top]))


def compare_schmidt(psi_dmrg_sec, psi_exact, basis, n,
                    cuts=None, label_dmrg="DMRG", label_exact="exact"):
    if cuts is None:
        cuts = list(range(1, n))
    rows = []
    print(f"\n{'─'*66}")
    print(f"  Entanglement entropy profile  [{label_dmrg}] vs [{label_exact}]")
    print(f"  {'cut':>4}  {'S_vN (DMRG)':>12}  {'S_vN (exact)':>13}  "
          f"{'ΔS':>10}  {'S2 (DMRG)':>10}  {'S2 (exact)':>11}")
    print(f"  {'─'*62}")
    for c in cuts:
        sp_d = schmidt_spectrum(psi_dmrg_sec, basis, n, cut=c)
        sp_e = schmidt_spectrum(psi_exact,    basis, n, cut=c)
        dS   = sp_d['entropy_vn'] - sp_e['entropy_vn']
        print(f"  {c:>4}  {sp_d['entropy_vn']:>12.6f}  "
              f"{sp_e['entropy_vn']:>13.6f}  {dS:>+10.6f}  "
              f"{sp_d['entropy_renyi2']:>10.6f}  "
              f"{sp_e['entropy_renyi2']:>11.6f}")
        rows.append(dict(cut=c,
            S_ucj=sp_d['entropy_vn'],   S_exact=sp_e['entropy_vn'],  dS=dS,
            S2_ucj=sp_d['entropy_renyi2'], S2_exact=sp_e['entropy_renyi2'],
            dS2=sp_d['entropy_renyi2'] - sp_e['entropy_renyi2'],
            gap_ucj=sp_d['schmidt_gap'], gap_exact=sp_e['schmidt_gap'],
            n_schmidt_ucj=sp_d['n_schmidt']))
    print(f"  {'─'*62}")
    return rows


# =============================================================================
# CIRCUIT SUMMARY WRITER  (same format as UCJ write_circuit_summary)
# =============================================================================
def write_circuit_summary(
    lattice_name: str,
    n: int,
    j1: float,
    j2: float,
    chi_max: int,
    gate_counts: dict,
    depth: int,
    out_dir: str = "circuit_summaries",
) -> pathlib.Path:
    lat_tag = lattice_name.replace(" ", "_").replace("/", "-")
    j1_tag  = f"{j1:.3f}".replace(".", "p")
    j2_tag  = f"{j2:.3f}".replace(".", "p")
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = out_path / f"circuit_DMRG_{lat_tag}_N{n}_chi{chi_max}_J1{j1_tag}_J2{j2_tag}.txt"

    total = sum(gate_counts.values())
    W, sep, thin = 60, "=" * 60, "-" * 60

    header = [
        sep,
        f"  CIRCUIT SUMMARY  –  DMRG (chi_max={chi_max})",
        f"  Lattice  : {lattice_name}   N={n}",
        f"  J1={j1:.4f}   J2={j2:.4f}",
        f"  Gate set : CNOT, RZ, S, Hadamard, Adjoint(S)",
        sep, "",
        f"  [ DMRG MPS circuit ]",
        thin,
        f"  {'depth':<28}  {depth:>8}",
        f"  {'total gates':<28}  {total:>8}",
        thin,
    ]
    for gate, count in sorted(gate_counts.items(), key=lambda x: (-x[1], x[0])):
        header.append(f"  {gate:<28}  {count:>8}")
    header += [thin, ""]

    fpath.write_text("\n".join(header) + "\n", encoding="utf-8")
    print(f"  [circuit_summary]  → {fpath}")
    return fpath


# =============================================================================
# HISTORY CSV  (mirrors UCJ write_history_csv — single row for DMRG)
# =============================================================================
def write_history_csv(
    row: dict,
    lattice_name: str,
    n: int,
    j1: float,
    j2: float,
    chi_max: int,
    out_dir: str = "layer_summaries",
) -> pathlib.Path:
    import csv
    lat_tag = lattice_name.replace(" ", "_").replace("/", "-")
    j1_tag  = f"{j1:.3f}".replace(".", "p")
    j2_tag  = f"{j2:.3f}".replace(".", "p")
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fpath = out_path / f"history_DMRG_{lat_tag}_N{n}_chi{chi_max}_J1{j1_tag}_J2{j2_tag}.csv"
    fieldnames = list(row.keys())
    with open(fpath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"  [history] wrote → {fpath}")
    return fpath


# =============================================================================
# SUMMARY FILE  (mirrors UCJ write_summary_file)
# =============================================================================
def write_summary_file(
    lattice_name: str,
    n: int,
    j1: float,
    j2: float,
    chi_max: int,
    e_exact: float,
    e_dmrg: float,
    e_pl: float,
    fidelity: float,
    overlap: float,
    gate_counts: dict,
    depth: int,
    rdm_data: dict,
    out_dir: str = "summaries",
) -> pathlib.Path:
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    lat_tag = lattice_name.replace(" ", "_").replace("/", "-")
    j2_tag  = f"{j2:.3f}".replace(".", "p")
    fname   = out_path / f"{lat_tag}_DMRG_chi{chi_max}_J2_{j2_tag}.txt"

    W, sep, thin = 80, "=" * 80, "-" * 80
    lines = [
        sep,
        f"  DMRG BASELINE SUMMARY",
        f"  Lattice : {lattice_name}   N={n}",
        f"  J1={j1:.4f}   J2={j2:.4f}   chi_max={chi_max}",
        f"  E_exact = {e_exact:.10f}   E/site = {e_exact/n:.10f}",
        sep, "",
        f"  [ DMRG ]",
        thin,
        f"    E_dmrg    = {e_dmrg:.10f}",
        f"    |ΔE|      = {abs(e_dmrg - e_exact):.6e}",
        f"    E_pl      = {e_pl:.10f}",
        f"    fidelity  = {fidelity:.8f}",
        f"    overlap   = {overlap:.8f}",
        "", f"  [ CIRCUIT ]",
        thin,
        f"    depth       = {depth}",
        f"    total gates = {sum(gate_counts.values())}",
    ]
    for gate, count in sorted(gate_counts.items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"    {gate:<28}  {count}")
    lines += [
        "", f"  [ FROBENIUS NORMS @ CONVERGENCE ]",
        thin,
        f"    ‖Re(ρ_ij)‖_F = {rdm_data.get('re_frob', float('nan')):.8f}",
        f"    ‖Im(ρ_ij)‖_F = {rdm_data.get('im_frob', float('nan')):.8f}",
        f"    Im/Re ratio  = {rdm_data.get('ratio', float('nan')):.8f}",
        "", sep,
        f"  END OF SUMMARY  –  {lattice_name}  chi_max={chi_max}  J2={j2:.4f}",
        sep,
    ]
    fname.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[summary]  written → {fname}")

    json_data = dict(
        lattice=lattice_name, n_sites=n, j1=j1, j2=j2, chi_max=chi_max,
        e_exact=e_exact, e_dmrg=e_dmrg, e_pl=e_pl,
        abs_dE=abs(e_dmrg - e_exact), fidelity=fidelity, overlap=overlap,
        gate_counts=gate_counts, depth=depth, rdm=rdm_data,
    )
    json_fname = fname.with_suffix(".json")
    json_fname.write_text(json.dumps(json_data, indent=2), encoding="utf-8")
    print(f"[summary]  JSON  → {json_fname}")
    return fname


# =============================================================================
# RDM NORMS  (same as UCJ rdm_norms_at_convergence)
# =============================================================================
def rdm_norms(psi_sector: np.ndarray, n: int,
              basis: np.ndarray, bindex: dict) -> dict:
    basis_np  = list(basis)
    basis_arr = np.array(
        [[(1 if (b >> s) & 1 else -1) for s in range(n)] for b in basis_np],
        dtype=np.float64)
    occ   = (basis_arr + 1) / 2
    probs = np.abs(psi_sector) ** 2
    rho   = np.zeros((n, n), dtype=np.complex128)
    n_mean = (probs[:, None] * occ).sum(0)
    for i in range(n):
        rho[i, i] = n_mean[i]
    for i in range(n):
        for j in range(i + 1, n):
            mask = (basis_arr[:, i] == 1) & (basis_arr[:, j] == -1)
            if not mask.any():
                continue
            sigma_v  = basis_arr[mask]
            fb       = np.array(basis_np)[mask]
            new_bits = (fb ^ (1 << i) ^ (1 << j)).astype(np.int64)
            col_idx  = np.array([bindex.get(int(b), -1) for b in new_bits])
            valid    = col_idx >= 0
            if not valid.any():
                continue
            rows_v  = np.where(mask)[0][valid]
            rows_f  = col_idx[valid]
            jw      = np.array([
                (-1) ** int(((sigma_v[k, i+1:j] + 1) / 2).sum())
                for k in range(sigma_v.shape[0])
            ])[valid]
            rho_ij    = (np.conj(psi_sector[rows_f]) * psi_sector[rows_v] * jw).sum()
            rho[i, j] = rho_ij
            rho[j, i] = np.conj(rho_ij)

    mask_off = ~np.eye(n, dtype=bool)
    re_frob  = np.linalg.norm(np.real(rho)[mask_off])
    im_frob  = np.linalg.norm(np.imag(rho)[mask_off])
    ratio    = im_frob / (re_frob + 1e-12)
    print(f"\n[RDM  DMRG  N={n}]")
    print(f"  ‖Re(ρ_ij)‖_F = {re_frob:.6f}")
    print(f"  ‖Im(ρ_ij)‖_F = {im_frob:.6f}")
    print(f"  Im/Re ratio  = {ratio:.6f}"
          + ("  ← Im negligible" if ratio < 0.1 else
             "  ← Im moderate"   if ratio < 0.5 else
             "  ← Im significant"))
    return dict(re_frob=re_frob, im_frob=im_frob, ratio=ratio)


# =============================================================================
# MAIN  run()
# =============================================================================
def run(n: int, j1: float = J1, j2: float = J2,
        chi_max: int = DMRG_CHI_MAX, timer: Timer | None = None,
        lattice_name: str | None = None):

    if lattice_name is None:
        lattice_name = f"chain L={n}"

    W = 70
    print("\n" + "=" * W)
    print(f"  DMRG run  |  {lattice_name}  N={n}  J1={j1}  J2={j2}  "
          f"chi_max={chi_max}")
    print("=" * W)

    nn_edges, nnn_edges = chain_edges(n, pbc=PBC)

    # ── 1. Exact diagonalisation ─────────────────────────────────────────────
    label = f"ED {lattice_name}"
    if timer: timer.start(label)
    e_exact, psi_exact, basis, bindex = exact_ground_state(
        n, nn_edges, nnn_edges, j1, j2)
    if timer: timer.stop(label)
    print(f"\n[Lanczos]  {lattice_name}  N={n}  J1={j1}  J2={j2}  "
          f"E_exact={e_exact:.8f}  E/site={e_exact/n:.8f}")

    # ── 2. DMRG ──────────────────────────────────────────────────────────────
    label = f"DMRG {lattice_name}"
    if timer: timer.start(label)
    e_dmrg, psi_mps = run_dmrg(n, j1, j2, chi_max)
    if timer: timer.stop(label)
    mps_arrays = [psi_mps.get_B(i).to_ndarray() for i in range(n)]
    ex_qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")
    qc_info(ex_qc, 'exact')
    apx_qc = mps_to_circuit(mps_arrays, method="approximate", shape="lpr", num_layers=10)
    qc_info(apx_qc, 'Approximate')
    return ex_qc

def qc_info(qc, circuit_type):
    print(f"\n[{circuit_type}]")

    basis = ["cx", "rz", "h", "s", "sdg"]

    qc_basis = transpile(
        qc,
        basis_gates=basis,
        optimization_level=0
    )

    print(f"Qubits: {qc_basis.num_qubits}")
    print(f"Depth: {qc_basis.depth()}")
    print(f"Total gates: {qc_basis.size()}")
    print("Gate counts:")

    for gate, count in qc_basis.count_ops().items():
        print(f"  {gate:<5} {count}")
# =============================================================================
# ENTRY
# =============================================================================
if __name__ == '__main__':
    n_list  = [8]
    J2_list = [0.0]

    for n in n_list:
        timer = Timer()
        for j2 in J2_list:
            run(n=n, j1=J1, j2=j2, chi_max=DMRG_CHI_MAX,
                timer=timer, lattice_name=f"chain L={n}")
        timer.summary(f"DMRG sweep  N={n}")
