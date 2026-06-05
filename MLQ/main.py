"""
main.py
=======
Single entry point comparing UCJ and DMRG ground-state circuits on the same
Heisenberg lattice.

Global config
-------------
  J1, J2    : Heisenberg couplings (defined below)
  LATTICE   : BaseLattice object shared by both methods

Design contract
---------------
  * Exact ground-state energy, eigenvector, and full spectrum come from a
    single Lanczos call (get_ground_state from ucj_circuit.py).  DMRG only
    adds its own variational energy; it does NOT re-run the diagonalisation.
  * Both circuits are transpiled to TARGET_BASIS before gate counts are printed.
  * Both circuits have the time-filter appended (append_filter from
    time_filter.py) using Lanczos energy gaps.
  * Metrics reported:
      - Fidelity with exact GS  before  filter
      - Fidelity with exact GS  after   filter
      - Max bond dimension used by DMRG
      - Gate counts (per gate type) for each circuit in TARGET_BASIS
"""

from __future__ import annotations

# ── stdlib ────────────────────────────────────────────────────────────────────
import textwrap
from typing import Optional

# ── numerics ──────────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse.linalg import eigsh

# ── Qiskit ────────────────────────────────────────────────────────────────────
from qiskit import transpile, QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp

# ── project modules ───────────────────────────────────────────────────────────
from lattices import BaseLattice, make_lattice          # shared lattice layer

# UCJ: imports every public symbol so nothing is re-implemented here
from ucj import (
    _get_n_up,
    _build_basis,
    _build_hamiltonian_op,
    get_ground_state,                  # ← single Lanczos call lives here
    build_jax_hamiltonian,
    build_jastrow_fn,
    build_givens_pairs,
    build_ucj,
    TARGET_BASIS as UCJ_TARGET_BASIS,
    VARIANT      as UCJ_VARIANT,
    K_LAYERS     as UCJ_K_LAYERS,
)

# DMRG: imports every public symbol so nothing is re-implemented here
from mps import (
    build_hamiltonian   as dmrg_build_hamiltonian,
    build_basis         as dmrg_build_basis,
    HeisenbergLattice,
    run_dmrg,
    mps_to_circuits,
    _transpile          as dmrg_transpile,
    BASIS_GATES         as DMRG_BASIS_GATES,
    DMRG_CHI_MAX,
)

# Time filter: imports every public symbol
from filter import (
    FilterBuilder,
    append_filter,
    new_func_v4,
    new_func_v5,
    fixtimes,
    unpack,
    timesconstraints,
    probability_constraints,
    probability_constraintsb,
    build_filter_circuit,
    post_filter_fidelity_analytic,
)

# =============================================================================
# ── GLOBAL CONFIGURATION ─────────────────────────────────────────────────────
# =============================================================================

J1       = 1.0
J2       = 0.7

# Define the lattice once; both methods share it.
# Change the arguments below to switch geometry / system size.
N_SITES  = 12
LATTICE: BaseLattice = make_lattice("chain", L=N_SITES)

# Target gate basis for transpilation and gate-count reporting
TARGET_BASIS = ["cx", "rz", "h", "s", "sdg"]

# DMRG bond dimension cap
CHI_MAX = DMRG_CHI_MAX

# UCJ hyper-parameters
VARIANT  = 'g'    # 're' | 'im' | 'g'
K_LAYERS = 1

# Time-filter settings
FILTER_TOTAL_TIME = 20.0
FILTER_A          = 4      # min number of pulses to sweep
FILTER_B          = 8      # max number of pulses to sweep
FILTER_METHOD     = "v5"   # 'v3' | 'v4' | 'v5'
FILTER_MAXITER    = 5000
FILTER_FTOL       = 1e-12
TROTTER_STEPS     = 1      # Trotter steps per filter pulse

# =============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# =============================================================================

_DIVIDER = "─" * 72


def _header(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _print_gate_counts(label: str, qc: QuantumCircuit) -> None:
    ops   = qc.count_ops()
    total = sum(ops.values())
    print(f"\n  [{label}]  qubits={qc.num_qubits}  depth={qc.depth()}  "
          f"total_gates={total}")
    for gate in TARGET_BASIS:
        print(f"    {gate:<8} {ops.get(gate, 0)}")
    others = {g: c for g, c in ops.items() if g not in TARGET_BASIS}
    if others:
        print("    --- non-basis gates (not yet decomposed) ---")
        for gate, cnt in sorted(others.items(), key=lambda x: -x[1]):
            print(f"    {gate:<8} {cnt}")


def _statevector_from_circuit(qc: QuantumCircuit) -> np.ndarray:
    """Simulate the circuit and return the statevector as a numpy array."""
    sv = Statevector(qc)
    return np.array(sv)


def _fidelity_with_sector(
    sv_full: np.ndarray,
    psi_exact: np.ndarray,
    basis: np.ndarray,
    n: int,
) -> float:
    """
    Project the full 2^n statevector onto the fixed-Sz sector and compute
    |<psi_exact | psi_sector>|^2.

    psi_exact lives in the sector basis (length = len(basis)).
    sv_full   lives in the full 2^n Hilbert space.
    """
    sector_sv = sv_full[basis]                      # pick sector amplitudes
    norm = np.linalg.norm(sector_sv)
    if norm < 1e-14:
        return 0.0
    sector_sv = sector_sv / norm                    # normalise within sector
    overlap   = np.dot(np.conj(psi_exact), sector_sv)
    return float(np.abs(overlap) ** 2)


def _build_hamiltonian_pauli(lattice: BaseLattice,
                             j1: float, j2: float) -> SparsePauliOp:
    """
    Build a SparsePauliOp for the Heisenberg J1-J2 model so that
    append_filter can use Qiskit's PauliEvolutionGate.
    """
    n = lattice.n_sites
    pauli_list = []

    def _zz(i: int, j: int, coeff: float) -> None:
        label = ["I"] * n
        label[i] = "Z"
        label[j] = "Z"
        pauli_list.append(("".join(reversed(label)), coeff * 0.25))

    def _xx(i: int, j: int, coeff: float) -> None:
        label = ["I"] * n
        label[i] = "X"
        label[j] = "X"
        pauli_list.append(("".join(reversed(label)), coeff * 0.5))

    def _yy(i: int, j: int, coeff: float) -> None:
        label = ["I"] * n
        label[i] = "Y"
        label[j] = "Y"
        pauli_list.append(("".join(reversed(label)), coeff * 0.5))

    for (i, j) in lattice.nn_edges:
        _zz(i, j, j1); _xx(i, j, j1); _yy(i, j, j1)
    for (i, j) in lattice.nnn_edges:
        _zz(i, j, j2); _xx(i, j, j2); _yy(i, j, j2)

    return SparsePauliOp.from_list(pauli_list).simplify()


def _full_spectrum_gaps(
    lattice: BaseLattice,
    j1: float,
    j2: float,
    n_eigvals: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (energies, gaps) where gaps = E_k - E_0,
    computed from the k lowest eigenstates of the sector Hamiltonian.
    This reuses _build_hamiltonian_op from ucj_circuit.py.
    """
    n    = lattice.n_sites
    n_up = _get_n_up(n)
    op, basis, _ = _build_hamiltonian_op(
        n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2)
    k    = min(n_eigvals, len(basis) - 1)
    evals, _ = eigsh(op, k=k, which="SA", tol=1e-10, maxiter=20_000)
    evals    = np.sort(evals.real)
    return evals, evals - evals[0]


def _coeffs_sq_in_eigenbasis(
    psi_trial: np.ndarray,
    lattice: BaseLattice,
    j1: float,
    j2: float,
    n_eigvals: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # ← added evecs
    n    = lattice.n_sites
    n_up = _get_n_up(n)
    op, basis, _ = _build_hamiltonian_op(
        n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2)
    k = min(n_eigvals, len(basis) - 1)
    evals, evecs = eigsh(op, k=k, which="SA", tol=1e-10, maxiter=20_000)
    order  = np.argsort(evals.real)
    evals  = evals[order].real
    evecs  = evecs[:, order]
    coeffs = evecs.conj().T @ psi_trial   # ← was (evecs.T @ ...).real
    return evals, np.abs(coeffs) ** 2, coeffs, evecs


# =============================================================================
# ── LANCZOS (single call) ─────────────────────────────────────────────────────
# =============================================================================

def run_lanczos(
    lattice: BaseLattice,
    j1: float,
    j2: float,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Single Lanczos diagonalisation shared by UCJ and DMRG.

    Returns
    -------
    e0        : ground-state energy
    psi_exact : ground-state vector in sector basis
    basis     : integer sector basis (length = dim)
    idx_map   : dict mapping integer bitstring → row index
    """
    _header("Lanczos exact diagonalisation  (single call)")
    n    = lattice.n_sites
    n_up = _get_n_up(n)
    e0, psi_exact, basis, idx_map = get_ground_state(
        n, n_up,
        lattice.nn_edges, lattice.nnn_edges,
        j1, j2,
    )
    print(f"  N={n}  n_up={n_up}  sector_dim={len(basis)}")
    print(f"  E0       = {e0:.10f}")
    print(f"  E0/site  = {e0/n:.10f}")
    return e0, psi_exact, basis, idx_map


# =============================================================================
# ── UCJ RUN ───────────────────────────────────────────────────────────────────
# =============================================================================

def run_ucj_comparison(
    lattice: BaseLattice,
    j1: float,
    j2: float,
    e0: float,
    psi_exact: np.ndarray,
    basis: np.ndarray,
    hamiltonian_pauli: SparsePauliOp,
    energies: np.ndarray,
    gaps: np.ndarray,
) -> dict:

    _header("UCJ circuit construction + optimisation")
    n = lattice.n_sites

    tqc_ucj, gate_counts_ucj, depth_ucj = build_ucj(
        lattice, j1, j2, variant=VARIANT, k_layers=K_LAYERS,
        basis_gates=TARGET_BASIS,
    )

    _header("UCJ — pre-filter fidelity")
    sv_ucj      = _statevector_from_circuit(tqc_ucj)
    fid_pre_ucj = _fidelity_with_sector(sv_ucj, psi_exact, basis, n)
    print(f"  UCJ fidelity (pre-filter)  = {fid_pre_ucj:.8f}")

    _header("UCJ — time filter optimisation")
    sector_sv_ucj = sv_ucj[basis]
    norm          = np.linalg.norm(sector_sv_ucj)
    sector_sv_ucj = sector_sv_ucj / (norm + 1e-30)
    overlap_ucj   = float(np.abs(np.dot(np.conj(psi_exact), sector_sv_ucj)))

    # ── CHANGE 1: unpack the new 4th return value ──────────────────────────
    evals_filter, coeffs_sq_ucj, coeffs_ucj, evecs_filter = \
        _coeffs_sq_in_eigenbasis(sector_sv_ucj, lattice, j1, j2)

    fb_ucj = FilterBuilder(
        total_time = FILTER_TOTAL_TIME,
        energies   = evals_filter,
        overlap    = overlap_ucj,
        a          = FILTER_A,
        b          = FILTER_B,
        maxiter    = FILTER_MAXITER,
        ftol       = FILTER_FTOL,
        coeffs_sq  = coeffs_sq_ucj,
    )
    filter_results_ucj = fb_ucj.build(method=FILTER_METHOD)
    best_ucj = _best_filter(filter_results_ucj, sector_sv_ucj, psi_exact, evecs_filter, evals_filter)
    print(f"\n  Best filter:  ntimes={best_ucj['ntimes']}  "
          f"fun={best_ucj['fun']:.4e}  success={best_ucj['success']}")

    # ── Step 4 unchanged — keep the Hadamard-test circuit for gate counts ──
    qc_ucj_filtered = transpile(
        build_filter_circuit(
            tqc_ucj, hamiltonian_pauli,
            best_ucj["times"], best_ucj["phases"],
            trotter_steps=TROTTER_STEPS,
        ),
        basis_gates=TARGET_BASIS,
        optimization_level=0,
    )
    
    # ── CHANGE 2: analytic post-filter fidelity (honest Hadamard-test prediction)
    _header("UCJ — post-filter fidelity (analytic, Hadamard test)")
    fid_post_ucj, postsel_prob = post_filter_fidelity_analytic(
        sector_sv_ucj, psi_exact,
        evecs_filter, evals_filter,
        best_ucj["times"], best_ucj["phases"],
    )
    print(f"  UCJ fidelity (post-filter, analytic)  = {fid_post_ucj:.8f}")
    print(f"  Postselection probability              = {postsel_prob:.6f}")

    gate_counts_filtered_ucj = dict(qc_ucj_filtered.count_ops())

    return {
        "qc_raw":               tqc_ucj,
        "qc_filtered":          qc_ucj_filtered,
        "fidelity_pre":         fid_pre_ucj,
        "fidelity_post":        fid_post_ucj,
        "gate_counts_raw":      gate_counts_ucj,
        "gate_counts_filtered": gate_counts_filtered_ucj,
        "postsel_prob":         postsel_prob
    }

# =============================================================================
# ── DMRG RUN ──────────────────────────────────────────────────────────────────
# =============================================================================

def run_dmrg_comparison(
    lattice: BaseLattice,
    j1: float,
    j2: float,
    e0: float,
    psi_exact: np.ndarray,
    basis: np.ndarray,
    hamiltonian_pauli: SparsePauliOp,
    energies: np.ndarray,
    gaps: np.ndarray,
) -> dict:
    _header("DMRG ground state")
    n = lattice.n_sites

    # ── 1. DMRG ───────────────────────────────────────────────────────────────
    dmrg_E, psi_mps = run_dmrg(lattice, j1, j2, chi_max=CHI_MAX)
    max_bond_dim    = int(max(psi_mps.chi))
    print(f"  DMRG energy   = {dmrg_E:.10f}")
    print(f"  Exact energy  = {e0:.10f}")
    print(f"  |ΔE|          = {abs(dmrg_E - e0):.4e}")
    print(f"  Max bond dim  = {max_bond_dim}")

    # ── 2. MPS → circuits ─────────────────────────────────────────────────────
    _header("DMRG — MPS → circuits")
    exact_qc, approx_qc = mps_to_circuits(psi_mps, n)

    # ── 3. Pre-filter fidelities ──────────────────────────────────────────────
    _header("DMRG — pre-filter fidelities")
    sv_exact  = _statevector_from_circuit(exact_qc)
    sv_approx = _statevector_from_circuit(approx_qc)

    fid_exact_pre  = _fidelity_with_sector(sv_exact,  psi_exact, basis, n)
    fid_approx_pre = _fidelity_with_sector(sv_approx, psi_exact, basis, n)
    print(f"  DMRG exact  fidelity (pre-filter)  = {fid_exact_pre:.8f}")
    print(f"  DMRG approx fidelity (pre-filter)  = {fid_approx_pre:.8f}")

    # ── 4. Sector statevectors (normalised) ───────────────────────────────────
    sector_sv_exact = sv_exact[basis]
    sector_sv_exact = sector_sv_exact / (np.linalg.norm(sector_sv_exact) + 1e-30)

    sector_sv_approx = sv_approx[basis]
    sector_sv_approx = sector_sv_approx / (np.linalg.norm(sector_sv_approx) + 1e-30)

    # ── 5. Filter for exact circuit ───────────────────────────────────────────
    _header("DMRG — time filter optimisation  (exact MPS circuit)")
    evals_exact, coeffs_sq_exact, coeffs_exact, evecs_exact = \
        _coeffs_sq_in_eigenbasis(sector_sv_exact, lattice, j1, j2)

    fb_exact = FilterBuilder(
        total_time = FILTER_TOTAL_TIME,
        energies   = evals_exact,
        overlap    = float(np.abs(np.dot(np.conj(psi_exact), sector_sv_exact))),
        a          = FILTER_A,
        b          = FILTER_B,
        maxiter    = FILTER_MAXITER,
        ftol       = FILTER_FTOL,
        coeffs_sq  = coeffs_sq_exact,
    )
    filter_results_exact = fb_exact.build(method=FILTER_METHOD)
    best_exact = _best_filter(filter_results_exact, sector_sv_exact, psi_exact, evecs_exact, evals_exact)
    print(f"\n  Best filter:  ntimes={best_exact['ntimes']}  "
          f"fun={best_exact['fun']:.4e}  success={best_exact['success']}")

    # ── 6. Filter for approx circuit (its own optimisation) ───────────────────
    _header("DMRG — time filter optimisation  (approx MPS circuit)")
    evals_approx, coeffs_sq_approx, coeffs_approx, evecs_approx = \
        _coeffs_sq_in_eigenbasis(sector_sv_approx, lattice, j1, j2)

    fb_approx = FilterBuilder(
        total_time = FILTER_TOTAL_TIME,
        energies   = evals_approx,
        overlap    = float(np.abs(np.dot(np.conj(psi_exact), sector_sv_approx))),
        a          = FILTER_A,
        b          = FILTER_B,
        maxiter    = FILTER_MAXITER,
        ftol       = FILTER_FTOL,
        coeffs_sq  = coeffs_sq_approx,
    )
    filter_results_approx = fb_approx.build(method=FILTER_METHOD)
    best_approx = _best_filter(filter_results_approx, sector_sv_approx, psi_exact, evecs_approx, evals_approx)    
    print(f"\n  Best filter:  ntimes={best_approx['ntimes']}  "
          f"fun={best_approx['fun']:.4e}  success={best_approx['success']}")

    # ── 7. Build Hadamard-test filter circuits ────────────────────────────────
    qc_exact_filtered  = transpile(build_filter_circuit(
        exact_qc, hamiltonian_pauli,
        best_exact["times"], best_exact["phases"],
        trotter_steps=TROTTER_STEPS),
        basis_gates=TARGET_BASIS,
        optimization_level=0,)
    
    qc_approx_filtered = transpile(build_filter_circuit(
        approx_qc, hamiltonian_pauli,
        best_approx["times"], best_approx["phases"],
        trotter_steps=TROTTER_STEPS),basis_gates=TARGET_BASIS,
        optimization_level=0,)

    # ── 8. Post-filter fidelities (analytic, Hadamard-test prediction) ────────
    _header("DMRG — post-filter fidelities (analytic, Hadamard test)")
    fid_exact_post, prob_exact = post_filter_fidelity_analytic(
        sector_sv_exact, psi_exact,
        evecs_exact, evals_exact,
        best_exact["times"], best_exact["phases"],
    )
    print(f"  DMRG exact  fidelity (post-filter) = {fid_exact_post:.8f}  "
          f"postsel_prob={prob_exact:.6f}")

    fid_approx_post, prob_approx = post_filter_fidelity_analytic(
        sector_sv_approx, psi_exact,
        evecs_approx, evals_approx,
        best_approx["times"], best_approx["phases"],
    )
    print(f"  DMRG approx fidelity (post-filter) = {fid_approx_post:.8f}  "
          f"postsel_prob={prob_approx:.6f}")

    return {
        "qc_exact_raw":               exact_qc,
        "qc_approx_raw":              approx_qc,
        "qc_exact_filtered":          qc_exact_filtered,
        "qc_approx_filtered":         qc_approx_filtered,
        "fidelity_exact_pre":         fid_exact_pre,
        "fidelity_approx_pre":        fid_approx_pre,
        "fidelity_exact_post":        fid_exact_post,
        "fidelity_approx_post":       fid_approx_post,
        "postsel_prob_exact":         prob_exact,
        "postsel_prob_approx":        prob_approx,
        "max_bond_dim":               max_bond_dim,
        "dmrg_energy":                dmrg_E,
        "gate_counts_exact_raw":      dict(exact_qc.count_ops()),
        "gate_counts_approx_raw":     dict(approx_qc.count_ops()),
        "gate_counts_exact_filtered": dict(qc_exact_filtered.count_ops()),
        "gate_counts_approx_filtered":dict(qc_approx_filtered.count_ops()),
    }


# =============================================================================
# ── SUMMARY PRINTER ───────────────────────────────────────────────────────────
# =============================================================================

def print_summary(
    e0: float,
    n: int,
    ucj_res: dict,
    dmrg_res: dict,
) -> None:
    _header("GATE COUNTS  —  target basis: " + str(TARGET_BASIS))

    _print_gate_counts("UCJ  raw (pre-filter)",      ucj_res["qc_raw"])
    _print_gate_counts("UCJ  filtered",              ucj_res["qc_filtered"])
    _print_gate_counts("DMRG exact  raw",            dmrg_res["qc_exact_raw"])
    _print_gate_counts("DMRG exact  filtered",       dmrg_res["qc_exact_filtered"])
    _print_gate_counts("DMRG approx raw",            dmrg_res["qc_approx_raw"])
    _print_gate_counts("DMRG approx filtered",       dmrg_res["qc_approx_filtered"])

    _header("METRICS SUMMARY")
    row_fmt = "  {:<38} {}"
    print(row_fmt.format("Exact E0 (Lanczos)",
                         f"{e0:.10f}"))
    print(row_fmt.format("Exact E0/site",
                         f"{e0/n:.10f}"))
    print()
    print(row_fmt.format("DMRG energy",
                         f"{dmrg_res['dmrg_energy']:.10f}"))
    print(row_fmt.format("DMRG |ΔE| vs Lanczos",
                         f"{abs(dmrg_res['dmrg_energy'] - e0):.4e}"))
    print(row_fmt.format("DMRG max bond dimension",
                         dmrg_res["max_bond_dim"]))
    print()
    print(row_fmt.format("UCJ  fidelity  pre-filter",
                         f"{ucj_res['fidelity_pre']:.8f}"))
    print(row_fmt.format("UCJ  fidelity  post-filter",
                         f"{ucj_res['fidelity_post']:.8f}"))
    print()
    print(row_fmt.format("DMRG exact  fidelity  pre-filter",
                         f"{dmrg_res['fidelity_exact_pre']:.8f}"))
    print(row_fmt.format("DMRG exact  fidelity  post-filter",
                         f"{dmrg_res['fidelity_exact_post']:.8f}"))
    print()
    print(row_fmt.format("DMRG approx fidelity  pre-filter",
                         f"{dmrg_res['fidelity_approx_pre']:.8f}"))
    print(row_fmt.format("DMRG approx fidelity  post-filter",
                         f"{dmrg_res['fidelity_approx_post']:.8f}"))
    
    print(row_fmt.format("UCJ  postselection probability",
                     f"{ucj_res['postsel_prob']:.6f}"))
    print(row_fmt.format("UCJ  shots per postselection",
                        f"{int(np.ceil(1/ucj_res['postsel_prob']))}"))
    print()
    print(row_fmt.format("DMRG exact  postsel. probability",
                        f"{dmrg_res['postsel_prob_exact']:.6f}"))
    print(row_fmt.format("DMRG approx postsel. probability",
                        f"{dmrg_res['postsel_prob_approx']:.6f}"))
    print(row_fmt.format("DMRG approx shots per postselection",
                        f"{int(np.ceil(1/dmrg_res['postsel_prob_approx']))}"))
    print()
    print(_DIVIDER)


# =============================================================================
# ── ENTRY POINT ───────────────────────────────────────────────────────────────
# =============================================================================

def main() -> tuple[dict, dict]:
    """
    Run the full UCJ-vs-DMRG comparison pipeline.

    Execution order
    ---------------
    1.  Lanczos         — exact GS energy + statevector (single call)
    2.  Lanczos gaps    — low-lying spectrum for filter optimisation
    3.  UCJ             — variational optimisation → circuit
    4.  DMRG            — TeNPy ground state → MPS circuits
    5.  Time filter     — optimise + append to all circuits
    6.  Gate counts     — print per gate type in TARGET_BASIS
    7.  Metrics         — fidelity (pre/post filter), max bond dim

    Returns
    -------
    ucj_res  : dict with UCJ results (see run_ucj_comparison)
    dmrg_res : dict with DMRG results (see run_dmrg_comparison)
    """
    _header(f"UCJ vs DMRG  —  N={N_SITES}  J1={J1}  J2={J2}  "
            f"lattice={LATTICE.name}")

    n = LATTICE.n_sites

    # ── 1. Single Lanczos call ────────────────────────────────────────────────
    e0, psi_exact, basis, idx_map = run_lanczos(LATTICE, J1, J2)

    # ── 2. Low-lying spectrum (energy gaps for filter) ────────────────────────
    _header("Low-lying spectrum for time-filter energy gaps")
    energies, gaps = _full_spectrum_gaps(LATTICE, J1, J2, n_eigvals=20)
    print(f"  Lowest {len(energies)} eigenvalues:")
    for k, (e, g) in enumerate(zip(energies, gaps)):
        print(f"    k={k:2d}  E={e:.8f}  gap={g:.8f}")

    # ── 3. Pauli Hamiltonian (needed by append_filter) ────────────────────────
    hamiltonian_pauli = _build_hamiltonian_pauli(LATTICE, J1, J2)

    # ── 4. UCJ ────────────────────────────────────────────────────────────────
    ucj_res = run_ucj_comparison(
        LATTICE, J1, J2,
        e0, psi_exact, basis,
        hamiltonian_pauli, energies, gaps,
    )

    # ── 5. DMRG ───────────────────────────────────────────────────────────────
    dmrg_res = run_dmrg_comparison(
        LATTICE, J1, J2,
        e0, psi_exact, basis,
        hamiltonian_pauli, energies, gaps,
    )

    # ── 6. Summary ────────────────────────────────────────────────────────────
    print_summary(e0, n, ucj_res, dmrg_res)

    return ucj_res, dmrg_res


if __name__ == "__main__":
    ucj_res, dmrg_res = main()
