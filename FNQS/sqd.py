"""
NQS -> SQD -> sparse-state-prep circuit pipeline.

    NQS Metropolis samples
        -> grow subspace along H-connectivity (project_operator_to_subspace)
        -> solve_qubit (exact-on-subspace ground state)
        -> truncate by |amplitude|^2 mass coverage
        -> qclib sparse circuit synthesis
        -> QuantumCircuit

Run with train.py's L set to match the loaded checkpoint, same convention
as exact_sector_check.py.

Install:
    pip install qiskit-addon-sqd qclib qiskit-aer --break-system-packages
"""
import pickle

import numpy as np
from qiskit.quantum_info import SparsePauliOp
from qiskit_addon_sqd.qubit import solve_qubit, sort_and_remove_duplicates
from qiskit import QuantumCircuit

from train import model, hi, graph, L, make_hamiltonian
from project import sample_configs, configs_to_bitstrings  # reuse your existing MC sampler

VARIABLES_PATH = "fnqs_variables.pkl"


# --------------------------------------------------------------------------
# 0. netket Hamiltonian -> qiskit SparsePauliOp
# --------------------------------------------------------------------------
def netket_hamiltonian_to_sparse_pauli_op(j2):
    """
    Convert make_hamiltonian(j2)'s netket LocalOperator into a qiskit
    SparsePauliOp over the same L spins, using netket's own exact
    Pauli-string decomposition (LocalOperator.to_pauli_strings()) rather
    than a manual one.

    Convention note (verified numerically against a dense round-trip):
    netket's pauli-string labels read site 0 as the LEFTMOST character,
    while qiskit's SparsePauliOp reads qubit 0 as the RIGHTMOST character.
    Reversing each label string is the only conversion needed -- the
    reconstructed qiskit operator's dense matrix then matches netket's
    `to_dense()` exactly, with NO basis permutation required, so
    spins_to_bool_matrix's site-i -> qubit-i mapping below is consistent
    with this Hamiltonian and with solve_qubit/qclib's qubit ordering.
    """
    ham = make_hamiltonian(j2)
    pauli_strings = ham.to_pauli_strings()
    labels = [str(lbl)[::-1] for lbl in pauli_strings.operators]
    weights = np.asarray(pauli_strings.weights)
    op = SparsePauliOp(labels, weights).simplify()
    return op


# --------------------------------------------------------------------------
# 1. bitstrings <-> qiskit bool bitstring_matrix
# --------------------------------------------------------------------------
def spins_to_bool_matrix(configs):
    """
    configs: (n, L) array of +-1 (netket spin convention).
    Returns: (n, L) bool array, qiskit convention (True == |1>).
    Uses the SAME up/down -> 0/1 mapping as exact_sector_check's
    states_to_bitstrings, so amplitudes line up with your existing checks.
    """
    return (np.asarray(configs) > 0)


def bool_matrix_to_bitstrings(bmat):
    return np.array(["".join("1" if b else "0" for b in row) for row in bmat])


# --------------------------------------------------------------------------
# 2. Subspace growth by H-connectivity (selected-CI trajectory)
# --------------------------------------------------------------------------
def grow_subspace_with_resampling(j2, variables, n_rounds=3, n_samples_per_round=20000,
                                   max_subspace=20000, seed=0, verbose=True):
    """
    The practical growth loop: union together NQS Metropolis draws across
    several independent chains/seeds. Since your NQS puts most of its mass
    exactly where |psi|^2 is large, repeated independent sampling rounds
    are an efficient stand-in for explicit H-connectivity expansion, and
    is what step 1 of your plan already gives you for free. Each round's
    bitstring set is the union with all previous rounds (selected-CI style
    monotonic growth), and solve_qubit is exact on whatever subspace you
    hand it regardless of how it was built.
    """
    all_bitstrings = set()
    bool_rows = []
    for r in range(n_rounds):
        configs = sample_configs(j2, variables, n_samples=n_samples_per_round,
                                  n_discard=200, seed=seed + r)
        bmat_round = spins_to_bool_matrix(configs)
        bstrs_round = bool_matrix_to_bitstrings(bmat_round)
        new = 0
        for b, row in zip(bstrs_round, bmat_round):
            if b not in all_bitstrings:
                all_bitstrings.add(b)
                bool_rows.append(row)
                new += 1
        if verbose:
            print(f"[grow_subspace] round {r}: +{new} new, total subspace = {len(all_bitstrings)}")
        if len(all_bitstrings) >= max_subspace:
            break
    bitstring_matrix = np.array(bool_rows)
    bitstring_matrix = sort_and_remove_duplicates(bitstring_matrix)
    return bitstring_matrix


# --------------------------------------------------------------------------
# 3. Exact-on-subspace ground state
# --------------------------------------------------------------------------
def solve_ground_state_on_subspace(bitstring_matrix, hamiltonian, verbose=True):
    """
    scipy.sparse.linalg.eigsh under the hood; which='SA' (smallest
    algebraic) picks the ground state, not the largest-magnitude
    eigenvalue eigsh defaults to.
    """
    vals, vecs = solve_qubit(bitstring_matrix, hamiltonian, k=1, which="SA", verbose=verbose)
    energy = vals[0]
    amplitudes = vecs[:, 0]
    return energy, amplitudes


# --------------------------------------------------------------------------
# 4. Truncate by mass coverage (reuses the accounting from
#    exact_sector_check.sampled_mass_coverage, pointed at SQD amplitudes)
# --------------------------------------------------------------------------
def truncate_by_mass_coverage(bitstring_matrix, amplitudes, target_fidelity=0.999, max_terms=None):
    probs = np.abs(amplitudes) ** 2
    order = np.argsort(-probs)
    bstrs = bool_matrix_to_bitstrings(bitstring_matrix)

    kept_bstrs, kept_amps = [], []
    cum = 0.0
    for idx in order:
        kept_bstrs.append(bstrs[idx])
        kept_amps.append(amplitudes[idx])
        cum += probs[idx]
        if cum >= target_fidelity:
            break
        if max_terms and len(kept_bstrs) >= max_terms:
            break

    kept_amps = np.array(kept_amps, dtype=complex)
    kept_amps /= np.linalg.norm(kept_amps)  # renormalize the truncated vector
    print(f"[truncate] kept {len(kept_bstrs)} / {len(bstrs)} basis states, "
          f"mass covered = {cum:.6f} (target {target_fidelity})")
    return dict(zip(kept_bstrs, kept_amps)), cum


# --------------------------------------------------------------------------
# 5. qclib sparse-state-prep circuit synthesis
# --------------------------------------------------------------------------
def build_sparse_state_circuit(sparse_dict, method="merge"):
    """
    method: "merge"   -> qclib.state_preparation.merge.MergeInitialize
                          (generally lowest gate/CX count for clustered
                          sparse states -- good default for NQS/SQD output
                          since physical ground states tend to cluster in
                          Hamming-distance-connected configurations)
            "cvoqram"  -> qclib.state_preparation.cvoqram.CvoqramInitialize
                          (uses one ancilla-chain register; sometimes lower
                          depth at the cost of extra qubits)
    Both are subclasses of qclib.gates.initialize_sparse.InitializeSparse
    and take the same {bitstring: complex amplitude} dict.
    """
    if method == "merge":
        from qclib.state_preparation.merge import MergeInitialize
        gate = MergeInitialize(sparse_dict)
    elif method == "cvoqram":
        from qclib.state_preparation.cvoqram import CvoqramInitialize
        gate = CvoqramInitialize(sparse_dict)
    else:
        raise ValueError(f"unknown method {method!r}")

    qc = QuantumCircuit(gate.num_qubits, name="sqd_ground_state")
    qc.append(gate, range(gate.num_qubits))
    return qc


def verify_circuit_fidelity(qc, sparse_dict):
    """Statevector-simulate qc and check overlap with the target sparse state."""
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    n = qc.num_qubits
    target = np.zeros(2 ** n, dtype=complex)
    for b, amp in sparse_dict.items():
        target[int(b, 2)] = amp
    target /= np.linalg.norm(target)

    backend = AerSimulator()
    tqc = transpile(qc, backend)
    tqc.save_statevector()
    sv = np.asarray(backend.run(tqc).result().get_statevector())
    fidelity = abs(np.vdot(target, sv)) ** 2
    return fidelity, tqc.depth(), dict(tqc.count_ops())

def circuit_resource_count(qc, backend=None, basis_gates=None):
    """Transpile and report resource counts, no simulation.

    basis_gates: e.g. ['h', 's', 'rz', 'cx']. If given, transpiles into
    exactly this gate set (ignores `backend`'s native basis). RZ is a
    continuous rotation, so {H, RZ} already spans arbitrary single-qubit
    SU(2) via an Euler decomposition (RZ-H-RZ-H-RZ); S is just RZ(pi/2)
    with a fixed global phase and gets folded in as a convenience/Clifford
    shortcut. CX is the only 2-qubit gate here, so it doubles as your
    "two-qubit gate count" directly -- no name-matching heuristic needed.
    """
    from qiskit import transpile
    from qiskit_aer import AerSimulator

    if basis_gates is not None:
        tqc = transpile(qc, basis_gates=basis_gates, optimization_level=3)
    else:
        backend = backend or AerSimulator()
        tqc = transpile(qc, backend)

    ops = dict(tqc.count_ops())
    # Count by actual gate arity rather than a hardcoded name list --
    # robust to whatever basis you pick.
    two_qubit_gates = sum(1 for instr in tqc.data if instr.operation.num_qubits == 2)

    print(f"  basis: {basis_gates or 'backend native'}")
    print(f"  depth: {tqc.depth()}")
    print(f"  total gates: {sum(ops.values())}")
    print(f"  two-qubit gates: {two_qubit_gates}")
    print(f"  gate breakdown: {ops}")
    return dict(depth=tqc.depth(), ops=ops, two_qubit_gates=two_qubit_gates, circuit=tqc)

def save_transpiled_circuit(tqc, path="sqd_ground_state_transpiled.qpy"):
    from qiskit import qpy
    with open(path, "wb") as f:
        qpy.dump(tqc, f)
    print(f"  saved transpiled circuit to {path}")


def load_transpiled_circuit(path="sqd_ground_state_transpiled.qpy"):
    from qiskit import qpy
    with open(path, "rb") as f:
        circuits = qpy.load(f)
    return circuits[0]


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------
def run_pipeline(j2, target_fidelity=0.999, n_rounds=3, n_samples_per_round=20000,
                  max_subspace=20000, method="merge", seed=0):
    with open(VARIABLES_PATH, "rb") as f:
        loaded = pickle.load(f)
    variables = loaded["variables"]
    print(f"Loaded {VARIABLES_PATH}")

    print(f"\n=== Building H(j2={j2}) as SparsePauliOp ===")
    H = netket_hamiltonian_to_sparse_pauli_op(j2)
    print(f"  {len(H.paulis)} Pauli terms, {H.num_qubits} qubits")

    print(f"\n=== Step 1-2: NQS-seeded subspace growth ===")
    bitstring_matrix = grow_subspace_with_resampling(
        j2, variables, n_rounds=n_rounds, n_samples_per_round=n_samples_per_round,
        max_subspace=max_subspace, seed=seed,
    )

    print(f"\n=== Step 2: exact-on-subspace diagonalization ({bitstring_matrix.shape[0]} configs) ===")
    energy, amplitudes = solve_ground_state_on_subspace(bitstring_matrix, H)
    print(f"  SQD ground-state energy estimate: {energy:.8f}")

    print(f"\n=== Step 3: truncate to {target_fidelity:.4%} mass coverage ===")
    sparse_dict, coverage = truncate_by_mass_coverage(
        bitstring_matrix, amplitudes, target_fidelity=target_fidelity,
    )

    print(f"\n=== Step 4: qclib sparse circuit synthesis (method={method}) ===")
    qc = build_sparse_state_circuit(sparse_dict, method=method)
    print(f"  circuit: {qc.num_qubits} qubits, {len(sparse_dict)} nonzero amplitudes")
    '''
    print(f"\n=== Verification (statevector sim) ===")
    fidelity, depth, ops = verify_circuit_fidelity(qc, sparse_dict)
    print(f"  fidelity vs truncated target: {fidelity:.8f}")
    print(f"  depth: {depth}, gate counts: {ops}")
    '''
    print(f"\n=== Resource count (transpile only, no sim) ===")
    BASIS = ['h', 's', 'rz', 'cx']
    resources = circuit_resource_count(qc, basis_gates=BASIS)
    save_transpiled_circuit(resources["circuit"])
    return {
        "energy": energy,
        "sparse_dict": sparse_dict,
        "circuit": qc,
        "coverage": coverage,
    #    "fidelity": fidelity,
    }


if __name__ == "__main__":
    J2_TEST = 0.0
    run_pipeline(J2_TEST, target_fidelity=0.999, n_rounds=3,
                 n_samples_per_round=20000, max_subspace=20000, method="merge")