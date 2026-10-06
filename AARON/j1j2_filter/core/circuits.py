"""Pulse / filter circuits and the convention check."""


import numpy as np
import scipy.sparse as sp
from hamiltonians import j1j2_hamiltonian
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.circuit import ControlFlowOp, IfElseOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, MatrixExponential, SuzukiTrotter
from scipy.linalg import expm

from core.simulate import apply_pulse_exact


def h_tensor_z(H):
    """H (x) Z_anc, ancilla = top qubit (leftmost label), term order preserved.
    exp(-i t H(x)Z) = e^{-iHt} on anc=0, e^{+iHt} on anc=1."""
    return SparsePauliOp.from_list(
        [("Z" + lab, c) for lab, c in zip(H.paulis.to_labels(), H.coeffs)])


def evolution_gate(H, t, trotter_steps=1, order=1):
    """exp(-i H t). order=0: exact (matrix exponential; verification only),
    order=1: LieTrotter, else SuzukiTrotter. Term order is preserved."""
    if order == 0:
        synth = MatrixExponential()
    elif order == 1:
        synth = LieTrotter(reps=trotter_steps, preserve_order=True)
    else:
        synth = SuzukiTrotter(order=order, reps=trotter_steps,
                              preserve_order=True)
    return PauliEvolutionGate(H, time=t, synthesis=synth)


def apply_filter_pulse(qc, sys_qubits, anc, H, t_i, phi_i, trotter_steps=1,
                       order=1):
    """One pulse. Ancilla must enter in |0>; the 0 branch applies
    cos(H t_i + phi_i) to the system, the 1 branch -i sin(H t_i + phi_i)."""
    qc.h(anc)
    qc.rz(2 * phi_i, anc)
    qc.append(evolution_gate(h_tensor_z(H), t_i, trotter_steps, order),
              [*sys_qubits, anc])
    qc.h(anc)


def _steps_per_pulse(trotter_steps, n_pulses):
    if np.isscalar(trotter_steps):
        return [int(trotter_steps)] * n_pulses
    steps = [int(s) for s in trotter_steps]
    if len(steps) != n_pulses:
        raise ValueError(f"need {n_pulses} step counts, got {len(steps)}")
    return steps


def build_filter_circuit(H, times, phases, trial_prep=None, trotter_steps=1,
                         order=1):
    """Full filter, measure + reset ancilla after each pulse (no early abort)."""
    N = H.num_qubits
    sys_reg, anc = QuantumRegister(N, "sys"), QuantumRegister(1, "anc")
    creg = ClassicalRegister(len(times), "anc_meas")
    qc = QuantumCircuit(sys_reg, anc, creg)
    steps = _steps_per_pulse(trotter_steps, len(times))
    if trial_prep is not None:
        trial_prep(qc, sys_reg)
    for k, (t_i, phi_i) in enumerate(zip(times, phases)):
        apply_filter_pulse(qc, sys_reg, anc[0], H, t_i, phi_i, steps[k], order)
        qc.measure(anc[0], creg[k])
        qc.reset(anc[0])
    return qc


def build_early_abort_circuit(H, times, phases, trial_prep=None,
                              trotter_steps=1, order=1):
    """Deterministic-time cosine filter with a single ancilla, mid-circuit
    measure + reset, and the remaining pulses skipped as soon as the ancilla
    reads 1 (needs dynamic circuits / if_test). (A14: renamed from the
    'rodeo' circuit -- the rodeo algorithm uses random times; here the times
    are the optimized deterministic ones.) trotter_steps: int or per-pulse list."""
    N = H.num_qubits
    sys_reg, anc = QuantumRegister(N, "sys"), QuantumRegister(1, "anc")
    creg = ClassicalRegister(len(times), "cycle")
    qc = QuantumCircuit(sys_reg, anc, creg)
    steps = _steps_per_pulse(trotter_steps, len(times))
    if trial_prep is not None:
        trial_prep(qc, sys_reg)

    def recurse(k):
        if k == len(times):
            return
        apply_filter_pulse(qc, sys_reg, anc[0], H, times[k], phases[k],
                           steps[k], order)
        qc.measure(anc[0], creg[k])
        qc.reset(anc[0])
        with qc.if_test((creg[k], 0)):
            recurse(k + 1)

    recurse(0)
    return qc, creg


def flatten_success_path(circ):
    """Inline every IfElseOp's true-body, leaving the all-zeros path."""
    flat = QuantumCircuit(list(circ.qubits), list(circ.clbits))
    for inst in circ.data:
        op = inst.operation
        if isinstance(op, IfElseOp):
            body = flatten_success_path(op.blocks[0])
            flat.compose(body, qubits=list(inst.qubits),
                         clbits=list(inst.clbits), inplace=True)
        elif isinstance(op, ControlFlowOp):
            raise NotImplementedError(f"unhandled control flow: {op.name}")
        else:
            flat.append(op, inst.qubits, inst.clbits)
    return flat


def verify_pulse_convention(seed=7, atol=1e-8, verbose=True):
    """Single pulse on N=3 with a NONCOMMUTING, field-carrying, randomly ordered
    H and a random COMPLEX state. Uses the exact (unsynthesized) evolution so
    the check isolates the pulse convention from Trotter error. Checks, up to
    one global phase: anc=0 block = cos(Ht+phi) psi, anc=1 block =
    -i sin(Ht+phi) psi; the numpy block pulse and the classical cosine filter
    agree; and the flipped sign phi -> -phi is REJECTED (test has power).
    Returns True/False; callers should treat False as fatal."""
    rng = np.random.default_rng(seed)
    N, dim = 3, 8
    base = j1j2_hamiltonian(N, 1.0, 0.7, fields=[0.3, 0.6, 0.9], order="bond")
    perm = rng.permutation(len(base))
    H = SparsePauliOp(base.paulis[perm], base.coeffs[perm])
    Hm = H.to_matrix()
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    t_i, phi_i = 0.83, 0.37

    def expected(phi):
        X = t_i * Hm + phi * np.eye(dim)
        c = (expm(1j * X) + expm(-1j * X)) / 2
        s = (expm(1j * X) - expm(-1j * X)) / 2j
        return np.concatenate([c @ psi, -1j * (s @ psi)])

    def aligned_err(got, exp_):
        th = np.angle(np.vdot(exp_, got))
        return float(np.linalg.norm(got - np.exp(1j * th) * exp_))

    qc = QuantumCircuit(N + 1)
    qc.prepare_state(psi, list(range(N)))
    apply_filter_pulse(qc, list(range(N)), N, H, t_i, phi_i, order=0)
    got = Statevector(qc).data
    err_pos = aligned_err(got, expected(phi_i))
    err_flip = aligned_err(got, expected(-phi_i))

    s_np = apply_pulse_exact(np.concatenate([psi, np.zeros(dim)]),
                             sp.csr_matrix(Hm), t_i, phi_i)
    err_np = aligned_err(s_np, expected(phi_i))

    w, V = np.linalg.eigh(Hm)
    cl = V @ (np.cos(w * t_i + phi_i) * (V.conj().T @ psi))
    err_cl = float(np.linalg.norm(cl - expected(phi_i)[:dim]))

    ok = (err_pos < atol and err_np < atol and err_cl < atol and err_flip > 1e-3)
    if verbose:
        print(f"pulse convention: circuit err={err_pos:.2e}  numpy err={err_np:.2e}  "
              f"classical err={err_cl:.2e}  flipped-phi err={err_flip:.2e} "
              f"(must be large)  ->  {'OK' if ok else 'FAIL'}")
    return bool(ok)
