"""Post-selected simulation (exact and synthesized) and state metrics."""


import numpy as np
import scipy.sparse as sp
from hamiltonians import reorder_axes
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from scipy.sparse.linalg import expm_multiply


def postselected_run(trial_qc, times, phases, pulse_fn, synthesize=True):
    """Apply each pulse, project ancilla -> 0, renormalize.
    pulse_fn(qc, sys_qubits, anc, t_i, phi_i, k) appends pulse k to qc.
    synthesize=True transpiles PauliEvolutionGate to real Trotter gates.
    Noiseless. Returns (product of success probabilities, system statevector)."""
    N = trial_qc.num_qubits
    dim = 2 ** N
    sys_q, anc = list(range(N)), N
    sv = Statevector(np.concatenate([Statevector(trial_qc).data,
                                     np.zeros(dim, dtype=complex)]))
    p_succ = 1.0
    for k, (t_i, phi_i) in enumerate(zip(times, phases)):
        pc = QuantumCircuit(N + 1)
        pulse_fn(pc, sys_q, anc, t_i, phi_i, k)
        if synthesize:
            pc = transpile(pc, basis_gates=["h", "s", "x", "cx", "rz"],
                           optimization_level=0)
        sv = sv.evolve(pc)
        proj = np.asarray(sv.data)[:dim]
        p = float(np.vdot(proj, proj).real)
        p_succ *= p
        sv = Statevector(np.concatenate([proj / np.sqrt(p),
                                         np.zeros(dim, dtype=complex)]))
    return p_succ, np.asarray(sv.data)[:dim]


def apply_pulse_exact(state, Hs, t_i, phi_i):
    """One exact pulse on [anc=0 | anc=1] using sparse expm_multiply (A12): no
    dense exponentials. Hs: sparse scaled H."""
    dim = Hs.shape[0]
    s = state.reshape(2, dim)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    rz = np.diag([np.exp(-1j * phi_i), np.exp(1j * phi_i)])
    s = (rz @ h) @ s                                       # H, then Rz(2 phi)
    s = np.stack([expm_multiply(-1j * t_i * Hs, s[0]),     # anc=0: e^{-iHt}
                  expm_multiply(+1j * t_i * Hs, s[1])])    # anc=1: e^{+iHt}
    s = h @ s                                              # final H
    return s.reshape(-1)


def postselected_run_exact(trial_qc, times, phases, Hm):
    """Exact-unitary post-selected run. Hm: scaled H, dense or sparse.
    The Trotter-error-free, noiseless reference. Returns (P_succ, sys vector)."""
    Hs = sp.csr_matrix(Hm)
    dim = Hs.shape[0]
    state = np.concatenate([Statevector(trial_qc).data.astype(complex),
                            np.zeros(dim, dtype=complex)])
    p_succ = 1.0
    for t_i, phi_i in zip(times, phases):
        state = apply_pulse_exact(state, Hs, t_i, phi_i)
        anc0 = state[:dim]
        p = float(np.vdot(anc0, anc0).real)
        p_succ *= p
        state = np.concatenate([anc0 / np.sqrt(p), np.zeros(dim, dtype=complex)])
    return p_succ, state[:dim]


def overlaps(vec_qiskit, psi0_dmrg, psi0_ed=None):
    """(|<DMRG|v>|^2, |<ED|v>|^2) for a qiskit-ordered vector. psi0_* in MPS
    ordering (site 0 = MSB). ED entry is NaN if psi0_ed is None."""
    t = reorder_axes(vec_qiskit)
    fd = float(abs(np.vdot(psi0_dmrg, t)) ** 2)
    fe = (float(abs(np.vdot(psi0_ed, t)) ** 2) if psi0_ed is not None
          else float("nan"))
    return fd, fe


def state_metrics(sys_vec, H_qk, psi0_dmrg, psi0_ed=None):
    """(energy w.r.t. unscaled H_qk, fidelity vs DMRG gs, fidelity vs ED gs)."""
    E = float(np.real(Statevector(sys_vec).expectation_value(H_qk)))
    fd, fe = overlaps(sys_vec, psi0_dmrg, psi0_ed)
    return E, fd, fe
