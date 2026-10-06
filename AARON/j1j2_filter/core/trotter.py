"""Trotter error: commutator bounds, total-error bound, ED checker."""


import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

from core.circuits import evolution_gate
from core.filter_design import certify_eta, filter_values


def _strip_identity(H):
    keep = [i for i, lab in enumerate(H.paulis.to_labels()) if set(lab) != {"I"}]
    return SparsePauliOp(H.paulis[keep], H.coeffs[keep])


def alpha_comm(H, tight=False, reverse=False):
    """alpha = sum_g || [H_g, sum_{g'>g} H_g'] || over the Pauli terms in stored
    order (reverse=True reverses that order). Identity terms are dropped
    (they commute with everything).
    tight=False: triangle inequality, sum_{i<j, noncommuting} 2|c_i||c_j|.
    tight=True:  exact spectral norm of each grouped commutator."""
    H = _strip_identity(H)
    if reverse:
        H = SparsePauliOp(H.paulis[::-1], H.coeffs[::-1])
    P, c = H.paulis, np.abs(H.coeffs)
    n = len(P)
    if not tight:
        return float(sum(2 * c[i] * c[j] for i in range(n)
                         for j in range(i + 1, n) if not P[i].commutes(P[j])))
    a = 0.0
    for i in range(n - 1):
        A = SparsePauliOp(P[i], [H.coeffs[i]])
        B = SparsePauliOp(P[i + 1:], H.coeffs[i + 1:])
        C = (1j * (A.dot(B) - B.dot(A))).simplify()      # Hermitian
        if np.allclose(C.coeffs, 0):
            continue
        if H.num_qubits <= 8:
            a += float(np.max(np.abs(np.linalg.eigvalsh(C.to_matrix()))))
        else:
            a += abs(sla.eigsh(C.to_matrix(sparse=True), k=1, which="LM",
                               return_eigenvectors=False)[0])
    return a


def trotter_alpha(H, tight=True):
    """Safe alpha: max over both term orders, so the bound holds whichever
    ordering convention the commutator theorem uses for the synthesized
    product. lie_trotter_order() tells you which one is actually realized."""
    return max(alpha_comm(H, tight, reverse=False),
               alpha_comm(H, tight, reverse=True))


def lie_trotter_order(H, tau=0.05, reps_list=(1, 3), tol=1e-9):
    N = H.num_qubits

    mats = [
        SparsePauliOp(H.paulis[i], [H.coeffs[i]]).to_matrix()
        for i in range(len(H))
    ]

    fwd_ok = rev_ok = True
    worst_f = worst_r = 0.0

    for reps in reps_list:
        t = tau * reps
        dt = t / reps

        U_f = np.eye(2**N, dtype=complex)
        U_r = np.eye(2**N, dtype=complex)

        for m in mats:
            s = expm(-1j * dt * m)
            U_f = s @ U_f
            U_r = U_r @ s

        U_f = np.linalg.matrix_power(U_f, reps)
        U_r = np.linalg.matrix_power(U_r, reps)

        # Build PauliEvolutionGate
        gate = evolution_gate(H, t, reps, 1)

        qc = QuantumCircuit(N)
        qc.append(gate, range(N))

        # IMPORTANT: synthesize/decompose the PauliEvolutionGate
        qc = qc.decompose(reps=10)

        U = Operator(qc).data

        ef = np.linalg.norm(U - U_f, 2)
        er = np.linalg.norm(U - U_r, 2)

        worst_f = max(worst_f, ef)
        worst_r = max(worst_r, er)

        fwd_ok &= ef < tol
        rev_ok &= er < tol

        print(
            f"reps={reps}: "
            f"forward={ef:.3e}, reverse={er:.3e}"
        )

    return {
        "forward": bool(fwd_ok),
        "reverse": bool(rev_ok),
        "err_forward": float(worst_f),
        "err_reverse": float(worst_r),
    }


def trotter_bounds(alpha, tg, k, p_g):
    """(R3) Rigorous post-selected-state bound.

    Setup: exact unnormalized vector a = A_m ... A_1 psi with A_i = <0|W_i|0>
    = cos(H t_i + phi_i); Trotterized b with A~_i = <0|W~_i|0>. Only the
    exp(-i t H(x)Z) factor differs, and [H_g(x)Z, H_g'(x)Z] = [H_g, H_g'](x)1,
    so ||W~_i - W_i|| <= alpha t_i^2 / (2 k_i) =: eps_i (same alpha for H(x)Z).
    All A_i, A~_i are blocks of unitaries (norm <= 1), so telescoping gives
    ||a - b|| <= sum_i eps_i = eps. On the uniform grid t_i = k_i dt,
    sum k_i = n: eps = alpha T^2 / (2n).
    Normalization: ||a/|a| - b/|b|| <= 2 ||a - b|| / ||a||, ||a|| = sqrt(p_g).
    Hence state distance <= 2 eps / sqrt(p_g). Valid when `alpha` is for the
    realized term order (trotter_alpha takes the max over both orders)."""
    tg = np.asarray(tg, dtype=float)
    k = np.asarray(k, dtype=float)
    total = float(np.sum(alpha * tg ** 2 / (2.0 * k)))
    raw = 2.0 * total / np.sqrt(p_g)
    return dict(bound_total=total, bound_state_raw=raw,
                bound_state=min(2.0, raw))


def total_error_bound(gamma, delta, alpha, times, phases, k, hi=1.0,
                      e0_slack=0.0, eta=None):
    """Upper bound on 1 - F_total from (gamma, delta, alpha, design).

    k : per-pulse Trotter step counts (array, same length as times).
    eta : optionally pass an already-certified eta (skips the certificate)."""
    times = np.asarray(times, float)
    k = np.asarray(k, float)
    gamma = min(float(gamma), 1.0)
    cert = certify_eta(times, phases, delta, hi, e0_slack)
    if eta is None:
        eta = cert["eta"]

    # leakage (exact filter)
    if gamma >= 1.0:
        leak = 0.0
    elif not np.isfinite(eta):
        leak = 1.0
    else:
        leak = (1 - gamma) * eta ** 2 / (gamma + (1 - gamma) * eta ** 2)

    # Trotter (post-selected state distance)
    p_g = gamma * cert["f0_lb"] ** 2
    eps_T = float(np.sum(alpha * times ** 2 / (2.0 * k)))
    d_T = 2.0 if p_g <= 0 else min(2.0, 2.0 * eps_T / np.sqrt(p_g))

    total = min(1.0, (np.sqrt(2.0 * leak) + d_T) ** 2)
    return dict(eps_bound=total, leak=leak, trotter_dist=d_T, eps_T=eps_T,
                eta=eta, f0=cert["f0"], p_g_lb=p_g, gamma=gamma)


def steps_needed(alpha, T, p_g, d_T_target):
    """Total Trotter steps n so that the Trotter distance <= d_T_target."""
    return int(np.ceil(alpha * T ** 2 / (np.sqrt(p_g) * d_T_target)))


def alpha_triangle(H):
    """Unscaled alpha = sum_{i<j, noncommuting} 2|c_i||c_j| (identity dropped).
    Divide by W^2 for the scaled value."""
    keep = [i for i, lab in enumerate(H.paulis.to_labels()) if set(lab) != {"I"}]
    P, c = H.paulis[keep], np.abs(H.coeffs[keep])
    n = len(P)
    return float(sum(2 * c[i] * c[j] for i in range(n) for j in range(i + 1, n)
                     if not P[i].commutes(P[j])))


def _trotter_run(Hs_op, psi, times, phases, k):
    """Dense/sparse Lie-Trotter simulation of the post-selected filter.
    Ancilla = MSB. Returns (system vector, success probability)."""
    N = Hs_op.num_qubits
    dim = 2 ** N
    terms = [(sp.csr_matrix(SparsePauliOp("Z" + lab).to_matrix(sparse=True)),
              float(np.real(c)))
             for lab, c in zip(Hs_op.paulis.to_labels(), Hs_op.coeffs)]
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    state = np.asarray(psi, dtype=complex)
    p_succ = 1.0
    for t_i, phi_i, k_i in zip(times, phases, k):
        s = np.stack([state, np.zeros(dim, dtype=complex)])
        rz = np.diag([np.exp(-1j * phi_i), np.exp(1j * phi_i)])
        flat = ((rz @ h) @ s).reshape(-1)
        for _ in range(int(k_i)):
            for P, c in terms:
                th = c * t_i / k_i
                flat = np.cos(th) * flat - 1j * np.sin(th) * (P @ flat)
        a0 = (h @ flat.reshape(2, dim))[0]
        p = float(np.vdot(a0, a0).real)
        p_succ *= p
        state = a0 / np.sqrt(p)
    return state, p_succ


def ed_error(H, psi, times, phases, k, alpha=None, simulate_trotter=True,
             shift=None, W=None, e0_slack=0.0):
    """Exact-diagonalization error report for a design.

    H   : unscaled SparsePauliOp (N <= ~14).   psi : trial vector (qiskit order).
    shift, W : scaling H_s = (H - shift)/W the design was built for (e.g. the
        pipeline's DMRG E0 and certified W). Default: ED E0 and E_top - E0.
    Returns the bound (computed from ED-exact gamma and gap) next to the
    actual infidelity with the exact filter and with the Trotterized circuit."""
    times = np.asarray(times, float)
    phases = np.asarray(phases, float)
    k = np.asarray(k, int)
    psi = np.asarray(psi, dtype=complex)
    psi = psi / np.linalg.norm(psi)

    evals, V = np.linalg.eigh(H.to_matrix())
    E0 = float(evals[0])
    shift = E0 if shift is None else float(shift)
    W = float(evals[-1] - evals[0]) if W is None else float(W)
    Es = (evals - shift) / W
    if Es[-1] > 1.0 + 1e-9:
        raise ValueError(f"scaled spectrum reaches {Es[-1]:.6f} > 1: W too small")
    if evals[1] - evals[0] < 1e-9:
        raise ValueError("degenerate ground state: fidelity bound not applicable")
    delta = float((evals[1] - evals[0]) / W)

    c = V.conj().T @ psi
    gamma = float(abs(c[0]) ** 2)
    amp = c * filter_values(times, phases, Es)
    p_exact = float(np.sum(np.abs(amp) ** 2))
    eps_exact = 1.0 - float(abs(amp[0]) ** 2) / p_exact

    if alpha is None:
        alpha = alpha_triangle(H) / W ** 2
    b = total_error_bound(gamma, delta, alpha, times, phases, k, hi=1.0,
                          e0_slack=e0_slack)

    out = dict(eps_bound=b["eps_bound"], eps_exact_filter=eps_exact,
               eps_trotter=None, gamma=gamma, delta=delta, W=W, E0=E0,
               alpha=alpha, shift=shift, eta=b["eta"], leak_bound=b["leak"],
               trotter_dist_bound=b["trotter_dist"],
               p_succ_exact=p_exact, p_succ_trotter=None, bound_holds=None)

    if simulate_trotter:
        N = H.num_qubits
        Hs = (H * (1.0 / W) + SparsePauliOp("I" * N, [-shift / W])).simplify(atol=0)
        # simplify may drop/merge terms; identity (ancilla Z rotation) is kept
        vec, p_t = _trotter_run(Hs, psi, times, phases, k)
        eps_t = 1.0 - float(abs(np.vdot(V[:, 0], vec)) ** 2)
        out.update(eps_trotter=eps_t, p_succ_trotter=p_t,
                   bound_holds=bool(eps_t <= b["eps_bound"] + 1e-12))
    return out
