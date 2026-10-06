"""
hamiltonians.py -- single home for the J1-J2 Hamiltonian builders (audit A7, A13).

Conventions (audit A6)
----------------------
* Qubit q  <->  chain site q.  Qiskit is little-endian (qubit 0 = least
  significant bit), so a qiskit statevector indexed [b_{N-1} ... b_0] must be
  axis-reversed (`reorder_axes`) to get the MPS/quimb ordering (site 0 = most
  significant).  `reorder_axes` is an involution.
* Pauli terms are built with `from_sparse_list` (explicit qubit indices), NOT by
  editing label strings, and `simplify()` is deliberately NOT called so the term
  order is exactly the construction order (the order LieTrotter uses with
  preserve_order=True).
* The rescaled Hamiltonian is built by `scale_hamiltonian`, which appends the
  identity (energy-shift) term LAST; it commutes with everything, so it does
  not affect any commutator bound.
"""
import numpy as np
import quimb as qu
import quimb.tensor as qtn
from qiskit.quantum_info import SparsePauliOp


# ----------------------------------------------------------------------
# Pauli form
# ----------------------------------------------------------------------
def j1j2_hamiltonian(N, j1=1.0, j2=0.5, fields=None, order="bond"):
    """Open-boundary J1-J2 chain, S.S = (XX+YY+ZZ)/4.
    order="bond": X,Y,Z consecutive per bond (each bond exponential is
                  SU(2)-symmetric); r=1 bonds first, then r=2 bonds.
    order="xyz":  three commuting families H_X, H_Y, H_Z (audit E6).
    fields: optional per-site Z fields h_q (used only by tests, audit A6)."""
    bonds = [(r, i, J) for r, J in ((1, j1), (2, j2)) if J != 0
             for i in range(N - r)]
    terms = []
    if order == "bond":
        for r, i, J in bonds:
            for P in "XYZ":
                terms.append((P + P, [i, i + r], 0.25 * J))
    elif order == "xyz":
        for P in "XYZ":
            for r, i, J in bonds:
                terms.append((P + P, [i, i + r], 0.25 * J))
    else:
        raise ValueError("order must be 'bond' or 'xyz'")
    if fields is not None:
        for q, h in enumerate(fields):
            if h != 0:
                terms.append(("Z", [q], float(h)))
    return SparsePauliOp.from_sparse_list(terms, num_qubits=N)


def n_pauli_terms(N, j1=1.0, j2=0.5):
    return 3 * ((N - 1) * (j1 != 0) + (N - 2) * (j2 != 0))


def pauli_l1_norm(H):
    """sum |c_beta| over non-identity terms: a certified bound on ||H - c_I||."""
    return float(sum(abs(c) for lab, c in zip(H.paulis.to_labels(), H.coeffs)
                     if set(lab) != {"I"}))


def scale_hamiltonian(H, shift, W):
    """(H - shift) / W with terms in the ORIGINAL order and the identity last."""
    N = H.num_qubits
    terms = [(lab, float(np.real(c)) / W)
             for lab, c in zip(H.paulis.to_labels(), H.coeffs)]
    terms.append(("I" * N, -float(shift) / W))
    return SparsePauliOp.from_list(terms)


# ----------------------------------------------------------------------
# Ordering helpers
# ----------------------------------------------------------------------
def _bit_reversal(N):
    return np.array([int(format(i, f"0{N}b")[::-1], 2) for i in range(2 ** N)])


def reorder_axes(vec):
    """qiskit (little-endian) <-> MPS (site 0 = MSB) ordering; an involution."""
    vec = np.asarray(vec)
    N = int(round(np.log2(len(vec))))
    return np.ascontiguousarray(
        np.transpose(vec.reshape([2] * N), tuple(range(N - 1, -1, -1)))
    ).reshape(-1)


# ----------------------------------------------------------------------
# MPO form (DMRG input)
# ----------------------------------------------------------------------
def mpo_two_body_heisenberg(N, J, r, S=0.5):
    """MPO for J * sum_i S_i . S_{i+r}, open boundary (bond dim 2 + 3r)."""
    ops = [np.asarray(qu.spin_operator(a, S=S)) for a in ("x", "y", "z")]
    d = ops[0].shape[0]
    I = np.eye(d)
    n_states = 2 + 3 * r
    FIN = n_states - 1

    def idx(a, k):
        return 1 + a * r + (k - 1)

    W = np.zeros((n_states, n_states, d, d), dtype=complex)
    W[0, 0] = I
    W[FIN, FIN] = I
    for a in range(3):
        W[0, idx(a, 1)] = ops[a]
        for k in range(1, r):
            W[idx(a, k), idx(a, k + 1)] = I
        W[idx(a, r), FIN] = J * ops[a]

    arrays = []
    for i in range(N):
        if i == 0:
            arrays.append(W[0, :, :, :])
        elif i == N - 1:
            arrays.append(W[:, FIN, :, :])
        else:
            arrays.append(W)
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def MPO_ham_j1j2(N, j1=1.0, j2=0.5, cyclic=False):
    if cyclic:
        raise NotImplementedError("this builder is open-boundary only")
    H = mpo_two_body_heisenberg(N, j1, r=1) + mpo_two_body_heisenberg(N, j2, r=2)
    H.compress(cutoff=1e-12)
    return H


def check_mpo_matches_pauli(N, j1, j2, atol=1e-9):
    """Audit A7 / test T1: dense MPO (site order) == Pauli matrix after the
    bit-reversal map.  Returns the max abs entry difference (N <= ~10)."""
    Hm = np.asarray(MPO_ham_j1j2(N, j1=j1, j2=j2).to_dense())
    Hp = j1j2_hamiltonian(N, j1, j2).to_matrix()
    perm = _bit_reversal(N)
    err = float(np.max(np.abs(Hm - Hp[np.ix_(perm, perm)])))
    if err > atol:
        raise RuntimeError(f"MPO and Pauli Hamiltonians differ (max |dH|={err:.2e})")
    return err


# ----------------------------------------------------------------------
# Symmetry-resolved gaps and variance (audit A3)
# ----------------------------------------------------------------------
def _total_spin_squared(N):
    def comp(P):
        return SparsePauliOp.from_sparse_list(
            [(P, [q], 0.5) for q in range(N)], num_qubits=N)
    Sx, Sy, Sz = comp("X"), comp("Y"), comp("Z")
    return (Sx.dot(Sx) + Sy.dot(Sy) + Sz.dot(Sz)).simplify()


def symmetry_resolved_gaps(H, tol=1e-6):
    """Dense ED (N <= ~12).  Returns E, S(S+1), reflection parity per level and
    the gap to the lowest level overall and to the lowest level in the same
    (S, parity) sector as the ground state.  Caveat: inside an exactly
    degenerate multiplet S^2/parity expectation values can mix; the ground
    state is assumed nondegenerate."""
    N = H.num_qubits
    w, v = np.linalg.eigh(H.to_matrix())
    S2 = _total_spin_squared(N).to_matrix(sparse=True)
    perm = _bit_reversal(N)
    s2 = np.real((v.conj() * (S2 @ v)).sum(axis=0))
    par = np.real((v.conj() * v[perm, :]).sum(axis=0))
    same = np.where((np.abs(s2[1:] - s2[0]) < tol)
                    & (np.abs(par[1:] - par[0]) < tol))[0]
    return dict(E=w, s2=s2, parity=par, gap_any=float(w[1] - w[0]),
                gap_sector=float(w[1 + same[0]] - w[0]) if len(same) else float("nan"))


def energy_variance(Hsp, vec):
    """<H^2>-<H>^2.  Some eigenvalue lies within sqrt(var) of <H> (rigorous)."""
    v = np.asarray(vec, dtype=complex)
    v = v / np.linalg.norm(v)
    Hv = Hsp @ v
    e = float(np.real(np.vdot(v, Hv)))
    return float(np.real(np.vdot(Hv, Hv))) - e * e