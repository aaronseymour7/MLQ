"""
DMRG energies/states (E0, E1, E_top) with an ED cross-check.

Changes from the audited version
--------------------------------
* ED Hamiltonian is built from the PAULI form (hamiltonians.j1j2_hamiltonian),
  so it is independent of the MPO used by DMRG (audit A7); the old
  `build_sparse_j1j2` (same MPO, dense) is removed.
* ED_CHECK_MAX_N = 12 for DENSE ED (2^24 dense was impossible); sparse Lanczos
  ED is used up to ED_SPARSE_MAX_N (audit A12).
* ED eigenvectors are returned in MPS site order (axis-reversed), as before.
* Callers should call get_energies ONCE and pass the result around (A9).
"""
import numpy as np
import quimb.tensor as qtn
from scipy.sparse.linalg import eigsh
from qiskit import transpile
from qiskit.quantum_info import Statevector

from hamiltonians import MPO_ham_j1j2, j1j2_hamiltonian, reorder_axes
from mps_to_circuit import mps_to_circuit

BOND_DIMS = [8, 16, 32, 64, 64]
CUTOFF = 1e-10
MAX_SWEEPS = 60
TOL = 1e-10
ED_CHECK_MAX_N = 12      # dense eigh
ED_SPARSE_MAX_N = 22     # Lanczos (low end + high end)


def build_projector_mpo(mps):
    n = mps.L
    arrays = mps.arrays
    new_arrays = []
    for i in range(n):
        A = arrays[i]
        Ac = np.conj(A)
        if i == 0:
            p, r = A.shape
            T = np.einsum('pr,qR->rRpq', A, Ac).reshape(r * r, p, p)
        elif i == n - 1:
            l, p = A.shape
            T = np.einsum('lp,Lq->lLpq', A, Ac).reshape(l * l, p, p)
        else:
            l, p, r = A.shape
            T = np.einsum('lpr,LqR->lLrRpq', A, Ac).reshape(l * l, r * r, p, p)
        new_arrays.append(T)
    return qtn.MatrixProductOperator(new_arrays, shape='lrud')


def run_dmrg(mpo, bond_dims=BOND_DIMS, cutoff=CUTOFF, tol=TOL,
             max_sweeps=MAX_SWEEPS, verbosity=0):
    dmrg = qtn.DMRG2(mpo, bond_dims=bond_dims, cutoffs=cutoff)
    converged = dmrg.solve(tol=tol, max_sweeps=max_sweeps, verbosity=verbosity)
    if not converged:
        print(f"  [warning] DMRG did not report convergence within {max_sweeps} sweeps")
    return dmrg.energy, dmrg.state


def ed_ground_and_excited(N, j1=1.0, j2=0.0, want_states=True):
    """Exact E0, E1, E_top from the PAULI Hamiltonian (independent of the MPO).
    States are returned in MPS site order."""
    if N > ED_SPARSE_MAX_N:
        raise ValueError(f"N={N} > ED_SPARSE_MAX_N={ED_SPARSE_MAX_N}")
    Hs = j1j2_hamiltonian(N, j1=j1, j2=j2).to_matrix(sparse=True)
    if N <= ED_CHECK_MAX_N:
        w, v = np.linalg.eigh(Hs.toarray())
        lo_w, lo_v = w[:3], v[:, :3]
        top_w, top_v = w[-1], v[:, -1]
    else:
        lw, lv = eigsh(Hs, k=3, which="SA")
        o = np.argsort(lw)
        lo_w, lo_v = lw[o], lv[:, o]
        tw, tv = eigsh(Hs, k=1, which="LA")
        top_w, top_v = tw[0], tv[:, 0]
    out = dict(E0_ed=float(lo_w[0]), E1_ed=float(lo_w[1]), E_top_ed=float(top_w))
    if want_states:
        out["psi0_ed"] = reorder_axes(lo_v[:, 0])
        out["psi1_ed"] = reorder_axes(lo_v[:, 1])
        out["psitop_ed"] = reorder_axes(top_v)
    return out


def ed_cross_check(N, j1=1.0, j2=0.0, E0_dmrg=None, E1_dmrg=None, Etop_dmrg=None,
                   rtol=1e-6, atol=1e-8, want_states=True):
    if N > ED_SPARSE_MAX_N:
        raise ValueError(f"N={N} exceeds ED_SPARSE_MAX_N={ED_SPARSE_MAX_N}; ED skipped.")
    print(f"Running ED cross-check (N={N}, dim=2^{N}={2**N})...")
    ed = ed_ground_and_excited(N, j1, j2, want_states=want_states)

    def _cmp(name, dmrg_val, ed_val):
        if dmrg_val is None:
            return None
        diff = abs(dmrg_val - ed_val)
        ok = diff <= atol + rtol * abs(ed_val)
        print(f"  [{'OK' if ok else 'MISMATCH'}] {name}: DMRG={dmrg_val:.10f}  "
              f"ED={ed_val:.10f}  |diff|={diff:.2e}")
        ed[f"{name}_dmrg"], ed[f"{name}_diff"], ed[f"{name}_ok"] = dmrg_val, diff, ok
        return ok

    checks = [_cmp("E0", E0_dmrg, ed["E0_ed"]),
              _cmp("E1", E1_dmrg, ed["E1_ed"]),
              _cmp("E_top", Etop_dmrg, ed["E_top_ed"])]
    ed["all_ok"] = all(c for c in checks if c is not None)
    if not ed["all_ok"]:
        print("  [warning] DMRG vs ED mismatch -- check bond dims / convergence / penalty.")
    return ed


def get_energies(H, N=None, j1=1.0, j2=0.0, safety_factor=1.1, run_ed_check=True):
    """Run once per configuration and pass the dict around (audit A9)."""
    if H is None:
        H = qtn.MPO_ham_heis(8, j=1, cyclic=False)
    if N is None:
        N = H.L

    print("Running DMRG for ground state...")
    E0, psi0 = run_dmrg(H)
    psi0.normalize()

    print("Running DMRG for highest excited state (via -H)...")
    E_top_neg, psi_top = run_dmrg(-1.0 * H)
    E_top = -E_top_neg

    bandwidth = E_top - E0
    if bandwidth <= 0:
        raise RuntimeError(f"Non-positive bandwidth (E_top={E_top:.6g}, E0={E0:.6g})")
    penalty = safety_factor * bandwidth
    print(f"  spectral bandwidth = {bandwidth:.6g}, PENALTY = {penalty:.6g}")

    P0 = build_projector_mpo(psi0)
    H_exc = H + penalty * P0
    H_exc.compress(cutoff=1e-12)

    print("Running DMRG for first excited state...")
    _, psi1 = run_dmrg(H_exc)
    psi1.normalize()
    E1 = np.real(psi1.H @ H.apply(psi1))

    ed_result = None
    if run_ed_check and N <= ED_SPARSE_MAX_N:
        ed_result = ed_cross_check(N, j1, j2, E0_dmrg=E0, E1_dmrg=E1, Etop_dmrg=E_top)
    elif run_ed_check:
        print(f"  [skip] N={N} > ED_SPARSE_MAX_N; no ED cross-check run.")

    return dict(E0=E0, E1=E1, E_top=E_top, psi0=psi0, psi1=psi1, ed=ed_result)


# ----------------------------------------------------------------------
# Trial-circuit scoring helpers (unchanged logic)
# ----------------------------------------------------------------------
def circuit_fidelity_and_energy(qc, psi0, H, reverse_qubit_order=True,
                                contract_optimize="greedy", psi0_ed=None):
    N = psi0.L
    if qc.num_qubits != N:
        raise ValueError(f"circuit has {qc.num_qubits} qubits, psi0 has {N} sites")
    vec = Statevector(qc).data
    tensor = vec.reshape([2] * N)
    if reverse_qubit_order:
        tensor = np.transpose(tensor, axes=range(N - 1, -1, -1))
    tensor = np.ascontiguousarray(tensor)

    psi0_dense = psi0.to_dense().reshape([2] * N)
    fidelity = float(np.abs(np.vdot(psi0_dense, tensor)) ** 2)

    ket = qtn.Tensor(tensor, inds=H.lower_inds, tags="KET")
    bra = qtn.Tensor(tensor.conj(), inds=H.upper_inds, tags="BRA")
    energy = float(np.real((H & ket & bra).contract(optimize=contract_optimize)))
    out = dict(fidelity=fidelity, energy=energy)
    if psi0_ed is not None:
        out["fidelity_ed"] = float(np.abs(np.vdot(np.asarray(psi0_ed).reshape([2] * N),
                                                  tensor)) ** 2)
    return out


BASIS = ["cx", "rz", "sx", "x"]


def count_gates_optimized(qc, basis_gates=BASIS, optimization_level=3):
    qct = transpile(qc, basis_gates=basis_gates, optimization_level=optimization_level)
    ops = dict(qct.count_ops())
    return dict(cnot=ops.get("cx", 0), rz=ops.get("rz", 0),
                clifford_total=sum(v for k, v in ops.items() if k != "rz"),
                depth=qct.depth(), all_counts=ops)


def resource_and_accuracy_sweep(mps_arrays, psi0, H, layer_range, shape="lpr",
                                psi0_ed=None, E0=None, E0_ed=None, **kwargs):
    if psi0_ed is not None:
        ceil = float(np.abs(np.vdot(np.asarray(psi0_ed).reshape(-1),
                                    psi0.to_dense().reshape(-1))) ** 2)
        print(f"  [baseline] DMRG-vs-ED fidelity ceiling = {ceil:.8f}")
    rows = []
    for L in layer_range:
        qc = mps_to_circuit(mps_arrays, method="approximate", shape=shape,
                            num_layers=L, **kwargs)
        g = count_gates_optimized(qc)
        a = circuit_fidelity_and_energy(qc, psi0, H, psi0_ed=psi0_ed)
        row = dict(num_layers=L, cnot=g["cnot"], rz=g["rz"],
                   fidelity=a["fidelity"], energy=a["energy"])
        line = (f"L={L:2d}  CNOT={g['cnot']:4d}  RZ={g['rz']:4d}  "
                f"fidelity={a['fidelity']:.6f}  E={a['energy']:.6f}")
        if psi0_ed is not None:
            row["fidelity_ed"] = a["fidelity_ed"]
            if E0_ed is not None:
                row["energy_err_ed"] = a["energy"] - E0_ed
            line += f"  fidelity_ed={a['fidelity_ed']:.6f}"
        print(line)
        rows.append(row)
    return rows