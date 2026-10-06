"""E7a: where does the 100-1000x slack of the Trotter bound come from?
Compare the worst-case constant alpha = sum_g ||[H_g, sum_{g'>g} H_g']|| with the first-order error coefficient actually
felt by (a) the ground state and (b) the trial state:  ||C psi||, C = sum_{g<g'} [H_g', H_g]  (S1(t)psi - e^{-itH}psi ~ (t^2/2) C psi)."""
from common import *
import numpy as np, io, contextlib
from qiskit.quantum_info import SparsePauliOp
from hamiltonians import j1j2_hamiltonian
import core.trotter as tr
out = []
for J2 in (0.0, 0.4):
    for N in (4, 6, 8, 10):
        H = j1j2_hamiltonian(N, 1.0, J2)
        Hm = H.to_matrix(sparse=True)
        from scipy.sparse.linalg import eigsh
        w, v = eigsh(Hm, k=3, which="SA"); o = np.argsort(w); w, v = w[o], v[:, o]
        wt = eigsh(Hm, k=1, which="LA", return_eigenvectors=False)[0]
        W = wt - w[0]
        mats = [SparsePauliOp(H.paulis[i], [H.coeffs[i]]).to_matrix(sparse=True) for i in range(len(H))]
        psi0 = v[:, 0]
        # C psi = sum_{g<g'} [H_g', H_g] psi  computed with running sum:  sum_g' (H_g' H_g - H_g H_g') psi  for g'>g
        def Cpsi(psi):
            acc = np.zeros_like(psi, dtype=complex)
            tail = [None] * len(mats)
            suffix = np.zeros_like(psi, dtype=complex)           # sum_{g'>g} H_g' psi
            sfx_list = []
            for g in range(len(mats) - 1, -1, -1):
                sfx_list.append(suffix.copy()); suffix = suffix + mats[g] @ psi
            sfx_list = sfx_list[::-1]                              # sfx_list[g] = sum_{g'>g} H_g' psi
            # sum_g [H_g', H_g]-type: sum_g ( (sum_{g'>g} H_g') H_g psi - H_g (sum_{g'>g} H_g') psi )
            Ssum = [sum(mats[g + 1:]) if g < len(mats) - 1 else None for g in range(len(mats))]
            for g in range(len(mats) - 1):
                acc += Ssum[g] @ (mats[g] @ psi) - mats[g] @ (Ssum[g] @ psi)
            return acc
        a = tr.alpha_comm(H, True, False) / W ** 2
        c0 = np.linalg.norm(Cpsi(psi0)) / W ** 2
        # first excited (triplet) and a mid-spectrum-ish random low-energy mix for context
        c1 = np.linalg.norm(Cpsi(v[:, 1])) / W ** 2
        row = dict(N=N, J2=J2, alpha_scaled=a, Cpsi0=c0, Cpsi_E1=c1, ratio_gs=a / max(c0, 1e-300), ratio_E1=a / max(c1, 1e-300))
        out.append(row); print({k: (round(x, 4) if isinstance(x, float) else x) for k, x in row.items()}, flush=True)
save("exp7a_looseness", out)
