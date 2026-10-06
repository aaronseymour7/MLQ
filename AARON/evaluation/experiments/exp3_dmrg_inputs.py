"""E3a: quality of the DMRG-derived spectral inputs (E0, E1 via penalty, E_top) vs exact diagonalisation,
and the certified-bandwidth bound vs true E_top.   usage: exp3_dmrg_inputs.py N_list J2_list"""
from common import *
import sys, time, numpy as np
import io, contextlib
from get_energies import get_energies, ed_ground_and_excited
from hamiltonians import MPO_ham_j1j2, j1j2_hamiltonian, pauli_l1_norm, reorder_axes, energy_variance
Ns = [int(x) for x in sys.argv[1].split(",")]
J2s = [float(x) for x in sys.argv[2].split(",")]
rows = []
for J2 in J2s:
    for N in Ns:
        t0 = time.time()
        H = MPO_ham_j1j2(N, 1.0, J2)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            r = get_energies(H, N=N, j1=1.0, j2=J2, run_ed_check=False)
        ed = ed_ground_and_excited(N, 1.0, J2)
        psi = np.asarray(r["psi0"].to_dense()).reshape(-1); psi /= np.linalg.norm(psi)
        ov0 = abs(np.vdot(ed["psi0_ed"], psi)) ** 2
        psi1 = np.asarray(r["psi1"].to_dense()).reshape(-1); psi1 /= np.linalg.norm(psi1)
        # weight of DMRG "excited" state on exact ground state and exact first excited state
        ov1_0 = abs(np.vdot(ed["psi0_ed"], psi1)) ** 2
        ov1_1 = abs(np.vdot(ed["psi1_ed"], psi1)) ** 2
        Hp = j1j2_hamiltonian(N, 1.0, J2)
        var = energy_variance(Hp.to_matrix(sparse=True), reorder_axes(psi))
        l1 = pauli_l1_norm(Hp)
        # tighter rigorous top-of-spectrum bound: each bond J S.S <= J/4 (J>=0)
        etop_bond = 0.25 * ((N - 1) + (N - 2) * (J2 != 0) * J2) if J2 >= 0 else None
        row = dict(N=N, J2=J2, E0_dmrg=r["E0"], E0_ed=ed["E0_ed"], dE0=r["E0"] - ed["E0_ed"],
                   E1_dmrg=r["E1"], E1_ed=ed["E1_ed"], dE1=r["E1"] - ed["E1_ed"],
                   gap_ed=ed["E1_ed"] - ed["E0_ed"], gap_dmrg=r["E1"] - r["E0"],
                   gap_rel_err=((r["E1"] - r["E0"]) - (ed["E1_ed"] - ed["E0_ed"])) / (ed["E1_ed"] - ed["E0_ed"]),
                   Etop_dmrg=r["E_top"], Etop_ed=ed["E_top_ed"], dEtop=r["E_top"] - ed["E_top_ed"],
                   Etop_cert_l1=l1, Etop_cert_bond=etop_bond, overlap_gs=ov0,
                   psi1_weight_on_ed_gs=ov1_0, psi1_weight_on_ed_E1=ov1_1, var_gs=var, time=time.time() - t0)
        rows.append(row)
        print({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
save(f"exp3a_N{sys.argv[1].replace(',','-')}_J2{sys.argv[2].replace(',','-')}", rows)
