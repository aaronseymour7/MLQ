"""E4: (a) sensitivity of the exact-filter infidelity to an over-estimated gap (the unsafe direction for a DMRG excited-state
energy), (b) how much the S=0 / parity sector structure would buy, (c) trial-state sector weights.
usage: exp4_gap_and_symmetry.py"""
from common import *
import io, contextlib, numpy as np
import floor, core.filter_design as fd
from hamiltonians import j1j2_hamiltonian, symmetry_resolved_gaps, _total_spin_squared, _bit_reversal, reorder_axes, MPO_ham_j1j2
from get_energies import get_energies
from mps_to_circuit import mps_to_circuit
from qiskit.quantum_info import Statevector
out = {}
# ---- (b) sector gaps
rows = []
for N in (6, 8, 10, 12):
    for J2 in (0.0, 0.2411, 0.5):
        Hp = j1j2_hamiltonian(N, 1.0, J2)
        sg = symmetry_resolved_gaps(Hp)
        w = sg["E"]; W = w[-1] - w[0]
        rows.append(dict(N=N, J2=J2, gap_any=sg["gap_any"], gap_sector=sg["gap_sector"],
                         ratio=sg["gap_sector"] / sg["gap_any"], n_saving=(sg["gap_sector"] / sg["gap_any"]) ** 2,
                         s2_first=float(sg["s2"][1]), par_first=float(sg["parity"][1]), W=W))
        print("b", rows[-1], flush=True)
out["sector"] = rows

# ---- (a) gap over-estimation, N=6, J2=0, trial L=1 DMRG circuit
N, J2 = 6, 0.0
Hp = j1j2_hamiltonian(N, 1.0, J2); Hm = Hp.to_matrix()
w, V = np.linalg.eigh(Hm)
W = w[-1] - w[0]; E = (w - w[0]) / W
gap_true = E[1]
with contextlib.redirect_stdout(io.StringIO()):
    r = get_energies(MPO_ham_j1j2(N, 1.0, J2), N=N, j1=1.0, j2=J2, run_ed_check=False)
qc = mps_to_circuit(r["psi0"].arrays, method="approximate", shape="lpr", num_layers=1)
psi = Statevector(qc).data; c = V.conj().T @ psi
gamma = abs(c[0]) ** 2
# sector weights of the trial state
S2 = _total_spin_squared(N).to_matrix(sparse=True)
s2 = np.real(np.einsum("ij,ij->j", V.conj(), S2 @ V))
wt_S = {f"S2={round(s,3)}": float(np.sum(abs(c[np.abs(s2 - s) < 1e-6]) ** 2)) for s in sorted(set(np.round(s2, 6)))}
out["trial_sector_weights_N6_L1"] = wt_S
print("trial sector weights", wt_S)
sens = []
for s_ in (-0.2, 0.0, 0.1, 0.2, 0.35, 0.5, 1.0):
    d_est = gap_true * (1 + s_)
    eps = 1e-3
    eta_t = floor.eta_target(eps, gamma)
    best = None
    for x in (0.6, 0.75, 1.0, 1.25):
        T = x * np.pi / d_est
        for m in (4, 6):
            res = floor.solve_floor(T, m, d_est, eta_t, 0.0, 1.0, n_random=2, seed=0)
            if res is not None and (best is None or x ** 2 / res["f0sq"] < best[0]):
                best = (x ** 2 / res["f0sq"], res)
    if best is None:
        sens.append(dict(s=s_, feasible=False)); continue
    res = best[1]
    F = fd.filter_values(res["times"], res["phases"], E)
    amp = c * F
    infid = 1 - abs(amp[0]) ** 2 / np.sum(abs(amp) ** 2)
    eta_true = floor.certify(res["times"], res["phases"], gap_true, 0.0, 1.0, rtol=0.005)["eta"]
    sens.append(dict(s=s_, gap_est=d_est, T=res["T"], m=res["m"], infid_exact_filter=infid,
                     target_leak=eps, eta_design=res["cert"]["eta"], eta_on_true_window=eta_true,
                     f0sq=res["f0sq"], T_phys=res["T"] * W))
    print("a", sens[-1], flush=True)
out["gap_sensitivity_N6"] = sens
out["gap_sensitivity_gamma"] = gamma
save("exp4_gap_symmetry", out)
