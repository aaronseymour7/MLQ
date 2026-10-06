"""E3b: is the filter worth it?  Trial-circuit layers L vs. exact/near-exact state preparation.
usage: exp3b_baseline.py N J2"""
from common import *
import sys, io, contextlib, numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from mps_to_circuit import mps_to_circuit
from get_energies import get_energies, ed_ground_and_excited
from hamiltonians import MPO_ham_j1j2, j1j2_hamiltonian, reorder_axes
from core.resources import resource_costs
N, J2 = int(sys.argv[1]), float(sys.argv[2])
H = MPO_ham_j1j2(N, 1.0, J2)
with contextlib.redirect_stdout(io.StringIO()):
    r = get_energies(H, N=N, j1=1.0, j2=J2, run_ed_check=False)
ed = ed_ground_and_excited(N, 1.0, J2)
out = dict(N=N, J2=J2, rows=[])
for L in range(1, 9):
    try:
        qc = mps_to_circuit(r["psi0"].arrays, method="approximate", shape="lpr", num_layers=L)
    except Exception as e:
        print("L", L, "failed", e); break
    v = Statevector(qc).data
    g = float(abs(np.vdot(ed["psi0_ed"], reorder_axes(v))) ** 2)
    c = resource_costs(qc, verbose=False)
    out["rows"].append(dict(L=L, gamma=g, infid=1 - g, cx=c["cx"], depth=c["depth"], rz_nc=c["rz_nonclifford"]))
    print(N, J2, "L", L, f"1-F={1-g:.3e}", "CX", c["cx"], "rz", c["rz_nonclifford"], flush=True)
# exact preparation of the ED ground state (generic isometry synthesis)
qc = QuantumCircuit(N)
qc.append(StatePreparation(reorder_axes(ed["psi0_ed"]).astype(complex)), range(N))
c = resource_costs(qc, verbose=False)
out["exact_prep"] = dict(cx=c["cx"], depth=c["depth"], rz_nc=c["rz_nonclifford"])
print("exact state-prep CX", c["cx"], "rz_nc", c["rz_nonclifford"])
save(f"exp3b_N{N}_J2_{J2}", out)
