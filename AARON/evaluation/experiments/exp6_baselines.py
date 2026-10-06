"""E6: resource-vs-fidelity baselines at equal target.  usage: exp6_baselines.py N J2
Methods (all measured against the exact ED ground state, noiseless):
  approx   : mps-to-circuit method='approximate', L = 1..10 layers (the pipeline's trial circuit)
  exact_chi: mps-to-circuit method='exact' (sequential/isometry) on the DMRG MPS truncated to bond dim chi
  genprep  : generic isometry state preparation of the ED ground state (no structure)
  filter   : SBC filter on top of an L-layer trial circuit; GUARANTEED design (a-priori bound = certified
             infidelity), CX = trial + filter.   Not simulated (certified quantity)."""
from common import *
import sys, io, time, contextlib, copy, numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector
from mps_to_circuit import mps_to_circuit
from get_energies import get_energies, ed_ground_and_excited
from hamiltonians import MPO_ham_j1j2, reorder_axes
from core.resources import resource_costs
import pipeline as P

N, J2 = int(sys.argv[1]), float(sys.argv[2])
t0 = time.time()
H = MPO_ham_j1j2(N, 1.0, J2)
with contextlib.redirect_stdout(io.StringIO()):
    r = get_energies(H, N=N, j1=1.0, j2=J2, run_ed_check=False)
ed = ed_ground_and_excited(N, 1.0, J2)
psi0 = ed["psi0_ed"]
def score(qc):
    v = Statevector(qc).data
    g = float(abs(np.vdot(psi0, reorder_axes(v))) ** 2)
    c = resource_costs(qc, verbose=False)
    return dict(infid=1 - g, gamma=g, cx=int(c["cx"]), depth=int(c["depth"]), rz_nc=int(c["rz_nonclifford"]))
out = dict(N=N, J2=J2, approx=[], exact_chi=[], filter=[])
for L in range(1, 11):
    try:
        qc = mps_to_circuit(r["psi0"].arrays, method="approximate", shape="lpr", num_layers=L)
        out["approx"].append(dict(L=L, **score(qc))); print("approx", out["approx"][-1], flush=True)
    except Exception as e:
        print("approx fail", L, e); break
for chi in (1, 2, 4, 8, 16):
    try:
        psi = r["psi0"].copy(); psi.compress(max_bond=chi, cutoff=0.0); psi.normalize()
        t = time.time()
        qc = mps_to_circuit(psi.arrays, method="exact", shape="lpr")
        row = dict(chi=chi, **score(qc), secs=time.time() - t)
        out["exact_chi"].append(row); print("exact", row, flush=True)
    except Exception as e:
        out["exact_chi"].append(dict(chi=chi, error=repr(e))); print("exact fail", chi, repr(e)[:150], flush=True)
qc = QuantumCircuit(N); qc.append(StatePreparation(reorder_axes(psi0).astype(complex)), range(N))
c = resource_costs(qc, verbose=False)
out["genprep"] = dict(cx=int(c["cx"]), rz_nc=int(c["rz_nonclifford"]), depth=int(c["depth"])); print("genprep", out["genprep"], flush=True)

# filter points (guaranteed, a priori; no time-evolution simulation)
P.SIM_MAX_N = 0
P.FLOOR_EPS_FRACS = [0.2, 0.1, 0.05]; P.FLOOR_M = [4, 6]
for L in (1, 3):
    P.SWEEP_L = L
    ctx = P.run_quiet(P.build_ctx, N, 1.0, J2)
    P._STEP_CX["cx"] = ctx["step"]["cx"]
    for eps in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
        row = dict(L=L, eps=eps, gamma=ctx["gamma"], trial_cx=ctx["trial_cost"]["cx"], trial_rz=ctx["trial_cost"]["rz_nonclifford"])
        try:
            des, _, _ = P.make_design(ctx, eps)
            row["design"] = des["source"]
            if des["source"] == "none":
                row.update(cx=ctx["trial_cost"]["cx"], n=0, eps_bound=0.0)
            else:
                pt = P.find_guaranteed(ctx, des, eps)
                if pt is None: row["error"] = "no guaranteed design"
                else:
                    cc = P.costs(ctx, pt, pt["p_g_lb"])
                    row.update(n=pt["n"], eps_bound=pt["eps_bound"], cx=cc["cx_total"], cx_filter=cc["cx"],
                               rz_nc=cc["rz_nc"], p_lb=pt["p_g_lb"], exp_cx=cc["exp_cx"], m=pt["n_nz"])
        except Exception as e:
            row["error"] = repr(e)
        out["filter"].append(row); print("filter", row, flush=True)
out["time"] = time.time() - t0
save(f"exp6_N{N}_J2_{J2}", out)
