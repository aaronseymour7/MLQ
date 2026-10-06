"""E5: a-priori guaranteed cost vs system size N (no time-evolution simulation; certified quantities only).
usage: exp5_scaling.py J2 L N1,N2,...   (eps fixed 1e-2)"""
from common import *
import sys, time, numpy as np
import pipeline as P
J2, L = float(sys.argv[1]), int(sys.argv[2]); Ns = [int(x) for x in sys.argv[3].split(",")]
P.SWEEP_L = L; P.SIM_MAX_N = 0
P.FLOOR_EPS_FRACS = [0.2, 0.1, 0.05]; P.FLOOR_M = [4, 6]
eps = 1e-2
rows = []
for N in Ns:
    t0 = time.time()
    ctx = P.run_quiet(P.build_ctx, N, 1.0, J2)
    row = dict(N=N, J2=J2, L=L, gamma=ctx["gamma"], gap=ctx["gap"], W=ctx["spec"]["W"], alpha=ctx["alpha"],
               gap_raw=ctx["gap"] * ctx["spec"]["W"], alpha_raw=ctx["alpha"] * ctx["spec"]["W"] ** 2,
               step_cx=ctx["step"]["cx"], trial_cx=ctx["trial_cost"]["cx"], trial_rz=ctx["trial_cost"]["rz_nonclifford"])
    try:
        P._STEP_CX["cx"] = ctx["step"]["cx"]
        des, c, ninf = P.make_design(ctx, eps)
        row.update(design=des["source"], x=des["x"], T=des["T"], m=len(des["times"]), eta=des["eta_cert"])
        pt = P.find_guaranteed(ctx, des, eps)
        if pt is not None:
            cc = P.costs(ctx, pt, pt["p_g_lb"])
            row.update(n_guar=pt["n"], p_lb=pt["p_g_lb"], eps_bound=pt["eps_bound"], cx=cc["cx_total"], cx_filter=cc["cx"],
                       exp_cx=cc["exp_cx"], rz_nc=cc["rz_nc"], depth=cc["depth"])
    except Exception as e:
        row["error"] = repr(e)
    row["time"] = time.time() - t0
    rows.append(row)
    print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    save(f"exp5_J2_{J2}_L{L}_N{Ns[0]}-{Ns[-1]}", rows)
