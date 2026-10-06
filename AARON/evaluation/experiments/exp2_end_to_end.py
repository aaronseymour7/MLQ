"""E2: end-to-end guaranteed vs empirical, bound tightness, sharper-bound what-if.
usage: exp2_end_to_end.py N J2 [L]"""
from common import *
import sys, time, numpy as np
import pipeline as P
from core.trotter import total_error_bound, steps_needed

N, J2 = int(sys.argv[1]), float(sys.argv[2])
L = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P.SWEEP_L = L
t0 = time.time()
ctx = P.run_quiet(P.build_ctx, N, 1.0, J2)
res = dict(N=N, J2=J2, L=L, gamma=ctx["gamma"], gap=ctx["gap"], W=ctx["spec"]["W"], alpha=ctx["alpha"],
           step_cx=ctx["step"]["cx"], trial_cx=ctx["trial_cost"]["cx"], rows=[])

def sharp_total(gamma, leak, eps_T, p_g):
    s = eps_T / np.sqrt(p_g)
    if s >= 1: return 1.0
    th = np.arcsin(np.sqrt(leak)) + np.arcsin(s)
    return 1.0 if th >= np.pi / 2 else float(np.sin(th) ** 2)

def n_sharp(ctx, des, eps):
    eb = total_error_bound(ctx["gamma"], ctx["gap"], ctx["alpha"], des["times"], des["phases"],
                           np.ones(len(des["times"])), hi=1.0, e0_slack=ctx["e0_slack"], eta=des["eta_cert"])
    a = np.arcsin(np.sqrt(eps)) - np.arcsin(np.sqrt(eb["leak"]))
    if a <= 0: return None
    epsT = np.sqrt(eb["p_g_lb"]) * np.sin(a)
    return int(np.ceil(ctx["alpha"] * des["T"] ** 2 / (2 * epsT)))

for eps in (1e-1, 1e-2, 1e-3):
    row = dict(eps=eps)
    try:
        P._STEP_CX["cx"] = ctx["step"]["cx"]
        des, cands, ninf = P.make_design(ctx, eps)
        row.update(design=des["source"], x=des["x"], T=des["T"], m=len(des["times"]), eta=des["eta_cert"])
        row["n_sharp_apriori"] = n_sharp(ctx, des, eps)
        pg = P.find_guaranteed(ctx, des, eps)
        if pg is not None:
            c = P.costs(ctx, pg, pg["p_g_lb"])
            row["guar"] = dict(n=pg["n"], eps_bound=pg["eps_bound"], eps_actual=pg.get("eps_actual"),
                               state_err=pg.get("state_err"), bound_state_raw=pg.get("bound_state_raw"),
                               p_lb=pg["p_g_lb"], p_meas=pg.get("p_succ"), cx=c["cx_total"], t=c["t_est"],
                               leak=pg["leak"], eta=pg["eta"], nz=pg["n_nz"])
            if pg.get("simulated"):
                row["guar"]["sharp_bound"] = sharp_total(ctx["gamma"], pg["leak"], pg["eps_T"], pg["p_g_lb"])
        cache = {}
        pe = P.find_empirical(ctx, des, eps, cache)
        if pe is not None:
            c = P.costs(ctx, pe, pe["p_succ"])
            row["emp"] = dict(n=pe["n"], eps_actual=pe["eps_actual"], eta=pe["eta"], p_meas=pe["p_succ"],
                              cx=c["cx_total"], k=[int(k) for k in pe["k"]], grid_ok=pe["grid_ok"],
                              state_err=pe["state_err"])
        row["sweep"] = [dict(n=n, eps_actual=p["eps_actual"], eta=p["eta"], state_err=p["state_err"],
                             bound_raw=p["bound_state_raw"]) for n, p in sorted(cache.items())]
    except Exception as e:
        row["error"] = repr(e)
    res["rows"].append(row)
    print(N, J2, eps, {k: v for k, v in row.items() if k not in ("sweep",)}, flush=True)
res["time"] = time.time() - t0
save(f"exp2_N{N}_J2_{J2}_L{L}", res)
