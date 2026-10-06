"""E7: a NON-ORACLE empirical protocol.  usage: exp7_nonoracle_empirical.py N J2 [L]
Selects the number of Trotter steps n without using the exact ground state:
  (i)  certified leakage of the snapped design (rigorous, classical): leak_cert(n)
  (ii) Richardson-type Trotter estimate from two runs of the SAME circuit family: delta_n = 1-|<psi_n|psi_2n>|^2,
       infidelity-to-limit ~ 4 delta_n (first-order error ~ 1/n)  -> accept smallest n on a grid with
       (sqrt(leak_cert) + 2 sqrt(delta_n))^2 <= eps.
Then the actual infidelity (ED) is evaluated for validation only, and compared with the oracle-empirical n and the
guaranteed n."""
from common import *
import sys, time, numpy as np
import pipeline as P
from core.trotter import _trotter_run
from core.simulate import state_metrics

N, J2 = int(sys.argv[1]), float(sys.argv[2]); L = int(sys.argv[3]) if len(sys.argv) > 3 else 1
P.SWEEP_L = L; P.FLOOR_EPS_FRACS = [0.2, 0.1, 0.05]; P.FLOOR_M = [4, 6]
t0 = time.time()
ctx = P.run_quiet(P.build_ctx, N, 1.0, J2); P._STEP_CX["cx"] = ctx["step"]["cx"]
GRID = [3, 4, 5, 6, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200, 256, 320, 400, 512, 640, 800, 1024, 1500, 2048]
out = dict(N=N, J2=J2, L=L, gamma=ctx["gamma"], step_cx=ctx["step"]["cx"], trial_cx=ctx["trial_cost"]["cx"], rows=[])
cache = {}
def run(des, n):
    if n not in cache:
        g = P.make_grid(ctx, des, n)
        if len(g["k"]) == 0: cache[n] = None
        else:
            sv, p = _trotter_run(ctx["H_scaled"], ctx["trial_vec"], g["tg"], g["ph"], g["k"])
            cache[n] = (g, sv, p)
    return cache[n]
for eps in (1e-1, 1e-2, 1e-3):
    row = dict(eps=eps)
    try:
        cache.clear()
        des, _, _ = P.make_design(ctx, eps)
        row["design"] = des["source"]
        if des["source"] == "none":
            out["rows"].append(row); continue
        chosen = None
        for n in GRID:
            a, b = run(des, n), run(des, 2 * n)
            if a is None or b is None: continue
            g, sv, p = a
            delta = 1 - abs(np.vdot(sv, b[1])) ** 2
            eb = P.total_error_bound(ctx["gamma"], ctx["gap"], ctx["alpha"], g["tg"], g["ph"], g["k"], hi=1.0, e0_slack=ctx["e0_slack"], eta=g["eta"])
            leak = eb["leak"]
            est = (np.sqrt(leak) + 2 * np.sqrt(max(delta, 0))) ** 2
            if est <= eps and g["eta"] < 1:
                _, Fd, Fe = state_metrics(sv, ctx["H_qk"], ctx["psi0_dmrg"], ctx["psi0_ed"])
                actual = 1 - (Fe if np.isfinite(Fe) else Fd)
                pt = P.point(ctx, des, n, simulate=False)
                c = P.costs(ctx, pt, p)
                chosen = dict(n=n, est=float(est), leak_cert=float(leak), delta=float(delta), eta=float(g["eta"]), actual=float(actual),
                              p_succ=float(p), cx=c["cx_total"], cx_filter=c["cx"], rz_nc=c["rz_nc"], exp_cx=c["exp_cx"], k=[int(x) for x in g["k"]])
                break
        row["protocol"] = chosen
        pe = P.find_empirical(ctx, des, eps, {})
        if pe is not None:
            c = P.costs(ctx, pe, pe["p_succ"]); row["oracle"] = dict(n=pe["n"], actual=pe["eps_actual"], eta=pe["eta"], cx=c["cx_total"])
        pg = P.find_guaranteed(ctx, des, eps)
        if pg is not None:
            c = P.costs(ctx, pg, pg["p_g_lb"]); row["guaranteed"] = dict(n=pg["n"], cx=c["cx_total"], eps_bound=pg["eps_bound"])
    except Exception as e:
        row["error"] = repr(e)
    out["rows"].append(row)
    print(N, J2, L, {k: v for k, v in row.items()}, flush=True)
out["time"] = time.time() - t0
save(f"exp7_N{N}_J2_{J2}_L{L}", out)
