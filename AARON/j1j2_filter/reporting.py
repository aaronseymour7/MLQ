"""Printed reports and per-case plots."""


import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from console import banner, fix, kv, sci, section
from pipeline import SIM_MAX_N, SWEEP_L, costs, ctx_cx, measured_cost


matplotlib.use("Agg")


def report_candidates(cands, n_inf, pick):
    section("DESIGN SELECTION (leakage budget -> guaranteed steps)")
    if not cands:
        print("  no feasible floor candidates")
        return
    print(f"  {'':2}{'eps_leak':>9}{'x':>7}{'m':>4}{'T':>8}{'eta':>9}"
          f"{'F0^2':>8}{'leak<=':>10}{'n_req':>10}{'cost':>12}")
    for c in cands:
        r = c["row"]
        print(f"  {'*' if c is pick else ' ':2}{r['eps']:>9.1e}{r['x']:>7.2f}"
              f"{r['m']:>4d}{r['T']:>8.2f}{r['eta_cert']:>9.3f}{r['f0sq']:>8.3f}"
              f"{c['leak']:>10.2e}{c['n_req']:>10d}{c['cost']:>12.0f}")
    if n_inf:
        print(f"  ({n_inf} infeasible candidate(s) skipped)")


def report_point(ctx, des, pt, title, p_cost, p_label, eps):
    section(title)
    c = costs(ctx, pt, p_cost)
    print("  Filter design")
    kv([("x = T*Delta/pi", f"{des['x']:.3f}"), ("T (scaled)", fix(des["T"], 4)),
        ("Delta (scaled)", fix(ctx["gap"], 6)),
        ("eta target", sci(des["eta_target"]) if des["eta_target"] else "n/a"),
        ("eta certified (n)", sci(pt["eta"], 3)), ("|F(0)|", fix(pt["f0"], 4)),
        ("pulses designed", str(len(des["times"]))),
        ("pulses nonzero", f"{pt['n_nz']}  (of {pt['n_designed']} on grid)"),
        ("gamma", fix(ctx["gamma"], 6)),
        ("leak budget", sci(des["leak_budget"]) if des["leak_budget"] else "n/a")])
    print(f"  k_i  = {pt['k'].tolist()}")
    print(f"  t_i  = {[round(float(t), 4) for t in pt['tg']]}")
    j = int(np.argmax(pt["k"]))
    kmax = int(pt["k"][j])
    print(f"  Largest evolution block: pulse {j + 1}/{pt['n_nz']}  "
          f"t={pt['tg'][j]:.4f}  Trotter steps={kmax}  "
          f"CX={ctx['step']['cx'] * kmax}  depth~{ctx['step']['depth'] * kmax}")
    print("  Trotter / bound")
    kv([("n (total steps)", str(pt["n"])), ("dt = T/n", fix(pt["dt"], 5)),
        ("alpha", fix(ctx["alpha"], 4)), ("eps_T = sum a t^2/2k", sci(pt["eps_T"])),
        ("leakage bound", sci(pt["leak"])), ("d_T (state dist)", sci(pt["d_T"])),
        ("P_succ lower bound", fix(pt["p_g_lb"], 5)),
        ("TOTAL eps bound", sci(pt["eps_bound"])),
        ("target eps", sci(eps)), ("bound <= target", str(pt["eps_bound"] <= eps))])
    if not pt["grid_ok"]:
        print("  [note] eta_target not reachable on this grid; bound uses achieved eta")
    if pt["simulated"]:
        print("  Measured (noiseless)")
        kv([("actual 1-F", sci(pt["eps_actual"])),
            ("F_dmrg", fix(pt["F_dmrg"], 6)), ("F_ed", fix(pt["F_ed"], 6)),
            ("P_succ", fix(pt["p_succ"], 5)),
            ("state err vs exact", sci(pt["state_err"])),
            ("Trotter bound (raw)", sci(pt["bound_state_raw"])),
            ("circuit vs numpy 1-F", sci(pt["circ_dev"]) if pt["circ_dev"] is not None else "not run")])
        if pt["warn"]:
            print(f"  [warning] {pt['warn']}")
    else:
        print(f"  Measured: not simulated (n={pt['n']} > SIM_MAX_N={SIM_MAX_N})")
    print("  Fidelity (trial -> after filter)")
    after_ed = fix(pt["F_ed"], 6) if pt["simulated"] else "not simulated"
    after_dm = fix(pt["F_dmrg"], 6) if pt["simulated"] else "not simulated"
    kv([("trial  F_ed / F_dmrg", f"{fix(ctx['g_ed'], 6)} / {fix(ctx['g_dmrg'], 6)}"),
        ("after  F_ed", after_ed), ("after  F_dmrg", after_dm),
        ("guaranteed F >=", f"{fix(1 - pt['eps_bound'], 6)}  (1 - eps bound)")],
       ncol=1)
    print("  Costs (incl. trial circuit)")
    kv([("CX (filter)", str(c["cx"])), ("CX (total)", str(c["cx_total"])),
        ("depth (total)", str(c["depth"])), ("non-Cliff. rz", str(c["rz_nc"])),
        ("approx T-count", f"{c['t_est']:.0f}"),
        (f"E[CX]/accept ({p_label})", f"{c['exp_cx']:.0f}")])
    m = measured_cost(ctx, pt)
    if m is not None:
        err = 100 * (m["cx"] - c["cx"]) / c["cx"] if c["cx"] else float("nan")
        print(f"  full-transpile check: filter CX={m['cx']} depth={m['depth']} "
              f"(estimate CX={c['cx']}, off {err:+.1f}%)")
    return c


def report_sweep(cache, pt_g, pt_e):
    pts = dict(cache)
    for p in (pt_g, pt_e):
        if p is not None and p["simulated"]:
            pts[p["n"]] = p
    if not pts:
        return
    section("CONVERGENCE TABLE (measured points)")
    print(f"  {'n':>7}{'nz':>4}{'eps_actual':>12}{'eps_bound':>12}{'state_err':>12}"
          f"{'P_succ':>9}{'eta':>10}{'CX':>9}")
    for n in sorted(pts):
        p = pts[n]
        tag = ("G" if p is pt_g else "") + ("E" if p is pt_e else "")
        print(f"  {n:>7}{p['n_nz']:>4}{p['eps_actual']:>12.2e}{p['eps_bound']:>12.2e}"
              f"{p['state_err']:>12.2e}{p['p_succ']:>9.4f}{p['eta']:>10.2e}"
              f"{ctx_cx(p):>9}  {tag}")
    print("  (G = guaranteed n, E = empirical n)")


def plot_case(ctx, eps, cache, pt_g, pt_e, tag):
    pts = dict(cache)
    for p in (pt_g, pt_e):
        if p is not None and p["simulated"]:
            pts[p["n"]] = p
    if not pts:
        return
    ns = sorted(pts)
    fig, ax = plt.subplots(1, 3, figsize=(16, 4))
    ax[0].loglog(ns, [max(pts[n]["state_err"], 1e-16) for n in ns], "o-",
                 label="actual state error")
    ax[0].loglog(ns, [pts[n]["bound_state_raw"] for n in ns], "k--",
                 label="rigorous bound (unclamped)")
    ax[0].axhline(2.0, color="gray", ls=":")
    ax[0].set(xlabel="n", ylabel="state error vs exact grid filter",
              title="Trotter error vs bound")
    ax[0].legend()
    ax[1].loglog(ns, [max(pts[n]["eps_actual"], 1e-16) for n in ns], "o-",
                 label=r"actual $1-F$")
    ax[1].loglog(ns, [pts[n]["eps_bound"] for n in ns], "k--", label="total bound")
    ax[1].axhline(eps, color="r", ls=":", label=f"target {eps:g}")
    for p, c, lab in ((pt_g, "g", "guaranteed n"), (pt_e, "m", "empirical n")):
        if p is not None:
            ax[1].axvline(p["n"], color=c, ls="--", alpha=0.6, label=lab)
    ax[1].set(xlabel="n", ylabel="ground-state infidelity", title="Total error")
    ax[1].legend()
    ax[2].loglog(ns, [ctx_cx(pts[n]) for n in ns], "o-", label="CX (estimate)")
    ax[2].set(xlabel="n", ylabel="CX count", title="Filter cost")
    ax[2].legend()
    fig.suptitle(f"N={ctx['N']} J2={ctx['j2']:g} eps={eps:g} L={SWEEP_L}")
    fig.tight_layout()
    fig.savefig(f"trotter_sweep_{tag}.png", dpi=150)
    plt.close(fig)


def print_summary(summary):
    banner("SUMMARY")
    print(f"{'N':>3}{'J2':>6}{'eps':>8}{'x':>6}{'T':>8}{'m':>3}{'eta':>9}"
          f"{'n_guar':>9}{'nz':>3}{'n_emp':>8}{'nz':>3}{'CX_guar':>10}{'CX_emp':>10}")
    for r in summary:
        f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
        print(f"{r['N']:>3}{r['J2']:>6g}{r['eps']:>8.0e}{r['x']:>6.2f}{r['T']:>8.2f}"
              f"{r['m_designed']:>3}{f(r['eta_guar'], '.2e'):>9}"
              f"{f(r['n_guar'], 'd'):>9}{f(r['nz_guar'], 'd'):>3}"
              f"{f(r['n_emp'], 'd'):>8}{f(r['nz_emp'], 'd'):>3}"
              f"{f(r['cx_guar'], 'd'):>10}{f(r['cx_emp'], 'd'):>10}")


def print_block_summary(summary):
    banner("FIDELITY AND LARGEST EVOLUTION BLOCK")
    print(f"{'N':>3}{'J2':>6}{'eps':>8}{'trial F':>10}{'F_guar':>10}{'F_emp':>10}"
          f"{'kmax_g':>8}{'CXblk_g':>9}{'kmax_e':>8}{'CXblk_e':>9}")
    for r in summary:
        f = lambda v, fmt: "n/a" if v is None else format(v, fmt)
        print(f"{r['N']:>3}{r['J2']:>6g}{r['eps']:>8.0e}{r['trial_F']:>10.5f}"
              f"{f(r['F_after_guar'], '.5f'):>10}{f(r['F_after_emp'], '.5f'):>10}"
              f"{f(r['kmax_guar'], 'd'):>8}{f(r['cx_block_guar'], 'd'):>9}"
              f"{f(r['kmax_emp'], 'd'):>8}{f(r['cx_block_emp'], 'd'):>9}")
