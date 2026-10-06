"""One (case, eps) run: design -> guaranteed / empirical -> report."""


from console import banner, section
from core.filter_design import fidelity_lower_bound
from pipeline import (
    N_SWEEP,
    SIM_MAX_N,
    _STEP_CX,
    f_after,
    find_empirical,
    find_guaranteed,
    make_design,
)
from reporting import plot_case, report_candidates, report_point, report_sweep


def run_target(ctx, eps):
    _STEP_CX["cx"] = ctx["step"]["cx"]
    banner(f"N={ctx['N']}  J2={ctx['j2']:g}   TARGET eps = {eps:g}", "=")
    des, cands, n_inf = make_design(ctx, eps)
    if des["source"] == "floor":
        report_candidates(cands, n_inf, next((c for c in cands
                          if c["row"]["eps"] == des["leak_budget"]), None))
    F_lb = fidelity_lower_bound(ctx["gamma"], des["eta_cert"])
    print(f"\n  design source: {des['source']}   exact-filter fidelity lower bound "
          f"(R2) = {F_lb:.6f}")

    # guaranteed
    pt_g = find_guaranteed(ctx, des, eps)
    if pt_g is None:
        section("GUARANTEED  (a-priori rigorous bound <= eps)")
        print("  NOT ACHIEVABLE: leakage alone exceeds the budget "
              "(or no n satisfied the bound). Lower eta / increase pulses.")
        c_g = None
    else:
        c_g = report_point(ctx, des, pt_g, f"GUARANTEED  (rigorous bound <= {eps:g})",
                           pt_g["p_g_lb"], "P_succ lb", eps)

    # empirical
    cache = {}
    pt_e = find_empirical(ctx, des, eps, cache)
    if pt_e is None:
        section("EMPIRICAL  (measured 1-F <= eps)")
        print(f"  target not reached within the sweep (n <= "
              f"{min(max(N_SWEEP), SIM_MAX_N)})")
        c_e = None
    else:
        c_e = report_point(ctx, des, pt_e, f"EMPIRICAL  (measured 1-F <= {eps:g})",
                           pt_e["p_succ"], "P_succ meas.", eps)
        if pt_g is not None:
            print(f"\n  n_guaranteed / n_empirical = {pt_g['n'] / pt_e['n']:.1f}x   "
                  f"CX ratio = {c_g['cx_total'] / c_e['cx_total']:.1f}x")

    report_sweep(cache, pt_g if pt_g and pt_g["simulated"] else None, pt_e)
    tag = f"N{ctx['N']}_J2_{ctx['j2']:g}_eps{eps:g}_{des['source']}"
    plot_case(ctx, eps, cache, pt_g, pt_e, tag)

    def g(p, key):
        return None if p is None else p[key]

    return dict(
        N=ctx["N"], J2=ctx["j2"], eps=eps, design=des["source"],
        gap=ctx["gap"], W=ctx["spec"]["W"], gamma=ctx["gamma"], alpha=ctx["alpha"],
        x=des["x"], T=des["T"], m_designed=len(des["times"]),
        eta_design=des["eta_cert"],
        n_guar=g(pt_g, "n"), nz_guar=g(pt_g, "n_nz"), eta_guar=g(pt_g, "eta"),
        eps_bound_guar=g(pt_g, "eps_bound"), eps_actual_guar=g(pt_g, "eps_actual"),
        p_succ_lb_guar=g(pt_g, "p_g_lb"),
        cx_guar=None if c_g is None else c_g["cx_total"],
        t_guar=None if c_g is None else c_g["t_est"],
        n_emp=g(pt_e, "n"), nz_emp=g(pt_e, "n_nz"), eta_emp=g(pt_e, "eta"),
        eps_actual_emp=g(pt_e, "eps_actual"), eps_bound_emp=g(pt_e, "eps_bound"),
        p_succ_emp=g(pt_e, "p_succ"),
        cx_emp=None if c_e is None else c_e["cx_total"],
        t_emp=None if c_e is None else c_e["t_est"],
        trial_F=ctx["gamma"],
        F_after_guar=f_after(pt_g), F_after_emp=f_after(pt_e),
        kmax_guar=None if pt_g is None else int(pt_g["k"].max()),
        cx_block_guar=None if pt_g is None else ctx["step"]["cx"] * int(pt_g["k"].max()),
        kmax_emp=None if pt_e is None else int(pt_e["k"].max()),
        cx_block_emp=None if pt_e is None else ctx["step"]["cx"] * int(pt_e["k"].max()))
