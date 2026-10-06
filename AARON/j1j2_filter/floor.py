"""
floor.py -- the filter "floor" for a given precision (patched version).

    f0sq(eps) = max_{t, phi, T*Delta, m}  F(0)^2 = prod cos^2(phi_i)
                s.t.  certified eta <= eta_target(eps, gamma),
                      sum t_i = T,  t_min <= t_i <= t_max   (no zero-time pulses)

F(0) = prod cos(phi_i) is the filter's amplitude on the ground state, so F(0)^2
is the factor multiplying the ground-state probability: P_succ >= gamma * F(0)^2.
This replaces the hard-coded p_ground_min = 0.9. eps is the LEAKAGE share of the
error budget only (Trotter, snapping and synthesis are budgeted separately).

Changes relative to the first version
-------------------------------------
 1. gamma is clipped to <= 1 (DMRG/ED overlaps can print 1+1e-15); gamma = 1 means
    no filter is needed and is returned as a trivial row instead of raising.
 2. No hidden floor: the old stage-A guard c >= sqrt(0.1) is gone. Feasibility is
    found with a LADDER of tiny-to-large guards (0.9, 0.5, 0.1, 1e-2, 1e-3); the
    guard is only a search device, the reported value comes from stage B, which
    maximizes F(0)^2 directly. A design needing F(0)^2 < 1e-3 is reported
    infeasible (it would be useless anyway).
 3. Analytic Jacobians for every SLSQP stage (no finite differences).
 4. Certification fixed point: the inner target is tightened/relaxed until the
    CERTIFIED eta lies just below eta_target (replaces the fixed margin list).
 5. More robust starts: warm starts from the previous eps AND the previous x,
    geometric time ladders, Dirichlet random starts.
 6. Window [Delta, hi]: pass hi = (E_top - E0)/W (closed-form E_top) to stop
    suppressing a region the spectrum never reaches. Default hi = 1.
 7. Cost proxy x^2 / f0sq (Trotter steps ~ T^2, repeat cost ~ 1/f0sq), ties
    broken toward fewer pulses. Pass cost_fn to override.
 8. Wider default x = T*Delta/pi range (lower bound T*Delta >= (1-eta) F(0)).
 9. Results are named f0sq (alias 'floor' kept in rows for old callers).
10. grid_design(): snaps to n equal steps and re-optimizes phases under
    eta <= eta_target (replaces reopt_phases' 0.9 constraint).
11. Optional process-parallel sweep over m (n_jobs).

Certificate: interval branch-and-bound with
    |F(x)| <= |F(m)| + |F'(m)| d + L^2 d^2 / 2,   L = sum|t_i|,
valid because |F''| <= (sum|t_i|)^2. Pruned intervals keep their upper bound, so
the result is a rigorous sup bound (looser only if the interval cap is hit).
"""
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from scipy import optimize as opt

PHASE_LIM = (-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3)
F0SQ_GUARDS = (0.9, 0.5, 0.1, 1e-2, 1e-3)      # search ladder, NOT a design floor
DEFAULT_X = (0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5)


# ------------------------------------------------------------ basic pieces
def filter_values(times, phases, E):
    """F(E) = prod_i cos(E t_i + phi_i), vectorized over E."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    return np.prod(np.cos(np.outer(E, np.asarray(times)) + np.asarray(phases)),
                   axis=1)


def _prod_except(c):
    """Row-wise product of all entries except column j (no division)."""
    c = np.atleast_2d(c)
    pre, suf = np.ones_like(c), np.ones_like(c)
    if c.shape[1] > 1:
        pre[:, 1:] = np.cumprod(c[:, :-1], axis=1)
        suf[:, :-1] = np.cumprod(c[:, :0:-1], axis=1)[:, ::-1]
    return pre * suf


def _F_Jp(times, phases, E):
    """F(E) and dF/dphi_j (n_E x m). dF/dt_j = E * dF/dphi_j."""
    th = np.outer(E, times) + phases
    c, s = np.cos(th), np.sin(th)
    return np.prod(c, axis=1), -s * _prod_except(c)


def _F_dF(times, phases, E):
    """F(E) and dF/dE."""
    F, Jp = _F_Jp(times, phases, E)
    return F, Jp @ np.asarray(times)


def _c0_grad(phases):
    """c0 = prod cos(phi_i) and its gradient wrt phi."""
    c = np.cos(phases)
    return float(np.prod(c)), -np.sin(phases) * _prod_except(c)[0]


# ---------------------------------------------------------------- targets
def eta_target(eps, gamma):
    """F_exact >= gamma/(gamma+(1-gamma) eta^2) >= 1-eps
       iff eta^2 <= gamma eps / ((1-eps)(1-gamma)).  inf if gamma >= 1."""
    if not 0.0 < eps < 1.0:
        raise ValueError(f"need 0 < eps < 1, got {eps}")
    gamma = min(float(gamma), 1.0)
    if gamma <= 0.0:
        raise ValueError(f"need gamma > 0, got {gamma}")
    if gamma >= 1.0 - 1e-12:
        return float("inf")
    return float(np.sqrt(gamma * eps / ((1.0 - eps) * (1.0 - gamma))))


# ---------------------------------------------------------- certification
def certify(times, phases, delta, e0_slack=0.0, hi=1.0, rtol=0.02, n0=2000,
            max_intervals=4_000_000, atol=1e-14):
    """Rigorous eta = sup_{[Delta,hi]}|F| / |F(E0)| with |F(E0)| >= F(0) - L e0_slack.
    Valid when spec(H) is inside {E0} U [Delta, hi]."""
    times = np.asarray(times, float)
    phases = np.asarray(phases, float)
    L = float(np.sum(np.abs(times)))
    d = (hi - delta) / (2.0 * n0)
    mids = delta + d * (2.0 * np.arange(n0) + 1.0)
    lb, ub_pruned, converged = 0.0, 0.0, True
    while mids.size:
        F, dF = _F_dF(times, phases, mids)
        aF = np.abs(F)
        lb = max(lb, float(aF.max()))
        ub = aF + np.abs(dF) * d + 0.5 * L * L * d * d
        keep = ub > lb * (1.0 + rtol) + atol
        if (~keep).any():
            ub_pruned = max(ub_pruned, float(ub[~keep].max()))
        mids = mids[keep]
        if mids.size == 0:
            break
        if 2 * mids.size > max_intervals:
            ub_pruned = max(ub_pruned, float(ub[keep].max()))
            converged = False
            break
        d *= 0.5
        mids = np.concatenate([mids - d, mids + d])
    sup = min(max(ub_pruned, lb), 1.0)
    f0 = float(abs(np.prod(np.cos(phases))))
    f0_lb = max(f0 - L * e0_slack, 0.0)
    return dict(eta=sup / f0_lb if f0_lb > 0 else float("inf"), sup_cert=sup,
                f0=f0, f0_lb=f0_lb, f0sq=f0_lb ** 2, floor=f0_lb ** 2,
                lipschitz=L, converged=converged)


# ------------------------------------------------------------ SLSQP stages
def _unpack(v, m, t_fixed):
    return (v[:m], v[m:2 * m]) if t_fixed is None else (t_fixed, v[:m])


def _cols(Jt, Jp, t_fixed):
    return np.hstack([Jt, Jp]) if t_fixed is None else Jp


def _dc_full(dc, m, t_fixed):
    return np.concatenate([np.zeros(m), dc]) if t_fixed is None else dc


def _eq_constraint(m, T, t_fixed, extra=0):
    if t_fixed is not None:
        return []
    return [{"type": "eq", "fun": lambda z: np.sum(z[:m]) - T,
             "jac": lambda z: np.concatenate([np.ones(m), np.zeros(m + extra)])}]


def _stage_A(v0, T, E, m, bnds, t_fixed, f0sq_guard):
    """Minimize eta = sup|F|/F(0) (epigraph r) subject to F(0)^2 >= f0sq_guard."""
    t0, ph0 = _unpack(v0, m, t_fixed)
    c00 = float(np.prod(np.cos(ph0)))
    ref = max(float(np.max(np.abs(filter_values(t0, ph0, E)))) / max(c00, 1e-6),
              1e-14)
    cg = float(np.sqrt(f0sq_guard))
    nv = len(v0)

    def ineq(z):
        v, r = z[:-1], z[-1]
        t, ph = _unpack(v, m, t_fixed)
        F = filter_values(t, ph, E) / ref
        c = float(np.prod(np.cos(ph)))
        return np.concatenate([r * c - F, r * c + F, [c - cg]])

    def ineq_jac(z):
        v, r = z[:-1], z[-1]
        t, ph = _unpack(v, m, t_fixed)
        F, Jp = _F_Jp(t, ph, E)
        Jt = E[:, None] * Jp
        c, dc = _c0_grad(ph)
        dcf = _dc_full(dc, m, t_fixed)
        J = _cols(Jt, Jp, t_fixed) / ref
        top = -J + r * dcf
        bot = J + r * dcf
        col_r = np.full((len(E), 1), c)
        last = np.concatenate([dcf, [0.0]])[None, :]
        return np.vstack([np.hstack([top, col_r]), np.hstack([bot, col_r]), last])

    z0 = np.append(v0, 1.0)
    cons = [{"type": "ineq", "fun": ineq, "jac": ineq_jac}] + \
        _eq_constraint(m, T, t_fixed, extra=1)
    grad_obj = np.zeros(nv + 1)
    grad_obj[-1] = 1.0
    res = opt.minimize(lambda z: z[-1], z0, jac=lambda z: grad_obj,
                       method="SLSQP", bounds=bnds + [(0.0, None)],
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12})
    return res.x[:-1]


def _stage_B(v0, T, E, m, bnds, t_fixed, eta_t):
    """Maximize F(0) subject to |F(E_j)| <= eta_t * F(0) on the grid E."""
    def obj(v):
        _, ph = _unpack(v, m, t_fixed)
        return -float(np.sum(np.log(np.cos(ph))))

    def obj_jac(v):
        _, ph = _unpack(v, m, t_fixed)
        g = np.tan(ph)
        return np.concatenate([np.zeros(m), g]) if t_fixed is None else g

    def ineq(v):
        t, ph = _unpack(v, m, t_fixed)
        F = filter_values(t, ph, E) / eta_t
        c = float(np.prod(np.cos(ph)))
        return np.concatenate([c - F, c + F])

    def ineq_jac(v):
        t, ph = _unpack(v, m, t_fixed)
        F, Jp = _F_Jp(t, ph, E)
        Jt = E[:, None] * Jp
        c, dc = _c0_grad(ph)
        dcf = _dc_full(dc, m, t_fixed)
        J = _cols(Jt, Jp, t_fixed) / eta_t
        return np.vstack([dcf - J, dcf + J])

    cons = [{"type": "ineq", "fun": ineq, "jac": ineq_jac}] + \
        _eq_constraint(m, T, t_fixed)
    res = opt.minimize(obj, v0, jac=obj_jac, method="SLSQP", bounds=bnds,
                       constraints=cons,
                       options={"maxiter": 300, "ftol": 1e-12})
    return res.x


def _eta_on(v, m, t_fixed, E):
    t, ph = _unpack(v, m, t_fixed)
    c = float(np.prod(np.cos(ph)))
    return float(np.max(np.abs(filter_values(t, ph, E)))) / c if c > 0 else np.inf


def _local_max_idx(a):
    return np.where((a[1:-1] >= a[:-2]) & (a[1:-1] >= a[2:]))[0] + 1


def _optimise_to_target(v0, ctx, tgt):
    """Drive a design to grid-eta <= tgt, then maximize F(0) (stage B) with
    cutting-plane refinement of the grid. Returns (v, reached)."""
    T, m, delta, hi = ctx["T"], ctx["m"], ctx["delta"], ctx["hi"]
    t_fixed, bnds, G = ctx["t_fixed"], ctx["bnds"], ctx["G"]
    E = np.linspace(delta, hi, ctx["n_grid"])
    lo_b = np.array([b[0] for b in bnds])
    hi_b = np.array([b[1] for b in bnds])
    v = np.clip(v0, lo_b, hi_b)
    reached = False
    for g in F0SQ_GUARDS:
        v = _stage_A(v, T, E, m, bnds, t_fixed, g)
        if _eta_on(v, m, t_fixed, E) <= tgt:
            reached = True
            break
    if not reached:
        return v, False
    for _ in range(ctx["rounds"]):
        v = _stage_B(v, T, E, m, bnds, t_fixed, tgt)
        t, ph = _unpack(v, m, t_fixed)
        aF = np.abs(filter_values(t, ph, G))
        c = float(np.prod(np.cos(ph)))
        if aF.max() <= tgt * c * (1.0 + 2e-3):
            break
        pk = _local_max_idx(aF)
        E = np.union1d(E, G[pk[np.argsort(-aF[pk])][:30]])
    return v, True


# ------------------------------------------------------------- start points
def _cap_times(raw, T, tmin, tmax):
    t = np.maximum(np.asarray(raw, float), 1e-12)
    t = tmin + (T - len(t) * tmin) * t / t.sum()
    for _ in range(100):
        over = t > tmax
        if not over.any():
            break
        ex = (t[over] - tmax).sum()
        t[over] = tmax
        room = ~over
        t[room] += ex * (t[room] - tmin) / (t[room] - tmin).sum()
    return t


def _starts(T, m, tmin, tmax, rng, n_random, warm):
    out = [np.asarray(w, float) for w in warm if w is not None]
    for r in (0.45, 0.6, 0.75):
        out.append(np.concatenate([_cap_times(r ** np.arange(m), T, tmin, tmax),
                                   np.zeros(m)]))
    for _ in range(n_random):
        out.append(np.concatenate([
            _cap_times(rng.dirichlet(np.full(m, 2.0)), T, tmin, tmax),
            rng.uniform(-0.3, 0.3, m)]))
    return out


def _sorted_design(v, m):
    o = np.argsort(v[:m])
    return np.concatenate([v[:m][o], v[m:2 * m][o]])


# ------------------------------------------------------------------ solver
def solve_floor(T, m, delta, eta_t, e0_slack=0.0, hi=1.0, warm=(), n_random=2,
                seed=0, t_min_frac=0.05, t_max_frac=1 / 3, n_grid=200,
                rounds=3, fp_iters=4):
    """Best certified design for fixed (T, m): maximize F(0)^2 s.t. certified
    eta <= eta_t. t_min = t_min_frac*T/m so no pulse has zero duration.
    Returns dict (with 'f0sq' = certified lower bound on F(0)^2) or None."""
    tmin, tmax = t_min_frac * T / m, T * t_max_frac
    if m * tmin > T or m * tmax < T:
        return None
    if eta_t >= 1.0:                       # zero phases already satisfy it
        t = np.full(m, T / m)
        cert = certify(t, np.zeros(m), delta, e0_slack, hi, rtol=0.005)
        if cert["eta"] > eta_t:
            return None
        return dict(T=T, m=m, times=t, phases=np.zeros(m),
                    z=np.concatenate([t, np.zeros(m)]), cert=cert,
                    f0sq=cert["f0sq"], eta_target=eta_t)
    rng = np.random.default_rng(seed)
    bnds = [(tmin, tmax)] * m + [PHASE_LIM] * m
    ctx = dict(T=T, m=m, delta=delta, hi=hi, t_fixed=None, bnds=bnds,
               G=np.linspace(delta, hi, 20001), n_grid=n_grid, rounds=rounds)
    best = None
    for z in _starts(T, m, tmin, tmax, rng, n_random, warm):
        tgt, v = eta_t * 0.98, z
        retried = False
        for _ in range(fp_iters):
            v, reached = _optimise_to_target(v, ctx, tgt)
            if not reached:
                if retried:
                    break
                retried, tgt, v = True, eta_t * 0.998, z   # margin too greedy
                continue
            v = _sorted_design(v, m)
            cert = certify(v[:m], v[m:], delta, e0_slack, hi, rtol=0.005)
            if cert["eta"] <= eta_t:
                if best is None or cert["f0sq"] > best["cert"]["f0sq"]:
                    best = dict(T=T, m=m, times=v[:m].copy(),
                                phases=v[m:].copy(), z=v.copy(), cert=cert,
                                f0sq=cert["f0sq"], eta_target=eta_t)
                if cert["eta"] >= 0.97 * eta_t or cert["f0sq"] > 1 - 1e-6:
                    break
                tgt *= 0.995 * eta_t / cert["eta"]      # slack left: relax
            else:
                tgt *= 0.995 * eta_t / cert["eta"]      # over target: tighten
        if best is not None and best["f0sq"] > 1 - 1e-6:
            break
    return best


# -------------------------------------------------------------------- sweep
def _scaled_warm(entry):
    if entry is None:
        return None
    z, T_old, T_new = entry
    z = z.copy()
    m = len(z) // 2
    z[:m] *= T_new / T_old
    return z


def _sweep_m(args):
    """All (x, eps) for one m, with warm starts over eps (same x) and over x."""
    m, x_list, eps_list, gamma, delta, hi, e0_slack, seed, verbose, skip_infeasible, kw = args
    table, warm_x = [], {}
    for x in sorted(x_list):
        T = float(x * np.pi / delta)
        z_eps, dead = None, False
        for eps in eps_list:                              # loose -> tight
            eta_t = eta_target(eps, gamma)
            if dead and skip_infeasible:       # tighter eps cannot be easier
                table.append(dict(eps=eps, x=x, m=m, eta_target=eta_t,
                                  feasible=False, res=None, skipped=True))
                continue
            warm = [z_eps, _scaled_warm(
                None if eps not in warm_x else (*warm_x[eps], T))]
            r = solve_floor(T, m, delta, eta_t, e0_slack, hi, warm=warm,
                            seed=seed, **kw)
            if r is not None:
                z_eps = r["z"]
                warm_x[eps] = (r["z"], T)
            else:
                dead = True
            table.append(dict(eps=eps, x=x, m=m, eta_target=eta_t,
                              feasible=r is not None, res=r))
            if verbose:
                msg = (f"f0sq={r['f0sq']:.4f} eta={r['cert']['eta']:.2e}"
                       if r else "infeasible")
                print(f"[eps={eps:.0e} x={x:.2f} m={m:2d}] "
                      f"eta_target={eta_t:.2e}  {msg}", flush=True)
    return table


def floor_vs_precision(eps_list, gamma, delta, e0_slack, x_list=DEFAULT_X,
                       m_list=(4, 6, 8, 10, 12), hi=1.0, seed=0, verbose=True,
                       cost_fn=None, n_jobs=1, skip_infeasible=True, **kw):
    """For each (x = T*Delta/pi, m): continuation over eps (loose -> tight) with
    warm starts over eps and x. Returns (rows, table):
      rows  : best (x, m) per eps by cost_fn(x, f0sq, m) (default x^2/f0sq, ties
              within 2% broken toward fewer pulses); f0sq = certified F(0)^2
      table : every (eps, x, m) result.
    A row with m = 0 means gamma = 1 (no filter needed).
    skip_infeasible: once (x, m) is infeasible at some eps, tighter eps are not
    attempted (eta_target only shrinks); entries are marked skipped=True.
    The cost proxy ignores the grid-validity floor on n (n >~ a few steps per
    pulse) and t_min, so very small x winners should be re-checked with grid_design."""
    gamma = min(float(gamma), 1.0)
    eps_sorted = sorted(eps_list, reverse=True)
    if gamma >= 1.0 - 1e-9:
        rows = [dict(eps=e, eta_target=float("inf"), feasible=True, x=0.0, m=0,
                     f0sq=1.0, floor=1.0, P_succ_lb=1.0, gamma=gamma,
                     note="gamma = 1: trial state already exact, no filter")
                for e in eps_sorted]
        return rows, []
    cost_fn = cost_fn or (lambda x, f0sq, m: x * x / f0sq)
    tasks = [(m, list(x_list), eps_sorted, gamma, delta, hi, e0_slack, seed,
              verbose, skip_infeasible, kw) for m in m_list]
    if n_jobs > 1:
        with ProcessPoolExecutor(max_workers=n_jobs) as ex:
            parts = list(ex.map(_sweep_m, tasks))
    else:
        parts = [_sweep_m(t) for t in tasks]
    table = [e for p in parts for e in p]
    rows = []
    for eps in eps_sorted:
        ok = [t for t in table if t["eps"] == eps and t["feasible"]]
        if not ok:
            rows.append(dict(eps=eps, eta_target=eta_target(eps, gamma),
                             feasible=False, gamma=gamma))
            continue
        cost = {id(t): cost_fn(t["x"], t["res"]["f0sq"], t["m"]) for t in ok}
        cmin = min(cost.values())
        near = [t for t in ok if cost[id(t)] <= 1.02 * cmin]
        b = min(near, key=lambda t: (t["m"], cost[id(t)]))
        c = b["res"]["cert"]
        rows.append(dict(eps=eps, eta_target=b["eta_target"], feasible=True,
                         x=b["x"], m=b["m"], T=b["res"]["T"],
                         f0sq=b["res"]["f0sq"], floor=b["res"]["f0sq"],
                         P_succ_lb=gamma * b["res"]["f0sq"], gamma=gamma,
                         eta_cert=c["eta"], cost_proxy=cost[id(b)],
                         times=b["res"]["times"], phases=b["res"]["phases"],
                         certified_converged=c["converged"]))
    return rows, table


# --------------------------------------------------------------- grid stage
def snap_steps(times, T, n):
    """Integer k_i with sum k_i = n (largest remainder); t_i = k_i * dt."""
    times = np.asarray(times, float) * T / np.sum(times)
    dt = T / n
    x = times / dt
    k = np.floor(x + 1e-12).astype(int)
    for j in np.argsort(-(x - k))[: n - k.sum()]:
        k[j] += 1
    return k, dt


def grid_design(times, phases, T, n, delta, eta_t, e0_slack=0.0, hi=1.0,
                n_grid=200, rounds=3, seed=0):
    """Snap a continuous design to n equal steps, drop k=0 pulses, and
    RE-OPTIMIZE PHASES under certified eta <= eta_t (maximizing F(0)^2).
    Replaces builder.grid_filter / reopt_phases (which used the 0.9 floor).
    Returns dict(k, dt, phases, cert, feasible). If eta_t cannot be reached on
    this grid, 'phases' are the min-eta phases and feasible = False."""
    k, dt = snap_steps(times, T, n)
    keep = k > 0
    k = k[keep]
    t = k * dt
    m = len(k)
    ph0 = np.asarray(phases, float)[keep]
    ctx = dict(T=float(t.sum()), m=m, delta=delta, hi=hi, t_fixed=t,
               bnds=[PHASE_LIM] * m, G=np.linspace(delta, hi, 20001),
               n_grid=n_grid, rounds=rounds)
    rng = np.random.default_rng(seed)
    starts = [ph0, np.zeros(m)] + [rng.uniform(-0.3, 0.3, m) for _ in range(2)]
    best, fallback = None, None
    for v in starts:
        tgt = eta_t * 0.98
        for _ in range(4):
            v, reached = _optimise_to_target(v, ctx, tgt)
            cert = certify(t, v, delta, e0_slack, hi, rtol=0.005)
            if not reached:
                if fallback is None or cert["eta"] < fallback["cert"]["eta"]:
                    fallback = dict(k=k, dt=dt, phases=v.copy(), cert=cert,
                                    feasible=False)
                break
            if cert["eta"] <= eta_t:
                if best is None or cert["f0sq"] > best["cert"]["f0sq"]:
                    best = dict(k=k, dt=dt, phases=v.copy(), cert=cert,
                                feasible=True)
                if cert["eta"] >= 0.97 * eta_t or cert["f0sq"] > 1 - 1e-6:
                    break
            tgt *= 0.995 * eta_t / cert["eta"]
        if best is not None and best["cert"]["f0sq"] > 1 - 1e-6:
            break
    return best or fallback


# Wiring (inside run_case, after gamma_ref and spec exist):
#   hi = (spec["Etop"] - spec["E0"]) / spec["W"]          # closed-form E_top
#   rows, table = floor_vs_precision(
#       eps_list=[1e-2, 1e-3, 1e-4, 1e-5, 1e-6], gamma=gamma_ref,
#       delta=spec["gap"], e0_slack=spec["e0_slack"], hi=hi, n_jobs=4)
#   floor_rows = rows            # do NOT reuse the name `rows` (sweep below)
#   pick = next(r for r in floor_rows if r["eps"] == EPS_LEAK and r["feasible"])
#   sw_times, sw_phases, T = pick["times"], pick["phases"], pick["T"]
#   ...
#   for n in N_SWEEP:
#       g = grid_design(sw_times, sw_phases, T, n, spec["gap"],
#                       pick["eta_target"], spec["e0_slack"], hi)
#       k, dt, ph = g["k"], g["dt"], g["phases"]       # g["feasible"] flags n
#       cert_g = g["cert"]