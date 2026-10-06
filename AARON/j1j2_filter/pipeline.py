"""Pipeline: per-case context, design, grid, search, circuit export.

Run-time settings (ORDER, SWEEP_L, DESIGN, ...) live at the top of this module."""


import math
import numpy as np
from floor import floor_vs_precision, grid_design
from get_energies import get_energies
from hamiltonians import (
    MPO_ham_j1j2,
    check_mpo_matches_pauli,
    j1j2_hamiltonian,
    scale_hamiltonian,
)
from mps_to_circuit import mps_to_circuit
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from console import VERBOSE, banner, fix, kv, run_quiet, sci, section
from core.circuits import (
    apply_filter_pulse,
    build_early_abort_circuit,
    verify_pulse_convention,
)
from core.filter_design import FilterBuilder, certify_filter, grid_filter, select_best
from core.resources import resource_costs, rotation_synthesis_t_count
from core.simulate import (
    overlaps,
    postselected_run,
    postselected_run_exact,
    state_metrics,
)
from core.spectrum import get_spectrum
from core.trotter import (
    _trotter_run,
    lie_trotter_order,
    steps_needed,
    total_error_bound,
    trotter_alpha,
    trotter_bounds,
)


SIZES = [6]
J1 = 1.0
J2_LIST = [0.0]
EPS_TARGETS = [1e-2]                  # target total infidelities (one report each)
N_SWEEP = [10, 20, 40, 80, 160, 320, 640, 1280, 2560]
SWEEP_L = 1                           # trial-circuit layers
ORDER = 1                             # first-order Trotter
SEED = 1234
BANDWIDTH = "certified"    
SPECTRUM_SOURCE = "ed"                # or "dmrg" (uncertified)
GAP_MODE = "any"                      # or "sector"
MPO_CHECK_MAX_N = 10
COST_TOPOLOGY = None                  # None (all-to-all) or "line"
SYNTH_EPS = 1e-3                      # per-rotation synthesis precision (T-count)
STRICT_BOUND = False                  # True: total-bound violation is fatal
SIM_MAX_N = 20_000                    # largest n simulated numerically
CIRCUIT_CHECK_MAX_N = 160             # also run the real qiskit circuit up to this n
VERIFY_COST_MAX_N = 200               # full-transpile cost check up to this n
DESIGN = "floor"                      # "floor" (precision-targeted) or "builder"
FLOOR_EPS_FRACS = [0.4, 0.2, 0.1, 0.05, 0.02, 0.01]   # leakage budget as fraction of eps
FLOOR_M = [4, 6, 8]
FLOOR_KW = dict(t_max_frac=0.5)
FLOOR_JOBS = 1


def run_global_checks():
    if not verify_pulse_convention(verbose=VERBOSE):
        raise RuntimeError("verify_pulse_convention() failed; fix before "
                           "trusting any fidelity or P_succ number.")
    H = j1j2_hamiltonian(4, 1.0, 0.7)
    order = run_quiet(lie_trotter_order, H)
    if not (order["forward"] or order["reverse"]):
        raise RuntimeError("LieTrotter equals neither explicit ordered product.")
    print(f"[checks] pulse convention OK; LieTrotter order: "
          f"forward={order['forward']} reverse={order['reverse']}")


def per_step_costs(H, cmap):
    """CX / depth / non-Clifford rz of ONE Trotter step of one pulse (phi=0, so
    the Rz(2 phi) is added separately per nonzero pulse)."""
    N = H.num_qubits
    qc = QuantumCircuit(N + 1)
    apply_filter_pulse(qc, list(range(N)), N, H, 1.0, 0.0, 1, ORDER)
    c = resource_costs(qc, verbose=False, coupling_map=cmap, synth_eps=SYNTH_EPS)
    return dict(cx=c["cx"], depth=c["depth"], rz_nc=c["rz_nonclifford"])


def build_ctx(N, j1, j2):
    ctx = dict(N=N, j1=j1, j2=j2)
    banner(f"CASE  N={N}  J1={j1:g}  J2={j2:g}")
    if N <= MPO_CHECK_MAX_N:
        ctx["mpo_err"] = check_mpo_matches_pauli(N, j1, j2)
        if ctx["mpo_err"] > 1e-9:
            print(f"  [warning] MPO vs Pauli mismatch: {ctx['mpo_err']:.2e}")

    H_mpo = MPO_ham_j1j2(N, j1=j1, j2=j2, cyclic=False)
    res = run_quiet(get_energies, H_mpo, N=N, j1=j1, j2=j2, run_ed_check=True)
    ed = res["ed"]
    psi0_ed = ed.get("psi0_ed") if ed else None
    psi0_dmrg = np.asarray(res["psi0"].to_dense()).reshape(-1)
    psi0_dmrg = psi0_dmrg / np.linalg.norm(psi0_dmrg)
    dmrg_ed = (float(abs(np.vdot(psi0_ed, psi0_dmrg)) ** 2)
               if psi0_ed is not None else float("nan"))

    H_qk = j1j2_hamiltonian(N, j1=j1, j2=j2)
    spec = run_quiet(get_spectrum, N, j1=j1, j2=j2, source=SPECTRUM_SOURCE, res=res,
                     bandwidth=BANDWIDTH, gap_mode=GAP_MODE)
    H_scaled = scale_hamiltonian(H_qk, spec["shift"], spec["W"])
    Hs = H_scaled.to_matrix(sparse=True)
    if N <= 16:
        w = np.linalg.eigvalsh(Hs.toarray())
        if w[0] < -1e-6 or w[-1] > 1 + 1e-9:
            print(f"  [warning] scaled spectrum [{w[0]:.2e}, {w[-1]:.6f}] leaves "
                  f"[0,1]: certification assumption violated.")

    trial_qc = mps_to_circuit(res["psi0"].arrays, method="approximate",
                              shape="lpr", num_layers=SWEEP_L)
    if trial_qc.num_qubits != N:
        raise ValueError(f"trial circuit has {trial_qc.num_qubits} qubits, "
                         f"expected {N}")
    trial_vec = Statevector(trial_qc).data
    g_dmrg, g_ed = overlaps(trial_vec, psi0_dmrg, psi0_ed)
    gamma = min(g_ed if np.isfinite(g_ed) else g_dmrg, 1.0)
    gamma_src = "ED" if np.isfinite(g_ed) else "DMRG (uncertified)"

    cmap = CouplingMap.from_line(N + 1) if COST_TOPOLOGY == "line" else None
    alpha = trotter_alpha(H_scaled, tight=True)
    ctx.update(
        spec=spec, H_qk=H_qk, H_scaled=H_scaled, Hs=Hs, trial_qc=trial_qc,
        trial_vec=trial_vec, psi0_dmrg=psi0_dmrg, psi0_ed=psi0_ed,
        gamma=gamma, gamma_src=gamma_src, g_dmrg=g_dmrg, g_ed=g_ed,
        dmrg_ed=dmrg_ed, alpha=alpha, gap=spec["gap"], e0_slack=spec["e0_slack"],
        energies=spec["energies"], cmap=cmap, E0_dmrg=float(np.real(res["E0"])),
        step=per_step_costs(H_scaled, cmap),
        trial_cost=resource_costs(trial_qc, verbose=False, synth_eps=SYNTH_EPS),
        builder_design=None)

    section("CASE INPUTS")
    gap_src = "ED-verified" if N <= 12 else "DMRG (uncertified)"
    src = SPECTRUM_SOURCE.upper()
    kv([(f"E0 ({src})", fix(spec["E0"], 8)), ("E1", fix(spec["E1"], 8)),
        ("Etop", fix(spec["Etop"], 6)), ("W = Etop-E0" if src == "ED" else "W",
        f"{fix(spec['W'], 5)} [{src if src == 'ED' else BANDWIDTH}]"),
        ("Delta (scaled gap)", fix(spec["gap"], 6)),
        ("gap mode / source", "any / ED" if src == "ED" else f"{GAP_MODE} / DMRG (uncertified)"),
        ("gamma", f"{fix(gamma, 6)} [{gamma_src}]"),
        ("trial F (ED)", fix(g_ed, 6)), ("trial F (DMRG)", fix(g_dmrg, 6)),
        ("1-gamma (trial)", sci(1 - gamma)),
        ("<DMRG|ED>^2", fix(dmrg_ed, 8)),
        ("alpha (tight, max ord)", fix(alpha, 4)),
        ("e0_slack (indicative)", sci(spec["e0_slack"])),
        ("trial layers L", str(SWEEP_L))])
    tc, st = ctx["trial_cost"], ctx["step"]
    kv([("trial CX / depth", f"{tc['cx']} / {tc['depth']}"),
        ("trial non-Cliff. rz", str(tc["rz_nonclifford"])),
        ("CX per Trotter step", str(st["cx"])),
        ("depth per step", str(st["depth"])),
        ("non-Cliff. rz / step", str(st["rz_nc"])),
        ("topology", COST_TOPOLOGY or "all-to-all")])
    return ctx


def choose_floor_design(ctx, eps_total):
    """Pick the leakage budget minimizing guaranteed n_req / P_succ_lb."""
    eps_list = [f * eps_total for f in FLOOR_EPS_FRACS]
    rows, _ = floor_vs_precision(
        eps_list=eps_list, gamma=ctx["gamma"], delta=ctx["gap"],
        e0_slack=ctx["e0_slack"], hi=1.0, m_list=FLOOR_M, n_jobs=FLOOR_JOBS,
        verbose=False, **FLOOR_KW)
    cands, n_infeasible = [], 0
    for r in rows:
        if not r.get("feasible") or r.get("m", 0) == 0:
            n_infeasible += 1
            continue
        eb = total_error_bound(ctx["gamma"], ctx["gap"], ctx["alpha"], r["times"],
                               r["phases"], np.ones(len(r["times"])), hi=1.0,
                               e0_slack=ctx["e0_slack"], eta=r["eta_cert"])
        dT = np.sqrt(eps_total) - np.sqrt(2.0 * eb["leak"])
        if dT <= 0 or eb["p_g_lb"] <= 0:
            n_infeasible += 1
            continue
        n_req = steps_needed(ctx["alpha"], r["T"], eb["p_g_lb"], dT)
        cands.append(dict(row=r, n_req=n_req, cost=n_req / eb["p_g_lb"],
                          leak=eb["leak"], p_g_lb=eb["p_g_lb"]))
    pick = min(cands, key=lambda c: c["cost"]) if cands else None
    return pick, cands, n_infeasible


def make_design(ctx, eps):
    cands, n_inf, pick = [], 0, None
    if DESIGN == "floor":
        pick, cands, n_inf = choose_floor_design(ctx, eps)
    if pick is not None:
        r = pick["row"]
        des = dict(source="floor", times=np.asarray(r["times"], float),
                   phases=np.asarray(r["phases"], float), T=float(r["T"]),
                   eta_cert=float(r["eta_cert"]), eta_target=float(r["eta_target"]),
                   x=float(r["x"]), leak_budget=float(r["eps"]))
    else:
        if DESIGN == "floor":
            print("  [warning] no feasible floor design; falling back to builder")
        if ctx["builder_design"] is None:
            b = FilterBuilder(total_time=ctx["spec"]["totaltime"],
                              energies=ctx["energies"], seed=SEED,
                              e0_slack=ctx["e0_slack"])
            best = select_best(b.build(verbose=False))
            T = float(np.sum(best["times"]))
            ctx["builder_design"] = dict(
                source="builder", times=best["times"], phases=best["phases"],
                T=T, eta_cert=best["eta"], eta_target=None,
                x=T * ctx["gap"] / np.pi, leak_budget=None)
        des = ctx["builder_design"]
    return des, cands, n_inf


def make_grid(ctx, des, n):
    """Snap the design to n equal steps; drop k=0 pulses (a t=0 pulse is just a
    scalar cos(phi) after post-selection, so eta and the state are unchanged)."""
    if des["source"] == "floor":
        g = grid_design(des["times"], des["phases"], des["T"], n, ctx["gap"],
                        des["eta_target"], ctx["e0_slack"], 1.0)
        k, dt, ph = g["k"], g["dt"], g["phases"]
        grid_ok, eta_bb = bool(g["feasible"]), g["cert"]["eta"]
    else:
        k, dt, ph = grid_filter(des["times"], des["phases"], des["T"], n,
                                ctx["energies"])
        grid_ok, eta_bb = True, np.inf
    k, ph = np.asarray(k, int), np.asarray(ph, float)
    n_designed = len(k)
    nz = k > 0
    k, ph = k[nz], ph[nz]
    tg = k * dt
    cert = certify_filter(tg, ph, ctx["energies"], e0_slack=ctx["e0_slack"])
    return dict(n=n, dt=dt, k=k, ph=ph, tg=tg, n_designed=n_designed,
                grid_ok=grid_ok, eta=min(cert["eta"], eta_bb))


def simulate_point(ctx, g, eps_bound):
    tg, ph, k = g["tg"], g["ph"], g["k"]
    p_ex, sv_ex = postselected_run_exact(ctx["trial_qc"], tg, ph, ctx["Hs"])
    sv, p = _trotter_run(ctx["H_scaled"], ctx["trial_vec"], tg, ph, k)
    _, Fd, Fe = state_metrics(sv, ctx["H_qk"], ctx["psi0_dmrg"], ctx["psi0_ed"])
    theta = np.angle(np.vdot(sv_ex, sv))
    state_err = float(np.linalg.norm(sv_ex - np.exp(-1j * theta) * sv))
    bnd = trotter_bounds(ctx["alpha"], tg, k, p_ex)
    if state_err > bnd["bound_state"] + 1e-9:
        raise RuntimeError(
            f"BOUND VIOLATION at n={g['n']}: state_err={state_err:.3e} > "
            f"{bnd['bound_state']:.3e}. Check term order, alpha, convention.")
    eps_actual = 1.0 - (Fe if np.isfinite(Fe) else Fd)
    out = dict(simulated=True, p_succ=p, p_succ_exact=p_ex, F_dmrg=Fd, F_ed=Fe,
               eps_actual=eps_actual, state_err=state_err,
               bound_state_raw=bnd["bound_state_raw"], circ_dev=None, warn=None)
    if g["n"] <= CIRCUIT_CHECK_MAX_N:
        _, svc = postselected_run(
            ctx["trial_qc"], tg, ph,
            lambda qc, s, a, t, phi, i: apply_filter_pulse(
                qc, s, a, ctx["H_scaled"], t, phi, int(k[i]), order=ORDER))
        out["circ_dev"] = float(1.0 - abs(np.vdot(sv, svc)) ** 2)
    if eps_actual > eps_bound + 1e-9:
        msg = (f"total-error bound violated at n={g['n']}: actual "
               f"{eps_actual:.3e} > bound {eps_bound:.3e} (check gap / gamma / "
               f"shift assumptions)")
        if STRICT_BOUND:
            raise RuntimeError(msg)
        out["warn"] = msg
    return out


def costs(ctx, pt, p_succ):
    s, tc, n = ctx["step"], ctx["trial_cost"], pt["n"]
    cx = s["cx"] * n
    rz = s["rz_nc"] * n + pt["n_nz"]                  # + one Rz(2 phi) per pulse
    cx_tot = cx + tc["cx"]
    return dict(cx=cx, cx_total=cx_tot, depth=s["depth"] * n + tc["depth"],
                rz_nc=rz + tc["rz_nonclifford"],
                t_est=rotation_synthesis_t_count(rz, SYNTH_EPS) + tc["t_count_est"],
                exp_cx=cx_tot / p_succ if p_succ > 0 else float("inf"))


def measured_cost(ctx, pt):
    if pt["n"] > VERIFY_COST_MAX_N:
        return None
    qc, _ = build_early_abort_circuit(
        ctx["H_scaled"], pt["tg"], pt["ph"], trial_prep=None,
        trotter_steps=[int(x) for x in pt["k"]], order=ORDER)
    c = resource_costs(qc, flatten=True, verbose=False, coupling_map=ctx["cmap"],
                       synth_eps=SYNTH_EPS)
    return dict(cx=c["cx"], depth=c["depth"])


def point(ctx, des, n, simulate=True):
    g = make_grid(ctx, des, n)
    if len(g["k"]) == 0:
        raise ValueError(f"n={n}: all pulses snapped to k=0")
    eb = total_error_bound(ctx["gamma"], ctx["gap"], ctx["alpha"], g["tg"],
                           g["ph"], g["k"], hi=1.0, e0_slack=ctx["e0_slack"],
                           eta=g["eta"])
    pt = dict(n=n, dt=g["dt"], k=g["k"], tg=g["tg"], ph=g["ph"],
              n_designed=g["n_designed"], n_nz=len(g["k"]), eta=g["eta"],
              grid_ok=g["grid_ok"], eps_bound=eb["eps_bound"], leak=eb["leak"],
              d_T=eb["trotter_dist"], eps_T=eb["eps_T"], p_g_lb=eb["p_g_lb"],
              f0=eb["f0"], simulated=False, eps_actual=None, p_succ=None,
              warn=None, circ_dev=None)
    if simulate:
        pt.update(simulate_point(ctx, g, eb["eps_bound"]))
    return pt


def n_required(ctx, des, eps):
    """A-priori n from the ungridded design (starting guess for the guarantee)."""
    eb = total_error_bound(ctx["gamma"], ctx["gap"], ctx["alpha"], des["times"],
                           des["phases"], np.ones(len(des["times"])), hi=1.0,
                           e0_slack=ctx["e0_slack"], eta=des["eta_cert"])
    dT = np.sqrt(eps) - np.sqrt(2.0 * eb["leak"])
    if dT <= 0 or eb["p_g_lb"] <= 0:
        return None
    return max(int(steps_needed(ctx["alpha"], des["T"], eb["p_g_lb"], dT)), 1)


def find_guaranteed(ctx, des, eps):
    """Smallest n (found by 5% increments from the a-priori value) whose bound
    on the SNAPPED grid design is <= eps; None if leakage alone exceeds eps."""
    n = n_required(ctx, des, eps)
    if n is None:
        return None
    for _ in range(60):
        pt = point(ctx, des, n, simulate=False)
        if pt["eps_bound"] <= eps:
            return point(ctx, des, n, simulate=(n <= SIM_MAX_N))
        n = int(math.ceil(n * 1.05)) + 1
    return None


def find_empirical(ctx, des, eps, cache):
    """Sweep N_SWEEP, then integer-bisect (to ~3%) between the last failing and
    first passing n. Measured error is not strictly monotone in n, so this is
    a practical value, not a guaranteed minimum."""
    def get(n):
        if n not in cache:
            cache[n] = point(ctx, des, n, simulate=True)
        return cache[n]

    prev, hit = 0, None
    for n in N_SWEEP:
        if n > SIM_MAX_N:
            break
        if get(n)["eps_actual"] <= eps:
            hit = n
            break
        prev = n
    if hit is None:
        return None
    lo, hi = prev, hit
    while hi - lo > max(1, int(0.03 * hi)):
        mid = (lo + hi) // 2
        if mid < 1:
            break
        if get(mid)["eps_actual"] <= eps:
            hi = mid
        else:
            lo = mid
    return get(hi)


_STEP_CX = {}


def ctx_cx(p):
    return _STEP_CX.get("cx", 0) * p["n"]


def f_after(p):
    """Measured fidelity after the filter (ED if available, else DMRG)."""
    if p is None or not p["simulated"]:
        return None
    return p["F_ed"] if np.isfinite(p["F_ed"]) else p["F_dmrg"]


def export_circuits(N=6, J2=0.0, eps=1e-2, which="guaranteed", J1_=None):
    """Return (trial_qc, filter_qc, full_qc, info) for one case.

    trial_qc  : N-qubit state-prep circuit (the MPS-derived ansatz)
    filter_qc : (N+1)-qubit filter circuit, ancilla = qubit N,
                built with build_early_abort_circuit (same one used for costs)
    full_qc   : trial_qc composed onto qubits 0..N-1 of filter_qc
    info      : dict with tg, ph, k, n, ancilla index, bounds, P_succ lb, etc.
    """
    j1 = J1 if J1_ is None else J1_
    ctx = run_quiet(build_ctx, N, j1, J2)
    des, _, _ = run_quiet(make_design, ctx, eps)

    if which == "guaranteed":
        pt = run_quiet(find_guaranteed, ctx, des, eps)
    else:
        pt = run_quiet(find_empirical, ctx, des, eps, {})
    if pt is None:
        raise RuntimeError(f"No '{which}' design found for eps={eps:g}")

    
    filter_qc, _ = build_early_abort_circuit(
        ctx["H_scaled"], pt["tg"], pt["ph"], trial_prep=None,
        trotter_steps=[int(x) for x in pt["k"]], order=ORDER)

    trial_qc = ctx["trial_qc"].copy()
    full_qc = filter_qc.copy()
    full_qc.compose(trial_qc, qubits=list(range(N)), front=True, inplace=True)

    info = dict(N=N, J2=J2, eps=eps, n=pt["n"], tg=pt["tg"], ph=pt["ph"],
                k=pt["k"], ancilla=N, eps_bound=pt["eps_bound"],
                p_succ_lb=pt["p_g_lb"], gamma=ctx["gamma"],
                H_scaled=ctx["H_scaled"], H_qk=ctx["H_qk"],
                shift=ctx["spec"]["shift"], W=ctx["spec"]["W"],
                psi0_ed=ctx["psi0_ed"], psi0_dmrg=ctx["psi0_dmrg"],
                trial_vec=ctx["trial_vec"])
    return trial_qc, filter_qc, full_qc, info
