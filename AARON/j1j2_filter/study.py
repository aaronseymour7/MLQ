"""
study.py -- full scaling study for the cosine-filter pipeline.

    from study import StudyConfig, run_study, line_routing, a2a_routing
    cfg = StudyConfig(out_dir="study_v1",
                      J2_list=(0.0, 0.2411, 0.4),
                      N_resource=(4, 6, 8, 10, 12),   # circuits built + counted, NOT simulated
                      N_ideal=(4, 6, 8, 10, 12),              # noiseless filter runs
                      eps_list=(1e-1, 3e-2, 1e-2, 3e-3, 1e-3),
                      eps_cases=((6, 0.0), (8, 0.0), (6, 0.4)),
                      routing=(a2a_routing(), line_routing()))
    out = run_study(cfg)            # compute (resumable) -> figures -> report.md
    out = run_study(cfg, compute_data=False)   # re-plot / re-report from cached data only

Three experiments (all share one cache, so nothing is ever recomputed):

  A. Resource scaling vs N  (no simulation): trial / filter / full-circuit CX,
     depth, Trotter steps, pulses, evolution time, T-count, per-step cost,
     gamma, p_succ lower bound, W, gap; optional routed 2q counts and
     estimated duration for any RoutingSpec (all-to-all, line, a fake backend).
     Power-law and exponential fits over N are reported.
  B. Ideal (noiseless) filter effect vs N and J2: energy and fidelity of the
     trial state, of the exact filter, and of the actual Trotterized circuit
     (post-selected statevector), plus P_succ and the Trotter-only distance.
  C. Ideal runs vs epsilon: does the design hit its target eps, how n and CX
     scale with 1/eps, and how Trotter error scales with n.

Outputs in cfg.out_dir:  data/*.json (one per point, resumable), figures/*.png|pdf,
all_resources.csv, all_ideal.csv, config.json, report.md, noisy/ (optional).

Notes / assumptions
  * Needs builder.py, hamiltonians.py and case_report.py importable, and your
    notebook namespace to define build_ctx, make_design, find_guaranteed,
    find_empirical, build_early_abort_circuit, run_quiet, J1, ORDER (or only
    export_circuits, in which case contexts are rebuilt for every eps).
  * "Ideal" = post-selected statevector of the *synthesized Trotter circuit*
    (builder.postselected_run), so it is the circuit that would be run, minus noise.
  * Statevector cost grows as 2^N: ideal runs are skipped for N > cfg.ideal_max_N.
  * Failed points are written as data/*.err and retried on the next run.
"""
from __future__ import annotations

import csv
import datetime
import json
import math
import os
import platform
import sys
import time
import traceback
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

import qiskit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap

from core.circuits import apply_filter_pulse, verify_pulse_convention
from core.simulate import overlaps, postselected_run, postselected_run_exact
from core.trotter import alpha_triangle, trotter_bounds
from pipeline import export_circuits
from case_report import Target, _default, _logical_resources, _prep, run_case

NAN = float("nan")


# ======================================================================
# Config
# ======================================================================
@dataclass
class RoutingSpec:
    """Where to transpile the full circuit for routed 2q counts / duration
    (no simulation). coupling: None (all-to-all), a CouplingMap, an edge list,
    or a callable n_qubits -> CouplingMap. backend: a (fake) backend instead."""
    label: str
    backend: Any = None
    coupling: Any = None
    basis: Tuple[str, ...] = ("rz", "sx", "x", "cx", "id")

    def to_target(self, nq: int):
        if self.backend is not None:
            if self.backend.num_qubits < nq:
                return None
            return Target(self.label, None, dict(backend=self.backend), self.backend, {})
        kw = dict(basis_gates=list(self.basis) + ["measure", "reset", "if_else", "delay"])
        if self.coupling is not None:
            cm = self.coupling(nq) if callable(self.coupling) else self.coupling
            if not isinstance(cm, CouplingMap):
                cm = CouplingMap(cm)
            if cm.size() < nq:
                return None
            kw["coupling_map"] = cm
        return Target(self.label, None, kw, None, {})


def a2a_routing():
    return RoutingSpec("a2a")


def line_routing():
    return RoutingSpec("line", coupling=lambda n: CouplingMap.from_line(n))


@dataclass
class StudyConfig:
    out_dir: str = "study"
    which: str = "guaranteed"             # design used for A and B ("empirical" is an oracle: it uses the exact ground state)
    eps_base: float = 1e-2
    J2_list: Tuple[float, ...] = (0.0, 0.2411, 0.4)   # J2=0.5 is the exactly solvable Majumdar-Ghosh point (gamma=1): sanity check only
    J1: Optional[float] = None            # None -> J1 from the notebook namespace
    # A: resource scaling (circuits built and counted only)
    N_resource: Tuple[int, ...] = (4, 6, 8, 10, 12)      # N>12 needs a non-ED spectrum source (uncertified)
    # B: ideal runs over N and J2
    N_ideal: Tuple[int, ...] = (4, 6, 8, 10, 12)
    ideal_max_N: int = 14
    # C: ideal runs over epsilon
    eps_list: Tuple[float, ...] = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3)
    eps_cases: Tuple[Tuple[int, float], ...] = ((6, 0.0), (8, 0.0), (6, 0.4))
    eps_which: Tuple[str, ...] = ("guaranteed", "empirical")   # empirical = oracle lower envelope
    # routing / transpile
    routing: Tuple[RoutingSpec, ...] = ()
    optimization_level: int = 2
    seed_transpiler: int = 1
    # fits
    fit_min_N: int = 6
    # optional noisy spot-checks (reuse case_report)
    noisy_cases: Tuple[dict, ...] = ()              # e.g. ({"N": 4, "J2": 0.0, "eps": 1e-2},)
    noisy_targets: Tuple[Target, ...] = ()
    noisy_shots: int = 4000
    # misc
    force: bool = False                   # recompute even if cached
    show: bool = False                    # plt.show() instead of closing figures
    fig_formats: Tuple[str, ...] = ("png", "pdf")


# ======================================================================
# Circuit factory (caches the expensive context per (N, J2))
# ======================================================================
class CaseFactory:
    """Mirrors export_circuits() but reuses build_ctx per (N, J2) across eps.
    Falls back to export_circuits() if the pieces are not in the namespace."""
    PARTS = ("build_ctx", "make_design", "find_guaranteed", "find_empirical",
             "build_early_abort_circuit", "run_quiet")

    def __init__(self, ns=None, J1=None, order=None, max_cached=3, circuits_fn=None):
        """circuits_fn(N, J2, eps, which) -> (trial_qc, filter_qc, full_qc, info);
        if given it is used for everything (e.g. pass your export_circuits).
        Otherwise names are looked up in `ns` (dict or module; default: the
        notebook namespace)."""
        self.circuits_fn = circuits_fn
        if ns is None:
            import __main__
            ns = vars(__main__)
        self.ns = ns if isinstance(ns, dict) else vars(ns)
        self.J1 = J1 if J1 is not None else self.ns.get("J1")
        self.order = order if order is not None else self.ns.get("ORDER", 1)
        self._ctx: "OrderedDict[tuple, dict]" = OrderedDict()
        self.max_cached = max_cached

    def _has_parts(self):
        return all(p in self.ns for p in self.PARTS) and self.J1 is not None

    def ctx(self, N, J2):
        key = (int(N), float(J2))
        if key in self._ctx:
            self._ctx.move_to_end(key)
            return self._ctx[key]
        c = self.ns["run_quiet"](self.ns["build_ctx"], N, self.J1, J2)
        self._ctx[key] = c
        while len(self._ctx) > self.max_cached:
            self._ctx.popitem(last=False)
        return c

    def circuits(self, N, J2, eps, which):
        if self.circuits_fn is not None:
            return self.circuits_fn(N, J2, eps, which)
        if not self._has_parts():
            if "export_circuits" not in self.ns:
                missing = [p for p in self.PARTS if p not in self.ns]
                if self.J1 is None:
                    missing.append("J1")
                raise NameError(
                    f"pipeline not found in this namespace (missing: {missing}, and no "
                    "export_circuits). Run the cells that define them in THIS kernel, or "
                    "pass ns=globals() / ns=vars(your_module), or circuits_fn=your_function "
                    "to run_study().")
            kw = dict(N=N, J2=J2, eps=eps, which=which)
            if self.J1 is not None:
                kw["J1_"] = self.J1
            return self.ns["export_circuits"](**kw)
        ns, rq = self.ns, self.ns["run_quiet"]
        ctx = self.ctx(N, J2)
        des, _, _ = rq(ns["make_design"], ctx, eps)
        if des.get("source") == "none":      # gamma = 1: nothing to filter
            from qiskit import QuantumCircuit
            trial_qc = ctx["trial_qc"].copy()
            filter_qc = QuantumCircuit(N + 1)
            full_qc = filter_qc.copy()
            full_qc.compose(trial_qc, qubits=list(range(N)), front=True, inplace=True)
            info = dict(N=N, J2=J2, eps=eps, n=0, tg=np.array([]), ph=np.array([]),
                        k=np.array([], dtype=int), ancilla=N, eps_bound=0.0,
                        p_succ_lb=1.0, gamma=ctx["gamma"], H_scaled=ctx["H_scaled"],
                        H_qk=ctx["H_qk"], shift=ctx["spec"]["shift"], W=ctx["spec"]["W"],
                        psi0_ed=ctx["psi0_ed"], psi0_dmrg=ctx["psi0_dmrg"],
                        trial_vec=ctx["trial_vec"], gap_scaled=float(ctx["spec"]["gap"]),
                        design_source="none")
            return trial_qc, filter_qc, full_qc, info
        if which == "guaranteed":
            pt = rq(ns["find_guaranteed"], ctx, des, eps)
        else:
            pt = rq(ns["find_empirical"], ctx, des, eps, {})
        if pt is None:
            raise RuntimeError(f"No '{which}' design found for N={N}, J2={J2}, eps={eps:g}")
        filter_qc, _ = ns["build_early_abort_circuit"](
            ctx["H_scaled"], pt["tg"], pt["ph"], trial_prep=None,
            trotter_steps=[int(x) for x in pt["k"]], order=self.order)
        trial_qc = ctx["trial_qc"].copy()
        full_qc = filter_qc.copy()
        full_qc.compose(trial_qc, qubits=list(range(N)), front=True, inplace=True)
        gap = None
        try:
            gap = float(ctx["spec"]["gap"])
        except Exception:
            pass
        info = dict(N=N, J2=J2, eps=eps, n=pt["n"], tg=pt["tg"], ph=pt["ph"], k=pt["k"],
                    ancilla=N, eps_bound=pt["eps_bound"], p_succ_lb=pt["p_g_lb"],
                    gamma=ctx["gamma"], H_scaled=ctx["H_scaled"], H_qk=ctx["H_qk"],
                    shift=ctx["spec"]["shift"], W=ctx["spec"]["W"],
                    psi0_ed=ctx["psi0_ed"], psi0_dmrg=ctx["psi0_dmrg"],
                    trial_vec=ctx["trial_vec"], gap_scaled=gap)
        return trial_qc, filter_qc, full_qc, info


# ======================================================================
# Per-point computation
# ======================================================================
def _flt(x):
    try:
        return float(x)
    except Exception:
        return NAN


def resource_row(cfg, circ, J2, eps, which, order, build_s):
    trial_qc, filter_qc, full_qc, info = circ
    N, m = info["N"], len(info["tg"])
    tg = np.asarray(info["tg"], float)
    k = np.asarray(info["k"], int)
    W = float(info["W"])
    t0 = time.time()
    lr = _logical_resources(trial_qc, filter_qc, full_qc, info, order)
    row = dict(kind="res", N=int(N), J2=float(J2), eps=float(eps), which=which,
               n_pulses=m, n_steps=int(info["n"]), k=k.tolist(), tg=tg.tolist(),
               phases=np.asarray(info["ph"], float).tolist(),
               T_scaled=float(tg.sum()), T_phys=float(tg.sum() / W), W=W,
               shift=float(info["shift"]), gamma=float(info["gamma"]),
               p_succ_lb=float(info["p_succ_lb"]), eps_bound=float(info["eps_bound"]),
               n_terms=int(len(info["H_scaled"]) - 1),
               gap_scaled=_flt(info.get("gap_scaled")), build_s=build_s,
               design_source=info.get("design_source", "filter"))
    row["gap_raw"] = row["gap_scaled"] * W if np.isfinite(row["gap_scaled"]) else NAN
    for part, s in (("trial", "trial"), ("filter_success_path", "filter"),
                    ("full_success_path", "full")):
        for c in ("cx", "h", "rz", "total", "depth", "rz_nonclifford", "t_count_est"):
            row[f"{s}_{c}"] = _flt(lr[part][c])
    row["step_cx"] = _flt(lr["per_trotter_step"].get("cx"))
    row["step_depth"] = _flt(lr["per_trotter_step"].get("depth"))
    for spec in cfg.routing:
        tgt = spec.to_target(N + 1)
        if tgt is None:
            continue
        pre = f"route[{spec.label}]"
        try:
            _, st, _, _ = _prep(full_qc, tgt, cfg.optimization_level, cfg.seed_transpiler, N)
            row[pre + "_n2q"] = int(st["n_2q"])
            row[pre + "_n1q"] = int(st["n_1q"])
            row[pre + "_depth"] = int(st["depth"])
            row[pre + "_dur_us"] = _flt(st.get("duration_us"))
            row[pre + "_T2_us"] = _flt(st.get("median_T2_us"))
        except Exception as e:
            row[pre + "_error"] = f"{type(e).__name__}: {e}"
    row["res_s"] = time.time() - t0
    return row


def _state_metrics(vec, info):
    vec = np.asarray(vec)
    E = float(np.real(Statevector(vec).expectation_value(info["H_qk"])))
    fd = fe = NAN
    if info.get("psi0_dmrg") is not None:
        try:
            fd = float(overlaps(vec, info["psi0_dmrg"], None)[0])
        except Exception:
            pass
    if info.get("psi0_ed") is not None:
        try:
            fe = float(overlaps(vec, info["psi0_ed"], None)[0])
        except Exception:
            pass
    F = fe if np.isfinite(fe) else fd
    return dict(E=E, F_dmrg=fd, F_ed=fe, F=F)


def ideal_row(cfg, circ, J2, eps, which, order):
    trial_qc, _, _, info = circ
    N = info["N"]
    tg = np.asarray(info["tg"], float)
    ph = np.asarray(info["ph"], float)
    k = np.asarray(info["k"], int)
    W = float(info["W"])
    Hs = info["H_scaled"]
    steps = [int(x) for x in k]
    t0 = time.time()

    v_b = Statevector(trial_qc).data
    P_ex, v_x = postselected_run_exact(trial_qc, tg, ph, Hs.to_matrix(sparse=True))

    def pulse_fn(pc, sq, anc, t, phi, idx):
        apply_filter_pulse(pc, sq, anc, Hs, t, phi, trotter_steps=steps[idx], order=order)
    P_tr, v_t = postselected_run(trial_qc, tg, ph, pulse_fn, synthesize=True)

    mb, mx, mt = (_state_metrics(v, info) for v in (v_b, v_x, v_t))
    d_trot = float(1.0 - abs(np.vdot(v_x, v_t)) ** 2)
    p_g = float(info["p_succ_lb"])
    alpha = float(alpha_triangle(info["H_qk"]) / W ** 2)
    bound = NAN
    if p_g > 0:
        bs = trotter_bounds(alpha, tg, k, p_g)["bound_state"]
        bound = float(min(1.0, bs ** 2))
    row = dict(kind="ideal", N=int(N), J2=float(J2), eps=float(eps), which=which,
               n_steps=int(info["n"]), n_pulses=len(tg), T_scaled=float(tg.sum()),
               gamma=float(info["gamma"]), p_succ_lb=p_g, eps_bound=float(info["eps_bound"]),
               E_ref=float(info["shift"]), alpha_scaled=alpha,
               P_exact=float(P_ex), P_trot=float(P_tr),
               d_trotter=d_trot, trotter_infid_bound=bound,
               ref_source="ED" if np.isfinite(mb["F_ed"]) else "DMRG")
    for suf, m in (("b", mb), ("x", mx), ("t", mt)):
        row[f"E_{suf}"] = m["E"]
        row[f"F_ed_{suf}"] = m["F_ed"]
        row[f"F_dmrg_{suf}"] = m["F_dmrg"]
        row[f"F_{suf}"] = m["F"]
        row[f"infid_{suf}"] = 1.0 - m["F"] if np.isfinite(m["F"]) else NAN
        row[f"dE_{suf}"] = m["E"] - row["E_ref"]
        row[f"dE_per_site_{suf}"] = (m["E"] - row["E_ref"]) / N
    row["gain_t"] = (row["infid_b"] / max(row["infid_t"], 1e-16)
                     if np.isfinite(row["infid_t"]) else NAN)
    row["ideal_s"] = time.time() - t0
    return row


# ---------------------------------------------------------------- cache
def _key(kind, N, J2, eps, which):
    return f"{kind}_N{N}_J2{J2:g}_eps{eps:g}_{which}"


def _ddir(cfg):
    d = os.path.join(cfg.out_dir, "data")
    os.makedirs(d, exist_ok=True)
    return d


def _dump(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_default)


def _plan_points(cfg):
    pts: Dict[tuple, Tuple[bool, bool]] = {}

    def add(N, J2, eps, which, res, ideal):
        key = (int(N), float(J2), float(eps), which)
        r, i = pts.get(key, (False, False))
        pts[key] = (r or res, i or ideal)
    for N in cfg.N_resource:
        for J2 in cfg.J2_list:
            add(N, J2, cfg.eps_base, cfg.which, True, False)
    for N in cfg.N_ideal:
        for J2 in cfg.J2_list:
            add(N, J2, cfg.eps_base, cfg.which, True, N <= cfg.ideal_max_N)
    for (N, J2) in cfg.eps_cases:
        for eps in cfg.eps_list:
            for w in cfg.eps_which:
                add(N, J2, eps, w, True, N <= cfg.ideal_max_N)
    return [(*k, *v) for k, v in sorted(pts.items())]


def compute(cfg, fac=None, verbose=True):
    """Run every missing point; each result is written as soon as it is done."""
    fac = fac or CaseFactory(J1=cfg.J1)
    if not verify_pulse_convention(verbose=False):
        raise RuntimeError("verify_pulse_convention failed: circuit convention is broken")
    ddir = _ddir(cfg)
    pts = _plan_points(cfg)
    for i, (N, J2, eps, which, want_res, want_ideal) in enumerate(pts, 1):
        paths = {k: os.path.join(ddir, _key(k, N, J2, eps, which) + ".json")
                 for k in ("res", "ideal")}
        need = {"res": want_res and (cfg.force or not os.path.exists(paths["res"])),
                "ideal": want_ideal and (cfg.force or not os.path.exists(paths["ideal"]))}
        tag = f"[{i}/{len(pts)}] N={N} J2={J2:g} eps={eps:g} {which}"
        if not any(need.values()):
            if verbose:
                print(f"{tag}: cached")
            continue
        t0 = time.time()

        def fail(kind, e):
            _dump(paths[kind] + ".err", dict(kind=kind, N=N, J2=J2, eps=eps, which=which,
                                             error=f"{type(e).__name__}: {e}",
                                             traceback=traceback.format_exc()))
            if verbose:
                print(f"{tag}: {kind} FAILED: {type(e).__name__}: {e}")
        try:
            circ = fac.circuits(N, J2, eps, which)
        except Exception as e:
            for k, v in need.items():
                if v:
                    fail(k, e)
            continue
        build_s = time.time() - t0
        for kind, fn in (("res", lambda: resource_row(cfg, circ, J2, eps, which, fac.order, build_s)),
                         ("ideal", lambda: ideal_row(cfg, circ, J2, eps, which, fac.order))):
            if not need[kind]:
                continue
            try:
                _dump(paths[kind], fn())
                if os.path.exists(paths[kind] + ".err"):
                    os.remove(paths[kind] + ".err")
                if verbose:
                    print(f"{tag}: {kind} ok ({time.time() - t0:.1f}s)")
            except Exception as e:
                fail(kind, e)
    errs = [fn for fn in os.listdir(ddir) if fn.endswith(".err")]
    if errs:
        print(f"\n[compute] {len(errs)} failed point file(s) in {ddir} (*.err). "
              "First error:")
        with open(os.path.join(ddir, sorted(errs)[0])) as f:
            print("   ", json.load(f)["error"])
    return fac


def run_noisy(cfg, fac=None, verbose=True):
    """Optional hardware-sim spot checks via case_report.run_case (cached)."""
    if not cfg.noisy_cases or not cfg.noisy_targets:
        return
    fac = fac or CaseFactory(J1=cfg.J1)
    out = os.path.join(cfg.out_dir, "noisy")
    for c in cfg.noisy_cases:
        N, J2 = c["N"], c.get("J2", 0.0)
        eps, which = c.get("eps", cfg.eps_base), c.get("which", cfg.which)
        tag = f"N{N}_J2{J2:g}_eps{eps:g}_{which}"
        if not cfg.force and os.path.exists(os.path.join(out, tag + ".json")):
            continue
        try:
            circ = fac.circuits(N, J2, eps, which)
            run_case(circuits=circ, targets=list(cfg.noisy_targets), shots=cfg.noisy_shots,
                     optimization_level=cfg.optimization_level,
                     seed_transpiler=cfg.seed_transpiler, order=fac.order,
                     out_dir=out, tag=tag, which=which, verbose=False)
            if verbose:
                print(f"[noisy] {tag}: ok")
        except Exception as e:
            print(f"[noisy] {tag}: FAILED {type(e).__name__}: {e}")


# ======================================================================
# Loading, selection, fits
# ======================================================================
def load_rows(cfg, kind):
    d = _ddir(cfg)
    rows = []
    for fn in sorted(os.listdir(d)):
        if fn.startswith(kind + "_") and fn.endswith(".json"):
            with open(os.path.join(d, fn)) as f:
                rows.append(json.load(f))
    return rows


def load_errors(cfg):
    d, out = _ddir(cfg), []
    for fn in sorted(os.listdir(d)):
        if fn.endswith(".err"):
            with open(os.path.join(d, fn)) as f:
                out.append(json.load(f))
    return out


def _same(a, b):
    if isinstance(a, str) or isinstance(b, str):
        return a == b
    try:
        return bool(np.isclose(float(a), float(b), rtol=1e-9, atol=1e-14))
    except Exception:
        return a == b


def _sel(rows, **f):
    return [r for r in rows if all(k in r and _same(r[k], v) for k, v in f.items())]


def _xy(rows, x, y):
    pts = []
    for r in rows:
        try:
            a, b = float(r[x]), float(r[y])
        except Exception:
            continue
        if np.isfinite(a) and np.isfinite(b):
            pts.append((a, b))
    pts.sort()
    if not pts:
        return np.array([]), np.array([])
    arr = np.array(pts)
    return arr[:, 0], arr[:, 1]


def _fit(x, y, logx):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (y > 0)
    if logx:
        m &= x > 0
    if m.sum() < 3:
        return None
    X = np.log(x[m]) if logx else x[m]
    Y = np.log(y[m])
    s, b = np.polyfit(X, Y, 1)
    ss = float(np.sum((Y - Y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((Y - (s * X + b)) ** 2)) / ss if ss > 0 else 1.0
    return dict(slope=float(s), intercept=float(b), r2=r2, npts=int(m.sum()))


def compute_fits(cfg, R, I):
    """Keys: (scope, model, metric, J2, N, which)."""
    fits: Dict[tuple, dict] = {}
    base = _sel(R, eps=cfg.eps_base, which=cfg.which)
    for J2 in sorted({r["J2"] for r in base}):
        rows = [r for r in _sel(base, J2=J2) if r["N"] >= cfg.fit_min_N]
        for metric in ("trial_cx", "filter_cx", "full_cx", "full_depth", "n_steps",
                       "T_scaled", "n_terms", "step_cx"):
            x, y = _xy(rows, "N", metric)
            for model, logx in (("power", True), ("exp", False)):
                f = _fit(x, y, logx)
                if f:
                    fits[("N", model, metric, J2, None, None)] = f
    for (N, J2) in cfg.eps_cases:
        for w in cfg.eps_which:
            r = _sel(R, N=N, J2=J2, which=w)
            i = _sel(I, N=N, J2=J2, which=w)
            for metric, rows in (("n_steps", r), ("full_cx", r), ("infid_t", i), ("infid_x", i)):
                x, y = _xy(rows, "eps", metric)
                f = _fit(x, y, True)
                if f:
                    fits[("eps", "power", metric, J2, N, w)] = f
            x, y = _xy(i, "n_steps", "d_trotter")
            f = _fit(x, y, True)
            if f:
                fits[("n", "power", "d_trotter", J2, N, w)] = f
    return fits


# ======================================================================
# Figures
# ======================================================================
def _style():
    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.3,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.family": "serif", "mathtext.fontset": "cm",
        "axes.spines.top": False, "axes.spines.right": False})


def _save_fig(fig, cfg, name):
    figdir = os.path.join(cfg.out_dir, "figures")
    os.makedirs(figdir, exist_ok=True)
    for ax in fig.axes:                      # log axes with no positive data -> linear
        for axis, getter, setter in (("x", ax.get_xscale, ax.set_xscale),
                                     ("y", ax.get_yscale, ax.set_yscale)):
            if getter() != "log":
                continue
            ok = False
            for ln in ax.lines:
                d = np.asarray(ln.get_xdata() if axis == "x" else ln.get_ydata(), float)
                if d.size and np.any(np.isfinite(d) & (d > 0)):
                    ok = True
                    break
            if not ok:
                setter("linear")
    fig.tight_layout()
    for ext in cfg.fig_formats:
        fig.savefig(os.path.join(figdir, f"{name}.{ext}"))
    if cfg.show:
        plt.show()
    else:
        plt.close(fig)
    return name


def _jlab(J2):
    return rf"$J_2/J_1={J2:g}$"


def _fl(fits, metric, J2):
    f = fits.get(("N", "power", metric, float(J2), None, None))
    return rf" ($\sim N^{{{f['slope']:.2f}}}$)" if f else ""


def _loglog(ax, Ns=None):
    ax.set_xscale("log")
    ax.set_yscale("log")
    if Ns is not None and len(Ns):
        ax.set_xticks(list(Ns))
        ax.set_xticklabels([str(int(n)) for n in Ns])
        ax.minorticks_off()


def fig_resources(cfg, R, fits):
    base = _sel(R, eps=cfg.eps_base, which=cfg.which)
    if not base:
        return None
    J2s = sorted({r["J2"] for r in base})
    Ns = sorted({int(r["N"]) for r in base})
    fig, axs = plt.subplots(2, 3, figsize=(7.4, 4.8))
    a = axs.ravel()
    for i, J2 in enumerate(J2s):
        rows, c = _sel(base, J2=J2), f"C{i}"
        for ax, metric in ((a[0], "full_cx"), (a[2], "full_depth")):
            x, y = _xy(rows, "N", metric)
            ax.plot(x, y, "o-", color=c, ms=3, label=_jlab(J2) + _fl(fits, metric, J2))
        x, y = _xy(rows, "N", "n_steps")
        a[3].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2) + " steps")
        x, y = _xy(rows, "N", "n_pulses")
        a[3].plot(x, y, "s--", color=c, ms=3)
        x, y = _xy(rows, "N", "T_scaled")
        a[4].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2))
        x, y = _xy(rows, "N", "T_phys")
        a[4].plot(x, y, "^--", color=c, ms=3)
    rows = _sel(base, J2=J2s[0])
    for metric, fmt, lab in (("trial_cx", "s-", "trial"), ("filter_cx", "^-", "filter"),
                             ("full_cx", "o-", "trial + filter")):
        x, y = _xy(rows, "N", metric)
        a[1].plot(x, y, fmt, ms=3, label=lab)
    for metric, fmt, lab in (("step_cx", "o-", "CX / Trotter step"),
                             ("n_terms", "s-", "Pauli terms"),
                             ("step_depth", "^-", "depth / step")):
        x, y = _xy(rows, "N", metric)
        a[5].plot(x, y, fmt, ms=3, label=lab)
    a[0].set_title("Full-circuit CX")
    a[1].set_title(f"CX decomposition ({_jlab(J2s[0])})")
    a[2].set_title("Full-circuit depth")
    a[3].set_title("Trotter steps (solid), pulses (dashed)")
    a[4].set_title("Evolution time: scaled (o), physical (^)")
    a[5].set_title(f"Per-step cost ({_jlab(J2s[0])})")
    for k in (0, 1, 2, 5):
        _loglog(a[k], Ns)
    for k in (3, 4):
        a[k].set_xticks(Ns)
    for k in range(6):
        a[k].set_xlabel("$N$")
        a[k].legend()
    return _save_fig(fig, cfg, "fig1_resources_vs_N")


def fig_aux(cfg, R):
    base = _sel(R, eps=cfg.eps_base, which=cfg.which)
    if not base:
        return None
    J2s = sorted({r["J2"] for r in base})
    Ns = sorted({int(r["N"]) for r in base})
    fig, axs = plt.subplots(2, 2, figsize=(5.4, 4.4))
    a = axs.ravel()
    for i, J2 in enumerate(J2s):
        rows, c = _sel(base, J2=J2), f"C{i}"
        x, y = _xy(rows, "N", "gamma")
        a[0].plot(x, 1 - y, "o-", color=c, ms=3, label=_jlab(J2))
        x, y = _xy(rows, "N", "p_succ_lb")
        a[1].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2))
        x, y = _xy(rows, "N", "W")
        a[2].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2) + " $W$")
        x, y = _xy(rows, "N", "gap_raw")
        a[2].plot(x, y, "s--", color=c, ms=3)
        x, y = _xy(rows, "N", "full_t_count_est")
        a[3].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2))
    a[0].set_title(r"Trial infidelity $1-\gamma$")
    a[0].set_yscale("log")
    a[1].set_title(r"$P_{succ}$ lower bound")
    a[2].set_title(r"Bandwidth $W$ (solid), gap (dashed)")
    a[3].set_title("Approx. T-count (full circuit)")
    _loglog(a[3], Ns)
    for ax in a:
        ax.set_xlabel("$N$")
        ax.legend()
    return _save_fig(fig, cfg, "fig2_aux_vs_N")


def fig_routing(cfg, R):
    base = _sel(R, eps=cfg.eps_base, which=cfg.which)
    labels = sorted({k[len("route["):k.index("]")] for r in base for k in r
                     if k.startswith("route[") and k.endswith("]_n2q")})
    if not labels:
        return None
    J2s = sorted({r["J2"] for r in base})
    Ns = sorted({int(r["N"]) for r in base})
    fig, axs = plt.subplots(1, 3, figsize=(7.4, 2.7))
    for j, J2 in enumerate(J2s):
        rows = _sel(base, J2=J2)
        ls = ["-", "--", ":"][j % 3]
        x, y = _xy(rows, "N", "full_cx")
        axs[0].plot(x, y, "k" + ("-" if j == 0 else ls), lw=0.8,
                    label="logical CX" if j == 0 else None)
        for li, lab in enumerate(labels):
            c = f"C{li}"
            x, y = _xy(rows, "N", f"route[{lab}]_n2q")
            axs[0].plot(x, y, "o", ls=ls, color=c, ms=3, label=f"{lab}, {_jlab(J2)}")
            lx, ly = _xy(rows, "N", "full_cx")
            if len(x) and len(lx):
                ratio = {int(a_): b_ for a_, b_ in zip(lx, ly)}
                axs[1].plot(x, [yy / ratio[int(xx)] for xx, yy in zip(x, y) if int(xx) in ratio],
                            "o", ls=ls, color=c, ms=3, label=f"{lab}, {_jlab(J2)}")
            x, y = _xy(rows, "N", f"route[{lab}]_dur_us")
            if len(x):
                axs[2].plot(x, y, "o", ls=ls, color=c, ms=3, label=f"{lab}, {_jlab(J2)}")
    axs[0].set_title("Routed 2q gates")
    axs[1].set_title("Routing overhead (2q / logical CX)")
    axs[2].set_title(r"Est. duration ($\mu$s)")
    _loglog(axs[0], Ns)
    for ax in axs:
        ax.set_xlabel("$N$")
        ax.legend()
    return _save_fig(fig, cfg, "fig3_routing_vs_N")


def fig_ideal_infidelity(cfg, I):
    base = _sel(I, eps=cfg.eps_base, which=cfg.which)
    if not base:
        return None
    J2s = sorted({r["J2"] for r in base})
    nc = min(len(J2s), 3)
    nr = math.ceil(len(J2s) / nc)
    fig, axs = plt.subplots(nr, nc, figsize=(2.6 * nc + 0.4, 2.5 * nr), squeeze=False)
    for ax, J2 in zip(axs.ravel(), J2s):
        rows = _sel(base, J2=J2)
        for key, fmt, lab in (("infid_b", "s-", "before filter"),
                              ("infid_x", "^--", "exact filter"),
                              ("infid_t", "o-", "Trotterized circuit")):
            x, y = _xy(rows, "N", key)
            ax.plot(x, y, fmt, ms=3, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("$N$")
        ax.set_title(_jlab(J2))
    axs.ravel()[0].set_ylabel(r"infidelity $1-F$")
    axs.ravel()[0].legend()
    for ax in axs.ravel()[len(J2s):]:
        ax.axis("off")
    return _save_fig(fig, cfg, "fig4_ideal_infidelity_vs_N")


def _heat(ax, M, Ns, J2s, title, fmt, cmap="viridis"):
    im = ax.imshow(M, origin="lower", aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(J2s)))
    ax.set_xticklabels([f"{j:g}" for j in J2s])
    ax.set_yticks(range(len(Ns)))
    ax.set_yticklabels([str(n) for n in Ns])
    ax.set_xlabel("$J_2/J_1$")
    ax.set_ylabel("$N$")
    ax.set_title(title)
    ax.grid(False)
    lo, hi = np.nanmin(M) if np.isfinite(M).any() else 0, np.nanmax(M) if np.isfinite(M).any() else 1
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                dark = (M[i, j] - lo) / (hi - lo + 1e-12) < 0.5
                ax.text(j, i, fmt.format(M[i, j]), ha="center", va="center", fontsize=6.5,
                        color="w" if dark else "k")
    return im


def fig_ideal_summary(cfg, I):
    base = _sel(I, eps=cfg.eps_base, which=cfg.which)
    if not base:
        return None
    J2s = sorted({r["J2"] for r in base})
    Ns = sorted({int(r["N"]) for r in base})

    def grid(f):
        M = np.full((len(Ns), len(J2s)), np.nan)
        for r in base:
            M[Ns.index(int(r["N"])), [_same(r["J2"], j) for j in J2s].index(True)] = f(r)
        return M
    Fb, Ft = grid(lambda r: r["F_b"]), grid(lambda r: r["F_t"])
    gain = grid(lambda r: math.log10(max(r["infid_b"], 1e-16) / max(r["infid_t"], 1e-16))
                if np.isfinite(r["infid_t"]) and np.isfinite(r["infid_b"]) else NAN)
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.0))
    a = axs.ravel()
    _heat(a[0], Fb, Ns, J2s, "$F$ before filter", "{:.3f}")
    _heat(a[1], Ft, Ns, J2s, "$F$ after Trotterized filter", "{:.4f}")
    _heat(a[2], gain, Ns, J2s, r"$\log_{10}$ infidelity reduction", "{:.2f}", cmap="magma")
    for i, J2 in enumerate(J2s):
        rows, c = _sel(base, J2=J2), f"C{i}"
        x, y = _xy(rows, "N", "dE_per_site_b")
        a[3].plot(x, y, "s--", color=c, ms=3, label=_jlab(J2) + " before")
        x, y = _xy(rows, "N", "dE_per_site_t")
        a[3].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2) + " after")
        x, y = _xy(rows, "N", "P_trot")
        a[4].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2) + " Trotter")
        x, y = _xy(rows, "N", "P_exact")
        a[4].plot(x, y, "^--", color=c, ms=3)
        x, y = _xy(rows, "N", "d_trotter")
        a[5].plot(x, y, "o-", color=c, ms=3, label=_jlab(J2))
    a[3].set_title(r"$(E-E_0)/N$"); a[3].set_yscale("log")
    a[4].set_title(r"$P_{succ}$: Trotter (o), exact (^)")
    a[5].set_title(r"Trotter-only infidelity $1-|\langle\psi_{ex}|\psi_{T}\rangle|^2$")
    a[5].set_yscale("log")
    for k in (3, 4, 5):
        a[k].set_xlabel("$N$")
        a[k].legend()
    return _save_fig(fig, cfg, "fig5_ideal_summary")


def fig_eps(cfg, R, I, fits):
    if not cfg.eps_cases or not any(_sel(R, N=N, J2=J2) for (N, J2) in cfg.eps_cases):
        return None
    fig, axs = plt.subplots(2, 3, figsize=(7.6, 5.0))
    a = axs.ravel()
    eg = np.array(sorted(cfg.eps_list))
    a[0].plot(eg, eg, "k:", lw=0.8, label=r"$1-F=\epsilon$")
    for (N, J2) in cfg.eps_cases:
        ci = list(cfg.eps_cases).index((N, J2))
        for w in cfg.eps_which:
            ls = "-" if w == "empirical" else "--"
            lab = rf"$N={N},\,J_2={J2:g}$" + ("" if len(cfg.eps_which) == 1 else f" ({w[:4]})")
            r, i = _sel(R, N=N, J2=J2, which=w), _sel(I, N=N, J2=J2, which=w)
            kw = dict(marker="o", ls=ls, color=f"C{ci}", ms=3)

            def sl(scope, metric):
                f = fits.get((scope, "power", metric, J2, N, w))
                return rf" ($\sim x^{{{f['slope']:.2f}}}$)" if f else ""
            x, y = _xy(i, "eps", "infid_t"); a[0].plot(x, y, label=lab, **kw)
            x, y = _xy(i, "eps", "infid_x"); a[1].plot(x, y, label=lab, **kw)
            x, y = _xy(r, "eps", "n_steps"); a[2].plot(x, y, label=lab + sl("eps", "n_steps"), **kw)
            x, y = _xy(r, "eps", "full_cx"); a[3].plot(x, y, label=lab + sl("eps", "full_cx"), **kw)
            x, y = _xy(i, "n_steps", "d_trotter"); a[4].plot(x, y, label=lab + sl("n", "d_trotter"), **kw)
            x, y = _xy(i, "eps", "P_trot"); a[5].plot(x, y, label=lab, **kw)
            x, y = _xy(r, "eps", "p_succ_lb"); a[5].plot(x, y, ls=":", color=f"C{ci}", lw=0.8)
    titles = [r"Final infidelity (Trotterized) vs target $\epsilon$",
              r"Exact-filter infidelity vs $\epsilon$", r"Trotter steps $n$ vs $\epsilon$",
              r"Full-circuit CX vs $\epsilon$", r"Trotter-only infidelity vs $n$",
              r"$P_{succ}$ (solid), lower bound (dotted)"]
    xl = [r"$\epsilon$", r"$\epsilon$", r"$\epsilon$", r"$\epsilon$", "$n$", r"$\epsilon$"]
    for ax, t, x_ in zip(a, titles, xl):
        ax.set_title(t, fontsize=8)
        ax.set_xlabel(x_)
        ax.legend()
    for k in range(5):
        _loglog(a[k])
    a[5].set_xscale("log")
    for k in (0, 2, 3):
        a[k].invert_xaxis()
    return _save_fig(fig, cfg, "fig6_eps_scaling")


# ======================================================================
# Report
# ======================================================================
def _n(x, f=".3g"):
    try:
        if x is None or (isinstance(x, float) and not np.isfinite(x)):
            return "–"
        return format(x, f)
    except Exception:
        return str(x)


def _md(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _write_csv(path, rows):
    if not rows:
        return
    keys: List[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (list, dict)) else v)
                        for k, v in r.items()})


def _meta():
    try:
        import qiskit_aer
        aer = qiskit_aer.__version__
    except Exception:
        aer = "n/a"
    return dict(date=datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                python=sys.version.split()[0], platform=platform.platform(),
                numpy=np.__version__, qiskit=qiskit.__version__, qiskit_aer=aer)


def _cfg_dict(cfg):
    d = {}
    for k, v in cfg.__dict__.items():
        if k == "routing":
            v = [dict(label=s.label, backend=getattr(s.backend, "name", None),
                      coupling=("callable" if callable(s.coupling) else str(s.coupling)))
                 for s in v]
        elif k == "noisy_targets":
            v = [dict(label=t.label, **t.description) for t in v]
        d[k] = v
    return d


def build_report(cfg, R, I, fits, figs, errs):
    meta = _meta()
    L = [f"# Filter study report", f"*generated {meta['date']}*", "",
         "## 1. Configuration", "```json", json.dumps(_cfg_dict(cfg), indent=1, default=str), "```",
         "", "Environment: " + ", ".join(f"{k}={v}" for k, v in meta.items() if k != "date"), ""]

    base = _sel(R, eps=cfg.eps_base, which=cfg.which)
    L += ["## 2. Circuit resources vs N (not simulated)",
          f"Design: `{cfg.which}`, eps = {cfg.eps_base:g}. Counts are on the all-zeros "
          "(success) path, before routing, in the h/s/cx/rz basis.", ""]
    for J2 in sorted({r["J2"] for r in base}):
        rows = sorted(_sel(base, J2=J2), key=lambda r: r["N"])
        L += [f"**J2/J1 = {J2:g}**", "", _md(
            ["N", "pulses", "n", "k", "T (scaled)", "T (phys)", "CX trial", "CX filter", "CX full",
             "depth full", "T-count", "gamma", "P_succ lb"],
            [[r["N"], r["n_pulses"], r["n_steps"], r["k"], _n(r["T_scaled"]), _n(r["T_phys"]),
              _n(r["trial_cx"], ".0f"), _n(r["filter_cx"], ".0f"), _n(r["full_cx"], ".0f"),
              _n(r["full_depth"], ".0f"), _n(r["full_t_count_est"], ".0f"),
              _n(r["gamma"], ".4f"), _n(r["p_succ_lb"], ".3f")] for r in rows]), ""]
    labels = sorted({k[len("route["):k.index("]")] for r in base for k in r
                     if k.startswith("route[") and k.endswith("]_n2q")})
    if labels:
        L += ["### Routed 2q counts", ""]
        for J2 in sorted({r["J2"] for r in base}):
            rows = sorted(_sel(base, J2=J2), key=lambda r: r["N"])
            L += [f"**J2/J1 = {J2:g}**", "", _md(
                ["N", "logical CX"] + [f"{l}: 2q / depth / dur µs" for l in labels],
                [[r["N"], _n(r["full_cx"], ".0f")] +
                 [f"{_n(r.get(f'route[{l}]_n2q'), '.0f')} / {_n(r.get(f'route[{l}]_depth'), '.0f')} / "
                  f"{_n(r.get(f'route[{l}]_dur_us'), '.0f')}" for l in labels] for r in rows]), ""]

    L += ["### Scaling fits over N (N >= %d)" % cfg.fit_min_N,
          "Power law y ~ N^p and exponential y ~ exp(bN); compare R². Few points: indicative only.", ""]
    fr = []
    for (scope, model, metric, J2, N_, w), f in sorted(fits.items(), key=lambda kv: str(kv[0])):
        if scope == "N" and model == "power":
            e = fits.get(("N", "exp", metric, J2, None, None))
            fr.append([f"{J2:g}", metric, _n(f["slope"], ".2f"), _n(f["r2"], ".4f"),
                       _n(e["slope"], ".3f") if e else "–", _n(e["r2"], ".4f") if e else "–", f["npts"]])
    L += [_md(["J2", "metric", "p", "R² (power)", "b", "R² (exp)", "pts"], fr), ""]

    idl = _sel(I, eps=cfg.eps_base, which=cfg.which)
    L += ["## 3. Ideal filter effect vs N and J2",
          f"Noiseless, post-selected, eps = {cfg.eps_base:g}. F is the overlap with the ED ground state "
          "when available, else DMRG (column `ref`).", ""]
    rows = sorted(idl, key=lambda r: (r["J2"], r["N"]))
    L += [_md(["J2", "N", "ref", "F before", "F exact filt.", "F Trotter", "1-F before", "1-F Trotter",
               "gain", "P_succ (T)", "(E-E0)/N before", "(E-E0)/N after", "Trotter-only 1-F"],
              [[f"{r['J2']:g}", r["N"], r["ref_source"], _n(r["F_b"], ".4f"), _n(r["F_x"], ".5f"),
                _n(r["F_t"], ".5f"), _n(r["infid_b"]), _n(r["infid_t"]), _n(r["gain_t"], ".1f"),
                _n(r["P_trot"], ".3f"), _n(r["dE_per_site_b"]), _n(r["dE_per_site_t"]),
                _n(r["d_trotter"])] for r in rows]), ""]

    L += ["## 4. Ideal runs vs epsilon", ""]
    for (N, J2) in cfg.eps_cases:
        for w in cfg.eps_which:
            rr = sorted(_sel(R, N=N, J2=J2, which=w), key=lambda r: -r["eps"])
            if not rr:
                continue
            L += [f"**N={N}, J2/J1={J2:g}, design `{w}`**", ""]
            tab = []
            for r in rr:
                i = (_sel(I, N=N, J2=J2, which=w, eps=r["eps"]) or [{}])[0]
                tab.append([f"{r['eps']:g}", r["n_steps"], r["n_pulses"], _n(r["full_cx"], ".0f"),
                            _n(r["full_depth"], ".0f"), _n(i.get("infid_x")), _n(i.get("infid_t")),
                            _n(i.get("d_trotter")), _n(i.get("trotter_infid_bound")),
                            _n(i.get("P_trot"), ".3f"), _n(r["p_succ_lb"], ".3f"),
                            _n(r["eps_bound"], ".3g")])
            L += [_md(["eps", "n", "pulses", "CX", "depth", "1-F exact", "1-F Trotter",
                       "Trotter-only", "rigorous bound", "P_succ", "P_succ lb", "eps_bound"], tab), ""]
    fe = [[f"{k[4]}", f"{k[3]:g}", k[5], k[2], k[0] + ("" if k[0] == "eps" else " (x=n)"),
           _n(f["slope"], ".2f"), _n(f["r2"], ".4f"), f["npts"]]
          for k, f in sorted(fits.items(), key=lambda kv: str(kv[0])) if k[0] in ("eps", "n")]
    L += ["### Fitted exponents (log-log slope)", "",
          _md(["N", "J2", "design", "y", "x", "slope", "R²", "pts"], fe), ""]

    noisy = []
    nd = os.path.join(cfg.out_dir, "noisy")
    if os.path.isdir(nd):
        for fn in sorted(os.listdir(nd)):
            if fn.endswith(".json"):
                with open(os.path.join(nd, fn)) as f:
                    noisy.append((fn[:-5], json.load(f)))
    if noisy:
        L += ["## 5. Hardware-sim spot checks", ""]
        for tag, rep in noisy:
            L += [f"**{tag}**", "", _md(
                ["row", "P_succ", "E", "E-E0", "F_ED", "purity", "dF*", "2q", "depth", "t µs"],
                [[s["label"], _n(s["P_succ"], ".3f"), _n(s["E"], ".4f"), _n(s["dE"], "+.4f"),
                  _n(s["F_ed"], ".4f"), _n(s["purity"], ".3f"), _n(s["dF_vs_trial"], "+.3f"),
                  _n(s["n_2q"], ".0f"), _n(s["depth"], ".0f"), _n(s["dur_us"], ".0f")]
                 for s in rep["summary"]]), ""]

    L += ["## Figures", ""] + [f"![{f}](figures/{f}.png)" for f in figs if f] + [""]
    L += ["## Failed points", ""]
    L += ([f"- {e['kind']} N={e['N']} J2={e['J2']:g} eps={e['eps']:g} {e['which']}: {e['error']}"
           for e in errs] or ["none"])
    return "\n".join(L)


# ======================================================================
# Entry points
# ======================================================================
def analyze(cfg, verbose=True):
    _style()
    R, I, errs = load_rows(cfg, "res"), load_rows(cfg, "ideal"), load_errors(cfg)
    if not R and not I:
        print(f"[analyze] no successful points in {cfg.out_dir}/data "
              f"({len(errs)} failed). Fix the errors above (see data/*.err), then rerun; "
              "figures and report will be mostly empty until then.")
    fits = compute_fits(cfg, R, I)
    figs = [fig_resources(cfg, R, fits), fig_aux(cfg, R), fig_routing(cfg, R),
            fig_ideal_infidelity(cfg, I), fig_ideal_summary(cfg, I), fig_eps(cfg, R, I, fits)]
    text = build_report(cfg, R, I, fits, figs, errs)
    os.makedirs(cfg.out_dir, exist_ok=True)
    with open(os.path.join(cfg.out_dir, "report.md"), "w") as f:
        f.write(text)
    _write_csv(os.path.join(cfg.out_dir, "all_resources.csv"), R)
    _write_csv(os.path.join(cfg.out_dir, "all_ideal.csv"), I)
    _dump(os.path.join(cfg.out_dir, "config.json"), dict(config=_cfg_dict(cfg), env=_meta()))
    if verbose:
        print(text)
        print(f"\n[saved] {cfg.out_dir}/report.md, figures/, all_resources.csv, all_ideal.csv")
    return dict(R=R, I=I, fits=fits, figures=[f for f in figs if f], errors=errs, report=text)


def run_study(cfg: Optional[StudyConfig] = None, *, compute_data=True, noisy=True,
              analyze_data=True, ns=None, circuits_fn=None, verbose=True, **overrides):
    """compute (resumable) -> optional noisy spot checks -> figures + report.
    ns: namespace holding the pipeline (default: notebook globals; e.g. globals()
        or vars(module)).  circuits_fn: alternatively, a function
        (N, J2, eps, which) -> (trial_qc, filter_qc, full_qc, info)."""
    cfg = cfg or StudyConfig()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    os.makedirs(cfg.out_dir, exist_ok=True)
    fac = None
    if compute_data or (noisy and cfg.noisy_cases):
        fac = CaseFactory(ns=ns, J1=cfg.J1, circuits_fn=circuits_fn)
    if compute_data:
        compute(cfg, fac, verbose)
    if noisy:
        run_noisy(cfg, fac, verbose)
    return analyze(cfg, verbose) if analyze_data else None
