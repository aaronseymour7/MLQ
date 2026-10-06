"""
case_report.py -- one call -> complete, saved report for one filter case.

    from case_report import run_case, fake_target, ideal_target, depolarizing_target
    rep = run_case(N=6, J2=0.0, eps=1e-2, which="emp",
                   targets=[ideal_target(), fake_target(FakeLagosV2())],
                   shots=8000, out_dir="reports")

Everything is printed AND stored in the returned dict (and in out_dir as .txt /
.json / .npz), so nothing has to be re-run to look something up:

  inputs            all case / run / target parameters
  design            pulses, times, phases, Trotter steps k, n, total time, gamma, bounds
  logical_resources pre-routing success-path gate counts (trial, filter, full), per-step cost
  reference         exact-diagonalization numbers: E0, trial state, exact filter, ED Trotter
  runs[target]      routed gate counts / depth / est. duration, P_succ, abort histogram,
                    E, F (vs ED and DMRG), purity -- for filtered AND trial-only circuits
  summary           one table with every row above

Modularity
  * Backends: a `Target` bundles (Aer simulator, transpile kwargs). Use
    ideal_target(), fake_target(backend), depolarizing_target(p2, ...), or build
    your own Target(...). Pass any list to run_case(targets=[...]).
  * Case variables: N, J1, J2, eps, which, shots, optimization_level, seeds.
  * Circuit source: export_fn (defaults to `export_circuits` in your notebook
    namespace), or pass circuits=(trial_qc, filter_qc, full_qc, info) directly.

Assumes `builder.py` and `hamiltonians.py` are importable (same modules the
pipeline already uses). Noise is simulated with the density-matrix method, so
the *transpiled circuit's active qubits* must be <= max_dm_qubits (13 default).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import scipy.sparse.linalg as sla

from qiskit import transpile
from qiskit.quantum_info import DensityMatrix, Statevector, partial_trace
from qiskit.transpiler import CouplingMap
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error

from core.circuits import flatten_success_path
from core.resources import cx_depth_per_step, resource_costs
from core.simulate import postselected_run_exact
from core.trotter import ed_error
from pipeline import export_circuits
from hamiltonians import reorder_axes

NAN = float("nan")
NATIVE_BASIS = ["rz", "sx", "x", "cx", "id"]
EXTRA_OPS = ["measure", "reset", "if_else", "delay"]
NONGATE = {"barrier", "delay", "snapshot", "measure", "reset"}


# ======================================================================
# Targets (swap these to change backend / noise)
# ======================================================================
@dataclass
class Target:
    """label: name in the report. simulator: AerSimulator (density_matrix).
    transpile_kwargs: passed to qiskit.transpile (backend=... or
    basis_gates=/coupling_map=). backend: object with .target (for durations)."""
    label: str
    simulator: Any
    transpile_kwargs: dict
    backend: Any = None
    description: dict = field(default_factory=dict)


def ideal_target(backend=None, basis=NATIVE_BASIS, label=None):
    """Noiseless. backend=None: all-to-all, no routing (algorithmic error only).
    backend given: same routing/basis as that device, but no noise."""
    sim = AerSimulator(method="density_matrix")
    if backend is None:
        kw = dict(basis_gates=list(basis) + EXTRA_OPS)
        desc = dict(noise="none", connectivity="all-to-all", basis=list(basis))
        lab = label or "ideal"
    else:
        kw = dict(backend=backend)
        desc = dict(noise="none", connectivity=f"routed to {backend.name}")
        lab = label or f"ideal@{backend.name}"
    return Target(lab, sim, kw, backend, desc)


def fake_target(backend, label=None):
    """Noisy simulation with the fake backend's noise model + coupling map."""
    sim = AerSimulator.from_backend(backend, method="density_matrix")
    desc = dict(noise="backend noise model", backend=backend.name,
                n_qubits=backend.num_qubits)
    return Target(label or backend.name, sim, dict(backend=backend), backend, desc)


def depolarizing_target(p2, p1_ratio=0.1, p_readout=0.0, coupling_map=None,
                        basis=("rz", "sx", "x", "cx"), label=None):
    """Simple depolarizing model: cx error p2, 1q (sx,x) error p2*p1_ratio,
    symmetric readout error p_readout. coupling_map=None -> all-to-all."""
    nm = NoiseModel(basis_gates=list(basis))
    nm.add_all_qubit_quantum_error(depolarizing_error(p2, 2), ["cx"])
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 * p1_ratio, 1), ["sx", "x"])
    if p_readout > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]]))
    kw = dict(basis_gates=list(basis) + ["id"] + EXTRA_OPS)
    conn = "all-to-all"
    if coupling_map is not None:
        cm = coupling_map if isinstance(coupling_map, CouplingMap) else CouplingMap(coupling_map)
        kw["coupling_map"] = cm
        conn = f"custom ({len(cm.get_edges())} edges)"
    sim = AerSimulator(method="density_matrix", noise_model=nm)
    desc = dict(noise="depolarizing", p_cx=p2, p_1q=p2 * p1_ratio,
                p_readout=p_readout, connectivity=conn)
    return Target(label or f"depol p={p2:g}", sim, kw, None, desc)


def default_targets():
    from qiskit_ibm_runtime.fake_provider import FakeLagosV2
    return [ideal_target(), fake_target(FakeLagosV2())]


# ======================================================================
# Helpers
# ======================================================================
def _from_main(name, given=None, default=None):
    if given is not None:
        return given
    import __main__
    if hasattr(__main__, name):
        return getattr(__main__, name)
    if default is not None:
        return default
    raise NameError(f"'{name}' not found in the notebook namespace; pass it explicitly")


def _circuit_stats(circ):
    """Success-path (control flow inlined) gate statistics of a (transpiled) circuit."""
    flat = flatten_success_path(circ)
    ops = {k: int(v) for k, v in flat.count_ops().items()}
    n2 = sum(1 for i in flat.data
             if i.operation.num_qubits == 2 and i.operation.name not in NONGATE)
    n1 = sum(1 for i in flat.data
             if i.operation.num_qubits == 1 and i.operation.name not in NONGATE)
    try:
        d2 = flat.depth(lambda i: i.operation.num_qubits == 2)
    except Exception:
        d2 = None
    return flat, dict(ops=ops, n_2q=n2, n_1q=n1, depth=flat.depth(), depth_2q=d2)


def _estimate_duration(flat, target):
    """ASAP critical-path duration (seconds) from the backend's instruction
    durations. Ignores classical feed-forward latency."""
    avail: Dict[int, float] = {}
    end_all, missing = 0.0, set()
    for inst in flat.data:
        name = inst.operation.name
        if name == "barrier":
            continue
        qs = tuple(flat.find_bit(q).index for q in inst.qubits)
        dur = None
        try:
            if name in target:
                p = target[name].get(qs)
                dur = None if p is None else p.duration
        except Exception:
            dur = None
        if dur is None:
            dur = 0.0
            if name != "rz":
                missing.add(name)
        start = max((avail.get(q, 0.0) for q in qs), default=0.0)
        for q in qs:
            avail[q] = start + dur
        end_all = max(end_all, start + dur)
    return end_all, sorted(missing)


def _t1_t2_us(target, qubits):
    try:
        t1 = [target.qubit_properties[q].t1 for q in qubits]
        t2 = [target.qubit_properties[q].t2 for q in qubits]
        return (float(np.median([x for x in t1 if x])) * 1e6,
                float(np.median([x for x in t2 if x])) * 1e6)
    except Exception:
        return None, None


def _prep(circ, tgt, opt, seed, N):
    tqc = transpile(circ, optimization_level=opt, seed_transpiler=seed,
                    **tgt.transpile_kwargs)
    flat, stats = _circuit_stats(tqc)
    lay = tqc.layout
    phys_of = list(lay.final_index_layout()) if lay is not None else list(range(N))
    active = sorted({tqc.find_bit(q).index for inst in tqc.data for q in inst.qubits}
                    | set(phys_of[:N + 1]))
    stats["active_qubits"] = active
    stats["layout_logical_to_physical"] = phys_of[:N + 1]
    if tgt.backend is not None:
        dur, missing = _estimate_duration(flat, tgt.backend.target)
        t1, t2 = _t1_t2_us(tgt.backend.target, active)
        stats.update(duration_us=dur * 1e6, duration_missing_ops=missing,
                     median_T1_us=t1, median_T2_us=t2)
    return tqc, stats, phys_of, active


def _system_rho(dm, active, phys_of, N):
    """Saved density matrix over `active` physical qubits -> N-qubit system
    density matrix in logical (little-endian) order, ancilla/idle traced out."""
    dm = dm if isinstance(dm, DensityMatrix) else DensityMatrix(np.asarray(dm))
    keep = [active.index(phys_of[i]) for i in range(N)]
    drop = [q for q in range(dm.num_qubits) if q not in keep]
    red = partial_trace(dm, drop) if drop else dm
    R = np.asarray(red.data)
    srt = sorted(keep)
    src = [srt.index(p) for p in keep]
    T = R.reshape([2] * (2 * N))
    ax = [N - 1 - src[N - 1 - a] for a in range(N)]
    R = T.transpose(ax + [N + a for a in ax]).reshape(2 ** N, 2 ** N)
    R = (R + R.conj().T) / 2
    return R / np.trace(R).real


def _build_reference(info):
    H = info["H_qk"]
    psi_ed = reorder_axes(np.asarray(info["psi0_ed"]).reshape(-1)).astype(complex)
    psi_ed /= np.linalg.norm(psi_ed)
    psi_dm = None
    try:
        psi_dm = reorder_axes(np.asarray(info["psi0_dmrg"]).reshape(-1)).astype(complex)
        psi_dm /= np.linalg.norm(psi_dm)
    except Exception:
        pass
    Hm = H.to_matrix(sparse=True)
    try:
        E0 = float(sla.eigsh(Hm, k=1, which="SA", return_eigenvectors=False)[0])
    except Exception:
        E0 = float(np.real(psi_ed.conj() @ (Hm @ psi_ed)))
    return dict(H=H, psi_ed=psi_ed, psi_dmrg=psi_dm, E0=E0)


def _metrics(rho, ref):
    if rho is None:
        return dict(E=NAN, dE=NAN, F_ed=NAN, F_dmrg=NAN, purity=NAN)
    rho = np.asarray(rho)
    E = float(DensityMatrix(rho).expectation_value(ref["H"]).real)
    F_ed = float(np.real(ref["psi_ed"].conj() @ rho @ ref["psi_ed"]))
    F_dm = (float(np.real(ref["psi_dmrg"].conj() @ rho @ ref["psi_dmrg"]))
            if ref["psi_dmrg"] is not None else NAN)
    return dict(E=E, dE=E - ref["E0"], F_ed=F_ed, F_dmrg=F_dm,
                purity=float(np.real(np.trace(rho @ rho))))


def _abort_histogram(counts, m, shots):
    """Fraction of shots that succeeded / aborted at pulse j (lowest set bit)."""
    hist = {"success": 0.0, **{f"abort@{j}": 0.0 for j in range(m)}}
    for key, c in counts.items():
        v = int(key.replace(" ", ""), 2)
        if v == 0:
            hist["success"] += c / shots
        else:
            hist[f"abort@{(v & -v).bit_length() - 1}"] += c / shots
    return hist


# ======================================================================
# Per-target runs
# ======================================================================
def _run_filtered(tgt, full_qc, ref, N, m, shots, opt, seed_t, seed_s, max_dm):
    tqc, stats, phys_of, active = _prep(full_qc, tgt, opt, seed_t, N)
    if len(active) > max_dm:
        raise RuntimeError(f"{len(active)} active qubits > max_dm_qubits={max_dm}")
    qc = tqc.copy()
    qc.save_density_matrix(qubits=active, label="rho", conditional=True)
    t0 = time.time()
    res = tgt.simulator.run(qc, shots=shots, seed_simulator=seed_s).result()
    counts = res.get_counts()
    p = counts.get("0" * m, 0) / shots
    d = res.data()["rho"]
    rho = _system_rho(d["0x0"], active, phys_of, N) if "0x0" in d else None
    out = dict(P_succ=p, P_succ_err=float(np.sqrt(max(p * (1 - p), 0) / shots)),
               abort_histogram=_abort_histogram(counts, m, shots),
               n_success_shots=int(counts.get("0" * m, 0)),
               wall_s=time.time() - t0, transpile=stats, **_metrics(rho, ref))
    return out, rho, tqc


def _run_trial(tgt, trial_qc, ref, N, opt, seed_t, seed_s, max_dm):
    tqc, stats, phys_of, active = _prep(trial_qc, tgt, opt, seed_t, N)
    if len(active) > max_dm:
        raise RuntimeError(f"{len(active)} active qubits > max_dm_qubits={max_dm}")
    qc = tqc.copy()
    qc.save_density_matrix(qubits=active, label="rho")
    res = tgt.simulator.run(qc, shots=1, seed_simulator=seed_s).result()
    rho = _system_rho(res.data()["rho"], active, phys_of, N)
    return dict(transpile=stats, **_metrics(rho, ref)), rho


# ======================================================================
# Logical (pre-routing) resources and references
# ======================================================================
def _logical_resources(trial_qc, filter_qc, full_qc, info, order):
    keys = ("cx", "h", "s", "rz", "total", "depth", "measure", "reset",
            "rz_nonclifford", "t_count_est")

    def pick(d):
        return {k: d[k] for k in keys}
    out = dict(
        trial=pick(resource_costs(trial_qc, verbose=False)),
        filter_success_path=pick(resource_costs(filter_qc, flatten=True, verbose=False)),
        full_success_path=pick(resource_costs(full_qc, flatten=True, verbose=False)),
    )
    try:
        cx, dp = cx_depth_per_step(info["H_scaled"], order=order)
        out["per_trotter_step"] = dict(cx=int(cx), depth=int(dp))
    except Exception as e:
        out["per_trotter_step"] = dict(error=str(e))
    return out


def _design(info, m):
    tg, ph, k = (np.asarray(info[x]) for x in ("tg", "ph", "k"))
    W = float(info["W"])
    return dict(
        n_pulses=m, tg_scaled=tg, phases=ph, k=k.astype(int), n_steps_total=int(info["n"]),
        trotter_dt_per_pulse=tg / k, T_total_scaled=float(tg.sum()),
        T_total_unscaled=float(tg.sum() / W), W=W, shift_E0_dmrg=float(info["shift"]),
        n_pauli_terms=int(len(info["H_scaled"]) - 1), gamma=float(info["gamma"]),
        p_succ_lower_bound=float(info["p_succ_lb"]), eps_bound=float(info["eps_bound"]))


def _references(trial_qc, info, ref):
    out, states = {}, {}
    v = Statevector(trial_qc).data
    states["trial_exact"] = np.outer(v, v.conj())
    out["trial_exact"] = _metrics(states["trial_exact"], ref)
    try:
        Hs = info["H_scaled"].to_matrix(sparse=True)
        P, vec = postselected_run_exact(trial_qc, info["tg"], info["ph"], Hs)
        states["exact_filter"] = np.outer(vec, np.conj(vec))
        out["exact_filter"] = dict(P_succ=float(P), **_metrics(states["exact_filter"], ref))
    except Exception as e:
        out["exact_filter"] = dict(error=str(e))
    try:
        ed = ed_error(info["H_qk"], v, info["tg"], info["ph"], info["k"],
                      shift=info["shift"], W=info["W"])
        out["ed_error"] = {k: ed[k] for k in
                           ("eps_bound", "eps_exact_filter", "eps_trotter", "gamma",
                            "delta", "eta", "leak_bound", "trotter_dist_bound",
                            "p_succ_exact", "p_succ_trotter", "bound_holds")}
    except Exception as e:
        out["ed_error"] = dict(error=str(e))
    return out, states


# ======================================================================
# Main entry point
# ======================================================================
def run_case(N=6, J2=0.0, eps=1e-2, which="empirical", J1=None, *,
             targets: Optional[List[Target]] = None, shots=8000,
             optimization_level=2, seed_transpiler=1, seed_simulator=1234,
             order=None, export_fn=None, circuits=None,
             run_trial_baseline=True, max_dm_qubits=13,
             out_dir=None, tag=None, verbose=True) -> dict:
    """Build the case, run it on every target, return + print + save the report."""
    order = _from_main("ORDER", order, default=1) if order is None else order
    if circuits is None:
        fn = export_circuits
        kw = dict(N=N, J2=J2, eps=eps, which=which)
        if J1 is not None:
            kw["J1_"] = J1
        circuits = fn(**kw)
    trial_qc, filter_qc, full_qc, info = circuits
    N, J2, eps = info["N"], info["J2"], info["eps"]
    m = len(info["tg"])
    targets = targets if targets is not None else default_targets()
    tag = tag or f"N{N}_J2{J2:g}_eps{eps:g}_{which}"

    ref = _build_reference(info)
    rep: Dict[str, Any] = dict(tag=tag)
    rep["inputs"] = dict(
        case=dict(N=N, J1=J1, J2=J2, eps=eps, which=which, trotter_order=order),
        run=dict(shots=shots, optimization_level=optimization_level,
                 seed_transpiler=seed_transpiler, seed_simulator=seed_simulator,
                 run_trial_baseline=run_trial_baseline, max_dm_qubits=max_dm_qubits),
        targets={t.label: t.description for t in targets})
    rep["design"] = _design(info, m)
    rep["logical_resources"] = _logical_resources(trial_qc, filter_qc, full_qc, info, order)
    refs, states = _references(trial_qc, info, ref)
    refs["E0_ed"] = ref["E0"]
    rep["reference"] = refs
    rep["runs"] = {}
    circuits_out = {}

    for tgt in targets:
        entry: Dict[str, Any] = dict(description=tgt.description)
        try:
            f, rho_f, tq = _run_filtered(tgt, full_qc, ref, N, m, shots, optimization_level,
                                         seed_transpiler, seed_simulator, max_dm_qubits)
            entry["filtered"] = f
            states[f"{tgt.label}|filtered"] = rho_f
            circuits_out[tgt.label] = tq
            lcx = rep["logical_resources"]["full_success_path"]["cx"]
            f["transpile"]["n_2q_over_logical_cx"] = f["transpile"]["n_2q"] / max(lcx, 1)
        except Exception as e:
            entry["filtered"] = dict(error=f"{type(e).__name__}: {e}")
        if run_trial_baseline:
            try:
                t, rho_t = _run_trial(tgt, trial_qc, ref, N, optimization_level,
                                      seed_transpiler, seed_simulator, max_dm_qubits)
                entry["trial_only"] = t
                states[f"{tgt.label}|trial_only"] = rho_t
            except Exception as e:
                entry["trial_only"] = dict(error=f"{type(e).__name__}: {e}")
        rep["runs"][tgt.label] = entry

    rep["summary"] = _summary_rows(rep)
    rep["_states"] = states
    rep["_circuits"] = dict(trial_qc=trial_qc, filter_qc=filter_qc, full_qc=full_qc,
                            transpiled=circuits_out)
    rep["_info"] = info
    text = format_report(rep)
    rep["text"] = text
    if verbose:
        print(text)
    if out_dir:
        _save(rep, out_dir, tag)
    return rep


def sweep(cases: List[dict], **common) -> List[dict]:
    """Run run_case for each dict of case overrides (e.g. [{'N':4},{'N':6,'eps':1e-3}]),
    then print one comparison table of the filtered rows."""
    reps = [run_case(**{**common, **c, "verbose": False}) for c in cases]
    lines = [f"{'case':34s}{'target':22s}{'P_succ':>8s}{'E':>10s}{'F_ED':>8s}{'dF_vs_trial':>13s}{'2q':>6s}"]
    for r in reps:
        for row in r["summary"]:
            if row["kind"] == "filtered":
                lines.append(f"{r['tag']:34s}{row['target']:22s}{_f(row['P_succ'],'.3f'):>8s}"
                             f"{_f(row['E'],'.4f'):>10s}{_f(row['F_ed'],'.4f'):>8s}"
                             f"{_f(row['dF_vs_trial'],'+.4f'):>13s}{_f(row['n_2q'],'d'):>6s}")
    print("\n".join(lines))
    return reps


# ======================================================================
# Summary + formatting + saving
# ======================================================================
def _summary_rows(rep):
    rows = []
    for key, lab in (("trial_exact", "trial (exact)"), ("exact_filter", "exact filter (no Trotter)")):
        r = rep["reference"].get(key, {})
        if "error" not in r:
            rows.append(dict(label=lab, target="-", kind="reference",
                             P_succ=r.get("P_succ", NAN), E=r["E"], dE=r["dE"],
                             F_ed=r["F_ed"], F_dmrg=r["F_dmrg"], purity=r["purity"],
                             n_2q=None, depth=None, dur_us=None, dF_vs_trial=NAN))
    ed = rep["reference"].get("ed_error", {})
    if "error" not in ed and ed.get("eps_trotter") is not None:
        rows.append(dict(label="ED Trotter ref (ed_error)", target="-", kind="reference",
                         P_succ=ed["p_succ_trotter"], E=NAN, dE=NAN,
                         F_ed=1 - ed["eps_trotter"], F_dmrg=NAN, purity=NAN,
                         n_2q=None, depth=None, dur_us=None, dF_vs_trial=NAN))
    for lab, e in rep["runs"].items():
        t = e.get("trial_only", {})
        for kind in ("trial_only", "filtered"):
            r = e.get(kind)
            if r is None or "error" in r:
                continue
            tr = r["transpile"]
            dF = (r["F_ed"] - t["F_ed"]) if (kind == "filtered" and t and "error" not in t) else NAN
            rows.append(dict(label=f"{lab}: {kind}", target=lab, kind=kind,
                             P_succ=r.get("P_succ", NAN), E=r["E"], dE=r["dE"],
                             F_ed=r["F_ed"], F_dmrg=r["F_dmrg"], purity=r["purity"],
                             n_2q=tr["n_2q"], depth=tr["depth"],
                             dur_us=tr.get("duration_us"), dF_vs_trial=dF))
    return rows


def _f(x, fmt=".4f"):
    if x is None:
        return "-"
    try:
        if isinstance(x, float) and not np.isfinite(x):
            return "nan"
        return format(x, fmt)
    except Exception:
        return str(x)


def _arr(x):
    return np.array2string(np.asarray(x), precision=5, separator=", ", max_line_width=100)



def format_report(rep) -> str:
    L: List[str] = []
    bar = lambda t: L.append(f"\n{'=' * 4} {t} " + "=" * max(0, 74 - len(t)))
    L.append("=" * 80)
    L.append(f"CASE REPORT  {rep['tag']}")
    L.append("=" * 80)
 
    bar("INPUTS")
    for sec, d in rep["inputs"].items():
        L.append(f"[{sec}]")
        for k, v in d.items():
            L.append(f"  {k:24s} {v}")
 
    bar("DESIGN")
    d = rep["design"]
    for k, v in d.items():
        L.append(f"  {k:24s} {_arr(v) if isinstance(v, np.ndarray) else v}")
 
    bar("LOGICAL RESOURCES (pre-routing, success path, h/s/cx/rz basis)")
    cols = ("cx", "h", "rz", "total", "depth", "rz_nonclifford", "t_count_est")
    L.append(f"  {'':22s}" + "".join(f"{c:>15s}" for c in cols))
    for name, r in rep["logical_resources"].items():
        if "error" in r:
            L.append(f"  {name:22s} ERROR: {r['error']}")
        else:
            L.append(f"  {name:22s}" + "".join(f"{_f(r.get(c), '.0f'):>15s}" for c in cols))
 
    bar("REFERENCE (exact numerics, noiseless)")
    ref = rep["reference"]
    L.append(f"  E0 (ED)                  {ref['E0_ed']:.8f}")
    for k in ("trial_exact", "exact_filter", "ed_error"):
        L.append(f"  {k}:")
        for kk, v in ref[k].items():
            L.append(f"      {kk:22s} {_f(v, '.6g') if isinstance(v, float) else v}")
 
    for lab, e in rep["runs"].items():
        bar(f"RUN: {lab}")
        L.append(f"  target: {e['description']}")
        for kind in ("filtered", "trial_only"):
            r = e.get(kind)
            if r is None:
                continue
            L.append(f"  [{kind}]")
            if "error" in r:
                L.append(f"      ERROR: {r['error']}")
                continue
            tr = r["transpile"]
            L.append(f"      2q gates={tr['n_2q']}  1q gates={tr['n_1q']}  depth={tr['depth']}  "
                     f"2q-depth={tr['depth_2q']}"
                     + (f"  2q/logical-cx={tr['n_2q_over_logical_cx']:.2f}"
                        if "n_2q_over_logical_cx" in tr else ""))
            L.append(f"      ops={tr['ops']}")
            L.append(f"      active physical qubits={tr['active_qubits']}  "
                     f"logical->physical={tr['layout_logical_to_physical']}")
            if "duration_us" in tr:
                L.append(f"      est. duration={tr['duration_us']:.1f} us  "
                         f"median T1={_f(tr.get('median_T1_us'), '.1f')} us  "
                         f"T2={_f(tr.get('median_T2_us'), '.1f')} us  "
                         f"(no duration for: {tr['duration_missing_ops']})")
            if kind == "filtered":
                L.append(f"      P_succ={r['P_succ']:.4f} +/- {r['P_succ_err']:.4f} "
                         f"({r['n_success_shots']} shots)  wall={r['wall_s']:.1f}s")
                L.append("      outcomes: " + "  ".join(
                    f"{k}={v:.3f}" for k, v in r["abort_histogram"].items()))
            L.append(f"      E={_f(r['E'], '.6f')}  E-E0={_f(r['dE'], '+.6f')}  "
                     f"F_ED={_f(r['F_ed'])}  F_DMRG={_f(r['F_dmrg'])}  "
                     f"purity={_f(r['purity'])}")
 
    bar("SUMMARY")
    hdr = (f"  {'row':36s}{'P_succ':>8s}{'E':>11s}{'E-E0':>10s}{'F_ED':>8s}"
           f"{'F_DMRG':>8s}{'purity':>8s}{'dF*':>8s}{'2q':>6s}{'depth':>7s}{'t_us':>8s}")
    L.append(hdr)
    for r in rep["summary"]:
        L.append(f"  {r['label']:36s}{_f(r['P_succ'], '.3f'):>8s}{_f(r['E'], '.5f'):>11s}"
                 f"{_f(r['dE'], '+.4f'):>10s}{_f(r['F_ed']):>8s}{_f(r['F_dmrg']):>8s}"
                 f"{_f(r['purity'], '.3f'):>8s}{_f(r['dF_vs_trial'], '+.3f'):>8s}"
                 f"{_f(r['n_2q'], 'd'):>6s}{_f(r['depth'], 'd'):>7s}{_f(r['dur_us'], '.0f'):>8s}")
    L.append("  dF* = F_ED(filtered) - F_ED(trial only) on the same target; "
             "positive means the filter helps.")
    return "\n".join(L)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def _save(rep, out_dir, tag):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}.txt"), "w") as f:
        f.write(rep["text"])
    clean = {k: v for k, v in rep.items() if not k.startswith("_") and k != "text"}
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(clean, f, indent=2, default=_default)
    np.savez(os.path.join(out_dir, f"{tag}_states.npz"),
             **{k.replace("|", "__").replace(" ", "_"): v for k, v in rep["_states"].items()})
    print(f"[saved] {out_dir}/{tag}.txt / .json / _states.npz")
