"""
Analyze sweeps for the Heisenberg MPS pipeline.
=================================================
Enhanced version with six improvements over the original single-trial sweep:

  1. Multi-trial runs per point   -- `repeat_approx_circuit_trials()` is now
     wired into every sweep so every data point carries mean ± std error bars
     for fidelity and energy error.

  2. Accuracy–resource tradeoff   -- `sweep_accuracy_resource_tradeoff()` runs
     `sweep_n_layers()` at a configurable set of N values (default: 20, 50)
     so you can plot fidelity vs. n_layers and show 3 layers is a Pareto choice.

  3. Bond-dimension sweep         -- `sweep_bond_dim()` varies max χ at fixed N
     (default χ ∈ {20,40,80,100} at N=20) so truncation error is separated from
     circuit-compilation error in the fidelity/energy narrative.

  4. DMRG convergence diagnostics -- `_run_one_point()` now captures per-sweep
     energy convergence data; `sweep_j2_with_dmrg_diag()` stores it and flags
     DMRG-limited points (energy still moving at the last sweep) so the
     J2>0.5 runtime jump is clearly attributed to DMRG, not circuit compilation.

  5. Resource extrapolation       -- `extrapolate_resources()` uses the fitted
     N^α scaling to project T-count and CX count to N=500 and N=1000, giving
     fault-tolerant context.

  6. Exact vs. approximate comparison -- `compare_exact_vs_approx()` builds a
     side-by-side DataFrame of exact-circuit and approx-circuit resource counts
     (and their ratio), justifying why approximate compilation is useful.

     *** PATCHED: compare_exact_vs_approx() now accepts a `contexts` dict of
     already-built GroundStateContext objects (e.g. from sweep_n) and reuses
     them instead of re-running DMRG + exact-circuit synthesis from scratch
     for every N. sweep_n() optionally returns those contexts via
     `return_contexts=True`. analyze() wires this together automatically. ***

Entry points
------------
    from analyze_sweeps import analyze
    results = analyze()          # returns AnalysisResults namedtuple

Or run individual sweeps:
    df = sweep_n(n_trials=5)
    df = sweep_j2_with_dmrg_diag(n_trials=5)
    df = sweep_bond_dim()
    df = sweep_accuracy_resource_tradeoff(n_values=[20, 50])
    df = compare_exact_vs_approx(n_values=[10, 20, 50])
    print(extrapolate_resources(df_n_sweep, target_ns=[500, 1000]))
"""

from __future__ import annotations

import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import linregress

from dmrg_circuit_pipeline import (
    J1,
    J2,
    DMRG_BOND_DIMS,
    RZ_SYNTHESIS_EPS,
    CIRCUIT_MPS_MAX_BOND,
    COMPRESS_EVERY_N_EDGES,
    N_LAYERS,
    DMRG_TOL,
    DMRG_CUTOFF,
    DMRG_MAX_SWEEPS,
    make_chain,
    build_ground_state_context,
    build_heisenberg_mpo,
    repeat_approx_circuit_trials,
    sweep_n_layers,
    _approx_trial,
    mps_diagnostics,
    _heisenberg_term_mpo,
    circuit_resource_report,
    mps_to_exact_circuit,
    mps_to_approx_circuit,
)

try:
    import quimb.tensor as qtn
    import numpy as np
except ImportError:
    raise ImportError("pip install quimb")


# =============================================================================
# TOP-LEVEL CONFIG
# =============================================================================
N_SWEEP_VALUES       = list(range(10, 101, 10))          # 10, 20, …, 100
N_SWEEP_FIXED_N      = 20
J2_SWEEP_VALUES      = [round(0.1 * k, 1) for k in range(11)]  # 0.0 … 1.0

# ---- Enhancement 1: trials per point ------------------------------------
DEFAULT_TRIALS_PER_POINT = 10   # increase to 10+ for publication

# ---- Enhancement 2: accuracy–resource tradeoff --------------------------
TRADEOFF_N_VALUES = [20, 50]
TRADEOFF_LAYERS   = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# ---- Enhancement 3: bond-dimension sweep --------------------------------
BOND_DIM_SWEEP_N       = 20
BOND_DIM_SWEEP_CHI_MAX = [20, 40, 80, 100]      # vary max χ; schedule auto-built

# ---- Enhancement 5: resource extrapolation targets ----------------------
EXTRAPOLATION_TARGETS = [200, 500, 1000]


# =============================================================================
# RESULT CONTAINER
# =============================================================================
@dataclass
class AnalysisResults:
    """All DataFrames returned by analyze()."""
    df_n_sweep:           pd.DataFrame   # sweep 1: N sweep, multi-trial
    df_j2_sweep:          pd.DataFrame   # sweep 2: J2 sweep, multi-trial + DMRG diag
    df_bond_dim:          pd.DataFrame   # sweep 3: bond-dim sweep
    df_tradeoff:          pd.DataFrame   # sweep 4: n_layers tradeoff (accuracy vs resources)
    df_exact_vs_approx:   pd.DataFrame   # sweep 6: exact vs approx comparison
    df_extrapolation:     pd.DataFrame   # sweep 5: resource extrapolation


# =============================================================================
# INTERNAL: ONE DATA POINT WITH MULTI-TRIAL + DMRG CONVERGENCE TRACKING
# =============================================================================
def _run_one_point(
    n_sites: int,
    j1: float,
    j2: float,
    bond_dims: list[int],
    n_layers: int,
    rz_eps: float,
    compress_every_n_edges: int,
    circuit_mps_max_bond: int,
    n_trials: int = 1,
    track_dmrg_convergence: bool = False,
    verbose: bool = False,
    return_ctx: bool = False,
) -> dict:
    """
    Build the ground-state context once, then run *n_trials* independent
    approximate-circuit fits and summarise them with mean ± std.

    Enhancement 1: n_trials > 1 → per-point error bars.
    Enhancement 4: track_dmrg_convergence=True → per-sweep energy log +
                   convergence flag stored in the row.

    PATCH: return_ctx=True → the built GroundStateContext is stashed in
    row["_ctx"] so callers (e.g. sweep_n) can cache and reuse it later
    (e.g. in compare_exact_vs_approx) instead of rebuilding DMRG from scratch.
    """
    chain = make_chain(n_sites)

    row: dict = dict(
        n_sites      = n_sites,
        j1           = j1,
        j2           = j2,
        j2_over_j1   = (j2 / j1) if j1 else float("nan"),
        n_layers     = n_layers,
        n_trials     = n_trials,
        ok           = False,
        error        = None,
    )

    # ---- DMRG with optional per-sweep energy logging (Enhancement 4) ----
    t0 = time.perf_counter()
    try:
        if track_dmrg_convergence:
            ctx, dmrg_log = _build_context_with_sweep_log(
                chain, j1, j2, bond_dims, rz_eps,
                circuit_mps_max_bond, compress_every_n_edges,
            )
            row["dmrg_sweep_energies"]    = dmrg_log["sweep_energies"]
            row["dmrg_n_sweeps"]          = dmrg_log["n_sweeps"]
            row["dmrg_converged"]         = dmrg_log["converged"]
            row["dmrg_delta_e_last"]      = dmrg_log["delta_e_last"]
            row["dmrg_limited"]           = not dmrg_log["converged"]
        else:
            ctx = build_ground_state_context(
                chain,
                j1=j1,
                j2=j2,
                bond_dims=bond_dims,
                rz_eps=rz_eps,
                circuit_mps_max_bond=circuit_mps_max_bond,
                compress_every_n_edges=compress_every_n_edges,
            )
    except Exception as e:
        row["context_time_s"] = time.perf_counter() - t0
        row["error"] = f"build_ground_state_context failed: {e!r}"
        if verbose:
            traceback.print_exc()
        return row
    row["context_time_s"] = time.perf_counter() - t0

    row["dmrg_energy"]        = ctx.E_dmrg
    row["max_chi"]            = ctx.diag.get("max_chi")
    row["mean_chi"]           = ctx.diag.get("mean_chi")
    row["half_chain_entropy"] = ctx.diag.get("half_chain_entropy")

    if return_ctx:
        row["_ctx"] = ctx

    # ---- Enhancement 1: multiple approximate-circuit trials ---------------
    t1 = time.perf_counter()
    try:
        df_trials = repeat_approx_circuit_trials(
            ctx,
            n_layers=n_layers,
            n_trials=n_trials,
            rz_eps=rz_eps,
            circuit_mps_max_bond=circuit_mps_max_bond,
            verbose=verbose,
        )
    except Exception as e:
        row["approx_trial_time_s"] = time.perf_counter() - t1
        row["error"] = f"repeat_approx_circuit_trials failed: {e!r}"
        if verbose:
            traceback.print_exc()
        return row
    row["approx_trial_time_s"] = time.perf_counter() - t1
    row["total_time_s"] = row["context_time_s"] + row["approx_trial_time_s"]

    # Aggregate: mean ± std for every numeric metric across trials
    for col in df_trials.columns:
        if col in ("trial", "n_layers", "n_retries"):
            continue
        vals = pd.to_numeric(df_trials[col], errors="coerce").dropna()
        if len(vals) == 0:
            continue
        row[f"{col}_mean"] = float(vals.mean())
        row[f"{col}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        row[f"{col}_min"]  = float(vals.min())
        row[f"{col}_max"]  = float(vals.max())
    row["n_retries_total"] = int(df_trials["n_retries"].sum()) if "n_retries" in df_trials.columns else 0

    row["ok"] = True
    return row


# =============================================================================
# ENHANCEMENT 4: DMRG CONVERGENCE TRACKING
# =============================================================================
def _build_context_with_sweep_log(
    chain,
    j1: float,
    j2: float,
    bond_dims: list[int],
    rz_eps: float,
    circuit_mps_max_bond: int,
    compress_every_n_edges: int,
) -> tuple:
    """
    Like build_ground_state_context() but intercepts DMRG to log per-sweep
    energies. Returns (ctx, dmrg_log) where dmrg_log contains:
        sweep_energies  : list[float]  energy after each sweep
        n_sweeps        : int
        converged       : bool
        delta_e_last    : float        |E[-1] - E[-2]|, 0 if only 1 sweep
    """
    import quimb.tensor as qtn
    import numpy as np
    from dmrg_circuit_pipeline import (
        build_heisenberg_mpo, mps_to_exact_circuit, circuit_resource_report,
        circuit_to_mps, verify_energy_mps, mps_diagnostics, GroundStateContext,
        EXACT_DIAG_MAX_N, exact_ground_state, build_hamiltonian, verify_energy,
    )

    n = chain.n_sites
    H_mpo = build_heisenberg_mpo(chain, j1, j2, compress_every_n_edges)

    # Optional ED for small chains
    use_ed = n <= EXACT_DIAG_MAX_N
    if use_ed:
        H, basis = build_hamiltonian(chain, j1, j2)
        e_exact, psi_exact, basis, _ = exact_ground_state(chain, j1, j2)
    else:
        H, basis, e_exact, psi_exact = None, None, None, None

    # Manual per-sweep DMRG loop (mirrors _harness_one_size logic)
    psi0 = qtn.MPS_neel_state(n)
    dmrg = qtn.DMRG2(H_mpo, bond_dims=bond_dims, cutoffs=DMRG_CUTOFF, p0=psi0)

    sweep_energies = []
    converged = False
    sweep_num = 0
    while sweep_num < DMRG_MAX_SWEEPS and not converged:
        sweep_num += 1
        converged = dmrg.solve(tol=DMRG_TOL, max_sweeps=1, verbosity=0)
        e = float(np.real(dmrg.energies[-1])) if dmrg.energies else float("nan")
        sweep_energies.append(e)

    E_dmrg  = sweep_energies[-1] if sweep_energies else float("nan")
    psi_mps = dmrg.state
    diag    = mps_diagnostics(psi_mps)

    delta_e_last = (
        abs(sweep_energies[-1] - sweep_energies[-2])
        if len(sweep_energies) >= 2 else 0.0
    )

    dmrg_log = dict(
        sweep_energies = sweep_energies,
        n_sweeps       = sweep_num,
        converged      = converged,
        delta_e_last   = delta_e_last,
    )

    # Exact circuit + MPS verification
    exact_qc        = mps_to_exact_circuit(psi_mps, n)
    exact_resources = circuit_resource_report(exact_qc, label="exact", rz_eps=rz_eps)
    exact_circ_mps  = circuit_to_mps(exact_qc, max_bond=circuit_mps_max_bond)
    exact_vfy_mps   = verify_energy_mps(exact_circ_mps, H_mpo, psi_mps, E_dmrg)
    exact_vfy       = (
        verify_energy(exact_qc, H, basis, E_dmrg, psi_exact) if use_ed else None
    )

    ctx = GroundStateContext(
        chain=chain, j1=j1, j2=j2, bond_dims=bond_dims, n=n,
        H=H, H_mpo=H_mpo, basis=basis,
        e_exact=e_exact, psi_exact=psi_exact,
        E_dmrg=E_dmrg, psi_mps=psi_mps, diag=diag,
        exact_qc=exact_qc,
        exact_resources=exact_resources,
        exact_verification=exact_vfy,
        exact_circ_mps=exact_circ_mps,
        exact_verification_mps=exact_vfy_mps,
    )
    return ctx, dmrg_log


# =============================================================================
# SWEEP 1: N SWEEP  (Enhancement 1: multi-trial)
# =============================================================================
def sweep_n(
    n_values: list[int] = N_SWEEP_VALUES,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    n_layers: int = N_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    n_trials: int = DEFAULT_TRIALS_PER_POINT,
    verbose: bool = False,
    return_contexts: bool = False,
):
    """
    Sweep chain size N at fixed J1, J2.
    Each point runs n_trials independent approx-circuit fits (Enhancement 1).
    Returns a DataFrame with _mean and _std columns for all approx metrics.

    PATCH: if return_contexts=True, returns (df, contexts) where contexts is
    a dict {n_sites: GroundStateContext} for every successfully-built point.
    This lets compare_exact_vs_approx() reuse the already-solved DMRG state
    instead of rebuilding it from scratch for the same (n_sites, j1, j2).
    """
    print(f"\n[sweep_n]  N in {n_values}  J1={j1}  J2={j2}  "
          f"n_layers={n_layers}  n_trials={n_trials}")
    rows = []
    contexts: dict[int, object] = {}
    for n_sites in n_values:
        print(f"\n  -- N={n_sites} --")
        row = _run_one_point(
            n_sites, j1, j2, bond_dims, n_layers, rz_eps,
            compress_every_n_edges, circuit_mps_max_bond,
            n_trials=n_trials, verbose=verbose,
            return_ctx=return_contexts,
        )
        fid_mean = row.get("approx_fidelity_mps_mean", float("nan"))
        fid_std  = row.get("approx_fidelity_mps_std",  float("nan"))
        cx_mean  = row.get("approx_cx_count_mean",      float("nan"))
        print(f"     ok={row['ok']}  total_time={row.get('total_time_s', float('nan')):.2f}s  "
              f"cx_mean={cx_mean:.1f}  fidelity={fid_mean:.4f}±{fid_std:.4f}")
        if return_contexts and "_ctx" in row:
            contexts[n_sites] = row.pop("_ctx")
        rows.append(row)
    df = pd.DataFrame(rows)
    if return_contexts:
        return df, contexts
    return df


# =============================================================================
# SWEEP 2: J2 SWEEP  (Enhancements 1 + 4: multi-trial + DMRG diagnostics)
# =============================================================================
def sweep_j2_with_dmrg_diag(
    j2_values: list[float] = J2_SWEEP_VALUES,
    n_sites: int = N_SWEEP_FIXED_N,
    j1: float = J1,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    n_layers: int = N_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    n_trials: int = DEFAULT_TRIALS_PER_POINT,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Sweep J2 at fixed N and J1.

    Enhancement 1: n_trials approx-circuit fits per point → error bars.
    Enhancement 4: captures per-sweep energy log and flags DMRG-limited points
                   (converged=False) so the J2>0.5 runtime spike is attributed
                   correctly to DMRG convergence failure, not circuit compilation.
    """
    print(f"\n[sweep_j2_with_dmrg_diag]  J2 in {j2_values}  N={n_sites}  "
          f"J1={j1}  n_trials={n_trials}")
    rows = []
    for j2_val in j2_values:
        print(f"\n  -- J2={j2_val} --")
        row = _run_one_point(
            n_sites, j1, j2_val, bond_dims, n_layers, rz_eps,
            compress_every_n_edges, circuit_mps_max_bond,
            n_trials=n_trials,
            track_dmrg_convergence=True,
            verbose=verbose,
        )

        # Print convergence flag prominently
        if row.get("dmrg_limited"):
            print(f"     *** DMRG-LIMITED ***  converged=False  "
                  f"n_sweeps={row.get('dmrg_n_sweeps')}  "
                  f"delta_E_last={row.get('dmrg_delta_e_last'):.2e}")
        else:
            print(f"     DMRG converged in {row.get('dmrg_n_sweeps')} sweeps")

        fid_mean = row.get("approx_fidelity_mps_mean", float("nan"))
        fid_std  = row.get("approx_fidelity_mps_std",  float("nan"))
        print(f"     ok={row['ok']}  total_time={row.get('total_time_s', float('nan')):.2f}s  "
              f"fidelity={fid_mean:.4f}±{fid_std:.4f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    _print_dmrg_convergence_table(df)
    return df


def _print_dmrg_convergence_table(df: pd.DataFrame) -> None:
    """Enhancement 4: print a convergence summary for the J2 sweep."""
    print(f"\n{'='*70}")
    print("  DMRG convergence summary (J2 sweep)")
    print(f"{'='*70}")
    cols = ["j2", "dmrg_n_sweeps", "dmrg_converged",
            "dmrg_delta_e_last", "dmrg_limited", "total_time_s"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    if "dmrg_limited" in df.columns:
        n_lim = df["dmrg_limited"].sum()
        print(f"\n  -> {int(n_lim)} of {len(df)} points are DMRG-limited "
              f"(converged=False); runtime at those points reflects DMRG cost, "
              f"not circuit compilation.")


# =============================================================================
# SWEEP 3: BOND-DIMENSION SWEEP  (Enhancement 3)
# =============================================================================
def sweep_bond_dim(
    chi_max_values: list[int] = BOND_DIM_SWEEP_CHI_MAX,
    n_sites: int = BOND_DIM_SWEEP_N,
    j1: float = J1,
    j2: float = J2,
    n_layers: int = N_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    n_trials: int = DEFAULT_TRIALS_PER_POINT,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Enhancement 3: Sweep maximum DMRG bond dimension χ at fixed N and J2.

    For each χ_max we build a bond_dims schedule [10, 20, …, chi_max] and
    run a fresh DMRG solve. This lets you plot:
      - DMRG energy vs. χ  (truncation error)
      - fidelity vs. χ     (how much truncation error bleeds into circuit fidelity)
    and thereby separate the two error sources that were conflated before.
    """
    print(f"\n[sweep_bond_dim]  chi_max in {chi_max_values}  "
          f"N={n_sites}  J1={j1}  J2={j2}  n_trials={n_trials}")
    rows = []
    for chi_max in chi_max_values:
        # Build a geometric schedule up to chi_max
        sched  = [10]
        bd     = 20
        while bd < chi_max:
            sched.append(bd)
            bd *= 2
        sched.append(chi_max)
        sched = sorted(set(sched))

        print(f"\n  -- chi_max={chi_max}  schedule={sched} --")
        row = _run_one_point(
            n_sites, j1, j2, sched, n_layers, rz_eps,
            compress_every_n_edges, circuit_mps_max_bond,
            n_trials=n_trials, verbose=verbose,
        )
        row["chi_max_target"] = chi_max
        row["bond_dim_sched"] = str(sched)

        fid_mean = row.get("approx_fidelity_mps_mean", float("nan"))
        fid_std  = row.get("approx_fidelity_mps_std",  float("nan"))
        print(f"     max_chi_actual={row.get('max_chi')}  "
              f"dmrg_energy={row.get('dmrg_energy'):.8f}  "
              f"fidelity={fid_mean:.6f}±{fid_std:.6f}")
        rows.append(row)

    df = pd.DataFrame(rows)
    _print_bond_dim_summary(df)
    return df


def _print_bond_dim_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  Bond-dimension sweep summary")
    print("  (separates truncation error from circuit-compilation error)")
    print(f"{'='*70}")
    cols = ["chi_max_target", "max_chi", "dmrg_energy",
            "approx_fidelity_mps_mean", "approx_fidelity_mps_std",
            "approx_energy_err_pct_mps_mean"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))


# =============================================================================
# SWEEP 4: ACCURACY–RESOURCE TRADEOFF  (Enhancement 2)
# =============================================================================
def sweep_accuracy_resource_tradeoff(
    n_values: list[int] = TRADEOFF_N_VALUES,
    layers: list[int] = TRADEOFF_LAYERS,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    n_trials: int = DEFAULT_TRIALS_PER_POINT,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Enhancement 2: run sweep_n_layers() at each N value to expose the
    fidelity vs. n_layers Pareto frontier.  DMRG is solved ONCE per N;
    the approx circuit is re-fit n_trials times per (N, n_layers) pair.

    Produces a DataFrame with columns:
        n_sites, n_layers, approx_fidelity_mps_{mean,std},
        approx_cx_count_{mean,std}, approx_t_count_estimate_{mean,std}, …
    """
    print(f"\n[sweep_accuracy_resource_tradeoff]  "
          f"N in {n_values}  layers={layers}  n_trials={n_trials}")
    all_dfs = []
    for n_sites in n_values:
        print(f"\n  -- N={n_sites} --")
        chain = make_chain(n_sites)
        df_summary = sweep_n_layers(
            chain,
            j1=j1,
            j2=j2,
            bond_dims=bond_dims,
            rz_eps=rz_eps,
            layers=layers,
            n_trials=n_trials,
            circuit_mps_max_bond=circuit_mps_max_bond,
            compress_every_n_edges=compress_every_n_edges,
            verbose_trials=verbose,
            return_all_trials=False,
        )
        df_summary["n_sites"] = n_sites
        all_dfs.append(df_summary)

    df = pd.concat(all_dfs, ignore_index=True)

    print(f"\n{'='*70}")
    print("  Accuracy–resource tradeoff summary")
    print("  (fidelity vs. n_layers; choose the Pareto-optimal n_layers)")
    print(f"{'='*70}")
    cols = ["n_sites", "n_layers",
            "approx_fidelity_mps_mean", "approx_fidelity_mps_std",
            "approx_cx_count_mean", "approx_t_count_estimate_mean",
            "approx_depth_mean"]
    available = [c for c in cols if c in df.columns]
    print(df[available].to_string(index=False))
    return df


# =============================================================================
# SWEEP 5: RESOURCE EXTRAPOLATION  (Enhancement 5)
# =============================================================================
def extrapolate_resources(
    df_n_sweep: pd.DataFrame,
    cx_col: str = "approx_cx_count_mean",
    t_col:  str = "approx_t_count_estimate_mean",
    target_ns: list[int] = EXTRAPOLATION_TARGETS,
) -> pd.DataFrame:
    """
    Enhancement 5: fit a power law (log-log linear regression) to the CX
    and T-count data from df_n_sweep, then project to target_ns.

    Prints the fitted exponents and returns a DataFrame with columns:
        n_sites, cx_projected, t_projected, cx_fit_alpha, t_fit_alpha
    """
    # Use mean columns if they exist, else fall back to non-mean
    def _resolve_col(df, preferred, fallback):
        return preferred if preferred in df.columns else fallback

    cx_c = _resolve_col(df_n_sweep, cx_col, "approx_cx_count")
    t_c  = _resolve_col(df_n_sweep, t_col,  "approx_t_count_estimate")

    df_fit = df_n_sweep[["n_sites", cx_c, t_c]].dropna()
    if len(df_fit) < 3:
        warnings.warn("extrapolate_resources: fewer than 3 clean points; "
                      "fit may be unreliable.")

    log_n  = np.log(df_fit["n_sites"].astype(float).values)
    log_cx = np.log(df_fit[cx_c].astype(float).values)
    log_t  = np.log(df_fit[t_c].astype(float).values)

    cx_slope, cx_intercept, cx_r, *_ = linregress(log_n, log_cx)
    t_slope,  t_intercept,  t_r,  *_ = linregress(log_n, log_t)

    print(f"\n{'='*70}")
    print("  Resource extrapolation (power-law fit: count ~ A * N^alpha)")
    print(f"{'='*70}")
    print(f"  CX count:  alpha={cx_slope:.4f}  R²={cx_r**2:.4f}")
    print(f"  T  count:  alpha={t_slope:.4f}  R²={t_r**2:.4f}")

    proj_rows = []
    for n_tgt in target_ns:
        cx_proj = np.exp(cx_intercept + cx_slope * np.log(n_tgt))
        t_proj  = np.exp(t_intercept  + t_slope  * np.log(n_tgt))
        proj_rows.append(dict(
            n_sites       = n_tgt,
            cx_projected  = cx_proj,
            t_projected   = t_proj,
            cx_fit_alpha  = cx_slope,
            t_fit_alpha   = t_slope,
        ))
        print(f"  N={n_tgt:>5d}:  CX ≈ {cx_proj:.1f}   T ≈ {t_proj:.1f}")

    return pd.DataFrame(proj_rows)


# =============================================================================
# SWEEP 6: EXACT VS. APPROXIMATE COMPARISON  (Enhancement 6)
# =============================================================================
def compare_exact_vs_approx(
    n_values: list[int] = N_SWEEP_VALUES,
    j1: float = J1,
    j2: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    n_layers: int = N_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    n_trials: int = DEFAULT_TRIALS_PER_POINT,
    verbose: bool = False,
    contexts: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Enhancement 6: for each N, compile both the exact MPS circuit and n_trials
    approximate circuits, then record a side-by-side resource comparison and
    the exact/approx ratio.

    The exact circuit is deterministic (no stochasticity); the approximate
    circuit metrics are averaged over n_trials random initializations.

    PATCH: `contexts`, if provided, is a dict {n_sites: GroundStateContext}
    of already-built contexts (e.g. returned by sweep_n(..., return_contexts=True)
    at the SAME j1/j2/bond_dims/rz_eps). For any n_sites found in this dict,
    DMRG and exact-circuit synthesis are skipped entirely and the cached
    context is reused -- only the (stochastic) approximate-circuit trials are
    re-run, since those are independent random fits each time. This avoids
    paying for DMRG + exact-circuit compilation twice for the same point,
    which previously made this sweep cost roughly as much as sweep_n() again.

    Key output columns (per row = one N value):
        exact_cx_count, exact_depth, exact_non_clifford_count, exact_t_count_estimate
        approx_cx_count_mean, approx_depth_mean, …
        cx_ratio  (= approx_cx_mean / exact_cx)
        depth_ratio
        t_ratio
        approx_fidelity_mps_mean, approx_fidelity_mps_std
        context_reused  (True if context was reused instead of rebuilt)
    """
    contexts = contexts or {}
    print(f"\n[compare_exact_vs_approx]  N in {n_values}  n_layers={n_layers}  "
          f"n_trials={n_trials}  (reusing {len(contexts)} cached context(s))")
    rows = []
    for n_sites in n_values:
        print(f"\n  -- N={n_sites} --")

        reused = n_sites in contexts
        t0 = time.perf_counter()
        if reused:
            # Reuse the already-solved DMRG + exact-circuit context. This
            # only works correctly if the cached context was built at the
            # same j1/j2/bond_dims/rz_eps/compress_every_n_edges -- callers
            # (analyze()) are responsible for that invariant.
            ctx = contexts[n_sites]
            print(f"     (reusing cached context, skipping DMRG + exact-circuit rebuild)")
        else:
            chain = make_chain(n_sites)
            try:
                ctx = build_ground_state_context(
                    chain, j1=j1, j2=j2, bond_dims=bond_dims, rz_eps=rz_eps,
                    circuit_mps_max_bond=circuit_mps_max_bond,
                    compress_every_n_edges=compress_every_n_edges,
                )
            except Exception as e:
                print(f"     FAILED (context): {e!r}")
                rows.append(dict(n_sites=n_sites, ok=False, error=str(e)))
                continue
        context_time = time.perf_counter() - t0

        # Exact circuit resources (deterministic)
        exact_res = ctx.exact_resources.as_row()
        exact_fid = ctx.exact_verification_mps.get("fidelity", float("nan"))
        exact_e_err = ctx.exact_verification_mps.get("energy_err_pct", float("nan"))

        # Approx circuit: n_trials stochastic fits (always re-run -- these
        # are independent random fits, not deterministic, so caching them
        # would change the statistics rather than just save time)
        t1 = time.perf_counter()
        try:
            df_trials = repeat_approx_circuit_trials(
                ctx, n_layers=n_layers, n_trials=n_trials, rz_eps=rz_eps,
                circuit_mps_max_bond=circuit_mps_max_bond, verbose=verbose,
            )
        except Exception as e:
            print(f"     FAILED (approx trials): {e!r}")
            rows.append(dict(n_sites=n_sites, ok=False, error=str(e),
                             context_time_s=context_time))
            continue
        approx_time = time.perf_counter() - t1

        # Aggregate approx stats
        approx_stats = {}
        for col in df_trials.columns:
            if col in ("trial", "n_layers", "n_retries"):
                continue
            vals = pd.to_numeric(df_trials[col], errors="coerce").dropna()
            if len(vals) == 0:
                continue
            approx_stats[f"approx_{col}_mean"] = float(vals.mean())
            approx_stats[f"approx_{col}_std"]  = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

        # Ratios: approx / exact  (a ratio < 1 means approx is cheaper)
        def _ratio(approx_key, exact_key):
            a = approx_stats.get(approx_key, float("nan"))
            e = exact_res.get(exact_key, float("nan"))
            return a / e if (e and not np.isnan(a) and not np.isnan(e)) else float("nan")

        row = dict(
            n_sites            = n_sites,
            j1                 = j1,
            j2                 = j2,
            n_layers           = n_layers,
            n_trials           = n_trials,
            ok                 = True,
            context_reused     = reused,
            dmrg_energy        = ctx.E_dmrg,
            max_chi            = ctx.diag.get("max_chi"),
            context_time_s     = context_time,
            approx_time_s      = approx_time,
            # Exact metrics
            exact_cx_count            = exact_res.get("cx_count"),
            exact_depth               = exact_res.get("depth"),
            exact_non_clifford_count  = exact_res.get("non_clifford_count"),
            exact_t_count_estimate    = exact_res.get("t_count_estimate"),
            exact_fidelity_mps        = exact_fid,
            exact_energy_err_pct_mps  = exact_e_err,
        )
        row.update(approx_stats)

        # Convenience ratios
        row["cx_ratio"]    = _ratio("approx_cx_count_mean",           "cx_count")
        row["depth_ratio"] = _ratio("approx_depth_mean",              "depth")
        row["t_ratio"]     = _ratio("approx_t_count_estimate_mean",   "t_count_estimate")
        row["fidelity_cost"] = row.get("approx_fidelity_mps_mean", float("nan"))

        cx_ratio = row["cx_ratio"]
        fid_mean = row.get("approx_fidelity_mps_mean", float("nan"))
        fid_std  = row.get("approx_fidelity_mps_std",  float("nan"))
        print(f"     exact_cx={exact_res.get('cx_count')}  "
              f"approx_cx_mean={approx_stats.get('approx_cx_count_mean'):.1f}  "
              f"cx_ratio={cx_ratio:.3f}  "
              f"approx_fidelity={fid_mean:.4f}±{fid_std:.4f}  "
              f"context_reused={reused}")
        rows.append(row)

    df = pd.DataFrame(rows)
    _print_exact_vs_approx_summary(df)
    return df


def _print_exact_vs_approx_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print("  Exact vs. Approximate circuit comparison")
    print("  (ratio < 1 = approx circuit is cheaper than exact)")
    print(f"{'='*70}")
    cols = [
        "n_sites", "context_reused",
        "exact_cx_count", "approx_cx_count_mean",  "cx_ratio",
        "exact_depth",    "approx_depth_mean",      "depth_ratio",
        "exact_t_count_estimate", "approx_t_count_estimate_mean", "t_ratio",
        "exact_fidelity_mps", "approx_fidelity_mps_mean", "approx_fidelity_mps_std",
    ]
    available = [c for c in cols if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df[available].to_string(index=False))


# =============================================================================
# PRETTY PRINT HELPER
# =============================================================================
_DISPLAY_COLS = [
    "approx_cx_count_mean",
    "approx_cx_count_std",
    "approx_depth_mean",
    "approx_non_clifford_count_mean",
    "approx_non_clifford_depth_mean",
    "approx_t_count_estimate_mean",
    "approx_t_count_estimate_std",
    "context_time_s",
    "approx_trial_time_s",
    "total_time_s",
    "approx_fidelity_mps_mean",
    "approx_fidelity_mps_std",
    "approx_energy_err_pct_mps_mean",
    "approx_energy_err_pct_mps_std",
]


def _print_summary(title: str, df: pd.DataFrame, x_col: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title} summary")
    print(f"{'='*70}")
    if df.empty:
        print("  (no rows)")
        return
    cols = [x_col] + [c for c in _DISPLAY_COLS if c in df.columns]
    with pd.option_context("display.width", 200, "display.max_columns", 40):
        print(df[cols].to_string(index=False))


# =============================================================================
# COMBINED ENTRY POINT
# =============================================================================
def analyze(
    n_values:             list[int]   = N_SWEEP_VALUES,
    n_fixed_for_j2:       int         = N_SWEEP_FIXED_N,
    j2_values:            list[float] = J2_SWEEP_VALUES,
    j1:                   float       = J1,
    j2_for_n_sweep:       float       = J2,
    bond_dims:            list[int]   = DMRG_BOND_DIMS,
    n_layers:             int         = N_LAYERS,
    rz_eps:               float       = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int       = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int         = CIRCUIT_MPS_MAX_BOND,
    n_trials:             int         = DEFAULT_TRIALS_PER_POINT,
    chi_max_values:       list[int]   = BOND_DIM_SWEEP_CHI_MAX,
    chi_sweep_n:          int         = BOND_DIM_SWEEP_N,
    tradeoff_n_values:    list[int]   = TRADEOFF_N_VALUES,
    tradeoff_layers:      list[int]   = TRADEOFF_LAYERS,
    extrapolation_targets: list[int]  = EXTRAPOLATION_TARGETS,
    verbose:              bool        = False,
) -> AnalysisResults:
    """
    Full analysis pipeline with all six enhancements.

    PATCH: sweep_n() and compare_exact_vs_approx() use the SAME n_values,
    j1, j2_for_n_sweep, bond_dims, rz_eps, and compress_every_n_edges, so
    the GroundStateContext built once in sweep_n() is cached and handed
    to compare_exact_vs_approx(), which reuses it instead of re-running
    DMRG + exact-circuit synthesis for every N a second time. Only the
    stochastic approximate-circuit trials are re-fit in sweep 6, since
    those are independent random fits each call.

    Returns
    -------
    AnalysisResults with fields:
        df_n_sweep         -- N sweep, multi-trial (Enhancements 1)
        df_j2_sweep        -- J2 sweep, multi-trial + DMRG diagnostics (1 + 4)
        df_bond_dim        -- bond-dimension sweep (Enhancement 3)
        df_tradeoff        -- fidelity vs. n_layers Pareto frontier (Enhancement 2)
        df_exact_vs_approx -- exact vs. approx side-by-side (Enhancement 6)
        df_extrapolation   -- resource projections to large N (Enhancement 5)
    """
    # ---- 1+4: N sweep with multi-trial (now also caches contexts) -------
    df_n_sweep, n_sweep_contexts = sweep_n(
        n_values=n_values, j1=j1, j2=j2_for_n_sweep,
        bond_dims=bond_dims, n_layers=n_layers, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond,
        n_trials=n_trials, verbose=verbose,
        return_contexts=True,
    )

    # ---- 1+4: J2 sweep with multi-trial + DMRG diagnostics ---------------
    df_j2_sweep = sweep_j2_with_dmrg_diag(
        j2_values=j2_values, n_sites=n_fixed_for_j2, j1=j1,
        bond_dims=bond_dims, n_layers=n_layers, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond,
        n_trials=n_trials, verbose=verbose,
    )

    # ---- 3: bond-dimension sweep -----------------------------------------
    df_bond_dim = sweep_bond_dim(
        chi_max_values=chi_max_values, n_sites=chi_sweep_n,
        j1=j1, j2=j2_for_n_sweep,
        n_layers=n_layers, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond,
        n_trials=n_trials, verbose=verbose,
    )

    # ---- 2: accuracy–resource tradeoff ------------------------------------
    df_tradeoff = sweep_accuracy_resource_tradeoff(
        n_values=tradeoff_n_values, layers=tradeoff_layers,
        j1=j1, j2=j2_for_n_sweep,
        bond_dims=bond_dims, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond,
        n_trials=n_trials, verbose=verbose,
    )

    # ---- 6: exact vs. approximate comparison (reuses sweep_n's contexts) -
    df_exact_vs_approx = compare_exact_vs_approx(
        n_values=n_values, j1=j1, j2=j2_for_n_sweep,
        bond_dims=bond_dims, n_layers=n_layers, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond,
        n_trials=n_trials, verbose=verbose,
        contexts=n_sweep_contexts,
    )

    # ---- 5: resource extrapolation (uses N-sweep data) -------------------
    ok_n = df_n_sweep[df_n_sweep["ok"].astype(bool)]
    df_extrapolation = extrapolate_resources(
        ok_n, target_ns=extrapolation_targets
    )

    # Print final summaries
    _print_summary("N sweep",  df_n_sweep,  x_col="n_sites")
    _print_summary("J2 sweep", df_j2_sweep, x_col="j2")

    return AnalysisResults(
        df_n_sweep          = df_n_sweep,
        df_j2_sweep         = df_j2_sweep,
        df_bond_dim         = df_bond_dim,
        df_tradeoff         = df_tradeoff,
        df_exact_vs_approx  = df_exact_vs_approx,
        df_extrapolation    = df_extrapolation,
    )


if __name__ == "__main__":
    results = analyze()
