"""
Analyze sweeps for the Heisenberg MPS pipeline.
=================================================
Adds a single `analyze()` entry point that:

  1. Sweeps chain size N = 10, 20, ..., 100 (J1, J2 held at module defaults).
  2. Sweeps J2 = 0.0, 0.1, ..., 1.0 at fixed N = 20 (J1 held at module default).

For every (N, J1, J2) point it:
  - builds the ground-state context (MPO build + DMRG) ONCE, timing that
    stage explicitly;
  - runs one approximate-circuit trial (`_approx_trial`) at the configured
    `n_layers`, timing that stage explicitly;
  - records gate counts and the full per-gate-type breakdown (from
    `ResourceReport.gate_counts`, surfaced as `n_<gate>` columns by
    `ResourceReport.as_row()`), non-Clifford depth/count, T-count estimate;
  - records the MPS-based fidelity/energy overlap between the prepared
    circuit state and the converged DMRG MPS (`approx_fidelity_mps`,
    `approx_energy_err_pct_mps`, etc., as already produced by `_approx_trial`
    via `verify_energy_mps`).

This file imports from the merged pipeline module rather than redefining
anything, so DMRG/MPO/circuit logic stays single-sourced.

Usage
-----
    from analyze_sweeps import analyze
    df_n, df_j2 = analyze()

Or, to reuse a context you already built for a one-off check (as in your
snippet), call `_approx_trial` directly -- `analyze()` is for the full sweep.
"""

from __future__ import annotations

import time
import traceback
from typing import Optional

import pandas as pd

from dmrg_circuit_pipeline import (
    J1,
    J2,
    DMRG_BOND_DIMS,
    RZ_SYNTHESIS_EPS,
    CIRCUIT_MPS_MAX_BOND,
    COMPRESS_EVERY_N_EDGES,
    N_LAYERS,
    make_chain,
    build_ground_state_context,
    _approx_trial,
)


# =============================================================================
# CONFIG -- edit these before calling analyze(), or pass overrides directly
# =============================================================================
N_SWEEP_VALUES   = list(range(10, 101, 10))          # 10, 20, ..., 100
N_SWEEP_FIXED_N  = 20                                 # N held fixed for the J2 sweep
J2_SWEEP_VALUES  = [round(0.1 * k, 1) for k in range(11)]   # 0.0, 0.1, ..., 1.0


# =============================================================================
# ONE DATA POINT
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
    verbose: bool,
) -> dict:
    """
    Build the ground-state context (MPO + DMRG) for one (N, J1, J2) point,
    run a single approximate-circuit trial, and return a flat dict with
    timing, gate breakdown, and DMRG-MPS overlap.
    """
    chain = make_chain(n_sites)

    row: dict = dict(
        n_sites=n_sites,
        j1=j1,
        j2=j2,
        j2_over_j1=(j2 / j1) if j1 else float("nan"),
        n_layers=n_layers,
        ok=False,
        error=None,
    )

    # ---- stage 1: ground-state context (MPO build + DMRG) -----------------
    t0 = time.perf_counter()
    try:
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

    row["dmrg_energy"] = ctx.E_dmrg
    row["max_chi"]      = ctx.diag.get("max_chi")
    row["mean_chi"]     = ctx.diag.get("mean_chi")
    row["half_chain_entropy"] = ctx.diag.get("half_chain_entropy")

    # ---- stage 2: one approximate-circuit trial ----------------------------
    t0 = time.perf_counter()
    try:
        trial_row = _approx_trial(
            ctx,
            n_layers=n_layers,
            rz_eps=rz_eps,
            verbose=verbose,
            circuit_mps_max_bond=circuit_mps_max_bond,
        )
    except Exception as e:
        row["approx_trial_time_s"] = time.perf_counter() - t0
        row["error"] = f"_approx_trial failed: {e!r}"
        if verbose:
            traceback.print_exc()
        return row
    row["approx_trial_time_s"] = time.perf_counter() - t0
    row["total_time_s"] = row["context_time_s"] + row["approx_trial_time_s"]

    # trial_row already carries, among others:
    #   approx_num_qubits, approx_depth, approx_non_clifford_depth,
    #   approx_cx_count, approx_non_clifford_count, approx_t_count_estimate,
    #   approx_rz_synthesis_eps, approx_n_<gate> for each gate type present,
    #   approx_circuit_energy_mps, approx_dmrg_energy_mps,
    #   approx_abs_error_mps, approx_energy_err_pct_mps,
    #   approx_fidelity_mps, approx_circuit_mps_norm_mps,
    #   n_retries
    row.update(trial_row)
    row["ok"] = True
    return row


# =============================================================================
# SWEEPS
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
    verbose: bool = False,
) -> pd.DataFrame:
    """Sweep chain size N at fixed J1, J2. One DMRG solve + one approx trial per N."""
    print(f"\n[sweep_n]  N in {n_values}  (J1={j1}, J2={j2}, n_layers={n_layers})")
    rows = []
    for n_sites in n_values:
        print(f"\n  -- N={n_sites} --")
        row = _run_one_point(
            n_sites, j1, j2, bond_dims, n_layers, rz_eps,
            compress_every_n_edges, circuit_mps_max_bond, verbose,
        )
        status = "OK" if row["ok"] else f"FAILED ({row['error']})"
        print(f"     status={status}  total_time_s={row.get('total_time_s', float('nan')):.3f}  "
              f"cx={row.get('approx_cx_count')}  fidelity_mps={row.get('approx_fidelity_mps')}")
        rows.append(row)
    return pd.DataFrame(rows)


def sweep_j2(
    j2_values: list[float] = J2_SWEEP_VALUES,
    n_sites: int = N_SWEEP_FIXED_N,
    j1: float = J1,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    n_layers: int = N_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    verbose: bool = False,
) -> pd.DataFrame:
    """Sweep J2 at fixed N (=20 by default) and J1. One DMRG solve + one approx trial per J2."""
    print(f"\n[sweep_j2]  J2 in {j2_values}  (N={n_sites}, J1={j1}, n_layers={n_layers})")
    rows = []
    for j2_val in j2_values:
        print(f"\n  -- J2={j2_val} --")
        row = _run_one_point(
            n_sites, j1, j2_val, bond_dims, n_layers, rz_eps,
            compress_every_n_edges, circuit_mps_max_bond, verbose,
        )
        status = "OK" if row["ok"] else f"FAILED ({row['error']})"
        print(f"     status={status}  total_time_s={row.get('total_time_s', float('nan')):.3f}  "
              f"cx={row.get('approx_cx_count')}  fidelity_mps={row.get('approx_fidelity_mps')}")
        rows.append(row)
    return pd.DataFrame(rows)


# =============================================================================
# COMBINED ENTRY POINT
# =============================================================================
def analyze(
    n_values: list[int] = N_SWEEP_VALUES,
    n_fixed_for_j2: int = N_SWEEP_FIXED_N,
    j2_values: list[float] = J2_SWEEP_VALUES,
    j1: float = J1,
    j2_for_n_sweep: float = J2,
    bond_dims: list[int] = DMRG_BOND_DIMS,
    n_layers: int = N_LAYERS,
    rz_eps: float = RZ_SYNTHESIS_EPS,
    compress_every_n_edges: int = COMPRESS_EVERY_N_EDGES,
    circuit_mps_max_bond: int = CIRCUIT_MPS_MAX_BOND,
    verbose: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full analysis: N-sweep (10..100 step 10, at j2=j2_for_n_sweep) and
    J2-sweep (0..1 step 0.1, at N=n_fixed_for_j2).

    Returns
    -------
    df_n_sweep, df_j2_sweep : pd.DataFrame
        Each row = one (N, J1, J2) point with timing, gate-type breakdown
        (n_<gate> columns), resource counts (cx_count, non_clifford_count,
        t_count_estimate, depth, non_clifford_depth), and DMRG-MPS overlap
        (approx_fidelity_mps, approx_energy_err_pct_mps, etc.).
    """
    df_n_sweep = sweep_n(
        n_values=n_values, j1=j1, j2=j2_for_n_sweep,
        bond_dims=bond_dims, n_layers=n_layers, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond, verbose=verbose,
    )

    df_j2_sweep = sweep_j2(
        j2_values=j2_values, n_sites=n_fixed_for_j2, j1=j1,
        bond_dims=bond_dims, n_layers=n_layers, rz_eps=rz_eps,
        compress_every_n_edges=compress_every_n_edges,
        circuit_mps_max_bond=circuit_mps_max_bond, verbose=verbose,
    )

    _print_summary("N sweep", df_n_sweep, x_col="n_sites")
    _print_summary("J2 sweep", df_j2_sweep, x_col="j2")

    return df_n_sweep, df_j2_sweep


# =============================================================================
# PRETTY PRINT HELPER
# =============================================================================
_DISPLAY_COLS = [
    "approx_cx_count",
    "approx_depth",
    "approx_non_clifford_count",
    "approx_non_clifford_depth",
    "approx_t_count_estimate",
    "context_time_s",
    "approx_trial_time_s",
    "total_time_s",
    "approx_fidelity_mps",
    "approx_energy_err_pct_mps",
]


def _print_summary(title: str, df: pd.DataFrame, x_col: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title} summary")
    print(f"{'='*70}")
    if df.empty:
        print("  (no rows)")
        return
    cols = [x_col] + [c for c in _DISPLAY_COLS if c in df.columns]
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print(df[cols].to_string(index=False))


if __name__ == "__main__":
    df_n, df_j2 = analyze()
