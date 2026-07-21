"""
rbm_to_ucj_pipeline.py
======================
Integrated pipeline: RBM/VMC observables → UCJ parameter prediction
→ warmstarted UCJ optimisation vs Néel baseline.

Stages
------
  1. COLLECT   – sweep J2, run VMC (NetKet RBM), record observables and UCJ
                 ground-truth params in one CSV (pipeline_data.csv).
  2. REGRESS   – train a per-parameter ridge regressor from
                 [E/N, C1, C2, C3] → θJ / θK.  Saves model weights.
  3. EVALUATE  – for each J2: run UCJ from (a) Néel init, (b) RBM-predicted
                 init.  Records energies, overlaps, iteration counts.
  4. ANALYSE   – load results, print a compact summary table, save plots.

Usage
-----
  # Run all stages in sequence (typical first run):
  python rbm_to_ucj_pipeline.py --mode collect
  python rbm_to_ucj_pipeline.py --mode regress
  python rbm_to_ucj_pipeline.py --mode evaluate
  python rbm_to_ucj_pipeline.py --mode analyse

  # One-shot on fresh machine:
  python rbm_to_ucj_pipeline.py --mode all

Configuration
-------------
  Edit the CONFIGURATION block below.  All paths, system sizes, and
  hyper-parameters live there — no magic numbers buried in functions.

Dependencies
------------
  numpy, scipy, jax, netket, matplotlib (analyse only)
  UCJ.py and lattices.py must be importable from the same directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
from scipy.optimize import minimize as scipy_minimize

# ---------------------------------------------------------------------------
# UCJ internals – must be importable from the same directory
# ---------------------------------------------------------------------------
from UCJ import (
    get_ground_state,
    build_jax_hamiltonian,
    build_jastrow_fn,
    build_givens_pairs,
    color_edges,
)
from lattices import make_lattice

# =============================================================================
# CONFIGURATION  — edit only this block
# =============================================================================

# --- System ---
SYSTEM_SIZE  = 16        # must match a square lattice, e.g. 16 = 4×4
J1_VAL       = 1.0
J2_VALUES    = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

# --- UCJ ansatz ---
UCJ_VARIANT   = "re"     # "re" | "im" | "g"
UCJ_K_LAYERS  = 1

# --- UCJ optimiser ---
LBFGS_MAXITER  = 800
LBFGS_MAXFUN   = 50_000
LBFGS_FTOL     = 1e-14
LBFGS_GTOL     = 1e-8
N_RESTARTS_REF = 3       # restarts for the Néel-init baseline
N_RESTARTS_PRE = 1       # restarts for the predicted-param init (already warm)
NOISE_SCALE    = 0.05
SEED           = 23

# --- VMC (NetKet) ---
VMC_ALPHA      = 6       # RBM hidden unit ratio
VMC_N_SAMPLES  = 4096
VMC_N_ITER     = 600
VMC_LR         = 0.01
VMC_DIAG_SHIFT = 0.05
VMC_DTYPE      = complex
VMC_RESTARTS   = 3       # keep the best VMC run per J2 point

# --- Regression ---
RIDGE_ALPHA    = 1e-3    # L2 regularisation strength

# --- Paths ---
DATA_CSV      = Path("pipeline_data.csv")
RESULTS_CSV   = Path("pipeline_results.csv")
REGRESSOR_PKL = Path("param_regressor.pkl")
PLOTS_DIR     = Path("pipeline_plots")

# =============================================================================
# HELPERS
# =============================================================================

_DEVICE = jax.devices("cpu")[0]

def _to_device(x):
    return jax.device_put(x, _DEVICE)


def _get_n_up(n: int) -> int:
    return n // 2


def _stride(variant: str, n_pair: int) -> int:
    return (3 if variant == "g" else 2) * n_pair


# =============================================================================
# STAGE 1 – COLLECT
# =============================================================================

# ── NetKet helpers ──────────────────────────────────────────────────────────

_Sp = np.array([[0, 1], [0, 0]], dtype=complex)
_Sm = np.array([[0, 0], [1, 0]], dtype=complex)
_Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
_SS = (
    np.kron(_Sz, _Sz)
    + 0.5 * np.kron(_Sp, _Sm)
    + 0.5 * np.kron(_Sm, _Sp)
)


def _build_nk_hamiltonian(n, j1, j2, pbc=True):
    n_up     = n // 2
    total_sz = (n_up - (n - n_up)) / 2.0
    hi       = nk.hilbert.Spin(s=0.5, N=n, total_sz=total_sz)
    graph    = nk.graph.Chain(n, pbc=pbc)  # topology only
    nn_edges = [(i, (i + 1) % n) for i in range(n)]
    nnn_edges= [(i, (i + 2) % n) for i in range(n)]
    ha = nk.operator.LocalOperator(hi, dtype=complex)
    for si, sj in nn_edges:
        ha += j1 * nk.operator.LocalOperator(hi, _SS, acting_on=[si, sj])
    for si, sj in nnn_edges:
        ha += j2 * nk.operator.LocalOperator(hi, _SS, acting_on=[si, sj])
    return ha, hi, graph


def _avg_spin_corr(vstate, hi, n, r):
    vals = []
    for i in range(n):
        j  = (i + r) % n
        op = nk.operator.LocalOperator(hi, _SS, acting_on=[i, j])
        vals.append(float(vstate.expect(op).mean.real))
    return float(np.mean(vals))


def _run_vmc_once(n, j1, j2):
    """One VMC run.  Returns dict or None on NaN energy."""
    ha, hi, graph = _build_nk_hamiltonian(n, j1, j2, pbc=True)
    model   = nk.models.RBM(alpha=VMC_ALPHA, param_dtype=VMC_DTYPE)
    sampler = nk.sampler.MetropolisExchange(hi, graph=graph)
    vstate  = nk.vqs.MCState(sampler, model, n_samples=VMC_N_SAMPLES)

    # Néel initialisation
    n_chains = vstate.sampler.n_chains
    neel     = np.ones(n); neel[1::2] = -1
    vstate.sampler_state = vstate.sampler_state.replace(
        σ=jnp.array(np.tile(neel, (n_chains, 1)).astype(np.int8))
    )
    optimizer = nk.optimizer.Sgd(learning_rate=VMC_LR)
    driver    = nk.driver.VMC_SR(
        ha, optimizer,
        diag_shift=VMC_DIAG_SHIFT,
        variational_state=vstate,
    )
    driver.run(n_iter=VMC_N_ITER)
    e_vmc = float(vstate.expect(ha).mean.real)
    if math.isnan(e_vmc):
        return None
    c1 = _avg_spin_corr(vstate, hi, n, r=1)
    c2 = _avg_spin_corr(vstate, hi, n, r=2)
    c3 = _avg_spin_corr(vstate, hi, n, r=3)
    return {"E_VMC": e_vmc, "C1": c1, "C2": c2, "C3": c3}


# ── UCJ helpers ─────────────────────────────────────────────────────────────

def _apply_jastrow(psi, theta_J, jastrow_fn):
    return psi * jnp.exp(1j * jastrow_fn(theta_J))


def _givens_scan(psi, thetas, srcs, dsts, row_ptr, imag=False):
    for k in range(row_ptr.shape[0] - 1):
        s, e = int(row_ptr[k]), int(row_ptr[k + 1])
        if s == e:
            continue
        c, ss = jnp.cos(thetas[k]), jnp.sin(thetas[k])
        ps, pd = psi[srcs[s:e]], psi[dsts[s:e]]
        if imag:
            ns = c * ps - 1j * ss * pd
            nd = -1j * ss * ps + c * pd
        else:
            ns, nd = c * ps - ss * pd, ss * ps + c * pd
        psi = psi.at[srcs[s:e]].set(ns).at[dsts[s:e]].set(nd)
    return psi


def _ucj_state(theta, variant, k_layers, psi0, n_pair, srcs, dsts, row_ptr, jastrow_fn):
    psi    = psi0
    stride = _stride(variant, n_pair)
    for l in range(k_layers):
        off = l * stride
        psi = _apply_jastrow(psi, theta[off:off + n_pair], jastrow_fn)
        psi = _givens_scan(psi, theta[off + n_pair:off + 2 * n_pair],
                           srcs, dsts, row_ptr, imag=(variant == "im"))
        if variant == "g":
            psi = _givens_scan(psi, theta[off + 2 * n_pair:off + 3 * n_pair],
                               srcs, dsts, row_ptr, imag=True)
    return psi


def _energy(psi, apply_H):
    norm = jnp.dot(jnp.conj(psi), psi)
    return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)


def _overlap_with_exact(psi, exact_gs_vec, basis, idx_map):
    """
    |<exact|ucj>|²  — both states normalised in the Fock basis subset.
    `exact_gs_vec` is the full Hilbert-sector vector from ED.
    """
    norm_ucj   = float(jnp.real(jnp.dot(jnp.conj(psi), psi)))
    norm_exact = float(np.dot(np.conj(exact_gs_vec), exact_gs_vec))
    overlap    = float(abs(jnp.dot(jnp.conj(exact_gs_vec), psi)) ** 2)
    return overlap / (norm_ucj * norm_exact)


def _make_val_grad(variant, k_layers, psi0, n_pair, srcs, dsts, row_ptr,
                   jastrow_fn, apply_H):
    def efn(theta):
        psi = _ucj_state(theta, variant, k_layers, psi0, n_pair,
                         srcs, dsts, row_ptr, jastrow_fn)
        return _energy(psi, apply_H)
    return jax.jit(jax.value_and_grad(efn))


def _optimise_ucj(val_grad_fn, x0):
    x0_gpu = _to_device(jnp.array(x0, dtype=jnp.float64))
    val_grad_fn(x0_gpu)  # JIT warm-up

    def scipy_fn(x_np):
        x_gpu = _to_device(jnp.array(x_np, dtype=jnp.float64))
        E, g  = val_grad_fn(x_gpu)
        return float(E), np.array(g, dtype=np.float64)

    result = scipy_minimize(
        scipy_fn, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER, "maxfun": LBFGS_MAXFUN,
                 "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL})
    return np.array(result.x), float(result.fun), int(result.nit), int(result.nfev)


def _run_ucj_with_init(x0, val_grad_fn, n_restarts, noise_rng):
    """Run UCJ from x0 (plus noise restarts).  Return best (params, E, nit)."""
    best_params, best_E, best_nit = None, np.inf, 0
    for r in range(n_restarts):
        noise = NOISE_SCALE * noise_rng.standard_normal(x0.shape)
        xr    = x0 + (noise if r > 0 else 0.0)
        opt_x, opt_E, nit, _ = _optimise_ucj(val_grad_fn, xr)
        print(f"    restart {r}: E={opt_E:.8f}  nit={nit}")
        if opt_E < best_E:
            best_E, best_params, best_nit = opt_E, opt_x, nit
    return best_params, best_E, best_nit


# ── Main collect loop ───────────────────────────────────────────────────────

def stage_collect():
    print("\n" + "=" * 64)
    print("  STAGE 1: COLLECT  (VMC observables + UCJ ground-truth params)")
    print("=" * 64)

    lattice = make_lattice("square", L=SYSTEM_SIZE)
    n       = lattice.n_sites
    pairs   = list(dict.fromkeys(
        (min(i, j), max(i, j)) for (i, j) in lattice.nn_edges
    ))
    n_pair  = len(pairs)
    stride  = _stride(UCJ_VARIANT, n_pair)

    # Dynamic CSV columns
    scalar_fields = [
        "j2", "E_VMC", "E_Lanczos", "E_per_site", "C1", "C2", "C3",
        "E_UCJ", "abs_error", "rel_error",
        "thetaJ_mean", "thetaJ_std", "thetaK_mean", "thetaK_std",
    ]
    tJ_fields = [f"thetaJ_P{k}" for k in range(n_pair)]
    tK_fields = [f"thetaK_P{k}" for k in range(n_pair)]
    fieldnames = scalar_fields + tJ_fields + tK_fields

    rows = []
    rng  = np.random.default_rng(SEED)

    for j2 in J2_VALUES:
        print(f"\n{'─' * 50}")
        print(f"  J2 = {j2:.4f}")

        # ── exact ED ────────────────────────────────────────────────────────
        e_exact, gs_vec, basis, idx_map = get_ground_state(
            n, _get_n_up(n),
            lattice.nn_edges, lattice.nnn_edges,
            J1_VAL, j2,
        )
        print(f"  E_exact = {e_exact:.10f}")

        # ── VMC  ─────────────────────────────────────────────────────────────
        best_vmc = None
        for trial in range(VMC_RESTARTS):
            print(f"  VMC trial {trial + 1}/{VMC_RESTARTS} ...", end=" ", flush=True)
            res = _run_vmc_once(n, J1_VAL, j2)
            if res is None:
                print("NaN — skipped")
                continue
            print(f"E/N={res['E_VMC']/n:.6f}  C1={res['C1']:.4f}")
            if best_vmc is None or res["E_VMC"] < best_vmc["E_VMC"]:
                best_vmc = res
        if best_vmc is None:
            print("  All VMC trials NaN — skipping J2 point")
            continue

        # ── UCJ ground-truth optimisation ────────────────────────────────────
        apply_H     = build_jax_hamiltonian(
            n, _get_n_up(n),
            lattice.nn_edges, lattice.nnn_edges,
            J1_VAL, j2, basis, idx_map,
        )
        jastrow_fn              = build_jastrow_fn(n, basis, pairs)
        srcs, dsts, row_ptr     = build_givens_pairs(n, basis, idx_map, pairs)

        neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
        psi_neel  = _to_device(
            jnp.zeros(len(basis), dtype=jnp.complex128)
            .at[idx_map[neel_bits]].set(1.0)
        )

        val_grad_fn = _make_val_grad(
            UCJ_VARIANT, UCJ_K_LAYERS, psi_neel,
            n_pair, srcs, dsts, row_ptr, jastrow_fn, apply_H,
        )

        best_params, best_E = None, np.inf
        for r in range(N_RESTARTS_REF):
            x0     = NOISE_SCALE * rng.standard_normal(UCJ_K_LAYERS * stride)
            opt_x, opt_E, nit, _ = _optimise_ucj(val_grad_fn, x0)
            print(f"  UCJ restart {r}: E={opt_E:.8f}  |ΔE|={abs(opt_E - e_exact):.2e}  nit={nit}")
            if opt_E < best_E:
                best_E, best_params = opt_E, opt_x

        # ── pack row ─────────────────────────────────────────────────────────
        tJ = best_params[:n_pair]
        tK = best_params[n_pair:2 * n_pair]

        row: dict[str, Any] = {
            "j2":          j2,
            "E_VMC":       best_vmc["E_VMC"],
            "E_Lanczos":   e_exact,
            "E_per_site":  best_vmc["E_VMC"] / n,
            "C1":          best_vmc["C1"],
            "C2":          best_vmc["C2"],
            "C3":          best_vmc["C3"],
            "E_UCJ":       best_E,
            "abs_error":   abs(best_E - e_exact),
            "rel_error":   abs(best_E - e_exact) / abs(e_exact),
            "thetaJ_mean": float(np.mean(tJ)),
            "thetaJ_std":  float(np.std(tJ)),
            "thetaK_mean": float(np.mean(tK)),
            "thetaK_std":  float(np.std(tK)),
        }
        for k in range(n_pair):
            row[f"thetaJ_P{k}"] = float(best_params[k])
            row[f"thetaK_P{k}"] = float(best_params[n_pair + k])
        rows.append(row)

        print(f"  → E_UCJ={best_E:.10f}  |ΔE|={row['abs_error']:.3e}")

    with DATA_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Saved {len(rows)} rows → {DATA_CSV.resolve()}")


# =============================================================================
# STAGE 2 – REGRESS
# =============================================================================

def stage_regress():
    """
    Train per-parameter ridge regressors:
        features: [E/N, C1, C2, C3]
        targets:  thetaJ_P0 … thetaJ_P{n-1}, thetaK_P0 … thetaK_P{n-1}

    Saves a dict to REGRESSOR_PKL:
        {
          "W":    ndarray (n_params, n_features) – regression weights,
          "b":    ndarray (n_params,)             – bias,
          "scaler_mean": ndarray,
          "scaler_std":  ndarray,
          "param_keys":  list[str],
          "n_pair":      int,
        }
    """
    print("\n" + "=" * 64)
    print("  STAGE 2: REGRESS  (features → UCJ param predictor)")
    print("=" * 64)

    if not DATA_CSV.exists():
        raise FileNotFoundError(f"{DATA_CSV} not found — run stage 'collect' first")

    with DATA_CSV.open() as fh:
        rows = list(csv.DictReader(fh))
    print(f"  Loaded {len(rows)} rows from {DATA_CSV}")

    if len(rows) < 3:
        raise ValueError("Need at least 3 data points to fit a regressor")

    # Identify param columns
    param_keys = [k for k in rows[0] if k.startswith("thetaJ_P") or k.startswith("thetaK_P")]
    param_keys.sort()  # thetaJ_P0, thetaJ_P1, …, thetaK_P0, …

    feature_keys = ["E_per_site", "C1", "C2", "C3"]
    X = np.array([[float(r[k]) for k in feature_keys] for r in rows])
    Y = np.array([[float(r[k]) for k in param_keys]   for r in rows])  # (N, n_params)

    # Standardise features
    mu  = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-12
    Xn  = (X - mu) / sig

    # Ridge regression:  (XᵀX + α I) W = XᵀY
    n_feat = Xn.shape[1]
    A = Xn.T @ Xn + RIDGE_ALPHA * np.eye(n_feat)
    W = np.linalg.solve(A, Xn.T @ Y)  # (n_feat, n_params)

    # Bias from residuals at training mean
    b = Y.mean(axis=0) - (Xn.mean(axis=0) @ W)

    # Diagnostics
    Y_pred = Xn @ W + b
    residuals = Y - Y_pred
    r2_per_param = 1.0 - (residuals ** 2).sum(axis=0) / ((Y - Y.mean(axis=0)) ** 2 + 1e-16).sum(axis=0)

    n_pair = len([k for k in param_keys if k.startswith("thetaJ_P")])
    print(f"\n  n_pair={n_pair}   n_param={len(param_keys)}")
    print(f"\n  Per-parameter R² on training data:")
    print(f"    {'param':<20}  R²")
    for k, r2 in zip(param_keys, r2_per_param):
        flag = " ◄ low" if r2 < 0.5 else ""
        print(f"    {k:<20}  {r2:.4f}{flag}")

    payload = {
        "W":            W,
        "b":            b,
        "scaler_mean":  mu,
        "scaler_std":   sig,
        "param_keys":   param_keys,
        "feature_keys": feature_keys,
        "n_pair":       n_pair,
        "ridge_alpha":  RIDGE_ALPHA,
        "mean_r2":      float(r2_per_param.mean()),
    }
    with REGRESSOR_PKL.open("wb") as fh:
        pickle.dump(payload, fh)
    print(f"\n  Regressor saved → {REGRESSOR_PKL.resolve()}")
    print(f"  Mean R² = {payload['mean_r2']:.4f}")


def _predict_params(observables: dict, regressor: dict) -> np.ndarray:
    """Use trained regressor to predict UCJ params from VMC observables."""
    x = np.array([observables[k] for k in regressor["feature_keys"]])
    xn = (x - regressor["scaler_mean"]) / regressor["scaler_std"]
    pred = xn @ regressor["W"] + regressor["b"]
    return pred  # shape: (n_params,)


# =============================================================================
# STAGE 3 – EVALUATE
# =============================================================================

def stage_evaluate():
    """
    For each J2:
      A) Néel init  – N_RESTARTS_REF random starts  (baseline)
      B) Predicted  – regressor → predicted θ as warm x0  (N_RESTARTS_PRE)

    Records for each: final energy, |ΔE|, overlap²(exact|UCJ), nit.
    Writes RESULTS_CSV.
    """
    print("\n" + "=" * 64)
    print("  STAGE 3: EVALUATE  (Néel init vs RBM-predicted init)")
    print("=" * 64)

    for p in [DATA_CSV, REGRESSOR_PKL]:
        if not p.exists():
            raise FileNotFoundError(f"{p} not found — run earlier stages first")

    with DATA_CSV.open() as fh:
        data_rows = {float(r["j2"]): r for r in csv.DictReader(fh)}
    with REGRESSOR_PKL.open("rb") as fh:
        regressor = pickle.load(fh)

    lattice = make_lattice("square", L=SYSTEM_SIZE)
    n       = lattice.n_sites
    pairs   = list(dict.fromkeys(
        (min(i, j), max(i, j)) for (i, j) in lattice.nn_edges
    ))
    n_pair  = len(pairs)
    stride  = _stride(UCJ_VARIANT, n_pair)

    result_fields = [
        "j2",
        # Néel baseline
        "neel_E", "neel_abs_error", "neel_rel_error", "neel_overlap2", "neel_nit",
        # RBM-predicted init
        "pred_E", "pred_abs_error", "pred_rel_error", "pred_overlap2", "pred_nit",
        # Comparison
        "delta_E",        # pred_E - neel_E  (negative = predicted wins)
        "overlap2_gain",  # pred_overlap2 - neel_overlap2
        "nit_reduction",  # neel_nit - pred_nit
        "better_energy",  # 1 if predicted < neel
        "better_overlap", # 1 if predicted overlap > neel overlap
        # Initial state overlaps (before optimisation)
        "neel_init_overlap2",
        "pred_init_overlap2",
        # Regressor quality
        "pred_param_rmse",
    ]
    results = []
    rng = np.random.default_rng(SEED)

    for j2 in J2_VALUES:
        print(f"\n{'─' * 50}")
        print(f"  J2 = {j2:.4f}")

        row = data_rows.get(j2)
        if row is None:
            print("  No data row — skipping (run collect first)")
            continue

        # ── ED ────────────────────────────────────────────────────────────
        e_exact, gs_vec, basis, idx_map = get_ground_state(
            n, _get_n_up(n),
            lattice.nn_edges, lattice.nnn_edges,
            J1_VAL, j2,
        )
        apply_H        = build_jax_hamiltonian(
            n, _get_n_up(n),
            lattice.nn_edges, lattice.nnn_edges,
            J1_VAL, j2, basis, idx_map,
        )
        jastrow_fn              = build_jastrow_fn(n, basis, pairs)
        srcs, dsts, row_ptr     = build_givens_pairs(n, basis, idx_map, pairs)

        # Néel state
        neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
        psi_neel  = _to_device(
            jnp.zeros(len(basis), dtype=jnp.complex128)
            .at[idx_map[neel_bits]].set(1.0)
        )
        neel_init_overlap2 = _overlap_with_exact(psi_neel, gs_vec, basis, idx_map)

        val_grad_fn = _make_val_grad(
            UCJ_VARIANT, UCJ_K_LAYERS, psi_neel,
            n_pair, srcs, dsts, row_ptr, jastrow_fn, apply_H,
        )

        # ── A) Néel baseline ───────────────────────────────────────────────
        print("  [A] Néel init:")
        x0_neel = np.zeros(UCJ_K_LAYERS * stride)
        neel_params, neel_E, neel_nit = _run_ucj_with_init(x0_neel, val_grad_fn, N_RESTARTS_REF, rng)
        neel_psi = _ucj_state(
            _to_device(jnp.array(neel_params)),
            UCJ_VARIANT, UCJ_K_LAYERS, psi_neel,
            n_pair, srcs, dsts, row_ptr, jastrow_fn,
        )
        neel_overlap2 = _overlap_with_exact(neel_psi, gs_vec, basis, idx_map)
        print(f"    E={neel_E:.8f}  |ΔE|={abs(neel_E - e_exact):.3e}  overlap²={neel_overlap2:.6f}  nit={neel_nit}")

        # ── B) Predicted init ──────────────────────────────────────────────
        print("  [B] RBM-predicted init:")
        vmc_obs   = {k: float(row[k]) for k in regressor["feature_keys"]}
        pred_x0   = _predict_params(vmc_obs, regressor).astype(np.float64)

        # If only 1 layer, pad/trim to correct size
        if pred_x0.size < UCJ_K_LAYERS * stride:
            pred_x0 = np.pad(pred_x0, (0, UCJ_K_LAYERS * stride - pred_x0.size))
        pred_x0 = pred_x0[:UCJ_K_LAYERS * stride]

        # RMSE vs ground-truth params from collect stage
        true_params = np.array(
            [float(row[f"thetaJ_P{k}"]) for k in range(n_pair)]
            + [float(row[f"thetaK_P{k}"]) for k in range(n_pair)]
        )
        pred_param_rmse = float(np.sqrt(np.mean((pred_x0[:len(true_params)] - true_params) ** 2)))

        # Overlap of predicted UCJ state BEFORE optimisation
        pred_psi_init = _ucj_state(
            _to_device(jnp.array(pred_x0)),
            UCJ_VARIANT, UCJ_K_LAYERS, psi_neel,
            n_pair, srcs, dsts, row_ptr, jastrow_fn,
        )
        pred_init_overlap2 = _overlap_with_exact(pred_psi_init, gs_vec, basis, idx_map)

        pred_params, pred_E, pred_nit = _run_ucj_with_init(pred_x0, val_grad_fn, N_RESTARTS_PRE, rng)
        pred_psi = _ucj_state(
            _to_device(jnp.array(pred_params)),
            UCJ_VARIANT, UCJ_K_LAYERS, psi_neel,
            n_pair, srcs, dsts, row_ptr, jastrow_fn,
        )
        pred_overlap2 = _overlap_with_exact(pred_psi, gs_vec, basis, idx_map)
        print(f"    E={pred_E:.8f}  |ΔE|={abs(pred_E - e_exact):.3e}  overlap²={pred_overlap2:.6f}  nit={pred_nit}")
        print(f"    param_rmse={pred_param_rmse:.4f}  init_overlap²={pred_init_overlap2:.6f}")

        results.append({
            "j2":                   j2,
            "neel_E":               neel_E,
            "neel_abs_error":       abs(neel_E  - e_exact),
            "neel_rel_error":       abs(neel_E  - e_exact) / abs(e_exact),
            "neel_overlap2":        neel_overlap2,
            "neel_nit":             neel_nit,
            "pred_E":               pred_E,
            "pred_abs_error":       abs(pred_E  - e_exact),
            "pred_rel_error":       abs(pred_E  - e_exact) / abs(e_exact),
            "pred_overlap2":        pred_overlap2,
            "pred_nit":             pred_nit,
            "delta_E":              pred_E - neel_E,
            "overlap2_gain":        pred_overlap2 - neel_overlap2,
            "nit_reduction":        neel_nit - pred_nit,
            "better_energy":        int(pred_E < neel_E),
            "better_overlap":       int(pred_overlap2 > neel_overlap2),
            "neel_init_overlap2":   neel_init_overlap2,
            "pred_init_overlap2":   pred_init_overlap2,
            "pred_param_rmse":      pred_param_rmse,
        })

    with RESULTS_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=result_fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\n  Saved {len(results)} rows → {RESULTS_CSV.resolve()}")


# =============================================================================
# STAGE 4 – ANALYSE
# =============================================================================

def stage_analyse():
    """Print summary table and save plots to PLOTS_DIR."""
    print("\n" + "=" * 64)
    print("  STAGE 4: ANALYSE")
    print("=" * 64)

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} not found — run 'evaluate' first")

    with RESULTS_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    def f(r, k):
        return float(r[k])

    # ── Summary table ────────────────────────────────────────────────────────
    hdr = (f"{'J2':>5}  {'|ΔE|_Néel':>10}  {'|ΔE|_pred':>10}  "
           f"{'ov²_Néel':>10}  {'ov²_pred':>10}  "
           f"{'init_ov²':>10}  {'nit↓':>6}  {'win':>4}")
    print("\n" + hdr)
    print("─" * len(hdr))
    wins_E = wins_ov = 0
    for r in rows:
        better = "✓" if int(r["better_energy"]) or int(r["better_overlap"]) else " "
        if int(r["better_energy"]):  wins_E  += 1
        if int(r["better_overlap"]): wins_ov += 1
        print(
            f"{f(r,'j2'):>5.2f}  "
            f"{f(r,'neel_abs_error'):>10.4e}  "
            f"{f(r,'pred_abs_error'):>10.4e}  "
            f"{f(r,'neel_overlap2'):>10.6f}  "
            f"{f(r,'pred_overlap2'):>10.6f}  "
            f"{f(r,'pred_init_overlap2'):>10.6f}  "
            f"{int(f(r,'nit_reduction')):>6d}  "
            f"{better:>4}"
        )
    n = len(rows)
    print(f"\n  Predicted init wins (energy)  : {wins_E}/{n}")
    print(f"  Predicted init wins (overlap) : {wins_ov}/{n}")

    # ── Plots ────────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("\n  matplotlib not installed — skipping plots")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    j2s    = [f(r, "j2")             for r in rows]
    neel_e = [f(r, "neel_abs_error") for r in rows]
    pred_e = [f(r, "pred_abs_error") for r in rows]
    neel_ov= [f(r, "neel_overlap2")  for r in rows]
    pred_ov= [f(r, "pred_overlap2")  for r in rows]
    init_ov= [f(r, "pred_init_overlap2") for r in rows]
    nit_r  = [f(r, "nit_reduction")  for r in rows]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    ax = axes[0]
    ax.semilogy(j2s, neel_e, "o-", label="Néel init")
    ax.semilogy(j2s, pred_e, "s--", label="RBM-predicted init")
    ax.set_xlabel("J₂"); ax.set_ylabel("|E_UCJ − E_exact|")
    ax.set_title("Energy error vs J₂"); ax.legend()

    ax = axes[1]
    ax.plot(j2s, neel_ov,  "o-",  label="Néel (after opt)")
    ax.plot(j2s, pred_ov,  "s--", label="Predicted (after opt)")
    ax.plot(j2s, init_ov,  "^:", label="Predicted (before opt)", alpha=0.7)
    ax.set_xlabel("J₂"); ax.set_ylabel("|⟨exact|UCJ⟩|²")
    ax.set_title("Overlap² vs J₂"); ax.legend()

    ax = axes[2]
    colors = ["green" if x > 0 else "red" for x in nit_r]
    ax.bar(j2s, nit_r, color=colors, width=0.03)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("J₂"); ax.set_ylabel("nit_Néel − nit_pred")
    ax.set_title("L-BFGS-B iteration reduction")

    fig.tight_layout()
    path = PLOTS_DIR / "summary.png"
    fig.savefig(path, dpi=150)
    print(f"\n  Plot saved → {path.resolve()}")
    plt.close(fig)

    # Scatter: init_overlap² vs final_overlap²
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(init_ov, pred_ov, c=j2s, cmap="viridis", zorder=3)
    lo = min(min(init_ov), min(neel_ov)) * 0.98
    hi = max(max(pred_ov), max(neel_ov)) * 1.02
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="y=x")
    ax.axhline(np.mean(neel_ov), color="tab:blue", linestyle=":", label=f"mean Néel ov² = {np.mean(neel_ov):.4f}")
    ax.set_xlabel("Init |⟨exact|UCJ_pred⟩|² (before opt)")
    ax.set_ylabel("Final |⟨exact|UCJ_pred⟩|² (after opt)")
    ax.set_title("RBM-init: starting vs final overlap")
    ax.legend(); fig.tight_layout()
    path2 = PLOTS_DIR / "overlap_scatter.png"
    fig.savefig(path2, dpi=150); plt.close(fig)
    print(f"  Plot saved → {path2.resolve()}")


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="RBM-observable → UCJ-param prediction pipeline"
    )
    parser.add_argument(
        "--mode",
        choices=["collect", "regress", "evaluate", "analyse", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()

    if args.mode in ("collect", "all"):
        stage_collect()
    if args.mode in ("regress", "all"):
        stage_regress()
    if args.mode in ("evaluate", "all"):
        stage_evaluate()
    if args.mode in ("analyse", "all"):
        stage_analyse()


if __name__ == "__main__":
    main()
