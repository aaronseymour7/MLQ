"""
rbm_ucj_filter.py
=================
Three stages, zero variational optimisation.

  COLLECT  – sweep J2, run NetKet RBM/VMC, record observables
             (E/N, C1, C2, C3) and exact-ED ground state.
             Also records the UCJ energy expectation of the Néel
             state (single forward pass, no opt) as a baseline.

  REGRESS  – ridge regression: [E/N, C1, C2, C3] → θ̂ (UCJ params).
             Ground-truth θ are taken from a reference UCJ sweep CSV
             (j2_sweep_re_ucj_N16.csv by default, or generated inline
             if that file is absent and GENERATE_REF=True).

  EVALUATE – for each J2:
               1. run VMC → observables
               2. predict θ̂ with regressor
               3. build UCJ state with θ̂  (one forward pass, no opt)
               4. record |<exact|UCJ(θ̂)>|²  vs  |<exact|Néel>|²
             Nothing is optimised.  The overlap ratio is the filter.

  ANALYSE  – print summary table, save plots.

Usage
-----
  python rbm_ucj_filter.py --mode collect
  python rbm_ucj_filter.py --mode regress
  python rbm_ucj_filter.py --mode evaluate
  python rbm_ucj_filter.py --mode analyse
  python rbm_ucj_filter.py --mode all
"""

from __future__ import annotations

import argparse
import csv
import math
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

from UCJ import (
    get_ground_state,
    build_jastrow_fn,
    build_givens_pairs,
    color_edges,
)
from lattices import make_lattice

# =============================================================================
# CONFIGURATION
# =============================================================================

SYSTEM_SIZE = 16
J1_VAL      = 1.0


J2_TRAIN = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.30,
    0.40,

]

J2_TEST = [
    0.025,
    0.075,
    0.125,
    0.175,
    0.25,
    0.35,
    0.45,
    0.50,
]

# Combined list kept for reference
J2_VALUES = sorted(set(J2_TRAIN) | set(J2_TEST))

UCJ_VARIANT = "re"
UCJ_K_LAYERS = 1

# VMC settings
VMC_ALPHA      = 4
VMC_N_SAMPLES  = 4096
VMC_N_ITER     = 1000
VMC_LR         = 0.01
VMC_DIAG_SHIFT = 0.05
VMC_DTYPE      = complex
VMC_RESTARTS   = 3      # keep best energy across restarts

# Regression
RIDGE_ALPHA = 1e-3

# Reference UCJ params CSV (output of j2_sweep_re_ucj_N16.py).
# Must contain columns  j2, thetaJ_L0_P{k}, thetaK_L0_P{k}.
REF_UCJ_CSV = Path("j2_sweep_re_ucj_N16.csv")

# Outputs
COLLECT_CSV   = Path("rbm_observables.csv")
RESULTS_CSV   = Path("filter_results.csv")
REGRESSOR_PKL = Path("filter_regressor.pkl")
PLOTS_DIR     = Path("filter_plots")

# =============================================================================
# SHARED UTILITIES
# =============================================================================

_DEVICE = jax.devices("cpu")[0]

def _to_dev(x):
    return jax.device_put(x, _DEVICE)

def _n_up(n):
    return n // 2

def _stride(variant, n_pair):
    return (3 if variant == "g" else 2) * n_pair


# --- Spin operators ---
_Sp = np.array([[0, 1], [0, 0]], dtype=complex)
_Sm = np.array([[0, 0], [1, 0]], dtype=complex)
_Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)
_SS = (np.kron(_Sz, _Sz)
       + 0.5 * np.kron(_Sp, _Sm)
       + 0.5 * np.kron(_Sm, _Sp))


# --- UCJ forward pass (no grad, no opt) ---

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


def ucj_state(theta, psi0, n_pair, srcs, dsts, row_ptr, jastrow_fn,
              variant=UCJ_VARIANT, k_layers=UCJ_K_LAYERS):
    """Single forward pass — returns unnormalised UCJ state vector."""
    psi    = psi0
    stride = _stride(variant, n_pair)
    theta  = _to_dev(jnp.array(theta, dtype=jnp.complex128))
    for l in range(k_layers):
        off = l * stride
        tJ  = jnp.real(theta[off : off + n_pair])
        psi = psi * jnp.exp(1j * jastrow_fn(tJ))
        psi = _givens_scan(psi, jnp.real(theta[off + n_pair : off + 2 * n_pair]),
                           srcs, dsts, row_ptr, imag=(variant == "im"))
        if variant == "g":
            psi = _givens_scan(psi,
                               jnp.real(theta[off + 2 * n_pair : off + 3 * n_pair]),
                               srcs, dsts, row_ptr, imag=True)
    return psi


def overlap2(psi, gs_vec):
    """
    |<gs|psi>|² / (<gs|gs> * <psi|psi>)
    gs_vec : numpy array in the same Fock-sector basis as psi
    """
    psi_np = np.array(psi)
    num    = abs(np.dot(np.conj(gs_vec), psi_np)) ** 2
    den    = float(np.real(np.dot(np.conj(gs_vec), gs_vec))) \
           * float(np.real(np.dot(np.conj(psi_np), psi_np)))
    return float(num / den) if den > 0 else 0.0


def neel_state(n, basis, idx_map):
    bits = sum(1 << i for i in range(n) if i % 2 == 0)
    return _to_dev(
        jnp.zeros(len(basis), dtype=jnp.complex128).at[idx_map[bits]].set(1.0)
    )


# =============================================================================
# STAGE 1 — COLLECT
# =============================================================================

def _nk_hamiltonian(n, j1, j2):
    n_up = _n_up(n)
    hi   = nk.hilbert.Spin(s=0.5, N=n, total_sz=(n_up - (n - n_up)) / 2.0)
    graph = nk.graph.Chain(n, pbc=True)
    ha    = nk.operator.LocalOperator(hi, dtype=complex)
    for i in range(n):
        ha += j1 * nk.operator.LocalOperator(hi, _SS, acting_on=[i, (i+1)%n])
        ha += j2 * nk.operator.LocalOperator(hi, _SS, acting_on=[i, (i+2)%n])
    return ha, hi, graph


def _avg_corr(vstate, hi, n, r):
    return float(np.mean([
        vstate.expect(nk.operator.LocalOperator(
            hi, _SS, acting_on=[i, (i+r)%n]
        )).mean.real
        for i in range(n)
    ]))


def _run_vmc(n, j1, j2):
    ha, hi, graph = _nk_hamiltonian(n, j1, j2)
    model   = nk.models.RBM(alpha=VMC_ALPHA, param_dtype=VMC_DTYPE)
    sampler = nk.sampler.MetropolisExchange(hi, graph=graph)
    vstate  = nk.vqs.MCState(sampler, model, n_samples=VMC_N_SAMPLES)
    neel    = np.ones(n); neel[1::2] = -1
    vstate.sampler_state = vstate.sampler_state.replace(
        σ=jnp.array(np.tile(neel, (vstate.sampler.n_chains, 1)).astype(np.int8))
    )
    driver = nk.driver.VMC_SR(
        ha, nk.optimizer.Sgd(learning_rate=VMC_LR),
        diag_shift=VMC_DIAG_SHIFT, variational_state=vstate,
    )
    driver.run(n_iter=VMC_N_ITER)
    e = float(vstate.expect(ha).mean.real)
    if math.isnan(e):
        return None
    return {
        "E_VMC": e,
        "E_per_site": e / n,
        "C1": _avg_corr(vstate, hi, n, 1),
        "C2": _avg_corr(vstate, hi, n, 2),
        "C3": _avg_corr(vstate, hi, n, 3),
    }


def stage_collect():
    print("\n" + "="*60)
    print("  COLLECT: VMC observables")
    print("="*60)

    fields = ["j2", "E_VMC", "E_per_site", "C1", "C2", "C3", "E_exact"]
    rows   = []
    n      = SYSTEM_SIZE

    for j2 in J2_VALUES: 
        print(f"\n  J2={j2:.3f}")
        lattice = make_lattice("chain", L=n)
        e_exact, _, _, _ = get_ground_state(
            n, _n_up(n), lattice.nn_edges, lattice.nnn_edges, J1_VAL, j2
        )
        print(f"  E_exact = {e_exact:.8f}")

        best = None
        for k in range(VMC_RESTARTS):
            print(f"  VMC trial {k+1}/{VMC_RESTARTS} ...", end=" ", flush=True)
            res = _run_vmc(n, J1_VAL, j2)
            if res is None:
                print("NaN — skip")
                continue
            print(f"E/N={res['E_per_site']:.6f}  C1={res['C1']:.4f}  C2={res['C2']:.4f}")
            if best is None or res["E_VMC"] < best["E_VMC"]:
                best = res

        if best is None:
            print("  All trials NaN — skipping")
            continue

        row = {"j2": j2, "E_exact": e_exact, **best}
        rows.append(row)

    with COLLECT_CSV.open("w", newline="") as fh:
        csv.DictWriter(fh, fieldnames=fields).writeheader()
        csv.DictWriter(fh, fieldnames=fields).writerows(rows)

    print(f"\n  Saved {len(rows)} rows → {COLLECT_CSV}")


# =============================================================================
# STAGE 2 — REGRESS
# =============================================================================

FEATURE_KEYS = ["E_per_site", "C1", "C2", "C3"]


def _load_ref_params(n_pair, allowed_j2=None):
    """
    Load ground-truth UCJ params from j2_sweep_re_ucj_N16.csv.
    Returns dict: j2 → np.ndarray of shape (2*n_pair,)
                  order: [θJ_P0…θJ_P{n-1}, θK_P0…θK_P{n-1}]
    """
    if not REF_UCJ_CSV.exists():
        raise FileNotFoundError(
            f"Reference UCJ CSV not found: {REF_UCJ_CSV}\n"
            "Run j2_sweep_re_ucj_N16.py first, or point REF_UCJ_CSV at your file."
        )
    result = {}
    with REF_UCJ_CSV.open() as fh:
        for row in csv.DictReader(fh):
            j2 = float(row["j2"])
            if allowed_j2 is not None and j2 not in allowed_j2:
                continue          # ← silently exclude test points
            tJ  = np.array([float(row[f"thetaJ_L0_P{k}"]) for k in range(n_pair)])
            tK  = np.array([float(row[f"thetaK_L0_P{k}"]) for k in range(n_pair)])
            result[j2] = np.concatenate([tJ, tK])
    return result


def stage_regress():
    print("\n" + "="*60)
    print("  REGRESS: VMC features → UCJ params")
    print("="*60)

    if not COLLECT_CSV.exists():
        raise FileNotFoundError(f"{COLLECT_CSV} missing — run collect first")

    with COLLECT_CSV.open() as fh:
        obs_rows = {float(r["j2"]): r for r in csv.DictReader(fh)}

    lattice = make_lattice("chain", L=SYSTEM_SIZE)
    pairs   = list(dict.fromkeys(
        (min(i,j), max(i,j)) for (i,j) in lattice.nn_edges
    ))
    n_pair  = len(pairs)

    ref_params = _load_ref_params(n_pair, allowed_j2=set(J2_TRAIN))
    
    # Align j2 values present in both
    common_j2 = sorted(set(obs_rows) & set(ref_params) & set(J2_TRAIN))
    if len(common_j2) < 3:
        raise ValueError(f"Only {len(common_j2)} common J2 points — need ≥3 to fit")
    print(f"  Fitting on {len(common_j2)} J2 points: {common_j2}")

    X = np.array([[float(obs_rows[j][k]) for k in FEATURE_KEYS] for j in common_j2])
    Y = np.array([ref_params[j]          for j in common_j2])   # (N, 2*n_pair)

    mu  = X.mean(0); sig = X.std(0) + 1e-12
    Xn  = (X - mu) / sig

    n_feat = Xn.shape[1]
    W = np.linalg.solve(Xn.T @ Xn + RIDGE_ALPHA * np.eye(n_feat), Xn.T @ Y)
    b = Y.mean(0) - Xn.mean(0) @ W

    Y_pred = Xn @ W + b
    ss_res = ((Y - Y_pred)**2).sum(0)
    ss_tot = ((Y - Y.mean(0))**2).sum(0) + 1e-16
    r2     = 1.0 - ss_res / ss_tot

    param_keys = [f"thetaJ_P{k}" for k in range(n_pair)] \
               + [f"thetaK_P{k}" for k in range(n_pair)]

    print(f"\n  Per-parameter R² (training):")
    for k, r2v in zip(param_keys, r2):
        flag = "  ◄ weak" if r2v < 0.5 else ""
        print(f"    {k:<20} R²={r2v:.4f}{flag}")
    print(f"\n  Mean R² = {r2.mean():.4f}")

    payload = {
        "W": W, "b": b,
        "scaler_mean": mu, "scaler_std": sig,
        "feature_keys": FEATURE_KEYS,
        "param_keys": param_keys,
        "n_pair": n_pair,
        "r2": r2,
        "mean_r2": float(r2.mean()),
    }
    with REGRESSOR_PKL.open("wb") as fh:
        pickle.dump(payload, fh)
    print(f"\n  Saved → {REGRESSOR_PKL}")


def predict_params(obs: dict, reg: dict) -> np.ndarray:
    x  = np.array([obs[k] for k in reg["feature_keys"]])
    xn = (x - reg["scaler_mean"]) / reg["scaler_std"]
    return (xn @ reg["W"] + reg["b"]).astype(np.float64)


# =============================================================================
# STAGE 3 — EVALUATE  (single UCJ forward pass per J2, no opt)
# =============================================================================

def stage_evaluate():
    print("\n" + "="*60)
    print("  EVALUATE: single UCJ forward pass, no optimisation")
    print("="*60)

    for p in [COLLECT_CSV, REGRESSOR_PKL]:
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — run earlier stages first")

    with COLLECT_CSV.open() as fh:
        obs_rows = {float(r["j2"]): r for r in csv.DictReader(fh)}
    with REGRESSOR_PKL.open("rb") as fh:
        reg = pickle.load(fh)

    n      = SYSTEM_SIZE
    lattice = make_lattice("chain", L=n)
    pairs  = list(dict.fromkeys(
        (min(i,j), max(i,j)) for (i,j) in lattice.nn_edges
    ))
    n_pair = len(pairs)

    fields = [
        "j2", "E_exact",
        "neel_overlap2",       # |<exact|Néel>|²  (no UCJ at all)
        "ucj_pred_overlap2",   # |<exact|UCJ(θ̂)>|²  (predicted θ, no opt)
        "overlap2_gain",       # ucj_pred - neel
        "overlap2_ratio",      # ucj_pred / neel
        "pred_param_rmse",     # vs ground-truth θ from ref CSV
        "better",              # 1 if predicted UCJ beats Néel
    ]
    rows = []

    # Load ground-truth params for RMSE
    try:
        ref_params = _load_ref_params(n_pair, allowed_j2=None)
    except FileNotFoundError:
        ref_params = {}

    for j2 in J2_TEST: 
        obs = obs_rows.get(j2)
        if obs is None:
            print(f"  J2={j2:.3f}  no VMC data — skip")
            continue

        print(f"\n  J2={j2:.3f}")

        # Exact ground state
        e_exact, gs_vec, basis, idx_map = get_ground_state(
            n, _n_up(n), lattice.nn_edges, lattice.nnn_edges, J1_VAL, j2
        )

        # Shared UCJ structures
        jastrow_fn          = build_jastrow_fn(n, basis, pairs)
        srcs, dsts, row_ptr = build_givens_pairs(n, basis, idx_map, pairs)
        psi0                = neel_state(n, basis, idx_map)

        # 1. Néel overlap (baseline, pure |Néel⟩ no circuit)
        ov_neel = overlap2(psi0, gs_vec)

        # 2. Predict θ̂ from VMC observables
        theta_hat = predict_params(
            {k: float(obs[k]) for k in reg["feature_keys"]}, reg
        )
        # Pad/trim to circuit size
        full_size = UCJ_K_LAYERS * _stride(UCJ_VARIANT, n_pair)
        theta_hat = np.resize(theta_hat, full_size)

        # 3. Single UCJ forward pass with θ̂
        psi_ucj = ucj_state(theta_hat, psi0, n_pair, srcs, dsts, row_ptr, jastrow_fn)
        ov_ucj  = overlap2(psi_ucj, gs_vec)

        # 4. RMSE vs reference (if available)
        true_theta = ref_params.get(j2)
        if true_theta is not None:
            rmse = float(np.sqrt(np.mean((theta_hat[:len(true_theta)] - true_theta)**2)))
        else:
            rmse = float("nan")

        gain  = ov_ucj - ov_neel
        ratio = ov_ucj / ov_neel if ov_neel > 0 else float("nan")

        print(f"    |<exact|Néel>|²        = {ov_neel:.6f}")
        print(f"    |<exact|UCJ(θ̂)>|²     = {ov_ucj:.6f}")
        print(f"    gain                    = {gain:+.6f}  ratio = {ratio:.3f}")
        print(f"    param RMSE vs ref       = {rmse:.4f}")
        print(f"    better than Néel?       = {'YES ✓' if ov_ucj > ov_neel else 'no'}")

        rows.append({
            "j2":                j2,
            "E_exact":           e_exact,
            "neel_overlap2":     ov_neel,
            "ucj_pred_overlap2": ov_ucj,
            "overlap2_gain":     gain,
            "overlap2_ratio":    ratio,
            "pred_param_rmse":   rmse,
            "better":            int(ov_ucj > ov_neel),
        })

    with RESULTS_CSV.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"\n  Saved {len(rows)} rows → {RESULTS_CSV}")


# =============================================================================
# STAGE 4 — ANALYSE
# =============================================================================

def stage_analyse():
    print("\n" + "="*60)
    print("  ANALYSE")
    print("="*60)

    if not RESULTS_CSV.exists():
        raise FileNotFoundError(f"{RESULTS_CSV} missing — run evaluate first")

    with RESULTS_CSV.open() as fh:
        rows = list(csv.DictReader(fh))

    def f(r, k): return float(r[k])

    wins = sum(int(r["better"]) for r in rows)
    print(f"\n  {'J2':>5}  {'|<gs|Néel>|²':>14}  {'|<gs|UCJ(θ̂)>|²':>15}  "
          f"{'gain':>9}  {'ratio':>7}  {'RMSE':>7}  result")
    print("  " + "─"*72)
    for r in rows:
        tag = "✓ better" if int(r["better"]) else "  worse "
        print(
            f"  {f(r,'j2'):>5.3f}  "
            f"{f(r,'neel_overlap2'):>14.6f}  "
            f"{f(r,'ucj_pred_overlap2'):>15.6f}  "
            f"{f(r,'overlap2_gain'):>+9.6f}  "
            f"{f(r,'overlap2_ratio'):>7.3f}  "
            f"{f(r,'pred_param_rmse'):>7.4f}  "
            f"{tag}"
        )
    print(f"\n  UCJ(θ̂) beats Néel in {wins}/{len(rows)} cases")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    PLOTS_DIR.mkdir(exist_ok=True)
    j2s    = [f(r, "j2")                for r in rows]
    ov_n   = [f(r, "neel_overlap2")     for r in rows]
    ov_u   = [f(r, "ucj_pred_overlap2") for r in rows]
    rmses  = [f(r, "pred_param_rmse")   for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    ax = axes[0]
    ax.plot(j2s, ov_n, "o-",  label="|⟨gs|Néel⟩|²  (baseline)")
    ax.plot(j2s, ov_u, "s--", label="|⟨gs|UCJ(θ̂)⟩|²  (RBM-predicted, no opt)")
    ax.fill_between(j2s, ov_n, ov_u,
                    where=[u > b for u, b in zip(ov_u, ov_n)],
                    alpha=0.15, color="green", label="gain region")
    ax.fill_between(j2s, ov_n, ov_u,
                    where=[u <= b for u, b in zip(ov_u, ov_n)],
                    alpha=0.15, color="red", label="loss region")
    ax.set_xlabel("J₂/J₁"); ax.set_ylabel("|⟨gs|ψ⟩|²")
    ax.set_title("Overlap with exact GS — no optimisation")
    ax.legend(fontsize=8)

    ax = axes[1]
    gains  = [f(r, "overlap2_gain") for r in rows]
    colors = ["tab:green" if g > 0 else "tab:red" for g in gains]
    ax.bar(j2s, gains, color=colors, width=0.03)
    ax.axhline(0, color="black", lw=0.8)
    ax2 = ax.twinx()
    ax2.plot(j2s, rmses, "k^--", markersize=5, alpha=0.6, label="param RMSE")
    ax2.set_ylabel("param RMSE", color="gray")
    ax.set_xlabel("J₂/J₁"); ax.set_ylabel("overlap² gain")
    ax.set_title("Gain and regressor quality vs J₂")
    ax2.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    out = PLOTS_DIR / "filter_summary.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"\n  Plot saved → {out}")


# =============================================================================
# CLI
# =============================================================================

def main():
    p = argparse.ArgumentParser(description="RBM → UCJ filter (no optimisation)")
    p.add_argument("--mode",
                   choices=["collect", "regress", "evaluate", "analyse", "all"],
                   default="all")
    args = p.parse_args()

    if args.mode in ("collect", "all"):   stage_collect()
    if args.mode in ("regress", "all"):   stage_regress()
    if args.mode in ("evaluate", "all"):  stage_evaluate()
    if args.mode in ("analyse", "all"):   stage_analyse()


if __name__ == "__main__":
    stage_collect()
