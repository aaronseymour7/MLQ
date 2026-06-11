"""
j2_sweep_re_ucj.py
------------------
Sweep J2 for the Re-UCJ ansatz on a 16-site (4×4) square lattice.
For each J2 value the optimised parameters are saved, plus summary
statistics for the Jastrow (thetaJ) and Givens (thetaK) blocks.

Output CSV columns
------------------
j2
E_exact              -- ground-state energy from exact diagonalisation
E_ucj                -- best variational energy found
abs_error            -- |E_ucj - E_exact|
rel_error            -- |E_ucj - E_exact| / |E_exact|
thetaJ_mean          -- mean of optimised Jastrow parameters (all layers)
thetaJ_std           -- std  of optimised Jastrow parameters (all layers)
thetaK_mean          -- mean of optimised Givens  parameters (all layers)
thetaK_std           -- std  of optimised Givens  parameters (all layers)
thetaJ_L{l}_P{k}     -- individual Jastrow parameter for layer l, pair index k
thetaK_L{l}_P{k}     -- individual Givens  parameter for layer l, pair index k

Usage
-----
    python j2_sweep_re_ucj.py

Adjust J2_VALUES and the constants at the top of this file as needed.
The script reuses UCJ.py internals to skip the Qiskit circuit build,
keeping the sweep fast.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax
# ---------------------------------------------------------------------------
# UCJ internals — import from the file in the same directory
# ---------------------------------------------------------------------------
from UCJ import *
from lattices import make_lattice

# =============================================================================
# SWEEP CONFIGURATION  — edit here
# =============================================================================
J2_VALUES   = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
J1_VAL      = 1          # 1.0  (from UCJ global)
VARIANT     = "re"
K_LAYERS_SW = 1    # 1    (from UCJ global)
N_RESTARTS = 3
OUTPUT_CSV  = Path("j2_sweep_re_ucj_N16.csv")
_DEVICE = jax.devices("cpu")[0]
NOISE_SCALE     = 0.05
SEED            = 23

LBFGS_MAXITER   = 800
LBFGS_MAXFUN    = 50_000
LBFGS_FTOL      = 1e-14
LBFGS_GTOL      = 1e-8

TARGET_BASIS    = ["cx", "rz", "h", "s", "sdg"]
OPT_LEVEL       = 3  

def _to_device(x):
    return jax.device_put(x, _DEVICE)
def _get_n_up(n: int) -> int:
    return n // 2
def _make_val_grad(variant, k_layers, psi0, n_pair, srcs, dsts, row_ptr,
                   jastrow_fn, apply_H):
    def efn(theta):
        psi = _ucj_state(theta, variant, k_layers, psi0, n_pair,
                         srcs, dsts, row_ptr, jastrow_fn)
        return _energy(psi, apply_H)

    return jax.jit(jax.value_and_grad(efn))


def _optimise(val_grad_fn, x0):
    x0_gpu = _to_device(jnp.array(x0, dtype=jnp.float64))
    val_grad_fn(x0_gpu)  # warm-up JIT

    def scipy_fn(x_np):
        x_gpu = _to_device(jnp.array(x_np, dtype=jnp.float64))
        E, g  = val_grad_fn(x_gpu)
        return float(E), np.array(g, dtype=np.float64)

    result = scipy_minimize(
        scipy_fn, x0, jac=True, method="L-BFGS-B",
        options={"maxiter": LBFGS_MAXITER, "maxfun": LBFGS_MAXFUN,
                 "ftol": LBFGS_FTOL, "gtol": LBFGS_GTOL})
    return np.array(result.x), float(result.fun), result


def _run_optimisation(variant, k_layers, n, n_pair, psi_neel,
                      srcs, dsts, row_ptr, jastrow_fn, apply_H, e_exact):
    val_grad_fn = _make_val_grad(variant, k_layers, psi_neel, n_pair,
                                 srcs, dsts, row_ptr, jastrow_fn, apply_H)
    stride = 3 * n_pair if variant == "g" else 2 * n_pair

    best_params, best_E = None, np.inf
    rng = np.random.default_rng(23)

    for restart in range(N_RESTARTS):
        x0 = NOISE_SCALE * rng.standard_normal(k_layers * stride)
        opt_x, opt_E, result = _optimise(val_grad_fn, x0)
        print(f"  [restart {restart}]  E={opt_E:.8f}  "
              f"|ΔE|={abs(opt_E - e_exact):.4e}  "
              f"nit={result.nit}  nfev={result.nfev}")
        if opt_E < best_E:
            best_E, best_params = opt_E, opt_x

    return best_params, best_E

def _givens_scan(psi, thetas, srcs, dsts, row_ptr, imag=False):
    for k in range(row_ptr.shape[0] - 1):
        s, e = int(row_ptr[k]), int(row_ptr[k + 1])
        if s == e:
            continue
        c, ss = jnp.cos(thetas[k]), jnp.sin(thetas[k])
        ps, pd = psi[srcs[s:e]], psi[dsts[s:e]]
        if imag:
            ns, nd = c * ps - 1j * ss * pd, -1j * ss * ps + c * pd
        else:
            ns, nd = c * ps - ss * pd, ss * ps + c * pd
        psi = psi.at[srcs[s:e]].set(ns).at[dsts[s:e]].set(nd)
    return psi


def _ucj_state(theta, variant, k_layers, psi0, n_pair, srcs, dsts, row_ptr, jastrow_fn):
    psi    = psi0
    stride = 3 * n_pair if variant == "g" else 2 * n_pair
    for l in range(k_layers):
        off = l * stride
        psi = apply_jastrow(psi, theta[off:off + n_pair], jastrow_fn)
        psi = _givens_scan(psi, theta[off + n_pair:off + 2 * n_pair],
                           srcs, dsts, row_ptr, imag=(variant == "im"))
        if variant == "g":
            psi = _givens_scan(psi, theta[off + 2 * n_pair:off + 3 * n_pair],
                               srcs, dsts, row_ptr, imag=True)
    return psi


def _energy(psi, apply_H):
    norm = jnp.dot(jnp.conj(psi), psi)
    return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)

def apply_jastrow(psi, theta_J, jastrow_phase_fn):
    return psi * jnp.exp(1j * jastrow_phase_fn(theta_J))
# =============================================================================
# PER-J2 ROUTINE
# =============================================================================

def _run_one(lattice, pairs, j2: float) -> dict:
    """Optimise UCJ for a single J2 value and return a result dict."""
    n    = lattice.n_sites
    n_up = _get_n_up(n)

    # ── exact diagonalisation ────────────────────────────────────────────────
    e_exact, _, basis, idx_map = get_ground_state(
        n, n_up,
        lattice.nn_edges, lattice.nnn_edges,
        J1_VAL, j2,
    )

    # ── JAX Hamiltonian + ansatz structures ──────────────────────────────────
    apply_H    = build_jax_hamiltonian(
        n, n_up,
        lattice.nn_edges, lattice.nnn_edges,
        J1_VAL, j2, basis, idx_map,
    )
    jastrow_fn              = build_jastrow_fn(n, basis, pairs)
    srcs, dsts, row_ptr     = build_givens_pairs(n, basis, idx_map, pairs)

    # Néel initial state  |↑↓↑↓…⟩
    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi_neel  = _to_device(
        jnp.zeros(len(basis), dtype=jnp.complex128)
        .at[idx_map[neel_bits]].set(1.0)
    )

    # ── variational optimisation ─────────────────────────────────────────────
    best_params, best_E = _run_optimisation(
        VARIANT, K_LAYERS_SW, n, len(pairs),
        psi_neel, srcs, dsts, row_ptr,
        jastrow_fn, apply_H, e_exact,
    )

    # ── split params into thetaJ / thetaK blocks ─────────────────────────────
    # For variant='re', stride = 2 * n_pair per layer.
    # Layout per layer:  [thetaJ_0 … thetaJ_{n_pair-1} | thetaK_0 … thetaK_{n_pair-1}]
    n_pair = len(pairs)
    stride = 2 * n_pair          # 're'/'im' only; 'g' would be 3*n_pair

    tJ_all, tK_all = [], []
    for layer in range(K_LAYERS_SW):
        off   = layer * stride
        tJ_all.append(best_params[off          : off + n_pair])
        tK_all.append(best_params[off + n_pair : off + 2 * n_pair])

    tJ = np.concatenate(tJ_all)
    tK = np.concatenate(tK_all)

    abs_err = abs(best_E - e_exact)
    rel_err = abs_err / abs(e_exact) if e_exact != 0.0 else float("nan")

    row: dict = {
        "j2":          j2,
        "E_exact":     e_exact,
        "E_ucj":       best_E,
        "abs_error":   abs_err,
        "rel_error":   rel_err,
        "thetaJ_mean": float(np.mean(tJ)),
        "thetaJ_std":  float(np.std(tJ)),
        "thetaK_mean": float(np.mean(tK)),
        "thetaK_std":  float(np.std(tK)),
    }

    # ── per-parameter values, labelled by layer and pair index ───────────────
    # columns: thetaJ_L{l}_P{k}  and  thetaK_L{l}_P{k}
    for layer in range(K_LAYERS_SW):
        off = layer * stride
        for k in range(n_pair):
            row[f"thetaJ_L{layer}_P{k}"] = float(best_params[off + k])
            row[f"thetaK_L{layer}_P{k}"] = float(best_params[off + n_pair + k])

    return row


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    lattice = make_lattice("square", L=16)   # 4×4 = 16 sites
    n = lattice.n_sites
    assert n == 16, f"Expected N=16, got N={n}"

    # Deduplicated NN pairs — same convention as build_ucj
    pairs = list(dict.fromkeys(
        (min(i, j), max(i, j)) for (i, j) in lattice.nn_edges
    ))
    rounds = color_edges(pairs)   # used only for the print below

    sep = "=" * 64
    print(f"\n{sep}")
    print(f"  Re-UCJ J2 sweep  |  N={n}  variant={VARIANT}  k={K_LAYERS_SW}")
    print(f"  J1={J1_VAL}  |  {len(pairs)} NN pairs  |  "
          f"{len(rounds)} parallel rounds")
    print(f"  J2 values : {J2_VALUES}")
    print(f"  Output    : {OUTPUT_CSV}")
    print(f"{sep}\n")

    # Build fieldnames dynamically so per-parameter columns match n_pair/k_layers.
    # Order: scalar summary columns first, then all thetaJ individual values
    # (layer-major), then all thetaK individual values (layer-major).
    n_pair = len(pairs)
    scalar_fields = [
        "j2",
        "E_exact", "E_ucj", "abs_error", "rel_error",
        "thetaJ_mean", "thetaJ_std",
        "thetaK_mean", "thetaK_std",
    ]
    thetaJ_fields = [
        f"thetaJ_L{l}_P{k}"
        for l in range(K_LAYERS_SW)
        for k in range(n_pair)
    ]
    thetaK_fields = [
        f"thetaK_L{l}_P{k}"
        for l in range(K_LAYERS_SW)
        for k in range(n_pair)
    ]
    FIELDNAMES = scalar_fields + thetaJ_fields + thetaK_fields

    print(f"  CSV columns : {len(FIELDNAMES)}  "
          f"({len(scalar_fields)} scalar + "
          f"{len(thetaJ_fields)} thetaJ + {len(thetaK_fields)} thetaK)")

    rows: list[dict] = []
    for j2 in J2_VALUES:
        print(f"\n{'─'*40}")
        print(f"  J2 = {j2:.4f}")
        result = _run_one(lattice, pairs, j2)
        rows.append(result)
        print(
            f"  E_exact = {result['E_exact']:.10f}  "
            f"E_UCJ = {result['E_ucj']:.10f}  "
            f"|ΔE| = {result['abs_error']:.4e}"
        )
        print(
            f"  thetaJ : mean={result['thetaJ_mean']:+.6f}  "
            f"std={result['thetaJ_std']:.6f}"
        )
        print(
            f"  thetaK : mean={result['thetaK_mean']:+.6f}  "
            f"std={result['thetaK_std']:.6f}"
        )

    with OUTPUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n{sep}")
    print(f"  Sweep complete.  CSV saved to: {OUTPUT_CSV.resolve()}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()
