"""
Tester: sample the ensemble-SR-trained FNQS model (see train.py) across a grid
of J2 values, both inside the training support [J2_LOW, J2_HIGH]
(interpolation) and outside it (extrapolation), and report the estimated
ground-state energy per site -- no exact diagonalization, no error metrics.

Usage
-----
1. Run this in the *same process* right after training, so `variables` and
   `model` are already in scope, OR
2. Save `variables` to disk after training (see `save_variables` /
   `load_variables` in the training script) and run this script standalone,
   importing the model definitions from your training script.

This script:
  - builds H(j2) for a grid of test couplings (only needed to define expect())
  - samples the trained NQS at each j2 and estimates energy/site + error bar
    on the *estimator itself* (Monte Carlo std error, not a comparison to ED)
  - saves a CSV of results and a plot of NQS energy vs j2
"""

import numpy as np
import jax
import jax.numpy as jnp
import netket as nk
import pandas as pd

jax.config.update("jax_enable_x64", True)

# --------------------------------------------------------------------------
# Model / Hamiltonian definitions + trained weights.
# --------------------------------------------------------------------------
# We import the *definitions* only (model class, Hilbert space, graph,
# make_hamiltonian, J2_LOW/J2_HIGH) from the training module. As long as the
# training loop in train.py is wrapped in `if __name__ == "__main__":`,
# this import is safe -- it will NOT re-run training.
from train import model, hi, graph, L, J1, J2_LOW, J2_HIGH, make_hamiltonian

# Load the already-trained weights saved by train.py, e.g. via
#   pickle.dump(variables, open("fnqs_variables.pkl", "wb"))
# at the end of the training script (see accompanying note).
import pickle

VARIABLES_PATH = "fnqs_variables.pkl"
with open(VARIABLES_PATH, "rb") as f:
    loaded = pickle.load(f)

variables = loaded["variables"]
print(f"Loaded trained weights from {VARIABLES_PATH} "
      f"(trained in {loaded.get('training_seconds', float('nan')):.1f}s)")

# --------------------------------------------------------------------------
# 1. Sweep: interpolation grid inside [J2_LOW, J2_HIGH] + extrapolation points
# --------------------------------------------------------------------------
def build_test_grid(j2_low, j2_high, n_interp=15, n_extrap_each_side=2, margin=0.15):
    interp = np.linspace(j2_low, j2_high, n_interp)
    extrap_below = np.linspace(j2_low - margin, j2_low - 1e-3, n_extrap_each_side)
    extrap_above = np.linspace(j2_high + 1e-3, j2_high + margin, n_extrap_each_side)
    grid = np.concatenate([extrap_below, interp, extrap_above])
    return np.round(grid, 4)


def run_energy_scan(j2_grid, variables, n_samples=4096, n_discard=16, verbose=True):
    """Sample the trained NQS at each j2 in the grid and record energy/site.

    No exact diagonalization is performed -- this only reports what the
    model itself thinks the energy is, plus the Monte Carlo error bar and
    relative variance of the local-energy estimator (a proxy for how
    well-converged/well-generalized the ansatz is at that j2).
    """
    rows = []
    sampler_eval = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=n_samples)
    vs = nk.vqs.MCState(
        sampler_eval, model,
        variables={**variables, "coupling": {"j2": jnp.asarray(j2_grid[0], jnp.float64)}},
        n_samples=n_samples,
        n_discard_per_chain=1000,   # long cold-start burn-in, once
    )

    for j2 in j2_grid:
        # warm-start: reuse the *previous* j2's chain state, just swap coupling
        vs.variables = {**variables, "coupling": {"j2": jnp.asarray(j2, jnp.float64)}}
        vs.sample(n_discard_per_chain=200)   # short re-equilibration between nearby j2's

        stats = vs.expect(make_hamiltonian(j2))
        e_nqs = float(stats.mean.real) / L
        e_err = float(stats.error_of_mean) / L
        rel_var = float(stats.variance) / (L * abs(e_nqs) + 1e-12) ** 2

        in_support = J2_LOW <= j2 <= J2_HIGH

        rows.append(
            dict(
                j2=j2,
                in_support=in_support,
                e_nqs=e_nqs,
                e_nqs_err=e_err,
                nqs_rel_variance=rel_var,
            )
        )

        if verbose:
            tag = "interp" if in_support else "EXTRAP"
            print(
                f"j2={j2:+.4f} [{tag}]  "
                f"nqs={e_nqs:.6f}+-{e_err:.6f}  "
                f"rel_var={rel_var:.2e}"
            )

    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 2. Plot NQS energy across j2
# --------------------------------------------------------------------------
def plot_energy_scan(df, outpath="fnqs_energy_scan.png"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))

    ax.axvspan(J2_LOW, J2_HIGH, color="tab:blue", alpha=0.08, label="training support")
    ax.errorbar(
        df["j2"], df["e_nqs"], yerr=df["e_nqs_err"],
        fmt="o-", color="tab:red", ms=5, capsize=3, label="FNQS (ensemble-SR)",
    )
    ax.set_xlabel("J2 / J1")
    ax.set_ylabel("Energy / site")
    ax.set_title(f"FNQS energy scan, N={L} J1-J2 Heisenberg chain")
    ax.legend()

    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")


# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    N_SAMPLES = 4096       # increase for tighter NQS error bars
    N_DISCARD = 16

    j2_grid = build_test_grid(J2_LOW, J2_HIGH, n_interp=9, n_extrap_each_side=2)

    # Always include the well-known frustration/dimerization point of the
    # N->inf J1-J2 chain (~0.2411) if it's not already in the grid.
    landmark = 0.50
    if not np.any(np.isclose(j2_grid, landmark, atol=1e-2)):
        j2_grid = np.sort(np.append(j2_grid, landmark))

    df = run_energy_scan(j2_grid, variables, n_samples=N_SAMPLES, n_discard=N_DISCARD)

    df.to_csv("fnqs_energy_scan_results.csv", index=False)
    print("\nSaved results to fnqs_energy_scan_results.csv")
    print(df.to_string(index=False))

    try:
        plot_energy_scan(df)
    except ImportError:
        print("matplotlib not installed -- skipping plot, CSV results still saved.")
