"""
Exact |<v|psi(.|j2)>|^2 via full Sz=0 sector enumeration, plus a comparison
against the Metropolis-sampled empirical frequencies for the same j2.

IMPORTANT: this must be run with train.py's L matching the L that
fnqs_variables.pkl was trained at (the model's patch reshape is fixed to
that L at construction time) -- e.g. for the L=20 checkpoint, edit train.py
so `L = 20` before running this script.
"""
import itertools
import pickle
from math import comb, log
from collections import Counter
import csv
import numpy as np
import jax
import jax.numpy as jnp
import netket as nk

from train import model, hi, graph, L, make_hamiltonian

VARIABLES_PATH = "fnqs_variables.pkl"


# --------------------------------------------------------------------------
# 1. Exact enumeration + exact |psi(v)|^2 / Z
# --------------------------------------------------------------------------
def enumerate_sz0_states(L):
    """All +-1 configs with equal #up/#down. Shape (C(L, L/2), L)."""
    half = L // 2
    n_total = comb(L, half)
    states = np.ones((n_total, L), dtype=np.float64)
    for i, down_positions in enumerate(itertools.combinations(range(L), half)):
        states[i, list(down_positions)] = -1.0
    return states


_apply_jit = jax.jit(lambda variables, batch: model.apply(variables, batch))


def exact_probs(variables, j2, states, batch_size=8192):
    """Exact |psi(v)|^2 / Z for every v in `states`. No MC noise."""
    variables_j2 = {**variables, "coupling": {"j2": jnp.asarray(j2, jnp.float64)}}
    n = states.shape[0]
    logpsi = np.empty(n, dtype=np.complex128)

    for start in range(0, n, batch_size):
        batch = jnp.asarray(states[start:start + batch_size])
        out = _apply_jit(variables_j2, batch)
        logpsi[start:start + batch_size] = np.asarray(out)

    log_p_unnorm = 2.0 * logpsi.real                      # log|psi|^2
    m = log_p_unnorm.max()
    logZ = m + np.log(np.sum(np.exp(log_p_unnorm - m)))   # log-sum-exp, stable
    probs = np.exp(log_p_unnorm - logZ)
    return probs, logpsi


def states_to_bitstrings(states):
    bits = ((states + 1) // 2).astype(int)
    return np.array(["".join(map(str, row)) for row in bits])


# --------------------------------------------------------------------------
# 2. MC sampling (reused from your sampler script)
# --------------------------------------------------------------------------
def sample_configs(j2, variables, n_samples=4096, n_discard=200, seed=0):
    sampler = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=16)
    vs = nk.vqs.MCState(
        sampler,
        model,
        variables={**variables, "coupling": {"j2": jnp.asarray(j2, jnp.float64)}},
        n_samples=n_samples,
        n_discard_per_chain=n_discard,
        seed=seed,
    )
    configs = np.asarray(vs.samples).reshape(-1, L)
    return configs


def configs_to_bitstrings(configs):
    bits = ((configs + 1) // 2).astype(int)
    return ["".join(map(str, row)) for row in bits]

def sampled_mass_coverage(exact_bitstrings, exact_p, counts):
    exact_lookup = dict(zip(exact_bitstrings, exact_p))
    sampled_mass = sum(exact_lookup[b] for b in counts.keys())
    print(f"Unique bitstrings sampled: {len(counts):,} / {len(exact_bitstrings):,} "
          f"({len(counts)/len(exact_bitstrings):.2%} of sector)")
    print(f"Exact probability mass covered by sampled support: {sampled_mass:.6f}")
    print(f"Missing mass (never sampled): {1 - sampled_mass:.6f}")
    return sampled_mass
# --------------------------------------------------------------------------
# 3. Compare exact probabilities vs. MC empirical frequencies
# --------------------------------------------------------------------------
def compare_exact_vs_sampled(j2, variables, n_samples=40000, n_discard=200,
                              top_k=200, seed=0):
    half = L // 2
    n_sector = comb(L, half)
    print(f"L={L}, Sz=0 sector size = {n_sector:,}")

    # exact
    states = enumerate_sz0_states(L)
    exact_p, _ = exact_probs(variables, j2, states)
    exact_bitstrings = states_to_bitstrings(states)
    exact_lookup = dict(zip(exact_bitstrings, exact_p))
    
    # sampled
    configs = sample_configs(j2, variables, n_samples=n_samples,
                              n_discard=n_discard, seed=seed)
    sampled_bitstrings = configs_to_bitstrings(configs)
    counts = Counter(sampled_bitstrings)
    n_total = len(sampled_bitstrings)
    print(f"j2={j2:+.4f}: drew {n_total} samples -> {len(counts)} unique "
          f"({len(counts)/n_total:.1%} of draws distinct)")
    coverage = sampled_mass_coverage(exact_bitstrings, exact_p, counts)
    # exact ranking, top_k by true probability
    order = np.argsort(-exact_p)[:top_k]
    print(f"Coverage: {coverage}")
    print(f"\nTop {top_k} bitstrings by EXACT probability "
          f"(exact_prob | sampled_count | sampled_freq):")
    for idx in order:
        b = exact_bitstrings[idx]
        p_exact = exact_p[idx]
        c = counts.get(b, 0)
        freq = c / n_total
        print(f"  {b}  exact={p_exact:.4e}  count={c:5d}  freq={freq:.4e}")

    # KL(empirical || exact), restricted to the sampled support --
    # a concrete number for the MC binning error you were worried about
    kl = 0.0
    for b, c in counts.items():
        p_hat = c / n_total
        p_exact = exact_lookup.get(b, 1e-300)   # shouldn't be missing, but guard
        kl += p_hat * (log(p_hat) - log(p_exact))
    print(f"\nKL(empirical || exact) over sampled support = {kl:.4e}")
    save_exact_vs_sampled_csv(exact_bitstrings, exact_p, counts, n_total,out_path=f"exact_vs_sampled_L{L}_j2{j2:.2f}.csv")
    return exact_p, exact_bitstrings, counts
def save_exact_vs_sampled_csv(exact_bitstrings, exact_p, counts, n_total,
                               out_path="exact_vs_sampled.csv",
                               include_unsampled=False):
    """
    Write one row per bitstring: bitstring, exact_prob, sampled_count,
    sampled_freq. By default only writes rows for bitstrings that were
    actually drawn by the sampler (include_unsampled=False) -- that's
    the natural set for a freq-vs-projection scatter plot, since points
    for zero-count bitstrings would just pile up on the freq=0 axis.
    Set include_unsampled=True to dump the full exact distribution too
    (useful if you want to see how much exact-probability mass sits on
    states the sampler never visited).
    """
    exact_lookup = dict(zip(exact_bitstrings, exact_p))

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["bitstring", "exact_prob", "sampled_count", "sampled_freq"])

        if include_unsampled:
            # iterate the full exact enumeration
            for b, p_exact in zip(exact_bitstrings, exact_p):
                c = counts.get(b, 0)
                writer.writerow([b, f"{p_exact:.10e}", c, f"{c / n_total:.10e}"])
        else:
            # iterate only what the sampler drew
            for b, c in counts.items():
                p_exact = exact_lookup.get(b, float("nan"))
                writer.writerow([b, f"{p_exact:.10e}", c, f"{c / n_total:.10e}"])

    print(f"Saved {out_path} "
          f"({'full sector' if include_unsampled else 'sampled bitstrings only'})")

if __name__ == "__main__":
    with open(VARIABLES_PATH, "rb") as f:
        loaded = pickle.load(f)
    variables = loaded["variables"]
    print(f"Loaded trained weights from {VARIABLES_PATH} "
          f"(trained in {loaded.get('training_seconds', float('nan')):.1f}s)")

    assert hi.size == L, (
        f"train.py's L={L} but this script assumes it matches the loaded "
        f"checkpoint's training L -- edit train.py's L to match before running."
    )

    J2_TEST = 0.0
    compare_exact_vs_sampled(J2_TEST, variables, n_samples=190000, n_discard=200)
