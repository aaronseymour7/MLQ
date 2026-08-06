"""
Sample bitstrings from the trained FNQS model at a fixed J2 and report the
resulting multiset of configurations (bitstring -> count).

This is step 1 of the ground-state-prep loop:
    sample bitstrings -> [build H projected onto sampled support] -> time
    evolve -> sample again -> time evolve again -> ...

Reuses the same model / Hilbert space / graph definitions as the tester
(the run_comparison / evaluate_nqs_at_j2 script), so sampling here is
consistent with what that script does internally.
"""
import pickle
from collections import Counter

import numpy as np
import jax.numpy as jnp
import netket as nk

from train import model, hi, graph, L, J1, J2_LOW, J2_HIGH, make_hamiltonian

VARIABLES_PATH = "fnqs_variables.pkl"
with open(VARIABLES_PATH, "rb") as f:
    loaded = pickle.load(f)

variables = loaded["variables"]
print(f"Loaded trained weights from {VARIABLES_PATH} "
      f"(trained in {loaded.get('training_seconds', float('nan')):.1f}s)")


# --------------------------------------------------------------------------
# 1. Draw raw samples from psi_theta( . | j2) via Metropolis
# --------------------------------------------------------------------------
def sample_configs(j2, variables, n_samples=4096, n_discard=200, seed=0):
    """Metropolis-sample the trained NQS at fixed j2, return raw +-1 configs.

    Returns an array of shape (n_total_samples, L), where n_total_samples =
    n_chains * n_samples_per_chain (netket reshapes internally). Note: since
    this is presumably a total-Sz=0 sector, every row will have equal counts
    of +1/-1 (or 0/1 after remapping below).
    """
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


# --------------------------------------------------------------------------
# 2. Map spin configs (+-1) to bitstrings ('0'/'1'), and count occurrences
# --------------------------------------------------------------------------
def configs_to_bitstrings(configs):
    """netket spins are +-1; map -1 -> '0', +1 -> '1'. Site 0 is leftmost char."""
    bits = ((configs + 1) // 2).astype(int)
    return ["".join(map(str, row)) for row in bits]


def count_bitstrings(j2, variables, n_samples=4096, n_discard=200, seed=0, top_k=200):
    configs = sample_configs(j2, variables, n_samples, n_discard, seed)
    bitstrings = configs_to_bitstrings(configs)
    counts = Counter(bitstrings)

    n_total = len(bitstrings)
    n_unique = len(counts)
    print(
        f"j2={j2:+.4f}: drew {n_total} samples -> {n_unique} unique bitstrings "
        f"({n_unique / n_total:.1%} of draws are distinct configs)"
    )

    print(f"\nTop {min(top_k, n_unique)} most-sampled bitstrings:")
    for bstr, c in counts.most_common(top_k):
        print(f"  {bstr}  count={c:5d}  freq={c / n_total:.4e}")

    return counts  

# --------------------------------------------------------------------------
# 3. Main
# --------------------------------------------------------------------------
if __name__ == "__main__":
    J2_TEST = 0  # pick any j2 in/near the training support
    N_SAMPLES = 40000
    N_DISCARD = 200
    print('starting count')
    counts = count_bitstrings(J2_TEST, variables, n_samples=N_SAMPLES, n_discard=N_DISCARD)

    # Sorted list of (bitstring, count), most frequent first -- convenient
    # for slicing off "top N configs" when you build the projected H.
    sorted_counts = counts.most_common()
