

import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import netket as nk
import pickle

jax.config.update("jax_enable_x64", True)
print("JAX devices:", jax.devices())

# --------------------------------------------------------------------------
# 1. Lattice, Hilbert space, and the family of Hamiltonians H(j2)
# --------------------------------------------------------------------------
L = 20
J1 = 1.0

nn_edges = [(i, (i + 1) % L, 0) for i in range(L)]   # nearest neighbors
nnn_edges = [(i, (i + 2) % L, 1) for i in range(L)]  # next-nearest neighbors
graph = nk.graph.Graph(edges=nn_edges + nnn_edges)

hi = nk.hilbert.Spin(s=0.5, N=L, total_sz=0.0)


def make_hamiltonian(j2, j1=J1):
    """Nearest + next-nearest neighbor Heisenberg chain for a given J2."""
    return nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[j1, j2])

# --------------------------------------------------------------------------
# 2. FNQS ansatz: a translation-invariant, PATCH-based self-attention network
#    conditioned on the coupling J2

def log_cosh(x):
    sgn = -2 * jnp.signbit(x.real) + 1
    x = x * sgn
    return x + jnp.log1p(jnp.exp(-2.0 * x)) - jnp.log(2.0)


def extract_patches_1d(x, b):
    """(batch, L) -> (batch, L_eff, b), contiguous non-overlapping patches."""
    batch, L = x.shape
    L_eff = L // b
    return x.reshape(batch, L_eff, b)


class FMHA(nn.Module):
    """Translation-invariant factored multi-head self-attention: the
    attention matrix is a learned circulant matrix over the L_eff PATCH
    tokens (not over raw sites), so it does not depend on the samples
    themselves. Structurally identical to the paper's FMHA (attentions.py),
    just written with reshape/transpose instead of einops."""
    d_model: int
    heads: int
    L_eff: int

    @nn.compact
    def __call__(self, x):
        v = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64)(x)
        v = v.reshape(x.shape[0], self.L_eff, self.heads, -1).transpose(0, 2, 1, 3)

        J = self.param("J", nn.initializers.xavier_uniform(),
                        (self.heads, self.L_eff), jnp.float64)
        Jmat = jax.vmap(lambda j, s: jnp.roll(j, s), (None, 0), out_axes=1)(
            J, jnp.arange(self.L_eff)
        )  # (heads, L_eff, L_eff)

        out = jnp.matmul(Jmat, v)                      # (batch, heads, L_eff, d_eff)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], self.L_eff, -1)
        return nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64)(out)


class EncoderBlock(nn.Module):
    d_model: int
    heads: int
    L_eff: int

    @nn.compact
    def __call__(self, x):
        ln1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        ln2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        x = x + FMHA(self.d_model, self.heads, self.L_eff)(ln1(x))
        # feedforward width/activation matched to the paper (2x, ReLU)
        # rather than the earlier 4x/GELU choice.
        ff = nn.Sequential([
            nn.Dense(2 * self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform()),
            nn.relu,
            nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform()),
        ])
        x = x + ff(ln2(x))
        return x


class FNQS1D(nn.Module):

    d_model: int = 48
    heads: int = 8
    num_layers: int = 4
    b: int = 2          # patch size; L=20, b=2 -> L_eff=10
    L: int = L

    @nn.compact
    def __call__(self, spins):
        j2 = self.variable("coupling", "j2", lambda: jnp.zeros((), jnp.float64)).value

        x = jnp.atleast_2d(spins)                                    # (batch, L)
        L_eff = self.L // self.b
        x = extract_patches_1d(x, self.b)                            # (batch, L_eff, b)

        j2 = jnp.broadcast_to(jnp.asarray(j2, dtype=jnp.float64), (x.shape[0],))
        j2_feat = jnp.broadcast_to(j2[:, None, None], (x.shape[0], L_eff, 1))
        x = jnp.concatenate([x, j2_feat], axis=-1)                   # (batch, L_eff, b+1)

        x = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform())(x)

        for _ in range(self.num_layers):
            x = EncoderBlock(self.d_model, self.heads, L_eff)(x)

        z = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)(x.sum(axis=1))
        amp = nn.LayerNorm(use_scale=True, use_bias=True,
                            dtype=jnp.float64, param_dtype=jnp.float64)(
            nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform())(z))
        sign = nn.LayerNorm(use_scale=True, use_bias=True,
                             dtype=jnp.float64, param_dtype=jnp.float64)(
            nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64,
                      kernel_init=nn.initializers.xavier_uniform())(z))
        out = amp + 1j * sign
        return jnp.sum(log_cosh(out), axis=-1)


# --------------------------------------------------------------------------
# 3. Ensemble-SR training loop (Eqs. 15-18)
# --------------------------------------------------------------------------
R = 9                        # number of couplings sampled per SR step ("systems")
M_PER_SYSTEM = 512           # MC samples per system  ->  total M = R * M_PER_SYSTEM
N_ITERS = 1400
LR = 0.02
J2_LOW, J2_HIGH = 0.0, 0.6   # support of P(j2) the model is trained to cover


HARD_REGION_LOW = 0.30
HARD_REGION_FRAC = 0.0


def sample_j2_batch(size):
    is_hard = np.random.rand(size) < HARD_REGION_FRAC
    hard_samples = np.random.uniform(HARD_REGION_LOW, J2_HIGH, size=size)
    full_samples = np.random.uniform(J2_LOW, J2_HIGH, size=size)
    return np.where(is_hard, hard_samples, full_samples)


# -- diag-shift annealing schedule --
# Starts high (strong regularization while the ensemble-averaged S is still
# poorly conditioned / far from any good basin) and decays toward a floor.

DIAG_SHIFT_START = 1e-2
DIAG_SHIFT_END = 3e-3
DIAG_SHIFT_DECAY_ITERS = 400   # exponential decay reaches ~DIAG_SHIFT_END by this iter


def diag_shift_at(it):
    frac = min(it / DIAG_SHIFT_DECAY_ITERS, 1.0)
    # exponential interpolation in log-space between START and END
    log_shift = (1 - frac) * np.log(DIAG_SHIFT_START) + frac * np.log(DIAG_SHIFT_END)
    return float(np.exp(log_shift))


# -- SR update clipping / rejection --
# Two-tier defense against a bad S_reg solve:
#   1. REJECT_UPDATE_NORM: if the *raw* (pre-clip) delta norm exceeds this,
#      S_reg was essentially singular at this step and the direction itself
#      is garbage, not just "large but legitimate". In that case we SKIP the
#      update entirely this iteration (params unchanged) rather than
#      rescaling a garbage direction and applying it anyway -- rescaling
#      doesn't fix a bad direction, and applying it is what corrupted
#      training in the iter-370/iter-790 blowups above.
#   2. MAX_UPDATE_NORM: for deltas below the reject threshold but still
#      larger than typical, clip (rescale) as before -- this handles
#      legitimately large-but-usable SR steps, mostly early in training.
MAX_UPDATE_NORM = 100.0
REJECT_UPDATE_NORM = 1000.0

model = FNQS1D()
rng = jax.random.PRNGKey(0)
dummy_spins = hi.random_state(jax.random.PRNGKey(1), 2)
variables = model.init(rng, dummy_spins)

np.random.seed(0)

sampler = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=M_PER_SYSTEM)

# one MCState per "system" (= one sampled j2), all sharing the same `params`;
# only the "coupling" collection differs from system to system.
vstates = [
    nk.vqs.MCState(sampler, model, variables=variables,
                    n_samples=M_PER_SYSTEM, chunk_size = 256, n_discard_per_chain=8)
    for _ in range(R)
]

optimizer = optax.sgd(LR)
opt_state = optimizer.init(variables["params"])
import jax.flatten_util


def bin_by_tercile(j2_batch, energies, j2_low, j2_high):
    """Free diagnostic: split this iteration's already-computed energies
    into low/mid/high thirds of [j2_low, j2_high] and report per-bin means.
    No extra sampling -- reuses what expect_and_grad already computed."""
    edges = np.linspace(j2_low, j2_high, 4)  # 3 bins
    labels = ["low ", "mid ", "high"]
    out = []
    for b in range(3):
        mask = (j2_batch >= edges[b]) & (j2_batch <= edges[b + 1] + 1e-9)
        if mask.any():
            vals = np.asarray(energies)[mask]
            out.append(f"{labels[b]}[{edges[b]:.2f}-{edges[b+1]:.2f}]="
                       f"{vals.mean():+.4f}(n={mask.sum()})")
        else:
            out.append(f"{labels[b]}[{edges[b]:.2f}-{edges[b+1]:.2f}]=  n/a")
    return "  ".join(out)


if __name__ == "__main__":

    for it in range(N_ITERS):
        j2_batch = sample_j2_batch(R)
        diag_shift_it = diag_shift_at(it)

        matvec_fns = []
        G_sum = None
        energies = []
        unravel = None
        
        for k in range(R):
            j2_k = float(j2_batch[k])
            vstates[k].variables = {**variables, "coupling": {"j2": jnp.asarray(j2_k, jnp.float64)}}
            H_k = make_hamiltonian(j2_k)
        
            e_k, G_k = vstates[k].expect_and_grad(H_k)
            G_k_flat, unravel = jax.flatten_util.ravel_pytree(G_k)
            G_sum = G_k_flat if G_sum is None else G_sum + G_k_flat
            energies.append(e_k.mean.real / L)
        
            Sk_op = nk.optimizer.qgt.QGTJacobianPyTree(vstates[k], diag_shift=0.0)
            matvec_fns.append(Sk_op)  # keep the operator itself, not a dense array
        
        G = G_sum / R
        
        def S_ensemble_mv(v_flat):
            v_tree = unravel(v_flat)
            acc = None
            for Sk_op in matvec_fns:
                out_flat, _ = jax.flatten_util.ravel_pytree(Sk_op @ v_tree)
                acc = out_flat if acc is None else acc + out_flat
            return acc / R + diag_shift_it * v_flat
        
        delta_flat, _ = jax.scipy.sparse.linalg.cg(S_ensemble_mv, G, maxiter=200, tol=1e-6)
        delta_norm = float(jnp.linalg.norm(delta_flat))

        if delta_norm > REJECT_UPDATE_NORM:
            # S_reg was essentially singular this step -- the direction is
            # garbage, not just large. Skip the update entirely rather than
            # rescaling and applying a corrupting step.
            rejected = True
        else:
            rejected = False
            if delta_norm > MAX_UPDATE_NORM:
                delta_flat = delta_flat * (MAX_UPDATE_NORM / delta_norm)

            delta = {"params": unravel(delta_flat)}
            updates, opt_state = optimizer.update(delta["params"], opt_state, variables["params"])
            new_params = optax.apply_updates(variables["params"], updates)
            variables = {**variables, "params": new_params}

        if it % 10 == 0:
            tercile_str = bin_by_tercile(j2_batch, energies, J2_LOW, J2_HIGH)
            if rejected:
                status = " [REJECTED - update skipped]"
            elif delta_norm > MAX_UPDATE_NORM:
                status = " [CLIPPED]"
            else:
                status = ""
            print(f"iter {it:4d}  diag_shift={diag_shift_it:.2e}  "
                  f"mean e/site={np.mean(energies):+.5f}  "
                  f"|delta|={delta_norm:.2f}{status}\n"
                  f"          {tercile_str}")

    with open("fnqs_variables.pkl", "wb") as f:
        pickle.dump(variables, f)
    print("Saved trained weights to fnqs_variables.pkl")
