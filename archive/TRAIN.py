import time
import csv
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

nn_edges = [(i, (i + 1) % L, 0) for i in range(L)]   # nearest neighbours
nnn_edges = [(i, (i + 2) % L, 1) for i in range(L)]  # next-nearest neighbours
graph = nk.graph.Graph(edges=nn_edges + nnn_edges)

hi = nk.hilbert.Spin(s=0.5, N=L, total_sz=0.0)


def make_hamiltonian(j2, j1=J1):
    """Nearest + next-nearest neighbour Heisenberg chain for a given J2."""
    return nk.operator.Heisenberg(hilbert=hi, graph=graph, J=[j1, j2])

# --------------------------------------------------------------------------
# 2. FNQS ansatz: a translation-invariant, PATCH-based self-attention network
#    conditioned on the coupling j2.
# --------------------------------------------------------------------------
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
    d_model: int
    heads: int
    L_eff: int

    @nn.compact
    def __call__(self, x):
        v = nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64, kernel_init=nn.initializers.xavier_uniform())(x)
        v = v.reshape(x.shape[0], self.L_eff, self.heads, -1).transpose(0, 2, 1, 3)

        J = self.param("J", nn.initializers.xavier_uniform(),
                        (self.heads, self.L_eff), jnp.float64)
        Jmat = jax.vmap(lambda j, s: jnp.roll(j, s), (None, 0), out_axes=1)(
            J, jnp.arange(self.L_eff)
        )  # (heads, L_eff, L_eff)

        out = jnp.matmul(Jmat, v)                      # (batch, heads, L_eff, d_eff)
        out = out.transpose(0, 2, 1, 3).reshape(x.shape[0], self.L_eff, -1)
        return nn.Dense(self.d_model, param_dtype=jnp.float64, dtype=jnp.float64, kernel_init=nn.initializers.xavier_uniform())(out)


class EncoderBlock(nn.Module):
    d_model: int
    heads: int
    L_eff: int

    @nn.compact
    def __call__(self, x):
        ln1 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        ln2 = nn.LayerNorm(dtype=jnp.float64, param_dtype=jnp.float64)
        x = x + FMHA(self.d_model, self.heads, self.L_eff)(ln1(x))
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
    """psi_theta(sigma | j2), patch-based."""
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
N_ITERS = 1500
LR = 0.02
J2_LOW, J2_HIGH = 0.0, 1.0   # support of P(j2) the model is trained to cover

HARD_REGION_LOW = 0.30
HARD_REGION_FRAC = 0.0


def sample_j2_batch(size):
    is_hard = np.random.rand(size) < HARD_REGION_FRAC
    hard_samples = np.random.uniform(HARD_REGION_LOW, J2_HIGH, size=size)
    full_samples = np.random.uniform(J2_LOW, J2_HIGH, size=size)
    return np.where(is_hard, hard_samples, full_samples)


DIAG_SHIFT_START = 1e-2
DIAG_SHIFT_END = 3e-3
DIAG_SHIFT_DECAY_ITERS = 400


def diag_shift_at(it):
    frac = min(it / DIAG_SHIFT_DECAY_ITERS, 1.0)
    log_shift = (1 - frac) * np.log(DIAG_SHIFT_START) + frac * np.log(DIAG_SHIFT_END)
    return float(np.exp(log_shift))


MAX_UPDATE_NORM = 100.0
REJECT_UPDATE_NORM = 1000.0

# --------------------------------------------------------------------------
# 3b. Fixed-grid evaluation -- the "grounding" signal for convergence
#
# The R systems used for the SR update are resampled from a random j2 draw
# every iteration, so the printed per-iteration energies aren't comparable
# across iterations. run_fixed_grid_eval re-evaluates the *same* small set
# of j2 points (EVAL_J2_GRID) with a much larger sample count, giving a
# stable, low-variance signal to actually judge convergence by.
# --------------------------------------------------------------------------
EVAL_J2_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
EVAL_EVERY = 25
EVAL_SAMPLES = 4096
EVAL_LOG_PATH = "fnqs_eval_log.csv"

eval_sampler = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=EVAL_SAMPLES)

model = FNQS1D()
rng = jax.random.PRNGKey(0)
dummy_spins = hi.random_state(jax.random.PRNGKey(1), 2)
variables = model.init(rng, dummy_spins)


_template_flat, unravel = jax.flatten_util.ravel_pytree(variables["params"])
del _template_flat

@jax.jit
def solve_sr_update(G, Sk_ops, diag_shift):
    def S_ensemble_mv(v_flat):
        v_tree = unravel(v_flat)
        acc = None
        for Sk_op in Sk_ops:
            out_flat, _ = jax.flatten_util.ravel_pytree(Sk_op @ v_tree)
            acc = out_flat if acc is None else acc + out_flat
        return acc / len(Sk_ops) + diag_shift * v_flat
    delta_flat, _ = jax.scipy.sparse.linalg.cg(S_ensemble_mv, G, maxiter=200, tol=1e-6)
    return delta_flat

np.random.seed(0)

sampler = nk.sampler.MetropolisExchange(hi, graph=graph, n_chains=M_PER_SYSTEM)

vstates = [
    nk.vqs.MCState(sampler, model, variables=variables,
                    n_samples=M_PER_SYSTEM, chunk_size=256, n_discard_per_chain=8)
    for _ in range(R)
]

# single dedicated MCState for grounding evals, kept separate from the R
# training vstates so it has its own sampler_state and doesn't disturb
# their chains; reused across EVAL_J2_GRID points and across iterations.
eval_vstate = nk.vqs.MCState(eval_sampler, model, variables=variables,
                              n_samples=EVAL_SAMPLES, chunk_size=256,
                              n_discard_per_chain=16)

eval_log_file = open(EVAL_LOG_PATH, "w", newline="")
eval_log_writer = csv.DictWriter(
    eval_log_file, fieldnames=["iter", "j2", "energy", "energy_error"]
)
eval_log_writer.writeheader()
eval_log_file.flush()


def run_fixed_grid_eval(it, variables):
    for j2_val in EVAL_J2_GRID:
        eval_vstate.variables = {**variables, "coupling": {"j2": jnp.asarray(j2_val, jnp.float64)}}
        H = make_hamiltonian(j2_val)
        e = eval_vstate.expect(H)
        eval_log_writer.writerow({
            "iter": it,
            "j2": j2_val,
            "energy": float(e.mean.real) / L,
            "energy_error": float(e.error_of_mean) / L,
        })
        eval_log_file.flush()


optimizer = optax.sgd(LR)
opt_state = optimizer.init(variables["params"])
import jax.flatten_util


def bin_by_tercile(j2_batch, energies, j2_low, j2_high):
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


def format_hms(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


if __name__ == "__main__":

    training_start = time.perf_counter()
    iter_times = []
    ETA_WINDOW = 50

    for it in range(N_ITERS):
        iter_start = time.perf_counter()

        j2_batch = sample_j2_batch(R)
        diag_shift_it = diag_shift_at(it)

        matvec_fns = []
        G_sum = None
        energies = []

        for k in range(R):
            j2_k = float(j2_batch[k])
            vstates[k].variables = {**variables, "coupling": {"j2": jnp.asarray(j2_k, jnp.float64)}}
            H_k = make_hamiltonian(j2_k)
    
            e_k, G_k = vstates[k].expect_and_grad(H_k)
            G_k_flat, _ = jax.flatten_util.ravel_pytree(G_k)   # unravel from outer scope, don't reassign it
            G_sum = G_k_flat if G_sum is None else G_sum + G_k_flat
            energies.append(e_k.mean.real / L)
    
            Sk_op = nk.optimizer.qgt.QGTJacobianPyTree(vstates[k], diag_shift=0.0)
            matvec_fns.append(Sk_op)
    
        G = G_sum / R
        delta_flat = solve_sr_update(G, tuple(matvec_fns), diag_shift_it)   # <-- one call, jitted, reused
        delta_norm = float(jnp.linalg.norm(delta_flat))


        if delta_norm > REJECT_UPDATE_NORM:
            rejected = True
        else:
            rejected = False
            if delta_norm > MAX_UPDATE_NORM:
                delta_flat = delta_flat * (MAX_UPDATE_NORM / delta_norm)

            delta = {"params": unravel(delta_flat)}
            updates, opt_state = optimizer.update(delta["params"], opt_state, variables["params"])
            new_params = optax.apply_updates(variables["params"], updates)
            variables = {**variables, "params": new_params}

        # -- fixed-grid grounding evaluation --
        # Runs every EVAL_EVERY iterations (and always on the last iteration)
        # using the *post-update* variables, so each CSV row can be read as
        # "energy at this j2 after training step `iter`".
        if it % EVAL_EVERY == 0 or it == N_ITERS - 1:
            run_fixed_grid_eval(it, variables)

        # -- timing bookkeeping --
        iter_elapsed = time.perf_counter() - iter_start
        iter_times.append(iter_elapsed)
        if len(iter_times) > ETA_WINDOW:
            iter_times.pop(0)

        if it % 10 == 0:
            tercile_str = bin_by_tercile(j2_batch, energies, J2_LOW, J2_HIGH)
            if rejected:
                status = " [REJECTED - update skipped]"
            elif delta_norm > MAX_UPDATE_NORM:
                status = " [CLIPPED]"
            else:
                status = ""

            total_elapsed = time.perf_counter() - training_start
            avg_iter_time = sum(iter_times) / len(iter_times)
            remaining_iters = N_ITERS - (it + 1)
            eta_seconds = avg_iter_time * remaining_iters

            grid_str = ""
            if it % EVAL_EVERY == 0 or it == N_ITERS - 1:
                grid_str = f"\n          [grounding eval] logged to {EVAL_LOG_PATH}"

            print(f"iter {it:4d}  diag_shift={diag_shift_it:.2e}  "
                  f"mean e/site={np.mean(energies):+.5f}  "
                  f"|delta|={delta_norm:.2f}{status}\n"
                  f"          {tercile_str}\n"
                  f"          time/iter={iter_elapsed:.2f}s  "
                  f"avg/iter={avg_iter_time:.2f}s  "
                  f"elapsed={format_hms(total_elapsed)}  "
                  f"ETA={format_hms(eta_seconds)}"
                  f"{grid_str}")

    total_training_time = time.perf_counter() - training_start
    print(f"Training finished in {format_hms(total_training_time)} "
          f"({total_training_time:.1f}s total, "
          f"{total_training_time / N_ITERS:.2f}s/iter average).")

    with open("fnqs_variables.pkl", "wb") as f:
        pickle.dump({"variables": variables, "training_seconds": total_training_time}, f)
    print("Saved trained weights to fnqs_variables.pkl")

    eval_log_file.close()
    print(f"Saved fixed-grid grounding log to {EVAL_LOG_PATH}")
