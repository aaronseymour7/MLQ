import netket as nk
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import math

# ============================================================
# Generic VMC runner
# ============================================================

def run_vmc_case(
    n,
    h_builder,
    reference_energy=None,
    state="ground",
    model_alpha=6,
    n_samples=4096,
    n_iter=600,
    learning_rate=0.01,
    diag_shift=0.05,
    param_dtype=complex,
    h_builder_kwargs=None,
):
    """
    Generic VMC workflow.

    Parameters
    ----------
    n : int
        System size.

    h_builder : callable
        Function like:
            ha, hi, graph = h_builder(n, **kwargs)

    state : str
        One of:
            "ground"
            "highest"
            "first_excited"

    h_builder_kwargs : dict
        Extra kwargs passed into h_builder.
    """

    if h_builder_kwargs is None:
        h_builder_kwargs = {}

    print(f"--- Running {state} state for N={n} ---")

    # --------------------------------------------------------
    # Build Hamiltonian
    # --------------------------------------------------------

    ha, hi, graph = h_builder(n, **h_builder_kwargs)



    # --------------------------------------------------------
    # Model / sampler
    # --------------------------------------------------------

    model = nk.models.RBM(alpha=model_alpha, param_dtype=param_dtype)
    sampler = nk.sampler.MetropolisExchange(hi, graph=graph)

    vstate = nk.vqs.MCState(
        sampler,
        model,
        n_samples=n_samples,
    )

    # --------------------------------------------------------
    # Initial states
    # --------------------------------------------------------

    n_chains = vstate.sampler.n_chains

    if state == "highest":

        ferro = np.ones(n)
        init_samples = np.tile(ferro, (n_chains, 1)).astype(np.int8)

    elif state == "first_excited":

        neel = np.ones(n)
        neel[2::2] = -1
        init_samples = np.tile(neel, (n_chains, 1)).astype(np.int8)

    else:

        neel = np.ones(n)
        neel[1::2] = -1
        init_samples = np.tile(neel, (n_chains, 1)).astype(np.int8)

    vstate.sampler_state = vstate.sampler_state.replace(
        σ=jnp.array(init_samples)
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    optimizer = nk.optimizer.Sgd(learning_rate=learning_rate)

    vmc_hamiltonian = -ha if state == "highest" else ha

    vmc = nk.driver.VMC_SR(
        vmc_hamiltonian,
        optimizer,
        diag_shift=diag_shift,
        variational_state=vstate,
    )

    vmc.run(n_iter=n_iter)

    # --------------------------------------------------------
    # Measurements
    # --------------------------------------------------------

    energy_stats = vstate.expect(vmc_hamiltonian)

    if state == "highest":
        e_vmc = -float(energy_stats.mean.real)
    else:
        e_vmc = float(energy_stats.mean.real)

    e_error = float(energy_stats.error_of_mean)


    return {
        "N": n,
        "VMC": e_vmc,
        "VMC_err": e_error,
    }


# ============================================================
# Batch wrapper
# ============================================================

def get_data(
    n_sites_list,
    h_builder,
    state="ground",
    h_builder_kwargs=None,
    **vmc_kwargs,
):
    """
    Run many system sizes.
    """

    results = []

    for n in n_sites_list:

        result = run_vmc_case(
            n=n,
            h_builder=h_builder,
            state=state,
            h_builder_kwargs=h_builder_kwargs,
            **vmc_kwargs,
        )

        results.append(result)

    return results


# ============================================================
# Printer
# ============================================================

def printer(data):

    for d in data:

        print(
            f"N={d['N']} | "
            f"VMC Energy: {d['VMC']:.8f} | "
            f"MC Error: {d['VMC_err']:.2e}"
        )


# ============================================================
# Hamiltonian builder
# ============================================================

def build_j1j2(n, j1, j2, pbc=False, total_sz=0):

    Sp = np.array([[0, 1], [0, 0]], dtype=complex)
    Sm = np.array([[0, 0], [1, 0]], dtype=complex)
    Sz = np.array([[0.5, 0], [0, -0.5]], dtype=complex)

    h2 = (
        np.kron(Sz, Sz)
        + 0.5 * np.kron(Sp, Sm)
        + 0.5 * np.kron(Sm, Sp)
    )


    hi = nk.hilbert.Spin(
        s=0.5,
        N=n,
        total_sz=total_sz,
    )

    graph = nk.graph.Chain(n, pbc=pbc)

    nn_e = (
        [(i, (i + 1) % n) for i in range(n)]
        if pbc
        else [(i, i + 1) for i in range(n - 1)]
    )

    nnn_e = (
        [(i, (i + 2) % n) for i in range(n)]
        if pbc
        else [(i, i + 2) for i in range(n - 2)]
    )

    ha = nk.operator.LocalOperator(hi, dtype=complex)

    for si, sj in nn_e:
        ha += j1 * nk.operator.LocalOperator(
            hi,
            h2,
            acting_on=[si, sj],
        )

    for si, sj in nnn_e:
        ha += j2 * nk.operator.LocalOperator(
            hi,
            h2,
            acting_on=[si, sj],
        )

    return ha, hi, graph


def main(n_sizes, J1, J2, K=5):
    """
    Runs K VMC trials per (N, J2) combination and records the lowest energy per level.
    
    Parameters
    ----------
    K : int
        Number of VMC trials to run per (N, J2, state) combination.
    """

    states = ["ground", "first_excited", "highest"]

    for n in n_sizes:
        for j2 in J2:
            print(f"\n{'='*60}")
            print(f"N={n}, J2={j2} | Running {K} trials per state")
            print(f"{'='*60}")

            for state in states:
                best_result = None
                if state == "ground":
                    sz = 0
                elif state == "first_excited":
                    sz = 1
                elif state == "highest":
                    sz = 0
                for k in range(K):
                    print(f"\n  [{state}] Trial {k+1}/{K}")
                    result = run_vmc_case(
                        n=n,
                        h_builder=build_j1j2,
                        state=state,
                        reference_energy=None,
                        h_builder_kwargs={
                            "j1": J1,
                            "j2": j2,
                            "pbc": False,
                            "total_sz": sz,
                        }
                    )
                    result["trial"] = k + 1



                    # For "highest", higher energy = better; for others, lower = better
                    if math.isnan(result["VMC"]):
                        print(f"      Trial {k+1} returned NaN — skipping.")
                        continue

                    if best_result is None:
                        best_result = result
                    else:
                        if state == "highest":
                            if result["VMC"] > best_result["VMC"]:
                                best_result = result
                        else:
                            if result["VMC"] < best_result["VMC"]:
                                best_result = result
                if best_result is None:
                    print(f"\n  >>> All {K} trials returned NaN for [{state}] — no valid result.")
                else:
                    print(f"\n  >>> Best result for [{state}] over {K} trials:")
                    print(
                        f"      Trial {best_result['trial']} | "
                        f"N={best_result['N']} | "
                        f"VMC: {best_result['VMC']:.8f} | "
                        f"Lanczos: {best_result['Lanczos']:.8f} | "
                        f"Abs Difference: {best_result['diff']:.8f}"
                    )



if __name__ == "__main__":
        
    n_sizes = list(range(10, 101, 10))
    J1 = 1.0
    J2 = [0.0]
    K = 1
    main(n_sizes, J1, J2, K=K)
