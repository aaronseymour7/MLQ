"""
Unitary Cluster Jastrow (uCJ) ansatz for the 1D Heisenberg spin chain.

Adapted from the k-fold uCJ framework of

    N. V. Tkachenko, H. Ren, W. M. Billings, R. Tomann, K. B. Whaley,
    M. Head-Gordon, "Beyond real: alternative unitary cluster Jastrow
    models for molecular electronic structure calculations on near-term
    quantum computers", Chem. Sci. (2025); arXiv:2505.10963.

The original paper defines, for k=1,

    |Psi> = exp(-K) exp(J) exp(K) |HF>

with K anti-Hermitian (orbital rotation generator) and J purely
imaginary & symmetric (density-density Jastrow generator), acting on
molecular spin-orbitals. Three variants are defined by restricting K:

    Re-uCJ : K real            -> real Givens rotations   (1 angle/pair)
    Im-uCJ : K purely imaginary-> "XY" hopping rotations   (1 angle/pair)
    g-uCJ  : K fully complex   -> generalized Givens       (2 angles/pair)

Here the same operator structure is reused with lattice sites of the
S=1/2 Heisenberg chain standing in for spin-orbitals (occupation
n_i = (1 - Z_i)/2 plays the role of the fermionic number operator; the
XX+YY exchange in the Heisenberg Hamiltonian conserves this occupation,
exactly as particle number is conserved in the molecular problem).
Because the Jastrow term J_pq n_p n_q is diagonal, exp(i J_pq n_p n_q)
is implemented *exactly* by a single ControlledPhaseShift gate; no
Trotterization is required for J. The K rotations are likewise applied
exactly (no Trotter error) as single two-qubit gates per pair, following
the same "no-Trotter" spirit of the original paper's Givens-rotation
circuits (there implemented via the Kivlichan et al. decomposition into
IsingXX/IsingYY (+Rz) gates; PennyLane's SingleExcitation/IsingXY gates
realize the same unitaries directly).

Usage
-----

Ground state    -> optimized within the Sz_tot = 0 sector.
Excited state   -> optimized within the Sz_tot = +1 sector (the lowest
                   state there is degenerate with the true first excited
                   triplet, so no explicit orthogonalization is needed).
Highest state   -> found by minimizing <-H> within the Sz_tot = 0 sector
                   (i.e. the top of that sector's spectrum.
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from scipy.optimize import minimize
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import eigsh
import jax
import jax.numpy as jnp
import netket as nk
from netket.operator.spin import sigmap, sigmam, sigmaz
import time
# =====================================================================
# 1. Hamiltonian: PennyLane operator (for the VQE cost) + exact
#    diagonalization in a fixed-magnetization sector (for benchmarking)
# =====================================================================


def heisenberg_bonds(N: int):
    """Nearest-neighbour bonds of the PBC chain."""
    return [(i, (i + 1) % N) for i in range(N)]


def heisenberg_hamiltonian(N: int, negate: bool = False) -> qml.Hamiltonian:
    """J=1 antiferromagnetic Heisenberg chain, S_i = sigma_i/2, PBC."""
    coeffs, ops = [], []
    sign = -1.0 if negate else 1.0
    for (i, j) in heisenberg_bonds(N):
        coeffs += [0.25 * sign] * 3
        ops += [
            qml.PauliX(i) @ qml.PauliX(j),
            qml.PauliY(i) @ qml.PauliY(j),
            qml.PauliZ(i) @ qml.PauliZ(j),
        ]
    return qml.Hamiltonian(coeffs, ops)


def build_sector_hamiltonian(N: int, n_up: int, negate: bool = False):
    """Sparse H (or -H, if negate) restricted to the fixed n_up
    (i.e. fixed Sz_tot) sector.

    Bit convention: bit=1 means spin-down at that site (n_up counts
    zero-bits). Returns (H_csr, basis_list).
    """
    sign = -1.0 if negate else 1.0
    basis = [s for s in range(2 ** N) if bin(s).count("1") == (N - n_up)]
    index = {s: i for i, s in enumerate(basis)}
    dim = len(basis)
    H = lil_matrix((dim, dim))
    bonds = heisenberg_bonds(N)
    for i, s in enumerate(basis):
        diag = 0.0
        for (p, q) in bonds:
            bp = (s >> p) & 1  # 1 = down
            bq = (s >> q) & 1
            sz_p = -0.5 if bp else 0.5
            sz_q = -0.5 if bq else 0.5
            diag += sz_p * sz_q
            if bp != bq:  # XX+YY flips a pair of unlike neighbours
                s2 = s ^ (1 << p) ^ (1 << q)
                H[index[s2], i] += 0.5 * sign
        H[i, i] += diag * sign
    return H.tocsr(), basis


def exact_low_energies(N: int, n_up: int, k: int = 3, negate: bool = False) -> np.ndarray:
    """Lowest k eigenvalues of H (or of -H, if negate) in the given
    sector. Used both for the ground/excited-state targets and, with
    negate=True, to get the *top* of the spectrum for the 'highest'
    target -- note this is diagonalizing -H directly, not just flipping
    the sign of the ground-state energy, since the in-sector spectrum
    of the isotropic Heisenberg model is not symmetric about zero.
    """
    H, _ = build_sector_hamiltonian(N, n_up, negate=negate)
    dim = H.shape[0]
    if dim <= k + 1:
        return np.sort(np.linalg.eigvalsh(H.toarray()))[:k]
    vals = eigsh(H, k=min(k, dim - 1), which="SA", return_eigenvectors=False)
    return np.sort(vals)


# =====================================================================
# 2. Reference (initial) product states
# =====================================================================


def neel_pattern(N: int) -> list[int]:
    """Up-down-up-down..., Sz_tot = 0. 1 = apply X (down spin)."""
    return [i % 2 for i in range(N)]


def triplet_pattern(N: int) -> list[int]:
    """up-up-down repeating unit cell -> Sz_tot = +1 (for N % 3 == 0)."""
    if N % 3 != 0:
        # fall back: Neel plus one flipped spin -> Sz_tot = +1
        pat = neel_pattern(N)
        pat[pat.index(1)] = 0
        return pat
    unit = [0, 0, 1]
    return unit * (N // 3)


def prepare_reference(wires, pattern):
    for w, b in zip(wires, pattern):
        if b:
            qml.PauliX(wires=w)


# =====================================================================
# 3. uCJ ansatz circuit
# =====================================================================


def all_pairs(N: int):
    return list(itertools.combinations(range(N), 2))


def n_pair(N: int) -> int:
    return N * (N - 1) // 2


def params_per_layer(N: int, variant: str) -> int:
    npair = n_pair(N)
    if variant == "g":
        return 3 * npair  # theta, phi, J
    return 2 * npair  # theta, J


def apply_generalized_givens(theta, phi, wires):
    p, q = wires
    qml.RZ(phi, wires=p)
    qml.IsingXY(-2.0 * theta, wires=[p, q])
    qml.RZ(-phi, wires=q)


def apply_K(theta, phi, wires, variant: str, sign: float, pairs, pair_index):
    """Apply exp(sign * K) as a product of pairwise rotations over ALL
    n_pair = N(N-1)/2 site pairs (matching the paper's parameter count),
    exactly (no Trotter splitting needed per pair). exp(-K) reverses the
    order of the exp(+K) product, as required for the two to be exact
    (non-commuting) inverses of one another."""
    order = pairs if sign > 0 else list(reversed(pairs))
    for (p, q) in order:
        orig_idx = pair_index[(p, q)]
        t = sign * theta[orig_idx]
        wp, wq = wires[p], wires[q]
        if variant == "re":
            # exp[t (a_p^dag a_q - a_q^dag a_p)] -> real Givens rotation
            qml.SingleExcitation(2 * t, wires=[wp, wq])
        elif variant == "im":
            # exp[i t (a_p^dag a_q + a_q^dag a_p)] -> XY hopping rotation
            qml.IsingXY(-2 * t, wires=[wp, wq])
        elif variant == "g":
            ph = phi[orig_idx]
            apply_generalized_givens(t, ph, [wp, wq])
        else:
            raise ValueError(f"unknown variant {variant!r}")


def apply_J(jpar, wires, pairs):
    """exp(i J_pq n_p n_q) for every pair -- exact single-gate diagonal
    phase, since n_p n_q is the projector onto |11>."""
    for idx, (p, q) in enumerate(pairs):
        qml.ControlledPhaseShift(jpar[idx], wires=[wires[p], wires[q]])


def ucj_circuit(params, wires, variant: str, k_layers: int, ref_pattern):
    prepare_reference(wires, ref_pattern)
    N = len(wires)
    npair = n_pair(N)
    ppl = params_per_layer(N, variant)
    pairs = all_pairs(N)
    pair_index = {pq: i for i, pq in enumerate(pairs)}
    for layer in range(k_layers):
        block = params[layer * ppl:(layer + 1) * ppl]
        theta = block[:npair]
        if variant == "g":
            phi = block[npair:2 * npair]
            jpar = block[2 * npair:3 * npair]
        else:
            phi = None
            jpar = block[npair:2 * npair]
        apply_K(theta, phi, wires, variant, +1.0, pairs, pair_index)
        apply_J(jpar, wires, pairs)
        apply_K(theta, phi, wires, variant, -1.0, pairs, pair_index)


# =====================================================================
# 4. VQE driver
# =====================================================================


@dataclass
class UCJResult:
    variant: str
    target: str
    N: int
    k_layers: int
    energy: float
    exact_energy: float
    params: np.ndarray
    n_params: int
    converged: bool
    history: list
    restart_histories: list


def get_reference_and_hamiltonian(N: int, target: str):
    """Returns (pattern, H_cost, exact_energy, cost_sign).

    H_cost is the operator the VQE cost function evaluates <H_cost>.
    exact_energy and the reported/optimized energy are both expressed
    in *physical* units, i.e. as an expectation value of the true
    Heisenberg H (never of -H) -- cost_sign converts between the two:
    <H_true> = cost_sign * <H_cost>. For "highest" we minimize <-H_true>
    (cost_sign=-1) so that L-BFGS-B, which always minimizes, drives the
    circuit toward the *top* of the spectrum.
    """
    if target == "ground":
        pattern = neel_pattern(N)
        H = heisenberg_hamiltonian(N, negate=False)
        n_up = N // 2
        exact = exact_low_energies(N, n_up, k=1, negate=False)[0]
        cost_sign = 1.0
    elif target == "excited":
        pattern = triplet_pattern(N)
        H = heisenberg_hamiltonian(N, negate=False)
        n_up = N // 2 + 1
        exact = exact_low_energies(N, n_up, k=1, negate=False)[0]
        cost_sign = 1.0
    elif target == "highest":
        pattern = neel_pattern(N)
        H = heisenberg_hamiltonian(N, negate=True)  # cost = <-H_true>
        n_up = N // 2
        # top of the Sz_tot=0 spectrum = -(lowest eigenvalue of -H);
        # diagonalize -H directly (spectrum is not symmetric about 0).
        exact = -exact_low_energies(N, n_up, k=1, negate=True)[0]
        cost_sign = -1.0
    else:
        raise ValueError(target)
    return pattern, H, exact, cost_sign


def RBM_gs(L = 8, alpha = 3, n_iter = 300, n_samples = 2016):

    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
    ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
    
    ma = nk.models.RBMSymm(symmetries=g.translation_group(), alpha=alpha)

    sa = nk.sampler.MetropolisExchange(hi, graph=g)

    # Optimizer
    op = nk.optimizer.Sgd(learning_rate=0.02)

    # The variational state
    vs = nk.vqs.MCState(sa, ma, n_samples=n_samples)

    # The ground-state optimization loop
    gs = nk.driver.VMC_SR(hamiltonian=ha, optimizer=op, diag_shift=0.01, variational_state=vs)

    start = time.time()
    gs.run(out="RBMSymmetric", n_iter=n_iter)
    end = time.time()
    return vs, hi


def extract_correlators(vs, hi):
    N = hi.size
    def n_op(hi, i):
        """n_i = (1 + sigma^z_i)/2"""
        return 0.5 * (sigmaz(hi, i) + 1)

    def cdag_c(hi, i, j):
        """c_i^dagger c_j for i < j, with explicit JW string phase."""
        op = sigmap(hi, i)
        for k in range(i + 1, j):
            op = op @ sigmaz(hi, k)
        sign = (-1) ** (j - i - 1)
        op = sign * (op @ sigmam(hi, j))
        return 0.25 * op

    # single-site occupations
    n_mean = np.array([vs.expect(n_op(hi, i)).mean.real for i in range(N)])

    C = np.zeros((N, N))
    for p in range(N):
        for q in range(N):
            if p == q:
                # <n_p^2> = <n_p> since n_p is a projector (eigenvalues 0/1)
                npnp = n_mean[p]
            else:
                npnq_op = n_op(hi, p) @ n_op(hi, q)
                npnp = vs.expect(npnq_op).mean.real
            C[p, q] = npnp - n_mean[p] * n_mean[q]




    rho = np.zeros((N, N), dtype=complex)

    # diagonal: rho_pp = <n_p>
    for p in range(N):
        rho[p, p] = n_mean[p]

    # off-diagonal: rho_pq for p < q, then Hermitian conjugate for p > q
    for p in range(N):
        for q in range(p + 1, N):
            val = vs.expect(cdag_c(hi, p, q)).mean
            rho[p, q] = val
            rho[q, p] = np.conj(val)

    def symmetrize_circulant(M, L):
        """Average M[p,q] over all translations to enforce circulant structure."""
        M_sym = np.zeros_like(M)
        for shift in range(L):
            M_sym += np.roll(np.roll(M, shift, axis=0), shift, axis=1)
        return M_sym / L

    rho_sym = symmetrize_circulant(rho, N)
    rho_rdm = np.eye(N) - rho_sym.T
    C_sym = symmetrize_circulant(C, N)

    return C_sym, rho_rdm


def warm_start_params(N, variant, k_layers, corr=None, rdm=None, rng=None):
    rng = rng or np.random.default_rng(0)
    npair = n_pair(N)
    ppl = params_per_layer(N, variant)
    pairs = all_pairs(N)
    x0 = np.zeros(k_layers * ppl)

    for layer in range(k_layers):
        base = layer * ppl

        if rdm is not None:
            if variant == "re":
                theta = np.array([rdm[p, q].real for (p, q) in pairs]) * 0.5
            elif variant == "im":
                theta = np.array([rdm[p, q].imag for (p, q) in pairs]) * 0.5
            elif variant == "g":
                theta = np.array([abs(rdm[p, q]) for (p, q) in pairs]) * 0.5
                phase = np.array([np.angle(rdm[p, q]) for (p, q) in pairs])
        else:
            theta = 0.01 * rng.standard_normal(npair)
            phase = 0.01 * rng.standard_normal(npair)

        jpar = (np.array([corr[p, q] for (p, q) in pairs])
                if corr is not None else 0.01 * rng.standard_normal(npair))

        if variant in ("re", "im"):
            x0[base:base+npair] = theta
            x0[base+npair:base+2*npair] = jpar
        elif variant == "g":
            x0[base:base+npair] = theta
            x0[base+npair:base+2*npair] = phase
            x0[base+2*npair:base+3*npair] = jpar
    return x0


def make_cost(N: int, variant: str, k_layers: int, pattern, H: qml.Hamiltonian):
    dev = qml.device("lightning.qubit", wires=N)
    wires = list(range(N))

    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def circuit(params):
        ucj_circuit(params, wires, variant, k_layers, pattern)
        return qml.expval(H)

    grad_fn = qml.grad(circuit)
    #value_and_grad = jax.jit(jax.value_and_grad(circuit))

    def cost(params_np):
        params = pnp.array(params_np, requires_grad=True)
        val = float(circuit(params))
        grad = np.asarray(grad_fn(params), dtype=np.float64)
        #val, grad = value_and_grad(params)
        return val, grad

    return cost, circuit


def optimize_layer(N, variant, k_layers, target, n_restarts=3, seed=0,
                   warm_start=None, maxiter=400, verbose=False):

    pattern, H, exact, cost_sign = get_reference_and_hamiltonian(N, target)
    cost, _ = make_cost(N, variant, k_layers, pattern, H)

    ppl = params_per_layer(N, variant)
    n_params = k_layers * ppl
    rng = np.random.default_rng(seed)

    best_val, best_x = np.inf, None
    best_history = None
    restart_histories = []

    for r in range(n_restarts):

        if r == 0 and warm_start is not None:
            x0 = warm_start
            start_type = "warm"
        else:
            x0 = 0.1 * rng.standard_normal(n_params)
            start_type = "random"

        # Store objective values for this restart
        history = []

        # Record initial energy
        initial_val, initial_grad = cost(x0)
        history.append({
            "iter": 0,
            "energy": initial_val,
            "grad_norm": np.linalg.norm(initial_grad),
        })

        iteration = [0]

        def callback(xk):
            iteration[0] += 1

            val, grad = cost(xk)

            history.append({
                "iter": iteration[0],
                "energy": val,
                "grad_norm": np.linalg.norm(grad),
            })

        res = minimize(
            cost,
            x0,
            jac=True,
            method="L-BFGS-B",
            callback=callback,
            options={"maxiter": maxiter},
        )

        restart_histories.append({
            "restart": r,
            "start_type": start_type,
            "history": history,
            "result": res,
        })

        if verbose:
            print(
                f"    restart {r} ({start_type}): "
                f"E={res.fun:.8f}, "
                f"iterations={len(history)-1}"
            )

        if res.fun < best_val:
            best_val = res.fun
            best_x = res.x
            best_history = history

    return UCJResult(
        variant=variant,
        target=target,
        N=N,
        k_layers=k_layers,
        energy=best_val,
        exact_energy=float(exact),
        params=best_x,
        n_params=n_params,
        converged=abs(best_val - exact) < 1e-3,
        history=best_history,
        restart_histories=restart_histories,
    )


def adaptive_layer_search(N, variant, target, k_max=4, tol=5e-3,
                           n_restarts=3, seed=0, verbose=True):
    """Increase k until |E - E_exact| < tol or k_max is reached."""
    history = []
    prev_x = None
    for k in range(1, k_max + 1):
        warm = None
        if prev_x is not None:
            # extend previous best params with a near-identity extra layer
            ppl = params_per_layer(N, variant)
            warm = np.concatenate(
                [prev_x, 0.01 * np.random.default_rng(seed + k).standard_normal(ppl)]
            )
        result = optimize_layer(N, variant, k, target, n_restarts=n_restarts,
                                 seed=seed + k, warm_start=warm, verbose=False)
        history.append(result)
        if verbose:
            err = result.energy - result.exact_energy
            print(f"  k={k}: E_uCJ={result.energy:.8f}  E_exact={result.exact_energy:.8f}"
                  f"  |err|={abs(err):.2e}  params={result.n_params}")
        prev_x = result.params
        if abs(result.energy - result.exact_energy) < tol:
            break
    return history



def run(L =  int, warmstart = bool, variant = 'g'):


    if warmstart:
        vs, hi = RBM_gs(L)
        C_sym, rho_rdm = extract_correlators(vs, hi)
        x0_g = warm_start_params(L, variant=variant, k_layers=1, corr=C_sym, rdm=rho_rdm)
        result = optimize_layer( N=L, variant=variant, k_layers=1, target="ground",
            n_restarts=1, seed=23, warm_start=x0_g, maxiter=200, verbose=True)
    else:
        result = optimize_layer( N=L, variant=variant, k_layers=1, target="ground",
            n_restarts=1, seed=23, maxiter=200, verbose=True)
    return result


def compare(L = int, exact = bool):
    import matplotlib.pyplot as plt
    def plot(result, label):
        hist = result.history
        iterations = [h["iter"] for h in hist]
        energies = [h["energy"] for h in hist]
        print(label,result.energy, result.exact_energy, result.converged)
        plt.plot(iterations, energies, marker=".", markersize=3,label=label)

    for variant in ("re", "im", "g"):
        print(f"{variant} variant:")
        warm_result = run(L=L, variant=variant, warmstart=True)
        cold_result = run(L=L, variant=variant, warmstart=False)
        plt.figure(figsize=(7, 5))
        plot(warm_result, 'Warm')
        plot(cold_result, 'Cold')
        
        if exact:

            plt.axhline(
            warm_result.exact_energy,
            linestyle="--",
            label="Exact"
            )
            print(f"exact energy: {warm_result.exact_energy}")
            print(f"cold energy error: {cold_result.energy}")
            print(f"warm energy error: {warm_result.energy - warm_result.exact_energy}")
            print(f"cold energy error: {cold_result.energy - warm_result.exact_energy}")

        plt.xlabel("L-BFGS-B iteration")
        plt.ylabel("Energy")
        plt.title(f"{variant} Warmstart uCJ convergence")
        plt.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    compare(L=8, exact=True)
