"""
main.py
=======
Builds UCJ and DMRG-MPS circuits for the J1-J2 Heisenberg chain,
applies an energy filter to each, and compares final fidelity with
the exact ground state obtained via Lanczos (scipy sparse eigsh).

Pipeline
--------
  1.  Exact diagonalisation (Lanczos) → full spectrum + ground state
  2.  Build UCJ circuit  (Jastrow + Givens, variational)
  3.  Build DMRG-MPS circuit  (TeNPy DMRG → mps-to-circuit)
  4.  Extract statevectors from each circuit (Qiskit Statevector)
  5.  Project onto fixed-Sz sector; decompose onto exact eigenbasis
  6.  Optimise energy filter (FilterBuilder, method v5)
  7.  Apply filter to both statevectors; compute fidelity |<ψ_filt|ψ_0>|²
  8.  Print comparison table

Default: 8-site chain, J1=1, J2=0, PBC

Usage
-----
    python main.py [--n N] [--j1 J1] [--j2 J2] [--chi CHI] [--layers K]
"""
from __future__ import annotations

import argparse
import warnings
import time
import functools
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy import optimize as opt

# =============================================================================
# OPTIONAL HEAVY IMPORTS  (fail loudly with install hint)
# =============================================================================
try:
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    _JAX_OK = True
except ImportError:
    _JAX_OK = False
    warnings.warn("JAX not found — UCJ optimisation disabled. pip install jax")

try:
    import tenpy
    from tenpy.networks.site import SpinHalfSite
    from tenpy.models.model import CouplingMPOModel
    from tenpy.models.lattice import Chain
    from tenpy.networks.mps import MPS
    from tenpy.algorithms import dmrg
    _TENPY_OK = True
except ImportError:
    _TENPY_OK = False
    warnings.warn("TeNPy not found — DMRG disabled. pip install physics-tenpy")

try:
    from mps_to_circuit import mps_to_circuit
    _MPS2CIRC_OK = True
except ImportError:
    _MPS2CIRC_OK = False
    warnings.warn("mps-to-circuit not found — DMRG circuit disabled. "
                  "pip install git+https://github.com/qiskit-community/mps-to-circuit.git")

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.circuit.library import RXXGate, RYYGate, RZGate, CPhaseGate
    from qiskit.quantum_info import Statevector
    _QISKIT_OK = True
except ImportError:
    _QISKIT_OK = False
    warnings.warn("Qiskit not found. pip install qiskit")

# =============================================================================
# CONFIG
# =============================================================================
DEFAULT_N      = 8
DEFAULT_J1     = 1.0
DEFAULT_J2     = 0.0
DEFAULT_PBC    = True
DEFAULT_CHI    = 64
DEFAULT_LAYERS = 1
DEFAULT_VARIANT = "re"    # 're' | 'im' | 'g'

# UCJ optimiser
N_RESTARTS   = 3
NOISE_SCALE  = 0.05
SEED         = 42
LBFGS_MAXITER = 1000
LBFGS_FTOL   = 1e-14
LBFGS_GTOL   = 1e-8

# Filter
FILTER_A = 4   # min pulses
FILTER_B = 10  # max pulses

TARGET_BASIS = ["cx", "rz", "h", "s", "sdg"]

# =============================================================================
# TIMER
# =============================================================================
class Timer:
    def __init__(self):
        self._laps = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = time.perf_counter()

    def stop(self, name):
        self._laps[name] = time.perf_counter() - self._starts.pop(name)
        return self._laps[name]

    def summary(self):
        W = 56
        print(f"\n{'='*W}\n  Runtime summary\n{'='*W}")
        total = 0.0
        for name, t in self._laps.items():
            total += t
            m, s = divmod(t, 60)
            print(f"  {name:<38}  {int(m):2d}m {s:05.2f}s")
        m, s = divmod(total, 60)
        print(f"  {'─'*52}")
        print(f"  {'TOTAL':<38}  {int(m):2d}m {s:05.2f}s")
        print(f"{'='*W}\n")

# =============================================================================
# LATTICE
# =============================================================================
@dataclass
class ChainLattice:
    """1-D chain with NN and NNN edges."""
    n_sites: int
    pbc: bool = True
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"chain_N{self.n_sites}_{'PBC' if self.pbc else 'OBC'}"
        n = self.n_sites
        if self.pbc:
            self.nn_edges  = [(i, (i + 1) % n) for i in range(n)]
            self.nnn_edges = [(i, (i + 2) % n) for i in range(n)]
        else:
            self.nn_edges  = [(i, i + 1) for i in range(n - 1)]
            self.nnn_edges = [(i, i + 2) for i in range(n - 2)]

# =============================================================================
# EXACT DIAGONALISATION  (Lanczos via scipy sparse eigsh)
# =============================================================================
def get_n_up(n: int) -> int:
    return n // 2


def build_basis(n: int, n_up: int) -> np.ndarray:
    return np.array([b for b in range(1 << n) if bin(b).count("1") == n_up],
                    dtype=np.int64)


def build_sparse_hamiltonian(n, n_up, nn_edges, nnn_edges, j1, j2):
    basis   = build_basis(n, n_up)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    dim     = len(basis)
    H       = lil_matrix((dim, dim), dtype=np.float64)
    for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
        for si, sj in edges:
            for row, bits in enumerate(basis):
                zi = 0.5 if (bits >> si) & 1 else -0.5
                zj = 0.5 if (bits >> sj) & 1 else -0.5
                H[row, row] += j * zi * zj
                if ((bits >> si) & 1) != ((bits >> sj) & 1):
                    fl  = bits ^ (1 << si) ^ (1 << sj)
                    col = idx_map.get(int(fl), -1)
                    if col >= 0:
                        H[row, col] += 0.5 * j
    return csr_matrix(H), basis, idx_map


def exact_full_spectrum(n, nn_edges, nnn_edges, j1, j2):
    """
    Return all eigenvalues and eigenvectors in the fixed n_up sector.
    Uses scipy dense eigh for small N (N<=16), Lanczos for larger.
    """
    n_up = get_n_up(n)
    H_sp, basis, idx_map = build_sparse_hamiltonian(n, n_up, nn_edges, nnn_edges, j1, j2)
    dim = H_sp.shape[0]
    print(f"  [ED]  sector dim={dim}  (n={n}, n_up={n_up})")

    if dim <= 512:
        from scipy.linalg import eigh
        evals, evecs = eigh(H_sp.toarray())
    else:
        # Lanczos — get all eigenvalues iteratively
        k = min(dim - 1, dim)
        evals, evecs = eigsh(H_sp, k=k, which="SA", tol=1e-12)
        order  = np.argsort(evals)
        evals  = evals[order]
        evecs  = evecs[:, order]

    # Normalise all columns
    norms = np.linalg.norm(evecs, axis=0)
    evecs /= norms[np.newaxis, :]
    print(f"  [ED]  E_0={evals[0]:.8f}  E_1={evals[1]:.8f}  gap={evals[1]-evals[0]:.6f}")
    return evals, evecs, basis, idx_map


# =============================================================================
# STATEVECTOR UTILITIES
# =============================================================================
def sv_to_sector(sv_full: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project full 2^n statevector onto fixed-Sz sector basis."""
    psi = sv_full[basis].astype(np.complex128)
    norm = np.linalg.norm(psi)
    if norm < 1e-10:
        warnings.warn("Statevector has near-zero projection onto Sz sector!")
    return psi / (norm + 1e-300)


def sv_eigenbasis_components(psi_sector: np.ndarray,
                              evecs: np.ndarray) -> np.ndarray:
    """
    Decompose |ψ⟩ onto exact eigenbasis: c_k = <φ_k|ψ>.
    Returns complex array of shape (dim,).
    """
    # evecs columns are eigenstates; psi_sector is a column vector
    return evecs.conj().T @ psi_sector   # shape (dim,)


def fidelity_with_ground_state(psi_sector: np.ndarray,
                                evecs: np.ndarray) -> float:
    """|<φ_0|ψ>|²"""
    return float(np.abs(np.dot(evecs[:, 0].conj(), psi_sector)) ** 2)


def reorder_sv_qiskit_to_site(sv: np.ndarray, n: int) -> np.ndarray:
    """Convert Qiskit little-endian → big-endian (site-0 = MSB) bit ordering."""
    out = np.zeros_like(sv)
    for i in range(len(sv)):
        j = int(f"{i:0{n}b}"[::-1], 2)
        out[j] = sv[i]
    return out


# =============================================================================
# ENERGY FILTER
# =============================================================================
def new_func_v5(coeffs_sq: np.ndarray, energies: np.ndarray):
    """
    Objective for filter optimisation.
    coeffs_sq[k] = |<φ_k|ψ_trial>|²
    Minimise  1 - (GS weight after filter) / (total weight after filter)
    """
    def objective(timesphases):
        n = len(timesphases) // 2
        times  = timesphases[:n]
        phases = timesphases[n:]
        cos_vals = np.cos(energies[:, None] * times + phases)   # (dim, n)
        weights  = np.prod(cos_vals ** 2, axis=1)               # (dim,)
        gs_w   = coeffs_sq[0] * weights[0]
        total  = np.dot(coeffs_sq, weights)
        return 1.0 - gs_w / (total + 1e-30)
    return objective


def timesconstraints(timesphases, total_time):
    n = len(timesphases) // 2
    return abs(np.sum(timesphases[:n]) - total_time)


def build_filter(coeffs_sq: np.ndarray,
                 energies: np.ndarray,
                 total_time: float,
                 a: int = FILTER_A,
                 b: int = FILTER_B) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimise filter pulse sequence.
    Returns (best_times, best_phases, best_fun).
    """
    objective = new_func_v5(coeffs_sq, energies)
    best_fun  = np.inf
    best_tp   = None

    for ntimes in range(a, b + 1):
        times = np.ones(ntimes)
        for i in range(1, ntimes):
            times[i] = times[i - 1] / 2.0
        times *= total_time / np.sum(np.abs(times))

        x0 = np.zeros(2 * ntimes)
        x0[:ntimes] = times

        bnd_t = [(0.0, total_time / 3.0)] * ntimes
        bnd_p = [(-np.pi / 2, np.pi / 2)] * ntimes
        bnds  = bnd_t + bnd_p

        constr = [{"type": "eq", "fun": timesconstraints, "args": (total_time,)}]

        res = opt.minimize(
            objective, x0,
            method="SLSQP",
            bounds=bnds,
            constraints=constr,
            options={"maxiter": 5000, "ftol": 1e-13},
            tol=1e-13,
        )

        print(f"    filter ntimes={ntimes:2d}  fun={res.fun:.3e}  success={res.success}")
        if res.fun < best_fun:
            best_fun = res.fun
            best_tp  = res.x.copy()

    n_best = len(best_tp) // 2
    return best_tp[:n_best], best_tp[n_best:], best_fun


def apply_filter(times: np.ndarray, phases: np.ndarray,
                 energies: np.ndarray,
                 coeffs_complex: np.ndarray) -> np.ndarray:
    """
    Apply filter in the eigenbasis: c_k → c_k * prod_i cos(E_k t_i + φ_i)
    Returns normalised filtered coefficient vector.
    """
    c = coeffs_complex.copy()
    for t, phi in zip(times, phases):
        c *= np.cos(energies * t + phi)
    norm = np.linalg.norm(c)
    if norm < 1e-12:
        warnings.warn("Filter produced near-zero state!")
        return c
    return c / norm


def filtered_fidelity(times, phases, energies, coeffs_complex):
    """
    |<φ_0|ψ_filtered>|² where |ψ_filtered⟩ is in the eigenbasis.
    """
    c_filt = apply_filter(times, phases, energies, coeffs_complex)
    return float(np.abs(c_filt[0]) ** 2)


# =============================================================================
# UCJ  (self-contained, JAX-based)
# =============================================================================
if _JAX_OK:
    _DEV = jax.devices("cpu")[0]

    def _to_dev(x):
        return jax.device_put(jnp.array(x, dtype=jnp.float64), _DEV)

    def _to_dev_c(x):
        return jax.device_put(jnp.array(x, dtype=jnp.complex128), _DEV)

    def build_jax_hamiltonian(n, n_up, nn_edges, nnn_edges, j1, j2,
                              basis, idx_map):
        rows, cols, vals = [], [], []
        for edges, j in [(nn_edges, j1), (nnn_edges, j2)]:
            for si, sj in edges:
                for row, bits in enumerate(basis):
                    zi = 0.5 if (bits >> si) & 1 else -0.5
                    zj = 0.5 if (bits >> sj) & 1 else -0.5
                    rows.append(row); cols.append(row); vals.append(j * zi * zj)
                    if ((bits >> si) & 1) != ((bits >> sj) & 1):
                        fl = int(bits ^ (1 << si) ^ (1 << sj))
                        if fl in idx_map:
                            rows.append(row)
                            cols.append(idx_map[fl])
                            vals.append(0.5 * j)
        h_r = jax.device_put(jnp.array(rows, dtype=jnp.int32), _DEV)
        h_c = jax.device_put(jnp.array(cols, dtype=jnp.int32), _DEV)
        h_v = jax.device_put(jnp.array(vals, dtype=jnp.float64), _DEV)
        dim = len(basis)

        @jax.jit
        def apply_H(psi):
            return jnp.zeros(dim, dtype=psi.dtype).at[h_r].add(h_v * psi[h_c])

        return apply_H

    def build_jastrow_fn(n, basis):
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pi = jax.device_put(jnp.array([p[0] for p in pairs], dtype=jnp.int32), _DEV)
        pj = jax.device_put(jnp.array([p[1] for p in pairs], dtype=jnp.int32), _DEV)
        bits = jax.device_put(jnp.array(basis, dtype=jnp.int32), _DEV)

        @jax.jit
        def jastrow_phase(theta_J):
            def acc(phase, args):
                tk, i, j = args
                bi = ((bits >> i) & 1).astype(jnp.float64)
                bj = ((bits >> j) & 1).astype(jnp.float64)
                return phase + tk * bi * bj, None
            phase, _ = jax.lax.scan(
                acc,
                jnp.zeros(bits.shape[0], dtype=jnp.float64),
                (theta_J, pi, pj),
            )
            return phase

        return jastrow_phase

    def build_givens_pairs(n, basis, idx_map):
        srcs_rag, dsts_rag = [], []
        for i in range(n):
            for j in range(i + 1, n):
                srcs, dsts = [], []
                for row, bits in enumerate(basis):
                    if ((bits >> i) & 1) and not ((bits >> j) & 1):
                        fl = int(bits ^ (1 << i) ^ (1 << j))
                        if fl in idx_map:
                            srcs.append(row)
                            dsts.append(idx_map[fl])
                srcs_rag.append(np.array(srcs, dtype=np.int32))
                dsts_rag.append(np.array(dsts, dtype=np.int32))

        counts  = np.array([len(s) for s in srcs_rag], dtype=np.int32)
        row_ptr = np.zeros(len(counts) + 1, dtype=np.int32)
        row_ptr[1:] = np.cumsum(counts)
        srcs_cat = np.concatenate(srcs_rag) if any(len(s) for s in srcs_rag) else np.array([], dtype=np.int32)
        dsts_cat = np.concatenate(dsts_rag) if any(len(d) for d in dsts_rag) else np.array([], dtype=np.int32)
        return (jax.device_put(jnp.array(srcs_cat, dtype=jnp.int32), _DEV),
                jax.device_put(jnp.array(dsts_cat, dtype=jnp.int32), _DEV),
                row_ptr)

    def _givens_scan(psi, thetas, srcs, dsts, row_ptr, imag=False):
        for k in range(len(row_ptr) - 1):
            s, e = int(row_ptr[k]), int(row_ptr[k + 1])
            if s == e:
                continue
            c, ss = jnp.cos(thetas[k]), jnp.sin(thetas[k])
            ps, pd = psi[srcs[s:e]], psi[dsts[s:e]]
            if imag:
                ns = c * ps - 1j * ss * pd
                nd = -1j * ss * ps + c * pd
            else:
                ns = c * ps - ss * pd
                nd = ss * ps + c * pd
            psi = psi.at[srcs[s:e]].set(ns).at[dsts[s:e]].set(nd)
        return psi

    def _ucj_state_fn(theta, variant, k_layers, psi0, n_pair,
                      srcs, dsts, row_ptr, jastrow_fn):
        psi    = psi0
        stride = 3 * n_pair if variant == "g" else 2 * n_pair
        for l in range(k_layers):
            off = l * stride
            phase = jastrow_fn(theta[off:off + n_pair])
            psi   = psi * jnp.exp(1j * phase)
            psi   = _givens_scan(psi, theta[off + n_pair:off + 2 * n_pair],
                                 srcs, dsts, row_ptr, imag=(variant == "im"))
            if variant == "g":
                psi = _givens_scan(psi, theta[off + 2 * n_pair:off + 3 * n_pair],
                                   srcs, dsts, row_ptr, imag=True)
        return psi

    def _energy_fn(psi, apply_H):
        norm = jnp.dot(jnp.conj(psi), psi)
        return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)


def run_ucj(lattice: ChainLattice,
            evals: np.ndarray,
            evecs: np.ndarray,
            basis: np.ndarray,
            idx_map: dict,
            j1: float, j2: float,
            variant: str = DEFAULT_VARIANT,
            k_layers: int = DEFAULT_LAYERS,
            ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Optimise UCJ ansatz.
    Returns (psi_sector, coeffs_eigenbasis, initial_fidelity).
    psi_sector : normalised statevector in the Sz sector.
    coeffs_eigenbasis : complex decomposition onto exact eigenstates.
    """
    if not _JAX_OK:
        raise RuntimeError("JAX required for UCJ.")

    n      = lattice.n_sites
    n_up   = get_n_up(n)
    pairs  = [(i, j) for i in range(n) for j in range(i + 1, n)]
    n_pair = len(pairs)

    apply_H    = build_jax_hamiltonian(n, n_up,
                                       lattice.nn_edges, lattice.nnn_edges,
                                       j1, j2, basis, idx_map)
    jastrow_fn = build_jastrow_fn(n, basis)
    srcs, dsts, row_ptr = build_givens_pairs(n, basis, idx_map)

    # Néel initial state in sector
    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi0 = jax.device_put(
        jnp.zeros(len(basis), dtype=jnp.complex128).at[idx_map[neel_bits]].set(1.0),
        _DEV
    )

    stride = 3 * n_pair if variant == "g" else 2 * n_pair

    def efn(theta_dev):
        psi = _ucj_state_fn(theta_dev, variant, k_layers, psi0, n_pair,
                            srcs, dsts, row_ptr, jastrow_fn)
        return _energy_fn(psi, apply_H)

    val_grad = jax.jit(jax.value_and_grad(efn))

    rng  = np.random.default_rng(SEED)
    best_E, best_theta = np.inf, None

    for restart in range(N_RESTARTS):
        x0 = NOISE_SCALE * rng.standard_normal(k_layers * stride)
        x0_dev = _to_dev(x0)
        val_grad(x0_dev)  # JIT warm-up on first call

        def scipy_fn(x_np):
            xd = _to_dev(x_np)
            E, g = val_grad(xd)
            return float(E), np.array(g, dtype=np.float64)

        res = opt.minimize(scipy_fn, x0, jac=True, method="L-BFGS-B",
                           options={"maxiter": LBFGS_MAXITER,
                                    "ftol": LBFGS_FTOL,
                                    "gtol": LBFGS_GTOL})
        print(f"    [UCJ restart {restart}]  E={res.fun:.8f}  "
              f"|ΔE|={abs(res.fun - evals[0]):.4e}")
        if res.fun < best_E:
            best_E, best_theta = res.fun, res.x.copy()

    # Extract final statevector
    psi_jax = _ucj_state_fn(_to_dev(best_theta), variant, k_layers, psi0,
                             n_pair, srcs, dsts, row_ptr, jastrow_fn)
    psi_np  = np.array(psi_jax, dtype=np.complex128)
    psi_np /= np.linalg.norm(psi_np)

    coeffs = sv_eigenbasis_components(psi_np, evecs)
    fid    = fidelity_with_ground_state(psi_np, evecs)
    print(f"    [UCJ]  E={best_E:.8f}  fidelity={fid:.8f}")
    return psi_np, coeffs, fid


# =============================================================================
# DMRG → MPS CIRCUIT
# =============================================================================
class HeisenbergChainJ1J2(CouplingMPOModel if _TENPY_OK else object):
    def init_sites(self, model_params):
        conserve = model_params.get('conserve', 'Sz')
        return SpinHalfSite(conserve=conserve)

    def init_lattice(self, model_params):
        L      = model_params['L']
        site   = self.init_sites(model_params)
        bc_MPS = model_params.get('bc_MPS', 'finite')
        return Chain(L, site, bc_MPS=bc_MPS, bc='periodic')

    def init_terms(self, model_params):
        J1 = model_params.get('J1', 1.0)
        J2 = model_params.get('J2', 0.0)
        self.add_coupling(J1 / 2, 0, 'Sp', 0, 'Sm', dx=1, op_string='JW')
        self.add_coupling(J1 / 2, 0, 'Sm', 0, 'Sp', dx=1, op_string='JW')
        self.add_coupling(J1,     0, 'Sz', 0, 'Sz', dx=1)
        if J2 != 0.0:
            self.add_coupling(J2 / 2, 0, 'Sp', 0, 'Sm', dx=2, op_string='JW')
            self.add_coupling(J2 / 2, 0, 'Sm', 0, 'Sp', dx=2, op_string='JW')
            self.add_coupling(J2,     0, 'Sz', 0, 'Sz', dx=2)


def run_dmrg_mps(lattice: ChainLattice,
                 evals: np.ndarray,
                 evecs: np.ndarray,
                 basis: np.ndarray,
                 j1: float, j2: float,
                 chi_max: int = DEFAULT_CHI,
                 ) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run DMRG on a PBC chain, convert MPS → Qiskit circuit, get statevector.
    Returns (psi_sector, coeffs_eigenbasis, initial_fidelity).
    """
    if not _TENPY_OK:
        raise RuntimeError("TeNPy required for DMRG.")
    if not _MPS2CIRC_OK:
        raise RuntimeError("mps-to-circuit required for MPS → circuit.")
    if not _QISKIT_OK:
        raise RuntimeError("Qiskit required.")

    n = lattice.n_sites
    model_params = dict(L=n, J1=j1, J2=j2, conserve='Sz', bc_MPS='finite')
    model = HeisenbergChainJ1J2(model_params)

    init_state = ['up' if i % 2 == 0 else 'down' for i in range(n)]
    psi_mps    = MPS.from_product_state(model.lat.mps_sites(), init_state, bc='finite')

    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'N_sweeps_check': 1,
        'mixer': True,
        'mixer_params': {'amplitude': 1e-4, 'decay': 1.2, 'disable_after': 50},
    }
    eng = dmrg.TwoSiteDMRGEngine(psi_mps, model, dmrg_params)
    E_dmrg, psi_mps = eng.run()
    print(f"    [DMRG]  E={E_dmrg:.8f}  chi_max={max(psi_mps.chi)}")

    # MPS tensors → circuit
    mps_arrays = [psi_mps.get_B(i).to_ndarray() for i in range(n)]
    qc = mps_to_circuit(mps_arrays, method="exact", shape="lpr")
    qc_t = transpile(qc, basis_gates=TARGET_BASIS, optimization_level=3)

    # Statevector (Qiskit little-endian → big-endian)
    sv_le = Statevector(qc_t).data
    sv_be = reorder_sv_qiskit_to_site(sv_le, n)

    psi_sec = sv_to_sector(sv_be, basis)
    coeffs  = sv_eigenbasis_components(psi_sec, evecs)
    fid     = fidelity_with_ground_state(psi_sec, evecs)
    print(f"    [DMRG-MPS circuit]  fidelity={fid:.8f}")
    return psi_sec, coeffs, fid


# =============================================================================
# PRINT COMPARISON TABLE
# =============================================================================
def print_comparison(results: dict):
    W   = 68
    sep = "=" * W
    thin = "-" * W
    print(f"\n{sep}")
    print(f"  FINAL COMPARISON  –  {results['lattice']}  "
          f"J1={results['j1']}  J2={results['j2']}")
    print(f"{sep}")
    print(f"  E_exact (Lanczos) = {results['e_exact']:.10f}  "
          f"(E/site = {results['e_exact']/results['n']:.10f})")
    print(f"{thin}")
    hdr = f"  {'Method':<22}  {'Pre-filter Fid':>14}  {'Post-filter Fid':>15}  {'ΔE_pre':>10}"
    print(hdr)
    print(f"  {thin}")
    for name, r in results['methods'].items():
        pf  = r.get('fid_pre',  float('nan'))
        pof = r.get('fid_post', float('nan'))
        dE  = r.get('dE_pre',   float('nan'))
        print(f"  {name:<22}  {pf:>14.8f}  {pof:>15.8f}  {dE:>10.4e}")
    print(f"{sep}\n")


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="UCJ vs DMRG-MPS with energy filter")
    parser.add_argument("--n",      type=int,   default=DEFAULT_N)
    parser.add_argument("--j1",     type=float, default=DEFAULT_J1)
    parser.add_argument("--j2",     type=float, default=DEFAULT_J2)
    parser.add_argument("--chi",    type=int,   default=DEFAULT_CHI,
                        help="DMRG max bond dim")
    parser.add_argument("--layers", type=int,   default=DEFAULT_LAYERS,
                        help="UCJ layers k")
    parser.add_argument("--variant",type=str,   default=DEFAULT_VARIANT,
                        help="UCJ variant: re | im | g")
    parser.add_argument("--filter-a", type=int, default=FILTER_A,
                        help="Min filter pulses")
    parser.add_argument("--filter-b", type=int, default=FILTER_B,
                        help="Max filter pulses")
    parser.add_argument("--no-ucj",  action="store_true",
                        help="Skip UCJ (useful if JAX unavailable)")
    parser.add_argument("--no-dmrg", action="store_true",
                        help="Skip DMRG (useful if TeNPy/mps-to-circuit unavailable)")
    args = parser.parse_args()

    n   = args.n
    j1  = args.j1
    j2  = args.j2
    timer = Timer()

    lattice = ChainLattice(n_sites=n, pbc=DEFAULT_PBC)

    print(f"\n{'='*60}")
    print(f"  UCJ vs DMRG-MPS  +  Energy Filter")
    print(f"  {lattice.name}  J1={j1}  J2={j2}")
    print(f"{'='*60}\n")

    # ── 1. Exact diagonalisation (full spectrum via Lanczos) ─────────────────
    print("[1/4]  Exact diagonalisation (Lanczos) ...")
    timer.start("ED full spectrum")
    evals, evecs, basis, idx_map = exact_full_spectrum(
        n, lattice.nn_edges, lattice.nnn_edges, j1, j2)
    timer.stop("ED full spectrum")

    # Energy window for filter: use all eigenvalues
    energies = evals.astype(np.float64)

    # Total filter time heuristic: span of the spectrum
    e_span     = float(energies[-1] - energies[0])
    total_time = max(4.0, 2.0 * np.pi / max(energies[1] - energies[0], 1e-3))
    print(f"  Filter total_time = {total_time:.4f}  (spectrum span {e_span:.4f})")

    # ── 2. UCJ ───────────────────────────────────────────────────────────────
    results_methods = {}

    if not args.no_ucj and _JAX_OK:
        print(f"\n[2a/4]  UCJ  (variant={args.variant}, k={args.layers}) ...")
        timer.start("UCJ optimisation")
        psi_ucj, coeff_ucj, fid_ucj_pre = run_ucj(
            lattice, evals, evecs, basis, idx_map,
            j1, j2, variant=args.variant, k_layers=args.layers)
        timer.stop("UCJ optimisation")

        # pre-filter energy
        e_ucj = float(np.real(np.dot(coeff_ucj.conj(), energies * coeff_ucj)))

        print(f"\n  [UCJ filter] ...")
        timer.start("UCJ filter")
        csq_ucj = np.abs(coeff_ucj) ** 2
        t_ucj, phi_ucj, fun_ucj = build_filter(
            csq_ucj, energies, total_time, args.filter_a, args.filter_b)
        fid_ucj_post = filtered_fidelity(t_ucj, phi_ucj, energies, coeff_ucj)
        timer.stop("UCJ filter")
        print(f"  [UCJ] post-filter fidelity = {fid_ucj_post:.8f}")

        results_methods["UCJ"] = dict(
            fid_pre=fid_ucj_pre, fid_post=fid_ucj_post,
            dE_pre=abs(e_ucj - evals[0]))
    elif args.no_ucj:
        print("\n[2a/4]  UCJ skipped (--no-ucj).")
    else:
        print("\n[2a/4]  UCJ skipped (JAX unavailable).")

    # ── 3. DMRG-MPS circuit ──────────────────────────────────────────────────
    if not args.no_dmrg and _TENPY_OK and _MPS2CIRC_OK and _QISKIT_OK:
        print(f"\n[2b/4]  DMRG + MPS circuit  (chi_max={args.chi}) ...")
        timer.start("DMRG + circuit")
        psi_dmrg, coeff_dmrg, fid_dmrg_pre = run_dmrg_mps(
            lattice, evals, evecs, basis, j1, j2, chi_max=args.chi)
        timer.stop("DMRG + circuit")

        e_dmrg_circ = float(np.real(np.dot(coeff_dmrg.conj(), energies * coeff_dmrg)))

        print(f"\n  [DMRG filter] ...")
        timer.start("DMRG filter")
        csq_dmrg = np.abs(coeff_dmrg) ** 2
        t_dmrg, phi_dmrg, fun_dmrg = build_filter(
            csq_dmrg, energies, total_time, args.filter_a, args.filter_b)
        fid_dmrg_post = filtered_fidelity(t_dmrg, phi_dmrg, energies, coeff_dmrg)
        timer.stop("DMRG filter")
        print(f"  [DMRG-MPS] post-filter fidelity = {fid_dmrg_post:.8f}")

        results_methods["DMRG-MPS circuit"] = dict(
            fid_pre=fid_dmrg_pre, fid_post=fid_dmrg_post,
            dE_pre=abs(e_dmrg_circ - evals[0]))
    elif args.no_dmrg:
        print("\n[2b/4]  DMRG skipped (--no-dmrg).")
    else:
        missing = []
        if not _TENPY_OK:       missing.append("TeNPy")
        if not _MPS2CIRC_OK:    missing.append("mps-to-circuit")
        if not _QISKIT_OK:      missing.append("Qiskit")
        print(f"\n[2b/4]  DMRG skipped (missing: {', '.join(missing)}).")

    # ── 4. Print comparison ──────────────────────────────────────────────────
    print_comparison(dict(
        lattice=lattice.name,
        n=n, j1=j1, j2=j2,
        e_exact=float(evals[0]),
        methods=results_methods,
    ))

    timer.summary()


if __name__ == "__main__":
    main()
