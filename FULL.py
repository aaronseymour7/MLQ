import numpy as np
import functools
import time as _time
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.optimize import minimize as scipy_minimize
from collections import deque

import jax
import jax.numpy as jnp
import optax
import netket as nk
import netket.nn as nknn
import flax.linen as nn

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.compiler import transpile
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import XXPlusYYGate
from qiskit.transpiler import Target, InstructionProperties   # NEW
from qiskit.quantum_info import Operator

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit_ibm_runtime.fake_provider import FakeBrisbane
from qiskit.providers.fake_provider import GenericBackendV2
from qiskit.transpiler import CouplingMap
from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error


try:
    import mthree
    _MTHREE_AVAILABLE = True
except ModuleNotFoundError:
    _MTHREE_AVAILABLE = False
    print("[M3]  mthree not installed — USE_M3 will be forced False. "
          "Install with: pip install mthree")

jax.config.update("jax_enable_x64", True)

# =============================================================================
# CONFIG
# =============================================================================
J_COUPLING  = 1.0
PBC         = True
ALPHA       = 3
VMC_SAMPLES = 1024
VMC_STEPS   = 600
K_MAX       = 1
E_TOL       = 5e-3
SEED        = 23
N_RESTARTS  = 3

SHOTS_EPSILON = 0.01
SHOTS_MIN     = 1024
SHOTS_MAX     = 65536

COBYLA_RHOBEG  = 0.3
COBYLA_RHOEND  = 1e-4
COBYLA_MAXITER = 300

SPSA_MAXITER = 300
SPSA_A       = 0.628
SPSA_C       = 0.1
SPSA_ALPHA   = 0.602
SPSA_GAMMA   = 0.101

QNSPSA_MAXITER        = 300
QNSPSA_A              = 0.628
QNSPSA_C              = 0.1
QNSPSA_ALPHA          = 0.602
QNSPSA_GAMMA          = 0.101
QNSPSA_REGULARISATION = 1e-3
QNSPSA_METRIC_SHOTS   = 1

ADAM_SPSA_LR      = 0.01
ADAM_SPSA_BETA1   = 0.9
ADAM_SPSA_BETA2   = 0.999
ADAM_SPSA_EPS     = 1e-8
ADAM_SPSA_MAXITER = 600
ADAM_SPSA_C       = 0.3
ADAM_SPSA_GAMMA   = 0.101

LBFGS_MAXITER = 300
LBFGS_MAXFUN  = 50_000

USE_M3       = True
M3_CAL_SHOTS = 8192
M3_METHOD    = 'direct'

TRANSPILE_SEED = 42
# =============================================================================

if USE_M3 and not _MTHREE_AVAILABLE:
    print("[M3]  USE_M3=True but mthree is not installed — disabling M3 automatically.")
    USE_M3 = False


# =============================================================================
# TIMER
# =============================================================================

class Timer:
    def __init__(self):
        self._laps   = {}
        self._starts = {}

    def start(self, name):
        self._starts[name] = _time.perf_counter()

    def stop(self, name):
        if name not in self._starts:
            raise KeyError(f"Timer '{name}' was never started.")
        self._laps[name] = _time.perf_counter() - self._starts.pop(name)
        return self._laps[name]

    def elapsed(self, name):
        return self._laps.get(name, float('nan'))

    def summary(self, title="Runtime summary"):
        W = 54
        print(f"\n{'='*W}")
        print(f"  {title}")
        print(f"{'='*W}")
        total = 0.0
        for name, t in self._laps.items():
            total += t
            m, s = divmod(t, 60)
            print(f"  {name:<36}  {int(m):2d}m {s:05.2f}s")
        print(f"  {'─'*48}")
        m, s = divmod(total, 60)
        print(f"  {'TOTAL':<36}  {int(m):2d}m {s:05.2f}s")
        print(f"{'='*W}\n")
        return self._laps.copy()


# =============================================================================
# EXACT DIAGONALISATION
# =============================================================================

def build_basis(n, n_up):
    return np.array([b for b in range(1 << n) if bin(b).count('1') == n_up], dtype=np.int64)


def build_hamiltonian(n, n_up, j=J_COUPLING, pbc=PBC):
    basis   = build_basis(n, n_up)
    idx_map = {int(b): i for i, b in enumerate(basis)}
    H       = lil_matrix((len(basis), len(basis)), dtype=np.float64)
    edges   = [(i, (i+1) % n) for i in range(n)] if pbc else [(i, i+1) for i in range(n-1)]
    for si, sj in edges:
        for row, bits in enumerate(basis):
            zi = 0.5 if (bits >> si) & 1 else -0.5
            zj = 0.5 if (bits >> sj) & 1 else -0.5
            H[row, row] += j * zi * zj
            if ((bits >> si) & 1) != ((bits >> sj) & 1):
                fl = bits ^ (1 << si) ^ (1 << sj)
                col = idx_map.get(int(fl), -1)
                if col >= 0:
                    H[row, col] += 0.5 * j
    return csr_matrix(H), basis, idx_map


# =============================================================================
# JAX HAMILTONIAN
# =============================================================================

def build_jax_hamiltonian(n, n_up, j=J_COUPLING, pbc=PBC):
    basis_list = [b for b in range(1 << n) if bin(b).count('1') == n_up]
    idx_map    = {b: i for i, b in enumerate(basis_list)}
    rows, cols, vals = [], [], []
    edges = [(i, (i+1) % n) for i in range(n)] if pbc else [(i, i+1) for i in range(n-1)]
    for i, js in edges:
        for row, bits in enumerate(basis_list):
            zi = 0.5 if (bits >> i) & 1 else -0.5
            zj = 0.5 if (bits >> js) & 1 else -0.5
            rows.append(row); cols.append(row); vals.append(j * zi * zj)
            if ((bits >> i) & 1) != ((bits >> js) & 1):
                fl = bits ^ (1 << i) ^ (1 << js)
                if fl in idx_map:
                    rows.append(row); cols.append(idx_map[fl]); vals.append(0.5 * j)
    return (jnp.array(rows, dtype=jnp.int32),
            jnp.array(cols, dtype=jnp.int32),
            jnp.array(vals, dtype=jnp.float64))


def make_apply_H(h_rows, h_cols, h_vals, dim):
    @jax.jit
    def apply_H(psi):
        return jnp.zeros(dim, dtype=psi.dtype).at[h_rows].add(h_vals * psi[h_cols])
    return apply_H


def neel_state(n, n_up, basis, idx_map):
    neel_bits = sum(1 << i for i in range(n) if i % 2 == 0)
    psi = jnp.zeros(len(basis), dtype=jnp.complex128)
    return psi.at[idx_map[neel_bits]].set(1.0)


def build_jastrow_matrix(n, basis):
    n_pair = n * (n - 1) // 2
    mat    = np.zeros((len(basis), n_pair))
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            for row, bits in enumerate(basis):
                mat[row, k] = ((bits >> i) & 1) * ((bits >> j) & 1)
            k += 1
    return jnp.array(mat, dtype=jnp.float64)


def build_givens_pairs(n, basis, idx_map):
    pairs_list, srcs_ragged, dsts_ragged = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            srcs, dsts = [], []
            for row, bits in enumerate(basis):
                if (bits >> i) & 1 and not (bits >> j) & 1:
                    fl = bits ^ (1 << i) ^ (1 << j)
                    if fl in idx_map:
                        srcs.append(row)
                        dsts.append(idx_map[fl])
            pairs_list.append((jnp.array(srcs, dtype=jnp.int32),
                               jnp.array(dsts, dtype=jnp.int32)))
            srcs_ragged.append(srcs)
            dsts_ragged.append(dsts)

    max_occ    = max((len(s) for s in srcs_ragged), default=1)
    n_pair     = len(srcs_ragged)
    srcs_mat   = np.zeros((n_pair, max_occ), dtype=np.int32)
    dsts_mat   = np.zeros((n_pair, max_occ), dtype=np.int32)
    valid_mask = np.zeros((n_pair, max_occ), dtype=bool)
    for k, (s, d) in enumerate(zip(srcs_ragged, dsts_ragged)):
        srcs_mat[k, :len(s)]   = s
        dsts_mat[k, :len(d)]   = d
        valid_mask[k, :len(s)] = True

    return (pairs_list,
            jnp.array(srcs_mat),
            jnp.array(dsts_mat),
            jnp.array(valid_mask))


# =============================================================================
# RBM (NetKet)
# =============================================================================

class RBMModel(nn.Module):
    alpha: int = 1

    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.complex128)
        a = self.param('visible_bias', nn.initializers.normal(0.01),
                       (x.shape[-1],), jnp.complex128)
        W = nn.Dense(self.alpha * x.shape[-1], use_bias=True,
                     dtype=jnp.complex128, param_dtype=jnp.complex128,
                     kernel_init=nn.initializers.normal(0.01),
                     bias_init=nn.initializers.normal(0.01))
        return jnp.dot(x, a) + jnp.sum(nknn.activation.log_cosh(W(x)), axis=-1)


def run_netket_vmc(n, e_exact, total_sz=0):
    hi  = nk.hilbert.Spin(s=0.5, N=n, total_sz=total_sz)
    ha  = nk.operator.Heisenberg(hilbert=hi,
                                  graph=nk.graph.Chain(n, pbc=PBC),
                                  J=J_COUPLING / 4.0)
    sa  = nk.sampler.MetropolisExchange(hi, n_chains=16, graph=nk.graph.Chain(n))
    vs  = nk.vqs.MCState(sa, RBMModel(alpha=ALPHA),
                          n_samples=VMC_SAMPLES, seed=SEED)
    opt = optax.sgd(learning_rate=0.02)
    gs  = nk.driver.VMC_SR(hamiltonian=ha, optimizer=opt,
                             diag_shift=0.01, variational_state=vs)
    gs.run(n_iter=VMC_STEPS, out=nk.logging.RuntimeLog())
    E_rbm = float(np.real(vs.expect(ha).mean))
    print(f"[VMC]  E_rbm={E_rbm:.6f}  E_exact={e_exact:.6f}")
    return vs, ha


def extract_ucj_correlators(vs, n, basis, idx_map):

    def log_psi(sigma_batch):
        return np.array(vs.log_value(
            jnp.array(sigma_batch, dtype=jnp.float32)))

    basis_arr = np.array(
        [[(1 if (b >> s) & 1 else -1) for s in range(n)] for b in basis],
        dtype=np.float32)

    log_vals = log_psi(basis_arr)
    log_amps = log_vals - np.max(np.real(log_vals))
    psi      = np.exp(log_amps)
    probs    = np.abs(psi) ** 2
    probs   /= probs.sum()

    occ    = (basis_arr + 1) / 2
    n_mean = (probs[:, None] * occ).sum(0)

    nn_mean   = np.einsum('d,di,dj->ij', probs, occ, occ)
    C_jastrow = nn_mean - np.outer(n_mean, n_mean)

    # 1-RDM with correct Jordan-Wigner strings
    rho = np.diag(n_mean.astype(complex))
    for i in range(n):
        for j in range(i + 1, n):
            mask = (basis_arr[:, i] == 1) & (basis_arr[:, j] == -1)
            if not mask.any():
                continue
            sigma_v       = basis_arr[mask]
            sigma_f       = sigma_v.copy()
            sigma_f[:, i] = -1
            sigma_f[:, j] =  1
            ratio    = np.exp(log_psi(sigma_f) - log_psi(sigma_v))
            jw_signs = np.array([
                (-1) ** int(((sigma_v[k, i+1:j] + 1) / 2).sum())
                for k in range(sigma_v.shape[0])])
            rho_ij     = (probs[mask] * jw_signs * ratio).sum()
            rho[i, j]  = rho_ij
            rho[j, i]  = np.conj(rho_ij)

    # K_real: antisymmetric part of Re(rho)
    # Generates e^{sum_{i<j} kappa_ij (a†_i a_j - a†_j a_i)}
    # Valid seed when Re(rho_ij) != 0, i.e. real hopping is present.
    rho_real_antisym = 0.5 * (np.real(rho) - np.real(rho).T)

    K_real = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            K_real[i, j] =  rho_real_antisym[i, j]
            K_real[j, i] = -K_real[i, j]

    # K_imag: symmetric part of Im(rho)
    # Generates e^{i sum_{i<j} kappa_ij (a†_i a_j + a†_j a_i)}
    # Nonzero only when wavefunction has complex phases (frustrated systems).
    rho_imag_sym = 0.5 * (np.imag(rho) + np.imag(rho).T)

    K_imag = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            K_imag[i, j] =  rho_imag_sym[i, j]
            K_imag[j, i] =  K_imag[i, j]

    print("K_real max off-diag:", np.max(np.abs(K_real - np.diag(np.diag(K_real)))))
    print("K_imag max off-diag:", np.max(np.abs(K_imag - np.diag(np.diag(K_imag)))))

    return C_jastrow, K_real, K_imagdef extract_ucj_correlators(vs, n, basis, idx_map):

    def log_psi(sigma_batch):
        return np.array(vs.log_value(
            jnp.array(sigma_batch, dtype=jnp.float32)))

    basis_arr = np.array(
        [[(1 if (b >> s) & 1 else -1) for s in range(n)] for b in basis],
        dtype=np.float32)

    log_vals = log_psi(basis_arr)
    log_amps = log_vals - np.max(np.real(log_vals))
    psi      = np.exp(log_amps)
    probs    = np.abs(psi) ** 2
    probs   /= probs.sum()

    occ    = (basis_arr + 1) / 2
    n_mean = (probs[:, None] * occ).sum(0)

    nn_mean   = np.einsum('d,di,dj->ij', probs, occ, occ)
    C_jastrow = nn_mean - np.outer(n_mean, n_mean)

    # 1-RDM with correct Jordan-Wigner strings
    rho = np.diag(n_mean.astype(complex))
    for i in range(n):
        for j in range(i + 1, n):
            mask = (basis_arr[:, i] == 1) & (basis_arr[:, j] == -1)
            if not mask.any():
                continue
            sigma_v       = basis_arr[mask]
            sigma_f       = sigma_v.copy()
            sigma_f[:, i] = -1
            sigma_f[:, j] =  1
            ratio    = np.exp(log_psi(sigma_f) - log_psi(sigma_v))
            jw_signs = np.array([
                (-1) ** int(((sigma_v[k, i+1:j] + 1) / 2).sum())
                for k in range(sigma_v.shape[0])])
            rho_ij     = (probs[mask] * jw_signs * ratio).sum()
            rho[i, j]  = rho_ij
            rho[j, i]  = np.conj(rho_ij)

    # K_real: antisymmetric part of Re(rho)
    # Generates e^{sum_{i<j} kappa_ij (a†_i a_j - a†_j a_i)}
    # Valid seed when Re(rho_ij) != 0, i.e. real hopping is present.
    rho_real_antisym = 0.5 * (np.real(rho) - np.real(rho).T)

    K_real = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            K_real[i, j] =  rho_real_antisym[i, j]
            K_real[j, i] = -K_real[i, j]

    # K_imag: symmetric part of Im(rho)
    # Generates e^{i sum_{i<j} kappa_ij (a†_i a_j + a†_j a_i)}
    # Nonzero only when wavefunction has complex phases (frustrated systems).
    rho_imag_sym = 0.5 * (np.imag(rho) + np.imag(rho).T)

    K_imag = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            K_imag[i, j] =  rho_imag_sym[i, j]
            K_imag[j, i] =  K_imag[i, j]

    print("K_real max off-diag:", np.max(np.abs(K_real - np.diag(np.diag(K_real)))))
    print("K_imag max off-diag:", np.max(np.abs(K_imag - np.diag(np.diag(K_imag)))))

    return C_jastrow, K_real, K_imag


'''
def extract_ucj_correlators(vs, n, basis, idx_map):
    def log_psi(sigma_batch):
        return np.array(vs.log_value(
            jnp.array(sigma_batch, dtype=jnp.float32)))

    basis_arr = np.array(
        [[(1 if (b >> s) & 1 else -1) for s in range(n)] for b in basis],
        dtype=np.float32)

    log_vals = log_psi(basis_arr)
    log_amps = log_vals - np.max(np.real(log_vals))
    psi      = np.exp(log_amps)
    probs    = np.abs(psi) ** 2
    probs   /= probs.sum()

    occ    = (basis_arr + 1) / 2
    n_mean = (probs[:, None] * occ).sum(0)

    # Physical density-density correlator — NOT directly theta_J
    nn_mean   = np.einsum('d,di,dj->ij', probs, occ, occ)
    C_jastrow = nn_mean - np.outer(n_mean, n_mean)

    # 1-RDM with correct Jordan-Wigner strings
    rho = np.diag(n_mean.astype(complex))
    for i in range(n):
        for j in range(i + 1, n):
            mask = (basis_arr[:, i] == 1) & (basis_arr[:, j] == -1)
            if not mask.any():
                continue
            sigma_v    = basis_arr[mask]
            sigma_f    = sigma_v.copy()
            sigma_f[:, i] = -1
            sigma_f[:, j] =  1
            ratio    = np.exp(log_psi(sigma_f) - log_psi(sigma_v))
            jw_signs = np.array([
                (-1) ** int(((sigma_v[k, i+1:j] + 1) / 2).sum())
                for k in range(sigma_v.shape[0])])
            rho_ij      = (probs[mask] * jw_signs * ratio).sum()
            rho[i, j]   = rho_ij
            rho[j, i]   = np.conj(rho_ij)


    from scipy.linalg import logm

    eigvals, U = np.linalg.eigh(rho)
    K_full = -1j * logm(U)
    K_full = 0.5 * (K_full - K_full.conj().T)

    K_real = np.zeros((n, n), dtype=np.float64)
    K_imag = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            K_real[i, j] =  np.real(K_full[i, j])
            K_real[j, i] = -K_real[i, j]
            K_imag[i, j] =  np.imag(K_full[i, j])
            K_imag[j, i] =  K_imag[i, j]

    return C_jastrow, K_real, K_imag
'''
# =============================================================================
# JAX WAVEFUNCTION OPS
# =============================================================================

def apply_jastrow(psi, theta_J, nn_mat):
    return psi * jnp.exp(1j * (nn_mat @ theta_J))


def _givens_scan(psi, thetas, srcs_mat, dsts_mat, valid_mask, imag=False):
    def _step(psi, args):
        theta_k, srcs_k, dsts_k, mask_k = args
        c, s = jnp.cos(theta_k), jnp.sin(theta_k)
        p_s, p_d = psi[srcs_k], psi[dsts_k]
        if imag:
            new_s = jnp.where(mask_k, c * p_s - 1j * s * p_d, p_s)
            new_d = jnp.where(mask_k, c * p_d - 1j * s * p_s, p_d)
        else:
            new_s = jnp.where(mask_k, c * p_s - s * p_d, p_s)
            new_d = jnp.where(mask_k, s * p_s + c * p_d, p_d)
        return psi.at[srcs_k].set(new_s).at[dsts_k].set(new_d), None

    psi, _ = jax.lax.scan(_step, psi, (thetas, srcs_mat, dsts_mat, valid_mask))
    return psi


def real_gauge_project(psi):
    idx = jnp.argmax(jnp.abs(psi))
    return psi * jnp.conj(psi[idx] / jnp.abs(psi[idx]))


# =============================================================================
# ANSATZ STATE BUILDERS
# =============================================================================

def ucj_state_re(theta, k_layers, psi0, n_pair, srcs_mat, dsts_mat, valid_mask, nn_mat):
    psi = psi0
    for layer in range(k_layers):
        off = layer * 2 * n_pair
        psi = apply_jastrow(psi, theta[off : off + n_pair], nn_mat)
        psi = _givens_scan(psi, theta[off + n_pair : off + 2*n_pair],
                           srcs_mat, dsts_mat, valid_mask)
    return psi


def ucj_state_g(theta, k_layers, psi0, n_pair, srcs_mat, dsts_mat, valid_mask, nn_mat,
                real_gauge=True):
    psi = psi0
    for layer in range(k_layers):
        off = layer * 3 * n_pair
        psi = apply_jastrow(psi, theta[off : off + n_pair], nn_mat)
        psi = _givens_scan(psi, theta[off + n_pair   : off + 2*n_pair],
                           srcs_mat, dsts_mat, valid_mask)
        psi = _givens_scan(psi, theta[off + 2*n_pair : off + 3*n_pair],
                           srcs_mat, dsts_mat, valid_mask, imag=True)
        if real_gauge:
            psi = real_gauge_project(psi)
    return psi


def ucj_state_im(theta, k_layers, psi0, n_pair, srcs_mat, dsts_mat, valid_mask, nn_mat):
    """
    Im-uCJ: imaginary orbital rotations only (K purely imaginary).
    Same parameter count as Re-uCJ (2*n_pair per layer: Jastrow + imag Givens).
    Circuit: cp layer + XXPlusYY(beta=0) layer — same gate count as Re-uCJ.
    From Tkachenko et al. 2025: Im-uCJ is strictly more accurate than Re-uCJ
    at the same circuit cost and is far more stable in optimisation.
    """
    psi = psi0
    for layer in range(k_layers):
        off = layer * 2 * n_pair
        psi = apply_jastrow(psi, theta[off : off + n_pair], nn_mat)
        psi = _givens_scan(psi, theta[off + n_pair : off + 2*n_pair],
                           srcs_mat, dsts_mat, valid_mask, imag=True)
    return psi


def _energy(psi, apply_H):
    norm = jnp.dot(jnp.conj(psi), psi)
    return jnp.real(jnp.dot(jnp.conj(psi), apply_H(psi)) / norm)


def make_energy_grad(variant, n, k_layers, psi0, srcs_mat, dsts_mat, valid_mask,
                     nn_mat, apply_H, real_gauge=True):
    n_pair = n * (n - 1) // 2
    if variant == 're':
        state_fn = functools.partial(ucj_state_re, k_layers=k_layers, psi0=psi0,
                                     n_pair=n_pair, srcs_mat=srcs_mat, dsts_mat=dsts_mat,
                                     valid_mask=valid_mask, nn_mat=nn_mat)
    elif variant == 'im':
        state_fn = functools.partial(ucj_state_im, k_layers=k_layers, psi0=psi0,
                                     n_pair=n_pair, srcs_mat=srcs_mat, dsts_mat=dsts_mat,
                                     valid_mask=valid_mask, nn_mat=nn_mat)
    else:
        state_fn = functools.partial(ucj_state_g, k_layers=k_layers, psi0=psi0,
                                     n_pair=n_pair, srcs_mat=srcs_mat, dsts_mat=dsts_mat,
                                     valid_mask=valid_mask, nn_mat=nn_mat, real_gauge=real_gauge)

    def efn(theta):
        return _energy(state_fn(theta), apply_H)

    return jax.jit(jax.value_and_grad(efn)), jax.jit(efn)


def fidelity(theta, variant, k_layers, psi0, psi_exact, srcs_mat, dsts_mat,
             valid_mask, nn_mat, n, real_gauge=True):
    n_pair = n * (n - 1) // 2
    if variant == 're':
        psi = ucj_state_re(theta, k_layers, psi0, n_pair, srcs_mat, dsts_mat, valid_mask, nn_mat)
    elif variant == 'im':
        psi = ucj_state_im(theta, k_layers, psi0, n_pair, srcs_mat, dsts_mat, valid_mask, nn_mat)
    else:
        psi = ucj_state_g(theta, k_layers, psi0, n_pair, srcs_mat, dsts_mat, valid_mask,
                          nn_mat, real_gauge=real_gauge)
    return float(jnp.abs(jnp.dot(jnp.conj(psi_exact.astype(jnp.complex128)), psi)) ** 2)


# =============================================================================
# WARM-START BUILDERS
# =============================================================================

def _upper_flat(mat, n):
    return np.array([mat[i, j] for i in range(n) for j in range(i+1, n)])


def warm_start(variant, C_jastrow, n, k_layers, K_real=None, K_imag=None,
               seed=SEED, noise_scale=0.01):
    n_pair    = n * (n - 1) // 2
    rng       = np.random.default_rng(seed)
    J_flat    = _upper_flat(C_jastrow, n)
    K_re_flat = _upper_flat(K_real, n) if K_real is not None else np.zeros(n_pair)
    K_im_flat = _upper_flat(K_imag, n) if K_imag is not None else np.zeros(n_pair)

    out = []
    for layer in range(k_layers):
        fresh = (layer == 0)
        out.append(J_flat    + noise_scale * rng.standard_normal(n_pair) if fresh
                   else       noise_scale * rng.standard_normal(n_pair))
        if variant == 'im':
            out.append(K_im_flat + noise_scale * rng.standard_normal(n_pair) if fresh
                       else        noise_scale * rng.standard_normal(n_pair))
        elif variant == 're':
            out.append(K_re_flat + noise_scale * rng.standard_normal(n_pair) if fresh
                       else        noise_scale * rng.standard_normal(n_pair))
        else:
            out.append(K_re_flat + noise_scale * rng.standard_normal(n_pair) if fresh
                       else        noise_scale * rng.standard_normal(n_pair))
            out.append(K_im_flat + noise_scale * rng.standard_normal(n_pair) if fresh
                       else        noise_scale * rng.standard_normal(n_pair))

    return np.concatenate(out)


# =============================================================================
# L-BFGS LAYER OPTIMISER
# =============================================================================

def optimize_layer(variant, n, k, x0, e_exact, psi_neel, psi_exact,
                   srcs_mat, dsts_mat, valid_mask, nn_mat, apply_H,
                   real_gauge=True):
    val_grad_fn, energy_fn = make_energy_grad(
        variant, n, k, psi_neel, srcs_mat, dsts_mat, valid_mask, nn_mat, apply_H, real_gauge)
    val_grad_fn(jnp.array(x0, dtype=jnp.float64))

    best_E = [np.inf]

    def scipy_fn(x_np):
        E, g = val_grad_fn(jnp.array(x_np, dtype=jnp.float64))
        E_f  = float(E)
        if E_f < best_E[0]:
            best_E[0] = E_f
        return E_f, np.array(g, dtype=np.float64)

    result = scipy_minimize(scipy_fn, x0, jac=True, method='L-BFGS-B',
                            options={'maxiter': LBFGS_MAXITER, 'maxfun': LBFGS_MAXFUN,
                                     'ftol': 1e-14, 'gtol': 1e-8})
    opt_x = np.array(result.x)
    opt_E = float(result.fun)
    fid   = fidelity(jnp.array(opt_x), variant, k, psi_neel, psi_exact,
                     srcs_mat, dsts_mat, valid_mask, nn_mat, n, real_gauge)
    print(f"  [{variant}-uCJ k={k}]  E={opt_E:.8f}  |<exact|uCJ>|²={fid:.6f}"
          f"  nit={result.nit}  nfev={result.nfev}")
    return opt_x, opt_E, fid


# =============================================================================
# ADAPTIVE LAYER SEARCH
# =============================================================================

def adaptive_ucj(variant, n, k_max, e_tol, C_jastrow, e_exact, psi_neel, psi_exact,
                 srcs_mat, dsts_mat, valid_mask, nn_mat, apply_H,
                 K_real=None, K_imag=None, real_gauge=True, n_restarts=N_RESTARTS):
    best        = dict(E=np.inf, fid=0., params=None, k=1)
    prev_params = None
    n_pair      = n * (n - 1) // 2

    for k in range(1, k_max + 1):
        layer_best = dict(E=np.inf, params=None, fid=0.)

        for restart in range(n_restarts):
            rng = np.random.default_rng(SEED + k * 100 + restart)

            if prev_params is None:
                # k=1 or no prior: full RBM-informed warm start
                x0 = warm_start(variant, C_jastrow, n, k, K_real, K_imag,
                                seed=SEED + restart, noise_scale=0.01)
            else:
                # Extend best params from previous layer with a new layer appended
                if variant == 'g':
                    new_layer = np.concatenate([
                        _upper_flat(C_jastrow, n) + 0.01 * rng.standard_normal(n_pair),
                        0.05 * rng.standard_normal(n_pair),
                        0.05 * rng.standard_normal(n_pair)])
                else:
                    new_layer = np.concatenate([
                        _upper_flat(C_jastrow, n) + 0.01 * rng.standard_normal(n_pair),
                        0.05 * rng.standard_normal(n_pair)])
                x0 = np.concatenate([prev_params, new_layer])

            opt_x, opt_E, fid = optimize_layer(
                variant, n, k, x0, e_exact, psi_neel, psi_exact,
                srcs_mat, dsts_mat, valid_mask, nn_mat, apply_H, real_gauge)

            if opt_E < layer_best['E']:
                layer_best.update(E=opt_E, params=opt_x, fid=fid)
            if abs(opt_E - e_exact) < e_tol:
                break

        # Always carry forward the best found params for next layer extension
        prev_params = layer_best['params']
        if layer_best['E'] < best['E']:
            best.update(**layer_best, k=k)

        delta = abs(layer_best['E'] - e_exact)
        print(f"[{variant}-uCJ k={k}]  E={layer_best['E']:.8f}  |ΔE|={delta:.4e}")
        if delta < e_tol:
            print(f"  Converged at k={k}.")
            break

    return best


# =============================================================================
# QISKIT CIRCUIT BUILDER
# =============================================================================

def nn_pairs(n, pbc=False):
    pairs = [(i, i+1) for i in range(n-1)]
    if pbc:
        pairs.append((0, n-1))
    return pairs


def build_ucj_circuit(n, k_layers, params, variant='re', pairs=None):
    if pairs is None:
        pairs = [(i, j) for i in range(n) for j in range(i+1, n)]
    n_pair = len(pairs)
    stride = 3 * n_pair if variant == 'g' else 2 * n_pair
    qreg   = QuantumRegister(n, 'q')
    qc     = QuantumCircuit(qreg)
    for i in range(n):
        if i % 2 == 0:
            qc.x(qreg[i])

    for l in range(k_layers):
        off  = l * stride
        tJ   = params[off          : off + n_pair]
        tK_r = params[off + n_pair : off + 2*n_pair]
        tK_i = params[off + 2*n_pair : off + 3*n_pair] if variant == 'g' else None

        for k, (i, j) in enumerate(pairs):
            qc.cp(float(tJ[k]), qreg[i], qreg[j])

        if variant == 'im':
            for k, (i, j) in enumerate(pairs):
                qc.append(XXPlusYYGate(2*float(tK_r[k]), beta=0.0),
                          [qreg[j], qreg[i]])
        else:
            for k, (i, j) in enumerate(pairs):
                qc.append(XXPlusYYGate(2*float(tK_r[k]), beta=-np.pi/2),
                          [qreg[j], qreg[i]])
            if variant == 'g':
                for k, (i, j) in enumerate(pairs):
                    qc.append(XXPlusYYGate(2*float(tK_i[k]), beta=0.0),
                              [qreg[j], qreg[i]])
    return qc


# =============================================================================
# PAULI HAMILTONIAN & BACKEND
# =============================================================================

def build_heisenberg_pauli(n, j=J_COUPLING, pbc=PBC):
    edges = [(i, (i+1)%n) for i in range(n)] if pbc else [(i, i+1) for i in range(n-1)]
    ops = []
    for si, sj in edges:
        for p in ('X', 'Y', 'Z'):
            lbl = ['I']*n; lbl[si] = p; lbl[sj] = p
            ops.append((''.join(reversed(lbl)), j/4.0))
    return SparsePauliOp.from_list(ops)

'''
def build_backend(n):

    edges = [[i, j] for i in range(n) for j in range(n) if i != j]

    fake = GenericBackendV2(
        num_qubits   = n,
        basis_gates  = ['rxx', 'ryy', 'cp', 'rz', 'sx', 'x', 'measure'],
        coupling_map = edges,
        seed         = 42,
    )

    nm = NoiseModel(basis_gates=['rxx', 'ryy', 'cp', 'rz', 'sx', 'x', 'measure'])

    err_1q = depolarizing_error(1e-4, 1)   # was 1e-3
    err_2q = depolarizing_error(1e-3, 2)   # was 5e-3

    for q in range(n):
        nm.add_quantum_error(err_1q, ['sx', 'x'], [q])

    for i in range(n):
        for j in range(n):
            if i != j:
                nm.add_quantum_error(err_2q, ['rxx', 'ryy', 'cp'], [i, j])

    aer = AerSimulator(noise_model=nm)
    return fake, aer, nm
'''
def build_backend(n):
    from qiskit_aer.noise import (
        thermal_relaxation_error, phase_damping_error,
        ReadoutError
    )

    edges = [[i, j] for i in range(n) for j in range(n) if i != j]

    fake = GenericBackendV2(
        num_qubits   = n,
        basis_gates  = ['rxx', 'ryy', 'cp', 'rz', 'sx', 'x', 'measure'],
        coupling_map = edges,
        seed         = 42,
    )

    # ------------------------------------------------------------------
    # Gate times (nanoseconds) — based on typical superconducting hardware
    # IBM Falcon / Eagle class devices
    # ------------------------------------------------------------------
    T1      = 100e3   # ns  (100 µs)
    T2      = 80e3    # ns  (80 µs  — must be <= 2*T1)
    t_sx    = 32      # ns  single-qubit gate time
    t_x     = 32      # ns
    t_rz    = 0       # ns  virtual Z gate — zero error
    t_2q    = 400     # ns  two-qubit gate time (rxx/ryy/cp after decomposition)
    t_meas  = 1000    # ns  measurement time

    # ------------------------------------------------------------------
    # Build noise model
    # ------------------------------------------------------------------
    nm = NoiseModel(basis_gates=['rxx', 'ryy', 'cp', 'rz', 'sx', 'x', 'measure'])

    # --- single-qubit gates: thermal relaxation + depolarizing ---
    # Thermal relaxation captures T1 (energy decay) and T2 (dephasing)
    # Depolarizing on top captures coherent/control errors
    err_1q_dep = depolarizing_error(1e-4, 1)
    for q in range(n):
        err_1q_th = thermal_relaxation_error(T1, T2, t_sx)
        err_sx    = err_1q_dep.compose(err_1q_th)
        err_x     = depolarizing_error(1e-4, 1).compose(
                        thermal_relaxation_error(T1, T2, t_x))
        nm.add_quantum_error(err_sx, ['sx'], [q])
        nm.add_quantum_error(err_x,  ['x'],  [q])

    # rz is a virtual gate (frame change) — no physical error
    # do not add noise to rz

    # --- two-qubit gates: thermal relaxation on both qubits + depolarizing ---
    err_2q_dep = depolarizing_error(1e-3, 2)
    for i in range(n):
        for j in range(n):
            if i != j:
                # thermal relaxation on each qubit independently during gate
                err_th_i  = thermal_relaxation_error(T1, T2, t_2q)
                err_th_j  = thermal_relaxation_error(T1, T2, t_2q)
                err_th_2q = err_th_i.expand(err_th_j)
                err_2q    = err_2q_dep.compose(err_th_2q)
                nm.add_quantum_error(err_2q, ['rxx', 'ryy', 'cp'], [i, j])

    # --- readout error — asymmetric (0→1 mislabel rarer than 1→0) ---
    # p(1|0): probability of measuring 1 when state is |0>
    # p(0|1): probability of measuring 0 when state is |1>
    p_meas_01 = 0.01   # 1% false positive
    p_meas_10 = 0.03   # 3% false negative (T1 decay during measurement)
    ro_error  = ReadoutError([
        [1 - p_meas_01,     p_meas_01],   # prepared |0>
        [p_meas_10,     1 - p_meas_10],   # prepared |1>
    ])
    for q in range(n):
        nm.add_readout_error(ro_error, [q])


    '''
    aer = AerSimulator(noise_model=nm)
    return fake, aer, nm
    '''
    method = 'matrix_product_state' if n >= 12 else 'automatic'
    aer = AerSimulator(method=method, noise_model=nm)
    return fake, aer, nm
    

# =============================================================================
# SUBGRAPH SELECTION outdated, was used for fakebrisbane
# =============================================================================

def get_best_subgraph(fake_backend, n):
    # GenericBackendV2 is already n-qubit all-to-all — no selection needed
    if not hasattr(fake_backend, 'properties'):
        best_qubits = list(range(n))
        print(f"  [Layout]  GenericBackendV2 — using qubits {best_qubits} (all-to-all)")
        return best_qubits




def build_subgraph_target(fake_backend, best_qubits):
    # GenericBackendV2 exposes a Target directly — just return it
    if not hasattr(fake_backend, 'properties'):
        print(f"  [Target]  GenericBackendV2 — using backend target directly")
        return fake_backend.target



# =============================================================================
# SHOTS FROM COMMUTING GROUPS
# =============================================================================

def get_shots_from_groups(groups, n_qubits, epsilon=SHOTS_EPSILON):
    n_groups        = len(groups)
    group_variances = [sum(coeff**2 for _, coeff in g['terms']) for g in groups]
    shots_needed    = max(var / (epsilon ** 2) for var in group_variances)
    shots           = int(np.clip(shots_needed, SHOTS_MIN, SHOTS_MAX))
    print(f"[Shots]  N={n_qubits}  n_groups={n_groups}"
          f"  epsilon={epsilon}"
          f"  group_variances={[f'{v:.4f}' for v in group_variances]}"
          f"  shots_needed={shots_needed:.0f}  shots={shots}")
    return shots


# =============================================================================
# CLIFFORD DIAGONALISATION
# =============================================================================

def find_diagonalizing_clifford(paulis):
    from qiskit.quantum_info import PauliList, Clifford
    from numpy import array

    num_paulis = len(paulis)
    num_qubits = len(paulis[0])
    paulis     = PauliList(list(str(pauli) for pauli in paulis))
    qc         = QuantumCircuit(num_qubits)
    X = array(list(p.x for p in paulis), dtype=int)
    Z = array(list(p.z for p in paulis), dtype=int)

    for q in range(num_qubits):
        if 1 not in X[:,q]:
            continue
        elif (1 in X[:,q] and 1 not in Z[:,q]):
            qc.h(q)
            new_z = X[:,q].copy(); new_x = Z[:,q].copy()
            Z[:,q] = new_z; X[:,q] = new_x
            continue
        elif all(X[:,q] == Z[:,q]):
            qc.sdg(q); qc.h(q)
            new_z = X[:,q].copy(); new_x = Z[:,q].copy() ^ X[:,q].copy()
            Z[:,q] = new_z; X[:,q] = new_x
            continue
        for r in range(q+1, num_qubits):
            if all(X[:,r] == X[:,q]):
                qc.cx(r, q)
                new_z = Z[:,q].copy() ^ Z[:,r].copy()
                new_x = X[:,q].copy() ^ X[:,r].copy()
                Z[:,r] = new_z; X[:,q] = new_x
                break
            elif all(Z[:,r] == Z[:,q]):
                qc.cx(q, r)
                new_z = Z[:,q].copy() ^ Z[:,r].copy()
                new_x = X[:,q].copy() ^ X[:,r].copy()
                Z[:,q] = new_z; X[:,r] = new_x
                qc.h(q)
                new_z = X[:,q].copy(); new_x = Z[:,q].copy()
                Z[:,q] = new_z; X[:,q] = new_x
                break
            elif all(Z[:,r] ^ Z[:,q] == X[:,q]):
                qc.cx(q, r)
                new_z = Z[:,q].copy() ^ Z[:,r].copy()
                new_x = X[:,q].copy() ^ X[:,r].copy()
                Z[:,q] = new_z; X[:,r] = new_x
                qc.sdg(q); qc.h(q)
                new_z = X[:,q].copy(); new_x = Z[:,q].copy() ^ X[:,q].copy()
                Z[:,q] = new_z; X[:,q] = new_x
                break
            elif all(Z[:,q] == X[:,r]):
                qc.cz(r, q)
                new_zr = Z[:,r].copy() ^ X[:,q].copy()
                new_zq = Z[:,q].copy() ^ X[:,r].copy()
                Z[:,r] = new_zr; Z[:,q] = new_zq
                qc.h(q)
                new_z = X[:,q].copy(); new_x = Z[:,q].copy()
                Z[:,q] = new_z; X[:,q] = new_x
                break
            if r == num_qubits - 1:
                for r1 in range(q+1, num_qubits):
                    for r2 in range(num_qubits):
                        if (all(X[:,q] == X[:,r1] ^ X[:,r2]) and r1 != q and r2 != r1):
                            qc.cx(r2, r1)
                            new_z = Z[:,r1].copy() ^ Z[:,r2].copy()
                            new_x = X[:,r1].copy() ^ X[:,r2].copy()
                            Z[:,r2] = new_z; X[:,r1] = new_x
                            qc.cx(r1, q)
                            new_z = Z[:,q].copy() ^ Z[:,r1].copy()
                            new_x = X[:,q].copy() ^ X[:,r1].copy()
                            Z[:,r1] = new_z; X[:,q] = new_x
                            break
                        if (all(Z[:,q] == Z[:,r1] ^ Z[:,r2]) and r1 != q and r2 != r1):
                            qc.cx(r2, r1)
                            new_z = Z[:,r2].copy() ^ Z[:,r1].copy()
                            new_x = X[:,r2].copy() ^ X[:,r1].copy()
                            Z[:,r2] = new_z; X[:,r1] = new_x
                            if r2 != q: qc.cx(r2, q)
                            new_z = Z[:,q].copy() ^ Z[:,r2].copy()
                            new_x = X[:,q].copy() ^ X[:,r2].copy()
                            Z[:,r2] = new_z; X[:,q] = new_x
                            qc.h(q)
                            new_z = X[:,q].copy(); new_x = Z[:,q].copy()
                            Z[:,q] = new_z; X[:,q] = new_x
                            break
                        if (all(Z[:,q] ^ (Z[:,r1] ^ Z[:,r2]) == X[:,q])
                                and r1 != q and r2 != r1 and r2 > q):
                            qc.cx(r2, r1)
                            new_z = Z[:,r2].copy() ^ Z[:,r1].copy()
                            new_x = X[:,r2].copy() ^ X[:,r1].copy()
                            Z[:,r2] = new_z; X[:,r1] = new_x
                            if r2 != q: qc.cx(q, r2)
                            new_z = Z[:,q].copy() ^ Z[:,r2].copy()
                            new_x = X[:,q].copy() ^ X[:,r2].copy()
                            Z[:,q] = new_z; X[:,r2] = new_x
                            qc.sdg(q); qc.h(q)
                            new_z = X[:,q].copy(); new_x = Z[:,q].copy() ^ X[:,q].copy()
                            Z[:,q] = new_z; X[:,q] = new_x
                            break
                        if (all(Z[:,q] == X[:,r1] ^ X[:,r2]) and r1 != q and r2 != r1):
                            qc.cx(r2, r1)
                            new_z = Z[:,r1].copy() ^ Z[:,r2].copy()
                            new_x = X[:,r1].copy() ^ X[:,r2].copy()
                            Z[:,r2] = new_z; X[:,r1] = new_x
                            qc.cz(r1, q)
                            new_zr = Z[:,r1].copy() ^ X[:,q].copy()
                            new_zq = Z[:,q].copy() ^ X[:,r1].copy()
                            Z[:,r1] = new_zr; Z[:,q] = new_zq
                            qc.h(q)
                            new_z = X[:,q].copy(); new_x = Z[:,q].copy()
                            Z[:,q] = new_z; X[:,q] = new_x
                            break
                        if (all(Z[:,q] ^ (X[:,r1] ^ X[:,r2]) == X[:,q])
                                and r1 != q and r2 != r1):
                            qc.cx(r2, r1)
                            new_z = Z[:,r1].copy() ^ Z[:,r2].copy()
                            new_x = X[:,r1].copy() ^ X[:,r2].copy()
                            Z[:,r2] = new_z; X[:,r1] = new_x
                            qc.cz(r1, q)
                            new_zr = Z[:,r1].copy() ^ X[:,q].copy()
                            new_zq = Z[:,q].copy() ^ X[:,r1].copy()
                            Z[:,r1] = new_zr; Z[:,q] = new_zq
                            qc.sdg(q); qc.h(q)
                            new_z = X[:,q].copy(); new_x = Z[:,q].copy() ^ X[:,q].copy()
                            Z[:,q] = new_z; X[:,q] = new_x
                            break
                    if 1 not in X[:,q]: break

    cliff      = Clifford(qc)
    paulis     = PauliList(list(pauli for pauli in paulis))
    new_paulis = ''.join(s for s in list(
        str(pauli) for sublist in paulis.evolve(cliff, frame='s')
        for pauli in sublist))
    if 'X' not in new_paulis and 'Y' not in new_paulis:
        return qc

    qc        = QuantumCircuit(num_qubits)
    X = array(list(p.x for p in paulis), dtype=int)
    Z = array(list(p.z for p in paulis), dtype=int)
    pivot_row = 0

    for q in range(num_qubits):
        if 1 not in X[:,q]:
            continue
        for s in range(num_paulis):
            if X[s, q] == 1:
                new_z = Z[pivot_row,:].copy(); new_x = X[pivot_row,:].copy()
                Z[pivot_row,:] = Z[s,:]; Z[s,:] = new_z
                X[pivot_row,:] = X[s,:]; X[s,:] = new_x
                break
        for r in range(num_qubits):
            if r == q: continue
            if X[pivot_row, r] == 1:
                qc.cx(q, r)
                new_z = Z[:,q].copy() ^ Z[:,r].copy()
                new_x = X[:,q].copy() ^ X[:,r].copy()
                Z[:,q] = new_z; X[:,r] = new_x
        if Z[pivot_row, q] == 0:
            qc.h(q)
            new_z = X[:,q].copy(); new_x = Z[:,q].copy()
            Z[:,q] = new_z; X[:,q] = new_x
        else:
            qc.sdg(q); qc.h(q)
            new_z = X[:,q].copy(); new_x = Z[:,q].copy() ^ X[:,q].copy()
            Z[:,q] = new_z; X[:,q] = new_x
        pivot_row += 1

    return qc

'''
def _commuting_groups(H_pauli):
    from qiskit.quantum_info import PauliList, Clifford

    n         = H_pauli.num_qubits
    groups_qk = H_pauli.group_commuting(qubit_wise=False)  # restore
    result    = []

    for group in groups_qk:
        terms  = [(p, float(np.real(c)))
                  for p, c in zip(group.paulis, group.coeffs)]
        paulis = [p for p, _ in terms]

        diag_qc = find_diagonalizing_clifford(paulis)  # restore

        cliff   = Clifford(diag_qc)
        evolved = PauliList([str(p) for p in paulis]).evolve(cliff, frame='s')

        z_signs = []
        for ep in evolved:
            label = ep.to_label().lstrip('+-')   # strip global phase prefix
            assert all(c in ('I', 'i', 'Z', 'z') for c in label), \
                f"Diagonalization incomplete: {label}"
            mask = np.array([1 if label[n - 1 - q] in ('Z', 'z') else 0
                            for q in range(n)], dtype=np.int8)
            z_signs.append(mask)

        result.append(dict(terms=terms, diag_qc=diag_qc, z_signs=z_signs))

    return result

'''
def _commuting_groups(H_pauli):
    from qiskit.quantum_info import Clifford, PauliList

    n         = H_pauli.num_qubits
    groups_qk = H_pauli.group_commuting(qubit_wise=True)
    result    = []

    for group in groups_qk:
        terms = [(p, float(np.real(c)))
                 for p, c in zip(group.paulis, group.coeffs)]

        # Find the non-identity Pauli type at each qubit across all terms.
        # qubit_wise=True guarantees at most one non-identity type per qubit.
        qubit_type = ['I'] * n
        for p in group.paulis:
            label = p.to_label()          # rightmost char = qubit 0
            for q in range(n):
                c = label[n - 1 - q]
                if c not in ('I', 'i'):
                    qubit_type[q] = c.upper()

        diag_qc = QuantumCircuit(n)
        for q in range(n):
            if qubit_type[q] == 'X':
                diag_qc.h(q)
            elif qubit_type[q] == 'Y':
                diag_qc.sdg(q)
                diag_qc.h(q)

        cliff   = Clifford(diag_qc)
        evolved = PauliList([str(p) for p in group.paulis]).evolve(cliff, frame='s')

        z_signs = []
        for ep in evolved:
            label = ep.to_label()
            assert all(c in ('I', 'i', 'Z', 'z') for c in label), \
                f"Diagonalization incomplete: {label}"
            mask = np.array([1 if label[n - 1 - q] in ('Z', 'z') else 0
                             for q in range(n)], dtype=np.int8)
            z_signs.append(mask)

        result.append(dict(terms=terms, diag_qc=diag_qc, z_signs=z_signs))

    return result




def _build_diag_circuit(tmpl, diag_qc, n):
    from qiskit import ClassicalRegister
    qc   = tmpl.copy()
    qc.compose(diag_qc, inplace=True)
    creg = ClassicalRegister(n, 'c')
    qc.add_register(creg)
    for i in range(n):
        qc.measure(i, creg[i])
    return qc


def _counts_to_energy_commuting(counts, terms, z_signs, n):
    total  = sum(counts.values())
    energy = 0.0
    for bitstring, count in counts.items():
        bits = np.array([int(b) for b in reversed(bitstring)], dtype=np.int8)
        for (_, coeff), mask in zip(terms, z_signs):
            eigen   = (-1) ** int(np.dot(bits, mask) % 2)
            energy += coeff * eigen * count
    return energy / total


# =============================================================================
# CIRCUIT CACHE  (subgraph-target transpilation)
# =============================================================================





def _xy(qc, theta, q0, q1):
    """XX+YY rotation using Aer-native RXX+RYY. Equivalent to XXPlusYYGate(2θ, 0)."""
    qc.rxx(-theta, q0, q1)
    qc.ryy(-theta, q0, q1)


def _xy_phased(qc, theta, q0, q1):
    """XXPlusYYGate(2θ, beta=-π/2) = rz(q1,-π/2) + rxx + ryy + rz(q1,+π/2)"""
    qc.rz(-np.pi/2, q1)
    qc.rxx(-theta, q0, q1)
    qc.ryy(-theta, q0, q1)
    qc.rz(np.pi/2, q1)


class CircuitCache:
    def __init__(self, fake_backend, aer_backend, noise_model):
        self.fake       = fake_backend
        self.aer        = aer_backend
        self.nm         = noise_model
        self._cache     = {}
        self._m3_mit    = {}
        self._subgraphs = {}   # n → best_qubits list
        self._targets   = {}   # n → Target

    def _get_subgraph(self, n):
        if n not in self._subgraphs:
            self._subgraphs[n] = get_best_subgraph(self.fake, n)
        return self._subgraphs[n]

    def _get_target(self, n):
        """
        Return the n-qubit Target for the best subgraph.
        Transpiling with target= means the compiler sees ONLY these n qubits —
        no other qubits exist, so no long SWAP chains through the full 127-qubit
        heavy-hex graph are possible.
        """
        if n not in self._targets:
            best_qubits      = self._get_subgraph(n)
            self._targets[n] = build_subgraph_target(self.fake, best_qubits)
        return self._targets[n]

    def _get_transpiled(self, n, k_layers, variant, H_pauli):
        key = (n, k_layers, variant)
        if key in self._cache:
            return self._cache[key]

        n_pair = n * (n - 1) // 2
        stride = 3*n_pair if variant == 'g' else 2*n_pair
        pv     = ParameterVector('θ', k_layers * stride)
        qreg   = QuantumRegister(n, 'q')
        tmpl   = QuantumCircuit(qreg)
        for i in range(n):
            if i % 2 == 0: tmpl.x(qreg[i])

        for l in range(k_layers):
            off  = l * stride
            tJ   = [pv[off+k] for k in range(n_pair)]
            tK_r = [pv[off+n_pair+k] for k in range(n_pair)]
            tK_i = [pv[off+2*n_pair+k] for k in range(n_pair)] if variant == 'g' else None
            k = 0
            for i in range(n):
                for j in range(i+1, n):
                    tmpl.cp(tJ[k], qreg[i], qreg[j]); k += 1


            if variant == 'im':
                k = 0
                for i in range(n):
                    for j in range(i+1, n):
                        _xy(tmpl, tK_r[k], qreg[j], qreg[i]); k += 1


            else:

                k = 0
                for i in range(n):
                    for j in range(i+1, n):
                        _xy_phased(tmpl, tK_r[k], qreg[j], qreg[i]); k += 1


                if variant == 'g':
                    k = 0
                    for i in range(n):
                        for j in range(i+1, n):
                            _xy(tmpl, tK_i[k], qreg[j], qreg[i]); k += 1

        groups = _commuting_groups(H_pauli)
        shots  = get_shots_from_groups(groups, n)
        print(f"Num shots: {shots}")
        # Use Target instead of backend= + initial_layout= so the transpiler
        # sees an n-qubit device.  This is the key fix: with backend=FakeBrisbane,
        # the output circuit always has 127 qubits even with initial_layout set,
        # because the router can still pick SWAP paths through any of the 127
        # qubits.  With target= the universe is n qubits only.
        target = self._get_target(n)
        best_qubits = self._get_subgraph(n)

        transpiled = []
        print(f"  [CircuitCache]  {len(groups)} commuting groups "
              f"(was 3 with X/Y/Z split):")
        for gi, g in enumerate(groups):
            print(f"    group {gi}  n_terms={len(g['terms'])}")
            qc_meas = _build_diag_circuit(tmpl, g['diag_qc'], n)
            t_qc    = transpile(
                qc_meas,
                target             = target,     # n-qubit universe — key fix
                optimization_level = 3,
                seed_transpiler    = TRANSPILE_SEED,
            )
            transpiled.append((t_qc, g['terms'], g['z_signs']))

        self._cache[key] = (transpiled, pv, shots)
        return self._cache[key]

    def _get_m3(self, transpiled_circuits, n):
        if not USE_M3:
            return None

        best_qubits = self._get_subgraph(n)
        layout_key  = tuple(best_qubits)

        if layout_key not in self._m3_mit:
            print(f"  [M3]  calibrating readout mitigation  "
                  f"qubits={layout_key}  shots={M3_CAL_SHOTS}")
            mit = mthree.M3Mitigation(self.aer)
            mit.cals_from_system(list(layout_key), shots=M3_CAL_SHOTS)
            self._m3_mit[layout_key] = mit
            print(f"  [M3]  calibration done  method={M3_METHOD}")

        return self._m3_mit[layout_key], layout_key

    def energy_fn(self, n, k_layers, variant, H_pauli, shots=None):
        transpiled, pv, cached_shots = self._get_transpiled(n, k_layers, variant, H_pauli)
        if shots is None:
            shots = cached_shots

        t_circuits = [t_qc for (t_qc, _, _) in transpiled]
        m3_result  = self._get_m3(t_circuits, n) if USE_M3 else None

        def _apply_m3(counts, t_qc):
            if m3_result is None:
                return counts
            mit, layout_key = m3_result
            quasi = mit.apply_correction(counts, list(layout_key), method=M3_METHOD)
            probs = quasi.nearest_probability_distribution()
            total = sum(counts.values())
            return {bs: max(1, round(p * total)) for bs, p in probs.items() if p > 0}

        def _eval(params):
            pd             = dict(zip(pv, params.tolist()))
            bound_circuits = [t_qc.assign_parameters(pd)
                              for (t_qc, _, _) in transpiled]
            job    = self.aer.run(bound_circuits, shots=shots, noise_model=self.nm)
            result = job.result()
            E = 0.0
            for idx, (t_qc, terms, z_signs) in enumerate(transpiled):
                raw_counts       = result.get_counts(idx)
                corrected_counts = _apply_m3(raw_counts, t_qc)
                E += _counts_to_energy_commuting(corrected_counts, terms, z_signs, n)
            return E

        return _eval


# =============================================================================
# ENERGY HELPERS
# =============================================================================

def energy_noiseless(qc, H_pauli):
    return float(np.real(Statevector(qc).expectation_value(H_pauli)))


def energy_noisy(qc_bound, H_pauli, fake_backend, aer_backend, noise_model,
                 shots=None, n=None, mitigator=None):
    n      = n or qc_bound.num_qubits
    groups = _commuting_groups(H_pauli)
    if shots is None:
        shots = get_shots_from_groups(groups, n)
    E = 0.0
    for g in groups:
        qc_meas = _build_diag_circuit(qc_bound, g['diag_qc'], n)
        t_qc    = transpile(qc_meas, backend=fake_backend, optimization_level=1,
                            seed_transpiler=TRANSPILE_SEED)
        counts  = aer_backend.run(t_qc, shots=shots,
                                  noise_model=noise_model).result().get_counts()
        if USE_M3 and mitigator is not None:
            mit, layout_key = mitigator
            quasi  = mit.apply_correction(counts, list(layout_key), method=M3_METHOD)
            probs  = quasi.nearest_probability_distribution()
            total  = sum(counts.values())
            counts = {bs: max(1, round(p * total)) for bs, p in probs.items() if p > 0}
        E += _counts_to_energy_commuting(counts, g['terms'], g['z_signs'], n)
    return E


# =============================================================================
# DIAGNOSTICS
# =============================================================================

def hamiltonian_info(H_pauli):
    n_terms = len(H_pauli)
    print(f"\n[Hamiltonian]  {n_terms} terms  n_qubits={H_pauli.num_qubits}")
    for term, coeff in zip(H_pauli.paulis, H_pauli.coeffs):
        print(f"  {term.to_label()}  {float(np.real(coeff)):+.6f}")
    groups = _commuting_groups(H_pauli)
    print(f"\n  Clifford commuting groups: {len(groups)}  "
          f"(vs 3 circuits with naive X/Y/Z split)")
    for gi, g in enumerate(groups):
        print(f"  group {gi}  n_terms={len(g['terms'])}")
        for term, coeff in g['terms']:
            print(f"    {term.to_label()}  {coeff:+.6f}")
    return H_pauli


def circuit_info(qc, label='circuit', fake_backend=None, subgraph_target=None):
    """
    Report logical circuit stats, then transpiled stats.
    If subgraph_target is provided, transpile against the n-qubit Target
    (accurate reflection of what the hardware actually runs).
    Falls back to full fake_backend transpilation if no target is given.
    """
    ops   = qc.count_ops()
    depth = qc.depth()
    total = sum(ops.values())

    print(f"\n[Circuit info: {label}]")
    print(f"  qubits={qc.num_qubits}  depth={depth}  total_gates={total}")
    for gate, count in sorted(ops.items(), key=lambda x: -x[1]):
        print(f"    {gate:<20} {count}")

    result = dict(label=label, num_qubits=qc.num_qubits,
                  depth=depth, total_gates=total, ops=dict(ops),
                  transpiled=None)

    if subgraph_target is not None:
        t_qc    = transpile(qc, target=subgraph_target,
                            optimization_level=3,
                            seed_transpiler=TRANSPILE_SEED)
        t_ops   = t_qc.count_ops()
        t_depth = t_qc.depth()
        t_total = sum(t_ops.values())
        print(f"\n  [After transpilation → subgraph target ({qc.num_qubits} qubits)]")
        print(f"  qubits={t_qc.num_qubits}  depth={t_depth}  total_gates={t_total}")
        for gate, count in sorted(t_ops.items(), key=lambda x: -x[1]):
            print(f"    {gate:<20} {count}")
        result['transpiled'] = dict(num_qubits=t_qc.num_qubits,
                                    depth=t_depth, total_gates=t_total, ops=dict(t_ops))
    elif fake_backend is not None:
        t_qc    = transpile(qc, backend=fake_backend, optimization_level=1,
                            seed_transpiler=TRANSPILE_SEED)
        t_ops   = t_qc.count_ops()
        t_depth = t_qc.depth()
        t_total = sum(t_ops.values())
        print(f"\n  [After transpilation → {fake_backend.name}]")
        print(f"  qubits={t_qc.num_qubits}  depth={t_depth}  total_gates={t_total}")
        for gate, count in sorted(t_ops.items(), key=lambda x: -x[1]):
            print(f"    {gate:<20} {count}")
        result['transpiled'] = dict(num_qubits=t_qc.num_qubits,
                                    depth=t_depth, total_gates=t_total, ops=dict(t_ops))

    return result


def state_overlap(params, variant, k_layers, psi_neel, psi_exact_np,
                  srcs_mat, dsts_mat, valid_mask, nn_mat, n,
                  H_pauli=None, real_gauge=True):
    n_pair = n * (n - 1) // 2
    theta  = jnp.array(params, dtype=jnp.float64)
    psi_ex = jnp.array(psi_exact_np, dtype=jnp.complex128)

    if variant == 're':
        psi_ucj = ucj_state_re(theta, k_layers, psi_neel, n_pair,
                               srcs_mat, dsts_mat, valid_mask, nn_mat)
    elif variant == 'im':
        psi_ucj = ucj_state_im(theta, k_layers, psi_neel, n_pair,
                               srcs_mat, dsts_mat, valid_mask, nn_mat)
    else:
        psi_ucj = ucj_state_g(theta, k_layers, psi_neel, n_pair,
                              srcs_mat, dsts_mat, valid_mask, nn_mat,
                              real_gauge=real_gauge)

    psi_ucj_norm = psi_ucj / jnp.sqrt(jnp.dot(jnp.conj(psi_ucj), psi_ucj))
    overlap_jax  = float(jnp.abs(jnp.dot(jnp.conj(psi_ex), psi_ucj_norm)) ** 2)

    qc            = build_ucj_circuit(n, k_layers, params, variant)
    sv            = Statevector(qc).data
    basis_list    = [b for b in range(1 << n) if bin(b).count('1') == n // 2]
    sv_sector     = np.array([sv[b] for b in basis_list])
    sv_norm       = sv_sector / np.linalg.norm(sv_sector)
    overlap_circuit = float(np.abs(np.dot(np.conj(psi_exact_np), sv_norm)) ** 2)

    E_sv = None
    if H_pauli is not None:
        E_sv = float(np.real(Statevector(qc).expectation_value(H_pauli)))

    sv_match = abs(overlap_jax - overlap_circuit) < 1e-5

    print(f"\n[State overlap]")
    print(f"  overlap_jax     (JAX wavefunction vs Lanczos) = {overlap_jax:.8f}")
    print(f"  overlap_circuit (Qiskit SV    vs Lanczos)     = {overlap_circuit:.8f}")
    if E_sv is not None:
        print(f"  E_sv (statevector)                           = {E_sv:.8f}")
    print(f"  JAX/circuit consistent: "
          f"{'YES' if sv_match else 'NO  ← mismatch'}")

    return dict(overlap_jax=overlap_jax, overlap_circuit=overlap_circuit,
                E_sv=E_sv, sv_match=sv_match)


# =============================================================================
# HARDWARE OPTIMISERS
# =============================================================================

def cobyla_vqe(energy_fn, x0,
               rhobeg=COBYLA_RHOBEG, rhoend=COBYLA_RHOEND, maxiter=COBYLA_MAXITER):
    best = dict(E=np.inf, x=np.array(x0, dtype=float))

    def _fn(x_np):
        E = float(energy_fn(x_np))
        if E < best['E']:
            best['E'] = E; best['x'] = x_np.copy()
        return E

    maxiter = max(maxiter, 10 * len(x0))
    scipy_minimize(_fn, x0, method='COBYLA',
                   options={'maxiter': maxiter, 'rhobeg': rhobeg, 'catol': 0.0},
                   tol=rhoend)
    print(f"[COBYLA]  best_E={best['E']:.8f}")
    return best['x'], best['E']


def spsa_vqe(energy_fn, x0,
             maxiter=SPSA_MAXITER,
             a=SPSA_A, c=SPSA_C,
             alpha=SPSA_ALPHA, gamma=SPSA_GAMMA,
             tol=1e-4, patience=30):
    n_p    = len(x0)
    x      = np.array(x0, dtype=float)
    A      = 0.1 * maxiter
    best   = dict(E=np.inf, x=x.copy())
    rng    = np.random.default_rng(SEED)
    no_improve = 0

    for k in range(maxiter):
        a_k   = a / (k + 1 + A) ** alpha
        c_k   = c / (k + 1) ** gamma
        delta = rng.choice([-1.0, 1.0], size=n_p)

        E_pos = float(energy_fn(x + c_k * delta))
        E_neg = float(energy_fn(x - c_k * delta))
        grad  = (E_pos - E_neg) / (2 * c_k) * delta
        x    -= a_k * grad

        E_mid = (E_pos + E_neg) / 2
        if E_mid < best['E'] - tol:
            best['E'] = E_mid; best['x'] = x.copy()
            no_improve = 0
        else:
            no_improve += 1

        if k % 10 == 0:
            print(f"  [SPSA {k:4d}]  E_mid={E_mid:.6f}  best={best['E']:.6f}"
                  f"  a_k={a_k:.2e}  c_k={c_k:.2e}")

        if no_improve >= patience:
            print(f"  [SPSA]  early stop at k={k} "
                  f"(no improvement for {patience} steps)")
            break

    print(f"[SPSA]  best_E={best['E']:.8f}  steps={k+1}  evals={2*(k+1)}")
    return best['x'], best['E']


def qnspsa_vqe(energy_fn, x0,
               maxiter=QNSPSA_MAXITER,
               a=QNSPSA_A, c=QNSPSA_C,
               alpha=QNSPSA_ALPHA, gamma=QNSPSA_GAMMA,
               reg=QNSPSA_REGULARISATION,
               metric_shots=QNSPSA_METRIC_SHOTS,
               tol=1e-4, patience=90):
    n      = len(x0)
    x      = np.array(x0, dtype=float)
    A      = 0.1 * maxiter
    best   = dict(E=np.inf, x=x.copy())
    rng    = np.random.default_rng(SEED)
    F_avg  = np.eye(n) * reg

    for k in range(maxiter):
        a_k = a / (k + 1 + A) ** alpha
        c_k = c / (k + 1) ** gamma

        d1    = rng.choice([-1.0, 1.0], size=n)
        E_pos = float(energy_fn(x + c_k * d1))
        E_neg = float(energy_fn(x - c_k * d1))
        grad  = (E_pos - E_neg) / (2 * c_k) * d1

        F_k = np.zeros((n, n))
        for _ in range(metric_shots):
            d2     = rng.choice([-1.0, 1.0], size=n)
            f_pp   = float(energy_fn(x + c_k * d1 + c_k * d2))
            f_pm   = float(energy_fn(x + c_k * d1 - c_k * d2))
            f_mp   = float(energy_fn(x - c_k * d1 + c_k * d2))
            f_mm   = float(energy_fn(x - c_k * d1 - c_k * d2))
            hess_e = (f_pp - f_pm - f_mp + f_mm) / (4 * c_k ** 2)
            F_k   += hess_e * np.outer(d1, d2)
        F_k /= metric_shots

        w     = 1.0 / (k + 2)
        F_avg = (1 - w) * F_avg + w * (F_k + reg * np.eye(n))

        try:
            nat_grad = np.linalg.solve(F_avg, grad)
        except np.linalg.LinAlgError:
            nat_grad = grad

        x -= a_k * nat_grad
        E_cur = (E_pos + E_neg) / 2
        if E_cur < best['E']:
            best['E'] = E_cur; best['x'] = x.copy()

        if k % 10 == 0:
            print(f"  [QNSPSA {k:4d}]  E_mid={E_cur:.6f}  best={best['E']:.6f}")

    print(f"[QNSPSA]  best_E={best['E']:.8f}")
    return best['x'], best['E']


def adam_spsa_vqe(energy_fn, x0,
                  lr=ADAM_SPSA_LR,
                  beta1=ADAM_SPSA_BETA1, beta2=ADAM_SPSA_BETA2,
                  eps=ADAM_SPSA_EPS,
                  maxiter=ADAM_SPSA_MAXITER,
                  c=ADAM_SPSA_C, gamma=ADAM_SPSA_GAMMA,
                  tol=1e-4, patience=30):
    """
    SPSA gradient estimates wrapped in Adam adaptive moment estimation.
    Per-parameter variance normalisation suppresses the large step-to-step
    noise visible in vanilla SPSA on noisy hardware landscapes.
    """
    n_p  = len(x0)
    x    = np.array(x0, dtype=float)
    m    = np.zeros(n_p)
    v    = np.zeros(n_p)
    best = dict(E=np.inf, x=x.copy())
    rng  = np.random.default_rng(SEED)
    no_improve = 0

    for k in range(1, maxiter + 1):
        c_k   = c / k ** gamma
        delta = rng.choice([-1.0, 1.0], size=n_p)

        E_pos = float(energy_fn(x + c_k * delta))
        E_neg = float(energy_fn(x - c_k * delta))
        grad  = (E_pos - E_neg) / (2 * c_k) * delta

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        m_hat = m  / (1 - beta1**k)
        v_hat = v  / (1 - beta2**k)
        x    -= lr * m_hat / (np.sqrt(v_hat) + eps)

        E_mid = (E_pos + E_neg) / 2
        if E_mid < best['E'] - tol:
            best.update(E=E_mid, x=x.copy())
            no_improve = 0
        else:
            no_improve += 1

        if k % 10 == 0:
            eff_lr = lr / (np.sqrt(v_hat.mean()) + eps)
            print(f"  [AdamSPSA {k:4d}]  E_mid={E_mid:.6f}"
                  f"  best={best['E']:.6f}  lr_eff={eff_lr:.2e}")

        if no_improve >= patience:
            print(f"  [AdamSPSA]  early stop at k={k} "
                  f"(no improvement for {patience} steps)")
            break

    print(f"[AdamSPSA]  best_E={best['E']:.8f}  steps={k}  evals={2*k}")
    return best['x'], best['E']


_OPTIMIZERS = {
    'cobyla':    cobyla_vqe,
    'spsa':      spsa_vqe,
    'qnspsa':    qnspsa_vqe,
    'adam_spsa': adam_spsa_vqe,
}


# =============================================================================
# FULL BENCHMARK
# =============================================================================

VARIANTS   = ['re', 'im', 'g']
OPTIMIZERS = ['cobyla', 'spsa', 'qnspsa']


def run_all(n, variants=VARIANTS, optimizers=OPTIMIZERS,
            k_max=K_MAX, e_tol=E_TOL, run_hardware=True, timer=None):
    W = 70
    print("\n" + "=" * W)
    print(f"  run_all  |  N={n}  variants={variants}  optimizers={optimizers}")
    print("=" * W)

    n_up = n // 2

    H_sp, basis, bindex = build_hamiltonian(n, n_up)
    evals, evecs         = eigsh(H_sp, k=1, which='SA')
    e_exact              = float(evals[0])
    psi_exact_np         = evecs[:, 0] / np.linalg.norm(evecs[:, 0])
    print(f"\n[Lanczos]  N={n}  E_exact={e_exact:.8f}  E/site={e_exact/n:.8f}")

    H_pauli = build_heisenberg_pauli(n)
    hamiltonian_info(H_pauli)

    vs_rbm, _               = run_netket_vmc(n, e_exact, total_sz=0)
    theta_J, K_real, K_imag = extract_ucj_correlators(vs_rbm, n, basis, bindex)

    h_rows, h_cols, h_vals            = build_jax_hamiltonian(n, n_up)
    apply_H                           = make_apply_H(h_rows, h_cols, h_vals, len(basis))
    _, srcs_mat, dsts_mat, valid_mask  = build_givens_pairs(n, list(basis), bindex)
    nn_mat   = build_jastrow_matrix(n, list(basis))
    psi_neel = neel_state(n, n_up, list(basis), bindex)
    psi_exact = jnp.array(psi_exact_np, dtype=jnp.complex128)

    fake, aer, nm = build_backend(n)
    cache = CircuitCache(fake, aer, nm)
    # pre-warm subgraph and target so they appear in logs before JAX runs
    _ = cache._get_target(n)

    jax_results = {}

    for variant in variants:
        print(f"\n{'─'*W}")
        print(f"  JAX L-BFGS  |  {variant}-uCJ  N={n}")
        print(f"{'─'*W}")

        _jax_label = f"jax_lbfgs {variant} N={n}"
        if timer: timer.start(_jax_label)
        jax_best = adaptive_ucj(
            variant, n, k_max, e_tol, theta_J, e_exact, psi_neel, psi_exact,
            srcs_mat, dsts_mat, valid_mask, nn_mat, apply_H,
            K_real=K_real, K_imag=K_imag)
        if timer: timer.stop(_jax_label)

        params_theory = np.array(jax_best['params'])
        qc_theory     = build_ucj_circuit(n, jax_best['k'], params_theory, variant)

        _sv_label = f"sv_noiseless {variant} N={n}"
        if timer: timer.start(_sv_label)
        E_sv = energy_noiseless(qc_theory, H_pauli)
        if timer: timer.stop(_sv_label)

        print(f"\n[{variant}-uCJ]  E_theory={jax_best['E']:.8f}"
              f"  E_sv={E_sv:.8f}  k_opt={jax_best['k']}")

        # circuit_info uses the subgraph target so reported transpiled counts
        # accurately reflect what runs in the cache, not a 127-qubit circuit
        circuit_info(qc_theory,
                     label=f'{variant}-uCJ  N={n}  k={jax_best["k"]}',
                     subgraph_target=cache._get_target(n))

        ov = state_overlap(
            params_theory, variant, jax_best['k'],
            psi_neel, psi_exact_np,
            srcs_mat, dsts_mat, valid_mask, nn_mat, n,
            H_pauli=H_pauli)

        jax_results[variant] = dict(
            jax_best      = jax_best,
            params_theory = params_theory,
            qc_theory     = qc_theory,
            E_sv          = E_sv,
            overlap       = ov)

    if not run_hardware:
        _print_summary(n, e_exact, jax_results, {}, variants, optimizers)
        return dict(jax_results=jax_results, hw_results={}, summary=None)

    hw_results = {}

    for variant in variants:
        jr = jax_results[variant]
        for optimizer in optimizers:
            print(f"\n{'─'*W}")
            print(f"  Hardware  |  {variant}-uCJ  ×  {optimizer.upper()}  N={n}")
            print(f"{'─'*W}")

            efn = cache.energy_fn(n, jr['jax_best']['k'], variant, H_pauli)

            E_noisy_init = efn(jr['params_theory'])
            print(f"  E_noisy_init={E_noisy_init:.6f}"
                  f"  (noise penalty="
                  f"{E_noisy_init - jr['jax_best']['E']:+.4f})")

            opt_fn = _OPTIMIZERS[optimizer]
            _hw_label = f"hw_opt {variant} {optimizer} N={n}"
            if timer: timer.start(_hw_label)
            params_hw_opt, E_hw_opt = opt_fn(efn, x0=jr['params_theory'])
            if timer: timer.stop(_hw_label)

            qc_hw   = build_ucj_circuit(
                n, jr['jax_best']['k'], params_hw_opt, variant)
            E_hw_sv = energy_noiseless(qc_hw, H_pauli)

            ov_hw = state_overlap(
                params_hw_opt, variant, jr['jax_best']['k'],
                psi_neel, psi_exact_np,
                srcs_mat, dsts_mat, valid_mask, nn_mat, n)

            hw_results[(variant, optimizer)] = dict(
                E_noisy_init  = E_noisy_init,
                E_hw_opt      = E_hw_opt,
                E_hw_sv       = E_hw_sv,
                params_hw_opt = params_hw_opt,
                overlap_hw    = ov_hw)

    summary = _print_summary(
        n, e_exact, jax_results, hw_results, variants, optimizers)
    return dict(jax_results=jax_results, hw_results=hw_results, summary=summary)


def _print_summary(n, e_exact, jax_results, hw_results, variants, optimizers):
    n_cols = len(variants) * len(optimizers)
    col    = 10 if n_cols > 6 else 14
    W      = max(90, 28 + col * n_cols + 2)
    print(f"\n{'='*W}")
    print(f"  SUMMARY  |  N={n}  E_exact={e_exact:.6f}  "
          f"({len(variants)} variants × {len(optimizers)} optimizers"
          f" = {n_cols} circuits)")
    print(f"{'='*W}")

    row_fmt = f"  {{:<28}}" + f"{{:>{col}}}" * n_cols
    headers = [f"{v}/{o[:3]}" for v in variants for o in optimizers]
    print(row_fmt.format("metric", *headers))
    print("  " + "─" * (28 + col * n_cols))

    rows = {}

    for v in variants:
        jr = jax_results[v]
        rows[f"E_theory ({v}-uCJ)"]       = {(v, o): jr['jax_best']['E'] for o in optimizers}
        rows[f"E_sv_theory ({v}-uCJ)"]    = {(v, o): jr['E_sv'] for o in optimizers}
        rows[f"overlap_theory ({v}-uCJ)"] = {(v, o): jr['overlap']['overlap_jax']
                                              for o in optimizers}

    if hw_results:
        for v in variants:
            val = hw_results.get(
                (v, optimizers[0]), {}).get('E_noisy_init', float('nan'))
            rows[f"E_noisy_init ({v}-uCJ)"] = {(v, o): val for o in optimizers}

        rows["E_hw_opt (noisy)"] = {
            (v, o): hw_results.get((v, o), {}).get('E_hw_opt', float('nan'))
            for v in variants for o in optimizers}

        rows["E_sv_hw (noiseless)"] = {
            (v, o): hw_results.get((v, o), {}).get('E_hw_sv', float('nan'))
            for v in variants for o in optimizers}

        rows["gap (hw_opt - theory)"] = {
            (v, o): (hw_results.get((v, o), {}).get('E_hw_opt', float('nan'))
                     - jax_results[v]['jax_best']['E'])
            for v in variants for o in optimizers}

        rows["overlap_hw (JAX)"] = {
            (v, o): hw_results.get((v, o), {}).get(
                'overlap_hw', {}).get('overlap_jax', float('nan'))
            for v in variants for o in optimizers}

    print(row_fmt.format("E_exact", *[f"{e_exact:.6f}"] * n_cols))
    for metric, vals in rows.items():
        cells = [f"{vals.get((v, o), float('nan')):.6f}"
                 for v in variants for o in optimizers]
        print(row_fmt.format(metric[:28], *cells))

    print(f"{'='*W}\n")
    return rows


# =============================================================================
# M3 COMPARISON HELPERS
# =============================================================================

def _print_m3_comparison(res_on, res_off, variants, optimizers, e_exact, timer):
    n_cols  = len(variants) * len(optimizers)
    col     = 10
    W       = max(96, 30 + col * n_cols + 2)
    headers = [f"{v}/{o[:3]}" for v in variants for o in optimizers]
    row_fmt = f"  {{:<30}}" + f"{{:>{col}}}" * n_cols
    sep     = "  " + "─" * (30 + col * n_cols)

    def _get(res, metric, v, o):
        hw = res.get('hw_results', {}).get((v, o), {})
        jr = res.get('jax_results', {}).get(v, {})
        if metric == 'E_hw_opt':     return hw.get('E_hw_opt', float('nan'))
        if metric == 'E_sv_hw':      return hw.get('E_hw_sv',  float('nan'))
        if metric == 'gap':
            return (hw.get('E_hw_opt', float('nan'))
                    - jr.get('jax_best', {}).get('E', float('nan')))
        if metric == 'overlap_hw':   return hw.get('overlap_hw', {}).get(
            'overlap_jax', float('nan'))
        if metric == 'E_noisy_init': return hw.get('E_noisy_init', float('nan'))
        return float('nan')

    METRICS         = ['E_hw_opt', 'E_sv_hw', 'gap', 'overlap_hw', 'E_noisy_init']
    BETTER_NEGATIVE = {'E_hw_opt', 'E_sv_hw', 'gap', 'E_noisy_init'}

    def _fmt_delta(val, metric):
        if np.isnan(val):
            return "     nan"
        if metric in BETTER_NEGATIVE:
            arrow = "↓" if val < 0 else "↑"
        else:
            arrow = "↑" if val > 0 else "↓"
        return f"{val:+.4f}{arrow}"

    print(f"\n{'='*W}")
    print(f"  M3 COMPARISON TABLE  |  N={variants[0]}  E_exact={e_exact:.6f}")
    print(f"  Columns: {', '.join(headers)}")
    print(f"{'='*W}")

    for block_label, res, is_delta in [
            ("M3 = ON  (mitigated)",   res_on,  False),
            ("M3 = OFF (raw counts)",  res_off, False),
            ("delta = ON − OFF  (↓ energy better, ↑ overlap better)",
             None, True)]:
        print(f"\n  ── {block_label} ──")
        print(row_fmt.format("metric", *headers))
        print(sep)
        for metric in METRICS:
            if is_delta:
                cells = [
                    _fmt_delta(_get(res_on, metric, v, o)
                               - _get(res_off, metric, v, o), metric)
                    for v in variants for o in optimizers]
            else:
                cells = [f"{_get(res, metric, v, o):.6f}"
                         for v in variants for o in optimizers]
            print(row_fmt.format(metric[:30], *cells))

    print(f"\n  ── Runtime ──")
    t_on  = timer.elapsed("[M3=ON] run_all")
    t_off = timer.elapsed("[M3=OFF] run_all")
    for label, t in [("M3=ON  total wall time", t_on),
                     ("M3=OFF total wall time", t_off)]:
        m, s = divmod(t, 60)
        print(f"  {label:<30}  {int(m):2d}m {s:05.2f}s")
    diff  = t_on - t_off
    m, s  = divmod(abs(diff), 60)
    sign  = "+" if diff > 0 else "-"
    print(f"  {'M3 overhead':<30}  {sign}{int(m):2d}m {s:05.2f}s")
    print(f"\n{'='*W}\n")

if __name__ == '__main__':

    n_list     = [4,6,8]
    variants   = ['re', 'im', 'g']
    optimizers = ['cobyla', 'spsa', 'adam_spsa']

    timer = Timer()
    for n in n_list:
        # ── Consistency check ─────────────────────────────────────────────────────
        timer = Timer()
        timer.start("consistency_check")
        print(f"=== Consistency check  N={n} ===")
        n_up_chk = n // 2
        H_sp_chk, basis_chk, bindex_chk = build_hamiltonian(n, n_up_chk)
        h_rows_c, h_cols_c, h_vals_c    = build_jax_hamiltonian(n, n_up_chk)
        apH_chk                         = make_apply_H(h_rows_c, h_cols_c, h_vals_c,
                                                      len(basis_chk))
        _, sm_c, dm_c, vm_c             = build_givens_pairs(
            n, list(basis_chk), bindex_chk)
        nn_c                            = build_jastrow_matrix(n, list(basis_chk))
        psi_n_c                         = neel_state(
            n, n_up_chk, list(basis_chk), bindex_chk)
        rng_c                           = np.random.default_rng(42)
        n_pair_c                        = n * (n - 1) // 2
        H_pauli_c                       = build_heisenberg_pauli(n)

        for chk_variant, n_params in [('re', 2*n_pair_c),
                                      ('im', 2*n_pair_c),
                                      ('g',  3*n_pair_c)]:
            params_c = rng_c.uniform(-np.pi/4, np.pi/4, n_params)
            E_jax_c  = float(make_energy_grad(
                chk_variant, n, 1, psi_n_c,
                sm_c, dm_c, vm_c, nn_c, apH_chk)[1](
                jnp.array(params_c, dtype=jnp.float64)))
            E_sv_c   = energy_noiseless(
                build_ucj_circuit(n, 1, params_c, chk_variant), H_pauli_c)
            delta_c  = abs(E_jax_c - E_sv_c)
            status   = 'PASS' if delta_c < 1e-6 else 'FAIL'
            print(f"  {chk_variant}-uCJ  E_jax={E_jax_c:.8f}"
                  f"  E_sv={E_sv_c:.8f}  delta={delta_c:.2e}  {status}")
        timer.stop("consistency_check")

        # ── Run 1: M3 ON ──────────────────────────────────────────────────────────
        print(f"\n{'#'*70}")
        print(f"#  FULL BENCHMARK — M3=ON  (USE_M3=True)")
        print(f"{'#'*70}\n")

        _orig_use_m3 = USE_M3
        USE_M3 = True


        res_m3_on = run_all(n=n, variants=variants, optimizers=optimizers,
                            run_hardware=True, timer=timer)


        timer.summary("Overall runtime breakdown")
