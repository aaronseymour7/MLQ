"""
builder.py -- ground-state filter (cos(H t_i + phi_i) pulses) for the J1-J2
Heisenberg chain: minimax pulse optimization with certification, Trotterized
circuits, rigorous error bounds, grid snapping, simulation and resource counts.

Conventions
-----------
* Hamiltonians passed to the circuit code are the *rescaled* ones
  H_s = (H - shift) / W  (built by hamiltonians.scale_hamiltonian, identity
  term LAST), so pulse times are in scaled-H units.
* Circuit layout: system qubits 0..N-1, ancilla = qubit N (the MSB of the
  statevector index): amplitudes are [anc=0 block | anc=1 block].
* One pulse = H(anc), Rz(2 phi), exp(-i t H (x) Z_anc), H(anc).
      anc = 0 block:  cos(H t + phi) psi
      anc = 1 block: -i sin(H t + phi) psi
* Qubit q <-> chain site q. Qiskit vectors are little-endian; use
  hamiltonians.reorder_axes to go to / from the MPS (site 0 = MSB) ordering.
* Nothing here reads script-level globals.

Scope
-----
Noiseless by design: exact statevector (or exact-unitary) simulation, no gate
or shot noise. The goal is to isolate algorithmic error (filter design,
spectral inputs, Trotter), not to predict device performance.

Rigorous results used (see docstrings of the functions that use them)
----------------------------------------------------------------------
(R1) certify_filter: |d/dE prod cos(E t_i + phi_i)| <= sum |t_i|, so a grid
     max plus (sum|t_i|) h/2 is a certified sup over a continuous interval.
(R2) fidelity_lower_bound: gamma / (gamma + (1-gamma) eta^2).
(R3) trotter_bounds: post-selected state distance <= 2 eps / sqrt(p_g) with
     eps = sum_i alpha t_i^2 / (2 k_i)  (= alpha T^2/(2n) on the uniform grid).
The Lie-Trotter commutator bound itself is the standard one from Childs et al.
(Theory of Trotter error, 2021)
"""
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import optimize as opt
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ControlFlowOp, IfElseOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, MatrixExponential, SuzukiTrotter

# A13: import from the home modules (no re-exports through builder).
from get_energies import get_energies
from hamiltonians import (
    MPO_ham_j1j2, energy_variance, j1j2_hamiltonian, pauli_l1_norm,
    reorder_axes, symmetry_resolved_gaps,
)

try:
    from get_energies import ED_CHECK_MAX_N
except ImportError:
    ED_CHECK_MAX_N = 14
ED_DENSE_MAX_N = min(ED_CHECK_MAX_N, 14)   # dense eigh / dense Pauli matrices
SECTOR_MAX_N = 12                          # dense symmetry-resolved ED

BASIS = ("h", "s", "cx", "rz")


# ======================================================================
# 1. Filter evaluation, certification, minimax optimization   (A2, A10)
# ======================================================================
def filter_values(times, phases, E):
    """F(E) = prod_i cos(E t_i + phi_i), vectorized over E."""
    E = np.atleast_1d(np.asarray(E, dtype=float))
    return np.prod(np.cos(np.outer(E, np.asarray(times)) + np.asarray(phases)),
                   axis=1)


def certify_filter(times, phases, energies, n_grid=200_001, e0_slack=0.0):
    """Certified bound on eta = sup_{E in [Delta,1]} |F(E)| / |F(E0)|, Delta =
    energies[1] (R1).

    Each factor has |cos|<=1 and |d cos(E t+phi)/dE| = |t sin| <= |t|, hence
    |F'(E)| <= L = sum|t_i|. On a grid of spacing h every point of the interval
    is within h/2 of a grid point, so sup <= max_grid + L h / 2.
    e0_slack (scaled units) widens the ground-state evaluation point:
    |F(e)| >= |F(0)| - L |e|. Pass sqrt(variance)/W only as an indicative
    number: variance gives *an* eigenvalue within sqrt(var), not necessarily E0.
    Valid when the spectrum is contained in {E0} U [Delta, 1]."""
    times = np.asarray(times, dtype=float)
    phases = np.asarray(phases, dtype=float)
    delta = float(energies[1])
    if not 0.0 < delta < 1.0:
        raise ValueError(f"need 0 < Delta < 1, got {delta}")
    grid = np.linspace(delta, 1.0, n_grid)
    h = grid[1] - grid[0]
    sup_grid = float(np.max(np.abs(filter_values(times, phases, grid))))
    lip = float(np.sum(np.abs(times)))
    sup_cert = min(sup_grid + lip * h / 2.0, 1.0)
    f0 = float(abs(np.prod(np.cos(phases))))
    f0_lb = max(f0 - lip * e0_slack, 0.0)
    eta = sup_cert / f0_lb if f0_lb > 0 else float("inf")
    return dict(eta=eta, sup_cert=sup_cert, sup_grid=sup_grid, f0=f0,
                f0_lb=f0_lb, lipschitz=lip, h=h)


def brute_force_sup(times, phases, energies, n_grid=1_000_001):
    """Dense-grid sup of |F| over [Delta,1] (test T8 reference)."""
    grid = np.linspace(float(energies[1]), 1.0, n_grid)
    return float(np.max(np.abs(filter_values(times, phases, grid))))


def fidelity_lower_bound(gamma, eta):
    """(R2) Ground-state fidelity after the EXACT filter, for a trial state with
    weight gamma = |<E0|trial>|^2 on the (nondegenerate) ground state and
    spectrum otherwise inside [Delta,1]:  F >= gamma / (gamma + (1-gamma) eta^2).
    Holds for the exact filter, not for the Trotterized one."""
    return float(gamma / (gamma + (1.0 - gamma) * eta ** 2))


def fixtimes(times, totaltime):
    """Rescale times in place so that sum |t_i| = totaltime."""
    times[:] = totaltime / np.sum(np.abs(times)) * times
    return times


def _cap_and_fix(times, T, tmax):
    """Rescale to sum T with every entry <= tmax (water-filling)."""
    t = np.maximum(np.asarray(times, dtype=float), 0.0)
    t = t * T / t.sum()
    for _ in range(100):
        over = t > tmax
        if not over.any():
            break
        excess = (t[over] - tmax).sum()
        t[over] = tmax
        room = ~over
        t[room] += excess * t[room] / t[room].sum()
    return t


def _solve_minimax(z0, E_opt, total_time, p_ground_min, fixed_times=None,
                   t_max_frac=1.0 / 3.0, maxiter=5000, ftol=1e-12):
    """Epigraph form of  min max_{E in grid} F(E)^2  s.t.  prod cos^2 phi_i >=
    p_ground_min  and  sum t_i = total_time.
    z = [times, phases, s] (free times) or [phases, s] (fixed_times given)."""
    free_t = fixed_times is None

    def split(z):
        if free_t:
            n = (len(z) - 1) // 2
            return z[:n], z[n:2 * n], z[-1]
        return fixed_times, z[:-1], z[-1]

    cons = [
        {"type": "ineq",
         "fun": lambda z: z[-1] - filter_values(split(z)[0], split(z)[1], E_opt) ** 2},
        {"type": "ineq",
         "fun": lambda z: np.prod(np.cos(split(z)[1]) ** 2) - p_ground_min},
    ]
    if free_t:
        n = (len(z0) - 1) // 2
        cons.append({"type": "eq",
                     "fun": lambda z: np.sum(split(z)[0]) - total_time})
        bnds = ([(0.0, total_time * t_max_frac)] * n
                + [(-np.pi / 2, np.pi / 2)] * n + [(0.0, None)])
    else:
        n = len(z0) - 1
        bnds = [(-np.pi / 2, np.pi / 2)] * n + [(0.0, None)]
    res = opt.minimize(lambda z: z[-1], z0, method="SLSQP", bounds=bnds,
                       constraints=cons,
                       options={"maxiter": maxiter, "ftol": ftol})
    times, phases, s = split(res.x)
    return np.array(times, dtype=float), np.array(phases, dtype=float), float(s), res


def _key(r):
    """Selection key: feasible-and-converged first, then smaller certified eta."""
    return (0 if r["ok"] else 1, r["eta"])


def select_best(results):
    """Pick by certified eta among feasible, converged results (A2)."""
    return min(results, key=_key)


class FilterBuilder:
    """Minimax optimization of (times, phases) for a range of pulse counts, with
    multistart and certification of every candidate. Selection is by certified
    eta, never by pulse count."""

    def __init__(self, total_time, energies, a=4, b=15, n_starts=6, seed=0,
                 p_ground_min=0.9, n_opt_grid=200, t_max_frac=1.0 / 3.0,
                 e0_slack=0.0, maxiter=5000, ftol=1e-12):
        self.total_time = float(total_time)
        self.energies = np.asarray(energies, dtype=float)
        self.delta = float(self.energies[1])
        self.a, self.b = int(a), int(b)
        self.n_starts, self.seed = int(n_starts), int(seed)
        self.p_ground_min = float(p_ground_min)
        self.n_opt_grid = int(n_opt_grid)
        self.t_max_frac = float(t_max_frac)
        self.e0_slack = float(e0_slack)
        self.maxiter, self.ftol = maxiter, ftol

    def _initial_times(self, ntimes, s_idx, rng, tmax):
        if s_idx == 0:
            raw = 0.7 ** np.arange(ntimes)
        else:
            raw = rng.dirichlet(np.full(ntimes, 2.0))
        return _cap_and_fix(raw, self.total_time, tmax)

    def build(self, verbose=True) -> List[Dict]:
        """For each ntimes in a..b: multistart minimax, keep the candidate with
        the best (feasible, certified eta). Returns one dict per ntimes."""
        rng = np.random.default_rng(self.seed)
        E_opt = np.linspace(self.delta, 1.0, self.n_opt_grid)
        tmax = self.total_time * self.t_max_frac
        results = []
        for ntimes in range(self.a, self.b + 1):
            best = None
            for s_idx in range(self.n_starts):
                t0 = self._initial_times(ntimes, s_idx, rng, tmax)
                ph0 = (np.zeros(ntimes) if s_idx == 0
                       else rng.uniform(-0.3, 0.3, ntimes))
                s0 = float(np.max(filter_values(t0, ph0, E_opt) ** 2))
                z0 = np.concatenate([t0, ph0, [s0]])
                times, phases, s, res = _solve_minimax(
                    z0, E_opt, self.total_time, self.p_ground_min,
                    t_max_frac=self.t_max_frac, maxiter=self.maxiter,
                    ftol=self.ftol)
                feasible = (abs(times.sum() - self.total_time)
                            <= 1e-8 * max(self.total_time, 1.0)
                            and np.prod(np.cos(phases) ** 2)
                            >= self.p_ground_min - 1e-9)
                cert = certify_filter(times, phases, self.energies,
                                      e0_slack=self.e0_slack)
                cand = dict(ntimes=ntimes, times=times.copy(),
                            phases=phases.copy(), s_opt=s, eta=cert["eta"],
                            cert=cert, success=bool(res.success),
                            feasible=bool(feasible),
                            ok=bool(res.success and feasible),
                            message=str(res.message))
                if best is None or _key(cand) < _key(best):
                    best = cand
            results.append(best)
            if verbose:
                print(f"ntimes={ntimes:2d}  eta_cert={best['eta']:.3e}  "
                      f"|F0|={best['cert']['f0']:.4f}  ok={best['ok']}  "
                      f"sum t={best['times'].sum():.6f}")
        return results

    @staticmethod
    def apply_filter(times, phases, energies, state):
        """Apply prod_i cos(E t_i + phi_i) to a spectrum-basis state.
        Returns (normalized filtered state, normalization factor)."""
        f0 = np.array(state, dtype=float, copy=True)
        for t_i, phi_i in zip(times, phases):
            f0 *= np.cos(energies * t_i + phi_i)
        fnorm = 1.0 / np.sqrt(np.sum(f0 ** 2))
        return f0 * fnorm, fnorm

    def evaluate(self, results, gs_state, trial_state, plot=True, ax=None):
        """Apply every optimized filter to a spectrum-basis trial state."""
        if ax is None and plot:
            _, ax = plt.subplots(figsize=(10, 6))
        out = []
        for res in results:
            f0, fnorm = self.apply_filter(res["times"], res["phases"],
                                          self.energies, trial_state)
            fdiff = float(np.sum((gs_state - f0) ** 2))
            out.append(dict(ntimes=res["ntimes"], fdiff=fdiff, f0=f0,
                            norm=fnorm, eta=res["eta"]))
            if plot:
                ax.plot(self.energies, f0, label=f"{res['ntimes']} pulses")
        if plot:
            ax.set_xlabel("Energy"); ax.set_ylabel("Filtered amplitude")
            ax.legend(); ax.grid(True, alpha=0.3); plt.show()
        return out


# ======================================================================
# 2. Spectral inputs: certified window, symmetry-resolved gap   (A3, A9)
# ======================================================================
def get_spectrum(N, j1=1.0, j2=0.0, source="dmrg", res=None, H_mpo=None,
                 bandwidth="certified", gap_mode="any"):
    """Spectrum for FilterBuilder, rescaled to (E - shift)/W with shift = E0.

    res: output of get_energies (pass it in -> DMRG runs once, A9). If None and
         source='dmrg', get_energies is called here.
    bandwidth='certified': W = (c_I + sum|c_beta|) - E0, an upper bound on
         E_top - E0 independent of DMRG's E_top (A3). The price is a smaller
         scaled gap (longer T). 'dmrg': W = E_top - E0 from DMRG (uncertified).
    gap_mode='any': DMRG first excited state (conservative for trial circuits
         that are not exactly symmetric). 'sector': lowest level in the ground
         state's (S, parity) sector (needs N <= SECTOR_MAX_N).

    Caveat (A3): shift = DMRG E0 >= E0_exact (variational), so the true ground
    energy may sit slightly below 0 in scaled units; e0_slack reports
    sqrt(var)/W as an indicative size of that offset."""
    if source not in ("dmrg", "ed"):
        raise ValueError("source must be 'dmrg' or 'ed'")
    if bandwidth not in ("certified", "dmrg"):
        raise ValueError("bandwidth must be 'certified' or 'dmrg'")
    H_p = j1j2_hamiltonian(N, j1, j2)
    l1 = pauli_l1_norm(H_p)
    cI = float(sum(np.real(c) for lab, c in zip(H_p.paulis.to_labels(), H_p.coeffs)
                   if set(lab) == {"I"}))

    if source == "ed":
        if N > ED_DENSE_MAX_N:
            raise ValueError(f"source='ed' requires N <= {ED_DENSE_MAX_N}")
        evals = np.linalg.eigvalsh(H_p.to_matrix())
        E0, E1, Etop = float(evals[0]), float(evals[1]), float(evals[-1])
        W = Etop - E0
        return dict(energies=(evals - E0) / W, gap=(E1 - E0) / W,
                    totaltime=np.pi * W / (E1 - E0), shift=E0, W=W, E0=E0,
                    E1=E1, Etop=Etop, l1=l1, source="ed", ed=None,
                    e0_slack=0.0, var=0.0, gap_mode="any", dmrg_res=None)

    if res is None:
        if H_mpo is None:
            H_mpo = MPO_ham_j1j2(N, j1=j1, j2=j2)
        res = get_energies(H_mpo, N=N, j1=j1, j2=j2)
    E0, E1, Etop = (float(np.real(res[k])) for k in ("E0", "E1", "E_top"))

    W_cert = (cI + l1) - E0
    W = max(W_cert, Etop - E0) if bandwidth == "certified" else Etop - E0
    gap_any = E1 - E0
    gap_sector = float("nan")
    if N <= SECTOR_MAX_N:
        sym = symmetry_resolved_gaps(H_p)
        gap_sector = sym["gap_sector"]
        if abs(sym["gap_any"] - gap_any) > 1e-6:
            print(f"  [warning] DMRG gap {gap_any:.8f} != exact lowest gap "
                  f"{sym['gap_any']:.8f} (excited-state DMRG not certified)")
    if gap_mode == "sector":
        if not np.isfinite(gap_sector):
            raise ValueError("gap_mode='sector' needs N <= SECTOR_MAX_N and a "
                             "level in the ground-state sector")
        gap_raw = gap_sector
    else:
        gap_raw = gap_any

    var = 0.0
    if res.get("psi0") is not None:
        v = reorder_axes(np.asarray(res["psi0"].to_dense()).reshape(-1))
        var = max(energy_variance(H_p.to_matrix(sparse=True), v), 0.0)
    gap = gap_raw / W
    print(f"[spectrum] E0={E0:.8f} E1={E1:.8f} Etop={Etop:.8f}  "
          f"W({bandwidth})={W:.6f} (W_cert={W_cert:.6f})  gap_any={gap_any:.6f} "
          f"gap_sector={gap_sector:.6f}  scaled gap={gap:.6f}  "
          f"sqrt(var)={np.sqrt(var):.2e}")
    return dict(energies=np.array([0.0, gap, 1.0]), gap=gap,
                totaltime=np.pi / gap, shift=E0, W=W, W_cert=W_cert, E0=E0,
                E1=E1, Etop=Etop, l1=l1, gap_any_raw=gap_any,
                gap_sector_raw=gap_sector, gap_mode=gap_mode, source="dmrg",
                ed=res.get("ed"), e0_slack=float(np.sqrt(var) / W), var=var,
                dmrg_res=res)


# ======================================================================
# 3. Trotter error: nested-commutator bound                      (A4)
# ======================================================================
def _strip_identity(H):
    keep = [i for i, lab in enumerate(H.paulis.to_labels()) if set(lab) != {"I"}]
    return SparsePauliOp(H.paulis[keep], H.coeffs[keep])


def alpha_comm(H, tight=False, reverse=False):
    """alpha = sum_g || [H_g, sum_{g'>g} H_g'] || over the Pauli terms in stored
    order (reverse=True reverses that order). Identity terms are dropped
    (they commute with everything).
    tight=False: triangle inequality, sum_{i<j, noncommuting} 2|c_i||c_j|.
    tight=True:  exact spectral norm of each grouped commutator."""
    H = _strip_identity(H)
    if reverse:
        H = SparsePauliOp(H.paulis[::-1], H.coeffs[::-1])
    P, c = H.paulis, np.abs(H.coeffs)
    n = len(P)
    if not tight:
        return float(sum(2 * c[i] * c[j] for i in range(n)
                         for j in range(i + 1, n) if not P[i].commutes(P[j])))
    a = 0.0
    for i in range(n - 1):
        A = SparsePauliOp(P[i], [H.coeffs[i]])
        B = SparsePauliOp(P[i + 1:], H.coeffs[i + 1:])
        C = (1j * (A.dot(B) - B.dot(A))).simplify()      # Hermitian
        if np.allclose(C.coeffs, 0):
            continue
        if H.num_qubits <= 8:
            a += float(np.max(np.abs(np.linalg.eigvalsh(C.to_matrix()))))
        else:
            a += abs(sla.eigsh(C.to_matrix(sparse=True), k=1, which="LM",
                               return_eigenvectors=False)[0])
    return a


def trotter_alpha(H, tight=True):
    """Safe alpha: max over both term orders, so the bound holds whichever
    ordering convention the commutator theorem uses for the synthesized
    product. lie_trotter_order() tells you which one is actually realized."""
    return max(alpha_comm(H, tight, reverse=False),
               alpha_comm(H, tight, reverse=True))


def lie_trotter_order(H, tau=0.05, reps_list=(1, 3), tol=1e-9):
    N = H.num_qubits

    mats = [
        SparsePauliOp(H.paulis[i], [H.coeffs[i]]).to_matrix()
        for i in range(len(H))
    ]

    fwd_ok = rev_ok = True
    worst_f = worst_r = 0.0

    for reps in reps_list:
        t = tau * reps
        dt = t / reps

        U_f = np.eye(2**N, dtype=complex)
        U_r = np.eye(2**N, dtype=complex)

        for m in mats:
            s = expm(-1j * dt * m)
            U_f = s @ U_f
            U_r = U_r @ s

        U_f = np.linalg.matrix_power(U_f, reps)
        U_r = np.linalg.matrix_power(U_r, reps)

        # Build PauliEvolutionGate
        gate = evolution_gate(H, t, reps, 1)

        qc = QuantumCircuit(N)
        qc.append(gate, range(N))

        # IMPORTANT: synthesize/decompose the PauliEvolutionGate
        qc = qc.decompose(reps=10)

        U = Operator(qc).data

        ef = np.linalg.norm(U - U_f, 2)
        er = np.linalg.norm(U - U_r, 2)

        worst_f = max(worst_f, ef)
        worst_r = max(worst_r, er)

        fwd_ok &= ef < tol
        rev_ok &= er < tol

        print(
            f"reps={reps}: "
            f"forward={ef:.3e}, reverse={er:.3e}"
        )

    return {
        "forward": bool(fwd_ok),
        "reverse": bool(rev_ok),
        "err_forward": float(worst_f),
        "err_reverse": float(worst_r),
    }


def h_tensor_z(H):
    """H (x) Z_anc, ancilla = top qubit (leftmost label), term order preserved.
    exp(-i t H(x)Z) = e^{-iHt} on anc=0, e^{+iHt} on anc=1."""
    return SparsePauliOp.from_list(
        [("Z" + lab, c) for lab, c in zip(H.paulis.to_labels(), H.coeffs)])


def trotter_bounds(alpha, tg, k, p_g):
    """(R3) Rigorous post-selected-state bound.

    Setup: exact unnormalized vector a = A_m ... A_1 psi with A_i = <0|W_i|0>
    = cos(H t_i + phi_i); Trotterized b with A~_i = <0|W~_i|0>. Only the
    exp(-i t H(x)Z) factor differs, and [H_g(x)Z, H_g'(x)Z] = [H_g, H_g'](x)1,
    so ||W~_i - W_i|| <= alpha t_i^2 / (2 k_i) =: eps_i (same alpha for H(x)Z).
    All A_i, A~_i are blocks of unitaries (norm <= 1), so telescoping gives
    ||a - b|| <= sum_i eps_i = eps. On the uniform grid t_i = k_i dt,
    sum k_i = n: eps = alpha T^2 / (2n).
    Normalization: ||a/|a| - b/|b|| <= 2 ||a - b|| / ||a||, ||a|| = sqrt(p_g).
    Hence state distance <= 2 eps / sqrt(p_g). Valid when `alpha` is for the
    realized term order (trotter_alpha takes the max over both orders)."""
    tg = np.asarray(tg, dtype=float)
    k = np.asarray(k, dtype=float)
    total = float(np.sum(alpha * tg ** 2 / (2.0 * k)))
    raw = 2.0 * total / np.sqrt(p_g)
    return dict(bound_total=total, bound_state_raw=raw,
                bound_state=min(2.0, raw))


# ======================================================================
# 4. Grid snapping: T = sum t_i split into n equal steps dt = T / n
# ======================================================================
def snap_to_grid(times, T, n):
    """Integer k_i with sum k_i = n (largest-remainder rounding); t_i = k_i dt.

    Uniform dt is the cost-optimal allocation: with the same alpha for all
    pulses, minimizing sum k_i subject to sum alpha t_i^2/(2 k_i) <= eps is a
    Cauchy-Schwarz problem with solution k_i proportional to t_i."""
    times = np.asarray(times) * T / np.sum(times)
    dt = T / n
    x = times / dt
    k = np.floor(x + 1e-12).astype(int)
    for j in np.argsort(-(x - k))[: n - k.sum()]:
        k[j] += 1
    return k, dt


def reopt_phases(k, dt, energies, phases0, p_ground_min=0.9, n_opt_grid=200):
    """Re-optimize phases (minimax objective) for fixed grid times t_i = k_i dt."""
    times = k * dt
    E_opt = np.linspace(float(energies[1]), 1.0, n_opt_grid)
    phases0 = np.asarray(phases0, dtype=float)
    s0 = float(np.max(filter_values(times, phases0, E_opt) ** 2))
    _, ph, _, res = _solve_minimax(np.concatenate([phases0, [s0]]), E_opt,
                                   times.sum(), p_ground_min,
                                   fixed_times=times)
    if not res.success:
        print(f"[reopt_phases] warning: {res.message}")
    return ph


def grid_filter(times, phases, T, n, energies, p_ground_min=0.9):
    """Snap to n equal steps, drop pulses with k = 0, re-optimize phases.
    Returns (k, dt, phases_new); pulse times are k * dt."""
    k, dt = snap_to_grid(times, T, n)
    keep = k > 0
    k = k[keep]
    return k, dt, reopt_phases(k, dt, energies, np.asarray(phases)[keep],
                               p_ground_min)


# ======================================================================
# 5. Circuits
# ======================================================================
def evolution_gate(H, t, trotter_steps=1, order=1):
    """exp(-i H t). order=0: exact (matrix exponential; verification only),
    order=1: LieTrotter, else SuzukiTrotter. Term order is preserved."""
    if order == 0:
        synth = MatrixExponential()
    elif order == 1:
        synth = LieTrotter(reps=trotter_steps, preserve_order=True)
    else:
        synth = SuzukiTrotter(order=order, reps=trotter_steps,
                              preserve_order=True)
    return PauliEvolutionGate(H, time=t, synthesis=synth)


def apply_filter_pulse(qc, sys_qubits, anc, H, t_i, phi_i, trotter_steps=1,
                       order=1):
    """One pulse. Ancilla must enter in |0>; the 0 branch applies
    cos(H t_i + phi_i) to the system, the 1 branch -i sin(H t_i + phi_i)."""
    qc.h(anc)
    qc.rz(2 * phi_i, anc)
    qc.append(evolution_gate(h_tensor_z(H), t_i, trotter_steps, order),
              [*sys_qubits, anc])
    qc.h(anc)


def _steps_per_pulse(trotter_steps, n_pulses):
    if np.isscalar(trotter_steps):
        return [int(trotter_steps)] * n_pulses
    steps = [int(s) for s in trotter_steps]
    if len(steps) != n_pulses:
        raise ValueError(f"need {n_pulses} step counts, got {len(steps)}")
    return steps


def build_filter_circuit(H, times, phases, trial_prep=None, trotter_steps=1,
                         order=1):
    """Full filter, measure + reset ancilla after each pulse (no early abort)."""
    N = H.num_qubits
    sys_reg, anc = QuantumRegister(N, "sys"), QuantumRegister(1, "anc")
    creg = ClassicalRegister(len(times), "anc_meas")
    qc = QuantumCircuit(sys_reg, anc, creg)
    steps = _steps_per_pulse(trotter_steps, len(times))
    if trial_prep is not None:
        trial_prep(qc, sys_reg)
    for k, (t_i, phi_i) in enumerate(zip(times, phases)):
        apply_filter_pulse(qc, sys_reg, anc[0], H, t_i, phi_i, steps[k], order)
        qc.measure(anc[0], creg[k])
        qc.reset(anc[0])
    return qc


def build_early_abort_circuit(H, times, phases, trial_prep=None,
                              trotter_steps=1, order=1):
    """Deterministic-time cosine filter with a single ancilla, mid-circuit
    measure + reset, and the remaining pulses skipped as soon as the ancilla
    reads 1 (needs dynamic circuits / if_test). (A14: renamed from the
    'rodeo' circuit -- the rodeo algorithm uses random times; here the times
    are the optimized deterministic ones.) trotter_steps: int or per-pulse list."""
    N = H.num_qubits
    sys_reg, anc = QuantumRegister(N, "sys"), QuantumRegister(1, "anc")
    creg = ClassicalRegister(len(times), "cycle")
    qc = QuantumCircuit(sys_reg, anc, creg)
    steps = _steps_per_pulse(trotter_steps, len(times))
    if trial_prep is not None:
        trial_prep(qc, sys_reg)

    def recurse(k):
        if k == len(times):
            return
        apply_filter_pulse(qc, sys_reg, anc[0], H, times[k], phases[k],
                           steps[k], order)
        qc.measure(anc[0], creg[k])
        qc.reset(anc[0])
        with qc.if_test((creg[k], 0)):
            recurse(k + 1)

    recurse(0)
    return qc, creg


# ======================================================================
# 6. Simulation (post-selected on ancilla = 0 after every pulse)
# ======================================================================
def postselected_run(trial_qc, times, phases, pulse_fn, synthesize=True):
    """Apply each pulse, project ancilla -> 0, renormalize.
    pulse_fn(qc, sys_qubits, anc, t_i, phi_i, k) appends pulse k to qc.
    synthesize=True transpiles PauliEvolutionGate to real Trotter gates.
    Noiseless. Returns (product of success probabilities, system statevector)."""
    N = trial_qc.num_qubits
    dim = 2 ** N
    sys_q, anc = list(range(N)), N
    sv = Statevector(np.concatenate([Statevector(trial_qc).data,
                                     np.zeros(dim, dtype=complex)]))
    p_succ = 1.0
    for k, (t_i, phi_i) in enumerate(zip(times, phases)):
        pc = QuantumCircuit(N + 1)
        pulse_fn(pc, sys_q, anc, t_i, phi_i, k)
        if synthesize:
            pc = transpile(pc, basis_gates=["h", "s", "x", "cx", "rz"],
                           optimization_level=0)
        sv = sv.evolve(pc)
        proj = np.asarray(sv.data)[:dim]
        p = float(np.vdot(proj, proj).real)
        p_succ *= p
        sv = Statevector(np.concatenate([proj / np.sqrt(p),
                                         np.zeros(dim, dtype=complex)]))
    return p_succ, np.asarray(sv.data)[:dim]


def apply_pulse_exact(state, Hs, t_i, phi_i):
    """One exact pulse on [anc=0 | anc=1] using sparse expm_multiply (A12): no
    dense exponentials. Hs: sparse scaled H."""
    dim = Hs.shape[0]
    s = state.reshape(2, dim)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    rz = np.diag([np.exp(-1j * phi_i), np.exp(1j * phi_i)])
    s = (rz @ h) @ s                                       # H, then Rz(2 phi)
    s = np.stack([expm_multiply(-1j * t_i * Hs, s[0]),     # anc=0: e^{-iHt}
                  expm_multiply(+1j * t_i * Hs, s[1])])    # anc=1: e^{+iHt}
    s = h @ s                                              # final H
    return s.reshape(-1)


def postselected_run_exact(trial_qc, times, phases, Hm):
    """Exact-unitary post-selected run. Hm: scaled H, dense or sparse.
    The Trotter-error-free, noiseless reference. Returns (P_succ, sys vector)."""
    Hs = sp.csr_matrix(Hm)
    dim = Hs.shape[0]
    state = np.concatenate([Statevector(trial_qc).data.astype(complex),
                            np.zeros(dim, dtype=complex)])
    p_succ = 1.0
    for t_i, phi_i in zip(times, phases):
        state = apply_pulse_exact(state, Hs, t_i, phi_i)
        anc0 = state[:dim]
        p = float(np.vdot(anc0, anc0).real)
        p_succ *= p
        state = np.concatenate([anc0 / np.sqrt(p), np.zeros(dim, dtype=complex)])
    return p_succ, state[:dim]


def overlaps(vec_qiskit, psi0_dmrg, psi0_ed=None):
    """(|<DMRG|v>|^2, |<ED|v>|^2) for a qiskit-ordered vector. psi0_* in MPS
    ordering (site 0 = MSB). ED entry is NaN if psi0_ed is None."""
    t = reorder_axes(vec_qiskit)
    fd = float(abs(np.vdot(psi0_dmrg, t)) ** 2)
    fe = (float(abs(np.vdot(psi0_ed, t)) ** 2) if psi0_ed is not None
          else float("nan"))
    return fd, fe


def state_metrics(sys_vec, H_qk, psi0_dmrg, psi0_ed=None):
    """(energy w.r.t. unscaled H_qk, fidelity vs DMRG gs, fidelity vs ED gs)."""
    E = float(np.real(Statevector(sys_vec).expectation_value(H_qk)))
    fd, fe = overlaps(sys_vec, psi0_dmrg, psi0_ed)
    return E, fd, fe


# ======================================================================
# 7. Resource counts                                              (A11)
# ======================================================================
def flatten_success_path(circ):
    """Inline every IfElseOp's true-body, leaving the all-zeros path."""
    flat = QuantumCircuit(list(circ.qubits), list(circ.clbits))
    for inst in circ.data:
        op = inst.operation
        if isinstance(op, IfElseOp):
            body = flatten_success_path(op.blocks[0])
            flat.compose(body, qubits=list(inst.qubits),
                         clbits=list(inst.clbits), inplace=True)
        elif isinstance(op, ControlFlowOp):
            raise NotImplementedError(f"unhandled control flow: {op.name}")
        else:
            flat.append(op, inst.qubits, inst.clbits)
    return flat


def count_nonclifford_rz(circ, atol=1e-9):
    """Number of rz gates whose angle is not a multiple of pi/2."""
    cnt = 0
    for inst in circ.data:
        if inst.operation.name == "rz":
            x = float(inst.operation.params[0]) / (np.pi / 2)
            if abs(x - round(x)) > atol:
                cnt += 1
    return cnt


def rotation_synthesis_t_count(n_rot, eps=1e-3):
    """APPROXIMATE T-count to synthesize n_rot arbitrary Rz rotations to
    per-rotation precision eps, using ~ 1.15 log2(1/eps) + 9.2 T per rotation
    (my recollection of the Kliuchnikov/Ross-Selinger-type scaling -- verify
    the constants before quoting). Only for fault-tolerant cost estimates."""
    return float(n_rot * (1.15 * np.log2(1.0 / eps) + 9.2))


def resource_costs(circuit, label="", flatten=False, verbose=True, basis=BASIS,
                   coupling_map=None, synth_eps=1e-3, seed_transpiler=0):
    """Transpile to `basis` (opt level 3) and count gates. flatten=True for the
    early-abort circuit. coupling_map: optional qiskit CouplingMap (routing
    overhead shows up as extra cx). Also returns non-Clifford rz count and an
    approximate T-count for rotation synthesis. Pass the trial circuit
    separately (or inside the circuit via trial_prep) to include it (A11)."""
    circ = flatten_success_path(circuit) if flatten else circuit
    rc = transpile(circ, basis_gates=list(basis) + ["measure", "reset"],
                   coupling_map=coupling_map, optimization_level=3,
                   seed_transpiler=seed_transpiler)
    counts = rc.count_ops()
    tidy = {g: int(counts.get(g, 0)) for g in basis}
    mid = {g: int(counts[g]) for g in ("measure", "reset") if g in counts}
    other = {g: int(c) for g, c in counts.items()
             if g not in basis and g not in ("measure", "reset", "barrier")}
    n_nc = count_nonclifford_rz(rc)
    out = {**tidy, "total": sum(tidy.values()), "depth": rc.depth(),
           "measure": mid.get("measure", 0), "reset": mid.get("reset", 0),
           "rz_nonclifford": n_nc,
           "t_count_est": rotation_synthesis_t_count(n_nc, synth_eps),
           "other": other}
    if verbose:
        print(f"\n--- resource costs: {label} ---")
        print("  " + "  ".join(f"{g.upper()}={tidy[g]}" for g in basis)
              + f"  | total={out['total']}  depth={out['depth']}"
              + (f"  | mid-circuit: {mid}" if mid else ""))
        print(f"  non-Clifford rz={n_nc}  approx T-count (eps={synth_eps:g}/rot)"
              f"={out['t_count_est']:.0f}")
        if other:
            print(f"  [warning] non-basis ops left after transpile: {other}")
    return out


def cx_depth_per_step(H, order=1, basis=BASIS, optimization_level=3):
    """(CX count, depth) of ONE Trotter step of one pulse, transpiled once.
    Ignores cancellation across step boundaries -- validate against a full
    resource_costs() transpile at more than one n."""
    N = H.num_qubits
    qc = QuantumCircuit(N + 1)
    apply_filter_pulse(qc, list(range(N)), N, H, 1.0, 0.0,
                       trotter_steps=1, order=order)
    rc = transpile(qc, basis_gates=list(basis),
                   optimization_level=optimization_level, seed_transpiler=0)
    return rc.count_ops().get("cx", 0), rc.depth()


# ======================================================================
# 8. Convention checks (run before trusting the circuit)         (A5)
# ======================================================================
def verify_pulse_convention(seed=7, atol=1e-8, verbose=True):
    """Single pulse on N=3 with a NONCOMMUTING, field-carrying, randomly ordered
    H and a random COMPLEX state. Uses the exact (unsynthesized) evolution so
    the check isolates the pulse convention from Trotter error. Checks, up to
    one global phase: anc=0 block = cos(Ht+phi) psi, anc=1 block =
    -i sin(Ht+phi) psi; the numpy block pulse and the classical cosine filter
    agree; and the flipped sign phi -> -phi is REJECTED (test has power).
    Returns True/False; callers should treat False as fatal."""
    rng = np.random.default_rng(seed)
    N, dim = 3, 8
    base = j1j2_hamiltonian(N, 1.0, 0.7, fields=[0.3, 0.6, 0.9], order="bond")
    perm = rng.permutation(len(base))
    H = SparsePauliOp(base.paulis[perm], base.coeffs[perm])
    Hm = H.to_matrix()
    psi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    psi /= np.linalg.norm(psi)
    t_i, phi_i = 0.83, 0.37

    def expected(phi):
        X = t_i * Hm + phi * np.eye(dim)
        c = (expm(1j * X) + expm(-1j * X)) / 2
        s = (expm(1j * X) - expm(-1j * X)) / 2j
        return np.concatenate([c @ psi, -1j * (s @ psi)])

    def aligned_err(got, exp_):
        th = np.angle(np.vdot(exp_, got))
        return float(np.linalg.norm(got - np.exp(1j * th) * exp_))

    qc = QuantumCircuit(N + 1)
    qc.prepare_state(psi, list(range(N)))
    apply_filter_pulse(qc, list(range(N)), N, H, t_i, phi_i, order=0)
    got = Statevector(qc).data
    err_pos = aligned_err(got, expected(phi_i))
    err_flip = aligned_err(got, expected(-phi_i))

    s_np = apply_pulse_exact(np.concatenate([psi, np.zeros(dim)]),
                             sp.csr_matrix(Hm), t_i, phi_i)
    err_np = aligned_err(s_np, expected(phi_i))

    w, V = np.linalg.eigh(Hm)
    cl = V @ (np.cos(w * t_i + phi_i) * (V.conj().T @ psi))
    err_cl = float(np.linalg.norm(cl - expected(phi_i)[:dim]))

    ok = (err_pos < atol and err_np < atol and err_cl < atol and err_flip > 1e-3)
    if verbose:
        print(f"pulse convention: circuit err={err_pos:.2e}  numpy err={err_np:.2e}  "
              f"classical err={err_cl:.2e}  flipped-phi err={err_flip:.2e} "
              f"(must be large)  ->  {'OK' if ok else 'FAIL'}")
    return bool(ok)
