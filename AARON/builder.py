"""
builder.py -- ground-state filter (cos(H t_i + phi_i) pulses) for the J1-J2
Heisenberg chain: pulse optimization, Trotterized circuits, error bounds,
grid snapping, and simulation / resource-count helpers.

Conventions
-----------
* Hamiltonians passed to the circuit code are the *rescaled* ones
  (E0 -> 0, E_top -> 1), so pulse times are in scaled-H units.
* Circuit layout: system qubits 0..N-1, ancilla = qubit N (the MSB).
* One pulse = H(anc), Rz(2 phi), exp(-i t_i H (x) Z_anc), H(anc).
  Ancilla = 0 branch applies cos(H t_i + phi_i) to the system.
* Nothing here reads script-level globals: N comes from the circuit / operator,
  everything else is passed in.
"""
from math import ceil
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import quimb as qu
import quimb.tensor as qtn
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from scipy import optimize as opt
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit import ControlFlowOp, IfElseOp
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from get_energies import get_energies

try:
    from get_energies import ED_CHECK_MAX_N
except ImportError:
    ED_CHECK_MAX_N = 14   # fallback: largest N for which dense ED is allowed

BASIS = ("h", "s", "cx", "rz")


# ======================================================================
# 1. Pulse optimization helpers
# ======================================================================
def fixtimes(times, totaltime):
    """Rescale times in place so that sum |t_i| = totaltime."""
    xf = totaltime / np.sum(np.abs(times))
    times[:] = xf * times
    return times


def unpack(timesphases):
    n = len(timesphases) // 2
    return timesphases[:n].copy(), timesphases[n:].copy()


def timesconstraints(timesphases, total_time):
    times, _ = unpack(timesphases)
    return abs(np.sum(times) - total_time)


def probability_constraints(timesphases, energies):
    _, phases = unpack(timesphases)
    return np.prod(np.cos(phases) ** 2) - 0.9


def probability_constraintsb(timesphases, energies, pos):
    times, phases = unpack(timesphases)
    return np.prod(np.cos(energies[pos] * times + phases) ** 2) - 0.9


def new_func_v3(timesphases, energies):
    times, phases = unpack(timesphases)
    num = np.prod(np.cos(phases))
    cos_vals = np.cos(energies[:, None] * times + phases)
    den = np.prod(cos_vals ** 2, axis=1).sum()
    return abs(1 - num / np.sqrt(den))


def new_func_v4(timesphases, energies):
    times, _ = unpack(timesphases)
    cos_vals = np.cos(energies[:, None] * times)
    den = np.prod(cos_vals ** 2, axis=1).sum()
    return abs(1 - 1.0 / np.sqrt(den))


class FilterBuilder:
    """Optimize (times, phases) for a range of pulse counts, then evaluate the
    resulting filters on a classical (spectrum-only) trial state."""

    _METHODS = {
        # name: (objective, extra inequality constraint, constraint needs `pos`)
        "v3":  (new_func_v3, probability_constraints,  False),
        "v3b": (new_func_v3, probability_constraintsb, True),
        "v4":  (new_func_v4, None,                     False),
    }

    def __init__(self, total_time, energies, a=4, b=15,
                 optimizer="SLSQP", maxiter=5000, ftol=1e-12):
        self.total_time = float(total_time)
        self.energies = np.asarray(energies, dtype=float)
        self.a, self.b = int(a), int(b)
        self.optimizer = optimizer
        self.opts = {"maxiter": maxiter, "ftol": ftol}

    # ------------------------------------------------------------------
    def build(self, method="v4") -> List[Dict]:
        """Optimize for ntimes = a..b pulses. Returns one dict per ntimes."""
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {list(self._METHODS)}")
        objective, extra_constr, needs_pos = self._METHODS[method]
        results = []

        for ntimes in range(self.a, self.b + 1):
            # initial guess: geometrically decreasing times summing to T
            times = 0.5 ** np.arange(ntimes)
            times = fixtimes(times, self.total_time)
            x0 = np.zeros(2 * ntimes)
            x0[:ntimes] = times

            bnds = ([(0.0, self.total_time / 3.0)] * ntimes
                    + [(-np.pi / 2, np.pi / 2)] * ntimes)

            constraints = [{"type": "eq", "fun": timesconstraints,
                            "args": (self.total_time,)}]
            if extra_constr is not None:
                args = (self.energies, 0) if needs_pos else (self.energies,)
                constraints.append({"type": "ineq", "fun": extra_constr,
                                    "args": args})

            res = opt.minimize(objective, x0=x0, args=(self.energies,),
                               method=self.optimizer, bounds=bnds,
                               constraints=constraints, options=self.opts,
                               tol=1e-13)

            times_opt, phases_opt = res.x[:ntimes], res.x[ntimes:]
            results.append({
                "ntimes": ntimes,
                "times": times_opt.copy(),
                "phases": phases_opt.copy(),
                "fun": float(res.fun),
                "success": bool(res.success),
                "message": str(res.message),
                "result": res,
            })
            print(f"ntimes={ntimes:2d}  time={times_opt.sum():.6f}  "
                  f"fun={res.fun:.3e}  success={res.success}")
        return results

    # ------------------------------------------------------------------
    @staticmethod
    def apply_filter(times, phases, energies, state) -> Tuple[np.ndarray, float]:
        """Apply prod_i cos(E t_i + phi_i) to a spectrum-basis state.
        Returns (normalized filtered state, normalization factor)."""
        f0 = np.array(state, dtype=float, copy=True)
        for t_i, phi_i in zip(times, phases):
            f0 *= np.cos(energies * t_i + phi_i)
        fnorm = 1.0 / np.sqrt(np.sum(f0 ** 2))
        return f0 * fnorm, fnorm

    # ------------------------------------------------------------------
    def evaluate(self, results, gs_state, trial_state, plot=True,
                 highlight_pos: Optional[int] = None,
                 ax: Optional[plt.Axes] = None) -> List[Dict]:
        """Evaluate every optimized filter on trial_state; optionally plot."""
        if ax is None and plot:
            _, ax = plt.subplots(figsize=(10, 6))

        eval_results = []
        for res in results:
            times, phases = res["times"], res["phases"]
            f0, fnorm = self.apply_filter(times, phases, self.energies,
                                          trial_state)
            fdiff = float(np.sum((gs_state - f0) ** 2))
            eval_results.append({"ntimes": res["ntimes"], "fdiff": fdiff,
                                 "f0": f0.copy(), "norm": fnorm,
                                 "times": times.copy(),
                                 "phases": phases.copy()})
            print(f"ntimes={res['ntimes']:2d}  totaltime={times.sum():.6f}  "
                  f"fdiff={fdiff:.6e}")
            if plot:
                ax.plot(self.energies, f0, label=f"{res['ntimes']} pulses")

        if plot:
            if highlight_pos is not None:
                ax.axvline(self.energies[highlight_pos], color="r", alpha=0.7,
                           label=f"E[{highlight_pos}]")
            ax.set_ylim(-1, 1)
            ax.set_xlabel("Energy")
            ax.set_ylabel("Filtered State Amplitude")
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        return eval_results

    # ------------------------------------------------------------------
    def build_and_evaluate(self, trial_state, method="v4", gs_state=None,
                           highlight_pos=1, plot=True):
        """Build, then evaluate on a caller-supplied trial_state (spectrum
        basis). gs_state defaults to the unit vector on energies[0]; that is
        just the definition of the basis, not an assumed overlap."""
        if gs_state is None:
            gs_state = np.zeros(len(self.energies))
            gs_state[0] = 1.0
        results = self.build(method)
        return self.evaluate(results, gs_state, trial_state, plot=plot,
                             highlight_pos=highlight_pos)


# ======================================================================
# 2. J1-J2 Hamiltonian: Pauli form, MPO form, spectrum
# ======================================================================
def j1j2_hamiltonian(N, j1=1.0, j2=0.5):
    """J1-J2 Heisenberg chain as a SparsePauliOp (open boundary).
    S_i . S_j = 1/4 (XX + YY + ZZ) for spin-1/2."""
    terms = []
    for r, J in ((1, j1), (2, j2)):
        if J == 0:
            continue
        for i in range(N - r):
            for P in "XYZ":
                label = ["I"] * N
                label[i], label[i + r] = P, P
                terms.append(("".join(label), 0.25 * J))
    return SparsePauliOp.from_list(terms).simplify()


def mpo_two_body_heisenberg(N, J, r, S=0.5):
    """MPO for J * sum_i S_i . S_{i+r}, open boundary.
    r=1: nearest neighbor (bond dim 5); r=2: next-nearest neighbor."""
    ops = [np.asarray(qu.spin_operator(a, S=S)) for a in ("x", "y", "z")]
    d = ops[0].shape[0]
    I = np.eye(d)

    n_states = 2 + 3 * r      # start + r waiting states per component + finish
    FIN = n_states - 1

    def idx(a, k):            # a: component 0,1,2 ; k: 1..r
        return 1 + a * r + (k - 1)

    W = np.zeros((n_states, n_states, d, d), dtype=complex)
    W[0, 0] = I               # nothing happened yet
    W[FIN, FIN] = I           # term already closed
    for a in range(3):
        W[0, idx(a, 1)] = ops[a]                 # open the term
        for k in range(1, r):
            W[idx(a, k), idx(a, k + 1)] = I      # wait
        W[idx(a, r), FIN] = J * ops[a]           # close at distance r

    arrays = []
    for i in range(N):
        if i == 0:
            arrays.append(W[0, :, :, :])         # (r, u, d)
        elif i == N - 1:
            arrays.append(W[:, FIN, :, :])       # (l, u, d)
        else:
            arrays.append(W)                     # (l, r, u, d)
    return qtn.MatrixProductOperator(arrays, shape="lrud")


def MPO_ham_j1j2(N, j1=1.0, j2=0.5, cyclic=False):
    if cyclic:
        raise NotImplementedError("this builder is open-boundary only")
    H = mpo_two_body_heisenberg(N, j1, r=1) + mpo_two_body_heisenberg(N, j2, r=2)
    H.compress(cutoff=1e-12)
    return H


def build_sparse_j1j2(N, j1=1.0, j2=0.0):
    """Sparse J1-J2 Hamiltonian via the MPO builder (dense conversion; fine for
    N <= ED_CHECK_MAX_N). Uses the same MPO as DMRG, so it guards against
    solver bugs, not MPO-construction bugs."""
    return sp.csr_matrix(MPO_ham_j1j2(N, j1=j1, j2=j2).to_dense())


def get_spectrum(N, j1=1.0, j2=0.0, source="dmrg", run_ed_check=True):
    """
    Energy spectrum for FilterBuilder, rescaled to E0 -> 0, E_top -> 1.

    source="dmrg": the three DMRG energies (E0, E1, E_top) only.
    source="ed":   full exact spectrum (N <= ED_CHECK_MAX_N).

    Returns dict: energies, gap, totaltime (= pi/gap), raw E0/E1/Etop,
    source, ed (cross-check dict if it ran).
    """
    if source not in ("dmrg", "ed"):
        raise ValueError("source must be 'dmrg' or 'ed'")

    H = MPO_ham_j1j2(N, j1=j1, j2=j2)

    if source == "ed":
        if N > ED_CHECK_MAX_N:
            raise ValueError(f"source='ed' requires N <= {ED_CHECK_MAX_N}, got N={N}")
        print(f"[source=ed] exact diagonalization, N={N} (dim=2^{N}={2**N})")
        evals = np.linalg.eigvalsh(H.to_dense())        # ascending
        E0, E1, Etop = evals[0], evals[1], evals[-1]
        energies = (evals - E0) / (Etop - E0)
        ed_result = None
    else:
        print(f"[source=dmrg] DMRG spectrum (E0, E1, E_top only), N={N}")
        res = get_energies(H, N=N, j1=j1, j2=j2, run_ed_check=run_ed_check)
        E0, E1, Etop = (np.real(res["E0"]), np.real(res["E1"]),
                        np.real(res["E_top"]))
        ed_result = res["ed"]
        energies = np.array([0.0, (E1 - E0) / (Etop - E0), 1.0])

    gap = energies[1]
    return dict(energies=energies, gap=gap, totaltime=np.pi / gap,
                E0=E0, E1=E1, Etop=Etop, source=source, ed=ed_result)


def run_filter(N, trial_state, j1=1.0, j2=0.0, source="dmrg", method="v3",
               a=8, b=10, highlight_pos=1, plot=True):
    """Spectrum -> FilterBuilder -> optimize + evaluate on trial_state
    (spectrum-basis amplitudes, length = len(spec['energies']))."""
    import time
    spec = get_spectrum(N, j1=j1, j2=j2, source=source)
    builder = FilterBuilder(total_time=spec["totaltime"],
                            energies=spec["energies"], a=a, b=b)
    start = time.time()
    eval_results = builder.build_and_evaluate(trial_state, method=method,
                                              highlight_pos=highlight_pos,
                                              plot=plot)
    print(f"[{source}] total time: {time.time() - start:.2f} sec")
    return spec, eval_results


# ======================================================================
# 3. Trotter error: nested-commutator bound and step planning
# ======================================================================
def alpha_comm(H, tight=False):
    """alpha = sum_g1 || [H_g1, sum_{g2>g1} H_g2] ||, Pauli terms in H.paulis
    order (the order LieTrotter uses). First-order error of time t with r steps
    is <= alpha t^2 / (2 r).
    tight=False: triangle-inequality version (safe upper bound).
    tight=True:  exact spectral norm of each nested commutator."""
    P, c = H.paulis, np.abs(H.coeffs)
    n = len(P)
    if not tight:
        return sum(2 * c[i] * c[j] for i in range(n) for j in range(i + 1, n)
                   if not P[i].commutes(P[j]))
    a = 0.0
    for i in range(n - 1):
        A = SparsePauliOp(P[i], H.coeffs[i])
        B = SparsePauliOp(P[i + 1:], H.coeffs[i + 1:])
        C = (1j * (A @ B - B @ A)).simplify()           # Hermitian
        if len(C.paulis) == 0 or np.allclose(C.coeffs, 0):
            continue
        a += abs(sla.eigsh(C.to_matrix(sparse=True), k=1, which="LM",
                           return_eigenvectors=False)[0])
    return a


def trotter1_plan(H, times, eps, tight=False):
    """Bound-based plan: fix dt from the largest pulse at error eps, reuse it
    for every pulse (r_i = ceil(t_i / dt))."""
    alpha = alpha_comm(H, tight)
    t_max = float(np.max(np.abs(times)))
    r_max = max(1, ceil(alpha * t_max ** 2 / (2 * eps)))
    dt = t_max / r_max
    steps = [max(1, ceil(abs(t) / dt - 1e-12)) for t in times]
    bound_total = sum(alpha * abs(t) * dt / 2 for t in times)
    return dict(alpha=alpha, dt=dt, r_max=r_max, steps=steps,
                bound_total=bound_total)


def h_tensor_z(H):
    """H (x) Z_anc, with the ancilla as the top qubit (leftmost Pauli label).
    exp(-i t H(x)Z) = e^{-iHt} on anc=0, e^{+iHt} on anc=1."""
    return SparsePauliOp.from_list(
        [("Z" + lab, c) for lab, c in zip(H.paulis.to_labels(), H.coeffs)])


def pulse_error(t, m, HZ, HZ_sp, v0, order=1):
    """|| Trotter(t, m steps) - exact ||  on v0 (use |+>_anc (x) trial state to
    exercise both ancilla branches). HZ_sp = HZ.to_matrix(sparse=True), built
    once by the caller."""
    exact = expm_multiply(-1j * t * HZ_sp, v0)
    qc = QuantumCircuit(HZ.num_qubits)
    qc.append(evolution_gate(HZ, t, m, order), range(HZ.num_qubits))
    qc = transpile(qc, basis_gates=["h", "s", "x", "cx", "rz"],
                   optimization_level=0)
    return np.linalg.norm(Statevector(v0).evolve(qc).data - exact)


def empirical_dt(t_max, eps, HZ, HZ_sp, v0, order=1, m_hi=4096):
    """Smallest m with pulse_error(t_max, m) <= eps. Returns (m, t_max / m)."""
    err = lambda m: pulse_error(t_max, m, HZ, HZ_sp, v0, order)
    m = 1
    while err(m) > eps:                       # bracket
        m *= 2
        if m > m_hi:
            raise RuntimeError("eps too small for m_hi")
    lo, hi = m // 2 + 1, m                    # bisect (error ~monotone in m)
    while lo < hi:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if err(mid) <= eps else (mid + 1, hi)
    return hi, t_max / hi


# ======================================================================
# 4. Grid snapping: T = sum t_i split into n equal steps dt = T / n
# ======================================================================
def snap_to_grid(times, T, n):
    """Integer k_i with sum k_i = n (largest-remainder rounding); t_i = k_i dt,
    dt = T / n."""
    times = np.asarray(times) * T / np.sum(times)       # enforce sum = T
    dt = T / n
    x = times / dt
    k = np.floor(x + 1e-12).astype(int)
    for j in np.argsort(-(x - k))[: n - k.sum()]:
        k[j] += 1
    return k, dt


def reopt_phases(k, dt, energies, phases0):
    """Re-optimize phases for fixed grid times t_i = k_i dt."""
    times = k * dt
    obj = lambda ph: new_func_v3(np.concatenate([times, ph]), energies)
    con = {"type": "ineq",
           "fun": lambda ph: probability_constraints(
               np.concatenate([times, ph]), energies)}
    res = opt.minimize(obj, phases0, method="SLSQP", constraints=[con],
                       bounds=[(-np.pi / 2, np.pi / 2)] * len(k),
                       options={"maxiter": 5000, "ftol": 1e-12})
    if not res.success:
        print(f"[reopt_phases] warning: {res.message}")
    return res.x


def grid_filter(times, phases, T, n, energies):
    """Snap to n equal steps, drop pulses with k = 0, re-optimize phases.
    Returns (k, dt, phases_new); pulse times are k * dt."""
    k, dt = snap_to_grid(times, T, n)
    keep = k > 0
    k = k[keep]
    return k, dt, reopt_phases(k, dt, energies, np.asarray(phases)[keep])


# ======================================================================
# 5. Circuits
# ======================================================================
def evolution_gate(H, t, trotter_steps=1, order=1):
    """Trotterized exp(-i H t). order=1 -> LieTrotter, else SuzukiTrotter."""
    synth = (LieTrotter(reps=trotter_steps) if order == 1
             else SuzukiTrotter(order=order, reps=trotter_steps))
    return PauliEvolutionGate(H, time=t, synthesis=synth)


def apply_filter_pulse(qc, sys_qubits, anc, H, t_i, phi_i, trotter_steps=1,
                       order=1):
    """One pulse. Ancilla must enter in |0>; measure it afterwards and keep the
    0 branch (that branch applies cos(H t_i + phi_i) to the system)."""
    qc.h(anc)
    qc.rz(2 * phi_i, anc)     # sign checked by verify_pulse_convention()
    qc.append(evolution_gate(h_tensor_z(H), t_i, trotter_steps, order),
              [*sys_qubits, anc])
    qc.h(anc)


def _steps_per_pulse(trotter_steps, n_pulses):
    """Accept an int (same for all pulses) or a per-pulse sequence."""
    if np.isscalar(trotter_steps):
        return [int(trotter_steps)] * n_pulses
    steps = [int(s) for s in trotter_steps]
    if len(steps) != n_pulses:
        raise ValueError(f"need {n_pulses} step counts, got {len(steps)}")
    return steps


def build_filter_circuit(H, times, phases, trial_prep=None, trotter_steps=1,
                         order=1):
    """Full filter, measure + reset ancilla after each pulse (no early abort).
    trial_prep(qc, sys_reg) prepares the trial state on the system register."""
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


def build_rodeo_circuit(H, times, phases, trial_prep=None, trotter_steps=1,
                        order=1):
    """Rodeo-style variant: single ancilla, mid-circuit measure + reset, and the
    remaining cycles are skipped as soon as the ancilla reads 1 (needs dynamic
    circuit support / if_test). trotter_steps: int or per-pulse list."""
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
        with qc.if_test((creg[k], 0)):          # continue only on success
            recurse(k + 1)

    recurse(0)
    return qc, creg


# ======================================================================
# 6. Simulation (post-selected on ancilla = 0 after every pulse)
# ======================================================================
def postselected_run(trial_qc, times, phases, pulse_fn, synthesize=True):
    """Apply each pulse, project ancilla -> 0, renormalize.
    pulse_fn(qc, sys_qubits, anc, t_i, phi_i, k) appends pulse k to qc.
    synthesize=True transpiles PauliEvolutionGate to real Trotter gates before
    simulating (otherwise Statevector would use the exact matrix).
    Returns (product of success probabilities, system statevector)."""
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


def apply_pulse_exact_np(state, Ufwd, Ubwd, phi_i, dim):
    """One exact pulse by numpy block manipulation (no qiskit control()).
    state layout: [anc=0 block, anc=1 block] (ancilla = MSB).
    Ufwd = e^{+iHt}, Ubwd = e^{-iHt}."""
    s = state.reshape(2, dim)
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    rz = np.diag([np.exp(-1j * phi_i), np.exp(1j * phi_i)])
    s = np.einsum("ab,b...->a...", rz @ h, s)          # H, then Rz(2 phi)
    s = np.stack([Ubwd @ s[0], Ufwd @ s[1]])           # anc=0: e^{-iHt}, anc=1: e^{+iHt}
    s = np.einsum("ab,b...->a...", h, s)               # final H
    return s.reshape(-1)


def postselected_run_exact(trial_qc, times, phases, Hm):
    """Exact-unitary post-selected run in numpy. Hm: dense scaled H matrix.
    Returns (product of success probabilities, system statevector)."""
    N = trial_qc.num_qubits
    dim = 2 ** N
    state = np.concatenate([Statevector(trial_qc).data.astype(complex),
                            np.zeros(dim, dtype=complex)])
    cache = {}                                          # local: keyed on t for this Hm only
    p_succ = 1.0
    for t_i, phi_i in zip(times, phases):
        key = round(float(t_i), 12)
        if key not in cache:
            cache[key] = (expm(1j * t_i * Hm), expm(-1j * t_i * Hm))
        state = apply_pulse_exact_np(state, *cache[key], phi_i, dim)
        anc0 = state[:dim]
        p = float(np.vdot(anc0, anc0).real)
        p_succ *= p
        state = np.concatenate([anc0 / np.sqrt(p), np.zeros(dim, dtype=complex)])
    return p_succ, state[:dim]


def state_metrics(sys_vec, H_qk, psi0_dmrg, psi0_ed):
    """(energy w.r.t. unscaled H_qk, fidelity vs DMRG gs, fidelity vs ED gs).
    The transpose converts qiskit's little-endian ordering to the MPS ordering."""
    N = int(np.log2(len(sys_vec)))
    E = float(np.real(Statevector(sys_vec).expectation_value(H_qk)))
    t = np.transpose(sys_vec.reshape([2] * N),
                     tuple(range(N - 1, -1, -1))).reshape(-1)
    return (E,
            float(abs(np.vdot(psi0_dmrg, t)) ** 2),
            float(abs(np.vdot(psi0_ed, t)) ** 2))


# ======================================================================
# 7. Resource counts
# ======================================================================
def flatten_success_path(circ):
    """Inline every IfElseOp's true-body, leaving the all-zeros (post-selected)
    path with no control flow."""
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


def resource_costs(circuit, label="", flatten=False, verbose=True, basis=BASIS):
    """Transpile to `basis` (opt level 3) and count gates. flatten=True for the
    rodeo circuit. Slow for large step counts: prefer cx_depth_per_step()."""
    circ = flatten_success_path(circuit) if flatten else circuit
    rc = transpile(circ, basis_gates=list(basis) + ["measure", "reset"],
                   optimization_level=3)
    counts = rc.count_ops()
    tidy = {g: int(counts.get(g, 0)) for g in basis}
    mid = {g: int(counts[g]) for g in ("measure", "reset") if g in counts}
    other = {g: int(c) for g, c in counts.items()
             if g not in basis and g not in ("measure", "reset", "barrier")}
    out = {**tidy, "total": sum(tidy.values()), "depth": rc.depth(),
           "measure": mid.get("measure", 0), "reset": mid.get("reset", 0),
           "other": other}
    if verbose:
        print(f"\n--- resource costs: {label} ---")
        print("  " + "  ".join(f"{g.upper()}={tidy[g]}" for g in basis)
              + f"  | total={out['total']}  depth={out['depth']}"
              + (f"  | mid-circuit: {mid}" if mid else ""))
        if other:
            print(f"  [warning] non-basis ops left after transpile: {other}")
    return out


def cx_depth_per_step(H, order=1, basis=BASIS):
    """(CX count, depth) of ONE Trotter step of one pulse, transpiled once.
    Multiply by the total step count for an estimate (ignores cancellation
    across step boundaries)."""
    N = H.num_qubits
    qc = QuantumCircuit(N + 1)
    apply_filter_pulse(qc, list(range(N)), N, H, 1.0, 0.0,
                       trotter_steps=1, order=order)
    rc = transpile(qc, basis_gates=list(basis), optimization_level=3)
    return rc.count_ops().get("cx", 0), rc.depth()


# ======================================================================
# 8. Convention check (run before trusting the circuit)
# ======================================================================
def verify_pulse_convention():
    """Single-pulse circuit vs FilterBuilder.apply_filter on a 2-level toy."""
    energies = np.array([0.0, 1.0])
    H = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])   # eigvals 0, 1
    t_i, phi_i = 0.7, 0.3
    psi0 = np.array([0.8, 0.6])

    f0_classical, _ = FilterBuilder.apply_filter(
        np.array([t_i]), np.array([phi_i]), energies, psi0)

    qc = QuantumCircuit(2)                       # qubit 0 = sys, qubit 1 = anc
    qc.initialize(psi0, 0)
    apply_filter_pulse(qc, [0], 1, H, t_i, phi_i)
    sv = Statevector(qc)
    f0_quantum = np.real(sv.data[:2])            # anc = 0 block
    f0_quantum = f0_quantum / np.linalg.norm(f0_quantum)

    print("classical f0:", f0_classical)
    print("quantum   f0:", f0_quantum)
    print("match:", np.allclose(np.abs(f0_classical), np.abs(f0_quantum),
                                atol=1e-6))
