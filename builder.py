import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit.synthesis import LieTrotter, SuzukiTrotter
import quimb as qu
import quimb.tensor as qtn
import time
import scipy.sparse as sp
from get_energies import get_energies

# ----------------------------------------------------------------------
# Helper functions (unchanged)
# ----------------------------------------------------------------------
def fixtimes(times, totaltime):
    xf = totaltime / np.sum(np.abs(times))
    times[:] = xf * times
    return times

def unpack(timesphases):
    ndouble = len(timesphases)
    n = ndouble // 2
    return timesphases[:n].copy(), timesphases[n:].copy()

def timesconstraints(timesphases, total_time):
    times, _ = unpack(timesphases)
    return abs(np.sum(times) - total_time)

def probability_constraints(timesphases, energies):
    _, phases = unpack(timesphases)
    return np.prod(np.cos(phases)**2) - 0.9

def probability_constraintsb(timesphases, energies, pos):
    times, phases = unpack(timesphases)
    return np.prod((np.cos(energies[pos] * times + phases))**2) - 0.9

def new_func_v3(timesphases, energies):
    ndouble = len(timesphases)
    n = ndouble // 2
    times = timesphases[:n]
    phases = timesphases[n:]
    num = np.prod(np.cos(phases))
    cos_vals = np.cos(energies[:, None] * times + phases)
    den = np.prod(cos_vals**2, axis=1).sum()
    return abs(1 - num / np.sqrt(den))

def new_func_v4(timesphases, energies):
    ndouble = len(timesphases)
    n = ndouble // 2
    times = timesphases[:n]
    phases = timesphases[n:]
    num = 1.0
    cos_vals = np.cos(energies[:, None] * times)
    den = np.prod(cos_vals**2, axis=1).sum()
    return abs(1 - num / np.sqrt(den))

# ----------------------------------------------------------------------
# FilterBuilder with evaluation & plotting
# ----------------------------------------------------------------------
class FilterBuilder:
    _METHODS = {
        "v3":  (new_func_v3,  probability_constraints,  False),
        "v3b": (new_func_v3,  probability_constraintsb, True),
        "v4":  (new_func_v4,  None,                     False),
    }

    def __init__(
        self,
        total_time,
        energies,
        a=4,
        b=15,
        optimizer="SLSQP",
        maxiter=5000,
        ftol=1e-12,
    ):
        self.total_time = float(total_time)
        self.energies = np.asarray(energies, dtype=float)
        self.a = int(a)
        self.b = int(b)
        self.optimizer = optimizer
        self.opts = {"maxiter": maxiter, "ftol": ftol}

    # ------------------------------------------------------------------
    def build(self, method="v4") -> List[Dict]:
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {list(self._METHODS)}")

        objective, extra_constr, needs_pos = self._METHODS[method]
        results = []

        for ntimes in range(self.a, self.b + 1):
            # initial guess
            times = np.ones(ntimes)
            for i in range(1, ntimes):
                times[i] = times[i - 1] / 2.0
            times = fixtimes(times, self.total_time)

            timesphases = np.zeros(2 * ntimes)
            timesphases[:ntimes] = times

            # bounds
            bnd_times = [(0.0, self.total_time / 3.0)] * ntimes
            bnd_phases = [(-np.pi / 2, np.pi / 2)] * ntimes
            bnds = bnd_times + bnd_phases

            # constraints
            constraints = [
                {"type": "eq", "fun": timesconstraints, "args": (self.total_time,)}
            ]
            if extra_constr is not None:
                if needs_pos:
                    constraints.append(
                        {"type": "ineq", "fun": extra_constr,
                         "args": (self.energies, 0)}
                    )
                else:
                    constraints.append(
                        {"type": "ineq", "fun": extra_constr,
                         "args": (self.energies,)}
                    )

            # optimize
            res = opt.minimize(
                objective,
                x0=timesphases,
                args=(self.energies,),
                method=self.optimizer,
                bounds=bnds,
                constraints=constraints,
                options=self.opts,
                tol=1e-13,
            )

            times_opt = res.x[:ntimes]
            phases_opt = res.x[ntimes:]

            results.append({
                "ntimes": ntimes,
                "times": times_opt.copy(),
                "phases": phases_opt.copy(),
                "fun": float(res.fun),
                "success": bool(res.success),
                "message": str(res.message),
                "result": res  # full scipy result
            })

            print(f"ntimes={ntimes:2d}  time={times_opt.sum():.6f}  fun={res.fun:.3e}  success={res.success}")

        return results

    # ------------------------------------------------------------------
    @staticmethod
    def apply_filter(times: np.ndarray, phases: np.ndarray, energies: np.ndarray, state: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Apply pulse sequence to a state (implements filterprint logic).
        Returns (fdiff, filtered_state)
        """
        f0 = state.copy()
        arg = np.zeros(len(energies))
        xscale = np.zeros(len(energies))

        for i in range(len(times)):
            arg[:] = times[i] * energies[:]
            xscale[:] = np.cos(phases[i]) * np.cos(arg[:]) - np.sin(phases[i]) * np.sin(arg[:])
            f0[:] = f0[:] * xscale[:]

        fnorm = 1.0 / np.sqrt(np.sum(f0**2))
        f0[:] = f0[:] * fnorm

        return f0, fnorm

    # ------------------------------------------------------------------
    def evaluate(
        self,
        results: List[Dict],
        gs_state: np.ndarray,
        trial_state: np.ndarray,
        plot: bool = True,
        highlight_pos: Optional[int] = None,
        ax: Optional[plt.Axes] = None,
    ) -> List[Dict]:
        """
        Evaluate all optimized filters on the trial state.

        Parameters
        ----------
        results : list of dicts from .build()
        gs_state : ground state vector (length = N_en)
        trial_state : initial trial state
        plot : whether to plot filtered states
        highlight_pos : index in energies to draw a red line
        ax : matplotlib axis (optional)

        Returns
        -------
        eval_results : list of dicts with fdiff, f0, etc.
        """
        if ax is None and plot:
            fig, ax = plt.subplots(figsize=(10, 6))

        eval_results = []
        for res in results:
            times = res["times"]
            phases = res["phases"]
            f0, fnorm = self.apply_filter(times, phases, self.energies, trial_state.copy())
            fdiff = np.sum((gs_state - f0) ** 2)

            eval_results.append({
                "ntimes": res["ntimes"],
                "fdiff": fdiff,
                "f0": f0.copy(),
                "norm": fnorm,
                "times": times.copy(),
                "phases": phases.copy(),
            })

            print(f"ntimes={res['ntimes']:2d}  totaltime={times.sum():.6f}  fdiff={fdiff:.6e}")

            if plot:
                ax.plot(self.energies, f0, label=f"{res['ntimes']} pulses")

        if plot:
            if highlight_pos is not None:
                ax.axvline(x=self.energies[highlight_pos], color='r', linestyle='-', alpha=0.7, label=f'E[{highlight_pos}]')
            ax.set_ylim(-1, 1)
            ax.set_xlabel("Energy")
            ax.set_ylabel("Filtered State Amplitude")
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return eval_results

    # ------------------------------------------------------------------
    def build_and_evaluate(
        self,
        method="v4",
        gs_state=None,
        trial_state=None,
        highlight_pos: int = 100,
        plot: bool = True
    ):
        """
        One-liner: build → evaluate → (optionally) plot.
        """
        results = self.build(method)

        if gs_state is None or trial_state is None:
            N = len(self.energies)
            gs_state = np.zeros(N)
            gs_state[0] = 1.0

            trial_state = np.zeros(N)
            trial_state[0] = self.overlap
            trial_state[1:] = np.sqrt((1 - self.overlap**2) / (N - 1))

            print(f"check norm trial state: {np.linalg.norm(trial_state):.6f}")
            print(f"check overlap with gs: {np.dot(gs_state, trial_state):.6f} (target: {self.overlap})")

        return self.evaluate(results, gs_state, trial_state, plot=plot, highlight_pos=highlight_pos)




# ----------------------------------------------------------------------
# 1. J1-J2 Heisenberg Hamiltonian as a SparsePauliOp (open boundary)
# ----------------------------------------------------------------------
def j1j2_hamiltonian(N, j1=1.0, j2=0.3):
    """S_i . S_j = 1/4 (XX + YY + ZZ) for spin-1/2. Open boundary, matches
    the r=1 / r=2 structure of mpo_two_body_heisenberg in builder.py."""
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


# ----------------------------------------------------------------------
# 2. Trotterized e^{-iHt} as a controllable Gate
# ----------------------------------------------------------------------
def evolution_gate(H, t, trotter_steps=4, order=2):
    synth = SuzukiTrotter(order=order, reps=trotter_steps) if order > 1 \
        else LieTrotter(reps=trotter_steps)
    return PauliEvolutionGate(H, time=t, synthesis=synth)


# ----------------------------------------------------------------------
# 3. One filter pulse: cos(H t_i + phi_i) via ancilla + Hadamard test
# ----------------------------------------------------------------------
def apply_filter_pulse(qc, sys_qubits, anc, H, t_i, phi_i, trotter_steps=4):
    """
    ancilla must enter in |0>. After this call, measure + reset `anc` and
    post-select on 0 -- that's the "success" branch of this pulse.

    U_i   = e^{i phi_i} * e^{i H t_i}  = e^{i phi_i} * PauliEvolutionGate(H, -t_i)
    U_i^d = e^{-i phi_i} * e^{-i H t_i} = e^{-i phi_i} * PauliEvolutionGate(H, t_i)
    """
    fwd = evolution_gate(H, -t_i, trotter_steps).control(1)   # e^{+iHt_i}
    bwd = evolution_gate(H, t_i, trotter_steps).control(1)    # e^{-iHt_i}

    qc.h(anc)
    qc.rz(2 * phi_i, anc)              # imprints e^{-i phi} on |0>, e^{+i phi} on |1>
                                        # -- verify sign with verify_pulse_convention()

    qc.append(fwd, [anc, *sys_qubits])         # ancilla=1 branch -> forward evolution
    qc.x(anc)
    qc.append(bwd, [anc, *sys_qubits])         # ancilla=0 (flipped to 1) -> backward evolution
    qc.x(anc)

    qc.h(anc)


# ----------------------------------------------------------------------
# 4. Chain a full optimized pulse sequence
# ----------------------------------------------------------------------
def build_filter_circuit(H, times, phases, trial_prep=None, trotter_steps=4):
    """
    trial_prep(qc, sys_reg): optional callback to prepare |trial_state> on the
    system register before filtering (e.g. your DMRG/variational trial state,
    loaded via `initialize` or a state-prep circuit).
    """
    N = H.num_qubits
    sys = QuantumRegister(N, "sys")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(len(times), "anc_meas")
    qc = QuantumCircuit(sys, anc, creg)

    if trial_prep is not None:
        trial_prep(qc, sys)

    for k, (t_i, phi_i) in enumerate(zip(times, phases)):
        apply_filter_pulse(qc, sys, anc[0], H, t_i, phi_i, trotter_steps)
        qc.measure(anc[0], creg[k])
        qc.reset(anc[0])

    return qc


# ----------------------------------------------------------------------
# 4b. Rodeo-style variant: single ancilla, mid-circuit measure+reset each
#     cycle, EARLY ABORT the remaining cycles the moment the ancilla reads 1
#     (standard Rodeo Algorithm behaviour -- don't waste circuit depth /
#     accumulate noise on a shot that has already failed post-selection).
#     Requires a backend/simulator that supports dynamic circuits (if_test).
# ----------------------------------------------------------------------
def build_rodeo_circuit(H, times, phases, trial_prep=None, trotter_steps=4):
    N = H.num_qubits
    sys = QuantumRegister(N, "sys")
    anc = QuantumRegister(1, "anc")
    creg = ClassicalRegister(len(times), "cycle")
    qc = QuantumCircuit(sys, anc, creg)

    if trial_prep is not None:
        trial_prep(qc, sys)

    def recurse(k):
        if k == len(times):
            return
        t_i, phi_i = times[k], phases[k]
        apply_filter_pulse(qc, sys, anc[0], H, t_i, phi_i, trotter_steps)
        qc.measure(anc[0], creg[k])
        qc.reset(anc[0])
        with qc.if_test((creg[k], 0)):   # only continue if this cycle succeeded
            recurse(k + 1)

    recurse(0)
    return qc, creg


# ----------------------------------------------------------------------
# 5. Sanity check against the classical FilterBuilder.apply_filter
#    Run this BEFORE trusting the circuit on the real spin Hamiltonian.
# ----------------------------------------------------------------------
def verify_pulse_convention():
    from builder import FilterBuilder  # your existing module

    # toy 2-level diagonal "Hamiltonian" so classical vs quantum is easy to compare
    energies = np.array([0.0, 1.0])
    H = SparsePauliOp.from_list([("I", 0.5), ("Z", -0.5)])  # eigvals 0, 1 on |0>,|1>

    t_i, phi_i = 0.7, 0.3
    psi0 = np.array([0.8, 0.6])  # trial_state, already normalized

    # classical reference
    f0_classical, _ = FilterBuilder.apply_filter(np.array([t_i]), np.array([phi_i]),
                                                  energies, psi0.copy())

    # quantum circuit, single pulse, post-select ancilla = 0
    sys = QuantumRegister(1, "sys")
    anc = QuantumRegister(1, "anc")
    qc = QuantumCircuit(sys, anc)
    qc.initialize(psi0, sys[0])
    apply_filter_pulse(qc, sys, anc[0], H, t_i, phi_i)

    sv = Statevector(qc)
    probs = sv.probabilities_dict()
    # amplitude on system register conditioned on anc = 0
    amp0 = sv.data[0] * 2  # index 0 -> anc=0, sys=0 (little-endian anc is top bit)
    amp1 = sv.data[1] * 2  # anc=0, sys=1
    f0_quantum = np.real([amp0, amp1])
    f0_quantum /= np.linalg.norm(f0_quantum)

    print("classical f0:", f0_classical)
    print("quantum   f0:", f0_quantum)
    print("match:", np.allclose(np.abs(f0_classical), np.abs(f0_quantum), atol=1e-6))


def build_sparse_j1j2(N, j1=1.0, j2=0.0):
    """Sparse J1-J2 Heisenberg Hamiltonian via quimb's qtn MPO builder,
    converted to a dense array then sparsified. Note: this is the SAME
    builder used for the DMRG MPO, so it no longer independently catches
    MPO-construction bugs the way a from-scratch build would — it only
    guards against DMRG-solver bugs (bad convergence, wrong penalty, etc.).
    Fine for N <= ED_CHECK_MAX_N where dense conversion is cheap."""
    H_mpo = MPO_ham_j1j2(N, j1=j1, j2=j2, cyclic=False)
    H_dense = H_mpo.to_dense()
    return sp.csr_matrix(H_dense)

def mpo_two_body_heisenberg(N, J, r, S=0.5):
    """MPO for J * sum_i S_i . S_{i+r}, open boundary conditions.
    r=1 gives standard nearest-neighbor coupling (bond dim 5, same as
    quimb's built-in MPO_ham_heis); r=2 gives next-nearest-neighbor."""
    ops = [np.asarray(qu.spin_operator(a, S=S)) for a in ('x', 'y', 'z')]
    d = ops[0].shape[0]
    I = np.eye(d)

    n_states = 2 + 3 * r          # start + (r waiting states per component) + finish
    FIN = n_states - 1

    def idx(a, k):                 # a: component 0,1,2 ; k: 1..r
        return 1 + a * r + (k - 1)

    W = np.zeros((n_states, n_states, d, d), dtype=complex)
    W[0, 0] = I                    # "nothing happened yet" propagates
    W[FIN, FIN] = I                # "term already closed" propagates
    for a in range(3):
        W[0, idx(a, 1)] = ops[a]                    # apply operator, start counting
        for k in range(1, r):
            W[idx(a, k), idx(a, k + 1)] = I          # pass identity while waiting
        W[idx(a, r), FIN] = J * ops[a]               # close the term at distance r

    arrays = []
    for i in range(N):
        if i == 0:
            arrays.append(W[0, :, :, :])       # (r, u, d)
        elif i == N - 1:
            arrays.append(W[:, FIN, :, :])     # (l, u, d)
        else:
            arrays.append(W)                    # (l, r, u, d)
    return qtn.MatrixProductOperator(arrays, shape='lrud')
def MPO_ham_j1j2(N, j1=1.0, j2=0.5, cyclic=False):
    if cyclic:
        raise NotImplementedError("this builder is open-boundary only")
    H1 = mpo_two_body_heisenberg(N, j1, r=1)   # nearest neighbor
    H2 = mpo_two_body_heisenberg(N, j2, r=2)   # next-nearest neighbor
    H = H1 + H2
    H.compress(cutoff=1e-12)                   # keeps combined bond dim tight
    return H
    return H
def get_spectrum(N, j1=1.0, j2=0.0, source="dmrg", run_ed_check=True):
    """
    Source the energy spectrum FilterBuilder needs, from either DMRG
    (default) or ED (exact, N <= ED_CHECK_MAX_N).

    source="dmrg": uses exactly the 3 energies the DMRG script gives you
      (E0, E1, E_top) as the spectrum, rescaled to E0->0, E_top->1.
    source="ed": uses the full exact spectrum (2**N states).

    Returns dict: energies (rescaled, ascending), gap, totaltime,
    E0/E1/Etop (raw), source, ed (cross-check dict, if run).
    """
    if source not in ("dmrg", "ed"):
        raise ValueError("source must be 'dmrg' or 'ed'")

    H = MPO_ham_j1j2(N, j1=j1, j2=j2)

    if source == "ed":
        if N > ED_CHECK_MAX_N:
            raise ValueError(
                f"source='ed' requires N <= ED_CHECK_MAX_N={ED_CHECK_MAX_N}, got N={N}"
            )
        print(f"[source=ed] exact diagonalization, N={N} (dim=2^{N}={2**N})")
        H_dense = H.to_dense()
        evals = np.linalg.eigvalsh(H_dense)     # ascending
        E0, E1, Etop = evals[0], evals[1], evals[-1]
        energies = (evals - E0) / (Etop - E0)
        ed_result = None                         # it *is* the ED result

    else:  # source == "dmrg"
        print(f"[source=dmrg] DMRG spectrum (E0, E1, E_top only), N={N}")
        res = get_energies(H, N=N, j1=j1, j2=j2, run_ed_check=run_ed_check)
        E0, E1, Etop = res["E0"], res["E1"], res["E_top"]
        ed_result = res["ed"]                    # cross-check, if it ran

        # exactly the 3 DMRG energies, rescaled -- no synthetic bulk
        energies = np.array([
            0.0,
            (E1 - E0) / (Etop - E0),
            1.0,
        ])

    gap = energies[1]
    totaltime = np.pi / gap

    return dict(energies=energies, gap=gap, totaltime=totaltime,
                E0=E0, E1=E1, Etop=Etop, source=source, ed=ed_result)


# --- wire into FilterBuilder ---
def run_filter(N, j1=1.0, j2=0.0, source="dmrg", overlap=0.34,
                method="v3", a=8, b=10, highlight_pos=1, plot=True):
    spec = get_spectrum(N, j1=j1, j2=j2, source=source)
    energies, totaltime = spec["energies"], spec["totaltime"]
    num_states = len(energies)

    builder = FilterBuilder(
        total_time=totaltime,
        energies=energies,
        overlap=overlap,
        a=a, b=b
    )

    gs_state = np.zeros(num_states); gs_state[0] = 1.0
    trial_state = np.zeros(num_states)
    trial_state[0]  = overlap
    trial_state[1:] = np.sqrt((1 - overlap**2) / (num_states - 1))

    start = time.time()
    eval_results = builder.build_and_evaluate(
        method=method,
        gs_state=gs_state,
        trial_state=trial_state,
        highlight_pos=highlight_pos,
        plot=plot
    )
    print(f"[{source}] total time: {time.time() - start:.2f} sec")
    return spec, eval_results