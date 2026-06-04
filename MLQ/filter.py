import numpy as np
from scipy import optimize as opt
import matplotlib.pyplot as plt
from typing import List, Dict, Optional, Tuple
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.synthesis import LieTrotter          # or SuzukiTrotter
from qiskit.quantum_info import SparsePauliOp



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

def probability_constraints(timesphases, energies, overlap):
    _, phases = unpack(timesphases)
    return np.prod(np.cos(phases)**2) - 0.9

def probability_constraintsb(timesphases, energies, overlap, pos):
    times, phases = unpack(timesphases)
    return np.prod((np.cos(energies[pos] * times + phases))**2) - 0.9

def new_func_v3(timesphases, energies, overlap):
    ndouble = len(timesphases)
    n = ndouble // 2
    times = timesphases[:n]
    phases = timesphases[n:]
    num = np.prod(np.cos(phases))
    cos_vals = np.cos(energies[:, None] * times + phases)
    den = np.prod(cos_vals**2, axis=1).sum()
    return abs(1 - num / np.sqrt(den))

def new_func_v4(timesphases, energies, overlap):
    ndouble = len(timesphases)
    n = ndouble // 2
    times = timesphases[:n]
    phases = timesphases[n:]
    num = 1.0
    cos_vals = np.cos(energies[:, None] * times)
    den = np.prod(cos_vals**2, axis=1).sum()
    return abs(1 - num / np.sqrt(den))


def new_func_v5(coeffs_sq, energies):
    """
    coeffs_sq : |<psi_k|psi_UCJ>|^2, shape (dim,)
    energies  : E_k, shape (dim,)
    Returns a scalar objective to minimize (0 = perfect filter).
    """
    def objective(timesphases):
        n = len(timesphases) // 2
        times  = timesphases[:n]
        phases = timesphases[n:]

        # cos(E_k * t_i + phi_i), shape (dim, n)
        cos_vals = np.cos(energies[:, None] * times + phases)

        # filter weight for each eigenstate: prod_i cos(...)^2
        weights = np.prod(cos_vals**2, axis=1)   # shape (dim,)

        # filtered overlap with GS (k=0) vs total
        gs_weight  = coeffs_sq[0] * weights[0]
        total      = np.dot(coeffs_sq, weights)

        return 1.0 - gs_weight / (total + 1e-30)

    return objective
# ----------------------------------------------------------------------
# FilterBuilder with evaluation & plotting
# ----------------------------------------------------------------------
class FilterBuilder:
    _METHODS = {
        "v3":  (new_func_v3,  probability_constraints,  False),
        "v3b": (new_func_v3,  probability_constraintsb, True),
        "v4":  (new_func_v4,  None,                     False),
        "v5":  (new_func_v5,  None,                     False),  # sentinel

    }

    def __init__(self, total_time, energies, overlap, a=4, b=15, 
                 optimizer="SLSQP", maxiter=5000, ftol=1e-12, coeffs_sq=None):

        self.total_time = float(total_time)
        self.energies = np.asarray(energies, dtype=float)
        self.overlap = float(overlap)
        self.a = int(a)
        self.b = int(b)
        self.optimizer = optimizer
        self.opts = {"maxiter": maxiter, "ftol": ftol}
        self.coeffs_sq = coeffs_sq 
    # ------------------------------------------------------------------
    def build(self, method="v4") -> List[Dict]:
        if method not in self._METHODS:
            raise ValueError(f"method must be one of {list(self._METHODS)}")

        if method == "v5":
            if self.coeffs_sq is None:
                raise ValueError("v5 requires coeffs_sq passed to FilterBuilder()")
            objective = new_func_v5(self.coeffs_sq, self.energies)
            extra_constr, needs_pos = None, False
        else:
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
            bnd_times  = [(0.0, self.total_time / 3.0)] * ntimes
            bnd_phases = [(-np.pi / 2, np.pi / 2)]      * ntimes
            bnds       = bnd_times + bnd_phases

            # constraints
            constraints = [
                {"type": "eq", "fun": timesconstraints, "args": (self.total_time,)}
            ]
            if extra_constr is not None:
                if needs_pos:
                    constraints.append(
                        {"type": "ineq", "fun": extra_constr,
                        "args": (self.energies, self.overlap, 0)}
                    )
                else:
                    constraints.append(
                        {"type": "ineq", "fun": extra_constr,
                        "args": (self.energies, self.overlap)}
                    )

            # ── optimize ──────────────────────────────────────────────────────────
            if method == "v5":
                # objective is already a closure; pass NO args
                res = opt.minimize(
                    objective,
                    x0=timesphases,
                    method=self.optimizer,
                    bounds=bnds,
                    constraints=constraints,
                    options=self.opts,
                    tol=1e-13,
                )
            else:
                res = opt.minimize(
                    objective,
                    x0=timesphases,
                    args=(self.energies, self.overlap),
                    method=self.optimizer,
                    bounds=bnds,
                    constraints=constraints,
                    options=self.opts,
                    tol=1e-13,
                )

            times_opt  = res.x[:ntimes]
            phases_opt = res.x[ntimes:]

            results.append({
                "ntimes":  ntimes,
                "times":   times_opt.copy(),
                "phases":  phases_opt.copy(),
                "fun":     float(res.fun),
                "success": bool(res.success),
                "message": str(res.message),
                "result":  res,
            })

            print(f"ntimes={ntimes:2d}  time={times_opt.sum():.6f}  "
                  f"fun={res.fun:.3e}  success={res.success}")

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
            fidelity = float(np.dot(gs_state, f0))**2   # should approach 1
            fdiff = 1.0 - fidelity

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

  





# --- Assume you already have: ---
#   qc        : your existing QuantumCircuit (state prep, etc.)
#   hamiltonian : your SparsePauliOp or Pauli Hamiltonian
#   n_qubits  : number of qubits

def append_filter(qc: QuantumCircuit, hamiltonian, times, phases, trotter_steps=1):
    """
    Append the filter pulse sequence to an existing Qiskit circuit.
    
    Each pulse is:
      1. Hamiltonian time evolution for `times[i]`
      2. Global phase rotation by `phases[i]`
    """
    n = qc.num_qubits

    for i, (t, phi) in enumerate(zip(times, phases)):
        # --- Step 1: time evolution e^{-i H t} ---
        evo_gate = PauliEvolutionGate(
            hamiltonian,
            time=float(t),
            synthesis=LieTrotter(reps=trotter_steps)
        )
        qc.append(evo_gate, range(n))

        # --- Step 2: phase rotation R(phi) ---
        # Global phase isn't observable, so apply as relative phase
        # on your "ground state" qubit/subspace.
        # Option A — if your GS lives in the |0...0⟩ subspace:
        qc.p(2 * float(phi), 0)          # single-qubit phase on qubit 0

        # Option B — if you need a full global phase (rarely needed):
        # qc.global_phase += float(phi)

        qc.barrier(label=f"pulse {i}")

    return qc


