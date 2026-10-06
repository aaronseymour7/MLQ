"""Filter design: evaluation, certification, minimax optimization, grid snapping."""


import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize as opt
from typing import Dict, List


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


def certify_eta(times, phases, delta, hi=1.0, e0_slack=0.0, n_grid=200_001):
    """Certified eta = sup_{[delta,hi]}|F| / |F(0)| (grid max + Lipschitz pad)."""
    times = np.asarray(times, float)
    phases = np.asarray(phases, float)
    L = float(np.sum(np.abs(times)))
    grid = np.linspace(delta, hi, n_grid)
    h = grid[1] - grid[0]
    sup = min(float(np.max(np.abs(filter_values(times, phases, grid)))) + L * h / 2, 1.0)
    f0 = float(abs(np.prod(np.cos(phases))))
    f0_lb = max(f0 - L * e0_slack, 0.0)
    eta = sup / f0_lb if f0_lb > 0 else float("inf")
    return dict(eta=eta, sup=sup, f0=f0, f0_lb=f0_lb, lipschitz=L)


def snap_steps(times, n):
    """Integer k_i with sum k_i = n (largest remainder); t_i ~ k_i * T/n."""
    times = np.asarray(times, float)
    T = times.sum()
    x = times / (T / n)
    k = np.floor(x + 1e-12).astype(int)
    for j in np.argsort(-(x - k))[: n - k.sum()]:
        k[j] += 1
    return k, T / n
