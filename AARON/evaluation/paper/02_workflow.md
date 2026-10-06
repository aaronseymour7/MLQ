## 4. The DMRG-to-filter workflow as implemented

The workflow in `AARON/j1j2_filter` has five stages. For each, we record *what the code does*, *what it certifies* and *what it merely assumes*.

| Stage | Module | What it produces | Status of the output |
|---|---|---|---|
| 1. Hamiltonian | `hamiltonians.py` | Pauli form (qiskit) and MPO (quimb); `check_mpo_matches_pauli` compares them for $N\le10$ | Verified to $10^{-9}$ |
| 2. DMRG | `get_energies.py` | $E_0$, MPS $|\psi_0\rangle$ (two-site DMRG, $\chi\le64$); $E_{\rm top}$ from DMRG on $-H$; $E_1$ from DMRG on $H+\lambda|\psi_0\rangle\langle\psi_0|$, $\lambda=1.1(E_{\rm top}-E_0)$ | *Estimates*, not bounds |
| 3. Spectral inputs | `core/spectrum.py` | shift, $W$, scaled gap $\Delta$, $e_0$ | ED path: exact ($N\le12$). DMRG path: $W$ certified, $\Delta$ **not**, $e_0$ **indicative only** |
| 4. Trial state | `mps_to_circuit` (external) | $L$-layer brickwork circuit approximating $|\psi_0\rangle$; $\gamma$ computed by overlap with the ED ground state if available, else with the DMRG MPS | Certified only on the ED path |
| 5. Filter + Trotter | `floor.py`, `core/filter_design.py`, `core/trotter.py`, `pipeline.py` | $(t_i,\phi_i,k_i)$, certified $\eta$, a-priori $n$, rigorous $1-F$ bound, CX/T counts | Rigorous *conditional on stages 3–4* |

### 4.1 Where DMRG enters, and what depends on it

1. **Trial state.** The MPS is compiled to an $L$-layer circuit. $L$ controls $\gamma$ and the circuit cost; the filter exists to repair $1-\gamma$.
2. **Spectral inputs.** In the default configuration (`SPECTRUM_SOURCE="ed"`) *no DMRG number enters the certificate*: $E_0,E_1,E_{\rm top}$ come from dense ED and DMRG is used only for the trial circuit and as a cross-check (`ed_cross_check`). This restricts every certified result to $N\le12$, where DMRG is not needed. With `SPECTRUM_SOURCE="dmrg"` the scaled window $[\Delta,1]$ is built from DMRG quantities, and the guarantees of §3 become conditional on three assumptions that DMRG cannot certify:
   * (A1) $E_1^{\rm DMRG}\le E_1$ (so that $\Delta$ is a *lower* bound on the true gap). The penalty method returns an energy of a state orthogonal to an *approximate* $|\psi_0\rangle$, i.e. an *upper* bound on $E_1$ when $\psi_0$ is exact; the wrong side for a certificate (§6.4 quantifies the consequence).
   * (A2) $\gamma$ is a lower bound. A DMRG–circuit overlap is an overlap with an approximation of $|E_0\rangle$.
   * (A3) the scaled ground energy is within $e_0$ of zero; `get_spectrum` sets $e_0=\sqrt{\operatorname{Var}H}/W$, which the code itself calls “indicative” (some eigenvalue lies within $\sqrt{\operatorname{Var}}$ of $\langle H\rangle$, not necessarily the lowest).

   The bandwidth, in contrast, *is* certified: $W=(c_I+\sum|c_\beta|)-E_0^{\rm DMRG}\ge E_{\rm top}-E_0^{\rm DMRG}$.
3. **Hilbert-space dimension.** Every number in stage 5 (exact reference, Trotter simulation, trial-circuit statevector) is computed with dense $2^{N}$ vectors, so the pipeline never exploits DMRG's ability to reach large $N$. It is a *verification harness* for $N\lesssim 14$, not (yet) a large-$N$ workflow.

### 4.2 Filter design (“floor” problem)

For fixed $(x{=}T\Delta/\pi,\,m)$ the solver (`floor.solve_floor`) maximises $F(0)^2$ subject to certified $\eta\le\eta_*$ using SLSQP with analytic Jacobians, a ladder of feasibility guards, a cutting-plane refinement of the constraint grid and a certification fixed point. Across $(x,m)$ and leakage budgets $\varepsilon_\ell=f\varepsilon$, $f\in\{0.4,\dots,0.01\}$, the pipeline picks the candidate minimising $n_{\rm req}/p_{g}^{\rm lb}$. The continuous design is then snapped to $n$ equal steps (largest-remainder), zero-length pulses are dropped (a $t=0$ pulse is a scalar $\cos\phi$ after post-selection, so $\eta$ and the state are unchanged), phases are re-optimised at the snapped times and $\eta$ is re-certified.

### 4.3 Two reported operating points

* **Guaranteed**: smallest $n$ (5 % search) whose *a-priori* bound $\varepsilon_{\rm bound}\le\varepsilon$.
* **Empirical**: smallest $n$ (sweep + bisection) whose *measured* infidelity to the ED ground state is $\le\varepsilon$.

The second is an oracle quantity (it uses the exact ground state that the algorithm is supposed to produce) and, as §6.2 shows, it can sit in a regime where the Trotterised filter is not an approximation of the designed filter at all.
