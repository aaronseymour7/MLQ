---
title: "Certified ground-state filtering from DMRG trial states — pipeline overview"
subtitle: "J1–J2 Heisenberg chain · Stetcu–Baroni projection · what we built, what we proved, what it costs"
---

# 0. One-paragraph summary

We want the ground state $|E_0\rangle$ of the open spin-½ $J_1$–$J_2$ chain on a quantum computer. **DMRG** (classical) gives a matrix-product state (MPS) that we compile into a shallow circuit; that circuit has ground-state weight $\gamma<1$. We repair it with the **Stetcu–Baroni–Carlson (SBC) projection**: a single ancilla, controlled time evolution, and post-selection implement a *filter* $F(E)=\prod_i\cos(Et_i+\phi_i)$ that suppresses excited states. We optimise the filter classically, **certify** how well it suppresses the spectrum, and control the Trotter error three ways (*certified*, *hybrid*, *oracle*). Result: the mathematics is sound and verified, the rigorous guarantee is $10^2$–$10^3\times$ more conservative than reality, a *hybrid* protocol recovers most of that gap — but for $N\le12$ an MPS-based circuit is still 5–74× cheaper than the filter, so **no resource advantage on these 1D chains**. The value is the certified-filtering method and its verification harness.

![Pipeline](figs/fig0_pipeline.png){width=100%}

# 1. The problem and the physics

| Symbol | Meaning |
|---|---|
| $N$ | number of spins (even; open boundaries) |
| $J_1, J_2$ | nearest / next-nearest-neighbour couplings; we use $J_1=1$, $J_2\in\{0,\,0.2411,\,0.4,\,0.5\}$ |
| $H$ | $\sum_{i}J_1\,\mathbf S_i\!\cdot\!\mathbf S_{i+1}+\sum_iJ_2\,\mathbf S_i\!\cdot\!\mathbf S_{i+2}$, with $\mathbf S_i\!\cdot\!\mathbf S_j=\tfrac14(XX+YY+ZZ)$ → $3(N-1)+3(N-2)$ Pauli terms |
| $E_0,\,E_1,\,E_{\rm top}$ | ground, first-excited and highest energy |
| gap | $E_1-E_0$ (a triplet; shrinks like $1/N$ near criticality) |
| $J_2=0.5$ | Majumdar–Ghosh point: the ground state is a product of dimers, exactly representable by one circuit layer ⇒ $\gamma=1$ (a *sanity check*, not a benchmark) |

The ground state is a non-degenerate singlet for even $N$, the Hamiltonian conserves total spin, and every bond exponential $e^{-i\theta\,\mathbf S_i\cdot\mathbf S_j}$ is SU(2)-symmetric — this matters later for symmetry-resolved certification.

# 2. Stage by stage

## Stage 1–2: Hamiltonian and DMRG (classical)

- `hamiltonians.py` builds the Pauli form (for circuits and ED) and the MPO (for DMRG); `check_mpo_matches_pauli` verifies they agree to $10^{-9}$.
- `get_energies.py` runs two-site DMRG (bond dims $\{8,16,32,64,64\}$, cutoff $10^{-10}$) for $E_0$ and the MPS $|\psi_{\rm DMRG}\rangle$; DMRG on $-H$ gives $E_{\rm top}$; DMRG on $H+\lambda|\psi_0\rangle\langle\psi_0|$ with $\lambda=1.1\,(E_{\rm top}-E_0)$ gives $E_1$ (penalty method).
- Cross-check against exact diagonalisation (ED): DMRG matches $E_0,E_1,E_{\rm top}$ to $\lesssim2\times10^{-9}$ up to $N=16$.
- **Caveat that drives the certification discussion:** DMRG gives *estimates*, not bounds. The penalty-method $E_1$ is variational ($\ge$ true), i.e. on the unsafe side if used as a gap lower bound.

## Stage 3: trial circuit

- `mps_to_circuit` (qiskit-community `mps-to-circuit`) compiles the MPS to an $L$-layer brickwork circuit of two-qubit isometries (`method="approximate"`); the `"exact"` method is a sequential (Schön et al.) preparation of an MPS with bond dimension $\chi$.
- $\gamma=|\langle E_0|\psi_{\rm trial}\rangle|^2$ is the **ground-state weight** of the trial state. It is what the filter must repair ($1-\gamma$ is the "error to remove"). On the certified path $\gamma$ is computed against the ED ground state (exact for $N\le12$).

## Stage 4: scaling the spectrum

$$H_s=\frac{H-E_0}{W},\qquad W\ge E_{\rm top}-E_0,\qquad \operatorname{spec}(H_s)\subset\{0\}\cup[\Delta,1],\quad \Delta=\frac{E_1-E_0}{W}.$$

All pulse times are in units of $W^{-1}$. $W$ drops out of the Trotter step count ($\alpha T^2$ is $W$-independent), so the choice only affects how wide a window $[\Delta,1]$ the filter must suppress. The code can take $W$ from ED (exact), or certify it as $c_I+\sum|c_\beta|-E_0$ (valid but ≈3× too large because $XX{+}YY{+}ZZ$ is bounded term by term; the per-bond bound $E_{\rm top}\le\tfrac14[J_1(N-1)+J_2(N-2)]$ is tighter).

## Stage 5: the filter (the core idea)

One **pulse** acts on the system and one ancilla (prepared in $|0\rangle$):

$$\mathcal W(t,\phi)=\mathsf H_{\rm a}\;e^{-itH_s\otimes Z_{\rm a}}\;\mathrm{Rz}_{\rm a}(2\phi)\;\mathsf H_{\rm a},\qquad \langle0_{\rm a}|\mathcal W|0_{\rm a}\rangle=\cos(H_st+\phi).$$

Measuring the ancilla after each of $m$ pulses and keeping only outcome 0 applies

$$F(E)=\prod_{i=1}^{m}\cos(Et_i+\phi_i)\quad\text{to the trial state},\qquad P_{\rm succ}=\sum_k|c_k|^2F(E_k)^2\;\ge\;\gamma\,F(0)^2 .$$

| Symbol | Meaning |
|---|---|
| $m$ | number of pulses (4–11 in practice) |
| $t_i,\ \phi_i$ | pulse duration and phase (optimised) |
| $T=\sum t_i$ | total scaled evolution time; $x=T\Delta/\pi$ is its natural dimensionless form |
| $\eta$ | $\sup_{E\in[\Delta,1]}|F(E)|\,/\,|F(0)|$ — *how strongly excited states are suppressed relative to the ground state* (smaller is better) |
| $F(0)^2$ | $\prod\cos^2\phi_i$ — the factor multiplying the ground-state success probability |
| $P_{\rm succ}$ | probability that every ancilla reads 0 (otherwise abort and retry; expected cost ÷ $P_{\rm succ}$) |

**Design problem (`floor.py`).** For a target leakage $\varepsilon_\ell$, set $\eta_*=\sqrt{\gamma\varepsilon_\ell/((1-\varepsilon_\ell)(1-\gamma))}$; then choose $(t_i,\phi_i)$ to maximise $F(0)^2$ subject to certified $\eta\le\eta_*$ (SLSQP with analytic Jacobians; sweep over $x$, $m$ and leakage fraction). **Certification (R1):** since $|F'|\le L=\sum|t_i|$ and $|F''|\le L^2$, interval bisection with a second-order envelope gives a *rigorous* upper bound on $\sup|F|$ (tested against $2\times10^6$-point brute force: 0 violations).

## Stage 6: the circuit

Each $e^{-it_iH_s\otimes Z}$ is split into $k_i$ first-order **Lie–Trotter** steps over the Pauli terms (term order preserved). Total steps $n=\sum_ik_i$, step size $dt=T/n$, pulse times snapped to $t_i=k_i\,dt$ (zero-length pulses dropped: after post-selection they are a scalar). One ancilla, mid-circuit measure+reset, *early abort* as soon as an outcome is 1. Per-step cost for $N=6$: 35 CX; total CX $\approx n\times$ step cost $+$ trial CX.

## Stage 7: how do we choose $n$ and claim an error? — three ways

The final state differs from $|E_0\rangle$ for two reasons:

1. **Leakage $\ell$** — the (ideal, exact-evolution) filter does not remove excited states completely.
 R2: $F_{\rm exact}\ge\gamma/(\gamma+(1-\gamma)\eta^2)$, so $\ell\le(1-\gamma)\eta^2/(\gamma+(1-\gamma)\eta^2)$. **Rigorous**, needs $\gamma$ and $\Delta$.
2. **Trotter error** — time evolution is discretised. Worst-case theory (Childs et al.): $\|\mathcal W-\widetilde{\mathcal W}\|\le\alpha t^2/2k$, with $\alpha=\sum_g\|[H_g,\sum_{g'>g}H_{g'}]\|$; telescoping over pulses and renormalising gives $\epsilon_T=\alpha T^2/(2n)$ and a state distance $d_T\le2\epsilon_T/\sqrt{p_g}$ with $p_g=\gamma F(0)^2$.

**Composition.** Original code: $1-F\le(\sqrt{2\ell}+d_T)^2$. Our proven sharpened version: $1-F\le\sin^2\!\big(\arcsin\sqrt\ell+\arcsin(\epsilon_T/\sqrt{p_g})\big)$ — about 3× tighter (verified, 0 violations).

| Mode | Leakage | Trotter error | Uses exact $|E_0\rangle$? | Rigorous? |
|---|---|---|---|---|
| **Certified** ("guaranteed") | R1+R2 | worst-case $\alpha T^2/2n$ | no | **yes** (given $\gamma$, $\Delta$) |
| **Oracle** ("empirical") | simulated | simulated | **yes** | no — a lower envelope, not usable by a practitioner |
| **Hybrid** (our protocol) | R1+R2, **rigorous** | **measured**, Richardson-type | no | leakage yes, Trotter no |

### What "hybrid" means, precisely

*Hybrid = rigorous for the part we can prove tightly, empirical for the part where the proof is too pessimistic.* The leakage half is a theorem (certified $\eta$). The Trotter half is replaced by a convergence test:

1. Pick $n$ from a geometric grid; build the snapped design at $n$ and at $2n$ and run both circuits (classically now; on hardware, compare an observable such as energy at $n$ vs $2n$).
2. $\delta_n=1-|\langle\psi_n|\psi_{2n}\rangle|^2$. Because the first-order error vector scales as $1/n$, $\psi_n-\psi_\infty\approx2(\psi_n-\psi_{2n})$, so the distance to the converged state is $\approx2\sqrt{\delta_n}$.
3. Accept the smallest $n$ with $\big(\sqrt{\ell(n)}+2\sqrt{\delta_n}\big)^2\le\varepsilon$ **and** certified $\eta<1$.

It never looks at the exact ground state (that is used afterwards only to *validate*). It is **not a theorem**: it assumes the asymptotic $1/n$ regime. Why it is needed: the worst-case constant $\alpha$ is attained on high-energy states, but the state we care about feels a first-order coefficient 28–41× smaller ($J_2=0$; 9–13× at $J_2=0.4$), a gap that grows with $N$ (§4.2).

# 3. Variable cheat-sheet

| Symbol | Where | Meaning |
|---|---|---|
| $\gamma$ | trial state | ground-state weight $|\langle E_0|\psi_{\rm trial}\rangle|^2$ |
| $L$ | trial circuit | brickwork layers (more ⇒ larger $\gamma$, more CX) |
| $\chi$ | MPS | bond dimension (sequential-MPS baseline cost grows with $\chi$) |
| $W,\ \Delta$ | scaling | bandwidth; scaled gap $(E_1-E_0)/W$ |
| $m,\ t_i,\ \phi_i,\ T,\ x$ | filter | pulses, times, phases, total time, $T\Delta/\pi$ |
| $\eta,\ F(0)^2$ | filter | suppression of excited states; ground-state amplitude² |
| $\varepsilon,\ \varepsilon_\ell$ | targets | total infidelity target $1-F$; leakage share of it |
| $k_i,\ n,\ dt$ | Trotter | steps per pulse, total steps, step size $T/n$ |
| $\alpha$ | Trotter | worst-case commutator constant |
| $\ell$ | error | leakage (ideal-filter infidelity) |
| $\epsilon_T,\ d_T,\ \delta_n$ | error | Trotter operator error; state distance; Richardson estimate |
| $P_{\rm succ},\ p_g$ | cost | success probability; lower bound $\gamma F(0)^2$ |
| CX | cost | two-qubit gate count including the trial circuit |

# 4. What we found

## 4.1 Correctness (verified)

Pulse convention; R1 certificates (60 random designs vs brute force); R2 ($2\times10^4$ random spectra); $\alpha$ vs dense commutator norms; pulse-level Trotter error ($\le0.24\times$ bound); both composition bounds ($5\times10^4$ random trials) — **no violation anywhere**, and every simulated design respected its bound.

## 4.2 Conservatism: why certified ≫ empirical

At $N=6,\ \varepsilon=10^{-2}$: certified $n=1450$ vs oracle $n=5$ (290×); measured error is ≈270× below the bound's prediction. Decomposition: low-energy suppression ≈28× (the ground state is almost annihilated by the Trotter commutator), pulse-level ≈4×, composition ≈2× ⇒ ≈240×.

| $N$ | $J_2$ | $\alpha$ (worst case) | $\|C\psi_0\|$ (ground state) | $\alpha/\|C\psi_0\|$ | $\alpha/\|C\psi_{E_1}\|$ |
|---|---|---|---|---|---|
| 4 | 0 | 0.2679 | 0.0000 | (=0) | 2.3 |
| 6 | 0 | 0.2141 | 0.0075 | 28.5 | 4.8 |
| 8 | 0 | 0.1713 | 0.0050 | 34.1 | 7.9 |
| 10 | 0 | 0.1417 | 0.0034 | 41.2 | 11.6 |
| 4 | 0.4 | 0.4177 | 0.0574 | 7.3 | 3.5 |
| 6 | 0.4 | 0.3480 | 0.0369 | 9.4 | 6.4 |
| 8 | 0.4 | 0.2832 | 0.0251 | 11.3 | 9.0 |
| 10 | 0.4 | 0.2361 | 0.0184 | 12.8 | 11.2 |

## 4.3 Hybrid protocol vs certified vs oracle

| $N$ | $J_2$ | $\varepsilon$ | $n$ (protocol) | CX | measured $1-F$ | $P_{\rm succ}$ | oracle $n$ / CX | certified $n$ / CX | certified ÷ protocol (CX) |
|---|---|---|---|---|---|---|---|---|---|
| 6 | 0 | 0.1 | 4 | 155 | 0.015 | 0.58 | 2 / 85 | 150 / 5,265 | 34× |
| 6 | 0 | 0.01 | 8 | 295 | 0.00099 | 0.6 | 5 / 190 | 1523 / 53,320 | 181× |
| 6 | 0 | 0.001 | 20 | 715 | 0.00036 | 0.55 | 16 / 575 | 10286 / 360,025 | 504× |
| 8 | 0 | 0.1 | 6 | 315 | 0.023 | 0.53 | 3 / 168 | 555 / 27,216 | 86× |
| 8 | 0 | 0.01 | 20 | 1,001 | 0.00092 | 0.54 | 14 / 707 | 4903 / 240,268 | 240× |
| 8 | 0 | 0.001 | 56 | 2,765 | 0.00022 | 0.54 | 25 / 1,246 | 32653 / 1,600,018 | 579× |
| 10 | 0 | 0.1 | 12 | 783 | 0.0064 | 0.44 | 7 / 468 | 1212 / 76,383 | 98× |
| 10 | 0 | 0.01 | 28 | 1,791 | 0.0013 | 0.51 | 21 / 1,350 | 11060 / 696,807 | 389× |
| 10 | 0 | 0.001 | – † | – | – | – | 50 / 3,177 | – / – | – |
| 6 | 0.4 | 0.1 | 5 | 330 | 0.014 | 0.91 | 3 / 204 | 177 / 11,166 | 34× |
| 6 | 0.4 | 0.01 | 10 | 645 | 0.0028 | 0.55 | 6 / 393 | 882 / 55,581 | 86× |
| 6 | 0.4 | 0.001 | 28 | 1,779 | 0.00053 | 0.49 | 22 / 1,401 | 8323 / 524,364 | 295× |
| 8 | 0.4 | 0.1 | 6 | 567 | 0.015 | 0.88 | 4 / 385 | 518 / 47,159 | 83× |
| 8 | 0.4 | 0.01 | 20 | 1,841 | 0.0046 | 0.53 | 14 / 1,295 | 3128 / 284,669 | 155× |
| 8 | 0.4 | 0.001 | 80 | 7,301 | 0.00043 | 0.8 | 54 / 4,935 | 28715 / 2,613,086 | 358× |
| 10 | 0.4 | 0.1 | 12 | 1,455 | 0.0076 | 0.79 | 5 / 622 | 1051 / 125,096 | 86× |
| 10 | 0.4 | 0.01 | 28 | 3,359 | 0.006 | 0.66 | 22 / 2,645 | 8251 / 981,896 | 292× |
| 10 | 0.4 | 0.001 | 96 | 11,451 | 0.00054 | 0.8 | 73 / 8,714 | 70053 / 8,336,334 | 728× |
| 12 | 0.4 | 0.1 | 12 | 1,797 | 0.023 | 0.67 | 9 / 1,356 | 1979 / 290,946 | 162× |
| 12 | 0.4 | 0.01 | 40 | 5,913 | 0.0064 | 0.53 | 35 / 5,178 | 18004 / 2,646,621 | 448× |
| 12 | 0.4 | 0.001 | 256 | 37,665 | 0.00035 | 0.82 | 122 / 17,967 | 175985 / 25,869,828 | 687× |

## 4.4 Is the filter competitive with classical-then-quantum MPS preparation? (equal-fidelity comparison)

![](figs/fig7_cx_vs_N.png){width=95%}

| $N$ | $J_2$ | $\varepsilon$ | best baseline (CX) | filter: certified (CX) | filter: hybrid protocol (CX) | certified / baseline | hybrid / baseline |
|---|---|---|---|---|---|---|---|
| 6 | 0 | 0.1 | MPS circuit (30) | 4,000 | 155 | 133× | 5.2× |
| 6 | 0 | 0.01 | generic prep (57) | 23,880 | 295 | 419× | 5.2× |
| 6 | 0 | 0.001 | generic prep (57) | 215,260 | 715 | 3776× | 13× |
| 8 | 0 | 0.1 | MPS circuit (63) | 14,861 | 315 | 236× | 5.0× |
| 8 | 0 | 0.01 | sequential MPS (100) | 137,998 | 1,001 | 1380× | 10× |
| 8 | 0 | 0.001 | generic prep (247) | 1,066,499 | 2,765 | 4318× | 11× |
| 10 | 0 | 0.1 | MPS circuit (81) | 43,551 | 783 | 538× | 9.7× |
| 10 | 0 | 0.01 | sequential MPS (137) | 447,822 | 1,791 | 3269× | 13× |
| 10 | 0 | 0.001 | sequential MPS (519) | 3,802,824 | – | 7327× | – |
| 12 | 0 | 0.1 | sequential MPS (174) | 125,532 | – | 721× | – |
| 12 | 0 | 0.01 | sequential MPS (174) | 1,292,390 | – | 7428× | – |
| 12 | 0 | 0.001 | sequential MPS (709) | – | – | – | – |
| 6 | 0.4 | 0.1 | MPS circuit (15) | 9,117 | 330 | 608× | 22× |
| 6 | 0.4 | 0.01 | MPS circuit (30) | 37,215 | 645 | 1240× | 22× |
| 6 | 0.4 | 0.001 | generic prep (57) | 285,372 | 1,779 | 5007× | 31× |
| 8 | 0.4 | 0.1 | MPS circuit (21) | 36,554 | 567 | 1741× | 27× |
| 8 | 0.4 | 0.01 | MPS circuit (84) | 178,878 | 1,841 | 2130× | 22× |
| 8 | 0.4 | 0.001 | sequential MPS (99) | 1,828,526 | 7,301 | 18470× | 74× |
| 10 | 0.4 | 0.1 | MPS circuit (27) | 100,755 | 1,455 | 3732× | 54× |
| 10 | 0.4 | 0.01 | sequential MPS (136) | 823,204 | 3,359 | 6053× | 25× |
| 10 | 0.4 | 0.001 | sequential MPS (136) | 6,707,397 | 11,451 | 49319× | 84× |
| 12 | 0.4 | 0.1 | MPS circuit (33) | 249,852 | 1,797 | 7571× | 54× |
| 12 | 0.4 | 0.01 | sequential MPS (174) | 1,943,145 | 5,913 | 11168× | 34× |
| 12 | 0.4 | 0.001 | sequential MPS (174) | 17,365,356 | 37,665 | 99801× | 216× |

**Reading the table:** at every size/coupling tested, an MPS circuit with bond dimension 4–8 (sequential `mps-to-circuit`) reaches $\varepsilon\lesssim10^{-2}$ with 63–174 CX at $N=6$–$12$, whereas the hybrid filter needs hundreds to thousands of CX and the certified filter $10^{4}$–$10^{7}$. The filter's cost is set by the *gap* ($T\propto1/\Delta$) and Trotter error, **not** by how entangled the state is — so it only wins where MPS preparation gets expensive, which none of these chains do.

## 4.5 DMRG's actual role

DMRG accuracy is excellent ($\sim10^{-9}$ up to $N=16$), but (i) in the certified path the spectral inputs come from ED ($N\le12$), (ii) DMRG's $E_1$ is variational (unsafe for a gap lower bound), (iii) $e_0=\sqrt{\rm Var}/W$ is "indicative", not a bound, and (iv) all downstream simulation uses dense vectors ($N\lesssim14$), so DMRG's large-$N$ reach is unused. For large $N$ the guarantees are *conditional* on three assumptions: gap lower bound, $\gamma$ lower bound, ground-energy offset.

# 5. Code audit (what we fixed / what remains)

Fixed: $\gamma=1$ now means *no filter* (previously a million-CX filter was built for an exact state); `ORDER≠1` raises (bound is first-order only); dense $2^N\times2^N$ check restricted to $N\le12$; `study.py` imports/defaults corrected (guaranteed designs, $J_2$ list, $N\le12$). Open: cost-proxy mismatch in the floor search ($x^2/F(0)^2$ vs the true $x^2/F(0)^3$), early-abort expected cost, over-wide window ($hi=1$), energy-shift rotation per step, verification of T-count constants and citations, tests directory and pinned environment. Full list: paper §6.

# 6. Suggested talk outline (8 slides)

1. **Goal & setting** — SBC projection on top of DMRG; the pipeline figure.
2. **The filter** — pulse circuit, $F(E)=\prod\cos(Et_i+\phi_i)$, $\gamma,\ \eta,\ P_{\rm succ}$.
3. **Certified error analysis** — R1 (suppression), R2 (leakage), R3 (Trotter), composition (+ our sharpening).
4. **Verification** — every ingredient tested against brute force; zero violations.
5. **Why the guarantee is loose** — table of §4.2; ground state feels 28–41× less Trotter error.
6. **Hybrid protocol** — definition, validation, 30–600× fewer CX than certified.
7. **Baselines** — equal-fidelity comparison; MPS preparation wins for $N\le12$.
8. **Takeaways & next steps** — methodology + harness; where an advantage might exist (large $\chi$, 2D, large $N$ with tensor-network simulation of the filter); certified gap bound for large $N$.

*Numbers in §4.2–4.4 are regenerated from the raw result files; rerun `paper/build_paper.py` after new experiments.*
