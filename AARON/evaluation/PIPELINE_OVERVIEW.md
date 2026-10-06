---
title: "Certified ground-state filtering from DMRG trial states — pipeline overview"
subtitle: "J1–J2 Heisenberg chain · Stetcu–Baroni projection · what we built, what we proved, what it costs"
---

# 0. One-paragraph summary

We want the ground state $|E_0\rangle$ of the open spin-½ $J_1$–$J_2$ chain on a quantum computer. **DMRG** (classical) gives a matrix-product state (MPS) that we compile into a shallow circuit; that circuit has ground-state weight $\gamma<1$. We repair it with the **Stetcu–Baroni–Carlson (SBC) projection**: a single ancilla, controlled time evolution, and post-selection implement a *filter* $F(E)=\prod_i\cos(Et_i+\phi_i)$ that suppresses excited states. We optimise the filter classically, **certify** how well it suppresses the spectrum, and control the Trotter error three ways (*certified*, *hybrid*, *oracle*). Result: the mathematics is sound and verified, the rigorous guarantee is $10^2$–$10^3\times$ more conservative than reality, a *hybrid* protocol recovers most of that gap — but for $N\le12$ an MPS-based circuit is still 5–216× (median ≈22×) cheaper than the filter, so **no resource advantage on these 1D chains**. The value is the certified-filtering method and its verification harness.

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
| $m$ | number of pulses (4–6 in the floor designs) |
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
| 10 | 0 | 0.001 | 48 | 3,051 | 0.0004 | 0.45 | 36 / 2,295 | 61683 / 3,886,056 | 1274× |
| 12 | 0 | 0.1 | 20 | 1,573 | 0.0071 | 0.55 | 7 / 572 | 2673 / 205,854 | 131× |
| 12 | 0 | 0.01 | 32 | 2,497 | 0.0015 | 0.48 | 29 / 2,266 | 21651 / 1,667,160 | 668× |
| 12 | 0 | 0.001 | 96 | 7,425 | 0.00019 | 0.52 | 44 / 3,421 | 119858 / 9,229,099 | 1243× |
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
| 10 | 0 | 0.001 | sequential MPS (519) | 3,802,824 | 3,051 | 7327× | 5.9× |
| 12 | 0 | 0.1 | sequential MPS (174) | 125,532 | 1,573 | 721× | 9.0× |
| 12 | 0 | 0.01 | sequential MPS (174) | 1,292,390 | 2,497 | 7428× | 14× |
| 12 | 0 | 0.001 | sequential MPS (709) | – | 7,425 | – | 10× |
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
6. **Hybrid protocol** — definition, validation, 30–1300× fewer CX than certified.
7. **Baselines** — equal-fidelity comparison; MPS preparation wins for $N\le12$.
8. **Takeaways & next steps** — methodology + harness; where an advantage might exist (large $\chi$, 2D, large $N$ with tensor-network simulation of the filter); certified gap bound for large $N$.

*Derivations of every formula above are in the appendix (§7). Numbers in §4.2–4.4 are regenerated from the raw result files; rerun `paper/build_paper.py` after new experiments.*

# 7. Appendix: derivations of the filter, leakage and error formulas

Notation: trial state $|\psi\rangle=\sum_kc_k|E_k\rangle$ with $\gamma=|c_0|^2$; scaled spectrum $\{0\}\cup[\Delta,1]$; $F(E)=\prod_{i=1}^m\cos(Et_i+\phi_i)$; $F_0=|F(0)|$; $L=\sum_i|t_i|$. Code locations: `core/filter_design.py` (A.2), `core/trotter.py` (A.3–A.4), `floor.py` (the $\eta_*$ target).

## A.1 The filter and the success probability

$\mathrm{Rz}(2\phi)=\mathrm{diag}(e^{-i\phi},e^{i\phi})$. Starting from $|0\rangle_{\rm a}|\psi\rangle$: the first $\mathsf H$ gives $\tfrac1{\sqrt2}(|0\rangle+|1\rangle)|\psi\rangle$; $\mathrm{Rz}(2\phi)$ multiplies the branches by $e^{\mp i\phi}$; $e^{-itH_s\otimes Z}$ multiplies them by $e^{\mp itH_s}$; the final $\mathsf H$ projects onto $|0\rangle$ with amplitude

$$\tfrac12\big(e^{-i(H_st+\phi)}+e^{+i(H_st+\phi)}\big)=\cos(H_st+\phi),$$

and onto $|1\rangle$ with $-i\sin(H_st+\phi)$. Keeping outcome 0 after each of the $m$ pulses (renormalising or not gives the same final direction, since projections commute with rescaling):

$$|\varphi\rangle=\frac{\sum_kc_kF(E_k)|E_k\rangle}{\sqrt p},\qquad P_{\rm succ}=p=\sum_k|c_k|^2F(E_k)^2\ \ge\ \gamma F_0^2 .$$

## A.2 Leakage (R2) and its certificate (R1)

*Leakage* $\ell$ is the infidelity of the ideal (exact-evolution) filtered state with the ground state. With $y=\sum_{k\ge1}|c_k|^2F(E_k)^2$,

$$1-F_{\rm exact}=\frac{y}{\gamma F_0^2+y}.$$

By definition of $\eta=\sup_{[\Delta,1]}|F|/F_0$, $F(E_k)^2\le\eta^2F_0^2$ for $k\ge1$, hence $y\le(1-\gamma)\eta^2F_0^2$. The right-hand side above is increasing in $y$, so substituting the upper bound gives

$$\boxed{\;\ell\ \le\ \frac{(1-\gamma)\eta^2}{\gamma+(1-\gamma)\eta^2},\qquad F_{\rm exact}\ge\frac{\gamma}{\gamma+(1-\gamma)\eta^2}\;}$$

**Design target.** For a leakage budget $\varepsilon_\ell$ set $\ell=\varepsilon_\ell$: $(1-\gamma)\eta^2(1-\varepsilon_\ell)=\varepsilon_\ell\gamma$, i.e.

$$\eta_*^2=\frac{\gamma\,\varepsilon_\ell}{(1-\varepsilon_\ell)(1-\gamma)} .$$

`floor.py` maximises $F_0^2=\prod\cos^2\phi_i$ subject to *certified* $\eta\le\eta_*$.

**Certifying $\eta$ (R1).** $|\cos|\le1$ and $|\tfrac{d}{dE}\cos(Et_i+\phi_i)|=|t_i\sin(\cdot)|\le|t_i|$ give $|F'|\le L$; differentiating once more (every term is a product of at most two $t$-factors) gives $|F''|\le L^2$. On an interval of half-width $d$ centred at $E_c$, Taylor's theorem with the remainder bounded by $\sup|F''|$ gives

$$|F(E)|\le|F(E_c)|+|F'(E_c)|\,d+\tfrac12L^2d^2 .$$

Bisecting $[\Delta,1]$, pruning intervals whose envelope falls below the running maximum (times $1+r_{\rm tol}$), and keeping the largest pruned envelope yields a rigorous upper bound on $\sup|F|$ (a cheaper variant uses the grid maximum plus $Lh/2$). If the true ground energy lies $e_0$ below the shift, $|F(-e_0)|\ge F_0-Le_0$, which lowers the denominator of $\eta$.

## A.3 Trotter error (R3)

**Lemma (two terms).** For Hermitian $A,B$, $U=e^{-i\delta(A+B)}$, $V=e^{-i\delta A}e^{-i\delta B}$:

$$U-V=-i\int_0^\delta U(\delta-s)\big(e^{-isA}Be^{isA}-B\big)V(s)\,ds$$

(Duhamel). Since $\|e^{-isA}Be^{isA}-B\|\le s\|[A,B]\|$ and $U,V$ are unitary, $\|U-V\|\le\tfrac{\delta^2}{2}\|[A,B]\|$.

**Many terms.** Writing $H=\sum_gH_g$ and applying the lemma inductively (peel off $H_1$ against $\sum_{g>1}H_g$, then recurse) gives
$\big\|e^{-i\delta H}-\prod_ge^{-i\delta H_g}\big\|\le\tfrac{\delta^2}{2}\alpha$, with
$$\alpha=\sum_g\Big\|\Big[H_g,\sum_{g'>g}H_{g'}\Big]\Big\|$$
(the first-order case of Childs, Su, Tran, Wiebe, Zhu). With $\delta=t/k$ and $k$ repetitions, telescoping $\|X^k-Y^k\|\le k\|X-Y\|$ gives $\alpha t^2/(2k)$. The ordering of the terms matters, so the code takes $\max$ over the forward and reversed orders.

**Ancilla.** $[H_gZ,H_{g'}Z]=[H_g,H_{g'}]\otimes Z^2=[H_g,H_{g'}]\otimes\mathbb 1$, so $H_s\otimes Z$ has the same $\alpha$ and
$$\|\mathcal W_i-\widetilde{\mathcal W}_i\|\le\epsilon_i:=\frac{\alpha t_i^2}{2k_i}.$$

**Across pulses.** The post-selected blocks $A_i=\langle0|\mathcal W_i|0\rangle$ are contractions with $\|A_i-\tilde A_i\|\le\|\mathcal W_i-\widetilde{\mathcal W}_i\|$. For $a=A_m\cdots A_1\psi$ and $b=\tilde A_m\cdots\tilde A_1\psi$,

$$a-b=\sum_{i=1}^m(A_m\cdots A_{i+1})(A_i-\tilde A_i)(\tilde A_{i-1}\cdots\tilde A_1\psi)\ \Rightarrow\ \|a-b\|\le\sum_i\epsilon_i=:\epsilon_T .$$

On the uniform grid $t_i=k_i\,dt$, $\sum_ik_i=n$, $dt=T/n$:
$$\epsilon_T=\alpha\,dt\sum_i\frac{t_i}{2}=\frac{\alpha T^2}{2n}.$$

**Normalisation.** $\|a\|\sin\angle(a,b)=\mathrm{dist}(a,\mathrm{span}\,b)\le\|a-b\|$, and $\|a\|^2=p\ge p_g:=\gamma(F_0-Le_0)^2$, so

$$\sin\angle(a,b)\le\frac{\epsilon_T}{\sqrt{p_g}}\quad(\text{valid when }\epsilon_T<\sqrt{p_g}).$$

(The original code used the weaker Euclidean bound $\|a/|a|-b/|b|\|\le2\epsilon_T/\sqrt{p_g}$.)

## A.4 Composition into a bound on $1-F$

For the final normalised Trotterised state $\tilde\varphi$, the ideal filtered state $\varphi$ and the ground state $g$, use the Fubini–Study angle $\theta(x,y)=\arccos|\langle x|y\rangle|$, which is a metric on rays:

- leakage: $\sin^2\theta(\varphi,g)=\ell'\le\ell$ (A.2);
- Trotter: $\theta(\tilde\varphi,\varphi)\le\arcsin(\epsilon_T/\sqrt{p_g})$ (A.3);
- triangle inequality: $\theta(\tilde\varphi,g)\le\theta(\tilde\varphi,\varphi)+\theta(\varphi,g)$;
- $1-F=\sin^2\theta(\tilde\varphi,g)$ and $\sin$ is increasing on $[0,\pi/2]$.

$$\boxed{\;1-F\ \le\ \sin^2\!\Big(\arcsin\sqrt{\ell}+\arcsin\frac{\epsilon_T}{\sqrt{p_g}}\Big)\ \le\ \Big(\sqrt\ell+\frac{\epsilon_T}{\sqrt{p_g}}\Big)^2\;}$$

*Original code bound:* with Euclidean (phase-optimised) distances, $1-\sqrt{1-\ell}\le\ell$ gives $\mathrm{dist}(\varphi,g)\le\sqrt{2\ell}$, so $\mathrm{dist}(\tilde\varphi,g)\le\sqrt{2\ell}+d_T$ with $d_T=2\epsilon_T/\sqrt{p_g}$, and $1-F\le2(1-|\langle g|\tilde\varphi\rangle|)=\mathrm{dist}^2\le(\sqrt{2\ell}+d_T)^2$. Both are valid; the angle form is about 3× tighter (checked on $5\times10^4$ random vector triples and every simulated design: 0 violations).

**Required step count.** Requiring $1-F\le\varepsilon$ in the angle form: $\epsilon_T\le\sqrt{p_g}\,\sin(\arcsin\sqrt\varepsilon-\arcsin\sqrt\ell)$, so with $\epsilon_T=\alpha T^2/2n$

$$n\ \ge\ \frac{\alpha T^2}{2\sqrt{p_g}\,\sin(\arcsin\sqrt\varepsilon-\arcsin\sqrt\ell)}\ \approx\ \frac{\alpha T^2}{2\sqrt{p_g}\,(\sqrt\varepsilon-\sqrt\ell)} .$$

(Original form: $n=\alpha T^2/(\sqrt{p_g}\,d_T)$ with $d_T=\sqrt\varepsilon-\sqrt{2\ell}$.) First-order Trotter therefore needs $n\propto\varepsilon^{-1/2}$ for an infidelity target, and $n\propto\alpha T^2$, i.e. $\propto N\cdot\Delta^{-2}$ in physical units.

## A.5 The hybrid protocol (a heuristic built on A.2 and A.4, not a theorem)

Assume the first-order error vector behaves as $e_n=\psi_n-\psi_\infty\approx c/n$. Then $\psi_n-\psi_{2n}\approx e_n/2$, so $\|\psi_n-\psi_\infty\|\approx2\|\psi_n-\psi_{2n}\|$. For small distances $1-|\langle\psi_n|\psi_{2n}\rangle|^2\approx\|\psi_n-\psi_{2n}\|^2$, i.e. $\delta_n\approx\|\psi_n-\psi_{2n}\|^2$, so the Trotter angle is $\approx2\sqrt{\delta_n}$. Inserting it for $\arcsin(\epsilon_T/\sqrt{p_g})$ in A.4, with the certified leakage $\ell(n)$ of the *snapped* design, gives the acceptance rule

$$\big(\sqrt{\ell(n)}+2\sqrt{\delta_n}\big)^2\le\varepsilon\quad\text{and certified }\eta<1 .$$

Leakage is rigorous; the Trotter term is only as reliable as the $1/n$ assumption. Observed margins: measured infidelity was $1.6$–$10\times$ below $\varepsilon$ in all 24 cases.

## A.6 Assumptions used above

1. $\mathrm{spec}(H_s)\subset\{0\}\cup[\Delta,1]$ with a non-degenerate ground state (even $N$, singlet).
2. $\gamma$, $\Delta$ and $e_0$ known exactly (ED, $N\le12$) or *assumed* (DMRG path).
3. First-order Lie–Trotter with preserved term order; $\alpha$ is the max over forward and reversed order. The Trotter bound does **not** cover second-order formulas.
4. Noiseless circuits; no synthesis error in rotations.
5. $\epsilon_T<\sqrt{p_g}$ (otherwise the Trotter angle bound is vacuous and the code falls back to 1).
