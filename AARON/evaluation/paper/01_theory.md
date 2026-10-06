## 2. Model, algorithm and conventions

**Model.** We study the open-boundary spin-½ $J_1$–$J_2$ chain
$$
H=\sum_{i=1}^{N-1}J_1\,\mathbf S_i\!\cdot\!\mathbf S_{i+1}+\sum_{i=1}^{N-2}J_2\,\mathbf S_i\!\cdot\!\mathbf S_{i+2},
\qquad \mathbf S_i\!\cdot\!\mathbf S_j=\tfrac14\left(X_iX_j+Y_iY_j+Z_iZ_j\right),
$$
with $J_1=1$, even $N$ (so that the ground state is a non-degenerate singlet) and $J_2\in\{0,\,0.2411,\,0.5\}$ (Heisenberg, the Okamoto–Nomura dimerisation point, and the Majumdar–Ghosh point). The Pauli form has $3(N-1)+3(N-2)\,[J_2\neq0]$ terms. $H$ conserves total spin $\mathbf S^2$ and spatial reflection; the lowest excitation is a triplet.

**Scaled Hamiltonian.** With $E_0$ the ground energy and $W\ge E_{\max}-E_0$, define $H_s=(H-E_0)/W$ so that $\operatorname{spec}(H_s)\subset\{0\}\cup[\Delta,1]$ with $\Delta=(E_1-E_0)/W$. All pulse times below are in units of $W^{-1}$.

**Projection pulse (Stetcu–Baroni–Carlson).** One pulse acts on the system register and one ancilla initialised in $|0\rangle$:
$$
\mathcal W(t,\phi)=\big(\mathsf H_{\rm a}\big)\;e^{-it\,H_s\otimes Z_{\rm a}}\;\mathrm{Rz}_{\rm a}(2\phi)\;\big(\mathsf H_{\rm a}\big).
$$
Using $\mathrm{Rz}(2\phi)=\mathrm{diag}(e^{-i\phi},e^{i\phi})$ one finds
$$
\langle 0_{\rm a}|\mathcal W|0_{\rm a}\rangle=\tfrac12\big(e^{-i(H_st+\phi)}+e^{+i(H_st+\phi)}\big)=\cos(H_st+\phi),
\qquad
\langle 1_{\rm a}|\mathcal W|0_{\rm a}\rangle=-i\sin(H_st+\phi).
$$
Measuring the ancilla and keeping only outcome $0$ after each of $m$ pulses (with the ancilla reset between pulses) applies, up to normalisation, the **filter**
$$
F(E)=\prod_{i=1}^{m}\cos(Et_i+\phi_i),\qquad T=\sum_i t_i ,
$$
to the trial state. The probability of the all-zeros record is $P_{\rm succ}=\sum_k|c_k|^2F(E_k)^2$ for a trial state $|\psi\rangle=\sum_kc_k|E_k\rangle$; because $F$ is a product of cosines, a failed pulse aborts the run immediately ("early abort"). We verified the sign convention (including that $\phi\to-\phi$ is rejected) directly on a random, non-commuting, field-carrying $H$ and a random complex state (`verify_pulse_convention`, error $<10^{-8}$).

**Relation to prior work.** The pulse is the deterministic-time, optimised-phase version of the projection/“rodeo” circuit of Stetcu, Baroni and Carlson (Phys. Rev. C 105, 064308 (2022)); with $\phi_i=0$ and random $t_i$ it reduces to the rodeo algorithm. The product-of-cosines filter is a real trigonometric polynomial in $E$ of total frequency $T$ and is therefore a restricted member of the family realised by quantum eigenvalue transformation of unitaries / QSVT (Lin–Tong, Dong–Lin–Tong), which achieve $T=O(\Delta^{-1}\log(1/\eta))$ with a single ancilla but need coherent (not measurement-based) control. The present scheme trades that for a *classically optimised* filter that can be certified and for early-abort statistics.

## 3. Rigorous error analysis

Notation: $\gamma=|\langle E_0|\psi\rangle|^2$ is the trial-state ground-state weight; $\eta=\sup_{E\in[\Delta,1]}|F(E)|/|F(0)|$; $L=\sum_i|t_i|$; $\alpha=\sum_g\big\|[H_g,\sum_{g'>g}H_{g'}]\big\|$ for the Pauli terms $H_g$ of $H_s$ in circuit order.

**R1 (certified suppression).** Since $|\cos|\le1$ and $|\partial_E\cos(Et_i+\phi_i)|\le|t_i|$, $|F'|\le L$ and $|F''|\le L^2$. On an interval of half-width $d$ centred at $E_c$, $|F(E)|\le|F(E_c)|+|F'(E_c)|d+\tfrac12L^2d^2$. Interval bisection with this envelope (`floor.certify`) returns a rigorous upper bound on $\sup|F|$; a cheaper variant (grid maximum $+\,Lh/2$, `filter_design.certify_filter`) is also provided. The ground-state amplitude is bounded below by $|F(0)|-L\,e_0$ where $e_0$ is the (scaled) uncertainty of the ground-state energy.

**R2 (leakage).** If $\operatorname{spec}H_s\subset\{0\}\cup[\Delta,1]$ then for the *exact* filter
$$
F_{\rm exact}\;\ge\;\frac{\gamma}{\gamma+(1-\gamma)\eta^2},\qquad\text{i.e. leakage }\ \ell\le\frac{(1-\gamma)\eta^2}{\gamma+(1-\gamma)\eta^2}.
$$
(Proof: numerator $\gamma F(0)^2$, denominator $\le\gamma F(0)^2+(1-\gamma)\sup|F|^2$.) The design problem is therefore: given $\gamma$ and a target leakage $\varepsilon_\ell$, find $(t_i,\phi_i)$ with certified $\eta\le\eta_*=\sqrt{\gamma\varepsilon_\ell/((1-\varepsilon_\ell)(1-\gamma))}$ that maximises $F(0)^2=\prod\cos^2\phi_i$ (so $P_{\rm succ}\ge\gamma F(0)^2$) at fixed $T$ and pulse number $m$ (`floor.py`).

**R3 (Lie–Trotter error of the post-selected state).** Replace $e^{-itH_s\otimes Z}$ by $k$ first-order Lie–Trotter steps over the Pauli terms. Because $[H_gZ,H_{g'}Z]=[H_g,H_{g'}]\otimes\mathbb 1$, the commutator constant of $H_s\otimes Z$ equals $\alpha$, and the standard bound (Childs et al., Phys. Rev. X 11, 011020, Eq. for $p=1$) gives $\|\mathcal W_i-\widetilde{\mathcal W}_i\|\le\alpha t_i^2/(2k_i)=:\epsilon_i$. The post-selected blocks $A_i=\langle0|\mathcal W_i|0\rangle$ are contractions, so telescoping gives for the un-normalised vectors $a=A_m\cdots A_1\psi$, $b=\tilde A_m\cdots\tilde A_1\psi$
$$
\|a-b\|\le\epsilon_T:=\sum_i\frac{\alpha t_i^2}{2k_i}\;\overset{t_i=k_i\,dt}{=}\;\frac{\alpha T^2}{2n},\qquad n=\sum_ik_i .
$$
Renormalisation after each pulse is irrelevant because the projections commute with global rescaling. The code uses $\|a/|a|-b/|b|\|\le2\epsilon_T/\sqrt{p_g}$ with $p_g=\|a\|^2\ge\gamma F(0)^2$. **Scope:** this is a first-order statement; the identical formula is used by the code whatever `ORDER` is set to (see §7).

**R4 (composition).** The code combines the two contributions as $1-F\le(\sqrt{2\ell}+d_T)^2$, $d_T=2\epsilon_T/\sqrt{p_g}$. This is valid (proved in the code docstring, re-derived by us: $1-\sqrt{1-\ell}\le\ell$ plus the triangle inequality for phase-optimised distance, and $1-F\le2(1-|\langle g|\tilde\phi\rangle|)$; 0 violations in $5\times10^4$ random trials) but **not tight**. Working with the Fubini–Study angle $\theta$ instead of the Euclidean distance:

> **Proposition (sharpened composition).** If $\|a-b\|<\|a\|$ then $\sin\angle(a,b)\le\|a-b\|/\|a\|$. With $\sin\theta_\ell=\sqrt\ell$ and the triangle inequality for $\angle$,
> $$1-F\;\le\;\sin^2\!\Big(\arcsin\sqrt{\ell}+\arcsin\frac{\epsilon_T}{\sqrt{p_g}}\Big)\;\le\;\Big(\sqrt\ell+\frac{\epsilon_T}{\sqrt{p_g}}\Big)^2 .$$

*Proof.* $\operatorname{dist}(a,\operatorname{span}b)\le\|a-b\|$ and $\operatorname{dist}(a,\operatorname{span}b)=\|a\|\sin\angle(a,b)$. The angle between rays is a metric on projective space, so $\theta(\tilde\phi,g)\le\theta(\tilde\phi,\phi)+\theta(\phi,g)$; finally $1-F=\sin^2\theta(\tilde\phi,g)$ and $\sin$ is increasing on $[0,\pi/2]$. $\square$

Relative to R4 this removes a factor 2 in the Trotter distance and a factor $\sqrt2$ in the leakage distance. Because the required step number scales as $n\propto\epsilon_T\propto\sqrt{\varepsilon}-\sqrt{\ell}$ (first-order Trotter gives $n\propto\varepsilon^{-1/2}$ for an *infidelity* target), the proposition lowers the *guaranteed* $n$ by about a factor 2 and the guaranteed $\varepsilon$ by about 4 at fixed $n$ (quantified in §6.2).

**Cost model.** One Trotter step of one pulse costs $c_{\rm step}$ CX (transpiled once); a design with $n$ steps and $m_{\rm nz}$ non-zero pulses costs $n\,c_{\rm step}+c_{\rm trial}$ CX and $n\,r_{\rm step}+m_{\rm nz}$ non-Clifford rotations; the expected cost to obtain a success is divided by $P_{\rm succ}$.
