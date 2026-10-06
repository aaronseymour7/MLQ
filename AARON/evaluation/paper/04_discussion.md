## 6. Code audit and recommended changes

Severity: **H** invalidates a claim or can silently produce a wrong/ruinous result; **M** weakens a claim or wastes resources; **L** hygiene. Line references are to `AARON/j1j2_filter`.

| # | Sev | Location | Finding | Recommended fix |
|---|---|---|---|---|
| 1 | H | `pipeline.choose_floor_design`, `make_design` | $\gamma\ge1-\epsilon_{\mathrm{mach}}$ produces $m=0$ rows that are counted infeasible; fallback to `builder` (hard-coded $P_{\mathrm{succ}}\ge0.9$, no $\eta$ target) builds a filter for an already-exact state (24 k–1.2 M CX at $J_2=0.5$, §5.7). Mixed `floor`/`builder` rows are also not comparable. | Return the identity ("no filter, cost 0") when $\gamma$ is within tolerance of 1, and never mix design sources in one table. |
| 2 | H | `core/spectrum.get_spectrum` (DMRG path), `pipeline.build_ctx` | Certificates are rigorous only if (A1) $\Delta\le\Delta_{\mathrm{true}}$, (A2) $\gamma\le\gamma_{\mathrm{true}}$, (A3) $|E_0^{\mathrm{shift}}-E_0|\le e_0$ hold; DMRG supplies none of them (§4.1, §5.5–5.6). In the default ED mode the certified results are limited to $N\le12$. | State the theorem *conditional on* (A1)–(A3) with these labels; report $\Delta$ with an explicit safety margin; for large $N$ obtain a gap lower bound by an independent method (e.g. symmetry-resolved DMRG plus a Temple/Weinstein-type bound) or present the large-$N$ numbers as estimates. |
| 3 | H | `pipeline.ORDER`, `core/trotter.trotter_bounds` | The bound is first-order; setting `ORDER=2` changes the circuits but the same $\alpha t^2/2k$ bound is still reported (and `lie_trotter_order` only checks order 1). | `assert ORDER == 1` next to the bound, or implement the second-order commutator bound. |
| 4 | H | `pipeline.build_ctx` (`if N <= 16: ... eigvalsh(Hs.toarray())`) | Dense $2^N\times2^N$ complex matrix: 4.3 GB at $N=14$, 68.7 GB at $N=16$; `SOURCE='dmrg'` is advertised for $N>12$. | Cap the dense check at $N\le12$ or use `eigsh` for the two extremal eigenvalues. |
| 5 | H | repo | Not reproducible as shipped: `scripts/run_study.py` imports a `study` module that is absent; `mps_to_circuit` is an unpinned external dependency that fails on Python ≥ 3.13; `main.ipynb` imports a stale 831-line monolith `builder.py`, and `hamiltonians.py`/`get_energies.py` exist twice; comments cite tests ("T1", "T8", audit items A1–A14) that are not in the repository. | Add `study.py`, a pinned `requirements.txt`/lock file, delete or archive `builder.py`, single-source the two duplicated modules, and add a `tests/` directory containing the checks of §5.1 (our `exp1_primitives.py` can be the seed). |
| 6 | M | `floor.floor_vs_precision` vs. `pipeline.choose_floor_design` | The $(x,m)$ winner for each $\varepsilon$ is picked with proxy $x^2/F(0)^2$, but the true cost is $n/P\propto x^2/F(0)^{3}$ (since $n\propto1/\sqrt{p_g}$); the later choice over leakage fractions uses the correct cost, so the two stages optimise different objectives. | Pass `cost_fn=lambda x,f,m: x*x/f**1.5`. |
| 7 | M | `pipeline.find_empirical` | "Empirical" $n$ uses the exact ground state and can land where $\eta\approx1$ (§5.2b). | Report it only as an oracle lower envelope; require certified $\eta<1$. |
| 8 | M | `pipeline.costs` | Expected cost $=\mathrm{CX}_{\mathrm{total}}/P_{\mathrm{succ}}$ ignores early abort (a failed pulse stops the run), so it overestimates; correct form: $\bigl(c_{\mathrm{trial}}+\sum_i c_{\mathrm{step}}k_i\prod_{j<i}p_j\bigr)/\prod_jp_j$. | Implement with the per-pulse $p_j$ that the simulation already has. |
| 9 | M | `pipeline` (`hi=1.0` everywhere); `core/spectrum` | The filter must suppress $[\Delta,1]$, but the spectrum ends at $(E_{\mathrm{top}}-E_0)/W<1$ when $W$ is the certified (looser) bandwidth; $W=c_I+\sum|c_\beta|-E_0$ overstates $E_{\mathrm{top}}$ by 3× because $XX{+}YY{+}ZZ$ is bounded termwise (true per-bond bound $J/4$, so $E_{\mathrm{top}}\le\frac14[J_1(N-1)+J_2(N-2)]$ for $J\ge0$). | Pass `hi=(E_top_bound-E0)/W` and use the per-bond bound, which is also rigorous and 3× tighter. |
| 10 | M | `core/trotter.total_error_bound` | Composition $(\sqrt{2\ell}+d_T)^2$ with $d_T=2\epsilon_T/\sqrt{p_g}$ is valid but 2.5–3.1× loose. | Use $\sin^2(\arcsin\sqrt\ell+\arcsin(\epsilon_T/\sqrt{p_g}))$ (Proposition, §3; verified in all simulated cases). |
| 11 | M | `core/__init__`, `core/resources.rotation_synthesis_t_count` | The authors themselves flag the Trotter theorem statement and the synthesis constants ($1.15\log_2(1/\epsilon)+9.2$) as "quoted from memory". | Verify against the sources and cite them in the paper before reporting T-counts. |
| 12 | L | `core/circuits.h_tensor_z` | The shift term $-E_0/W\cdot Z_{\mathrm{anc}}$ is applied as a separate non-Clifford $R_z$ in every Trotter step; it commutes with everything and can be folded into $\phi_i$ ($\phi_i\to\phi_i-E_0t_i/W$), saving one of 47 non-Clifford rotations per step ($N=6$). | Fold into the phase. |
| 13 | L | `pipeline.build_ctx` | $\gamma$ is clipped with `min(·,1)` but the $\gamma=1$ branch is never propagated (see 1); the ED-vs-DMRG $\gamma$ fallback is labelled but not carried into reports. | Carry `gamma_src` into every exported number. |

### 6.1 Methodological improvements that follow from the analysis

1. **Sharpened composition** (item 10): a free, proven 2.5–3.1× tighter bound and 2.3–3.2× fewer guaranteed steps.
2. **Symmetry-resolved certification**: let $w_s$ be the trial weight in sector $s$ and $\Delta_s$ the gap of that sector. Then $\ell\le\sum_sw_s\,(\sup_{[\Delta_s,1]}|F|)^2/F(0)^2$ over the ground-state sector and the (small) remainder bounded with the global gap. Because bond-ordered Trotter steps are exactly SU(2)-symmetric, the sector weights are conserved by the (Trotterised) evolution, so this is rigorous given the sector weights, which are cheap to compute from the trial MPS. Potential saving up to $(\Delta_s/\Delta)^2\approx7\times$ in $n$ at $J_2=0$ (§5.6).
3. **Subspace-restricted Trotter bounds**: replace the global $\alpha$ by commutator norms restricted to the low-energy subspace populated by the trial state (the observed state error is ≈ 300× below the bound), or use higher-order formulas with the matching commutator bound.
4. **Tensor-network simulation of the filter.** The DMRG advantage is lost as long as every number comes from dense vectors; the controlled evolution $e^{-itH\otimes Z}$ is an MPO-friendly operation, so MPS/TEBD simulation of the Trotterised filter would allow $N\sim30$–$100$ and show the real DMRG→circuit→filter story.

## 7. Conclusions and what a publishable paper still needs

**Established.** The mathematics of the filter, leakage, Trotter and composition bounds is correct, and verified numerically in every test performed; the pipeline's reference implementation (pulse convention, MPO/Pauli agreement, circuit-vs-numpy agreement to $10^{-16}$) is consistent; DMRG supplies ground-state and excitation energies to $\sim10^{-9}$ up to $N=16$.

**Not established.** (i) A resource advantage: at $N\le12$ the guaranteed filter costs $10^{2}$–$10^{3}\times$ exact preparation and $\gg$ shallow MPS circuits, and the empirical savings come from an oracle. (ii) A DMRG-specific role in certification: the certified numbers are ED-based. (iii) Anything at $N>12$, near criticality or in the frustrated regime away from the exactly solvable MG point.

**Required for submission (in priority order).**

1. *Baselines.* Compare resources at equal fidelity against (a) deeper/optimised MPS circuits and sequential MPS preparation, (b) QETU/Lin–Tong filtering with the same trial state, (c) rodeo with random times. Without this the paper cannot claim usefulness.
2. *Large-$N$ data* from a tensor-network simulation of the filter (item 4 above), with $\gamma(N,L)$ and the gap from DMRG labelled as estimates, and a stated rigorous or heuristic gap input.
3. *Tighter guarantees*: items 10 and 2 of §6.1 at minimum, to close the 100–1000× gap to the empirical cost, and an honest empirical definition (certified $\eta<1$).
4. *Benchmarks that stress the algorithm*: $J_2\in[0.2,0.45]$, $N\ge16$, and trial states with $\gamma\ll1$; drop $J_2=0.5$ or use it only as a sanity check.
5. *Repository hygiene* (items 1–5 of §6), a test suite, pinned environment, and archived raw data for every table and figure.
6. *Resource model*: Hamiltonian-aware compilation of $e^{-it\,h_b\otimes Z}$ (bond exponentials are SU(2)-symmetric), verified T-count constants, and a fault-tolerant estimate with early-abort statistics.

**Suggested framing if the above is only partially achievable:** a methods paper on *certified* SBC filtering — rigorous a-priori bounds, symmetry-resolved certification, and a verification harness — rather than a performance paper.

## 8. Reproducibility

Code: `AARON/evaluation/experiments` (`exp1_primitives.py` building-block tests; `exp2_end_to_end.py N J2 L`; `exp3_dmrg_inputs.py`; `exp3b_baseline.py`; `exp4_gap_and_symmetry.py`; `exp5_scaling.py`; `make_figures.py`). Raw outputs: `AARON/evaluation/results/*.json`; logs: `AARON/evaluation/logs`. All experiments call `AARON/j1j2_filter` unmodified. Environment: Python 3.12, numpy 2.x, scipy 1.18, qiskit 2.5.2, quimb 1.15, `mps-to-circuit`; run with `OMP_NUM_THREADS=1` (multi-threaded BLAS made the SLSQP filter design several times slower). Floor-design settings for the scaling study were reduced for speed (`FLOOR_EPS_FRACS=[0.2,0.1,0.05]`, `FLOOR_M=[4,6]`); end-to-end runs used the defaults.

**Limitations of this evaluation.** Single trial-circuit compilation per $(N,J_2,L)$ (the compiler is a local optimiser, so $\gamma$ is not unique); $J_2\neq0,0.5$ end-to-end runs and $N>10$ end-to-end runs were not completed; the $N=8$, $\varepsilon=10^{-3}$ guaranteed design was not simulated (beyond the simulation cap); the scaling fits use four sizes; the crossover estimate is an extrapolation. References were written from memory and must be checked before submission.

## References

1. I. Stetcu, A. Baroni, J. Carlson, *Projection algorithm for state preparation on quantum computers*, Phys. Rev. C **105**, 064308 (2022).
2. K. Choi, D. Lee, J. Bonitati, Z. Qian, J. Watkins, *Rodeo algorithm for quantum computing*, Phys. Rev. Lett. **127**, 040505 (2021).
3. Z. Qian et al., *Demonstration of the rodeo algorithm on a quantum computer*, arXiv:2110.07747.
4. A. M. Childs, Y. Su, M. C. Tran, N. Wiebe, S. Zhu, *Theory of Trotter error with commutator scaling*, Phys. Rev. X **11**, 011020 (2021).
5. L. Lin, Y. Tong, *Near-optimal ground state preparation*, Quantum **4**, 372 (2020).
6. Y. Dong, L. Lin, Y. Tong, *Ground-state preparation and energy estimation on early fault-tolerant quantum computers via quantum eigenvalue transformation of unitary matrices*, PRX Quantum **3**, 040305 (2022).
7. C. Schön, E. Solano, F. Verstraete, J. I. Cirac, M. M. Wolf, *Sequential generation of entangled multiqubit states*, Phys. Rev. Lett. **95**, 110503 (2005).
8. C. K. Majumdar, D. K. Ghosh, *On next-nearest-neighbor interaction in linear chain*, J. Math. Phys. **10**, 1388 (1969).
9. S. R. White, *Density matrix formulation for quantum renormalization groups*, Phys. Rev. Lett. **69**, 2863 (1992); U. Schollwöck, Ann. Phys. **326**, 96 (2011).
10. K. Okamoto, K. Nomura, Phys. Lett. A **169**, 433 (1992).
