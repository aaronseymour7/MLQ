"""Rebuild the generated paper sections from results/*.json and assemble paper.{md,html,docx,tex}.
Run again whenever new results arrive:   python3 build_paper.py   (needs numpy, matplotlib, pandoc)"""
import json, glob, subprocess, pathlib, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; RES = HERE.parent / "results"; FIG = HERE / "figs"; FIG.mkdir(exist_ok=True)
J = lambda p: json.load(open(p))
def fmt(x, nd=3):
    if x is None: return "–"
    if isinstance(x, int) or (isinstance(x, float) and abs(x) >= 100 and float(x).is_integer()): return f"{int(x):,}"
    return f"{x:.{nd}g}"
def cx(x): return "–" if x is None else f"{int(round(x)):,}"
def ratio(a, b): return "–" if (a is None or b is None or b == 0) else f"{a / b:.0f}×" if a / b >= 10 else f"{a / b:.1f}×"

# ---------- load
eq = {(r["N"], r["J2"], r["eps"]): r for r in J(RES / "exp6_equal_fidelity.json")}
e7 = {}
for p in glob.glob(str(RES / "exp7_N*_L1.json")):
    d = J(p)
    for r in d["rows"]: e7[(d["N"], d["J2"], r["eps"])] = dict(r, step_cx=d["step_cx"], gamma=d["gamma"])
loose = J(RES / "exp7a_looseness.json")
EPS = (1e-1, 1e-2, 1e-3)

def best_baseline(r):
    c = {k: r[k] for k in ("MPS circuit", "sequential MPS", "generic prep") if r.get(k) is not None}
    if not c: return None, None
    k = min(c, key=c.get); return k, c[k]

# ---------- table A: equal-fidelity (certified filter vs baselines vs hybrid protocol)
L = []
L.append("| $N$ | $J_2$ | $\\varepsilon$ | best baseline (CX) | filter: certified (CX) | filter: hybrid protocol (CX) | certified / baseline | hybrid / baseline |")
L.append("|---|---|---|---|---|---|---|---|")
for (N, J2, eps), r in sorted(eq.items(), key=lambda kv: (kv[0][1], kv[0][0], -kv[0][2])):
    nm, b = best_baseline(r)
    e = e7.get((N, J2, eps), {}); pr = (e.get("protocol") or {})
    L.append(f"| {N} | {J2:g} | {eps:g} | {nm or '–'} ({cx(b)}) | {cx(r.get('filter (certified)'))} | {cx(pr.get('cx'))} | {ratio(r.get('filter (certified)'), b)} | {ratio(pr.get('cx'), b)} |")
tabA = "\n".join(L)

# ---------- table B: hybrid protocol details
L = ["| $N$ | $J_2$ | $\\varepsilon$ | $n$ (protocol) | CX | measured $1-F$ | $P_{\\rm succ}$ | oracle $n$ / CX | certified $n$ / CX | certified ÷ protocol (CX) |", "|---|---|---|---|---|---|---|---|---|---|"]
for (N, J2, eps), e in sorted(e7.items(), key=lambda kv: (kv[0][1], kv[0][0], -kv[0][2])):
    pr, o, g = e.get("protocol") or {}, e.get("oracle") or {}, e.get("guaranteed") or {}
    if e.get("design") == "builder":
        note = " †"
    else: note = ""
    L.append(f"| {N} | {J2:g} | {eps:g} | {pr.get('n', '–')}{note} | {cx(pr.get('cx'))} | {fmt(pr.get('actual'),2) if pr else '–'} | {fmt(pr.get('p_succ'),2) if pr else '–'} | {o.get('n','–')} / {cx(o.get('cx'))} | {g.get('n','–')} / {cx(g.get('cx'))} | {ratio(g.get('cx'), pr.get('cx'))} |")
tabB = "\n".join(L)

# ---------- table C: looseness
L = ["| $N$ | $J_2$ | $\\alpha$ (worst case) | $\\|C\\psi_0\\|$ (ground state) | $\\alpha/\\|C\\psi_0\\|$ | $\\alpha/\\|C\\psi_{E_1}\\|$ |", "|---|---|---|---|---|---|"]
for r in sorted(loose, key=lambda r: (r["J2"], r["N"])):
    g = "(=0)" if r["Cpsi0"] < 1e-12 else f"{r['ratio_gs']:.1f}"
    L.append(f"| {r['N']} | {r['J2']:g} | {r['alpha_scaled']:.4f} | {r['Cpsi0']:.4f} | {g} | {r['ratio_E1']:.1f} |")
tabC = "\n".join(L)

# ---------- figure 7: CX vs N at eps = 1e-2
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .25, "figure.dpi": 150, "lines.linewidth": 1.1})
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
for ax, J2 in zip(axs, (0.0, 0.4)):
    Ns = sorted({k[0] for k in eq if k[1] == J2})
    def series(f):
        xs, ys = [], []
        for N in Ns:
            v = f(N)
            if v is not None: xs.append(N); ys.append(v)
        return xs, ys
    ax.semilogy(*series(lambda N: best_baseline(eq[(N, J2, 1e-2)])[1]), "s-", color="#009E73", label="best baseline (MPS / sequential MPS / generic)")
    ax.semilogy(*series(lambda N: ((e7.get((N, J2, 1e-2)) or {}).get("protocol") or {}).get("cx")), "o-", color="#0072B2", label="filter, hybrid protocol")
    ax.semilogy(*series(lambda N: ((e7.get((N, J2, 1e-2)) or {}).get("oracle") or {}).get("cx")), "o--", mfc="white", color="#0072B2", label="filter, oracle empirical")
    ax.semilogy(*series(lambda N: eq[(N, J2, 1e-2)].get("filter (certified)")), "x-", color="#D55E00", label="filter, certified")
    ax.set_title(f"$J_2$={J2:g}, $\\varepsilon=10^{{-2}}$", fontsize=8); ax.set_xlabel("N"); ax.set_xticks(Ns)
axs[0].set_ylabel("CX count (incl. trial circuit)")
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc="lower center", ncol=2, frameon=False, fontsize=7)
fig.tight_layout(rect=(0, 0.12, 1, 1)); fig.savefig(FIG / "fig7_cx_vs_N.png"); plt.close(fig)

gen = f"""### 5.8 Baselines at equal fidelity: the filter does not beat MPS preparation here

![CX versus infidelity for the DMRG-derived MPS circuit ($L=1..10$, `mps-to-circuit` *approximate*), the sequential (*exact*) MPS circuit at bond dimension $\\chi=2,4,8,16$, generic isometry state preparation, and the certified filter on $L=1$ and $L=3$ trial circuits. Filter points are certified bounds. Infidelity floor $10^{{-6}}$.](figs/fig6_baselines.png)

Table 5.8 lists, for each target $\\varepsilon$, the cheapest baseline that reaches it (measured against the exact ground state) next to the filter's cost (including the trial circuit): the *certified* design of §3 and the *hybrid protocol* of §5.10.

{tabA}

The best baseline is an MPS-based circuit with bond dimension $\\chi\\le8$ in nearly every row; generic preparation wins only at the smallest sizes and tightest targets ($N\\le8$). The sequential MPS method needs $\\chi=4$ for $\\varepsilon=10^{{-2}}$ at every $N=6$–$12$ and its cost grows slowly (63–174 CX); for $J_2=0.4$ the state is nearly a dimer product and $\\chi=2$ already gives $\\varepsilon\\lesssim10^{{-1}}$ for 15–33 CX. The certified filter is $10^{{2}}$–$10^{{4}}\\times$ more expensive and the gap *widens* with $N$ (figure 7). The hybrid protocol closes most of the gap but, at these sizes, remains 5–31× above the best baseline (up to 74× in the tightest $J_2=0.4$ row).

![CX at $\\varepsilon=10^{{-2}}$ versus $N$: best baseline, certified filter, hybrid protocol and oracle-empirical filter.](figs/fig7_cx_vs_N.png)

### 5.9 Where the certified-versus-empirical gap comes from

The worst-case constant $\\alpha=\\sum_g\\|[H_g,\\sum_{{g'>g}}H_{{g'}}]\\|$ is an operator norm over the whole Hilbert space. The first-order Lie–Trotter error vector acting on a state $\\psi$ is $\\tfrac{{t^2}}{{2k}}C\\psi$ with $C=\\sum_{{g<g'}}[H_{{g'}},H_g]$ (anti-Hermitian), so what a low-energy state actually feels is $\\|C\\psi\\|$, not $\\alpha$:

{tabC}

($H$ scaled by $W$; for $N=4$, $J_2=0$ the ground state is annihilated by $C$.) The ground state feels a first-order error coefficient 28–41× below $\\alpha$ at $J_2=0$ and 9–13× below at $J_2=0.4$, and the ratio *grows* with $N$ ($\\alpha$ grows linearly while $\\|C\\psi_0\\|$ falls), so the bound becomes looser, not tighter, as the system grows. Combined with the pulse-level slack ($\\le0.24$ of the bound, §5.1) and the factor $\\approx2$ of the original composition, this accounts for the measured $\\approx270\\times$ gap between predicted and observed state error at $N=6$ ($28\\times4\\times2\\approx240$). The measured state error scales as $1/n$ exactly as the first-order theory predicts, only with a much smaller constant: the oracle-empirical step counts are therefore *consistent with* the theory, not evidence against it.

### 5.10 A hybrid, non-oracle protocol: rigorous leakage plus empirical Trotter control

The "empirical" step counts of §5.2 use the exact ground state and are not available to a user. We therefore replace them by a protocol that uses only quantities a practitioner can obtain:

1. **Leakage — rigorous.** For the snapped step grid, certify $\\eta$ (R1) and compute the leakage bound $\\ell(n)$ from R2 (needs $\\gamma$ and $\\Delta$).
2. **Trotter — empirical, Richardson-type.** Run the same circuit family at $n$ and $2n$ and compute $\\delta_n=1-|\\langle\\psi_n|\\psi_{{2n}}\\rangle|^2$; because the error vector scales as $1/n$, the distance to the $n\\to\\infty$ state is $\\approx2\\sqrt{{\\delta_n}}$.
3. **Accept** the smallest $n$ on a geometric grid with $(\\sqrt{{\\ell(n)}}+2\\sqrt{{\\delta_n}})^2\\le\\varepsilon$ and certified $\\eta<1$.

The exact ground state is used only afterwards, to validate. (On hardware the two-run comparison is replaced by the convergence of an energy or other observable estimated at $n$ and $2n$.) Results ($L=1$ trial, $J_1=1$; † marks rows where the floor design search found no feasible design within the reduced search space used here and the legacy `builder` design was used, so these are not comparable):

{tabB}

The protocol met its target in every completed row, with a safety margin of roughly 2–10× in infidelity, and it needs roughly 30–600× fewer CX than the certified circuits (the larger factors at tighter $\\varepsilon$). It is, however, *not rigorous*: step 2 is an extrapolation that assumes the asymptotic $1/n$ regime. The rigorous and the hybrid numbers should be reported side by side, labelled as such.
"""
(HERE / "05_generated.md").write_text(gen)

parts = ["00_front.md", "01_theory.md", "02_workflow.md", "03_results.md", "05_generated.md", "04_discussion.md"]
md = "\n\n".join((HERE / p).read_text() for p in parts)
(HERE / "paper.md").write_text(md)
for out, extra in (("paper.html", ["--mathml", "--embed-resources"]), ("paper.docx", []), ("paper.tex", [])):
    r = subprocess.run(["pandoc", "paper.md", "-s", "--resource-path=.", "-o", out, *extra], cwd=HERE, capture_output=True, text=True)
    if r.returncode: print(out, r.stderr[-300:])
print("built:", len(md), "chars;", len(e7), "empirical rows;", len(eq), "baseline rows")
