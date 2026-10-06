"""Figures for the paper from results/*.json."""
from common import *
import glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = RES.parent / "paper" / "figs"; FIG.mkdir(exist_ok=True)
C = dict(blue="#0072B2", orange="#E69F00", green="#009E73", red="#D55E00", purple="#CC79A7", grey="#666666")
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5, "figure.dpi": 150, "lines.linewidth": 1.2})
def load(p): return json.load(open(p))

# ---- Fig 1 & 2 from exp2
runs = [load(p) for p in sorted(glob.glob(str(RES / "exp2_N*_J2_0.0_L1.json")))]
fig, ax = plt.subplots(figsize=(4.6, 3.3))
cols = [C["blue"], C["orange"], C["green"]]
for run, c in zip(runs, cols):
    eps = [r["eps"] for r in run["rows"] if "guar" in r and "emp" in r]
    g = [r["guar"]["n"] for r in run["rows"] if "guar" in r and "emp" in r]
    e = [r["emp"]["n"] for r in run["rows"] if "guar" in r and "emp" in r]
    ax.plot(eps, g, "o-", color=c, label=f"N={run['N']} guaranteed")
    ax.plot(eps, e, "s--", color=c, mfc="white", label=f"N={run['N']} empirical (oracle)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("target infidelity $\\varepsilon$"); ax.set_ylabel("Trotter steps $n$")
ax.legend(fontsize=6.5, ncol=1, frameon=False); fig.tight_layout(); fig.savefig(FIG / "fig1_guar_vs_emp.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(4.2, 3.3))
bd, ac, sh = [], [], []
for run in runs:
    for r in run["rows"]:
        g = r.get("guar")
        if g and g.get("eps_actual") is not None:
            bd.append(g["eps_bound"]); ac.append(g["eps_actual"]); sh.append(g.get("sharp_bound"))
bd, ac, sh = map(np.array, (bd, ac, sh))
ax.loglog(bd, ac, "o", color=C["blue"], label="measured vs. current bound")
ax.loglog(sh, ac, "^", color=C["green"], label="measured vs. sharpened bound")
lo, hi = min(ac.min(), 1e-5), 0.2
ax.plot([lo, hi], [lo, hi], color=C["grey"], lw=0.8); ax.text(hi * 0.5, hi * 0.35, "$y=x$", color=C["grey"], fontsize=7)
ax.set_xlabel("a-priori bound on $1-F$"); ax.set_ylabel("measured $1-F$ (noiseless)")
ax.legend(frameon=False, fontsize=7); fig.tight_layout(); fig.savefig(FIG / "fig2_bound_tightness.png"); plt.close(fig)

# ---- Fig 3 baseline
fig, ax = plt.subplots(figsize=(4.8, 4.0))
for p, c in zip(sorted(glob.glob(str(RES / "exp3b_N*_J2_0.0.json")), key=lambda s: int(s.split("_N")[1].split("_")[0])), [C["blue"], C["orange"], C["green"], C["red"]]):
    b = load(p)
    ax.loglog([r["cx"] for r in b["rows"]], [r["infid"] for r in b["rows"]], "o-", ms=3, color=c, label=f"N={b['N']}: MPS circuit, $L$=1..8")
    ax.plot(b["exact_prep"]["cx"], 1e-9 + 1e-3, marker="*", ms=9, color=c, ls="none")
for run, c in zip(runs, [C["blue"], C["orange"], C["green"]]):
    for r in run["rows"]:
        if "guar" in r:
            ax.plot(r["guar"]["cx"], r["eps"], "x", color=c, ms=6)
ax.plot([], [], "*", color=C["grey"], ms=8, label="exact state prep (plotted at $10^{-3}$)")
ax.plot([], [], "x", color=C["grey"], label="filter, guaranteed (trial $L$=1)")
ax.set_xlabel("CX count"); ax.set_ylabel("infidelity to exact ground state")
ax.legend(frameon=False, fontsize=6.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2); fig.tight_layout(); fig.savefig(FIG / "fig3_baseline.png", bbox_inches="tight"); plt.close(fig)

# ---- Fig 4 scaling
rows = load(RES / "exp5_J2_0.0_L1_N4-10.json")
N = np.array([r["N"] for r in rows]); n = np.array([r["n_guar"] for r in rows]); cx = np.array([r["cx"] for r in rows])
ex = {4: None}
for p in glob.glob(str(RES / "exp3b_N*_J2_0.0.json")):
    b = load(p); ex[b["N"]] = b["exact_prep"]["cx"]
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.0))
ax[0].loglog(N, n, "o-", color=C["blue"], label="guaranteed $n$ ($\\varepsilon=10^{-2}$)")
ref = n[1] * (N / N[1]) ** 4; ax[0].plot(N, ref, color=C["grey"], ls=":", label="$\\propto N^4$")
ax[0].set_xlabel("N"); ax[0].set_ylabel("Trotter steps"); ax[0].legend(frameon=False, fontsize=7)
ax[1].loglog(N, cx, "o-", color=C["blue"], label="filter (guaranteed)")
ax[1].plot(N, cx[1] * (N / N[1]) ** 5, color=C["grey"], ls=":", label="$\\propto N^5$")
Ne = sorted(k for k, v in ex.items() if v); ax[1].plot(Ne, [ex[k] for k in Ne], "*-", color=C["red"], label="exact state prep")
ax[1].set_xlabel("N"); ax[1].set_ylabel("CX count"); ax[1].legend(frameon=False, fontsize=7)
from matplotlib.ticker import ScalarFormatter
for a in ax:
    a.set_xticks([4, 6, 8, 10, 12]); a.xaxis.set_major_formatter(ScalarFormatter()); a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.tight_layout(); fig.savefig(FIG / "fig4_scaling.png"); plt.close(fig)

# ---- Fig 5 gap sensitivity
d = load(RES / "exp4_gap_symmetry.json")["gap_sensitivity_N6"]
d = [r for r in d if r.get("feasible", True)]
s = [100 * r["s"] for r in d]
fig, ax = plt.subplots(figsize=(4.4, 3.0))
ax.plot(s, [r["eta_on_true_window"] / r["eta_design"] for r in d], "o-", color=C["red"], label="certified $\\eta$ on true window / design $\\eta$")
ax.axhline(1, color=C["grey"], lw=0.8)
ax.set_xlabel("gap over-estimate (%)"); ax.set_ylabel("$\\eta$ inflation factor"); ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); fig.savefig(FIG / "fig5_gap_sensitivity.png"); plt.close(fig)
print("ok")
