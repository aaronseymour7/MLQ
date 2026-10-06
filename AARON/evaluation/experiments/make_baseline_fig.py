"""Figure + equal-fidelity table from exp6 results."""
from common import *
import glob, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FIG = RES.parent / "paper" / "figs"; FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": .25, "grid.linewidth": .5, "figure.dpi": 150, "lines.linewidth": 1.1})
C = dict(approx="#0072B2", exact="#009E73", gen="#666666", f1="#D55E00", f3="#E69F00")
runs = {}
for p in glob.glob(str(RES / "exp6_N*_J2_*.json")):
    d = json.load(open(p)); runs[(d["N"], d["J2"])] = d
J2s = sorted({k[1] for k in runs}); Ns = sorted({k[0] for k in runs})
fig, axs = plt.subplots(len(J2s), len(Ns), figsize=(3.0 * len(Ns), 2.7 * len(J2s)), squeeze=False, sharey=True)
FLOOR = 1e-6
for i, J2 in enumerate(J2s):
    for j, N in enumerate(Ns):
        ax = axs[i][j]; d = runs.get((N, J2))
        if d is None: ax.axis("off"); continue
        a = d["approx"]; ax.loglog([x["cx"] for x in a], [max(x["infid"], FLOOR) for x in a], "o-", ms=2.5, color=C["approx"], label="MPS circuit, $L$=1..10")
        e = [x for x in d["exact_chi"] if "cx" in x and x["cx"] > 0]
        ax.loglog([x["cx"] for x in e], [max(x["infid"], FLOOR) for x in e], "s-", ms=3, color=C["exact"], label="sequential MPS, $\\chi$=2..16")
        ax.plot(d["genprep"]["cx"], FLOOR, "*", ms=8, color=C["gen"], label="generic state prep")
        for L, c, lab in ((1, C["f1"], "filter on $L$=1 (certified)"), (3, C["f3"], "filter on $L$=3 (certified)")):
            f = [x for x in d["filter"] if x["L"] == L and "cx" in x and x.get("design") != "none"]
            if f: ax.loglog([x["cx"] for x in f], [x["eps_bound"] for x in f], "x-", ms=4, color=c, label=lab)
        ax.set_title(f"$N$={N}, $J_2$={J2:g}", fontsize=8)
        if i == len(J2s) - 1: ax.set_xlabel("CX count")
        if j == 0: ax.set_ylabel("infidelity (floor $10^{-6}$)")
axs[0][0].legend(fontsize=5.5, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(FIG / "fig6_baselines.png"); plt.close(fig)

# equal-fidelity table: min CX reaching infidelity <= eps
rows = []
for (N, J2), d in sorted(runs.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    for eps in (1e-1, 1e-2, 1e-3):
        cand = {}
        for nm, lst in (("MPS circuit", [(x["cx"], x["infid"]) for x in d["approx"]]),
                        ("sequential MPS", [(x["cx"], x["infid"]) for x in d["exact_chi"] if "cx" in x]),
                        ("generic prep", [(d["genprep"]["cx"], 0.0)]),
                        ("filter (certified)", [(x["cx"], x["eps_bound"]) for x in d["filter"] if "cx" in x and x.get("design") != "none"])):
            ok = [c for c, f in lst if f <= eps]
            cand[nm] = min(ok) if ok else None
        rows.append(dict(N=N, J2=J2, eps=eps, **cand))
json.dump(rows, open(RES / "exp6_equal_fidelity.json", "w"), indent=1)
for r in rows: print(r)
