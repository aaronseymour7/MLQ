"""Pipeline block diagram (figs/fig0_pipeline.png)."""
import pathlib, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
FIG = pathlib.Path(__file__).resolve().parent / "figs"; FIG.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(11, 4.6)); ax.set_xlim(0, 11); ax.set_ylim(0, 4.6); ax.axis("off")
def box(x, y, w, h, title, body, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=color, ec="#444", lw=0.8))
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", fontsize=8.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.48, body, ha="center", va="top", fontsize=7, linespacing=1.35)
C1, C2, C3, C4 = "#DCEBF7", "#E3F3E8", "#FCEBD2", "#F6DDD3"
box(0.1, 2.75, 2.5, 1.65, "1  Model", "open $J_1$–$J_2$ chain, $N$ sites\n$H=\\sum J_r\\vec{S}_i\\cdot\\vec{S}_{i+r}$\nPauli form + MPO", C1)
box(2.85, 2.75, 2.5, 1.65, "2  DMRG (classical)", "MPS $|\\psi_{DMRG}\\rangle$, $E_0$, $E_1$, $E_{top}$\ncross-checked with ED", C1)
box(5.6, 2.75, 2.5, 1.65, "3  Trial circuit", "MPS → $L$-layer circuit\n$|\\psi_{trial}\\rangle$,  $\\gamma=|\\langle E_0|\\psi_{trial}\\rangle|^2$", C2)
box(8.35, 2.75, 2.5, 1.65, "4  Scale spectrum", "$H_s=(H-E_0)/W$\nspec $\\subset\\{0\\}\\cup[\\Delta,1]$", C2)
box(0.1, 0.1, 2.5, 2.35, "5  Filter design", "$F(E)=\\prod_i\\cos(Et_i+\\phi_i)$\n$m$ pulses, times $t_i$, phases $\\phi_i$\nmax $F(0)^2$ s.t. certified\n$\\eta\\leqq\\eta_*(\\varepsilon_\\ell,\\gamma)$", C3)
box(2.85, 0.1, 2.5, 2.35, "6  Circuit", "pulse: H · Rz(2φ) · $e^{-it H_s\\otimes Z}$ · H\nLie–Trotter, $k_i$ steps\n$n=\\sum k_i$, $dt=T/n$\nmeasure ancilla; abort on 1", C3)
box(5.6, 0.1, 2.5, 2.35, "7  Choose $n$ (error control)", "certified: $\\ell,\\ \\epsilon_T=\\alpha T^2/2n$\nhybrid: certified $\\ell$ + Richardson\n$\\delta_n$ from runs at $n$, $2n$\noracle: compare to exact $|E_0\\rangle$", C4)
box(8.35, 0.1, 2.5, 2.35, "8  Compare", "CX / T count, $P_{succ}$\nvs MPS circuit ($L$ layers),\nsequential MPS ($\\chi$),\ngeneric state prep", C4)
for (x0, y0, x1, y1) in [(2.6, 3.57, 2.85, 3.57), (5.35, 3.57, 5.6, 3.57), (8.1, 3.57, 8.35, 3.57), (2.6, 1.27, 2.85, 1.27), (5.35, 1.27, 5.6, 1.27), (8.1, 1.27, 8.35, 1.27)]:
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", lw=1))
ax.text(10.95, 2.6, "continues  ↓ (5)", ha="right", va="center", fontsize=7, color="#555")
fig.tight_layout(); fig.savefig(FIG / "fig0_pipeline.png", dpi=170); print("ok")
