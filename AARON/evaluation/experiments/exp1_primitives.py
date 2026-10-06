"""E1: soundness of the analytic building blocks, tested against brute force.
 A. spectral-sup certificates (floor.certify, filter_design.certify_filter)
 B. fidelity lower bound R2
 C. commutator constant alpha and the Lie-Trotter pulse-error claim
 D. state-distance -> infidelity conversion (current vs sharper)"""
from common import *
import numpy as np, scipy.linalg as sl
import floor, core.filter_design as fd, core.trotter as tr
from hamiltonians import j1j2_hamiltonian
from qiskit.quantum_info import SparsePauliOp
rng = np.random.default_rng(0)
out = {}

# ---- A: certificates
rows = []
for trial in range(60):
    m = rng.integers(3, 8)
    delta = rng.uniform(0.05, 0.4)
    T = rng.uniform(0.3, 3.0) * np.pi / delta
    t = rng.dirichlet(np.ones(m) * 2) * T
    ph = rng.uniform(-0.6, 0.6, m)
    grid = np.linspace(delta, 1, 2_000_001)
    brute = float(np.max(np.abs(fd.filter_values(t, ph, grid))))
    c1 = floor.certify(t, ph, delta, 0.0, 1.0, rtol=0.01)
    c2 = fd.certify_filter(t, ph, [0, delta, 1.0])
    rows.append(dict(brute=brute, bnb=c1["sup_cert"], grid_lip=c2["sup_cert"],
                     conv=c1["converged"]))
bn = np.array([r["bnb"] / r["brute"] for r in rows]); gl = np.array([r["grid_lip"] / r["brute"] for r in rows])
out["A"] = dict(n=len(rows), bnb_min_ratio=bn.min(), bnb_max_ratio=bn.max(), bnb_median=float(np.median(bn)),
                gridlip_min_ratio=gl.min(), gridlip_max_ratio=gl.max(),
                violations=int((bn < 1 - 1e-12).sum() + (gl < 1 - 1e-12).sum()),
                unconverged=int(sum(not r["conv"] for r in rows)))
print("A", out["A"])

# ---- B: R2 fidelity lower bound, random spectra / trial states
viol, slack = 0, []
for trial in range(20000):
    d = rng.integers(3, 40)
    delta = rng.uniform(0.02, 0.5)
    E = np.concatenate([[0.0], rng.uniform(delta, 1.0, d - 1)])
    E[1] = delta
    m = rng.integers(2, 6)
    t = rng.dirichlet(np.ones(m)) * rng.uniform(0.5, 3) * np.pi / delta
    ph = rng.uniform(-0.8, 0.8, m)
    gam = rng.uniform(0.01, 0.99)
    w = rng.dirichlet(np.ones(d - 1) * rng.uniform(0.1, 2)) * (1 - gam)
    p = np.concatenate([[gam], w])
    F = fd.filter_values(t, ph, E)
    fid = p[0] * F[0] ** 2 / np.sum(p * F ** 2)
    eta = np.max(np.abs(F[1:])) / abs(F[0])
    lb = fd.fidelity_lower_bound(gam, eta)
    if fid < lb - 1e-12: viol += 1
    slack.append(fid - lb)
out["B"] = dict(n=20000, violations=viol, min_slack=float(min(slack)))
print("B", out["B"])

# ---- C: alpha vs brute-force commutator sums, and Lie-Trotter pulse-level error
def dense_terms(H):
    return [SparsePauliOp(H.paulis[i], [H.coeffs[i]]).to_matrix() for i in range(len(H))]
def alpha_dense(mats, reverse=False):
    mats = mats[::-1] if reverse else mats
    a = 0.0
    for i in range(len(mats) - 1):
        S = sum(mats[i + 1:])
        C = mats[i] @ S - S @ mats[i]
        a += np.linalg.norm(C, 2)
    return a
rowsC = []
for N, j2 in [(4, 0.0), (4, 0.5), (5, 0.3), (6, 0.0), (6, 0.5)]:
    H = j1j2_hamiltonian(N, 1.0, j2)
    mats = dense_terms(H)
    a_f, a_r = alpha_dense(mats), alpha_dense(mats, True)
    a_code_f, a_code_r = tr.alpha_comm(H, True, False), tr.alpha_comm(H, True, True)
    a_tri = tr.alpha_triangle(H)
    # actual first-order error at t small, forward product (first term applied first)
    t = 0.02
    U = sl.expm(-1j * t * H.to_matrix())
    Sf = np.eye(2 ** N, dtype=complex)
    for m_ in mats: Sf = sl.expm(-1j * t * m_) @ Sf
    err = np.linalg.norm(U - Sf, 2)
    rowsC.append(dict(N=N, J2=j2, alpha_dense_fwd=a_f, alpha_dense_rev=a_r, alpha_code_fwd=a_code_f,
                      alpha_code_rev=a_code_r, alpha_triangle=a_tri,
                      err_over_bound=err / (a_f * t * t / 2)))
out["C"] = rowsC
for r in rowsC: print("C", r)

# pulse-level (H (x) Z) check, N=4: ||W - W~|| <= alpha t^2/(2k)
N = 4
H = j1j2_hamiltonian(N, 1.0, 0.5)
Hz = SparsePauliOp.from_list([("Z" + l, c) for l, c in zip(H.paulis.to_labels(), H.coeffs)])
mz = dense_terms(Hz)
a = max(alpha_dense(mz), alpha_dense(mz, True))
a_sys = max(tr.alpha_comm(H, True, False), tr.alpha_comm(H, True, True))
pulse = []
for t, k in [(0.5, 1), (1.0, 1), (2.0, 4), (3.0, 8), (6.0, 20)]:
    W = sl.expm(-1j * t * Hz.to_matrix())
    S = np.eye(2 ** (N + 1), dtype=complex)
    for _ in range(k):
        for m_ in mz: S = sl.expm(-1j * t / k * m_) @ S
    e = np.linalg.norm(W - S, 2)
    pulse.append(dict(t=t, k=k, err=e, bound=a * t * t / (2 * k), ratio=e / (a * t * t / (2 * k))))
out["C_pulse"] = dict(alpha_HZ=a, alpha_H=a_sys, rows=pulse)
print("C_pulse", out["C_pulse"])

# ---- D: state-distance -> infidelity conversion.  random unit vectors
# current: 1-F <= (sqrt(2 leak) + d)^2 ; sharper: sin(theta1+theta2)^2 with sin theta1 = sqrt(leak),
# theta2 = 2 arcsin(d/2)  (angle for ||phi - phi~|| = d).  Also check the normalisation step:
# sin(angle(a,b)) <= ||a-b||/||a||  vs  ||a/|a| - b/|b|| <= 2||a-b||/||a||.
v1 = v2 = 0; ratios = []
for trial in range(50000):
    dim = 8
    g = np.zeros(dim, complex); g[0] = 1
    phi = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    phi = phi * np.array([1] + [rng.uniform(0, 0.3)] * (dim - 1)); phi /= np.linalg.norm(phi)
    pert = (rng.normal(size=dim) + 1j * rng.normal(size=dim)) * rng.uniform(0, 0.5)
    phit = phi + pert; phit /= np.linalg.norm(phit)
    d = np.linalg.norm(phit - phi)
    leak = 1 - abs(np.vdot(g, phi)) ** 2
    actual = 1 - abs(np.vdot(g, phit)) ** 2
    cur = (np.sqrt(2 * leak) + d) ** 2
    th = np.arcsin(np.sqrt(leak)) + 2 * np.arcsin(min(d / 2, 1))
    sharp = np.sin(min(th, np.pi / 2)) ** 2 if th < np.pi / 2 else 1.0
    if actual > cur + 1e-12: v1 += 1
    if actual > sharp + 1e-12: v2 += 1
    if cur > 1e-9: ratios.append((actual / cur, actual / sharp if sharp > 1e-9 else np.nan))
r = np.array(ratios)
vn = 0; rn = []
for trial in range(50000):
    a_ = rng.normal(size=6) + 1j * rng.normal(size=6)
    b_ = a_ + (rng.normal(size=6) + 1j * rng.normal(size=6)) * rng.uniform(0, 0.8)
    na = np.linalg.norm(a_)
    dist = np.linalg.norm(a_ / na - b_ / np.linalg.norm(b_))
    sinang = np.sqrt(max(0, 1 - abs(np.vdot(a_, b_)) ** 2 / (na ** 2 * np.linalg.norm(b_) ** 2)))
    ab = np.linalg.norm(a_ - b_) / na
    if sinang > ab + 1e-12: vn += 1
    rn.append((dist / (2 * ab), sinang / ab))
rn = np.array(rn)
out["D"] = dict(current_violations=v1, sharper_violations=v2,
                median_actual_over_current=float(np.median(r[:, 0])),
                median_actual_over_sharper=float(np.nanmedian(r[:, 1])),
                norm_sinangle_violations=vn,
                norm_dist_over_2x_max=float(rn[:, 0].max()), norm_sinangle_over_x_max=float(rn[:, 1].max()))
print("D", out["D"])
save("exp1_primitives", out)
