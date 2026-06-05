
from dataclasses import dataclass
from scipy.linalg import eigh
from ucj import build_ucj
from lattices import BaseLattice, make_lattice
from filter import append_filter





j1 = 1.0 
j2 = .6
n = 12
n_up = _get_n_up(n)
lattice = make_lattice('kagome', L=n)

# ── build + transpile circuit ─────────────────────────────────────────────

tqc, ham, counts, depth, psi_ucj = build_ucj(n, lattice, j1=j1, j2=j2, variant="g", k_layers=1, state = True)
norm = np.linalg.norm(np.array(psi_ucj))
print("UCJ norm =", norm)
# ── after build_ucj and UCJ optimisation ─────────────────────────────────────

H_sparse, basis, idx_map = _build_hamiltonian_sparse(
    n, n_up, lattice.nn_edges, lattice.nnn_edges, j1, j2
)
H_dense = H_sparse.toarray()                    # fine for n=8, dim~70
evals_full, evecs_full = eigh(H_dense)          # all eigenpairs, sorted ascending

psi_gs  = evecs_full[:, 0]
# ── Gap: find first excited state strictly above E_0 ─────────────────────────
E_0   = evals_full[0]
tol   = 1e-6   # degeneracy threshold

# first index where eval differs from E_0 by more than tol
first_excited_idx = np.argmax(evals_full - E_0 > tol)

if first_excited_idx == 0:
    raise RuntimeError("No excited state found above ground state — check Hamiltonian.")

E_1 = evals_full[first_excited_idx]
gap = E_1 - E_0

print(f"E_0={E_0:.8f}  E_1={E_1:.8f}  gap={gap:.8f}  (first excited at index {first_excited_idx})")
E_max   = evals_full[-1]

# ── UCJ state: keep complex, project onto full eigenbasis ─────────────────────
psi_ucj_np = np.array(psi_ucj, dtype=complex)
psi_ucj_np = psi_ucj_np / np.linalg.norm(psi_ucj_np)

E_ucj = np.real(np.vdot(psi_ucj_np, H_dense @ psi_ucj_np))

print("Energy from returned state:", E_ucj)
print("Ground energy:", E_0)
print("DeltaE:", E_ucj - E_0)

# components c_k = <psi_k | psi_ucj>  (complex)
coeffs = evecs_full.conj().T @ psi_ucj_np       # shape (dim,), complex
coeffs_sq = np.abs(coeffs)**2

print(f"norm in full eigenbasis: {np.linalg.norm(coeffs):.8f}")   # should be ~1.0

overlap = float(np.abs(coeffs[0]))
print(f"<GS|UCJ> = {overlap:.6f}")

# ── Filter inputs ─────────────────────────────────────────────────────────────
# The filter multiplies component-wise: c_k -> c_k * prod_i cos(E_k * t_i + phi_i)
# So trial_state = |c_k| (magnitudes), gs_state = delta_{k,0}
trial_state = np.abs(coeffs)                    # real, non-negative amplitudes
gs_state    = np.zeros(len(evals_full))
gs_state[0] = 1.0

totaltime = np.pi / gap
print(f"gap={gap:.6f}  totaltime={totaltime:.4f}  dim={len(evals_full)}")

# ── FilterBuilder ─────────────────────────────────────────────────────────────
fb = FilterBuilder(
    total_time=totaltime,
    energies=evals_full,       # full real spectrum
    overlap=overlap,
    a=8, b=12,
    coeffs_sq=coeffs_sq,
)

eval_results = fb.build_and_evaluate(
    method="v5",
    gs_state=gs_state,
    trial_state=trial_state,
    highlight_pos=1,
    plot=True
)
