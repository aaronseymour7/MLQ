import numpy as np
from netket.operator.spin import sigmap, sigmam, sigmaz

N = hi.size

def n_op(hi, i):
    """n_i = (1 + sigma^z_i)/2"""
    return 0.5 * (sigmaz(hi, i) + 1)

def cdag_c(hi, i, j):
    """c_i^dagger c_j for i < j, with explicit JW string phase."""
    op = sigmap(hi, i)
    for k in range(i + 1, j):
        op = op @ sigmaz(hi, k)
    sign = (-1) ** (j - i - 1)
    op = sign * (op @ sigmam(hi, j))
    return 0.25 * op

# single-site occupations
n_mean = np.array([vs.expect(n_op(hi, i)).mean.real for i in range(N)])

C = np.zeros((N, N))
for p in range(N):
    for q in range(N):
        if p == q:
            # <n_p^2> = <n_p> since n_p is a projector (eigenvalues 0/1)
            npnp = n_mean[p]
        else:
            npnq_op = n_op(hi, p) @ n_op(hi, q)
            npnp = vs.expect(npnq_op).mean.real
        C[p, q] = npnp - n_mean[p] * n_mean[q]




rho = np.zeros((N, N), dtype=complex)

# diagonal: rho_pp = <n_p>
for p in range(N):
    rho[p, p] = n_mean[p]

# off-diagonal: rho_pq for p < q, then Hermitian conjugate for p > q
for p in range(N):
    for q in range(p + 1, N):
        val = vs.expect(cdag_c(hi, p, q)).mean
        rho[p, q] = val
        rho[q, p] = np.conj(val)

def symmetrize_circulant(M, L):
    """Average M[p,q] over all translations to enforce circulant structure."""
    M_sym = np.zeros_like(M)
    for shift in range(L):
        M_sym += np.roll(np.roll(M, shift, axis=0), shift, axis=1)
    return M_sym / L

rho_sym = symmetrize_circulant(rho, N)
rho_rdm = np.eye(N) - rho_sym.T
C_sym = symmetrize_circulant(C, N)
theta_J = C_sym.copy()
Re_rho = rho_rdm.real
Im_rho = rho_rdm.imag

# Givens seeds
theta_Givens_real = Re_rho
theta_Givens_imag = Im_rho

print(f"Theta G_real: {theta_Givens_real}")
print(f"Theta G_imag: {theta_Givens_imag}")
print(f"Theta J: {theta_J}")
