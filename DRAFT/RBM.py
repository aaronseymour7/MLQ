import netket as nk
import time

L = 8
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)


## Symmetric RBM Spin Machine
ma = nk.models.RBMSymm(symmetries=g.translation_group(), alpha=3)

# Metropolis Exchange Sampling
# Notice that this sampler exchanges two neighboring sites
# thus preservers the total magnetization
sa = nk.sampler.MetropolisExchange(hi, graph=g)

# Optimizer
op = nk.optimizer.Sgd(learning_rate=0.02)

# The variational state
vs = nk.vqs.MCState(sa, ma, n_samples=2016)

# The ground-state optimization loop
gs = nk.driver.VMC_SR(hamiltonian=ha, optimizer=op, diag_shift=0.01, variational_state=vs)

start = time.time()
gs.run(out="RBMSymmetric", n_iter=300)
end = time.time()
E = (vs.expect(ha).mean.real)


evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
exact_gs_energy = evals[0]
print("The exact ground-state energy is E0=", exact_gs_energy)
print("### Symmetric RBM calculation")
print("Has", vs.n_parameters, "parameters")
print("The Symmetric RBM calculation took", end - start, "seconds")
print(f"Final RBM energy: {E:.10f}")
