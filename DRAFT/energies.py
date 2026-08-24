import netket as nk
import time

L = 8
g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
ha = nk.operator.Heisenberg(hilbert=hi, graph=g)


def get_energies(L=8, level = 'ground', exact = bool):
    
    g = nk.graph.Hypercube(length=L, n_dim=1, pbc=True)
    
    if level == 'ground':     
        hi = nk.hilbert.Spin(s=0.5, total_sz=0, N=g.n_nodes)
        ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
        if exact == True:
            evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
            exact_energy = evals[0]
            
    elif level == 'first':
        hi = nk.hilbert.Spin(s=0.5, total_sz=1, N=g.n_nodes)
        ha = nk.operator.Heisenberg(hilbert=hi, graph=g)
        if exact == True:
            evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
            exact_energy = evals[0]
    elif level == 'highest':
        hi = nk.hilbert.Spin(s=0.5, N=g.n_nodes)
        ha = -1 * nk.operator.Heisenberg(hilbert=hi, graph=g)
        if exact == True:
            evals = nk.exact.lanczos_ed(ha, compute_eigenvectors=False)
            exact_energy = -1 * evals[0]

    ## Symmetric RBM 
    ma = nk.models.RBMSymm(symmetries=g.translation_group(), alpha=3)

    if level == 'highest':
        sa = nk.sampler.MetropolisLocal(hi)
    else:
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
    
    if level == 'highest':
        E =  -1 * E
    if exact:
      return E, exact_energy, end - start
    return E, end - start


rbm, ex, runtime = get_energies(L=8, level = 'ground', exact = True)
print(rbm, ex, runtime)
print(rbm-ex)
rbm, ex, runtime = get_energies(L=8, level = 'first', exact = True)
print(rbm, ex, runtime)
print(rbm-ex)
rbm, ex, runtime = get_energies(L=8, level = 'highest', exact = True)
print(rbm, ex, runtime)
print(rbm-ex)
