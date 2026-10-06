"""core -- the filter library (split from builder.py).


builder.py -- ground-state filter (cos(H t_i + phi_i) pulses) for the J1-J2
Heisenberg chain: minimax pulse optimization with certification, Trotterized
circuits, rigorous error bounds, grid snapping, simulation and resource counts.

Conventions
-----------
* Hamiltonians passed to the circuit code are the *rescaled* ones
  H_s = (H - shift) / W  (built by hamiltonians.scale_hamiltonian, identity
  term LAST), so pulse times are in scaled-H units.
* Circuit layout: system qubits 0..N-1, ancilla = qubit N (the MSB of the
  statevector index): amplitudes are [anc=0 block | anc=1 block].
* One pulse = H(anc), Rz(2 phi), exp(-i t H (x) Z_anc), H(anc).
      anc = 0 block:  cos(H t + phi) psi
      anc = 1 block: -i sin(H t + phi) psi
* Qubit q <-> chain site q. Qiskit vectors are little-endian; use
  hamiltonians.reorder_axes to go to / from the MPS (site 0 = MSB) ordering.
* Nothing here reads script-level globals.

Scope
-----
Noiseless by design: exact statevector (or exact-unitary) simulation, no gate
or shot noise. The goal is to isolate algorithmic error (filter design,
spectral inputs, Trotter), not to predict device performance.

Rigorous results used (see docstrings of the functions that use them)
----------------------------------------------------------------------
(R1) certify_filter: |d/dE prod cos(E t_i + phi_i)| <= sum |t_i|, so a grid
     max plus (sum|t_i|) h/2 is a certified sup over a continuous interval.
(R2) fidelity_lower_bound: gamma / (gamma + (1-gamma) eta^2).
(R3) trotter_bounds: post-selected state distance <= 2 eps / sqrt(p_g) with
     eps = sum_i alpha t_i^2 / (2 k_i)  (= alpha T^2/(2n) on the uniform grid).
The Lie-Trotter commutator bound itself is the standard one from Childs et al.
(Theory of Trotter error, 2021); I'm quoting it from memory, so please
double-check the exact statement/ordering convention in the paper.
"""
