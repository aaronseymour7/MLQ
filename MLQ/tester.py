"""
main.py
=======
UCJ ground-state circuit runner with fidelity against exact Lanczos solution.

Global config
-------------
  J1, J2    : Heisenberg couplings
  LATTICE   : BaseLattice object
  VARIANT   : UCJ variant ('re' | 'im' | 'g')
  K_LAYERS  : number of UCJ layers
"""



# =============================================================================
# ── GLOBAL CONFIGURATION ─────────────────────────────────────────────────────
# =============================================================================

J1       = 1.0
J2       = 0.0

N_SITES  = 12
LATTICE: BaseLattice = make_lattice("chain", L=N_SITES)

TARGET_BASIS = ["cx", "rz", "h", "s", "sdg"]

VARIANT  = 'g'    # 're' | 'im' | 'g'
K_LAYERS = 1

# =============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# =============================================================================

_DIVIDER = "─" * 72


def _header(title: str) -> None:
    print(f"\n{_DIVIDER}")
    print(f"  {title}")
    print(_DIVIDER)


def _print_gate_counts(label: str, qc) -> None:
    ops   = qc.count_ops()
    total = sum(ops.values())
    print(f"\n  [{label}]  qubits={qc.num_qubits}  depth={qc.depth()}  "
          f"total_gates={total}")
    for gate in TARGET_BASIS:
        print(f"    {gate:<8} {ops.get(gate, 0)}")
    others = {g: c for g, c in ops.items() if g not in TARGET_BASIS}
    if others:
        print("    --- non-basis gates (not yet decomposed) ---")
        for gate, cnt in sorted(others.items(), key=lambda x: -x[1]):
            print(f"    {gate:<8} {cnt}")


def _fidelity_with_sector(
    sv_full: np.ndarray,
    psi_exact: np.ndarray,
    basis: np.ndarray,
    n: int,
) -> float:
    """
    Project the full 2^n statevector onto the fixed-Sz sector and compute
    |<psi_exact | psi_sector>|^2.
    """
    sector_sv = sv_full[basis]
    norm = np.linalg.norm(sector_sv)
    if norm < 1e-14:
        return 0.0
    sector_sv = sector_sv / norm
    overlap   = np.dot(np.conj(psi_exact), sector_sv)
    return float(np.abs(overlap) ** 2)


# =============================================================================
# ── MAIN ─────────────────────────────────────────────────────────────────────
# =============================================================================

def main() -> dict:
    n = LATTICE.n_sites

    _header(f"UCJ circuit runner  —  N={N_SITES}  J1={J1}  J2={J2}  "
            f"lattice={LATTICE.name}")

    # ── 1. Exact ground state (Lanczos) ───────────────────────────────────────
    _header("Lanczos exact diagonalisation")
    n_up = _get_n_up(n)
    e0, psi_exact, basis, idx_map = get_ground_state(
        n, n_up,
        LATTICE.nn_edges, LATTICE.nnn_edges,
        J1, J2,
    )
    print(f"  N={n}  n_up={n_up}  sector_dim={len(basis)}")
    print(f"  E0       = {e0:.10f}")
    print(f"  E0/site  = {e0/n:.10f}")

    # ── 2. Build + optimise UCJ circuit ───────────────────────────────────────
    _header("UCJ circuit construction + optimisation")
    tqc_ucj, gate_counts_ucj, depth_ucj = build_ucj(
        LATTICE, J1, J2,
        variant=VARIANT,
        k_layers=K_LAYERS,
        basis_gates=TARGET_BASIS,
    )

    # ── 3. Fidelity ───────────────────────────────────────────────────────────
    _header("UCJ fidelity vs exact ground state")
    sv_ucj  = np.array(Statevector(tqc_ucj))
    fidelity = _fidelity_with_sector(sv_ucj, psi_exact, basis, n)
    print(f"  Fidelity = {fidelity:.8f}")

    # ── 4. Gate counts ────────────────────────────────────────────────────────
    _header("Gate counts  —  target basis: " + str(TARGET_BASIS))
    _print_gate_counts("UCJ", tqc_ucj)

    print(f"\n{_DIVIDER}")

    return {
        "e0":        e0,
        "fidelity":  fidelity,
        "qc":        tqc_ucj,
        "gate_counts": gate_counts_ucj,
        "depth":     depth_ucj,
    }


if __name__ == "__main__":
    results = main()
