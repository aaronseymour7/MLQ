# tenpy_lattices.py

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

try:
    import tenpy
    import tenpy.networks.site as tsite
    from tenpy.models.lattice import (
        Chain, Square, Triangular, Honeycomb, Kagome
    )
except ImportError:
    raise ImportError(
        "pip install physics-tenpy   (or: conda install -c conda-forge tenpy)"
    )


# ---------------------------------------------------------------------------
# Protocol — unchanged from your original interface
# ---------------------------------------------------------------------------
@runtime_checkable
class BaseLattice(Protocol):
    name:       str
    n_sites:    int
    nn_edges:   list[tuple[int, int]]
    nnn_edges:  list[tuple[int, int]]


# ---------------------------------------------------------------------------
# Core adapter
# ---------------------------------------------------------------------------
class TenpyLattice:
    """
    Wraps a TeNPy lattice and exposes the BaseLattice interface.

    Parameters
    ----------
    lat       : TeNPy lattice instance (already constructed)
    name      : human-readable label, e.g. "chain_L12"
    pbc       : whether periodic boundaries were used
    nnn_key   : key in lat.pairs for next-nearest neighbours
                (None → nnn_edges will be empty)
    """

    def __init__(self, lat, name, pbc=True, nnn_key="next_nearest_neighbors"):
        self._lat  = lat
        self.name  = name
        self.pbc   = pbc

        order      = lat.order          # shape (N, dim+1): columns are [x, y, ..., u]
        self.n_sites = order.shape[0]

        # TeNPy order layout: first `dim` columns = spatial coords, last column = u
        dim = lat.dim
        # _idx keys are plain-int tuples of the full row (x, y, ..., u)
        self._idx: dict[tuple, int] = {
            tuple(int(v) for v in row): i for i, row in enumerate(order)
        }

        self.nn_edges  = self._build_edges("nearest_neighbors")
        self.nnn_edges = self._build_edges(nnn_key) if nnn_key else []

    # ------------------------------------------------------------------
    def _build_edges(self, pair_key: str) -> list[tuple[int, int]]:
        if pair_key not in self._lat.pairs:
            return []

        bonds = self._lat.pairs[pair_key]
        dim   = self._lat.dim
        order = self._lat.order          # (N, dim+1): [x, y, ..., u]
        Ls    = np.asarray(self._lat.Ls, dtype=int)   # shape (dim,)

        seen:  set[tuple[int, int]] = set()
        edges: list[tuple[int, int]] = []

        for site_row in order:
            # spatial coords are first `dim` columns; u is the last column
            xy1 = site_row[:dim].astype(int)    # shape (dim,)
            u1  = int(site_row[dim])            # unit-cell index

            for bond in bonds:
                bu1 = int(bond[0])
                bu2 = int(bond[1])
                dxy = np.asarray(bond[2], dtype=int).reshape(dim)

                if bu1 != u1:
                    continue

                xy2_raw = xy1 + dxy
                xy2     = xy2_raw % Ls if self.pbc else xy2_raw

                if not self.pbc and (np.any(xy2_raw < 0) or np.any(xy2_raw >= Ls)):
                    continue

                # key is (x, y, ..., u) — spatial first, u last
                key2 = tuple(xy2.tolist()) + (bu2,)
                j    = self._idx.get(key2, -1)
                if j < 0:
                    continue

                i    = self._idx[tuple(int(v) for v in site_row)]
                edge = (min(i, j), max(i, j))
                if edge[0] != edge[1] and edge not in seen:
                    seen.add(edge)
                    edges.append(edge)

        edges.sort()
        return edges

    def __repr__(self) -> str:
        return (
            f"TenpyLattice(name={self.name!r}, N={self.n_sites}, "
            f"nn={len(self.nn_edges)}, nnn={len(self.nnn_edges)})"
        )


# ---------------------------------------------------------------------------
# Factory — same signature as the old make_lattice()
# ---------------------------------------------------------------------------
def _spin_half_site():
    """Minimal spinless site (TeNPy needs *something*; we only use geometry)."""
    return tsite.SpinHalfSite(conserve="Sz")


def make_lattice(
    kind:    str,
    L:       int  = 8,
    pbc:     bool = True,
    j2:      float = 0.0,        # informational only; geometry is independent
) -> TenpyLattice:
    """
    Factory that mirrors the old make_lattice(kind, L=...) signature.

    Supported kinds
    ---------------
    'chain'       → Chain(L, 1, …)          N = L
    'square'      → Square(Lx, Ly, …)       N = Lx*Ly   (Lx=Ly=√L if square)
    'triangular'  → Triangular(Lx, Ly, …)   N = Lx*Ly
    'honeycomb'   → Honeycomb(Lx, Ly, …)    N = 2*Lx*Ly
    'kagome'      → Kagome(Lx, Ly, …)       N = 3*Lx*Ly

    For 2-D lattices L is interpreted as the *total* number of sites;
    the factory finds integer Lx, Ly such that Lx*Ly*n_uc == L.
    For Honeycomb (n_uc=2) and Kagome (n_uc=3) L must be divisible
    by the unit-cell size.

    PBC is applied along every direction when pbc=True.
    """
    kind = kind.lower()
    site = _spin_half_site()
    bc   = "periodic" if pbc else "open"

    if kind == "chain":
        lat = Chain(L, site, bc=bc)
        name = f"chain L={L}"

    elif kind == "square":
        Lx, Ly = _factor2(L, n_uc=1)
        lat  = Square(Lx, Ly, site, bc=bc)
        name = f"square {Lx}x{Ly}"

    elif kind == "triangular":
        Lx, Ly = _factor2(L, n_uc=1)
        lat  = Triangular(Lx, Ly, site, bc=bc)
        name = f"triangular {Lx}x{Ly}"

    elif kind == "honeycomb":
        Lx, Ly = _factor2(L, n_uc=2)
        lat  = Honeycomb(Lx, Ly, site, bc=bc)
        name = f"honeycomb {Lx}x{Ly}"

    elif kind == "kagome":
        Lx, Ly = _factor2(L, n_uc=3)
        lat  = Kagome(Lx, Ly, site, bc=bc)
        name = f"kagome {Lx}x{Ly}"

    else:
        raise ValueError(
            f"Unknown lattice kind {kind!r}. "
            "Choose from: chain, square, triangular, honeycomb, kagome"
        )

    # TeNPy's Triangular and Kagome use 'next_nearest_neighbors' for NNN.
    # Honeycomb has no NNN key by default (it's a bipartite lattice
    # with a natural NNN across the "other" sublattice — same key).
    nnn_key = (
        "next_nearest_neighbors"
        if kind in ("chain", "square", "triangular", "honeycomb", "kagome")
        else None
    )

    return TenpyLattice(lat, name=name, pbc=pbc, nnn_key=nnn_key)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _factor2(n_total: int, n_uc: int) -> tuple[int, int]:
    """
    Find (Lx, Ly) such that Lx * Ly * n_uc == n_total and Lx >= Ly.
    Prefer the most square factorisation.
    """
    if n_total % n_uc != 0:
        raise ValueError(
            f"n_total={n_total} is not divisible by unit-cell size n_uc={n_uc}"
        )
    sites = n_total // n_uc
    best  = None
    for Ly in range(1, int(sites**0.5) + 1):
        if sites % Ly == 0:
            best = (sites // Ly, Ly)
    if best is None:
        raise ValueError(f"Cannot factorise {sites} into two positive integers")
    return best
