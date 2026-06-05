# @title
from __future__ import annotations

import numpy as np
from typing import Protocol, runtime_checkable

try:
    import tenpy.networks.site as tsite
    from tenpy.models.lattice import Chain, Square, Triangular, Honeycomb, Kagome
except ImportError:
    raise ImportError("pip install physics-tenpy")


@runtime_checkable
class BaseLattice(Protocol):
    name:      str
    n_sites:   int
    nn_edges:  list[tuple[int, int]]
    nnn_edges: list[tuple[int, int]]


class TenpyLattice:
    def __init__(self, lat, name: str, pbc: bool = True,
                 nnn_key: str | None = "next_nearest_neighbors"):
        self._lat    = lat          # keep the full TeNPy object for downstream use
        self.name    = name
        self.pbc     = pbc
        self.n_sites = lat.N_sites  # use TeNPy's own count, not order.shape[0]

        dim   = lat.dim
        order = lat.order           # (N, dim+1): [x, y, ..., u]

        # Build index map with plain Python ints to avoid numpy scalar hash issues
        self._idx: dict[tuple, int] = {
            tuple(int(v) for v in row): i for i, row in enumerate(order)
        }

        self.nn_edges  = self._build_edges("nearest_neighbors")
        self.nnn_edges = self._build_edges(nnn_key) if nnn_key else []

        self._validate()

    def _build_edges(self, pair_key: str) -> list[tuple[int, int]]:
        if pair_key not in self._lat.pairs:
            return []

        bonds = self._lat.pairs[pair_key]
        dim   = self._lat.dim
        Ls    = np.asarray(self._lat.Ls, dtype=int)

        seen:  set[tuple[int, int]] = set()
        edges: list[tuple[int, int]] = []

        for site_row in self._lat.order:
            xy1 = site_row[:dim].astype(int)   # spatial coords
            u1  = int(site_row[dim])            # unit-cell index

            for bond in bonds:
                bu1 = int(bond[0])
                bu2 = int(bond[1])
                dxy = np.asarray(bond[2], dtype=int).reshape(dim)

                if bu1 != u1:
                    continue

                xy2_raw = xy1 + dxy

                if self.pbc:
                    xy2 = xy2_raw % Ls
                else:
                    if np.any(xy2_raw < 0) or np.any(xy2_raw >= Ls):
                        continue
                    xy2 = xy2_raw

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

    def _validate(self):
        """Cross-check edge count against known coordination numbers."""
        N   = self.n_sites
        nn  = len(self.nn_edges)
        nnn = len(self.nnn_edges)

        expected_z = {
            'chain':       (2,  2),
            'square':      (4,  4),
            'triangular':  (6,  6),
            'honeycomb':   (3,  6),
            'kagome':      (4,  4),
        }
        kind = self.name.split()[0]
        if kind not in expected_z or not self.pbc:
            return

        # Skip validation for lattices so small that PBC causes bond collapse
        # (e.g. 2×2 square: the +x and -x neighbours of a site are the same site).
        # This is a degenerate geometry, not a bug in edge-building.
        Ls = self._lat.Ls                         # e.g. (2, 2) for a 2×2 square
        if any(l < 3 for l in Ls):
            print(f"[{self.name}]  skipping validation: lattice too small "
                  f"(Ls={list(Ls)}) for PBC edge-count check.")
            return

        z_nn, z_nnn  = expected_z[kind]
        expected_nn  = N * z_nn  // 2
        expected_nnn = N * z_nnn // 2

        if nn != expected_nn:
            raise ValueError(
                f"{self.name}: expected {expected_nn} nn edges (z={z_nn}), "
                f"got {nn}. Lattice geometry is wrong."
            )
        if nnn != expected_nnn:
            raise ValueError(
                f"{self.name}: expected {expected_nnn} nnn edges (z={z_nnn}), "
                f"got {nnn}. Lattice geometry is wrong."
            )

        print(f"[{self.name}]  nn={nn} ({z_nn}/site)  "
              f"nnn={nnn} ({z_nnn}/site)  ✓")

    @property
    def tenpy_lat(self):
        """The raw TeNPy lattice object, for Schmidt spectra etc."""
        return self._lat

    def __repr__(self):
        return (f"TenpyLattice(name={self.name!r}, N={self.n_sites}, "
                f"nn={len(self.nn_edges)}, nnn={len(self.nnn_edges)})")


def _site():
    return tsite.SpinHalfSite(conserve="Sz")


def _bc(pbc: bool, dim: int) -> list[str] | str:
    """TeNPy bc: string for 1D, list for 2D."""
    if dim == 1:
        return "periodic" if pbc else "open"
    return ["periodic"] * dim if pbc else ["open"] * dim


def make_lattice(kind: str, L: int = 8, pbc: bool = True) -> TenpyLattice:
    kind = kind.lower()
    site = _site()

    if kind == "chain":
        lat  = Chain(L, site, bc=_bc(pbc, 1))
        name = f"chain L={L}"
        nnn  = "next_nearest_neighbors"

    elif kind == "square":
        Lx, Ly = _factor2(L, n_uc=1, prefer_square=True)
        lat    = Square(Lx, Ly, site, bc=_bc(pbc, 2))
        name   = f"square {Lx}x{Ly}"
        nnn    = "next_nearest_neighbors"

    elif kind == "triangular":
        Lx, Ly = _factor2(L, n_uc=1, prefer_square=True)
        lat    = Triangular(Lx, Ly, site, bc=_bc(pbc, 2))
        name   = f"triangular {Lx}x{Ly}"
        nnn    = "next_nearest_neighbors"

    elif kind == "honeycomb":
        Lx, Ly = _factor2(L, n_uc=2, prefer_square=True)
        lat    = Honeycomb(Lx, Ly, site, bc=_bc(pbc, 2))
        name   = f"honeycomb {Lx}x{Ly}"
        nnn    = "next_nearest_neighbors"

    elif kind == "kagome":
        Lx, Ly = _factor2(L, n_uc=3, prefer_square=True)
        lat    = Kagome(Lx, Ly, site, bc=_bc(pbc, 2))
        name   = f"kagome {Lx}x{Ly}"
        nnn    = "next_nearest_neighbors"

    else:
        raise ValueError(
            f"Unknown lattice {kind!r}. "
            "Choose: chain, square, triangular, honeycomb, kagome"
        )

    return TenpyLattice(lat, name=name, pbc=pbc, nnn_key=nnn)


def _factor2(n_total: int, n_uc: int, prefer_square: bool = True) -> tuple[int, int]:
    if n_total % n_uc != 0:
        raise ValueError(f"{n_total} not divisible by unit-cell size {n_uc}")
    sites = n_total // n_uc
    best  = None
    for Ly in range(1, int(sites**0.5) + 1):
        if sites % Ly == 0:
            best = (sites // Ly, Ly)
    if best is None:
        raise ValueError(f"Cannot factorise {sites}")
    return best
