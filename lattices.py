"""
lattices.py
-----------
Lattice geometry definitions for the J1-J2 UCJ ansatz code.

Each lattice class exposes:
    .n_sites      : int   – total number of sites
    .nn_edges     : list of (i, j) tuples  – nearest-neighbour bonds
    .nnn_edges    : list of (i, j) tuples  – next-nearest-neighbour bonds (may be empty)
    .name         : str   – human-readable label
    .dim          : int   – spatial dimension (1 or 2)
    .coords       : list of (x, y) – real-space coordinates (for visualisation)

Factory function:
    make_lattice(kind, L, pbc=True, **kwargs) -> BaseLattice
    kind ∈ {'chain', 'square', 'triangular', 'honeycomb', 'kagome'}

All edge lists are de-duplicated and store i < j.

Notes on periodic boundary conditions
--------------------------------------
  chain        : standard mod-L wrap.
  square       : torus (Lx×Ly).  Requires L = Lx*Ly; pass Lx, Ly as kwargs
                 (default Lx=Ly=sqrt(L)).
  triangular   : torus with the oblique shear vector (default Lx×Ly, same as square).
  honeycomb    : even–odd sub-lattice trick; requires L divisible by 2 and Lx even.
  kagome       : 3-site unit cell; requires L divisible by 3.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Sequence


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dedup(edges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Remove duplicates; always store with i < j."""
    seen = set()
    out  = []
    for a, b in edges:
        e = (min(a, b), max(a, b))
        if e not in seen and e[0] != e[1]:
            seen.add(e)
            out.append(e)
    return sorted(out)


def _grid_idx(ix: int, iy: int, Lx: int, Ly: int, pbc: bool = True) -> int | None:
    if pbc:
        return (iy % Ly) * Lx + (ix % Lx)
    if 0 <= ix < Lx and 0 <= iy < Ly:
        return iy * Lx + ix
    return None


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class BaseLattice:
    name:      str
    n_sites:   int
    nn_edges:  list[tuple[int, int]]
    nnn_edges: list[tuple[int, int]]
    coords:    list[tuple[float, float]]
    dim:       int = 2

    def __post_init__(self):
        self.nn_edges  = _dedup(self.nn_edges)
        self.nnn_edges = _dedup(self.nnn_edges)

    def describe(self):
        print(f"[Lattice]  {self.name}  n_sites={self.n_sites}  "
              f"NN={len(self.nn_edges)}  NNN={len(self.nnn_edges)}  PBC embedded")


# ---------------------------------------------------------------------------
# 1 – Chain  (1D)
# ---------------------------------------------------------------------------

def make_chain(L: int, pbc: bool = True) -> BaseLattice:
    """Standard 1D chain with NN and NNN bonds."""
    nn, nnn = [], []
    for i in range(L):
        j1 = (i + 1) % L
        j2 = (i + 2) % L
        if pbc or i + 1 < L:
            nn.append((i, j1))
        if pbc or i + 2 < L:
            nnn.append((i, j2))
    coords = [(float(i), 0.0) for i in range(L)]
    return BaseLattice(
        name=f"chain-{L}{'p' if pbc else 'o'}",
        n_sites=L, nn_edges=nn, nnn_edges=nnn,
        coords=coords, dim=1)


# ---------------------------------------------------------------------------
# 2 – Square lattice  (2D, Lx × Ly)
# ---------------------------------------------------------------------------

def make_square(Lx: int, Ly: int, pbc: bool = True) -> BaseLattice:
    """
    Square lattice.  NN = horizontal + vertical bonds.
    NNN = diagonal bonds (the two diagonals of each square plaquette).
    """
    N  = Lx * Ly
    nn, nnn = [], []
    for iy in range(Ly):
        for ix in range(Lx):
            i = iy * Lx + ix
            # NN: right and up
            for dix, diy in [(1, 0), (0, 1)]:
                j = _grid_idx(ix + dix, iy + diy, Lx, Ly, pbc)
                if j is not None:
                    nn.append((i, j))
            # NNN: diagonals
            for dix, diy in [(1, 1), (1, -1)]:
                j = _grid_idx(ix + dix, iy + diy, Lx, Ly, pbc)
                if j is not None:
                    nnn.append((i, j))
    coords = [(float(ix), float(iy)) for iy in range(Ly) for ix in range(Lx)]
    return BaseLattice(
        name=f"square-{Lx}x{Ly}{'p' if pbc else 'o'}",
        n_sites=N, nn_edges=nn, nnn_edges=nnn,
        coords=coords, dim=2)


# ---------------------------------------------------------------------------
# 3 – Triangular lattice  (2D, Lx × Ly with oblique PBC)
# ---------------------------------------------------------------------------

def make_triangular(Lx: int, Ly: int, pbc: bool = True) -> BaseLattice:
    """
    Triangular lattice with basis vectors a1=(1,0) and a2=(1/2, √3/2).
    Each site has 6 NN.  NNN are the 6 next-nearest neighbours (distance √3).

    Under PBC the lattice wraps on the oblique torus; open boundary simply
    drops out-of-range neighbours.
    """
    N  = Lx * Ly
    nn, nnn = [], []

    # NN displacements in (ix, iy) lattice coords on the oblique grid
    # The oblique coord offset in x depends on the row: row iy has x-offset iy//2
    # We just work in (ix, iy) indices directly; the "shear" is implicit.
    nn_deltas  = [(1, 0), (0, 1), (1, -1)]   # 3 unique directions (i<j enforced later)
    nnn_deltas = [(2, -1), (1, 1), (-1, 2)]  # 3 unique NNN directions

    for iy in range(Ly):
        for ix in range(Lx):
            i = iy * Lx + ix
            for dix, diy in nn_deltas:
                j = _grid_idx(ix + dix, iy + diy, Lx, Ly, pbc)
                if j is not None:
                    nn.append((i, j))
            for dix, diy in nnn_deltas:
                j = _grid_idx(ix + dix, iy + diy, Lx, Ly, pbc)
                if j is not None:
                    nnn.append((i, j))

    sq3h = math.sqrt(3) / 2
    coords = [(ix + iy * 0.5, iy * sq3h) for iy in range(Ly) for ix in range(Lx)]
    return BaseLattice(
        name=f"triangular-{Lx}x{Ly}{'p' if pbc else 'o'}",
        n_sites=N, nn_edges=nn, nnn_edges=nnn,
        coords=coords, dim=2)


# ---------------------------------------------------------------------------
# 4 – Honeycomb lattice  (2D, Lx × Ly unit cells, 2 sites/cell)
# ---------------------------------------------------------------------------

def make_honeycomb(Lx: int, Ly: int, pbc: bool = True) -> BaseLattice:
    """
    Honeycomb lattice.  Unit cell contains 2 sites: A (sublattice 0) and B (sublattice 1).
    Total sites N = 2 * Lx * Ly.

    NN bonds: each A connects to 3 B neighbours (intra- and inter-cell).
    NNN bonds: same-sublattice second-shell (the 6 NNN of each site are all same sublattice).
    """
    N = 2 * Lx * Ly

    def cell(ix, iy):
        return (iy % Ly) * Lx + (ix % Lx) if pbc else (
            iy * Lx + ix if 0 <= ix < Lx and 0 <= iy < Ly else None)

    def A(ix, iy): c = cell(ix, iy); return 2 * c     if c is not None else None
    def B(ix, iy): c = cell(ix, iy); return 2 * c + 1 if c is not None else None

    nn, nnn = [], []
    for iy in range(Ly):
        for ix in range(Lx):
            a = A(ix, iy)
            b = B(ix, iy)
            if a is None or b is None:
                continue
            # NN: A(ix,iy) -- B(ix,iy)              (intra-cell)
            nn.append((a, b))
            # NN: A(ix,iy) -- B(ix-1, iy)
            nb = B(ix - 1, iy)
            if nb is not None: nn.append((a, nb))
            # NN: A(ix,iy) -- B(ix, iy-1)
            nb = B(ix, iy - 1)
            if nb is not None: nn.append((a, nb))

            # NNN A-A: (ix,iy) -> (ix+1,iy), (ix,iy+1), (ix-1,iy+1)
            for dix, diy in [(1, 0), (0, 1), (-1, 1)]:
                nb = A(ix + dix, iy + diy)
                if nb is not None: nnn.append((a, nb))
            # NNN B-B: (ix,iy) -> (ix+1,iy), (ix,iy+1), (ix+1,iy-1)
            for dix, diy in [(1, 0), (0, 1), (1, -1)]:
                nb = B(ix + dix, iy + diy)
                if nb is not None: nnn.append((b, nb))

    sq3h = math.sqrt(3) / 2
    coords = []
    for iy in range(Ly):
        for ix in range(Lx):
            # A sub-lattice site
            coords.append((ix + iy * 0.5,         iy * sq3h))
            # B sub-lattice site (shifted by (1,0) * 1/3 of bond length)
            coords.append((ix + iy * 0.5 + 1./3., iy * sq3h + sq3h / 3.))

    return BaseLattice(
        name=f"honeycomb-{Lx}x{Ly}{'p' if pbc else 'o'}",
        n_sites=N, nn_edges=nn, nnn_edges=nnn,
        coords=coords, dim=2)


# ---------------------------------------------------------------------------
# 5 – Kagome lattice  (2D, Lx × Ly unit cells, 3 sites/cell)
# ---------------------------------------------------------------------------

def make_kagome(Lx: int, Ly: int, pbc: bool = True) -> BaseLattice:
    """
    Kagome lattice.  Unit cell has 3 sites (A, B, C). Total N = 3 * Lx * Ly.

    NN = bonds forming the corner-sharing triangles (z=4 per site).
    NNN = bonds connecting sites in adjacent hexagons at distance √3 a
          (same sublattice, z=4 per site).

    Basis within unit cell (fractional units):
        A: (0,   0  )
        B: (0.5, 0  )
        C: (0.25, √3/4)   [top of upward triangle]
    """
    N = 3 * Lx * Ly

    def cell(ix, iy):
        if pbc:
            return (iy % Ly) * Lx + (ix % Lx)
        return iy * Lx + ix if 0 <= ix < Lx and 0 <= iy < Ly else None

    def S(sub, ix, iy):
        c = cell(ix, iy)
        return 3 * c + sub if c is not None else None

    nn, nnn = [], []

    for iy in range(Ly):
        for ix in range(Lx):
            a = S(0, ix, iy)   # A
            b = S(1, ix, iy)   # B
            c = S(2, ix, iy)   # C

            if None in (a, b, c):
                continue

            # ──── NN bonds (edges of up-triangle + shared edges with neighbours) ────
            # Upward triangle: A-B, A-C, B-C  (all intra-cell)
            nn += [(a, b), (a, c), (b, c)]

            # Down-triangle bonds connecting cells:
            # B(ix,iy) -- A(ix+1, iy)
            nb = S(0, ix + 1, iy)
            if nb is not None: nn.append((b, nb))
            # C(ix,iy) -- A(ix, iy+1)
            nb = S(0, ix, iy + 1)
            if nb is not None: nn.append((c, nb))
            # C(ix,iy) -- B(ix-1, iy+1)
            nb = S(1, ix - 1, iy + 1)
            if nb is not None: nn.append((c, nb))

            # ──── NNN bonds (same-sublattice next-shell) ────
            # A: neighbours at (±1,0) and (0,±1)
            for dix, diy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = S(0, ix + dix, iy + diy)
                if nb is not None: nnn.append((a, nb))
            # B: same pattern
            for dix, diy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = S(1, ix + dix, iy + diy)
                if nb is not None: nnn.append((b, nb))
            # C: same pattern
            for dix, diy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nb = S(2, ix + dix, iy + diy)
                if nb is not None: nnn.append((c, nb))

    sq3h = math.sqrt(3) / 2
    coords = []
    for iy in range(Ly):
        for ix in range(Lx):
            ox = ix + iy * 0.5     # oblique offset
            oy = iy * sq3h
            coords.append((ox,          oy))            # A
            coords.append((ox + 0.5,    oy))            # B
            coords.append((ox + 0.25,   oy + sq3h/2))  # C

    return BaseLattice(
        name=f"kagome-{Lx}x{Ly}{'p' if pbc else 'o'}",
        n_sites=N, nn_edges=nn, nnn_edges=nnn,
        coords=coords, dim=2)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_lattice(kind: str, L: int | None = None,
                 Lx: int | None = None, Ly: int | None = None,
                 pbc: bool = True, **kwargs) -> BaseLattice:
    """
    Create a lattice by name.

    Parameters
    ----------
    kind : 'chain' | 'square' | 'triangular' | 'honeycomb' | 'kagome'
    L    : total number of sites (used when Lx/Ly not given).
           For 2D lattices a square layout is inferred: Lx=Ly=sqrt(N_cells).
           For honeycomb, N_cells = L//2; for kagome N_cells = L//3.
    Lx, Ly : explicit grid dimensions (override L).
    pbc  : periodic boundary conditions.

    Examples
    --------
    make_lattice('chain', L=16)
    make_lattice('square', L=16)            # → 4×4
    make_lattice('square', Lx=4, Ly=6)     # → 4×6 = 24 sites
    make_lattice('triangular', L=12)        # → 3×4
    make_lattice('honeycomb', L=8)          # → 2×2 unit cells, 8 sites
    make_lattice('kagome', L=12)            # → 2×2 unit cells, 12 sites
    """
    kind = kind.lower().replace('-', '').replace('_', '')

    if kind == 'chain':
        n = L or (Lx or 8)
        return make_chain(n, pbc=pbc)

    # 2D lattices — infer Lx, Ly from L if not given
    def _infer_2d(sites_per_cell: int):
        nonlocal Lx, Ly
        if Lx is not None and Ly is not None:
            return
        if L is None:
            raise ValueError("Provide L or both Lx and Ly.")
        n_cells = L // sites_per_cell
        root    = int(round(math.sqrt(n_cells)))
        if root * root != n_cells:
            # Try rectangular: find factor pair closest to square
            for r in range(root, 0, -1):
                if n_cells % r == 0:
                    Lx, Ly = r, n_cells // r
                    return
            raise ValueError(f"Cannot factor {n_cells} into a rectangle.")
        Lx = Ly = root

    if kind == 'square':
        _infer_2d(1)
        return make_square(Lx, Ly, pbc=pbc)
    elif kind in ('triangular', 'tri'):
        _infer_2d(1)
        return make_triangular(Lx, Ly, pbc=pbc)
    elif kind in ('honeycomb', 'hc'):
        _infer_2d(2)
        if pbc and (Lx < 3 or Ly < 3):
            import warnings
            warnings.warn(
                f"Honeycomb PBC with Lx={Lx}, Ly={Ly}: unit cell may be too small "
                "for correct NNN connectivity (need Lx≥3, Ly≥3). "
                "NNN bond count will be reduced.", stacklevel=2)
        return make_honeycomb(Lx, Ly, pbc=pbc)
    elif kind in ('kagome', 'kag'):
        _infer_2d(3)
        if pbc and (Lx < 3 or Ly < 3):
            import warnings
            warnings.warn(
                f"Kagome PBC with Lx={Lx}, Ly={Ly}: unit cell may be too small "
                "for correct NNN connectivity (need Lx≥3, Ly≥3). "
                "NNN bond count will be reduced.", stacklevel=2)
        return make_kagome(Lx, Ly, pbc=pbc)
    else:
        raise ValueError(f"Unknown lattice kind '{kind}'. "
                         f"Choose from: chain, square, triangular, honeycomb, kagome.")


# ---------------------------------------------------------------------------
# Quick connectivity test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    for spec in [
        ('chain',       dict(L=8)),
        ('square',      dict(L=16)),
        ('triangular',  dict(L=12)),
        ('honeycomb',   dict(L=8)),
        ('kagome',      dict(L=12)),
    ]:
        kind, kw = spec
        lat = make_lattice(kind, **kw)
        lat.describe()
