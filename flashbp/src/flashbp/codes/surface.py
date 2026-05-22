import numpy as np

from .base import Code
from ._gf2 import null_space, quotient_basis


class SurfaceCode(Code):
    """
    Rotated planar surface code at odd distance ``d``.

    Layout
    ------
    Data qubits sit on a d × d grid::

        (0,0) (0,1) ... (0,d-1)
        (1,0) (1,1) ... (1,d-1)
          ...
        (d-1,0)      ...

    with linear index ``qidx(r, c) = r*d + c``.

    Stabilisers
    -----------
    Bulk weight-4 plaquettes at every 2×2 corner-cluster
    ``{(r, c), (r, c+1), (r+1, c), (r+1, c+1)}`` for
    ``r, c ∈ [0, d-1)``.  Their type alternates with ``(r + c) mod 2``
    (even → X, odd → Z).

    Boundary weight-2 plaquettes tile the four edges:

    - top / bottom rough boundaries → X stabilisers
      (top: ``c`` odd, bottom: ``c`` even)
    - left / right smooth boundaries → Z stabilisers
      (left: ``r`` even, right: ``r`` odd)

    Counts
    ------
    - ``n = d²`` physical qubits
    - ``r_x = r_z = (d² − 1) / 2`` stabilisers each
    - ``k = 1`` logical qubit
    - distance ``d`` (X and Z)

    The Tanner-graph girth grows quickly with `d`: at small `d` the bulk
    is too small to host the canonical length-8 cycle around a face, so
    local GBP regions at modest ``degree`` stay small.
    """

    def __init__(self, d: int):
        if d < 3 or d % 2 == 0:
            raise ValueError("SurfaceCode: d must be an odd integer >= 3.")
        self.d = d
        n = d * d

        def qidx(r: int, c: int) -> int:
            return r * d + c

        X_stabs: list[list[int]] = []
        Z_stabs: list[list[int]] = []

        # ── Bulk weight-4 plaquettes ───────────────────────────────────────
        for r in range(d - 1):
            for c in range(d - 1):
                quad = [qidx(r, c),     qidx(r, c + 1),
                        qidx(r + 1, c), qidx(r + 1, c + 1)]
                if (r + c) % 2 == 0:
                    X_stabs.append(quad)
                else:
                    Z_stabs.append(quad)

        # ── Boundary weight-2 plaquettes ───────────────────────────────────
        # Top rough (X): c odd
        for c in range(d - 1):
            if c % 2 == 1:
                X_stabs.append([qidx(0, c), qidx(0, c + 1)])
        # Bottom rough (X): c even
        for c in range(d - 1):
            if c % 2 == 0:
                X_stabs.append([qidx(d - 1, c), qidx(d - 1, c + 1)])
        # Left smooth (Z): r even
        for r in range(d - 1):
            if r % 2 == 0:
                Z_stabs.append([qidx(r, 0), qidx(r + 1, 0)])
        # Right smooth (Z): r odd
        for r in range(d - 1):
            if r % 2 == 1:
                Z_stabs.append([qidx(r, d - 1), qidx(r + 1, d - 1)])

        H_X = np.zeros((len(X_stabs), n), dtype=np.uint8)
        for i, supp in enumerate(X_stabs):
            for q in supp:
                H_X[i, q] = 1
        H_Z = np.zeros((len(Z_stabs), n), dtype=np.uint8)
        for i, supp in enumerate(Z_stabs):
            for q in supp:
                H_Z[i, q] = 1

        self.H_X = H_X
        self.H_Z = H_Z

        self.logical_z, self.logical_x = self._compute_logicals()

    def _compute_logicals(self):
        log_z = quotient_basis(null_space(self.H_X), self.H_Z)
        log_x = quotient_basis(null_space(self.H_Z), self.H_X)
        return log_z, log_x

    def __repr__(self) -> str:
        return f"SurfaceCode(d={self.d}, n={self.n}, k={self.k})"
