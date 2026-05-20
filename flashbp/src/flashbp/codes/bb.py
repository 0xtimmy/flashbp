import numpy as np
from .base import Code
from ._gf2 import null_space, quotient_basis


class BBCode(Code):
    """
    Bivariate Bicycle (BB) code from Bravyi et al., Nature 2024.

    Defined over the group Z_l x Z_m by two polynomials:
        a(x, y) = sum of x^i * y^j  for (i, j) in a_terms
        b(x, y) = sum of x^i * y^j  for (i, j) in b_terms

    Parity check matrices:
        A = poly_matrix(a_terms, l, m)   [lm x lm]
        B = poly_matrix(b_terms, l, m)   [lm x lm]

        H_X = [A | B]        [lm x 2lm]
        H_Z = [B^T | A^T]    [lm x 2lm]

    Parameters
    ----------
    l, m      : dimensions of Z_l x Z_m
    a_terms   : list of (i, j) exponent pairs for a(x, y)
    b_terms   : list of (i, j) exponent pairs for b(x, y)
    d         : distance (None if unknown)
    """

    def __init__(self, l: int, m: int, a_terms: list, b_terms: list, d: int | None = None):
        self.l = l
        self.m = m
        self.d = d

        A = self._poly_matrix(a_terms, l, m)
        B = self._poly_matrix(b_terms, l, m)

        self.H_X = np.hstack([A,    B   ])          # (lm x 2lm)
        self.H_Z = np.hstack([B.T,  A.T ])          # (lm x 2lm)

        self.logical_z, self.logical_x = self._compute_logicals()

    # ------------------------------------------------------------------

    @staticmethod
    def _cyclic_shift(n: int, k: int) -> np.ndarray:
        """n x n cyclic permutation matrix that shifts by k positions."""
        S = np.zeros((n, n), dtype=np.uint8)
        for i in range(n):
            S[i, (i + k) % n] = 1
        return S

    @staticmethod
    def _poly_matrix(terms: list, l: int, m: int) -> np.ndarray:
        """Build lm x lm GF(2) matrix from exponent pairs (i, j) -> x^i * y^j."""
        A = np.zeros((l * m, l * m), dtype=np.uint8)
        for (i, j) in terms:
            A ^= np.kron(
                BBCode._cyclic_shift(l, i),
                BBCode._cyclic_shift(m, j),
            )
        return A

    def _compute_logicals(self):
        log_z = quotient_basis(null_space(self.H_X), self.H_Z)
        log_x = quotient_basis(null_space(self.H_Z), self.H_X)
        return log_z, log_x

    # ------------------------------------------------------------------
    # Named instances from Bravyi et al. (2024), Table 1
    # ------------------------------------------------------------------

    @classmethod
    def smbb_code(cls) -> "BBCode":
        """
        Small BB code for exact ML contraction demos.

        This is a tiny l=m=2 bivariate-bicycle instance with parameters
        [[8, 2, ?]].  Its DEM has 8 detectors and 2 observables, so the dense
        ML contraction state space has only 2^(8+2) = 1024 states.
        """
        return cls(
            l=2, m=2,
            a_terms=[(0, 0), (0, 1)],
            b_terms=[(0, 0), (1, 0)],
            d=None,
        )

    @classmethod
    def hbb_code(cls) -> "BBCode":
        """
        Heavy-but-exact BB code for GPU ML contraction studies.

        This l=4, m=5 instance has parameters [[40, 2, ?]].  Its DEM has
        40 detectors and 2 observables, but the CSS halves split cleanly into
        two connected components.  With split ML decoding, each half contracts
        2^(20+1) = 2,097,152 states instead of a full 2^(40+2) state tensor.
        """
        return cls(
            l=4, m=5,
            a_terms=[(0, 0), (0, 1)],
            b_terms=[(0, 0), (1, 0)],
            d=None,
        )

    @classmethod
    def gross_code(cls) -> "BBCode":
        """
        [[72, 12, 6]] bivariate bicycle code.

        a(x,y) = x^3 + y + y^2
        b(x,y) = y^3 + x + x^2
        """
        return cls(
            l=6, m=6,
            a_terms=[(3, 0), (0, 1), (0, 2)],
            b_terms=[(0, 3), (1, 0), (2, 0)],
            d=6,
        )

    @classmethod
    def bb_144_12_12(cls) -> "BBCode":
        """
        [[144, 12, 12]] bivariate bicycle code.

        a(x,y) = x^3 + y + y^2
        b(x,y) = y^3 + x + x^2
        """
        return cls(
            l=12, m=6,
            a_terms=[(3, 0), (0, 1), (0, 2)],
            b_terms=[(0, 3), (1, 0), (2, 0)],
            d=12,
        )
