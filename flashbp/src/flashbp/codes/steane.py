import numpy as np
from .base import Code


class SteaneCode(Code):
    """
    Steane [[7, 1, 3]] code.

    CSS code based on the [7,4,3] Hamming code.
    H_X = H_Z = parity check matrix of the classical [7,4,3] code.
    """

    def __init__(self):
        # Rows correspond to stabilizers:
        #   D0: X0 X2 X4 X6
        #   D1: X1 X2 X5 X6
        #   D2: X3 X4 X5 X6
        H = np.array([
            [1, 0, 1, 0, 1, 0, 1],
            [0, 1, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 1, 1],
        ], dtype=np.uint8)

        self.H_X = H.copy()
        self.H_Z = H.copy()
        self.d   = 3

        # Logical Z = Z0 Z1 Z2  (weight-3 representative)
        # Logical X = X0 X1 X2  (weight-3 representative)
        self.logical_z = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
        self.logical_x = np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.uint8)
