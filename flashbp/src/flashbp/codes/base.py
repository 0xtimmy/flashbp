from abc import ABC, abstractmethod

import numpy as np
import stim


class Code(ABC):
    """
    Abstract base class for CSS quantum error-correcting codes.

    Subclasses must set:
        H_X       : np.ndarray, shape (r_x, n), X-type parity check matrix
        H_Z       : np.ndarray, shape (r_z, n), Z-type parity check matrix
        logical_z : np.ndarray, shape (k, n),   logical Z operators
        logical_x : np.ndarray, shape (k, n),   logical X operators
        d         : int | None,                 distance (None if unknown)
    """

    H_X: np.ndarray
    H_Z: np.ndarray
    logical_z: np.ndarray
    logical_x: np.ndarray
    d: int | None

    @property
    def n(self) -> int:
        """Number of physical qubits."""
        return self.H_X.shape[1]

    @property
    def k(self) -> int:
        """Number of logical qubits."""
        return self.logical_z.shape[0]

    def to_dem(self, p: float) -> stim.DetectorErrorModel:
        """
        Build a stim DetectorErrorModel for independent X/Z noise at rate p.

        Detectors D0 .. D(r_x-1)       : X-syndrome (rows of H_X, detect Z errors)
        Detectors D(r_x) .. D(r_x+r_z-1): Z-syndrome (rows of H_Z, detect X errors)
        Observables L0 .. L(k-1)        : logical qubits
        """
        r_x = self.H_X.shape[0]
        r_z = self.H_Z.shape[0]
        lines = []

        # Z errors — detected by X stabilizers
        for j in range(self.n):
            dets = [f"D{i}"       for i in range(r_x) if self.H_X[i, j]]
            obs  = [f"L{o}"       for o in range(self.k) if self.logical_z[o, j]]
            targets = " ".join(dets + obs)
            if targets:
                lines.append(f"error({p}) {targets}")

        # X errors — detected by Z stabilizers
        for j in range(self.n):
            dets = [f"D{r_x + i}" for i in range(r_z) if self.H_Z[i, j]]
            obs  = [f"L{o}"       for o in range(self.k) if self.logical_x[o, j]]
            targets = " ".join(dets + obs)
            if targets:
                lines.append(f"error({p}) {targets}")

        return stim.DetectorErrorModel("\n".join(lines))

    def __repr__(self) -> str:
        return (f"{type(self).__name__}"
                f"([[{self.n}, {self.k}, {self.d if self.d is not None else '?'}]])")
