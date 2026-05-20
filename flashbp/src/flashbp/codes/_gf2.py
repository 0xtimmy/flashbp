"""GF(2) linear algebra utilities."""

import numpy as np


def rref(M):
    """
    Reduced row echelon form of M over GF(2).
    Returns (rref_M, pivot_cols) where rref_M contains only the non-zero rows.
    """
    M = M.astype(np.uint8).copy()
    rows, cols = M.shape
    pivot_row = 0
    pivot_cols = []

    for col in range(cols):
        found = -1
        for row in range(pivot_row, rows):
            if M[row, col]:
                found = row
                break
        if found == -1:
            continue
        M[[pivot_row, found]] = M[[found, pivot_row]]
        pivot_cols.append(col)
        for row in range(rows):
            if row != pivot_row and M[row, col]:
                M[row] ^= M[pivot_row]
        pivot_row += 1

    return M[:pivot_row], pivot_cols


def null_space(H):
    """
    Basis for the null space of H over GF(2).
    Returns array of shape (n - rank, n).
    """
    m, n = H.shape
    M = H.astype(np.uint8).copy()
    pivot_row = 0
    pivot_cols = []

    for col in range(n):
        found = -1
        for row in range(pivot_row, m):
            if M[row, col]:
                found = row
                break
        if found == -1:
            continue
        M[[pivot_row, found]] = M[[found, pivot_row]]
        pivot_cols.append(col)
        for row in range(m):
            if row != pivot_row and M[row, col]:
                M[row] ^= M[pivot_row]
        pivot_row += 1

    free_cols = [c for c in range(n) if c not in pivot_cols]
    pivot_dict = {pc: i for i, pc in enumerate(pivot_cols)}

    null_vecs = []
    for fc in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[fc] = 1
        for pc, idx in pivot_dict.items():
            if M[idx, fc]:
                vec[pc] = 1
        null_vecs.append(vec)

    return np.array(null_vecs, dtype=np.uint8) if null_vecs else np.zeros((0, n), dtype=np.uint8)


def quotient_basis(V, H_stab):
    """
    Basis for the quotient space rowspace(V) / rowspace(H_stab) over GF(2).

    Used to find logical operators: V = ker(H_X), H_stab = H_Z gives logical Z,
    and vice versa for logical X.
    """
    n = V.shape[1]
    if V.shape[0] == 0:
        return np.zeros((0, n), dtype=np.uint8)

    stab, pivot_cols = rref(H_stab)
    pivot_dict = {col: idx for idx, col in enumerate(pivot_cols)}

    result = []
    result_pivots = {}

    for v in V:
        v = v.copy()

        # Reduce by stabilizer rows
        for col, idx in pivot_dict.items():
            if v[col]:
                v ^= stab[idx]

        if not v.any():
            continue  # v is in rowspace(H_stab), not a logical

        # Reduce by already-found logical operators
        for col, idx in result_pivots.items():
            if v[col]:
                v ^= result[idx]

        if not v.any():
            continue

        first = int(np.where(v)[0][0])
        result_pivots[first] = len(result)
        result.append(v.copy())

    return np.array(result, dtype=np.uint8) if result else np.zeros((0, n), dtype=np.uint8)
