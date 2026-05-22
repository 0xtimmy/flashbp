"""
Compute fixed node positions for Tanner-graph animation.

Positions are computed once and reused across all frames so nodes don't jitter.
"""
import numpy as np


def bipartite_layout(num_vars: int, num_checks: int) -> dict:
    """
    Two-column layout with variable nodes on the left and check nodes on the right.

    Returns a dict with:
      var_pos:   {var_idx:   (x, y)}
      check_pos: {check_idx: (x, y)}
    Coordinates are in [0, 1] x [0, 1].
    """
    var_ys   = np.linspace(0.95, 0.05, num_vars)   if num_vars   > 1 else [0.5]
    check_ys = np.linspace(0.95, 0.05, num_checks) if num_checks > 1 else [0.5]

    var_pos   = {i: (0.15, float(y)) for i, y in enumerate(var_ys)}
    check_pos = {i: (0.85, float(y)) for i, y in enumerate(check_ys)}
    return {"var_pos": var_pos, "check_pos": check_pos}


def bb_torus_layout(l: int, m: int) -> dict:
    """
    Layout for a Bivariate-Bicycle code on the Z_l x Z_m torus.

    The Z-error and X-error Tanner subgraphs are disconnected (X errors only
    trigger Z-checks and vice versa), so we draw them as two side-by-side
    panels.  Within each panel, at lattice site (i, j):

        check at        (j,        l-1-i)
        L-qubit at      (j + 0.5,  l-1-i      )
        R-qubit at      (j,        l-1-i + 0.5)

    Index convention (matches the order in `Code.to_dem`):
        detectors    [0, lm)     : X-checks
        detectors    [lm, 2lm)   : Z-checks
        variables    [0, lm)     : Z errors on L qubits
        variables    [lm, 2lm)   : Z errors on R qubits
        variables    [2lm, 3lm)  : X errors on L qubits
        variables    [3lm, 4lm)  : X errors on R qubits

    Returns
    -------
    A dict with keys: var_pos, check_pos, figsize, node_size.
    """
    lm        = l * m
    panel_w   = float(m)
    panel_h   = float(l)
    gap       = 1.0
    total_w   = 2 * panel_w + gap
    total_h   = panel_h
    margin    = 0.04
    usable    = 1.0 - 2 * margin

    z_off = 0.0                  # left panel
    x_off = panel_w + gap        # right panel

    def norm(x: float, y: float, panel_x: float) -> tuple[float, float]:
        nx = margin + usable * (x + panel_x) / total_w
        ny = margin + usable * y           / total_h
        return (nx, ny)

    var_pos:   dict[int, tuple[float, float]] = {}
    check_pos: dict[int, tuple[float, float]] = {}

    for i in range(l):
        for j in range(m):
            yi  = (l - 1) - i           # flip so small i sits at the top
            idx = i * m + j

            # ── Z-panel (left): X-checks + Z errors ────────────────────────
            check_pos[idx]              = norm(j,       yi,       z_off)
            var_pos  [idx]              = norm(j + 0.5, yi,       z_off)   # Z on L
            var_pos  [lm + idx]         = norm(j,       yi + 0.5, z_off)   # Z on R

            # ── X-panel (right): Z-checks + X errors ───────────────────────
            check_pos[lm + idx]         = norm(j,       yi,       x_off)
            var_pos  [2 * lm + idx]     = norm(j + 0.5, yi,       x_off)   # X on L
            var_pos  [3 * lm + idx]     = norm(j,       yi + 0.5, x_off)   # X on R

    # Heuristic figure size — ~0.7 inch per lattice unit, plus title/margin.
    figsize   = (total_w * 0.7 + 2.0, total_h * 0.7 + 2.0)
    node_size = max(20.0, 500.0 / max(l, m))

    return {
        "var_pos":   var_pos,
        "check_pos": check_pos,
        "figsize":   figsize,
        "node_size": node_size,
    }


def surface_code_layout(H_X: np.ndarray, H_Z: np.ndarray, d: int) -> dict:
    """
    Layout for the rotated planar surface code.

    The CSS detector-error model has two disconnected Tanner subgraphs:
    Z errors are checked by X stabilizers, and X errors are checked by
    Z stabilizers.  We draw those as side-by-side copies of the d x d data
    lattice, placing each stabilizer at the centroid of its qubit support.

    Index convention (matches ``Code.to_dem``):
        detectors [0, r_x)       : X-checks
        detectors [r_x, r_x+r_z) : Z-checks
        variables [0, n)         : Z errors on data qubits
        variables [n, 2n)        : X errors on data qubits
    """
    r_x, n = H_X.shape
    r_z, n_z = H_Z.shape
    if n_z != n or n != d * d:
        raise ValueError("surface_code_layout expects d*d columns in H_X/H_Z")

    panel_w = float(max(d - 1, 1))
    panel_h = float(max(d - 1, 1))
    gap = max(1.0, 0.35 * d)
    total_w = 2 * panel_w + gap
    total_h = panel_h
    margin = 0.05
    usable = 1.0 - 2 * margin

    z_off = 0.0
    x_off = panel_w + gap

    def qcoord(q: int) -> tuple[float, float]:
        row, col = divmod(int(q), d)
        return float(col), float((d - 1) - row)

    def norm(x: float, y: float, panel_x: float) -> tuple[float, float]:
        nx = margin + usable * (x + panel_x) / total_w
        ny = margin + usable * y / total_h if total_h else 0.5
        return (nx, ny)

    def check_centroid(H: np.ndarray, row: int) -> tuple[float, float]:
        supp = np.flatnonzero(H[row])
        if len(supp) == 0:
            return 0.0, 0.0
        coords = np.asarray([qcoord(q) for q in supp], dtype=np.float64)
        return float(coords[:, 0].mean()), float(coords[:, 1].mean())

    var_pos: dict[int, tuple[float, float]] = {}
    check_pos: dict[int, tuple[float, float]] = {}

    for q in range(n):
        x, y = qcoord(q)
        var_pos[q] = norm(x, y, z_off)
        var_pos[n + q] = norm(x, y, x_off)

    for chk in range(r_x):
        x, y = check_centroid(H_X, chk)
        check_pos[chk] = norm(x, y, z_off)
    for chk in range(r_z):
        x, y = check_centroid(H_Z, chk)
        check_pos[r_x + chk] = norm(x, y, x_off)

    figsize = (total_w * 0.8 + 2.0, total_h * 0.8 + 2.0)
    node_size = max(28.0, 520.0 / max(d, 1))

    return {
        "var_pos": var_pos,
        "check_pos": check_pos,
        "figsize": figsize,
        "node_size": node_size,
    }


def edges_from_H(H: np.ndarray) -> list[tuple[int, int]]:
    """
    Enumerate (check, var) edges in the same row-major order the C++ decoder uses.
    The order matters because msg_v2c / msg_c2v arrays are indexed by edge.
    """
    num_checks, num_vars = H.shape
    edges: list[tuple[int, int]] = []
    for d in range(num_checks):
        for v in range(num_vars):
            if H[d, v]:
                edges.append((d, v))
    return edges
