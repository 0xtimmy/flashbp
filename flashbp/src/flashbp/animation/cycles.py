"""
Find cycles in a Tanner graph and render each as a green-highlighted frame.

A Tanner graph is bipartite (variable nodes ↔ check nodes), so every simple
cycle has even length ≥ 4.  Short cycles are the dominant cause of BP failures
— this utility makes them visible.

Node-id convention used internally:
    0 .. num_vars - 1                            : variable v
    num_vars .. num_vars + num_checks - 1        : check  d  (d = id - num_vars)
"""
from pathlib import Path

import matplotlib.pyplot as plt
import networkx     as nx
import numpy        as np

from .layout import bipartite_layout, edges_from_H
from .video  import make_video


# ---------------------------------------------------------------------------

def _build_tanner_graph(H: np.ndarray) -> nx.Graph:
    num_checks, num_vars = H.shape
    G = nx.Graph()
    G.add_nodes_from(range(num_vars + num_checks))
    rows, cols = np.nonzero(H)
    for d, v in zip(rows, cols):
        G.add_edge(int(v), num_vars + int(d))
    return G


def find_cycles(H: np.ndarray, max_length: int, syndrome=None) -> list[list[int]]:
    """
    Enumerate all simple cycles in the Tanner graph with length ≤ `max_length`,
    sorted by ascending length.

    Cycles are returned as lists of tagged node ids (see module docstring).
    """
    num_checks, num_vars = H.shape
    if max_length < 4:
        return []
    G = _build_tanner_graph(H)
    cycles: list[list[int]] = []
    for c in nx.simple_cycles(G, length_bound=max_length):
        if len(c) >= 4:
            if syndrome is None or any(syndrome[n - num_vars] for n in c if n >= num_vars):
                cycles.append(c)
    cycles.sort(key=len)
    return cycles


# ---------------------------------------------------------------------------

CYCLE_COLOR = "#2ca02c"   # tab:green
FAINT_EDGE  = "#d8d8d8"
FAINT_NODE  = "#bbbbbb"


def render_cycle_frame(
    cycle:        list[int],
    cycle_idx:    int,
    total_cycles: int,
    H:            np.ndarray,
    layout:       dict,
    output_path:  str | Path,
    figsize:      tuple[float, float] | None = None,
) -> None:
    """Render one cycle highlighted in green over a faded Tanner graph."""
    num_checks, num_vars = H.shape

    if figsize is None:
        figsize = layout.get("figsize") or (
            8.0, min(max(8.0, 0.25 * max(num_vars, num_checks)), 30.0)
        )
    base_size = layout.get(
        "node_size",
        max(60.0, 4000.0 / max(num_vars, num_checks)),
    )

    var_pos   = layout["var_pos"]
    check_pos = layout["check_pos"]
    edges     = edges_from_H(H)

    cycle_edges_set = {
        frozenset((cycle[i], cycle[(i + 1) % len(cycle)]))
        for i in range(len(cycle))
    }
    cycle_vars   = {n         for n in cycle if n <  num_vars}
    cycle_checks = {n - num_vars for n in cycle if n >= num_vars}

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ── edges: faint background first, then cycle on top ────────────────────
    cycle_segments = []
    for d, v in edges:
        x1, y1 = var_pos[v]
        x2, y2 = check_pos[d]
        if frozenset((v, num_vars + d)) in cycle_edges_set:
            cycle_segments.append(((x1, x2), (y1, y2)))
        else:
            ax.plot([x1, x2], [y1, y2],
                    color=FAINT_EDGE, linewidth=0.4, alpha=0.5, zorder=1)
    for (xs, ys) in cycle_segments:
        ax.plot(xs, ys, color=CYCLE_COLOR, linewidth=3.0, alpha=1.0, zorder=2)

    # ── variable nodes ──────────────────────────────────────────────────────
    var_xs    = [var_pos[v][0] for v in range(num_vars)]
    var_ys    = [var_pos[v][1] for v in range(num_vars)]
    var_faces = [CYCLE_COLOR if v in cycle_vars else "white"
                 for v in range(num_vars)]
    var_edges = ["black" if v in cycle_vars else FAINT_NODE
                 for v in range(num_vars)]
    var_lws   = [2.0    if v in cycle_vars else 0.4
                 for v in range(num_vars)]
    ax.scatter(var_xs, var_ys, s=base_size, c=var_faces,
               edgecolors=var_edges, linewidths=var_lws, zorder=3)

    # ── check nodes ─────────────────────────────────────────────────────────
    chk_xs    = [check_pos[d][0] for d in range(num_checks)]
    chk_ys    = [check_pos[d][1] for d in range(num_checks)]
    chk_faces = [CYCLE_COLOR if d in cycle_checks else "white"
                 for d in range(num_checks)]
    chk_edges = ["black" if d in cycle_checks else FAINT_NODE
                 for d in range(num_checks)]
    chk_lws   = [2.0    if d in cycle_checks else 0.4
                 for d in range(num_checks)]
    ax.scatter(chk_xs, chk_ys, s=base_size, c=chk_faces,
               edgecolors=chk_edges, linewidths=chk_lws,
               marker="s", zorder=3)

    ax.set_title(
        f"cycle {cycle_idx + 1}/{total_cycles}    length={len(cycle)}",
        fontsize=12,
    )

    fig.savefig(output_path, dpi=150, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


# ---------------------------------------------------------------------------

def animate_cycles(
    bp,
    output_dir: str | Path,
    max_dist:   int,
    framerate:  float = 2.0,
    video_name: str   = "cycles.mp4",
    layout:     dict | None = None,
    syndrome=None
) -> Path:
    """
    Enumerate every simple cycle in `bp`'s Tanner graph with length ≤ max_dist,
    render each as a frame (sorted ascending by length), and stitch into an mp4.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H = np.asarray(bp.H)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)

    cycles = find_cycles(H, max_length=max_dist, syndrome=syndrome)
    if not cycles:
        raise ValueError(f"No cycles of length <= {max_dist} found.")

    lengths = [len(c) for c in cycles]
    print(f"Found {len(cycles)} cycles  "
          f"(lengths {min(lengths)}..{max(lengths)})")

    for i, cycle in enumerate(cycles):
        render_cycle_frame(cycle, i, len(cycles), H, layout,
                           frames_dir / f"frame_{i:04d}.png")

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)
