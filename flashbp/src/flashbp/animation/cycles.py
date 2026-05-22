from __future__ import annotations

from pathlib import Path

import numpy as np

from flashbp.analytics.cycles import find_cycles, render_cycle_frame

from .layout import bipartite_layout
from .video import make_video


def animate_cycles(
    bp,
    output_dir: str | Path,
    max_dist: int,
    framerate: float = 2.0,
    video_name: str = "cycles.mp4",
    layout: dict | None = None,
    syndrome=None,
) -> Path:
    """
    Enumerate every simple Tanner-graph cycle with length <= max_dist, render
    each as a frame, and stitch the frames into an mp4.
    """
    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H = np.asarray(bp.H, dtype=np.uint8)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)

    cycles = find_cycles(
        H,
        max_length=max_dist,
        syndrome=syndrome,
        require_active_check=syndrome is not None,
    )
    if not cycles:
        raise ValueError(f"No cycles of length <= {max_dist} found.")

    lengths = [len(c) for c in cycles]
    print(f"Found {len(cycles)} cycles  "
          f"(lengths {min(lengths)}..{max(lengths)})")

    for i, cycle in enumerate(cycles):
        render_cycle_frame(
            cycle,
            i,
            len(cycles),
            H,
            layout,
            frames_dir / f"frame_{i:04d}.png",
            syndrome=syndrome,
        )

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)
