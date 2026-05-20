"""
End-to-end: a recording from RecordLogger → directory of PNG frames → mp4.
"""
from pathlib import Path

import numpy as np

from .layout import bipartite_layout
from .frames import render_frame
from .video  import make_video


def animate(
    bp,
    recording:   list,
    output_dir:  str | Path,
    shot_index:  int   = 0,
    framerate:   float = 2.0,
    video_name:  str   = "decode.mp4",
    layout:      dict | None = None,
    true_errors: np.ndarray | None = None,
) -> Path:
    """
    Render every iteration of one recorded shot to PNG and stitch into an mp4.

    Parameters
    ----------
    bp : flashbp.FlashBP
        Needed for the parity-check matrix `bp.H`.
    recording : list of shot dicts
        Output of `bp.get_recording()`.
    output_dir : str or Path
        PNG frames are written to `<output_dir>/frames/`, video to
        `<output_dir>/<video_name>`.
    shot_index : int
        Which shot in the recording to animate.
    framerate : float
        Frames per second for the output video.
    video_name : str
        Filename for the mp4 inside `output_dir`.

    Returns
    -------
    Path to the produced mp4.
    """
    if not recording:
        raise ValueError("recording is empty — did the decoder record any shots?")

    output_dir = Path(output_dir)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    H = np.asarray(bp.H)
    num_checks, num_vars = H.shape
    if layout is None:
        layout = bipartite_layout(num_vars, num_checks)

    shot = recording[shot_index]
    iters = shot["iterations"]
    if not iters:
        raise ValueError(f"shot {shot_index} has no recorded iterations.")

    for i, it in enumerate(iters):
        render_frame(it, H, layout, frames_dir / f"frame_{i:04d}.png",
                     true_errors=true_errors)

    return make_video(frames_dir, output_dir / video_name, framerate=framerate)
