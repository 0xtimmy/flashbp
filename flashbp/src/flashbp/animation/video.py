"""
Stitch a sequence of PNG frames into an MP4 by shelling out to ffmpeg.
"""
import shutil
import subprocess
from pathlib import Path


def check_ffmpeg() -> None:
    """Raise a clear error if `ffmpeg` is not on PATH."""
    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "ffmpeg not found on PATH. Install it and ensure it's accessible.\n"
            "  Windows : winget install ffmpeg  (then restart shell)\n"
            "  macOS   : brew install ffmpeg\n"
            "  Linux   : apt install ffmpeg / dnf install ffmpeg"
        )


def make_video(
    frame_dir:   str | Path,
    output_path: str | Path,
    framerate:   float = 2.0,
    pattern:     str   = "frame_%04d.png",
) -> Path:
    """
    Run ffmpeg to combine `frame_dir/<pattern>` into an h264 mp4.

    The `pad` filter rounds frame dimensions to even numbers, since libx264
    rejects odd dimensions.
    """
    check_ffmpeg()
    frame_dir   = Path(frame_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(framerate),
        "-i", str(frame_dir / pattern),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed (exit {result.returncode}):\n{result.stderr}"
        )
    return output_path
