def _add_torch_dll_directory():
    """Let Windows find LibTorch DLLs before importing the pybind module."""
    import os
    import sys
    from pathlib import Path

    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return

    try:
        import torch
    except Exception:
        return

    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.exists():
        os.add_dll_directory(str(torch_lib))


_add_torch_dll_directory()

from ._flashbp import FlashBP, torch_available, torch_diagnostics
from .config import DecoderConfig
from . import analytics, animation, codes

__all__ = [
    "FlashBP", "DecoderConfig",
    "torch_available", "torch_diagnostics",
    "analytics", "animation", "codes",
]
