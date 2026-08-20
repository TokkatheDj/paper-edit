"""Make the venv's bundled CUDA DLLs visible to ctranslate2 on Windows.

faster-whisper's CUDA backend loads cublas64_12.dll / cudnn64_9.dll by name.
pip puts them under site-packages/nvidia/*/bin, which is not on the DLL search
path, so `device="cuda"` constructs fine and then fails at first inference with
"Library cublas64_12.dll is not found". Call enable_cuda() before importing
faster_whisper.
"""
import os
import sys
from pathlib import Path

_NVIDIA_SUBDIRS = ("cublas", "cudnn", "cuda_nvrtc")


def enable_cuda() -> list[str]:
    """Add bundled NVIDIA DLL dirs to the search path. Returns dirs added."""
    if sys.platform != "win32":
        return []
    site = Path(sys.executable).parent.parent / "Lib" / "site-packages" / "nvidia"
    added = []
    for sub in _NVIDIA_SUBDIRS:
        d = site / sub / "bin"
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = f"{d}{os.pathsep}" + os.environ.get("PATH", "")
            added.append(str(d))
    return added


def cuda_available() -> bool:
    """True if a CUDA device is usable end-to-end (not just constructible)."""
    enable_cuda()
    try:
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
