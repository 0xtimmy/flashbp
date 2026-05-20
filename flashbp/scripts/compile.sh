#!/usr/bin/env bash
set -euo pipefail

# Build/install flashbp against the CUDA-enabled LibTorch bundled with the
# active Python environment's PyTorch install.
#
# Usage:
#   bash scripts/compile.sh
#   bash scripts/compile.sh --clean
#   PYTHON=/path/to/python bash scripts/compile.sh --clean
#
# Before running, install a CUDA PyTorch wheel, for example:
#   python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python}"
CLEAN=0
CUDA_ARCH="${CUDA_ARCH:-89}"  # RTX 4070 Ti / Ada Lovelace

for arg in "$@"; do
    case "$arg" in
        --clean)
            CLEAN=1
            ;;
        --cuda-arch=*)
            CUDA_ARCH="${arg#--cuda-arch=}"
            ;;
        -h|--help)
            sed -n '1,18p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

cd "$ROOT_DIR"

TORCH_INFO="$("$PYTHON" - <<'PY'
import sys
try:
    import torch
except Exception as exc:
    raise SystemExit(f"ERROR: failed to import torch: {exc}")

print(torch.utils.cmake_prefix_path)
print(torch.__version__)
print(torch.version.cuda or "")
print(torch.cuda.is_available())
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
else:
    print("")
PY
)"

TORCH_PREFIX="$(printf '%s\n' "$TORCH_INFO" | sed -n '1p')"
TORCH_VERSION="$(printf '%s\n' "$TORCH_INFO" | sed -n '2p')"
TORCH_CUDA_VERSION="$(printf '%s\n' "$TORCH_INFO" | sed -n '3p')"
CUDA_AVAILABLE="$(printf '%s\n' "$TORCH_INFO" | sed -n '4p')"
CUDA_DEVICE_COUNT="$(printf '%s\n' "$TORCH_INFO" | sed -n '5p')"
CUDA_DEVICE_NAME="$(printf '%s\n' "$TORCH_INFO" | sed -n '6p')"

echo "Python          : $("$PYTHON" -c 'import sys; print(sys.executable)')"
echo "PyTorch         : $TORCH_VERSION"
echo "Torch CUDA      : ${TORCH_CUDA_VERSION:-not built with CUDA}"
echo "CUDA available  : $CUDA_AVAILABLE"
echo "CUDA devices    : $CUDA_DEVICE_COUNT"
if [[ -n "$CUDA_DEVICE_NAME" ]]; then
    echo "CUDA device[0]  : $CUDA_DEVICE_NAME"
fi
echo "Torch CMake path: $TORCH_PREFIX"
echo "CUDA arch       : $CUDA_ARCH"

if [[ -z "$TORCH_CUDA_VERSION" ]]; then
    echo "ERROR: this PyTorch install is CPU-only. Install a CUDA wheel first." >&2
    exit 1
fi

if [[ "$CUDA_AVAILABLE" != "True" ]]; then
    echo "WARNING: PyTorch is CUDA-built, but CUDA is not available at runtime." >&2
    echo "         The extension will still link LibTorch, but GPU smoke tests may fail." >&2
fi

CUDA_ARGS=()
if [[ -n "$TORCH_CUDA_VERSION" ]]; then
    CUDA_MAJOR_MINOR="$(printf '%s' "$TORCH_CUDA_VERSION" | awk -F. '{print $1 "." $2}')"
    CUDA_ROOT_CANDIDATE="/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v$CUDA_MAJOR_MINOR"
    if [[ -d "$CUDA_ROOT_CANDIDATE" ]]; then
        echo "CUDA toolkit    : $CUDA_ROOT_CANDIDATE"
        CUDA_ARGS+=("-C" "cmake.args=-DCUDAToolkit_ROOT=$CUDA_ROOT_CANDIDATE")
        CUDA_ARGS+=("-C" "cmake.args=-DCMAKE_CUDA_COMPILER=$CUDA_ROOT_CANDIDATE/bin/nvcc.exe")
    else
        echo "WARNING: matching CUDA toolkit was not found at:" >&2
        echo "         $CUDA_ROOT_CANDIDATE" >&2
        echo "         CMake will use the first nvcc on PATH." >&2
        echo "         If configure fails with a CUDA version mismatch, install" >&2
        echo "         CUDA Toolkit v$CUDA_MAJOR_MINOR or install a PyTorch wheel" >&2
        echo "         matching your installed toolkit." >&2
    fi
fi

if [[ "$CLEAN" -eq 1 ]]; then
    echo "Removing build/"
    rm -rf build
fi

"$PYTHON" -m pip install -e . --no-build-isolation \
    -C cmake.args="-DFLASHBP_ENABLE_TORCH=ON" \
    -C cmake.args="-DCMAKE_PREFIX_PATH=$TORCH_PREFIX" \
    -C cmake.args="-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH" \
    "${CUDA_ARGS[@]}"

"$PYTHON" - <<'PY'
import torch
import flashbp
print(flashbp.torch_diagnostics())
PY
