#!/usr/bin/env bash
#
# setup-cuda-blackwell.sh — build the local CUDA toolchain Syrinx needs on a
# Blackwell (RTX 50-series, sm_120) box running a rolling-release distro.
#
# WHY THIS EXISTS (three constraints that admit exactly one combination):
#
#   1. candle 0.8.4 pins cudarc 0.13.9, whose build.rs hard-rejects any CUDA
#      toolkit newer than 12.8 ("Unsupported cuda toolkit version").
#   2. Blackwell (sm_120) needs CUDA >= 12.8.
#      => CUDA 12.8 exactly. Not the distro's 13.x.
#      (Bumping candle does NOT rescue this: cudarc 0.17.8, used by candle 0.11,
#       still tops out at CUDA 13.0, and Arch currently ships 13.3.)
#   3. CUDA 12.8's nvcc rejects gcc > 14, and its headers predate the glibc 2.41+
#      C23 math additions (cospi/sinpi/rsqrt), which collide even under gcc 14.
#      => a local gcc 14 AND a 6-line header fix.
#
# Everything installs under $HOME. No root, nothing touches the system toolkit.
#
#   ./scripts/setup-cuda-blackwell.sh
#   source scripts/test-all.env      # exports CUDA_ROOT / NVCC_CCBIN / ...
#   cargo build --release --features cuda
#
set -euo pipefail

CUDA_DEST="${CUDA_DEST:-$HOME/cuda-12.8}"
GCC_DEST="${GCC_DEST:-$HOME/gcc14}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
REDIST="https://developer.download.nvidia.com/compute/cuda/redist"
ARCHIVE="https://archive.archlinux.org/packages"

# Only what candle/cudarc actually touch: nvcc + headers to compile the kernels,
# and the .so.12 runtime libs cudarc dlopens (the system has .so.13 — wrong soname).
COMPONENTS=(
  "cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-12.8.93-archive.tar.xz"
  "cuda_cudart/linux-x86_64/cuda_cudart-linux-x86_64-12.8.90-archive.tar.xz"
  "cuda_cccl/linux-x86_64/cuda_cccl-linux-x86_64-12.8.90-archive.tar.xz"
  "cuda_nvrtc/linux-x86_64/cuda_nvrtc-linux-x86_64-12.8.93-archive.tar.xz"
  "libcublas/linux-x86_64/libcublas-linux-x86_64-12.8.4.1-archive.tar.xz"
  "libcurand/linux-x86_64/libcurand-linux-x86_64-10.3.9.90-archive.tar.xz"
)

echo ">> CUDA 12.8 toolkit -> $CUDA_DEST"
mkdir -p "$CUDA_DEST"
for rel in "${COMPONENTS[@]}"; do
  echo "   $(basename "$rel")"
  curl -sSL --retry 8 --retry-all-errors --retry-delay 3 -C - --max-time 1800 \
       -o "$WORK/$(basename "$rel")" "$REDIST/$rel"
  tar -xf "$WORK/$(basename "$rel")" -C "$WORK"
done
for d in "$WORK"/*-archive/; do cp -a "$d"/. "$CUDA_DEST"/; done
[ -d "$CUDA_DEST/lib" ] && [ ! -e "$CUDA_DEST/lib64" ] && ln -s lib "$CUDA_DEST/lib64"

# glibc >= 2.41 declares the C23 cospi/sinpi/rsqrt (+f variants) as noexcept.
# CUDA 12.8's crt/math_functions.h declares the same six WITHOUT an exception
# spec, and C++ rejects the mismatched redeclaration. NVIDIA fixed this in 12.9,
# which cudarc 0.13.9 will not accept — so add __THROW to the six declarations.
# Host-side only: __device__ codegen and all kernel numerics are untouched.
H="$CUDA_DEST/include/crt/math_functions.h"
cp -n "$H" "$H.orig"
python3 - "$H" <<'PY'
import re, sys
p = sys.argv[1]
s = open(p).read()
pat = re.compile(
    r'^(extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ +(?:double|float) +'
    r'(?:cospi|cospif|rsqrt|rsqrtf|sinpi|sinpif)\([^)]*\));$', re.M)
s, n = pat.subn(r'\1 __THROW;', s)
open(p, 'w').write(s)
print(f"   patched {n} declarations (expected 6)")
assert n == 6, f"expected 6 declarations to patch, patched {n}"
PY

echo ">> gcc 14 -> $GCC_DEST"
GCC_PKG=$(curl -sS --max-time 60 "$ARCHIVE/g/gcc/" \
  | grep -oE 'href="gcc-14\.2\.1[^"]*\.pkg\.tar\.zst"' | sed 's/href="//;s/"//' | tail -1)
LIB_PKG=$(curl -sS --max-time 60 "$ARCHIVE/g/gcc-libs/" \
  | grep -oE 'href="gcc-libs-14\.2\.1[^"]*\.pkg\.tar\.zst"' | sed 's/href="//;s/"//' | tail -1)
mkdir -p "$WORK/gccroot" "$GCC_DEST"
for pkg_path in "g/gcc/$GCC_PKG" "g/gcc-libs/$LIB_PKG"; do
  curl -sSL --retry 5 --retry-all-errors --max-time 900 -o "$WORK/g.pkg.tar.zst" "$ARCHIVE/$pkg_path"
  tar --zstd -xf "$WORK/g.pkg.tar.zst" -C "$WORK/gccroot"
done
cp -a "$WORK/gccroot/usr/." "$GCC_DEST"/

echo
echo "=== verify ==="
"$CUDA_DEST/bin/nvcc" --version | tail -1
"$GCC_DEST/bin/g++" --version | head -1
echo 'extern "C" __global__ void k(float* x){ x[threadIdx.x] *= 2.0f; }' > "$WORK/t.cu"
"$CUDA_DEST/bin/nvcc" --gpu-architecture=sm_120 --ptx \
  -ccbin "$GCC_DEST/bin/g++" -o "$WORK/t.ptx" "$WORK/t.cu"
grep -q '\.target sm_120' "$WORK/t.ptx" && echo "sm_120 PTX: OK"
echo
echo "Done. Now:  source scripts/test-all.env && cargo build --release --features cuda"
