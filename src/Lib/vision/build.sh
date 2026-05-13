#!/usr/bin/env bash
# PML Vision — production build script
# Usage:  bash src/Lib/vision/build.sh [--debug] [--avx512]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SCRIPT_DIR}/libvision.so.1"
LINK="${SCRIPT_DIR}/libvision.so"

CFLAGS="-O3 -march=native -mtune=native -mfma"
CFLAGS+=" -fno-math-errno -funsafe-math-optimizations"
CFLAGS+=" -fopenmp -funroll-loops -fomit-frame-pointer"
CFLAGS+=" -D_GNU_SOURCE -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_WRITE_IMPLEMENTATION"
CFLAGS+=" -shared -fPIC"
LDFLAGS="-lm"

for arg in "$@"; do
    case $arg in
        --debug)
            CFLAGS="-O0 -g3 -fsanitize=address,undefined -D_GNU_SOURCE"
            CFLAGS+=" -DVISION_DEBUG_MEMORY=1 -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_WRITE_IMPLEMENTATION"
            CFLAGS+=" -shared -fPIC"
            LDFLAGS="-lm"
            ;;
        --avx512)
            CFLAGS+=" -mavx512f -mavx512bw -mavx512vl -DVISION_ENABLE_AVX512=1"
            ;;
    esac
done

SOURCES=(
    "${SCRIPT_DIR}/vision_core.c"
    "${SCRIPT_DIR}/vision_resize.c"
    "${SCRIPT_DIR}/vision_color.c"
    "${SCRIPT_DIR}/vision_filter.c"
    "${SCRIPT_DIR}/vision_feature.c"
    "${SCRIPT_DIR}/vision_augment.c"
    "${SCRIPT_DIR}/vision_detect.c"
    "${SCRIPT_DIR}/vision_segment.c"
)

echo "[PML Vision] Compiling ${#SOURCES[@]} C files..."
# shellcheck disable=SC2086
gcc $CFLAGS -o "$OUT" "${SOURCES[@]}" $LDFLAGS

ln -sf "$(basename "$OUT")" "$LINK"
echo "[PML Vision] Built: $OUT"
echo "[PML Vision] Symbols: $(nm -D "$LINK" | grep -c "^[0-9a-f]* T vision_") exported functions"
