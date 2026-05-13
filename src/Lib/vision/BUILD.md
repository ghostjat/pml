# PML Vision — Build Reference

## Quick build

```bash
cd /path/to/project
bash src/Lib/vision/build.sh
```

Or let VisionEngine.php auto-compile on first use (development mode).

## Manual GCC command

```bash
gcc -O3 -march=native -mtune=native -mfma \
    -fno-math-errno -funsafe-math-optimizations \
    -fopenmp -funroll-loops -fomit-frame-pointer \
    -D_GNU_SOURCE -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_WRITE_IMPLEMENTATION \
    -shared -fPIC \
    -o src/Lib/vision/libvision.so.1 \
    src/Lib/vision/vision_core.c    \
    src/Lib/vision/vision_resize.c  \
    src/Lib/vision/vision_color.c   \
    src/Lib/vision/vision_filter.c  \
    src/Lib/vision/vision_feature.c \
    src/Lib/vision/vision_augment.c \
    src/Lib/vision/vision_detect.c  \
    src/Lib/vision/vision_segment.c \
    -lm && \
ln -sf libvision.so.1 src/Lib/vision/libvision.so
```

## Verify symbols

```bash
nm -D src/Lib/vision/libvision.so | grep "^[0-9a-f]* T vision_"
```

## Separate from libtensor.so

libvision.so is intentionally separate from libtensor.so:
- TensorEngine.php glob: `src/Lib/*.c`  → does NOT pick up vision/*.c
- VisionEngine.php glob: `src/Lib/vision/*.c` → separate .so
- Zero ABI conflicts: all vision symbols are prefixed `vision_`

## AVX-512 build (if CPU supports it)

```bash
gcc ... -mavx512f -mavx512bw -mavx512vl ... -DVISION_ENABLE_AVX512=1 ...
```

Runtime dispatch via `vision_cpu_features()` handles mixed-capability clusters.

## Debug / memory leak build

```bash
gcc -O0 -g3 -fsanitize=address,undefined \
    -DVISION_DEBUG_MEMORY=1 \
    ... same sources ... -lm
```

Then run any PHP script with `ASAN_OPTIONS=detect_leaks=1`.
