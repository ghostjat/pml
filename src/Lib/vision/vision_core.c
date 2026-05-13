/*
 * PML Vision — Core: lifecycle, I/O, format conversion, tensor bridge
 *
 * Build (from project root):
 *   gcc -O3 -march=native -mtune=native -mfma -fno-math-errno \
 *       -funsafe-math-optimizations -fopenmp -funroll-loops    \
 *       -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC       \
 *       -DSTB_IMAGE_IMPLEMENTATION -DSTB_IMAGE_WRITE_IMPLEMENTATION \
 *       -o src/Lib/vision/libvision.so.1                       \
 *       src/Lib/vision/vision_core.c src/Lib/vision/vision_resize.c \
 *       src/Lib/vision/vision_color.c src/Lib/vision/vision_filter.c \
 *       src/Lib/vision/vision_feature.c src/Lib/vision/vision_augment.c \
 *       src/Lib/vision/vision_detect.c src/Lib/vision/vision_segment.c \
 *       -lm
 */

#define VISION_INTERNAL
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#define STBI_NO_HDR          /* exclude HDR/PFM to reduce binary size */

#include "vision.h"
#include "../stb_image.h"
#include "../stb_image_write.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <inttypes.h>
#include <cpuid.h>
#include <omp.h>

#ifdef __AVX2__
#  include <immintrin.h>
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * 0. ERROR HANDLING
 * ═══════════════════════════════════════════════════════════════════════ */
static char  _vision_err_buf[512] = {0};
static int   _vision_had_error    = 0;

int vision_check_error(void) { return _vision_had_error; }

const char* vision_get_last_error(void) { return _vision_err_buf; }

void vision_clear_error(void)
{
    _vision_had_error   = 0;
    _vision_err_buf[0]  = '\0';
}

void vision_set_error(const char* msg)
{
    snprintf(_vision_err_buf, sizeof(_vision_err_buf), "%s", msg);
    _vision_had_error = 1;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 1. MEMORY DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */
static VisionMemStats _mem_stats = {0, 0, 0, 0, 0};
static omp_lock_t     _mem_lock;
static int            _mem_lock_init = 0;

static void _ensure_mem_lock(void)
{
    if (!_mem_lock_init) {
        omp_init_lock(&_mem_lock);
        _mem_lock_init = 1;
    }
}

void* vision_alloc(size_t bytes)
{
    _ensure_mem_lock();
    if (bytes == 0) return NULL;

    void* ptr = NULL;
    if (posix_memalign(&ptr, VISION_ALIGN, bytes) != 0 || !ptr) {
        VISION_ERR("FATAL: vision_alloc failed for %zu bytes", bytes);
    }
    memset(ptr, 0, bytes);

    omp_set_lock(&_mem_lock);
    _mem_stats.bytes_allocated += (int64_t)bytes;
    _mem_stats.images_allocated++;
    int64_t live = _mem_stats.bytes_allocated - _mem_stats.bytes_freed;
    if (live > _mem_stats.peak_bytes) _mem_stats.peak_bytes = live;
    omp_unset_lock(&_mem_lock);

    return ptr;
}

void vision_dealloc(void* ptr, size_t bytes)
{
    if (!ptr) return;
    _ensure_mem_lock();
    free(ptr);

    omp_set_lock(&_mem_lock);
    _mem_stats.bytes_freed += (int64_t)bytes;
    _mem_stats.images_freed++;
    omp_unset_lock(&_mem_lock);
}

const VisionMemStats* vision_mem_stats(void)
{
    return &_mem_stats;
}

void vision_mem_stats_reset(void)
{
    _ensure_mem_lock();
    omp_set_lock(&_mem_lock);
    memset(&_mem_stats, 0, sizeof(_mem_stats));
    omp_unset_lock(&_mem_lock);
}

/* ═══════════════════════════════════════════════════════════════════════
 * 2. CPU FEATURE DETECTION
 * ═══════════════════════════════════════════════════════════════════════ */
static VisionCPUFeatures _cpu_features = {0, 0, 0, 0, 0, 0};
static int               _cpu_detected = 0;

const VisionCPUFeatures* vision_cpu_features(void)
{
    if (_cpu_detected) return &_cpu_features;

    unsigned int eax, ebx, ecx, edx;

    /* SSE 4.2 and AVX */
    if (__get_cpuid(1, &eax, &ebx, &ecx, &edx)) {
        _cpu_features.has_sse42 = (ecx >> 20) & 1;
        _cpu_features.has_avx   = (ecx >> 28) & 1;
        _cpu_features.has_fma   = (ecx >> 12) & 1;
    }

    /* AVX2 and AVX-512 via leaf 7 */
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        _cpu_features.has_avx2    = (ebx >> 5)  & 1;
        _cpu_features.has_avx512f = (ebx >> 16) & 1;
        _cpu_features.has_avx512bw= (ebx >> 30) & 1;
    }

    _cpu_detected = 1;
    return &_cpu_features;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 3. IMAGE LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_image_create(int width, int height, int channels,
                                  int format, int layout, int color_space)
{
    if (width <= 0 || height <= 0 || channels <= 0 || channels > 4)
        VISION_ERR("vision_image_create: invalid dims %dx%dx%d", width, height, channels);

    VisionImage* img = (VisionImage*)malloc(sizeof(VisionImage));
    if (!img) VISION_ERR("vision_image_create: struct alloc failed");
    memset(img, 0, sizeof(VisionImage));

    size_t elem_sz = vision_element_size(format);
    size_t stride;

    if (layout == VISION_LAYOUT_HWC) {
        stride = vision_row_stride(width, channels, format);
        img->data_size = stride * (size_t)height;
    } else {
        /* CHW: stride = bytes per channel plane */
        stride = (size_t)height * (size_t)width * elem_sz;
        /* align each plane */
        stride = (stride + VISION_ALIGN - 1) & ~(size_t)(VISION_ALIGN - 1);
        img->data_size = stride * (size_t)channels;
    }

    img->data = (uint8_t*)vision_alloc(img->data_size);
    if (!img->data) { free(img); return NULL; }

    img->width       = width;
    img->height      = height;
    img->channels    = channels;
    img->format      = format;
    img->layout      = layout;
    img->color_space = color_space;
    img->stride      = stride;
    img->owns_data   = 1;
    img->_pad        = 0;
    return img;
}

VisionImage* vision_image_create_from_data(void* data, int width, int height,
                                            int channels, int format, int layout,
                                            int take_ownership)
{
    if (!data || width <= 0 || height <= 0 || channels <= 0)
        VISION_ERR("vision_image_create_from_data: null data or bad dims");

    VisionImage* img = (VisionImage*)malloc(sizeof(VisionImage));
    if (!img) VISION_ERR("vision_image_create_from_data: struct alloc failed");

    size_t elem_sz = vision_element_size(format);
    size_t stride  = layout == VISION_LAYOUT_HWC
        ? vision_row_stride(width, channels, format)
        : ((size_t)height * width * elem_sz + VISION_ALIGN - 1) & ~(size_t)(VISION_ALIGN - 1);

    img->data        = (uint8_t*)data;
    img->stride      = stride;
    img->data_size   = stride * (layout == VISION_LAYOUT_HWC
                                  ? (size_t)height : (size_t)channels);
    img->width       = width;
    img->height      = height;
    img->channels    = channels;
    img->format      = format;
    img->layout      = layout;
    img->color_space = VISION_COLOR_RGB;
    img->owns_data   = take_ownership ? 1 : 0;
    img->_pad        = 0;
    return img;
}

VisionImage* vision_image_clone(const VisionImage* src)
{
    if (!src) VISION_ERR("vision_image_clone: null src");

    VisionImage* dst = vision_image_create(src->width, src->height, src->channels,
                                            src->format, src->layout);
    if (!dst) return NULL;

    memcpy(dst->data, src->data, src->data_size);
    dst->color_space = src->color_space;
    return dst;
}

VisionImage* vision_image_view(VisionImage* src)
{
    if (!src) VISION_ERR("vision_image_view: null src");

    VisionImage* v = (VisionImage*)malloc(sizeof(VisionImage));
    if (!v) VISION_ERR("vision_image_view: struct alloc failed");
    *v = *src;
    v->owns_data = 0;   /* view does not own the buffer */
    return v;
}

void vision_image_free(VisionImage* img)
{
    if (!img) return;
    if (img->owns_data && img->data) {
        vision_dealloc(img->data, img->data_size);
        img->data = NULL;
    }
    free(img);
}

/* Accessors */
int    vision_image_width(const VisionImage* i)       { return i ? i->width : 0; }
int    vision_image_height(const VisionImage* i)      { return i ? i->height : 0; }
int    vision_image_channels(const VisionImage* i)    { return i ? i->channels : 0; }
int    vision_image_format(const VisionImage* i)      { return i ? i->format : 0; }
int    vision_image_layout(const VisionImage* i)      { return i ? i->layout : 0; }
int    vision_image_color_space(const VisionImage* i) { return i ? i->color_space : 0; }
size_t vision_image_stride(const VisionImage* i)      { return i ? i->stride : 0; }
size_t vision_image_data_size(const VisionImage* i)   { return i ? i->data_size : 0; }
void*  vision_image_data_ptr(const VisionImage* i)    { return i ? i->data : NULL; }

/* ═══════════════════════════════════════════════════════════════════════
 * 4. IMAGE I/O  (stb_image backend)
 * ═══════════════════════════════════════════════════════════════════════ */

/* stb_image gives us a heap buffer; we copy into aligned memory then
   free the stb buffer. This keeps our ownership model clean. */

VisionImage* vision_imread(const char* path, int desired_channels)
{
    if (!path) VISION_ERR("vision_imread: null path");

    int w, h, ch;
    uint8_t* raw = stbi_load(path, &w, &h, &ch, desired_channels);
    if (!raw) VISION_ERR("vision_imread: %s — %s", path, stbi_failure_reason());

    int actual_ch = desired_channels > 0 ? desired_channels : ch;
    VisionImage* img = vision_image_create(w, h, actual_ch,
                                            VISION_FMT_UINT8, VISION_LAYOUT_HWC);
    if (!img) { stbi_image_free(raw); return NULL; }

    /* stb_image uses tightly packed rows; we use padded rows */
    size_t src_row = (size_t)w * actual_ch;
    for (int r = 0; r < h; r++) {
        memcpy(img->data + r * img->stride, raw + (size_t)r * src_row, src_row);
    }
    stbi_image_free(raw);
    return img;
}

int vision_imwrite(const char* path, const VisionImage* img)
{
    if (!path || !img) { vision_set_error("vision_imwrite: null arg"); return 0; }
    if (img->format != VISION_FMT_UINT8) {
        vision_set_error("vision_imwrite: only UINT8 supported directly; convert first");
        return 0;
    }

    /* flatten padded rows into a tight buffer for stb */
    size_t row_bytes = (size_t)img->width * img->channels;
    uint8_t* tight = (uint8_t*)malloc(row_bytes * img->height);
    if (!tight) { vision_set_error("vision_imwrite: out of memory"); return 0; }
    for (int r = 0; r < img->height; r++) {
        memcpy(tight + r * row_bytes, img->data + r * img->stride, row_bytes);
    }

    int ok = 0;
    const char* ext = strrchr(path, '.');
    if (!ext) ext = ".png";

    if (strcasecmp(ext, ".png") == 0 || strcasecmp(ext, "png") == 0) {
        ok = stbi_write_png(path, img->width, img->height, img->channels,
                             tight, (int)row_bytes);
    } else if (strcasecmp(ext, ".jpg") == 0 || strcasecmp(ext, ".jpeg") == 0) {
        ok = stbi_write_jpg(path, img->width, img->height, img->channels, tight, 90);
    } else if (strcasecmp(ext, ".bmp") == 0) {
        ok = stbi_write_bmp(path, img->width, img->height, img->channels, tight);
    } else if (strcasecmp(ext, ".tga") == 0) {
        ok = stbi_write_tga(path, img->width, img->height, img->channels, tight);
    } else {
        ok = stbi_write_png(path, img->width, img->height, img->channels,
                             tight, (int)row_bytes);
    }

    free(tight);
    if (!ok) vision_set_error("vision_imwrite: stb write failed");
    return ok;
}

VisionImage* vision_imdecode(const uint8_t* buf, size_t len, int desired_channels)
{
    if (!buf || len == 0) VISION_ERR("vision_imdecode: null buffer");

    int w, h, ch;
    uint8_t* raw = stbi_load_from_memory(buf, (int)len, &w, &h, &ch,
                                          desired_channels);
    if (!raw) VISION_ERR("vision_imdecode: %s", stbi_failure_reason());

    int actual_ch = desired_channels > 0 ? desired_channels : ch;
    VisionImage* img = vision_image_create(w, h, actual_ch,
                                            VISION_FMT_UINT8, VISION_LAYOUT_HWC);
    if (!img) { stbi_image_free(raw); return NULL; }

    size_t src_row = (size_t)w * actual_ch;
    for (int r = 0; r < h; r++) {
        memcpy(img->data + r * img->stride, raw + r * src_row, src_row);
    }
    stbi_image_free(raw);
    return img;
}

/* In-memory encode callback state */
typedef struct { uint8_t* buf; size_t len; size_t cap; } _EncBuf;

static void _enc_write_func(void* ctx, void* data, int size)
{
    _EncBuf* b = (_EncBuf*)ctx;
    size_t need = b->len + (size_t)size;
    if (need > b->cap) {
        size_t new_cap = VISION_MAX(need, b->cap * 2);
        b->buf = (uint8_t*)realloc(b->buf, new_cap);
        b->cap = new_cap;
    }
    memcpy(b->buf + b->len, data, (size_t)size);
    b->len += (size_t)size;
}

uint8_t* vision_imencode(const VisionImage* img, const char* ext, size_t* out_len)
{
    if (!img || !out_len) { vision_set_error("vision_imencode: null arg"); return NULL; }
    if (img->format != VISION_FMT_UINT8) {
        vision_set_error("vision_imencode: only UINT8 supported"); return NULL;
    }

    size_t row_bytes = (size_t)img->width * img->channels;
    uint8_t* tight = (uint8_t*)malloc(row_bytes * img->height);
    if (!tight) { vision_set_error("vision_imencode: oom"); return NULL; }
    for (int r = 0; r < img->height; r++) {
        memcpy(tight + r * row_bytes, img->data + r * img->stride, row_bytes);
    }

    _EncBuf enc = { (uint8_t*)malloc(65536), 0, 65536 };
    if (!enc.buf) { free(tight); vision_set_error("vision_imencode: oom"); return NULL; }

    int ok = 0;
    if (!ext || strcasecmp(ext, "png") == 0 || strcasecmp(ext, ".png") == 0) {
        ok = stbi_write_png_to_func(_enc_write_func, &enc, img->width, img->height,
                                     img->channels, tight, (int)row_bytes);
    } else if (strcasecmp(ext, "jpg") == 0 || strcasecmp(ext, ".jpg") == 0
           || strcasecmp(ext, "jpeg") == 0) {
        ok = stbi_write_jpg_to_func(_enc_write_func, &enc, img->width, img->height,
                                     img->channels, tight, 90);
    } else if (strcasecmp(ext, "bmp") == 0 || strcasecmp(ext, ".bmp") == 0) {
        ok = stbi_write_bmp_to_func(_enc_write_func, &enc, img->width, img->height,
                                     img->channels, tight);
    } else if (strcasecmp(ext, "tga") == 0 || strcasecmp(ext, ".tga") == 0) {
        ok = stbi_write_tga_to_func(_enc_write_func, &enc, img->width, img->height,
                                     img->channels, tight);
    } else {
        ok = stbi_write_png_to_func(_enc_write_func, &enc, img->width, img->height,
                                     img->channels, tight, (int)row_bytes);
    }

    free(tight);
    if (!ok) {
        free(enc.buf);
        vision_set_error("vision_imencode: encode failed");
        return NULL;
    }
    *out_len = enc.len;
    return enc.buf;
}

void vision_imencode_free(uint8_t* buf) { free(buf); }

/* ═══════════════════════════════════════════════════════════════════════
 * 5. FORMAT CONVERSION
 * ═══════════════════════════════════════════════════════════════════════ */

VisionImage* vision_to_float32(const VisionImage* src, float scale)
{
    if (!src) VISION_ERR("vision_to_float32: null src");
    if (src->format == VISION_FMT_FLOAT32) return vision_image_clone(src);

    VisionImage* dst = vision_image_create(src->width, src->height, src->channels,
                                            VISION_FMT_FLOAT32, src->layout);
    if (!dst) return NULL;

    int n_total = src->width * src->height * src->channels;

    if (src->format == VISION_FMT_UINT8) {
        const uint8_t* in  = src->data;
        float*         out = (float*)dst->data;

        /* AVX2: convert 8 uint8→float at once, in a packed HWC image */
        int i = 0;
#ifdef __AVX2__
        if (_vision_has_avx2()) {
            __m256 vscale = _mm256_set1_ps(scale);
            __m128i vzero  = _mm_setzero_si128();
            for (; i <= n_total - 8; i += 8) {
                __m128i b8  = _mm_loadl_epi64((const __m128i*)(in + i));
                __m128i b16 = _mm_unpacklo_epi8(b8, vzero);
                __m128i b32_lo = _mm_unpacklo_epi16(b16, vzero);
                __m128i b32_hi = _mm_unpackhi_epi16(b16, vzero);
                __m256i b32 = _mm256_set_m128i(b32_hi, b32_lo);
                __m256  f   = _mm256_cvtepi32_ps(b32);
                _mm256_storeu_ps(out + i, _mm256_mul_ps(f, vscale));
            }
        }
#endif
        for (; i < n_total; i++) out[i] = in[i] * scale;
    } else if (src->format == VISION_FMT_INT8) {
        const int8_t* in  = (const int8_t*)src->data;
        float*        out = (float*)dst->data;
        for (int i = 0; i < n_total; i++) out[i] = in[i] * scale;
    } else if (src->format == VISION_FMT_FLOAT16) {
        /* naive fp16→fp32 */
        const uint16_t* in  = (const uint16_t*)src->data;
        float*          out = (float*)dst->data;
        for (int i = 0; i < n_total; i++) {
            uint32_t h = in[i];
            uint32_t s = (h & 0x8000u) << 16;
            uint32_t e = (h & 0x7C00u);
            uint32_t m = (h & 0x03FFu);
            uint32_t v;
            if (e == 0)      v = s | (m << 13);           /* denormal/zero */
            else if (e==0x7C00u) v = s | 0x7F800000u | (m<<13); /* inf/nan */
            else v = s | ((e + 0x1C000u) << 13) | (m << 13);
            float f; memcpy(&f, &v, 4);
            out[i] = f * scale;
        }
    } else {
        vision_image_free(dst);
        VISION_ERR("vision_to_float32: unsupported source format %d", src->format);
    }

    dst->color_space = src->color_space;
    return dst;
}

VisionImage* vision_to_uint8(const VisionImage* src, float scale)
{
    if (!src) VISION_ERR("vision_to_uint8: null src");
    if (src->format == VISION_FMT_UINT8) return vision_image_clone(src);

    VisionImage* dst = vision_image_create(src->width, src->height, src->channels,
                                            VISION_FMT_UINT8, src->layout);
    if (!dst) return NULL;

    int n = src->width * src->height * src->channels;
    if (src->format == VISION_FMT_FLOAT32) {
        const float* in  = (const float*)src->data;
        uint8_t*     out = dst->data;
        for (int i = 0; i < n; i++) {
            float v = in[i] * scale;
            out[i]  = (uint8_t)VISION_CLAMP((int)(v + 0.5f), 0, 255);
        }
    } else {
        vision_image_free(dst);
        VISION_ERR("vision_to_uint8: unsupported source format %d", src->format);
    }

    dst->color_space = src->color_space;
    return dst;
}

VisionImage* vision_to_int8(const VisionImage* src, float scale, float zero_point)
{
    if (!src) VISION_ERR("vision_to_int8: null src");

    VisionImage* dst = vision_image_create(src->width, src->height, src->channels,
                                            VISION_FMT_INT8, src->layout);
    if (!dst) return NULL;

    int n = src->width * src->height * src->channels;
    if (src->format == VISION_FMT_FLOAT32) {
        const float* in  = (const float*)src->data;
        int8_t*      out = (int8_t*)dst->data;
        for (int i = 0; i < n; i++) {
            float v = in[i] * scale + zero_point;
            out[i]  = (int8_t)VISION_CLAMP((int)(v + 0.5f), -128, 127);
        }
    } else if (src->format == VISION_FMT_UINT8) {
        const uint8_t* in  = src->data;
        int8_t*        out = (int8_t*)dst->data;
        for (int i = 0; i < n; i++) {
            float v = in[i] * scale + zero_point;
            out[i]  = (int8_t)VISION_CLAMP((int)(v + 0.5f), -128, 127);
        }
    } else {
        vision_image_free(dst);
        VISION_ERR("vision_to_int8: unsupported source format");
    }

    dst->color_space = src->color_space;
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 6. LAYOUT CONVERSION: HWC ↔ CHW
 * ═══════════════════════════════════════════════════════════════════════ */

VisionImage* vision_hwc_to_chw(const VisionImage* src)
{
    if (!src) VISION_ERR("vision_hwc_to_chw: null src");
    if (src->layout == VISION_LAYOUT_CHW) return vision_image_clone(src);

    int H = src->height, W = src->width, C = src->channels;
    size_t esz = vision_element_size(src->format);

    VisionImage* dst = vision_image_create(W, H, C, src->format, VISION_LAYOUT_CHW);
    if (!dst) return NULL;

    /* CHW stride = aligned(H*W*esz) per channel plane */
    size_t plane_stride = dst->stride;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int c = 0; c < C; c++) {
        for (int r = 0; r < H; r++) {
            const uint8_t* src_row = src->data + (size_t)r * src->stride + c * esz;
            uint8_t*       dst_col = dst->data + (size_t)c * plane_stride
                                   + (size_t)r * W * esz;
            /* copy W elements, each strided by C*esz in src */
            for (int col = 0; col < W; col++) {
                memcpy(dst_col + col * esz, src_row + (size_t)col * C * esz, esz);
            }
        }
    }

    dst->color_space = src->color_space;
    return dst;
}

VisionImage* vision_chw_to_hwc(const VisionImage* src)
{
    if (!src) VISION_ERR("vision_chw_to_hwc: null src");
    if (src->layout == VISION_LAYOUT_HWC) return vision_image_clone(src);

    int H = src->height, W = src->width, C = src->channels;
    size_t esz = vision_element_size(src->format);
    size_t plane_stride = src->stride;

    VisionImage* dst = vision_image_create(W, H, C, src->format, VISION_LAYOUT_HWC);
    if (!dst) return NULL;

    #pragma omp parallel for schedule(static) collapse(2)
    for (int r = 0; r < H; r++) {
        for (int c = 0; c < C; c++) {
            const uint8_t* src_col = src->data + (size_t)c * plane_stride
                                   + (size_t)r * W * esz;
            uint8_t*       dst_row = dst->data + (size_t)r * dst->stride + c * esz;
            for (int col = 0; col < W; col++) {
                memcpy(dst_row + (size_t)col * C * esz, src_col + col * esz, esz);
            }
        }
    }

    dst->color_space = src->color_space;
    return dst;
}

/* ═══════════════════════════════════════════════════════════════════════
 * 7. TENSOR BRIDGE
 * ═══════════════════════════════════════════════════════════════════════ */

/*
 * Returns a raw float32 CHW buffer (void*) the same size as the existing
 * Pml Tensor would use for [C, H, W].  The CALLER owns this memory and
 * must free() it.  We do NOT return a Tensor* struct here to keep the
 * vision library free of tensor.h dependencies.
 */
void* vision_image_to_tensor(const VisionImage* img)
{
    if (!img) VISION_ERR("vision_image_to_tensor: null img");

    /* Ensure float32 CHW */
    VisionImage* f32 = NULL;
    VisionImage* chw = NULL;
    const VisionImage* src = img;

    if (img->format != VISION_FMT_FLOAT32) {
        f32 = vision_to_float32(img, 1.0f / 255.0f);
        if (!f32) return NULL;
        src = f32;
    }
    if (src->layout != VISION_LAYOUT_CHW) {
        chw = vision_hwc_to_chw(src);
        if (f32) vision_image_free(f32);
        if (!chw) return NULL;
        src = chw;
    }

    /* Allocate tight CHW buffer (no padding between planes) */
    size_t sz = (size_t)src->channels * src->height * src->width * sizeof(float);
    float* buf = (float*)vision_alloc(sz);
    if (!buf) {
        if (f32) vision_image_free(f32);
        if (chw) vision_image_free(chw);
        VISION_ERR("vision_image_to_tensor: alloc failed");
    }

    /* Copy plane by plane, stripping any alignment padding */
    int H = src->height, W = src->width, C = src->channels;
    size_t plane_elems = (size_t)H * W;
    for (int c = 0; c < C; c++) {
        const float* src_plane = (const float*)(src->data) + c * (src->stride / sizeof(float));
        float* dst_plane = buf + c * plane_elems;
        for (int r = 0; r < H; r++) {
            memcpy(dst_plane + r * W,
                   src_plane + r * (src->stride / sizeof(float) / C) ,
                   W * sizeof(float));
        }
    }

    if (f32) vision_image_free(f32);
    if (chw) vision_image_free(chw);
    return buf;
}

VisionImage* vision_tensor_to_image(void* tensor_data, int width, int height,
                                     int channels, int format)
{
    if (!tensor_data || width <= 0 || height <= 0 || channels <= 0)
        VISION_ERR("vision_tensor_to_image: invalid args");

    /* Wrap without taking ownership — caller still owns tensor_data */
    return vision_image_create_from_data(tensor_data, width, height,
                                          channels, format, VISION_LAYOUT_CHW, 0);
}
