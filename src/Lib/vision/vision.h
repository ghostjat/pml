/**
 * PML Vision Engine — Master C Header
 * ════════════════════════════════════════════════════════════════════════
 * Single-header C API for the Pml\Vision module.
 *
 *  Design principles
 *  ─────────────────
 *  • Every function that returns a pointer allocates a new VisionImage
 *    (or associated struct) via vision_image_create(). The caller
 *    ALWAYS owns the result and MUST call vision_image_free().
 *  • "view" variants share the underlying buffer (owns_data = 0).
 *    Freeing a view is safe (it never frees the data).
 *  • Error state is thread-local per translation unit (mirrors tensor.c).
 *  • Structs use int for boolean flags to guarantee FFI alignment.
 *  • size_t fields are placed after int fields to minimise padding.
 *  • All image buffers are 64-byte aligned (AVX-512 ready).
 *  • NO C++, NO external runtime, stb_image only for I/O.
 * ════════════════════════════════════════════════════════════════════════
 */

#ifndef VISION_H
#define VISION_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ═══════════════════════════════════════════════════════════════════════
 * 0.  VERSION
 * ═══════════════════════════════════════════════════════════════════════ */
#define PML_VISION_VERSION_MAJOR 1
#define PML_VISION_VERSION_MINOR 0
#define PML_VISION_VERSION_PATCH 0

/* ═══════════════════════════════════════════════════════════════════════
 * 1.  ENUMERATIONS
 * ═══════════════════════════════════════════════════════════════════════ */

/* Pixel element type stored in VisionImage.data */
typedef enum {
    VISION_FMT_UINT8   = 0,   /* uint8_t per channel, range 0–255          */
    VISION_FMT_FLOAT32 = 1,   /* float per channel, typically 0.0–1.0      */
    VISION_FMT_INT8    = 2,   /* int8_t per channel, range −128–127        */
    VISION_FMT_FLOAT16 = 3,   /* half-float as uint16_t, 0.0–1.0 encoded  */
} VisionPixelFormat;

/* Memory layout of the pixel buffer */
typedef enum {
    VISION_LAYOUT_HWC = 0,    /* [H][W][C] — stb_image default, OpenCV    */
    VISION_LAYOUT_CHW = 1,    /* [C][H][W] — PyTorch / ML standard        */
} VisionLayout;

/* Resize interpolation algorithm */
typedef enum {
    VISION_INTERP_NEAREST  = 0,
    VISION_INTERP_BILINEAR = 1,
    VISION_INTERP_BICUBIC  = 2,
    VISION_INTERP_AREA     = 3,   /* average-pool downscale                */
} VisionInterp;

/* Border padding mode */
typedef enum {
    VISION_BORDER_CONSTANT  = 0,
    VISION_BORDER_REFLECT   = 1,
    VISION_BORDER_REPLICATE = 2,
    VISION_BORDER_WRAP      = 3,
} VisionBorderMode;

/* Logical colour space tag (informational, not enforced) */
typedef enum {
    VISION_COLOR_RGB  = 0,
    VISION_COLOR_BGR  = 1,
    VISION_COLOR_GRAY = 2,
    VISION_COLOR_RGBA = 3,
    VISION_COLOR_HSV  = 4,
    VISION_COLOR_LAB  = 5,
} VisionColorSpace;

/* Short aliases used in .c implementations */
#define VISION_CS_RGB  VISION_COLOR_RGB
#define VISION_CS_BGR  VISION_COLOR_BGR
#define VISION_CS_GRAY VISION_COLOR_GRAY
#define VISION_CS_RGBA VISION_COLOR_RGBA
#define VISION_CS_HSV  VISION_COLOR_HSV
#define VISION_CS_LAB  VISION_COLOR_LAB

/* ═══════════════════════════════════════════════════════════════════════
 * 2.  CORE DATA STRUCTURES
 * ═══════════════════════════════════════════════════════════════════════ */

/**
 * VisionImage — central image container.
 *
 * Memory contract
 * ───────────────
 * • data is ALWAYS 64-byte aligned.
 * • stride >= width * channels * sizeof(element) (may be padded).
 * • owns_data == 1  → vision_image_free() releases data.
 * • owns_data == 0  → vision_image_free() releases the struct only.
 * • layout == HWC   → stride is bytes per row.
 * • layout == CHW   → stride is bytes per channel plane (height * width * elem).
 *
 * struct layout chosen to minimise padding on LP64:
 *   offset  0 : data        (8)
 *   offset  8 : stride      (8)
 *   offset 16 : data_size   (8)
 *   offset 24 : width       (4)
 *   offset 28 : height      (4)
 *   offset 32 : channels    (4)
 *   offset 36 : format      (4)
 *   offset 40 : layout      (4)
 *   offset 44 : color_space (4)
 *   offset 48 : owns_data   (4)  ← int, not bool, for FFI clarity
 *   offset 52 : _pad        (4)
 *   sizeof = 56
 */
typedef struct VisionImage {
    uint8_t* data;
    size_t   stride;
    size_t   data_size;
    int      width;
    int      height;
    int      channels;
    int      format;        /* VisionPixelFormat cast to int */
    int      layout;        /* VisionLayout cast to int      */
    int      color_space;   /* VisionColorSpace cast to int  */
    int      owns_data;     /* 1 = owns, 0 = view            */
    int      _pad;
} VisionImage;

/* Axis-aligned bounding box with detection score */
typedef struct {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
} VisionBBox;

/* Heap-allocated growable bounding-box array */
typedef struct {
    VisionBBox* boxes;
    int         count;
    int         capacity;
} VisionBBoxArray;

/* 2-D convolution kernel weights (row-major, kh×kw) */
typedef struct {
    float* data;
    int    kh;
    int    kw;
} VisionKernel;

/* HOG descriptor result */
typedef struct {
    float* descriptors;    /* [n_cells_y * n_cells_x * blocks_per_cell * n_bins] */
    int    n_features;
    int    n_cells_y;
    int    n_cells_x;
} HOGResult;

/* LBP histogram result */
typedef struct {
    float* descriptors;    /* [grid_x * grid_y * 256] */
    int    n_features;
} LBPResult;

/* Sparse set of keypoints (Harris / FAST) */
typedef struct {
    float* x;
    float* y;
    float* score;
    int    count;
} VisionCorners;

/* Connected-components result */
typedef struct {
    uint16_t* labels;       /* [height * width] — 0 = background */
    int*      areas;        /* pixel count per component         */
    int*      bbox_x1;
    int*      bbox_y1;
    int*      bbox_x2;
    int*      bbox_y2;
    int       n_components;
    int       width;
    int       height;
} VisionCC;

/* Runtime CPU feature flags */
typedef struct {
    int has_sse42;
    int has_avx;
    int has_avx2;
    int has_avx512f;
    int has_avx512bw;
    int has_fma;
} VisionCPUFeatures;

/* xoshiro128+ RNG — fast, non-cryptographic */
typedef struct {
    uint32_t state[4];
} VisionRNG;

/* Global memory diagnostics */
typedef struct {
    int64_t images_allocated;
    int64_t images_freed;
    int64_t bytes_allocated;
    int64_t bytes_freed;
    int64_t peak_bytes;
} VisionMemStats;

/* Opaque handle — video architecture stub (no implementation yet) */
typedef struct VisionVideoCapture VisionVideoCapture;

/* ═══════════════════════════════════════════════════════════════════════
 * 3.  ERROR HANDLING  (thread-local, mirrors tensor.c pattern)
 * ═══════════════════════════════════════════════════════════════════════ */
int         vision_check_error(void);
const char* vision_get_last_error(void);
void        vision_clear_error(void);
void        vision_set_error(const char* msg);

/* ═══════════════════════════════════════════════════════════════════════
 * 4.  CPU FEATURE DETECTION
 * ═══════════════════════════════════════════════════════════════════════ */
const VisionCPUFeatures* vision_cpu_features(void);

/* ═══════════════════════════════════════════════════════════════════════
 * 5.  IMAGE LIFECYCLE
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_image_create(int width, int height, int channels,
                                  int format, int layout, int color_space);
VisionImage* vision_image_create_from_data(void* data, int width, int height,
                                            int channels, int format, int layout,
                                            int take_ownership);
VisionImage* vision_image_clone(const VisionImage* src);
VisionImage* vision_image_view(VisionImage* src);
void         vision_image_free(VisionImage* img);

/* Field accessors — PHP never touches struct fields directly */
int    vision_image_width(const VisionImage* img);
int    vision_image_height(const VisionImage* img);
int    vision_image_channels(const VisionImage* img);
int    vision_image_format(const VisionImage* img);
int    vision_image_layout(const VisionImage* img);
int    vision_image_color_space(const VisionImage* img);
size_t vision_image_stride(const VisionImage* img);
size_t vision_image_data_size(const VisionImage* img);
void*  vision_image_data_ptr(const VisionImage* img);

/* ═══════════════════════════════════════════════════════════════════════
 * 6.  IMAGE I/O  (stb_image backend)
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_imread(const char* path, int desired_channels);
int          vision_imwrite(const char* path, const VisionImage* img);
VisionImage* vision_imdecode(const uint8_t* buf, size_t len, int desired_channels);
uint8_t*     vision_imencode(const VisionImage* img, const char* ext, size_t* out_len);
void         vision_imencode_free(uint8_t* buf);

/* ═══════════════════════════════════════════════════════════════════════
 * 7.  FORMAT & LAYOUT CONVERSION
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_to_float32(const VisionImage* src, float scale);
VisionImage* vision_to_uint8(const VisionImage* src, float scale);
VisionImage* vision_to_int8(const VisionImage* src, float scale, float zero_point);
VisionImage* vision_hwc_to_chw(const VisionImage* src);
VisionImage* vision_chw_to_hwc(const VisionImage* src);

/* ═══════════════════════════════════════════════════════════════════════
 * 8.  RESIZE & SPATIAL TRANSFORMS
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_resize(const VisionImage* src, int dst_w, int dst_h, VisionInterp interp);
VisionImage* vision_resize_long_edge(const VisionImage* src, int long_edge, VisionInterp interp);
VisionImage* vision_center_crop(const VisionImage* src, int crop_w, int crop_h);
VisionImage* vision_crop(const VisionImage* src, int x, int y, int w, int h);
VisionImage* vision_pad(const VisionImage* src, int top, int bottom,
                         int left, int right, VisionBorderMode border, float fill_value);
VisionImage* vision_flip_horizontal(const VisionImage* src);
VisionImage* vision_flip_vertical(const VisionImage* src);
VisionImage* vision_rotate90(const VisionImage* src, int times);
VisionImage* vision_rotate(const VisionImage* src, float angle_deg,
                            VisionInterp interp, VisionBorderMode border, float fill_value);
VisionImage* vision_affine(const VisionImage* src, const float* M6,
                            int dst_w, int dst_h, VisionInterp interp,
                            VisionBorderMode border, float fill_value);
VisionImage* vision_perspective(const VisionImage* src,
                                 const float* src_pts8, const float* dst_pts8,
                                 int dst_w, int dst_h, VisionInterp interp);

/* ═══════════════════════════════════════════════════════════════════════
 * 9.  COLOR OPERATIONS
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_to_grayscale(const VisionImage* src);
VisionImage* vision_rgb_to_bgr(const VisionImage* src);
VisionImage* vision_rgb_to_hsv(const VisionImage* src);
VisionImage* vision_hsv_to_rgb(const VisionImage* src);
VisionImage* vision_normalize(const VisionImage* src,
                               const float* mean, const float* std_dev);
VisionImage* vision_denormalize(const VisionImage* src,
                                 const float* mean, const float* std_dev);
VisionImage* vision_adjust_brightness(const VisionImage* src, float delta);
VisionImage* vision_adjust_contrast(const VisionImage* src, float factor);
VisionImage* vision_adjust_gamma(const VisionImage* src, float gamma);
VisionImage* vision_adjust_hue(const VisionImage* src, float delta_hue);
VisionImage* vision_bgr_to_rgb(const VisionImage* src);
VisionImage* vision_histogram_equalize(const VisionImage* src);

/* ═══════════════════════════════════════════════════════════════════════
 * 10. FILTERING
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_gaussian_blur(const VisionImage* src, int radius, float sigma);
VisionImage* vision_box_blur(const VisionImage* src, int radius);
VisionImage* vision_median_blur(const VisionImage* src, int radius);
VisionImage* vision_sobel(const VisionImage* src,
                           VisionImage** out_gx, VisionImage** out_gy);
VisionImage* vision_laplacian(const VisionImage* src);
VisionImage* vision_canny(const VisionImage* src, float low_thresh, float high_thresh,
                           int gaussian_radius, float gaussian_sigma);
VisionImage* vision_convolve2d(const VisionImage* src, const VisionKernel* kernel,
                                VisionBorderMode border);

/* ═══════════════════════════════════════════════════════════════════════
 * 11. MORPHOLOGY
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_erode(const VisionImage* src, int radius);
VisionImage* vision_dilate(const VisionImage* src, int radius);
VisionImage* vision_morph_open(const VisionImage* src, int radius);
VisionImage* vision_morph_close(const VisionImage* src, int radius);
VisionImage* vision_morph_gradient(const VisionImage* src, int radius);

/* ═══════════════════════════════════════════════════════════════════════
 * 12. FEATURE EXTRACTION
 * ═══════════════════════════════════════════════════════════════════════ */
HOGResult*    vision_hog(const VisionImage* src, int cell_size, int block_size,
                          int nbins, int* out_len);
void          vision_hog_free(HOGResult* r);
LBPResult*    vision_lbp(const VisionImage* src, int radius,
                          int grid_x, int grid_y, int* out_len);
void          vision_lbp_free(LBPResult* r);
void          vision_integral_image(const VisionImage* src, double* integral);
VisionCorners* vision_harris_corners(const VisionImage* src,
                                      float k, float threshold,
                                      int nms_radius, int* out_count);
VisionCorners* vision_fast_corners(const VisionImage* src, int threshold,
                                    int n_consecutive, int* out_count);
void           vision_corners_free(VisionCorners* c);

/* ═══════════════════════════════════════════════════════════════════════
 * 13. DATA AUGMENTATION
 * ═══════════════════════════════════════════════════════════════════════ */
void         vision_rng_init(VisionRNG* rng, uint64_t seed);
uint32_t     vision_rng_next(VisionRNG* rng);
VisionImage* vision_random_crop(const VisionImage* src, int crop_w, int crop_h,
                                 VisionRNG* rng);
VisionImage* vision_random_resize_crop(const VisionImage* src,
                                        int out_w, int out_h,
                                        float scale_lo, float scale_hi,
                                        float ratio_lo, float ratio_hi,
                                        VisionRNG* rng, VisionInterp interp);
VisionImage* vision_random_flip_horizontal(const VisionImage* src, float prob,
                                            VisionRNG* rng);
VisionImage* vision_random_flip_vertical(const VisionImage* src, float prob,
                                          VisionRNG* rng);
VisionImage* vision_random_brightness(const VisionImage* src, float max_delta,
                                       VisionRNG* rng);
VisionImage* vision_random_contrast(const VisionImage* src, float lo, float hi,
                                     VisionRNG* rng);
VisionImage* vision_random_hue(const VisionImage* src, float max_delta,
                                VisionRNG* rng);
VisionImage* vision_cutout(const VisionImage* src, int n_holes, int hole_size,
                            float fill_value, VisionRNG* rng);
VisionImage* vision_mixup(const VisionImage* a, const VisionImage* b,
                           float alpha, VisionRNG* rng, float* out_lambda);
VisionImage* vision_cutmix(const VisionImage* a, const VisionImage* b,
                            float alpha, VisionRNG* rng, float* out_lambda);
VisionImage* vision_random_rotation(const VisionImage* src, float max_angle_deg,
                                     VisionRNG* rng, VisionInterp interp,
                                     VisionBorderMode border, float fill_value);

/* ═══════════════════════════════════════════════════════════════════════
 * 14. DETECTION UTILITIES
 * ═══════════════════════════════════════════════════════════════════════ */
float vision_iou(const VisionBBox* a, const VisionBBox* b);
float vision_giou(const VisionBBox* a, const VisionBBox* b);
float vision_diou(const VisionBBox* a, const VisionBBox* b);
VisionBBoxArray* vision_nms(const VisionBBoxArray* boxes, float iou_thresh);
VisionBBoxArray* vision_soft_nms(const VisionBBoxArray* boxes,
                                  float sigma, float score_thresh);

VisionBBoxArray* vision_bbox_array_create(int capacity);
int              vision_bbox_array_push(VisionBBoxArray* arr, const VisionBBox* box);
void             vision_bbox_array_free(VisionBBoxArray* arr);

VisionBBoxArray* vision_generate_anchors(int feat_w, int feat_h,
                                          int stride,
                                          const float* scales, int n_scales,
                                          const float* ratios, int n_ratios);

void vision_bbox_encode(const VisionBBox* anchor, const VisionBBox* gt,
                         float* dx, float* dy, float* dw, float* dh);
void vision_bbox_decode(const VisionBBox* anchor,
                         float dx, float dy, float dw, float dh,
                         VisionBBox* out);

/* ═══════════════════════════════════════════════════════════════════════
 * 15. SEGMENTATION UTILITIES
 * ═══════════════════════════════════════════════════════════════════════ */
VisionImage* vision_mask_resize(const VisionImage* mask, int dst_w, int dst_h);
VisionImage* vision_polygon_rasterize(const float* pts_xy, int n_pts,
                                       int img_w, int img_h, uint8_t fill_val);
VisionCC*    vision_connected_components(const VisionImage* binary_mask);
void         vision_cc_free(VisionCC* cc);

/* ═══════════════════════════════════════════════════════════════════════
 * 16. MODEL DECODE & MASK ASSEMBLY
 *     Used by SSDLite, NanoDet, PicoDet, YOLO11n, FastSAM PHP classes.
 * ═══════════════════════════════════════════════════════════════════════ */

/* SSDLite prior box generator */
VisionBBoxArray* vision_ssd_prior_boxes(const int*   feat_sizes,  int n_scales,
                                         const float* min_sizes,
                                         const float* max_sizes,
                                         const float* ratios,     int n_ratios,
                                         int img_size);

/* SSD prediction decode (loc deltas + class logits → filtered boxes) */
VisionBBoxArray* vision_ssd_decode(const float* loc_pred, const float* cls_pred,
                                    const VisionBBoxArray* anchors,
                                    int n_cls,    float conf_thr,
                                    float var_xy, float var_wh);

/* NanoDet FCOS + GFL distribution decode */
VisionBBoxArray* vision_nanodet_decode(const float* cls_pred, const float* reg_pred,
                                        int feat_h, int feat_w, int stride,
                                        int n_cls, int reg_max,
                                        int img_w, int img_h, float conf_thr);

/* PicoDet DFL + aligned head decode (same math as NanoDet) */
VisionBBoxArray* vision_picodet_decode(const float* cls_pred, const float* reg_pred,
                                        int feat_h, int feat_w, int stride,
                                        int n_cls, int reg_max,
                                        int img_w, int img_h, float conf_thr);

/* YOLO11n DFL + anchor-free decode (single scale output tensor) */
VisionBBoxArray* vision_yolo11_decode(const float* output,
                                       int feat_h, int feat_w, int stride,
                                       int n_cls, int reg_max,
                                       int img_w, int img_h, float conf_thr);

/* FastSAM prototype-bank mask assembly → CHW uint8 image (n_dets channels) */
VisionImage* vision_fastsam_assemble_masks(const float* proto, int n_proto,
                                            int proto_h, int proto_w,
                                            const float* coeffs, int n_dets,
                                            const VisionBBoxArray* boxes,
                                            int out_w, int out_h, float mask_thr);

/* Multi-scale decode (NanoDet/YOLO11 FPN) + NMS */
VisionBBoxArray* vision_multiscale_decode(const float** cls_preds,
                                           const float** reg_preds,
                                           const int*    feat_hs,
                                           const int*    feat_ws,
                                           const int*    strides,
                                           int n_scales,
                                           int n_cls, int reg_max,
                                           int img_w, int img_h,
                                           float conf_thr, float iou_thr,
                                           int decode_fn);

/* ═══════════════════════════════════════════════════════════════════════
 * 17. TENSOR BRIDGE  (round-trips to/from existing Pml Tensor)
 *
 * The Tensor layout expected:  CHW float32, owned.
 * Caller is responsible for casting void* to/from their Tensor type.
 * ═══════════════════════════════════════════════════════════════════════ */
void*        vision_image_to_tensor(const VisionImage* img);
VisionImage* vision_tensor_to_image(void* tensor_data, int width, int height,
                                     int channels, int format);
void         vision_free_raw(void* ptr);   /* free() a vision_image_to_tensor buffer */

/* ═══════════════════════════════════════════════════════════════════════
 * 18. VIDEO FOUNDATION (architecture stub — no implementation)
 *
 * The opaque VisionVideoCapture type is declared here so future
 * translation units can extend it without ABI breaks.
 * ═══════════════════════════════════════════════════════════════════════ */
/* VisionVideoCapture* vision_video_open(const char* path);        */
/* VisionImage*        vision_video_read_frame(VisionVideoCapture*);*/
/* void                vision_video_close(VisionVideoCapture*);     */
/* int                 vision_video_width(VisionVideoCapture*);     */
/* int                 vision_video_height(VisionVideoCapture*);    */
/* double              vision_video_fps(VisionVideoCapture*);       */

/* ═══════════════════════════════════════════════════════════════════════
 * 19. MEMORY DIAGNOSTICS
 * ═══════════════════════════════════════════════════════════════════════ */
const VisionMemStats* vision_mem_stats(void);
void                  vision_mem_stats_reset(void);

/* ═══════════════════════════════════════════════════════════════════════
 * 20. INTERNAL HELPERS  (used across .c files, not exposed to PHP)
 * ═══════════════════════════════════════════════════════════════════════ */
#ifdef VISION_INTERNAL
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>

#ifndef VISION_MIN
#  define VISION_MIN(a,b) ((a)<(b)?(a):(b))
#endif
#ifndef VISION_MAX
#  define VISION_MAX(a,b) ((a)>(b)?(a):(b))
#endif
#ifndef VISION_CLAMP
#  define VISION_CLAMP(x,lo,hi) VISION_MIN(VISION_MAX(x,lo),hi)
#endif

#define VISION_ALIGN 64    /* AVX-512 alignment requirement */

/* Aligned allocation that tracks in mem_stats */
void* vision_alloc(size_t bytes);
void  vision_dealloc(void* ptr, size_t bytes);

/* Error macros — return NULL on failure */
#define VISION_ERR(fmt, ...)  do { \
    char _buf[512]; \
    snprintf(_buf, sizeof(_buf), fmt, ##__VA_ARGS__); \
    vision_set_error(_buf); \
    return NULL; \
} while(0)

#define VISION_ERR_VOID(fmt, ...) do { \
    char _buf[512]; \
    snprintf(_buf, sizeof(_buf), fmt, ##__VA_ARGS__); \
    vision_set_error(_buf); \
    return; \
} while(0)

#define VISION_ERR_INT(ret, fmt, ...) do { \
    char _buf[512]; \
    snprintf(_buf, sizeof(_buf), fmt, ##__VA_ARGS__); \
    vision_set_error(_buf); \
    return ret; \
} while(0)

/* Runtime SIMD dispatch */
static inline int _vision_has_avx2(void) {
    const VisionCPUFeatures* f = vision_cpu_features();
    return f->has_avx2;
}
static inline int _vision_has_avx512f(void) {
    const VisionCPUFeatures* f = vision_cpu_features();
    return f->has_avx512f;
}

/* Element size from format */
static inline size_t vision_element_size(int fmt) {
    switch (fmt) {
        case VISION_FMT_UINT8:   return 1;
        case VISION_FMT_FLOAT32: return 4;
        case VISION_FMT_INT8:    return 1;
        case VISION_FMT_FLOAT16: return 2;
        default:                 return 1;
    }
}

/* Bytes-per-row for HWC layout */
static inline size_t vision_row_stride(int width, int channels, int fmt) {
    size_t raw = (size_t)width * channels * vision_element_size(fmt);
    /* round up to VISION_ALIGN */
    return (raw + VISION_ALIGN - 1) & ~(size_t)(VISION_ALIGN - 1);
}

#endif /* VISION_INTERNAL */

#ifdef __cplusplus
}
#endif

#endif /* VISION_H */
