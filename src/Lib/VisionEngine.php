<?php

declare(strict_types=1);

namespace Pml\Lib;

use FFI;

/**
 * VisionEngine — singleton FFI bridge to libvision.so.
 *
 * PHP orchestration only. No image math in PHP.
 */
final class VisionEngine
{
    private static ?self $instance = null;
    private FFI $ffi;

    private const LIB_DIR  = __DIR__ . '/vision';
    private const LIB_SO   = __DIR__ . '/vision/libvision.so';
    private const BUILD_SH  = __DIR__ . '/vision/build.sh';

    /* FFI cdef — mirrors vision.h public API (enum values inlined as ints for FFI compat) */
    private const CDEF = <<<'CDEF'
/* enums as int constants */
static const int VISION_FMT_UINT8   = 0;
static const int VISION_FMT_FLOAT32 = 1;
static const int VISION_FMT_INT8    = 2;
static const int VISION_LAYOUT_HWC  = 0;
static const int VISION_LAYOUT_CHW  = 1;
static const int VISION_INTERP_NEAREST  = 0;
static const int VISION_INTERP_BILINEAR = 1;
static const int VISION_INTERP_BICUBIC  = 2;
static const int VISION_INTERP_AREA     = 3;
static const int VISION_BORDER_CONSTANT  = 0;
static const int VISION_BORDER_REFLECT   = 1;
static const int VISION_BORDER_REPLICATE = 2;
static const int VISION_BORDER_WRAP      = 3;
static const int VISION_COLOR_RGB  = 0;
static const int VISION_COLOR_BGR  = 1;
static const int VISION_COLOR_GRAY = 2;
static const int VISION_COLOR_RGBA = 3;
static const int VISION_COLOR_HSV  = 4;
static const int VISION_COLOR_LAB  = 5;

typedef struct VisionImage {
    uint8_t* data;
    size_t   stride;
    size_t   data_size;
    int      width;
    int      height;
    int      channels;
    int      format;
    int      layout;
    int      color_space;
    int      owns_data;
    int      _pad;
} VisionImage;

typedef struct {
    float x1, y1, x2, y2;
    float score;
    int   class_id;
} VisionBBox;

typedef struct {
    VisionBBox* boxes;
    int         count;
    int         capacity;
} VisionBBoxArray;

typedef struct {
    float* data;
    int    kh;
    int    kw;
} VisionKernel;

typedef struct {
    float* descriptors;
    int    n_features;
    int    n_cells_y;
    int    n_cells_x;
} HOGResult;

typedef struct {
    float* descriptors;
    int    n_features;
} LBPResult;

typedef struct {
    float* x;
    float* y;
    float* score;
    int    count;
} VisionCorners;

typedef struct {
    uint16_t* labels;
    int*      areas;
    int*      bbox_x1;
    int*      bbox_y1;
    int*      bbox_x2;
    int*      bbox_y2;
    int       n_components;
    int       width;
    int       height;
} VisionCC;

typedef struct {
    int has_sse42;
    int has_avx;
    int has_avx2;
    int has_avx512f;
    int has_avx512bw;
    int has_fma;
} VisionCPUFeatures;

typedef struct { uint32_t state[4]; } VisionRNG;

typedef struct {
    int64_t images_allocated;
    int64_t images_freed;
    int64_t bytes_allocated;
    int64_t bytes_freed;
    int64_t peak_bytes;
} VisionMemStats;

/* Error handling */
int         vision_check_error(void);
const char* vision_get_last_error(void);
void        vision_clear_error(void);
void        vision_set_error(const char* msg);

/* CPU features */
const VisionCPUFeatures* vision_cpu_features(void);

/* Image lifecycle */
VisionImage* vision_image_create(int width, int height, int channels,
                                  int format, int layout, int color_space);
VisionImage* vision_image_clone(const VisionImage* src);
void         vision_image_free(VisionImage* img);
int    vision_image_width(const VisionImage* img);
int    vision_image_height(const VisionImage* img);
int    vision_image_channels(const VisionImage* img);
int    vision_image_format(const VisionImage* img);
int    vision_image_layout(const VisionImage* img);
int    vision_image_color_space(const VisionImage* img);
size_t vision_image_stride(const VisionImage* img);
size_t vision_image_data_size(const VisionImage* img);
void*  vision_image_data_ptr(const VisionImage* img);

/* I/O */
VisionImage* vision_imread(const char* path, int desired_channels);
int          vision_imwrite(const char* path, const VisionImage* img);
VisionImage* vision_imdecode(const uint8_t* buf, size_t len, int desired_channels);
uint8_t*     vision_imencode(const VisionImage* img, const char* ext, size_t* out_len);
void         vision_imencode_free(uint8_t* buf);

/* Format & layout */
VisionImage* vision_to_float32(const VisionImage* src, float scale);
VisionImage* vision_to_uint8(const VisionImage* src, float scale);
VisionImage* vision_hwc_to_chw(const VisionImage* src);
VisionImage* vision_chw_to_hwc(const VisionImage* src);

/* Resize & spatial */
VisionImage* vision_resize(const VisionImage* src, int dst_w, int dst_h, int interp);
VisionImage* vision_resize_long_edge(const VisionImage* src, int long_edge, int interp);
VisionImage* vision_center_crop(const VisionImage* src, int crop_w, int crop_h);
VisionImage* vision_crop(const VisionImage* src, int x, int y, int w, int h);
VisionImage* vision_pad(const VisionImage* src, int top, int bottom,
                         int left, int right, int border, float fill_value);
VisionImage* vision_flip_horizontal(const VisionImage* src);
VisionImage* vision_flip_vertical(const VisionImage* src);
VisionImage* vision_rotate90(const VisionImage* src, int times);
VisionImage* vision_rotate(const VisionImage* src, float angle_deg,
                            int interp, int border, float fill_value);
VisionImage* vision_affine(const VisionImage* src, const float* M6,
                            int dst_w, int dst_h, int interp,
                            int border, float fill_value);
VisionImage* vision_perspective(const VisionImage* src,
                                 const float* src_pts8, const float* dst_pts8,
                                 int dst_w, int dst_h, int interp);

/* Color */
VisionImage* vision_to_grayscale(const VisionImage* src);
VisionImage* vision_rgb_to_bgr(const VisionImage* src);
VisionImage* vision_bgr_to_rgb(const VisionImage* src);
VisionImage* vision_rgb_to_hsv(const VisionImage* src);
VisionImage* vision_hsv_to_rgb(const VisionImage* src);
VisionImage* vision_normalize(const VisionImage* src, const float* mean, const float* std_dev);
VisionImage* vision_denormalize(const VisionImage* src, const float* mean, const float* std_dev);
VisionImage* vision_adjust_brightness(const VisionImage* src, float delta);
VisionImage* vision_adjust_contrast(const VisionImage* src, float factor);
VisionImage* vision_adjust_gamma(const VisionImage* src, float gamma);
VisionImage* vision_adjust_hue(const VisionImage* src, float delta_hue);
VisionImage* vision_histogram_equalize(const VisionImage* src);

/* Filtering */
VisionImage* vision_gaussian_blur(const VisionImage* src, int radius, float sigma);
VisionImage* vision_box_blur(const VisionImage* src, int radius);
VisionImage* vision_median_blur(const VisionImage* src, int radius);
VisionImage* vision_sobel(const VisionImage* src, VisionImage** out_gx, VisionImage** out_gy);
VisionImage* vision_laplacian(const VisionImage* src);
VisionImage* vision_canny(const VisionImage* src, float low_thresh, float high_thresh,
                           int gaussian_radius, float gaussian_sigma);
VisionImage* vision_convolve2d(const VisionImage* src, const VisionKernel* kernel, int border);

/* Morphology */
VisionImage* vision_erode(const VisionImage* src, int radius);
VisionImage* vision_dilate(const VisionImage* src, int radius);
VisionImage* vision_morph_open(const VisionImage* src, int radius);
VisionImage* vision_morph_close(const VisionImage* src, int radius);
VisionImage* vision_morph_gradient(const VisionImage* src, int radius);

/* Feature extraction */
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

/* Augmentation */
void         vision_rng_init(VisionRNG* rng, uint64_t seed);
uint32_t     vision_rng_next(VisionRNG* rng);
VisionImage* vision_random_crop(const VisionImage* src, int crop_w, int crop_h, VisionRNG* rng);
VisionImage* vision_random_resize_crop(const VisionImage* src,
                                        int out_w, int out_h,
                                        float scale_lo, float scale_hi,
                                        float ratio_lo, float ratio_hi,
                                        VisionRNG* rng, int interp);
VisionImage* vision_random_flip_horizontal(const VisionImage* src, float prob, VisionRNG* rng);
VisionImage* vision_random_flip_vertical(const VisionImage* src, float prob, VisionRNG* rng);
VisionImage* vision_random_brightness(const VisionImage* src, float max_delta, VisionRNG* rng);
VisionImage* vision_random_contrast(const VisionImage* src, float lo, float hi, VisionRNG* rng);
VisionImage* vision_random_hue(const VisionImage* src, float max_delta, VisionRNG* rng);
VisionImage* vision_cutout(const VisionImage* src, int n_holes, int hole_size,
                            float fill_value, VisionRNG* rng);
VisionImage* vision_mixup(const VisionImage* a, const VisionImage* b,
                           float alpha, VisionRNG* rng, float* out_lambda);
VisionImage* vision_cutmix(const VisionImage* a, const VisionImage* b,
                            float alpha, VisionRNG* rng, float* out_lambda);
VisionImage* vision_random_rotation(const VisionImage* src, float max_angle_deg,
                                     VisionRNG* rng, int interp, int border, float fill_value);

/* Detection */
float vision_iou(const VisionBBox* a, const VisionBBox* b);
float vision_giou(const VisionBBox* a, const VisionBBox* b);
float vision_diou(const VisionBBox* a, const VisionBBox* b);
VisionBBoxArray* vision_nms(const VisionBBoxArray* boxes, float iou_thresh);
VisionBBoxArray* vision_soft_nms(const VisionBBoxArray* boxes, float sigma, float score_thresh);
VisionBBoxArray* vision_bbox_array_create(int capacity);
int              vision_bbox_array_push(VisionBBoxArray* arr, const VisionBBox* box);
void             vision_bbox_array_free(VisionBBoxArray* arr);
VisionBBoxArray* vision_generate_anchors(int feat_w, int feat_h, int stride,
                                          const float* scales, int n_scales,
                                          const float* ratios, int n_ratios);
void vision_bbox_encode(const VisionBBox* anchor, const VisionBBox* gt,
                         float* dx, float* dy, float* dw, float* dh);
void vision_bbox_decode(const VisionBBox* anchor,
                         float dx, float dy, float dw, float dh, VisionBBox* out);

/* Segmentation */
VisionImage* vision_mask_resize(const VisionImage* mask, int dst_w, int dst_h);
VisionImage* vision_polygon_rasterize(const float* pts_xy, int n_pts,
                                       int img_w, int img_h, uint8_t fill_val);
VisionCC*    vision_connected_components(const VisionImage* binary_mask);
void         vision_cc_free(VisionCC* cc);

/* Tensor bridge */
void*        vision_image_to_tensor(const VisionImage* img);
VisionImage* vision_tensor_to_image(void* tensor_data, int width, int height,
                                     int channels, int format);
void         vision_free_raw(void* ptr);

/* ── Model decode (vision_model.c) ──────────────────────────────────────── */
VisionBBoxArray* vision_ssd_prior_boxes(const int* feat_sizes, int n_scales,
                                         const float* min_sizes,
                                         const float* max_sizes,
                                         const float* ratios, int n_ratios,
                                         int img_size);
VisionBBoxArray* vision_ssd_decode(const float* loc_pred, const float* cls_pred,
                                    const VisionBBoxArray* anchors,
                                    int n_cls, float conf_thr,
                                    float var_xy, float var_wh);
VisionBBoxArray* vision_nanodet_decode(const float* cls_pred, const float* reg_pred,
                                        int feat_h, int feat_w, int stride,
                                        int n_cls, int reg_max,
                                        int img_w, int img_h, float conf_thr);
VisionBBoxArray* vision_picodet_decode(const float* cls_pred, const float* reg_pred,
                                        int feat_h, int feat_w, int stride,
                                        int n_cls, int reg_max,
                                        int img_w, int img_h, float conf_thr);
VisionBBoxArray* vision_yolo11_decode(const float* output,
                                       int feat_h, int feat_w, int stride,
                                       int n_cls, int reg_max,
                                       int img_w, int img_h, float conf_thr);
VisionImage* vision_fastsam_assemble_masks(const float* proto, int n_proto,
                                            int proto_h, int proto_w,
                                            const float* coeffs, int n_dets,
                                            const VisionBBoxArray* boxes,
                                            int out_w, int out_h, float mask_thr);
CDEF;

    private function __construct()
    {
        if (!file_exists(self::LIB_SO)) {
            $this->compile();
        }
        $this->ffi = FFI::cdef(self::CDEF, self::LIB_SO);
    }

    public static function get(): self
    {
        if (self::$instance === null) {
            self::$instance = new self();
        }
        return self::$instance;
    }

    public function ffi(): FFI
    {
        return $this->ffi;
    }

    /** Auto-compile libvision.so from C sources (dev mode only). */
    private function compile(): void
    {
        // §36 — never auto-compile in production
        if (getenv('PML_ENV') === 'production') {
            throw new \RuntimeException(
                '[VisionEngine] libvision.so not found. Pre-build it before deploying.'
            );
        }
        if (!file_exists(self::BUILD_SH)) {
            throw new \RuntimeException('Vision build script not found: ' . self::BUILD_SH);
        }
        // §36 — use proc_open with 30s timeout instead of bare exec()
        $desc = [1 => ['pipe', 'w'], 2 => ['pipe', 'w']];
        $proc = proc_open('bash ' . escapeshellarg(self::BUILD_SH), $desc, $pipes);
        if (!is_resource($proc)) {
            throw new \RuntimeException('[VisionEngine] Failed to start build process.');
        }
        stream_set_blocking($pipes[1], false);
        stream_set_blocking($pipes[2], false);
        $deadline = microtime(true) + 30.0;
        $output   = '';
        while (proc_get_status($proc)['running']) {
            $output .= (string) stream_get_contents($pipes[1]);
            $output .= (string) stream_get_contents($pipes[2]);
            if (microtime(true) > $deadline) {
                proc_terminate($proc);
                fclose($pipes[1]); fclose($pipes[2]); proc_close($proc);
                throw new \RuntimeException('[VisionEngine] Build timed out after 30 s.');
            }
            usleep(50000);
        }
        $output .= (string) stream_get_contents($pipes[1]);
        $output .= (string) stream_get_contents($pipes[2]);
        $code = proc_close($proc);
        fclose($pipes[1]); fclose($pipes[2]);
        if ($code !== 0) {
            throw new \RuntimeException("Failed to compile libvision.so:\n" . $output);
        }
    }

    /** Allocate a new C VisionImage struct pointer. Caller must free. */
    public function createImage(int $w, int $h, int $channels,
                                 int $format, int $layout, int $colorSpace): FFI\CData
    {
        $img = $this->ffi->vision_image_create($w, $h, $channels, $format, $layout, $colorSpace);
        if (FFI::isNull($img)) {
            throw new \RuntimeException('vision_image_create failed: ' . $this->lastError());
        }
        return $img;
    }

    public function lastError(): string
    {
        $err = $this->ffi->vision_get_last_error();
        return $err !== null ? FFI::string($err) : '(no error)';
    }

    public function checkError(): void
    {
        if ($this->ffi->vision_check_error()) {
            $msg = $this->lastError();
            $this->ffi->vision_clear_error();
            throw new \RuntimeException('VisionEngine: ' . $msg);
        }
    }

    /** Allocate a typed C array (e.g. float[n]). */
    public function newArray(string $type, int $n): FFI\CData
    {
        return $this->ffi->new("{$type}[{$n}]");
    }

    /** Allocate a VisionRNG on the C side. */
    public function newRng(): FFI\CData
    {
        $rng = $this->ffi->new('VisionRNG');
        $this->ffi->vision_rng_init(FFI::addr($rng), (int)(microtime(true) * 1e6));
        return $rng;
    }
}
