#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

/* ------------------------------------------------------------------ helpers */

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}
static inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* sample one pixel channel at (fy,fx) with border clamp */
static inline float sample_bilinear_f32(const float *src,
                                        int width, int height, int channels,
                                        float fy, float fx, int c,
                                        size_t stride_bytes) {
    int x0 = (int)fx; int y0 = (int)fy;
    int x1 = x0+1;    int y1 = y0+1;
    float wx = fx - x0; float wy = fy - y0;
    x0 = clampi(x0,0,width-1);  x1 = clampi(x1,0,width-1);
    y0 = clampi(y0,0,height-1); y1 = clampi(y1,0,height-1);
    size_t stride = stride_bytes / sizeof(float);
    float v00 = src[y0*stride + x0*channels + c];
    float v01 = src[y0*stride + x1*channels + c];
    float v10 = src[y1*stride + x0*channels + c];
    float v11 = src[y1*stride + x1*channels + c];
    return (v00*(1-wx) + v01*wx)*(1-wy) + (v10*(1-wx) + v11*wx)*wy;
}

static inline uint8_t sample_bilinear_u8(const uint8_t *src,
                                         int width, int height, int channels,
                                         float fy, float fx, int c,
                                         size_t stride_bytes) {
    int x0 = (int)fx; int y0 = (int)fy;
    int x1 = x0+1;    int y1 = y0+1;
    float wx = fx - x0; float wy = fy - y0;
    x0 = clampi(x0,0,width-1);  x1 = clampi(x1,0,width-1);
    y0 = clampi(y0,0,height-1); y1 = clampi(y1,0,height-1);
    float v00 = src[y0*stride_bytes + x0*channels + c];
    float v01 = src[y0*stride_bytes + x1*channels + c];
    float v10 = src[y1*stride_bytes + x0*channels + c];
    float v11 = src[y1*stride_bytes + x1*channels + c];
    float v = (v00*(1-wx)+v01*wx)*(1-wy) + (v10*(1-wx)+v11*wx)*wy;
    return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : v + 0.5f);
}

/* cubic interpolation weight */
static inline float cubic_w(float t, float a) {
    float at = fabsf(t);
    if (at <= 1.f) return (a+2.f)*at*at*at - (a+3.f)*at*at + 1.f;
    if (at < 2.f)  return a*at*at*at - 5.f*a*at*at + 8.f*a*at - 4.f*a;
    return 0.f;
}

/* ------------------------------------------------------------------ nearest */

static VisionImage *resize_nearest(const VisionImage *src, int dw, int dh) {
    VisionImage *dst = vision_image_create(dw, dh, src->channels,
                                           src->format, src->layout, src->color_space);
    if (!dst) return NULL;

    float sx = (float)src->width  / dw;
    float sy = (float)src->height / dh;
    size_t elem = vision_element_size(src->format);
    int C = src->channels;

    #pragma omp parallel for schedule(static)
    for (int oy = 0; oy < dh; oy++) {
        int iy = clampi((int)(oy * sy), 0, src->height-1);
        for (int ox = 0; ox < dw; ox++) {
            int ix = clampi((int)(ox * sx), 0, src->width-1);
            uint8_t *d = dst->data + oy * dst->stride + ox * C * elem;
            const uint8_t *s = src->data + iy * src->stride + ix * C * elem;
            memcpy(d, s, C * elem);
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ bilinear */

#ifdef __AVX2__
/* Process 8 float output pixels along X for one row+channel strip.
   Caller handles the remainder. */
static void bilinear_row_avx2(const float *srow0, const float *srow1,
                               float *drow, int dx_start, int dx_end,
                               int src_width, int C, float sx, float ox_off) {
    for (int ox = dx_start; ox < dx_end - 7; ox += 8) {
        /* we stay scalar per pixel but vectorise over 8 x-positions */
        /* (full AVX2 gather-based bilinear is complex; this is AVX2-friendly loop) */
        for (int k = 0; k < 8; k++) {
            float fx = (ox + k + ox_off) * sx - 0.5f;
            if (fx < 0) fx = 0;
            int x0 = (int)fx; int x1 = x0+1;
            if (x1 >= src_width) x1 = src_width-1;
            float wx = fx - x0;
            for (int c = 0; c < C; c++) {
                float v = srow0[x0*C+c]*(1-wx) + srow0[x1*C+c]*wx; /* wy handled outside */
                drow[(ox+k)*C+c] = v; /* caller blends y */
            }
        }
    }
    (void)srow1; /* merged into caller */
}
#endif

static VisionImage *resize_bilinear(const VisionImage *src, int dw, int dh) {
    VisionImage *dst = vision_image_create(dw, dh, src->channels,
                                           src->format, src->layout, src->color_space);
    if (!dst) return NULL;

    int C = src->channels;
    float sx = (float)src->width  / dw;
    float sy = (float)src->height / dh;

    if (src->format == VISION_FMT_FLOAT32) {
        #pragma omp parallel for schedule(static)
        for (int oy = 0; oy < dh; oy++) {
            float fy = ((float)oy + 0.5f) * sy - 0.5f;
            if (fy < 0) fy = 0;
            int y0 = (int)fy; int y1 = y0+1;
            if (y1 >= src->height) y1 = src->height-1;
            float wy = fy - y0;
            const float *sr0 = (const float*)(src->data + y0 * src->stride);
            const float *sr1 = (const float*)(src->data + y1 * src->stride);
            float *dr = (float*)(dst->data + oy * dst->stride);

            for (int ox = 0; ox < dw; ox++) {
                float fx = ((float)ox + 0.5f) * sx - 0.5f;
                if (fx < 0) fx = 0;
                int x0 = (int)fx; int x1 = x0+1;
                if (x1 >= src->width) x1 = src->width-1;
                float wx = fx - x0;
                for (int c = 0; c < C; c++) {
                    float v00 = sr0[x0*C+c], v01 = sr0[x1*C+c];
                    float v10 = sr1[x0*C+c], v11 = sr1[x1*C+c];
                    dr[ox*C+c] = (v00*(1-wx)+v01*wx)*(1-wy)
                               + (v10*(1-wx)+v11*wx)*wy;
                }
            }
        }
    } else {
        /* uint8 path */
        #pragma omp parallel for schedule(static)
        for (int oy = 0; oy < dh; oy++) {
            float fy = ((float)oy + 0.5f) * sy - 0.5f;
            if (fy < 0) fy = 0;
            int y0 = (int)fy; int y1 = y0+1;
            if (y1 >= src->height) y1 = src->height-1;
            float wy = fy - y0;
            const uint8_t *sr0 = src->data + y0 * src->stride;
            const uint8_t *sr1 = src->data + y1 * src->stride;
            uint8_t *dr = dst->data + oy * dst->stride;

            for (int ox = 0; ox < dw; ox++) {
                float fx = ((float)ox + 0.5f) * sx - 0.5f;
                if (fx < 0) fx = 0;
                int x0 = (int)fx; int x1 = x0+1;
                if (x1 >= src->width) x1 = src->width-1;
                float wx = fx - x0;
                for (int c = 0; c < C; c++) {
                    float v00 = sr0[x0*C+c], v01 = sr0[x1*C+c];
                    float v10 = sr1[x0*C+c], v11 = sr1[x1*C+c];
                    float v = (v00*(1-wx)+v01*wx)*(1-wy)
                            + (v10*(1-wx)+v11*wx)*wy;
                    dr[ox*C+c] = (uint8_t)(v + 0.5f);
                }
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ bicubic */

static VisionImage *resize_bicubic(const VisionImage *src, int dw, int dh) {
    VisionImage *dst = vision_image_create(dw, dh, src->channels,
                                           src->format, src->layout, src->color_space);
    if (!dst) return NULL;

    int C = src->channels; float a = -0.75f;
    float sx = (float)src->width  / dw;
    float sy = (float)src->height / dh;

    if (src->format == VISION_FMT_FLOAT32) {
        #pragma omp parallel for schedule(static)
        for (int oy = 0; oy < dh; oy++) {
            float fy = ((float)oy + 0.5f) * sy - 0.5f;
            float *dr = (float*)(dst->data + oy * dst->stride);
            for (int ox = 0; ox < dw; ox++) {
                float fx = ((float)ox + 0.5f) * sx - 0.5f;
                int ixc = (int)fx; int iyc = (int)fy;
                for (int c = 0; c < C; c++) {
                    float acc = 0;
                    for (int m = -1; m <= 2; m++) {
                        float wy = cubic_w(fy - (iyc+m), a);
                        int yr = clampi(iyc+m, 0, src->height-1);
                        const float *row = (const float*)(src->data + yr*src->stride);
                        for (int n = -1; n <= 2; n++) {
                            float wx = cubic_w(fx - (ixc+n), a);
                            int xr = clampi(ixc+n, 0, src->width-1);
                            acc += wx * wy * row[xr*C+c];
                        }
                    }
                    dr[ox*C+c] = acc;
                }
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int oy = 0; oy < dh; oy++) {
            float fy = ((float)oy + 0.5f) * sy - 0.5f;
            uint8_t *dr = dst->data + oy * dst->stride;
            for (int ox = 0; ox < dw; ox++) {
                float fx = ((float)ox + 0.5f) * sx - 0.5f;
                int ixc = (int)fx; int iyc = (int)fy;
                for (int c = 0; c < C; c++) {
                    float acc = 0;
                    for (int m = -1; m <= 2; m++) {
                        float wy = cubic_w(fy - (iyc+m), a);
                        int yr = clampi(iyc+m, 0, src->height-1);
                        const uint8_t *row = src->data + yr*src->stride;
                        for (int n = -1; n <= 2; n++) {
                            float wx = cubic_w(fx - (ixc+n), a);
                            int xr = clampi(ixc+n, 0, src->width-1);
                            acc += wx * wy * row[xr*C+c];
                        }
                    }
                    dr[ox*C+c] = (uint8_t)(clampf(acc,0,255)+0.5f);
                }
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ area (box) downscale */

static VisionImage *resize_area(const VisionImage *src, int dw, int dh) {
    /* area averaging: best for downscaling */
    VisionImage *dst = vision_image_create(dw, dh, src->channels,
                                           src->format, src->layout, src->color_space);
    if (!dst) return NULL;

    int C = src->channels;
    float sx = (float)src->width  / dw;
    float sy = (float)src->height / dh;

    #pragma omp parallel for schedule(static)
    for (int oy = 0; oy < dh; oy++) {
        float y0f = oy * sy;
        float y1f = (oy+1) * sy;
        int y0 = (int)y0f; int y1 = clampi((int)ceilf(y1f), 0, src->height);
        for (int ox = 0; ox < dw; ox++) {
            float x0f = ox * sx;
            float x1f = (ox+1) * sx;
            int x0 = (int)x0f; int x1 = clampi((int)ceilf(x1f), 0, src->width);
            for (int c = 0; c < C; c++) {
                double acc = 0; double weight = 0;
                for (int iy = y0; iy < y1; iy++) {
                    float wy = 1.f;
                    if (iy == y0) wy = 1.f - (y0f - y0);
                    else if (iy == y1-1) wy = y1f - (y1-1);
                    for (int ix = x0; ix < x1; ix++) {
                        float wx = 1.f;
                        if (ix == x0) wx = 1.f - (x0f - x0);
                        else if (ix == x1-1) wx = x1f - (x1-1);
                        float w = wx * wy;
                        if (src->format == VISION_FMT_FLOAT32) {
                            const float *row = (const float*)(src->data + iy*src->stride);
                            acc += w * row[ix*C+c];
                        } else {
                            acc += w * src->data[iy*src->stride + ix*C+c];
                        }
                        weight += w;
                    }
                }
                double v = weight > 0 ? acc / weight : 0;
                if (src->format == VISION_FMT_FLOAT32) {
                    float *dr = (float*)(dst->data + oy*dst->stride);
                    dr[ox*C+c] = (float)v;
                } else {
                    dst->data[oy*dst->stride + ox*C+c] = (uint8_t)(v + 0.5);
                }
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ public resize */

VisionImage *vision_resize(const VisionImage *src, int new_width, int new_height,
                           VisionInterp interp) {
    if (!src || new_width <= 0 || new_height <= 0) {
        VISION_ERR("vision_resize: invalid args"); return NULL;
    }
    if (src->layout != VISION_LAYOUT_HWC) {
        VISION_ERR("vision_resize: only HWC layout supported"); return NULL;
    }
    switch (interp) {
        case VISION_INTERP_NEAREST:  return resize_nearest(src, new_width, new_height);
        case VISION_INTERP_BILINEAR: return resize_bilinear(src, new_width, new_height);
        case VISION_INTERP_BICUBIC:  return resize_bicubic(src, new_width, new_height);
        case VISION_INTERP_AREA:     return resize_area(src, new_width, new_height);
        default: VISION_ERR("vision_resize: unknown interp"); return NULL;
    }
}

VisionImage *vision_resize_long_edge(const VisionImage *src, int long_edge,
                                     VisionInterp interp) {
    if (!src || long_edge <= 0) { VISION_ERR("vision_resize_long_edge: invalid"); return NULL; }
    int W = src->width, H = src->height;
    int nw, nh;
    if (W >= H) { nw = long_edge; nh = (int)roundf((float)H * long_edge / W); }
    else        { nh = long_edge; nw = (int)roundf((float)W * long_edge / H); }
    if (nw < 1) nw = 1;
    if (nh < 1) nh = 1;
    return vision_resize(src, nw, nh, interp);
}

/* ------------------------------------------------------------------ crop */

VisionImage *vision_crop(const VisionImage *src, int x, int y, int w, int h) {
    if (!src || x < 0 || y < 0 || w <= 0 || h <= 0
     || x + w > src->width || y + h > src->height) {
        VISION_ERR("vision_crop: out of bounds"); return NULL;
    }
    size_t elem = vision_element_size(src->format);
    int C = src->channels;
    VisionImage *dst = vision_image_create(w, h, C, src->format,
                                           src->layout, src->color_space);
    if (!dst) return NULL;
    for (int row = 0; row < h; row++) {
        const uint8_t *s = src->data + (y+row)*src->stride + x*C*elem;
        uint8_t *d = dst->data + row*dst->stride;
        memcpy(d, s, (size_t)w * C * elem);
    }
    return dst;
}

VisionImage *vision_center_crop(const VisionImage *src, int w, int h) {
    int x = (src->width  - w) / 2;
    int y = (src->height - h) / 2;
    return vision_crop(src, x, y, w, h);
}

/* ------------------------------------------------------------------ pad */

VisionImage *vision_pad(const VisionImage *src,
                        int top, int bottom, int left, int right,
                        VisionBorderMode mode, float fill_value) {
    if (!src) { VISION_ERR("vision_pad: null"); return NULL; }
    int nw = src->width  + left + right;
    int nh = src->height + top  + bottom;
    size_t elem = vision_element_size(src->format);
    int C = src->channels;
    VisionImage *dst = vision_image_create(nw, nh, C, src->format,
                                           src->layout, src->color_space);
    if (!dst) return NULL;

    /* fill background */
    if (mode == VISION_BORDER_CONSTANT) {
        for (int oy = 0; oy < nh; oy++) {
            uint8_t *dr = dst->data + oy * dst->stride;
            for (int ox = 0; ox < nw; ox++) {
                for (int c = 0; c < C; c++) {
                    if (src->format == VISION_FMT_FLOAT32)
                        ((float*)dr)[ox*C+c] = fill_value;
                    else
                        dr[ox*C+c] = (uint8_t)clampf(fill_value,0,255);
                }
            }
        }
    }

    /* copy source rows into padded region */
    for (int sy = 0; sy < src->height; sy++) {
        int src_sy = sy;
        if (mode == VISION_BORDER_REFLECT)
            src_sy = (sy < 0) ? -sy : (sy >= src->height ? 2*src->height-2-sy : sy);
        const uint8_t *s = src->data + src_sy * src->stride;
        uint8_t *d = dst->data + (top+sy) * dst->stride + left * C * elem;
        memcpy(d, s, (size_t)src->width * C * elem);
    }

    /* fill left/right columns for non-constant border */
    if (mode != VISION_BORDER_CONSTANT) {
        for (int oy = top; oy < top + src->height; oy++) {
            int sy = oy - top;
            for (int ox = 0; ox < left; ox++) {
                int sx = (mode == VISION_BORDER_REFLECT) ? left - 1 - ox :
                         (mode == VISION_BORDER_REPLICATE) ? 0 : ox % src->width;
                sx = clampi(sx, 0, src->width-1);
                const uint8_t *s = src->data + sy*src->stride + sx*C*elem;
                uint8_t *d = dst->data + oy*dst->stride + ox*C*elem;
                memcpy(d, s, C*elem);
            }
            for (int ox = left+src->width; ox < nw; ox++) {
                int sx_off = ox - (left + src->width);
                int sx = (mode == VISION_BORDER_REFLECT) ? src->width-2-sx_off :
                         (mode == VISION_BORDER_REPLICATE) ? src->width-1
                         : sx_off % src->width;
                sx = clampi(sx, 0, src->width-1);
                const uint8_t *s = src->data + sy*src->stride + sx*C*elem;
                uint8_t *d = dst->data + oy*dst->stride + ox*C*elem;
                memcpy(d, s, C*elem);
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ flip */

VisionImage *vision_flip_horizontal(const VisionImage *src) {
    if (!src) { VISION_ERR("vision_flip_horizontal: null"); return NULL; }
    int W = src->width, H = src->height, C = src->channels;
    size_t elem = vision_element_size(src->format);
    VisionImage *dst = vision_image_create(W, H, C, src->format,
                                           src->layout, src->color_space);
    if (!dst) return NULL;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        const uint8_t *sr = src->data + y * src->stride;
        uint8_t *dr = dst->data + y * dst->stride;
        for (int x = 0; x < W; x++) {
            int rx = W - 1 - x;
            memcpy(dr + x*C*elem, sr + rx*C*elem, C*elem);
        }
    }
    return dst;
}

VisionImage *vision_flip_vertical(const VisionImage *src) {
    if (!src) { VISION_ERR("vision_flip_vertical: null"); return NULL; }
    int W = src->width, H = src->height, C = src->channels;
    size_t elem = vision_element_size(src->format);
    VisionImage *dst = vision_image_create(W, H, C, src->format,
                                           src->layout, src->color_space);
    if (!dst) return NULL;
    for (int y = 0; y < H; y++) {
        memcpy(dst->data + y*dst->stride,
               src->data + (H-1-y)*src->stride,
               (size_t)W*C*elem);
    }
    return dst;
}

/* ------------------------------------------------------------------ rotate90 */

VisionImage *vision_rotate90(const VisionImage *src, int k) {
    if (!src) { VISION_ERR("vision_rotate90: null"); return NULL; }
    k = ((k % 4) + 4) % 4;
    if (k == 0) return vision_image_clone(src);

    int W = src->width, H = src->height, C = src->channels;
    size_t elem = vision_element_size(src->format);
    int dw = (k & 1) ? H : W;
    int dh = (k & 1) ? W : H;
    VisionImage *dst = vision_image_create(dw, dh, C, src->format,
                                           src->layout, src->color_space);
    if (!dst) return NULL;

    #pragma omp parallel for schedule(static)
    for (int sy = 0; sy < H; sy++) {
        const uint8_t *sr = src->data + sy * src->stride;
        for (int sx = 0; sx < W; sx++) {
            int dx, dy;
            if      (k==1) { dx = H-1-sy; dy = sx; }
            else if (k==2) { dx = W-1-sx; dy = H-1-sy; }
            else           { dx = sy;     dy = W-1-sx; }
            memcpy(dst->data + dy*dst->stride + dx*C*elem,
                   sr + sx*C*elem, C*elem);
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ affine */

VisionImage *vision_affine(const VisionImage *src, const float M[6],
                           int out_width, int out_height,
                           VisionInterp interp, VisionBorderMode border,
                           float fill_value) {
    if (!src) { VISION_ERR("vision_affine: null"); return NULL; }
    int C = src->channels;
    size_t elem = vision_element_size(src->format);
    VisionImage *dst = vision_image_create(out_width, out_height, C,
                                           src->format, src->layout, src->color_space);
    if (!dst) return NULL;

    /* M is 2×3 forward matrix dst→src: [a b tx; c d ty] */
    float a=M[0],b=M[1],tx=M[2],c=M[3],d=M[4],ty=M[5];

    #pragma omp parallel for schedule(static)
    for (int oy = 0; oy < out_height; oy++) {
        uint8_t *dr = dst->data + oy * dst->stride;
        for (int ox = 0; ox < out_width; ox++) {
            float sx = a*ox + b*oy + tx;
            float sy = c*ox + d*oy + ty;
            uint8_t *dpix = dr + ox*C*elem;

            int in_bounds = (sx >= 0 && sx < src->width-1 && sy >= 0 && sy < src->height-1);
            if (!in_bounds && border == VISION_BORDER_CONSTANT) {
                for (int ch = 0; ch < C; ch++) {
                    if (src->format == VISION_FMT_FLOAT32)
                        ((float*)dpix)[ch] = fill_value;
                    else
                        dpix[ch] = (uint8_t)clampf(fill_value,0,255);
                }
                continue;
            }
            /* clamp for replicate / reflect */
            sx = clampf(sx, 0, src->width-1);
            sy = clampf(sy, 0, src->height-1);

            if (interp == VISION_INTERP_NEAREST) {
                int ix = clampi((int)(sx+0.5f), 0, src->width-1);
                int iy = clampi((int)(sy+0.5f), 0, src->height-1);
                memcpy(dpix, src->data + iy*src->stride + ix*C*elem, C*elem);
            } else {
                /* bilinear default */
                int x0=(int)sx, y0=(int)sy;
                int x1=clampi(x0+1,0,src->width-1);
                int y1=clampi(y0+1,0,src->height-1);
                float wx=sx-x0, wy=sy-y0;
                for (int ch = 0; ch < C; ch++) {
                    if (src->format == VISION_FMT_FLOAT32) {
                        const float *r0=(const float*)(src->data+y0*src->stride);
                        const float *r1=(const float*)(src->data+y1*src->stride);
                        float v=(r0[x0*C+ch]*(1-wx)+r0[x1*C+ch]*wx)*(1-wy)
                               +(r1[x0*C+ch]*(1-wx)+r1[x1*C+ch]*wx)*wy;
                        ((float*)dpix)[ch]=v;
                    } else {
                        const uint8_t *r0=src->data+y0*src->stride;
                        const uint8_t *r1=src->data+y1*src->stride;
                        float v=(r0[x0*C+ch]*(1-wx)+r0[x1*C+ch]*wx)*(1-wy)
                               +(r1[x0*C+ch]*(1-wx)+r1[x1*C+ch]*wx)*wy;
                        dpix[ch]=(uint8_t)(v+0.5f);
                    }
                }
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ rotate (arbitrary angle) */

VisionImage *vision_rotate(const VisionImage *src, float angle_deg,
                           VisionInterp interp, VisionBorderMode border,
                           float fill_value) {
    if (!src) { VISION_ERR("vision_rotate: null"); return NULL; }
    float rad = angle_deg * 3.14159265358979f / 180.f;
    float cosA = cosf(rad), sinA = sinf(rad);
    float cx = src->width * 0.5f, cy = src->height * 0.5f;
    /* inverse mapping: dst_pixel → src_pixel */
    /* sx = cosA*(ox-cx) + sinA*(oy-cy) + cx */
    /* sy = -sinA*(ox-cx) + cosA*(oy-cy) + cy */
    float M[6] = {
         cosA,  sinA, cx - cosA*cx - sinA*cy,
        -sinA,  cosA, cy + sinA*cx - cosA*cy
    };
    return vision_affine(src, M, src->width, src->height, interp, border, fill_value);
}

/* ------------------------------------------------------------------ perspective */

VisionImage *vision_perspective(const VisionImage *src,
                                const float src_pts[8], const float dst_pts[8],
                                int out_width, int out_height, VisionInterp interp) {
    if (!src) { VISION_ERR("vision_perspective: null"); return NULL; }

    /* solve 3×3 homography H such that dst_pt ~ H * src_pt (using 4 point pairs) */
    /* build 8×8 linear system A*h=b */
    double A[8][8], bv[8];
    for (int i = 0; i < 4; i++) {
        double sx=src_pts[i*2], sy=src_pts[i*2+1];
        double dx=dst_pts[i*2], dy=dst_pts[i*2+1];
        A[i*2][0]=sx; A[i*2][1]=sy; A[i*2][2]=1;
        A[i*2][3]=0;  A[i*2][4]=0;  A[i*2][5]=0;
        A[i*2][6]=-dx*sx; A[i*2][7]=-dx*sy; bv[i*2]=dx;
        A[i*2+1][0]=0; A[i*2+1][1]=0; A[i*2+1][2]=0;
        A[i*2+1][3]=sx; A[i*2+1][4]=sy; A[i*2+1][5]=1;
        A[i*2+1][6]=-dy*sx; A[i*2+1][7]=-dy*sy; bv[i*2+1]=dy;
    }
    /* Gaussian elimination */
    for (int col = 0; col < 8; col++) {
        int pivot = col;
        for (int row = col+1; row < 8; row++)
            if (fabs(A[row][col]) > fabs(A[pivot][col])) pivot = row;
        if (pivot != col) {
            for (int k=0;k<8;k++){double t=A[col][k];A[col][k]=A[pivot][k];A[pivot][k]=t;}
            double t=bv[col]; bv[col]=bv[pivot]; bv[pivot]=t;
        }
        double dv = A[col][col];
        if (fabs(dv) < 1e-12) { VISION_ERR("perspective: degenerate"); return NULL; }
        for (int row = col+1; row < 8; row++) {
            double factor = A[row][col] / dv;
            for (int k=col; k<8; k++) A[row][k] -= factor * A[col][k];
            bv[row] -= factor * bv[col];
        }
    }
    double h[9];
    for (int i = 7; i >= 0; i--) {
        h[i] = bv[i];
        for (int j = i+1; j < 8; j++) h[i] -= A[i][j]*h[j];
        h[i] /= A[i][i];
    }
    h[8] = 1.0;
    /* h[0..8] is forward H (src→dst). We need inverse for dst→src sampling. */
    /* build inverse of 3x3 */
    double det = h[0]*(h[4]*h[8]-h[5]*h[7]) - h[1]*(h[3]*h[8]-h[5]*h[6])
               + h[2]*(h[3]*h[7]-h[4]*h[6]);
    if (fabs(det) < 1e-12) { VISION_ERR("perspective: singular H"); return NULL; }
    double invH[9];
    invH[0]=(h[4]*h[8]-h[5]*h[7])/det; invH[1]=(h[2]*h[7]-h[1]*h[8])/det;
    invH[2]=(h[1]*h[5]-h[2]*h[4])/det; invH[3]=(h[5]*h[6]-h[3]*h[8])/det;
    invH[4]=(h[0]*h[8]-h[2]*h[6])/det; invH[5]=(h[2]*h[3]-h[0]*h[5])/det;
    invH[6]=(h[3]*h[7]-h[4]*h[6])/det; invH[7]=(h[1]*h[6]-h[0]*h[7])/det;
    invH[8]=(h[0]*h[4]-h[1]*h[3])/det;

    int C = src->channels;
    size_t elem = vision_element_size(src->format);
    VisionImage *dst = vision_image_create(out_width, out_height, C,
                                           src->format, src->layout, src->color_space);
    if (!dst) return NULL;

    #pragma omp parallel for schedule(static)
    for (int oy = 0; oy < out_height; oy++) {
        uint8_t *dr = dst->data + oy * dst->stride;
        for (int ox = 0; ox < out_width; ox++) {
            double pw = invH[6]*ox + invH[7]*oy + invH[8];
            double sx_d = (invH[0]*ox + invH[1]*oy + invH[2]) / pw;
            double sy_d = (invH[3]*ox + invH[4]*oy + invH[5]) / pw;
            float sx = (float)sx_d, sy = (float)sy_d;
            uint8_t *dpix = dr + ox*C*elem;
            if (sx < 0 || sx >= src->width || sy < 0 || sy >= src->height) {
                memset(dpix, 0, C*elem);
                continue;
            }
            if (interp == VISION_INTERP_NEAREST) {
                int ix=clampi((int)(sx+0.5f),0,src->width-1);
                int iy=clampi((int)(sy+0.5f),0,src->height-1);
                memcpy(dpix, src->data + iy*src->stride + ix*C*elem, C*elem);
            } else {
                int x0=(int)sx, y0=(int)sy;
                int x1=clampi(x0+1,0,src->width-1), y1=clampi(y0+1,0,src->height-1);
                float wx=sx-x0, wy=sy-y0;
                for (int ch=0; ch<C; ch++) {
                    if (src->format == VISION_FMT_FLOAT32) {
                        const float *r0=(const float*)(src->data+y0*src->stride);
                        const float *r1=(const float*)(src->data+y1*src->stride);
                        float v=(r0[x0*C+ch]*(1-wx)+r0[x1*C+ch]*wx)*(1-wy)
                               +(r1[x0*C+ch]*(1-wx)+r1[x1*C+ch]*wx)*wy;
                        ((float*)dpix)[ch]=v;
                    } else {
                        const uint8_t *r0=src->data+y0*src->stride;
                        const uint8_t *r1=src->data+y1*src->stride;
                        float v=(r0[x0*C+ch]*(1-wx)+r0[x1*C+ch]*wx)*(1-wy)
                               +(r1[x0*C+ch]*(1-wx)+r1[x1*C+ch]*wx)*wy;
                        dpix[ch]=(uint8_t)(v+0.5f);
                    }
                }
            }
        }
    }
    return dst;
}
