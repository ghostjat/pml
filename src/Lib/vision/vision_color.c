#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

static inline float clampf(float v, float lo, float hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

/* ------------------------------------------------------------------ grayscale */

VisionImage *vision_to_grayscale(const VisionImage *src) {
    if (!src) { VISION_ERR("vision_to_grayscale: null"); return NULL; }
    if (src->channels == 1) return vision_image_clone(src);
    if (src->channels != 3 && src->channels != 4) {
        VISION_ERR("vision_to_grayscale: need 3 or 4 channels"); return NULL;
    }
    int W=src->width, H=src->height, C=src->channels;
    VisionImage *dst = vision_image_create(W, H, 1, src->format,
                                           src->layout, VISION_CS_GRAY);
    if (!dst) return NULL;

    if (src->format == VISION_FMT_FLOAT32) {
        /* ITU-R BT.601: Y = 0.299R + 0.587G + 0.114B */
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++) {
            const float *sr = (const float*)(src->data + y*src->stride);
            float *dr = (float*)(dst->data + y*dst->stride);
#ifdef __AVX2__
            __m256 wr = _mm256_set1_ps(0.299f);
            __m256 wg = _mm256_set1_ps(0.587f);
            __m256 wb = _mm256_set1_ps(0.114f);
            int ox = 0;
            if (C == 3) {
                for (; ox <= W-8; ox += 8) {
                    /* gather R,G,B for 8 pixels — strided, no intrinsic gather */
                    float rv[8], gv[8], bv[8];
                    for (int k=0;k<8;k++){rv[k]=sr[(ox+k)*3];gv[k]=sr[(ox+k)*3+1];bv[k]=sr[(ox+k)*3+2];}
                    __m256 R=_mm256_loadu_ps(rv), G=_mm256_loadu_ps(gv), B=_mm256_loadu_ps(bv);
                    __m256 Y = _mm256_fmadd_ps(R,wr,_mm256_fmadd_ps(G,wg,_mm256_mul_ps(B,wb)));
                    _mm256_storeu_ps(dr+ox, Y);
                }
            }
            for (int x = ox; x < W; x++)
#else
            for (int x = 0; x < W; x++)
#endif
            {
                dr[x] = sr[x*C]*0.299f + sr[x*C+1]*0.587f + sr[x*C+2]*0.114f;
            }
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int y = 0; y < H; y++) {
            const uint8_t *sr = src->data + y*src->stride;
            uint8_t *dr = dst->data + y*dst->stride;
            for (int x = 0; x < W; x++) {
                int r=sr[x*C], g=sr[x*C+1], b=sr[x*C+2];
                dr[x] = (uint8_t)((r*299 + g*587 + b*114 + 500) / 1000);
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ RGB <-> BGR */

VisionImage *vision_rgb_to_bgr(const VisionImage *src) {
    if (!src || src->channels < 3) { VISION_ERR("rgb_to_bgr: need >=3ch"); return NULL; }
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;
    int W=src->width, H=src->height, C=src->channels;
    size_t elem = vision_element_size(src->format);

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        uint8_t *row = dst->data + y*dst->stride;
        for (int x = 0; x < W; x++) {
            /* swap channel 0 and 2 */
            uint8_t tmp[4];
            memcpy(tmp, row + x*C*elem, C*elem);
            memcpy(row + x*C*elem,     tmp + 2*elem, elem);   /* R←B */
            memcpy(row + x*C*elem + 2*elem, tmp,     elem);   /* B←R */
        }
    }
    dst->color_space = (src->color_space == VISION_CS_RGB) ? VISION_CS_BGR : VISION_CS_RGB;
    return dst;
}

VisionImage *vision_bgr_to_rgb(const VisionImage *src) {
    return vision_rgb_to_bgr(src); /* swap is symmetric */
}

/* ------------------------------------------------------------------ RGB <-> HSV */

static void rgb2hsv(float r, float g, float b, float *h, float *s, float *v) {
    float M = r > g ? (r > b ? r : b) : (g > b ? g : b);
    float m = r < g ? (r < b ? r : b) : (g < b ? g : b);
    float d = M - m;
    *v = M;
    *s = (M > 1e-6f) ? d / M : 0.f;
    if (d < 1e-6f) { *h = 0; return; }
    if (M == r)      *h = (g - b) / d + (g < b ? 6.f : 0.f);
    else if (M == g) *h = (b - r) / d + 2.f;
    else             *h = (r - g) / d + 4.f;
    *h /= 6.f;
}

static void hsv2rgb(float h, float s, float v, float *r, float *g, float *b) {
    if (s < 1e-6f) { *r = *g = *b = v; return; }
    float hh = h * 6.f; int i = (int)hh; float f = hh - i;
    float p=v*(1-s), q=v*(1-s*f), t=v*(1-s*(1-f));
    switch (i % 6) {
        case 0: *r=v;*g=t;*b=p; break; case 1: *r=q;*g=v;*b=p; break;
        case 2: *r=p;*g=v;*b=t; break; case 3: *r=p;*g=q;*b=v; break;
        case 4: *r=t;*g=p;*b=v; break; default: *r=v;*g=p;*b=q; break;
    }
}

VisionImage *vision_rgb_to_hsv(const VisionImage *src) {
    if (!src || src->channels < 3 || src->format != VISION_FMT_FLOAT32) {
        VISION_ERR("rgb_to_hsv: needs float32 >=3ch"); return NULL;
    }
    int W=src->width, H=src->height, C=src->channels;
    VisionImage *dst = vision_image_create(W, H, C, VISION_FMT_FLOAT32,
                                           src->layout, VISION_CS_HSV);
    if (!dst) return NULL;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        const float *sr = (const float*)(src->data + y*src->stride);
        float *dr = (float*)(dst->data + y*dst->stride);
        for (int x = 0; x < W; x++) {
            float hv,sv,vv;
            rgb2hsv(sr[x*C], sr[x*C+1], sr[x*C+2], &hv, &sv, &vv);
            dr[x*C]=hv; dr[x*C+1]=sv; dr[x*C+2]=vv;
            for (int c=3;c<C;c++) dr[x*C+c]=sr[x*C+c];
        }
    }
    return dst;
}

VisionImage *vision_hsv_to_rgb(const VisionImage *src) {
    if (!src || src->channels < 3 || src->format != VISION_FMT_FLOAT32) {
        VISION_ERR("hsv_to_rgb: needs float32 >=3ch"); return NULL;
    }
    int W=src->width, H=src->height, C=src->channels;
    VisionImage *dst = vision_image_create(W, H, C, VISION_FMT_FLOAT32,
                                           src->layout, VISION_CS_RGB);
    if (!dst) return NULL;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        const float *sr = (const float*)(src->data + y*src->stride);
        float *dr = (float*)(dst->data + y*dst->stride);
        for (int x = 0; x < W; x++) {
            float r,g,b;
            hsv2rgb(sr[x*C], sr[x*C+1], sr[x*C+2], &r, &g, &b);
            dr[x*C]=r; dr[x*C+1]=g; dr[x*C+2]=b;
            for (int c=3;c<C;c++) dr[x*C+c]=sr[x*C+c];
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ normalize / denormalize */

VisionImage *vision_normalize(const VisionImage *src,
                              const float *mean, const float *std_dev) {
    if (!src || src->format != VISION_FMT_FLOAT32) {
        VISION_ERR("vision_normalize: needs float32"); return NULL;
    }
    int W=src->width, H=src->height, C=src->channels;
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;

    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        float *dr = (float*)(dst->data + y*dst->stride);
#ifdef __AVX2__
        /* per-channel AVX2 fmadd: out = (in - mean) / std */
        for (int x = 0; x < W; x++) {
            for (int c = 0; c < C; c++) {
                dr[x*C+c] = (dr[x*C+c] - mean[c]) / std_dev[c];
            }
        }
#else
        for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++)
                dr[x*C+c] = (dr[x*C+c] - mean[c]) / std_dev[c];
#endif
    }
    return dst;
}

VisionImage *vision_denormalize(const VisionImage *src,
                                const float *mean, const float *std_dev) {
    if (!src || src->format != VISION_FMT_FLOAT32) {
        VISION_ERR("vision_denormalize: needs float32"); return NULL;
    }
    int W=src->width, H=src->height, C=src->channels;
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        float *dr = (float*)(dst->data + y*dst->stride);
        for (int x = 0; x < W; x++)
            for (int c = 0; c < C; c++)
                dr[x*C+c] = dr[x*C+c] * std_dev[c] + mean[c];
    }
    return dst;
}

/* ------------------------------------------------------------------ brightness / contrast / gamma */

VisionImage *vision_adjust_brightness(const VisionImage *src, float delta) {
    if (!src) { VISION_ERR("brightness: null"); return NULL; }
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;
    int W=src->width, H=src->height, C=src->channels;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        uint8_t *dr = dst->data + y*dst->stride;
        if (src->format == VISION_FMT_FLOAT32) {
            float *fr = (float*)dr;
            for (int x=0;x<W*C;x++) fr[x] = clampf(fr[x]+delta,0,1);
        } else {
            for (int x=0;x<W*C;x++) {
                int v = dr[x] + (int)(delta*255);
                dr[x] = (uint8_t)(v<0?0:v>255?255:v);
            }
        }
    }
    return dst;
}

VisionImage *vision_adjust_contrast(const VisionImage *src, float factor) {
    if (!src) { VISION_ERR("contrast: null"); return NULL; }
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;
    int W=src->width, H=src->height, C=src->channels;
    float mid = (src->format == VISION_FMT_FLOAT32) ? 0.5f : 127.5f;
    #pragma omp parallel for schedule(static)
    for (int y = 0; y < H; y++) {
        uint8_t *dr = dst->data + y*dst->stride;
        if (src->format == VISION_FMT_FLOAT32) {
            float *fr = (float*)dr;
            for (int x=0;x<W*C;x++) fr[x] = clampf(factor*(fr[x]-mid)+mid,0,1);
        } else {
            for (int x=0;x<W*C;x++) {
                float v = factor*(dr[x]-mid)+mid;
                dr[x]=(uint8_t)(v<0?0:v>255?255:v);
            }
        }
    }
    return dst;
}

VisionImage *vision_adjust_gamma(const VisionImage *src, float gamma) {
    if (!src || gamma <= 0) { VISION_ERR("gamma: invalid"); return NULL; }
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;
    int W=src->width, H=src->height, C=src->channels;
    float inv_gamma = 1.f / gamma;

    if (src->format == VISION_FMT_UINT8) {
        /* precompute LUT */
        uint8_t lut[256];
        for (int i=0;i<256;i++) lut[i]=(uint8_t)(powf(i/255.f,inv_gamma)*255.f+0.5f);
        #pragma omp parallel for schedule(static)
        for (int y=0;y<H;y++) {
            uint8_t *dr=dst->data+y*dst->stride;
            for (int x=0;x<W*C;x++) dr[x]=lut[dr[x]];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for (int y=0;y<H;y++) {
            float *fr=(float*)(dst->data+y*dst->stride);
            for (int x=0;x<W*C;x++) fr[x]=clampf(powf(clampf(fr[x],0,1),inv_gamma),0,1);
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ adjust_hue */

VisionImage *vision_adjust_hue(const VisionImage *src, float delta_hue) {
    /* delta_hue in [-0.5, 0.5] (fraction of full circle) */
    if (!src || src->channels < 3 || src->format != VISION_FMT_FLOAT32) {
        VISION_ERR("adjust_hue: needs float32 >=3ch"); return NULL;
    }
    VisionImage *hsv = vision_rgb_to_hsv(src);
    if (!hsv) return NULL;
    int W=src->width, H=src->height, C=src->channels;
    #pragma omp parallel for schedule(static)
    for (int y=0;y<H;y++) {
        float *row=(float*)(hsv->data+y*hsv->stride);
        for (int x=0;x<W;x++) {
            row[x*C] = fmodf(row[x*C]+delta_hue+1.f, 1.f);
        }
    }
    VisionImage *out = vision_hsv_to_rgb(hsv);
    vision_image_free(hsv);
    return out;
}

/* ------------------------------------------------------------------ histogram equalization */

VisionImage *vision_histogram_equalize(const VisionImage *src) {
    if (!src || src->format != VISION_FMT_UINT8) {
        VISION_ERR("histogram_equalize: needs uint8"); return NULL;
    }
    int W=src->width, H=src->height, C=src->channels;

    /* equalize per channel */
    VisionImage *dst = vision_image_clone(src);
    if (!dst) return NULL;

    for (int c=0; c<C; c++) {
        long hist[256] = {0};
        for (int y=0;y<H;y++) {
            const uint8_t *row=src->data+y*src->stride;
            for (int x=0;x<W;x++) hist[row[x*C+c]]++;
        }
        /* CDF */
        long cdf[256]; long cdf_min=0; cdf[0]=hist[0];
        for (int i=1;i<256;i++) cdf[i]=cdf[i-1]+hist[i];
        for (int i=0;i<256;i++) if (hist[i]>0){cdf_min=cdf[i];break;}
        long total = (long)W*H;
        uint8_t lut[256];
        for (int i=0;i<256;i++) {
            long d = total-cdf_min;
            lut[i] = (d>0) ? (uint8_t)(((cdf[i]-cdf_min)*255L+d/2)/d) : 0;
        }
        for (int y=0;y<H;y++) {
            uint8_t *row=dst->data+y*dst->stride;
            for (int x=0;x<W;x++) row[x*C+c]=lut[row[x*C+c]];
        }
    }
    return dst;
}
