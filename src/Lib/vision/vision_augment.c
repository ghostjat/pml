#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static inline int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}
static inline float clampf(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}

/* ------------------------------------------------------------------ xoshiro128+ RNG */

static inline uint32_t rotl32(uint32_t x, int k) {
    return (x << k) | (x >> (32 - k));
}

void vision_rng_init(VisionRNG *rng, uint64_t seed) {
    /* splitmix64 to seed 4 × uint32 */
    uint64_t s = seed;
    for(int i=0;i<4;i++){
        s += 0x9e3779b97f4a7c15ULL;
        uint64_t z = s;
        z = (z ^ (z>>30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z>>27)) * 0x94d049bb133111ebULL;
        z = z ^ (z>>31);
        rng->state[i] = (uint32_t)z;
    }
}

uint32_t vision_rng_next(VisionRNG *rng) {
    uint32_t result = rng->state[0] + rng->state[3];
    uint32_t t = rng->state[1] << 9;
    rng->state[2] ^= rng->state[0];
    rng->state[3] ^= rng->state[1];
    rng->state[1] ^= rng->state[2];
    rng->state[0] ^= rng->state[3];
    rng->state[2] ^= t;
    rng->state[3] = rotl32(rng->state[3], 11);
    return result;
}

/* [0,1) float */
static inline float rng_float(VisionRNG *rng) {
    return (float)(vision_rng_next(rng) >> 8) / 16777216.f;
}
/* [lo,hi) float */
static inline float rng_range(VisionRNG *rng, float lo, float hi) {
    return lo + rng_float(rng)*(hi-lo);
}
/* [0,n) int */
static inline int rng_int(VisionRNG *rng, int n) {
    return (int)(rng_float(rng)*n);
}

/* ------------------------------------------------------------------ random crop */

VisionImage *vision_random_crop(const VisionImage *src, int crop_w, int crop_h,
                                VisionRNG *rng) {
    if(!src||crop_w<=0||crop_h<=0||crop_w>src->width||crop_h>src->height){
        VISION_ERR("random_crop: invalid size"); return NULL;
    }
    int x=rng_int(rng, src->width -crop_w+1);
    int y=rng_int(rng, src->height-crop_h+1);
    return vision_crop(src, x, y, crop_w, crop_h);
}

/* ------------------------------------------------------------------ random resize crop (TorchVision semantics) */

VisionImage *vision_random_resize_crop(const VisionImage *src,
                                       int out_w, int out_h,
                                       float scale_lo, float scale_hi,
                                       float ratio_lo, float ratio_hi,
                                       VisionRNG *rng, VisionInterp interp) {
    if(!src){VISION_ERR("rrc: null");return NULL;}
    int W=src->width, H=src->height;
    int area=W*H;
    int crop_w,crop_h,ox,oy;
    int found=0;
    for(int attempt=0;attempt<10&&!found;attempt++){
        float scale=rng_range(rng,scale_lo,scale_hi);
        float ratio=expf(rng_range(rng,logf(ratio_lo),logf(ratio_hi)));
        int cw=(int)sqrtf((float)area*scale*ratio);
        int ch=(int)sqrtf((float)area*scale/ratio);
        if(cw>0&&ch>0&&cw<=W&&ch<=H){
            crop_w=cw; crop_h=ch;
            ox=rng_int(rng,W-cw+1);
            oy=rng_int(rng,H-ch+1);
            found=1;
        }
    }
    if(!found){
        /* fallback: center crop preserving ratio */
        float ratio=(float)out_w/out_h;
        if((float)W/H>ratio){ crop_h=H; crop_w=(int)(H*ratio); }
        else { crop_w=W; crop_h=(int)(W/ratio); }
        ox=(W-crop_w)/2; oy=(H-crop_h)/2;
    }
    VisionImage *cropped=vision_crop(src,ox,oy,crop_w,crop_h);
    if(!cropped)return NULL;
    VisionImage *resized=vision_resize(cropped,out_w,out_h,interp);
    vision_image_free(cropped);
    return resized;
}

/* ------------------------------------------------------------------ random flip */

VisionImage *vision_random_flip_horizontal(const VisionImage *src, float prob,
                                           VisionRNG *rng) {
    if(rng_float(rng)<prob) return vision_flip_horizontal(src);
    return vision_image_clone(src);
}

VisionImage *vision_random_flip_vertical(const VisionImage *src, float prob,
                                         VisionRNG *rng) {
    if(rng_float(rng)<prob) return vision_flip_vertical(src);
    return vision_image_clone(src);
}

/* ------------------------------------------------------------------ random brightness */

VisionImage *vision_random_brightness(const VisionImage *src,
                                      float max_delta, VisionRNG *rng) {
    float delta=rng_range(rng,-max_delta,max_delta);
    return vision_adjust_brightness(src,delta);
}

/* ------------------------------------------------------------------ random contrast */

VisionImage *vision_random_contrast(const VisionImage *src,
                                    float lo, float hi, VisionRNG *rng) {
    float factor=rng_range(rng,lo,hi);
    return vision_adjust_contrast(src,factor);
}

/* ------------------------------------------------------------------ random hue */

VisionImage *vision_random_hue(const VisionImage *src, float max_delta,
                               VisionRNG *rng) {
    float delta=rng_range(rng,-max_delta,max_delta);
    return vision_adjust_hue(src,delta);
}

/* ------------------------------------------------------------------ cutout */

VisionImage *vision_cutout(const VisionImage *src, int n_holes, int hole_size,
                           float fill_value, VisionRNG *rng) {
    if(!src){VISION_ERR("cutout: null");return NULL;}
    VisionImage *dst=vision_image_clone(src);
    if(!dst)return NULL;
    int W=src->width, H=src->height, C=src->channels;
    size_t elem=vision_element_size(src->format);

    for(int n=0;n<n_holes;n++){
        int cx=rng_int(rng,W), cy=rng_int(rng,H);
        int x0=clampi(cx-hole_size/2,0,W-1);
        int y0=clampi(cy-hole_size/2,0,H-1);
        int x1=clampi(cx+hole_size/2,0,W);
        int y1=clampi(cy+hole_size/2,0,H);
        for(int y=y0;y<y1;y++){
            uint8_t *row=dst->data+y*dst->stride;
            for(int x=x0;x<x1;x++){
                for(int c=0;c<C;c++){
                    if(src->format==VISION_FMT_FLOAT32)
                        ((float*)row)[x*C+c]=fill_value;
                    else
                        row[x*C+(int)(c*elem)]=(uint8_t)clampf(fill_value*255.f,0,255);
                }
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ mixup */

VisionImage *vision_mixup(const VisionImage *a, const VisionImage *b,
                          float alpha, VisionRNG *rng, float *out_lambda) {
    if(!a||!b){VISION_ERR("mixup: null");return NULL;}
    if(a->width!=b->width||a->height!=b->height||a->channels!=b->channels||
       a->format!=b->format){VISION_ERR("mixup: shape mismatch");return NULL;}

    /* sample lambda from Beta(alpha,alpha) via ratio of gammas approximation */
    /* simple approximation: lambda = rng in [alpha_lo, 1-alpha_lo] */
    float lo=alpha/(alpha+1.f); /* crude but avoids special function */
    float lam=(alpha>0)?clampf(rng_range(rng,0,1),lo,1-lo):0.5f;
    if(out_lambda)*out_lambda=lam;

    int W=a->width,H=a->height,C=a->channels;
    VisionImage *dst=vision_image_clone(a);
    if(!dst)return NULL;

    if(a->format==VISION_FMT_FLOAT32){
        #pragma omp parallel for schedule(static)
        for(int y=0;y<H;y++){
            float *dr=(float*)(dst->data+y*dst->stride);
            const float *br=(const float*)(b->data+y*b->stride);
            for(int x=0;x<W*C;x++) dr[x]=lam*dr[x]+(1-lam)*br[x];
        }
    } else {
        #pragma omp parallel for schedule(static)
        for(int y=0;y<H;y++){
            uint8_t *dr=dst->data+y*dst->stride;
            const uint8_t *br=b->data+y*b->stride;
            for(int x=0;x<W*C;x++){
                float v=lam*dr[x]+(1-lam)*br[x];
                dr[x]=(uint8_t)(v+0.5f);
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ cutmix */

VisionImage *vision_cutmix(const VisionImage *a, const VisionImage *b,
                           float alpha, VisionRNG *rng, float *out_lambda) {
    if(!a||!b){VISION_ERR("cutmix: null");return NULL;}
    if(a->width!=b->width||a->height!=b->height||a->channels!=b->channels||
       a->format!=b->format){VISION_ERR("cutmix: shape mismatch");return NULL;}

    float lam=(alpha>0)?clampf(rng_float(rng),alpha/(alpha+1.f),1-alpha/(alpha+1.f)):0.5f;
    int W=a->width,H=a->height;
    int cw=(int)(W*sqrtf(1-lam));
    int ch=(int)(H*sqrtf(1-lam));
    int cx=rng_int(rng,W), cy=rng_int(rng,H);
    int x0=clampi(cx-cw/2,0,W), x1=clampi(cx+cw/2,0,W);
    int y0=clampi(cy-ch/2,0,H), y1=clampi(cy+ch/2,0,H);
    float real_lam=1.f-(float)((x1-x0)*(y1-y0))/(float)(W*H);
    if(out_lambda)*out_lambda=real_lam;

    VisionImage *dst=vision_image_clone(a);
    if(!dst)return NULL;
    int C=a->channels; size_t elem=vision_element_size(a->format);
    for(int y=y0;y<y1;y++){
        const uint8_t *br=b->data+y*b->stride+x0*C*elem;
        uint8_t *dr=dst->data+y*dst->stride+x0*C*elem;
        memcpy(dr,br,(size_t)(x1-x0)*C*elem);
    }
    return dst;
}

/* ------------------------------------------------------------------ random rotation */

VisionImage *vision_random_rotation(const VisionImage *src,
                                    float max_angle_deg, VisionRNG *rng,
                                    VisionInterp interp, VisionBorderMode border,
                                    float fill_value) {
    float angle=rng_range(rng,-max_angle_deg,max_angle_deg);
    return vision_rotate(src,angle,interp,border,fill_value);
}
