#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

static inline float clampf(float v, float lo, float hi){
    return v<lo?lo:(v>hi?hi:v);
}
static inline int clampi(int v,int lo,int hi){
    return v<lo?lo:(v>hi?hi:v);
}

/* ------------------------------------------------------------------ generic 2D convolution */

VisionImage *vision_convolve2d(const VisionImage *src, const VisionKernel *kernel,
                               VisionBorderMode border) {
    if (!src || !kernel) { VISION_ERR("convolve2d: null"); return NULL; }
    if (src->format != VISION_FMT_FLOAT32) { VISION_ERR("convolve2d: needs float32"); return NULL; }
    int W=src->width,H=src->height,C=src->channels;
    int KH=kernel->height,KW=kernel->width;
    int padY=KH/2, padX=KW/2;
    VisionImage *dst=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!dst)return NULL;

    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        float *dr=(float*)(dst->data+y*dst->stride);
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                float acc=0;
                for(int ky=0;ky<KH;ky++){
                    int sy=y+ky-padY;
                    if(border==VISION_BORDER_REFLECT) sy=(sy<0)?-sy:(sy>=H?2*H-2-sy:sy);
                    else sy=clampi(sy,0,H-1);
                    const float *srow=(const float*)(src->data+sy*src->stride);
                    for(int kx=0;kx<KW;kx++){
                        int sx=x+kx-padX;
                        if(border==VISION_BORDER_REFLECT) sx=(sx<0)?-sx:(sx>=W?2*W-2-sx:sx);
                        else sx=clampi(sx,0,W-1);
                        acc+=kernel->data[ky*KW+kx]*srow[sx*C+c];
                    }
                }
                dr[x*C+c]=acc;
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ separable Gaussian */

static float *gaussian_kernel1d(int radius, float sigma) {
    int size=2*radius+1;
    float *k=(float*)malloc(size*sizeof(float));
    if(!k)return NULL;
    float sum=0;
    for(int i=0;i<size;i++){
        float x=(float)(i-radius);
        k[i]=expf(-x*x/(2*sigma*sigma));
        sum+=k[i];
    }
    for(int i=0;i<size;i++) k[i]/=sum;
    return k;
}

VisionImage *vision_gaussian_blur(const VisionImage *src, int radius, float sigma) {
    if(!src||radius<1){VISION_ERR("gaussian_blur: bad args");return NULL;}
    if(src->format!=VISION_FMT_FLOAT32){VISION_ERR("gaussian_blur: needs float32");return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    float *k=gaussian_kernel1d(radius,sigma);
    if(!k)return NULL;
    int ksize=2*radius+1;

    /* horizontal pass */
    VisionImage *tmp=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!tmp){free(k);return NULL;}
    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        const float *sr=(const float*)(src->data+y*src->stride);
        float *dr=(float*)(tmp->data+y*tmp->stride);
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                float acc=0;
                for(int i=0;i<ksize;i++){
                    int sx=clampi(x+i-radius,0,W-1);
                    acc+=k[i]*sr[sx*C+c];
                }
                dr[x*C+c]=acc;
            }
        }
    }
    /* vertical pass */
    VisionImage *dst=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!dst){free(k);vision_image_free(tmp);return NULL;}
    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        float *dr=(float*)(dst->data+y*dst->stride);
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                float acc=0;
                for(int i=0;i<ksize;i++){
                    int sy=clampi(y+i-radius,0,H-1);
                    const float *tr=(const float*)(tmp->data+sy*tmp->stride);
                    acc+=k[i]*tr[x*C+c];
                }
                dr[x*C+c]=acc;
            }
        }
    }
    free(k);
    vision_image_free(tmp);
    return dst;
}

/* ------------------------------------------------------------------ box blur (integral image) */

VisionImage *vision_box_blur(const VisionImage *src, int radius) {
    if(!src||radius<1){VISION_ERR("box_blur: bad args");return NULL;}
    if(src->format!=VISION_FMT_FLOAT32){VISION_ERR("box_blur: needs float32");return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    /* build integral image */
    double *integral=(double*)calloc((size_t)(H+1)*(W+1)*C,sizeof(double));
    if(!integral)return NULL;
    for(int y=0;y<H;y++){
        const float *sr=(const float*)(src->data+y*src->stride);
        for(int x=0;x<W;x++)
            for(int c=0;c<C;c++){
                integral[((y+1)*(W+1)+x+1)*C+c]=
                    sr[x*C+c]
                    +integral[(y*(W+1)+x+1)*C+c]
                    +integral[((y+1)*(W+1)+x)*C+c]
                    -integral[(y*(W+1)+x)*C+c];
            }
    }
    VisionImage *dst=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!dst){free(integral);return NULL;}
    float area_inv=1.f/((2*radius+1)*(2*radius+1));
    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        float *dr=(float*)(dst->data+y*dst->stride);
        int y1=clampi(y+radius+1,0,H), y0=clampi(y-radius,0,H);
        for(int x=0;x<W;x++){
            int x1=clampi(x+radius+1,0,W), x0=clampi(x-radius,0,W);
            float area=(float)((y1-y0)*(x1-x0));
            for(int c=0;c<C;c++){
                double s=integral[(y1*(W+1)+x1)*C+c]
                        -integral[(y0*(W+1)+x1)*C+c]
                        -integral[(y1*(W+1)+x0)*C+c]
                        +integral[(y0*(W+1)+x0)*C+c];
                dr[x*C+c]=(float)(s/area);
            }
        }
    }
    free(integral);
    return dst;
}

/* ------------------------------------------------------------------ median blur */

static int cmp_float(const void *a,const void *b){
    float fa=*(const float*)a, fb=*(const float*)b;
    return fa<fb?-1:fa>fb?1:0;
}

VisionImage *vision_median_blur(const VisionImage *src, int radius) {
    if(!src||radius<1){VISION_ERR("median_blur: bad args");return NULL;}
    if(src->format!=VISION_FMT_FLOAT32){VISION_ERR("median_blur: needs float32");return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    int ksize=(2*radius+1)*(2*radius+1);
    float *buf=(float*)malloc(ksize*sizeof(float));
    if(!buf)return NULL;
    VisionImage *dst=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!dst){free(buf);return NULL;}
    for(int y=0;y<H;y++){
        float *dr=(float*)(dst->data+y*dst->stride);
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                int cnt=0;
                for(int ky=-radius;ky<=radius;ky++){
                    int sy=clampi(y+ky,0,H-1);
                    const float *sr=(const float*)(src->data+sy*src->stride);
                    for(int kx=-radius;kx<=radius;kx++){
                        int sx=clampi(x+kx,0,W-1);
                        buf[cnt++]=sr[sx*C+c];
                    }
                }
                qsort(buf,cnt,sizeof(float),cmp_float);
                dr[x*C+c]=buf[cnt/2];
            }
        }
    }
    free(buf);
    return dst;
}

/* ------------------------------------------------------------------ Sobel */

VisionImage *vision_sobel(const VisionImage *src,
                          VisionImage **out_gx, VisionImage **out_gy) {
    if(!src||src->format!=VISION_FMT_FLOAT32){VISION_ERR("sobel: needs float32");return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    VisionImage *gx=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    VisionImage *gy=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    VisionImage *mag=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!gx||!gy||!mag){
        vision_image_free(gx);vision_image_free(gy);vision_image_free(mag);return NULL;
    }
    /* Kx = [-1 0 1; -2 0 2; -1 0 1]  Ky = [-1 -2 -1; 0 0 0; 1 2 1] */
    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        float *drx=(float*)(gx->data+y*gx->stride);
        float *dry=(float*)(gy->data+y*gy->stride);
        float *drm=(float*)(mag->data+y*mag->stride);
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                /* fetch 3x3 neighbourhood */
                float p[3][3];
                for(int ky=-1;ky<=1;ky++){
                    int sy=clampi(y+ky,0,H-1);
                    const float *sr=(const float*)(src->data+sy*src->stride);
                    for(int kx=-1;kx<=1;kx++){
                        int sx=clampi(x+kx,0,W-1);
                        p[ky+1][kx+1]=sr[sx*C+c];
                    }
                }
                float vx=(-p[0][0]+p[0][2])+(-2*p[1][0]+2*p[1][2])+(-p[2][0]+p[2][2]);
                float vy=(-p[0][0]-2*p[0][1]-p[0][2])+(p[2][0]+2*p[2][1]+p[2][2]);
                drx[x*C+c]=vx; dry[x*C+c]=vy;
                drm[x*C+c]=sqrtf(vx*vx+vy*vy);
            }
        }
    }
    if(out_gx)*out_gx=gx; else vision_image_free(gx);
    if(out_gy)*out_gy=gy; else vision_image_free(gy);
    return mag;
}

/* ------------------------------------------------------------------ Laplacian */

VisionImage *vision_laplacian(const VisionImage *src) {
    if(!src||src->format!=VISION_FMT_FLOAT32){VISION_ERR("laplacian: needs float32");return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    VisionImage *dst=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!dst)return NULL;
    /* kernel [0 1 0; 1 -4 1; 0 1 0] */
    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        float *dr=(float*)(dst->data+y*dst->stride);
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                int yn=clampi(y-1,0,H-1),yp=clampi(y+1,0,H-1);
                int xn=clampi(x-1,0,W-1),xp=clampi(x+1,0,W-1);
                const float *r0=(const float*)(src->data+yn*src->stride);
                const float *r1=(const float*)(src->data+y*src->stride);
                const float *r2=(const float*)(src->data+yp*src->stride);
                dr[x*C+c]=r0[x*C+c]+r1[xn*C+c]-4*r1[x*C+c]+r1[xp*C+c]+r2[x*C+c];
            }
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ Canny */

VisionImage *vision_canny(const VisionImage *src,
                          float low_thresh, float high_thresh,
                          int gaussian_radius, float gaussian_sigma) {
    if(!src){VISION_ERR("canny: null");return NULL;}
    /* ensure float32 single channel */
    VisionImage *gray=NULL;
    int free_gray=0;
    if(src->channels>1){gray=vision_to_grayscale(src);free_gray=1;}
    else if(src->format!=VISION_FMT_FLOAT32){
        gray=vision_to_float32(src);free_gray=1;
    } else {gray=(VisionImage*)src;}

    VisionImage *blurred=vision_gaussian_blur(gray,gaussian_radius,gaussian_sigma);
    if(free_gray)vision_image_free(gray);
    if(!blurred)return NULL;

    VisionImage *gxImg=NULL,*gyImg=NULL;
    VisionImage *mag=vision_sobel(blurred,&gxImg,&gyImg);
    vision_image_free(blurred);
    if(!mag){return NULL;}

    int W=mag->width,H=mag->height;
    /* Non-max suppression + hysteresis on single-channel float */
    VisionImage *nms=vision_image_create(W,H,1,VISION_FMT_FLOAT32,
                                         VISION_LAYOUT_HWC,VISION_CS_GRAY);
    if(!nms){vision_image_free(mag);vision_image_free(gxImg);vision_image_free(gyImg);return NULL;}

    const float *MG=(const float*)mag->data;
    const float *GX=(gxImg)?(const float*)gxImg->data:NULL;
    const float *GY=(gyImg)?(const float*)gyImg->data:NULL;
    float *NM=(float*)nms->data;
    size_t ms=mag->stride/sizeof(float), ns=nms->stride/sizeof(float);

    /* NMS */
    for(int y=1;y<H-1;y++){
        for(int x=1;x<W-1;x++){
            float gx=GX?GX[y*ms+x]:0, gy=GY?GY[y*ms+x]:0;
            float angle=atan2f(gy,gx)*180.f/3.14159265f;
            if(angle<0)angle+=180.f;
            float m=MG[y*ms+x];
            float m1,m2;
            if((angle<22.5f)||(angle>=157.5f)){
                m1=MG[y*ms+x-1]; m2=MG[y*ms+x+1];
            } else if(angle<67.5f){
                m1=MG[(y-1)*ms+x+1]; m2=MG[(y+1)*ms+x-1];
            } else if(angle<112.5f){
                m1=MG[(y-1)*ms+x]; m2=MG[(y+1)*ms+x];
            } else {
                m1=MG[(y-1)*ms+x-1]; m2=MG[(y+1)*ms+x+1];
            }
            NM[y*ns+x]=(m>=m1&&m>=m2)?m:0.f;
        }
    }
    vision_image_free(mag);
    if(gxImg)vision_image_free(gxImg);
    if(gyImg)vision_image_free(gyImg);

    /* double threshold + hysteresis */
    VisionImage *out=vision_image_create(W,H,1,VISION_FMT_UINT8,
                                          VISION_LAYOUT_HWC,VISION_CS_GRAY);
    if(!out){vision_image_free(nms);return NULL;}
    uint8_t *ED=out->data;
    size_t os=out->stride;
    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            float v=NM[y*ns+x];
            if(v>=high_thresh) ED[y*os+x]=255;
            else if(v>=low_thresh) ED[y*os+x]=128; /* weak */
            else ED[y*os+x]=0;
        }
    }
    /* hysteresis: promote weak if connected to strong */
    int changed=1;
    while(changed){
        changed=0;
        for(int y=1;y<H-1;y++)
            for(int x=1;x<W-1;x++)
                if(ED[y*os+x]==128){
                    int strong=0;
                    for(int dy=-1;dy<=1&&!strong;dy++)
                        for(int dx=-1;dx<=1;dx++)
                            if(ED[(y+dy)*os+x+dx]==255){strong=1;break;}
                    if(strong){ED[y*os+x]=255;changed=1;}
                }
        for(int y=0;y<H;y++)
            for(int x=0;x<W;x++)
                if(ED[y*os+x]==128) ED[y*os+x]=0; /* suppress remaining weak */
    }
    vision_image_free(nms);
    return out;
}

/* ------------------------------------------------------------------ morphology helpers */

static VisionImage *morph_op(const VisionImage *src, int radius, int dilate) {
    if(!src||src->format!=VISION_FMT_UINT8){VISION_ERR("morph: needs uint8");return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    VisionImage *dst=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!dst)return NULL;
    #pragma omp parallel for schedule(static)
    for(int y=0;y<H;y++){
        uint8_t *dr=dst->data+y*dst->stride;
        for(int x=0;x<W;x++){
            for(int c=0;c<C;c++){
                uint8_t best=dilate?0:255;
                for(int ky=-radius;ky<=radius;ky++){
                    int sy=clampi(y+ky,0,H-1);
                    const uint8_t *sr=src->data+sy*src->stride;
                    for(int kx=-radius;kx<=radius;kx++){
                        int sx=clampi(x+kx,0,W-1);
                        uint8_t v=sr[sx*C+c];
                        if(dilate){if(v>best)best=v;}
                        else{if(v<best)best=v;}
                    }
                }
                dr[x*C+c]=best;
            }
        }
    }
    return dst;
}

VisionImage *vision_erode(const VisionImage *src, int radius) {
    return morph_op(src,radius,0);
}
VisionImage *vision_dilate(const VisionImage *src, int radius) {
    return morph_op(src,radius,1);
}
VisionImage *vision_morph_open(const VisionImage *src, int radius) {
    VisionImage *t=vision_erode(src,radius); if(!t)return NULL;
    VisionImage *out=vision_dilate(t,radius); vision_image_free(t); return out;
}
VisionImage *vision_morph_close(const VisionImage *src, int radius) {
    VisionImage *t=vision_dilate(src,radius); if(!t)return NULL;
    VisionImage *out=vision_erode(t,radius); vision_image_free(t); return out;
}
VisionImage *vision_morph_gradient(const VisionImage *src, int radius) {
    VisionImage *d=vision_dilate(src,radius); if(!d)return NULL;
    VisionImage *e=vision_erode(src,radius);
    if(!e){vision_image_free(d);return NULL;}
    int W=src->width,H=src->height,C=src->channels;
    VisionImage *out=vision_image_create(W,H,C,src->format,src->layout,src->color_space);
    if(!out){vision_image_free(d);vision_image_free(e);return NULL;}
    for(int y=0;y<H;y++){
        const uint8_t *dr=d->data+y*d->stride;
        const uint8_t *er=e->data+y*e->stride;
        uint8_t *or_=out->data+y*out->stride;
        for(int x=0;x<W*C;x++) or_[x]=dr[x]-er[x];
    }
    vision_image_free(d); vision_image_free(e);
    return out;
}
