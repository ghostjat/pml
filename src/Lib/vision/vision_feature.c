#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifdef __AVX2__
#include <immintrin.h>
#endif

static inline int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}
static inline float clampf(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}

/* ------------------------------------------------------------------ integral image */

void vision_integral_image(const VisionImage *src, double *integral) {
    /* integral is caller-allocated: (H+1)*(W+1) doubles, row-major */
    int W=src->width, H=src->height;
    memset(integral, 0, (size_t)(H+1)*(W+1)*sizeof(double));
    for(int y=0;y<H;y++){
        double row_sum=0;
        const float *sr=(const float*)(src->data+y*src->stride);
        for(int x=0;x<W;x++){
            row_sum+=sr[x]; /* single-channel only */
            integral[(y+1)*(W+1)+(x+1)]=row_sum+integral[y*(W+1)+(x+1)];
        }
    }
}

/* ------------------------------------------------------------------ HOG (Dalal-Triggs) */

HOGResult *vision_hog(const VisionImage *src, int cell_size, int block_size,
                      int nbins, int *out_len) {
    if(!src||src->format!=VISION_FMT_FLOAT32||src->channels<1){
        VISION_ERR("hog: needs float32");return NULL;
    }
    /* work on grayscale */
    VisionImage *gray=NULL;
    int free_gray=0;
    if(src->channels>1){gray=vision_to_grayscale(src);free_gray=1;}
    else{gray=(VisionImage*)src;}
    if(!gray)return NULL;

    int W=gray->width,H=gray->height;
    int cells_x=W/cell_size, cells_y=H/cell_size;
    /* gradient magnitudes and angles */
    float *mag=(float*)calloc((size_t)H*W,sizeof(float));
    float *ang=(float*)calloc((size_t)H*W,sizeof(float));
    if(!mag||!ang){free(mag);free(ang);if(free_gray)vision_image_free(gray);return NULL;}

    #pragma omp parallel for schedule(static)
    for(int y=1;y<H-1;y++){
        const float *r0=(const float*)(gray->data+(y-1)*gray->stride);
        const float *r2=(const float*)(gray->data+(y+1)*gray->stride);
        const float *r1=(const float*)(gray->data+y*gray->stride);
        for(int x=1;x<W-1;x++){
            float dx=r1[x+1]-r1[x-1];
            float dy=r2[x]-r0[x];
            mag[y*W+x]=sqrtf(dx*dx+dy*dy);
            float a=atan2f(dy,dx)*180.f/3.14159265f;
            if(a<0)a+=180.f;
            ang[y*W+x]=a;
        }
    }

    /* cell histograms */
    float *cell_hist=(float*)calloc((size_t)cells_y*cells_x*nbins,sizeof(float));
    if(!cell_hist){free(mag);free(ang);if(free_gray)vision_image_free(gray);return NULL;}

    for(int cy=0;cy<cells_y;cy++){
        for(int cx=0;cx<cells_x;cx++){
            float *h=cell_hist+(cy*cells_x+cx)*nbins;
            for(int iy=0;iy<cell_size;iy++){
                int py=cy*cell_size+iy;
                if(py>=H)continue;
                for(int ix=0;ix<cell_size;ix++){
                    int px=cx*cell_size+ix;
                    if(px>=W)continue;
                    float m=mag[py*W+px];
                    float a=ang[py*W+px];
                    /* bilinear interpolation between bins */
                    float bin_f=a/(180.f/nbins);
                    int bin0=(int)bin_f%nbins;
                    int bin1=(bin0+1)%nbins;
                    float w1=bin_f-floorf(bin_f);
                    h[bin0]+=m*(1-w1);
                    h[bin1]+=m*w1;
                }
            }
        }
    }
    free(mag); free(ang);
    if(free_gray)vision_image_free(gray);

    /* block normalization: block_size × block_size cells, stride=1 */
    int blocks_x=clampi(cells_x-block_size+1,0,cells_x);
    int blocks_y=clampi(cells_y-block_size+1,0,cells_y);
    int feat_per_block=block_size*block_size*nbins;
    int total=blocks_y*blocks_x*feat_per_block;

    HOGResult *res=(HOGResult*)malloc(sizeof(HOGResult));
    if(!res){free(cell_hist);return NULL;}
    res->descriptors=(float*)calloc((size_t)total,sizeof(float));
    if(!res->descriptors){free(res);free(cell_hist);return NULL;}
    res->length=total;

    float *out=res->descriptors;
    for(int by=0;by<blocks_y;by++){
        for(int bx=0;bx<blocks_x;bx++){
            float *blk=out+(by*blocks_x+bx)*feat_per_block;
            float norm_sq=0;
            for(int dy=0;dy<block_size;dy++)
                for(int dx=0;dx<block_size;dx++){
                    const float *h=cell_hist+((by+dy)*cells_x+(bx+dx))*nbins;
                    float *dst=blk+(dy*block_size+dx)*nbins;
                    for(int b=0;b<nbins;b++){dst[b]=h[b]; norm_sq+=h[b]*h[b];}
                }
            float inv=1.f/sqrtf(norm_sq+1e-6f);
            for(int k=0;k<feat_per_block;k++) blk[k]*=inv;
        }
    }
    free(cell_hist);
    if(out_len)*out_len=total;
    return res;
}

void vision_hog_free(HOGResult *r){
    if(r){free(r->descriptors);free(r);}
}

/* ------------------------------------------------------------------ LBP */

LBPResult *vision_lbp(const VisionImage *src, int radius, int grid_x, int grid_y,
                      int *out_len) {
    if(!src||src->format!=VISION_FMT_FLOAT32){VISION_ERR("lbp: needs float32");return NULL;}
    VisionImage *gray=NULL; int free_g=0;
    if(src->channels>1){gray=vision_to_grayscale(src);free_g=1;}
    else{gray=(VisionImage*)src;}
    if(!gray)return NULL;

    int W=gray->width,H=gray->height;
    int npoints=8; /* 8 neighbors */
    int nuniform=58+2; /* 58 uniform + 1 non-uniform + 1 extra for safety */
    int hist_size=npoints+2; /* simplified: 59-bin */
    hist_size=60;
    int total=grid_x*grid_y*hist_size;

    LBPResult *res=(LBPResult*)malloc(sizeof(LBPResult));
    if(!res){if(free_g)vision_image_free(gray);return NULL;}
    res->histogram=(float*)calloc((size_t)total,sizeof(float));
    if(!res->histogram){free(res);if(free_g)vision_image_free(gray);return NULL;}
    res->length=total;

    int cell_w=W/grid_x, cell_h=H/grid_y;
    for(int gy=0;gy<grid_y;gy++){
        for(int gx=0;gx<grid_x;gx++){
            float *h=res->histogram+(gy*grid_x+gx)*hist_size;
            int x0=gx*cell_w, y0=gy*cell_h;
            int x1=x0+cell_w, y1=y0+cell_h;
            if(x1>W)x1=W; if(y1>H)y1=H;
            for(int y=y0+radius;y<y1-radius;y++){
                const float *row=(const float*)(gray->data+y*gray->stride);
                for(int x=x0+radius;x<x1-radius;x++){
                    float center=row[x];
                    uint8_t code=0;
                    for(int n=0;n<npoints;n++){
                        float angle=2.f*3.14159265f*n/npoints;
                        float nx=x+radius*cosf(angle);
                        float ny=y-radius*sinf(angle);
                        int ix=(int)nx, iy=(int)ny;
                        float wx=nx-ix, wy=ny-iy;
                        ix=clampi(ix,0,W-1); iy=clampi(iy,0,H-1);
                        int ix1=clampi(ix+1,0,W-1), iy1=clampi(iy+1,0,H-1);
                        const float *r0=(const float*)(gray->data+iy*gray->stride);
                        const float *r1=(const float*)(gray->data+iy1*gray->stride);
                        float v=(r0[ix]*(1-wx)+r0[ix1]*wx)*(1-wy)
                               +(r1[ix]*(1-wx)+r1[ix1]*wx)*wy;
                        if(v>=center) code|=(1<<n);
                    }
                    /* count transitions (uniform) */
                    int transitions=0;
                    uint8_t prev=(code>>7)&1;
                    for(int n=0;n<npoints;n++){
                        int cur=(code>>n)&1;
                        if(cur!=prev)transitions++;
                        prev=cur;
                    }
                    int bin=(transitions<=2)?__builtin_popcount(code):npoints+1;
                    if(bin<hist_size) h[bin]+=1.f;
                }
            }
            /* normalize */
            float sum=0; for(int b=0;b<hist_size;b++) sum+=h[b];
            if(sum>0) for(int b=0;b<hist_size;b++) h[b]/=sum;
        }
    }
    if(free_g)vision_image_free(gray);
    if(out_len)*out_len=total;
    return res;
}

void vision_lbp_free(LBPResult *r){
    if(r){free(r->histogram);free(r);}
}

/* ------------------------------------------------------------------ Harris corners */

VisionCorners *vision_harris_corners(const VisionImage *src,
                                     float k, float threshold, int nms_radius,
                                     int *out_count) {
    if(!src||src->format!=VISION_FMT_FLOAT32){VISION_ERR("harris: needs float32");return NULL;}
    VisionImage *gray=NULL; int fg=0;
    if(src->channels>1){gray=vision_to_grayscale(src);fg=1;}
    else{gray=(VisionImage*)src;}
    if(!gray)return NULL;

    int W=gray->width,H=gray->height;
    float *Ix2=(float*)calloc((size_t)H*W,sizeof(float));
    float *Iy2=(float*)calloc((size_t)H*W,sizeof(float));
    float *Ixy=(float*)calloc((size_t)H*W,sizeof(float));
    float *R  =(float*)calloc((size_t)H*W,sizeof(float));
    if(!Ix2||!Iy2||!Ixy||!R){
        free(Ix2);free(Iy2);free(Ixy);free(R);
        if(fg)vision_image_free(gray);return NULL;
    }

    for(int y=1;y<H-1;y++){
        const float *r0=(const float*)(gray->data+(y-1)*gray->stride);
        const float *r1=(const float*)(gray->data+y*gray->stride);
        const float *r2=(const float*)(gray->data+(y+1)*gray->stride);
        for(int x=1;x<W-1;x++){
            float dx=r1[x+1]-r1[x-1];
            float dy=r2[x]-r0[x];
            Ix2[y*W+x]=dx*dx;
            Iy2[y*W+x]=dy*dy;
            Ixy[y*W+x]=dx*dy;
        }
    }
    /* box blur with radius=2 */
    int br=2;
    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            float sx2=0,sy2=0,sxy=0;
            for(int dy=-br;dy<=br;dy++)
                for(int dx=-br;dx<=br;dx++){
                    int sy=clampi(y+dy,0,H-1), sx=clampi(x+dx,0,W-1);
                    sx2+=Ix2[sy*W+sx]; sy2+=Iy2[sy*W+sx]; sxy+=Ixy[sy*W+sx];
                }
            float det=sx2*sy2-sxy*sxy;
            float trace=sx2+sy2;
            R[y*W+x]=det-k*trace*trace;
        }
    }
    free(Ix2);free(Iy2);free(Ixy);
    if(fg)vision_image_free(gray);

    /* count above threshold */
    int cap=1024; int cnt=0;
    float *xs=(float*)malloc(cap*sizeof(float));
    float *ys=(float*)malloc(cap*sizeof(float));
    float *sc=(float*)malloc(cap*sizeof(float));
    if(!xs||!ys||!sc){free(xs);free(ys);free(sc);free(R);return NULL;}

    for(int y=nms_radius;y<H-nms_radius;y++){
        for(int x=nms_radius;x<W-nms_radius;x++){
            float rv=R[y*W+x];
            if(rv<threshold) continue;
            /* local NMS */
            int is_max=1;
            for(int dy=-nms_radius;dy<=nms_radius&&is_max;dy++)
                for(int dx=-nms_radius;dx<=nms_radius;dx++)
                    if(!(dy==0&&dx==0)&&R[(y+dy)*W+(x+dx)]>=rv){is_max=0;break;}
            if(!is_max)continue;
            if(cnt>=cap){
                cap*=2;
                xs=(float*)realloc(xs,cap*sizeof(float));
                ys=(float*)realloc(ys,cap*sizeof(float));
                sc=(float*)realloc(sc,cap*sizeof(float));
                if(!xs||!ys||!sc){free(xs);free(ys);free(sc);free(R);return NULL;}
            }
            xs[cnt]=(float)x; ys[cnt]=(float)y; sc[cnt]=rv; cnt++;
        }
    }
    free(R);

    VisionCorners *res=(VisionCorners*)malloc(sizeof(VisionCorners));
    if(!res){free(xs);free(ys);free(sc);return NULL;}
    res->x=xs; res->y=ys; res->score=sc; res->count=cnt;
    if(out_count)*out_count=cnt;
    return res;
}

void vision_corners_free(VisionCorners *c){
    if(c){free(c->x);free(c->y);free(c->score);free(c);}
}

/* ------------------------------------------------------------------ FAST-9 */

static const int FAST_CIRCLE_X[16]={0,1,2,3,3,3,2,1,0,-1,-2,-3,-3,-3,-2,-1};
static const int FAST_CIRCLE_Y[16]={-3,-3,-2,-1,0,1,2,3,3,3,2,1,0,-1,-2,-3};

VisionCorners *vision_fast_corners(const VisionImage *src, int threshold,
                                   int n_consecutive, int *out_count) {
    if(!src||src->format!=VISION_FMT_FLOAT32){VISION_ERR("fast: needs float32");return NULL;}
    VisionImage *gray=NULL; int fg=0;
    if(src->channels>1){gray=vision_to_grayscale(src);fg=1;}
    else{gray=(VisionImage*)src;}
    if(!gray)return NULL;

    int W=gray->width,H=gray->height;
    int cap=1024,cnt=0;
    float *xs=(float*)malloc(cap*sizeof(float));
    float *ys=(float*)malloc(cap*sizeof(float));
    float *sc=(float*)malloc(cap*sizeof(float));
    if(!xs||!ys||!sc){free(xs);free(ys);free(sc);if(fg)vision_image_free(gray);return NULL;}

    float thresh_f=(float)threshold/255.f;
    for(int y=3;y<H-3;y++){
        const float *row=(const float*)(gray->data+y*gray->stride);
        for(int x=3;x<W-3;x++){
            float ctr=row[x];
            float lo=ctr-thresh_f, hi=ctr+thresh_f;
            /* fast rejection: check N,S,E,W first */
            int brighter=0, darker=0;
            for(int k=0;k<16;k++){
                int cx=x+FAST_CIRCLE_X[k], cy=y+FAST_CIRCLE_Y[k];
                cx=clampi(cx,0,W-1); cy=clampi(cy,0,H-1);
                float v=((const float*)(gray->data+cy*gray->stride))[cx];
                if(v>hi) brighter++;
                else if(v<lo) darker++;
            }
            if(brighter<n_consecutive&&darker<n_consecutive) continue;
            /* full test */
            int consecutive=0,best_consec=0;
            for(int k=0;k<32;k++){
                int idx=k%16;
                int cx=x+FAST_CIRCLE_X[idx], cy=y+FAST_CIRCLE_Y[idx];
                cx=clampi(cx,0,W-1); cy=clampi(cy,0,H-1);
                float v=((const float*)(gray->data+cy*gray->stride))[cx];
                if((brighter>=n_consecutive&&v>hi)||(darker>=n_consecutive&&v<lo))
                    consecutive++;
                else consecutive=0;
                if(consecutive>best_consec)best_consec=consecutive;
            }
            if(best_consec>=n_consecutive){
                if(cnt>=cap){cap*=2;xs=(float*)realloc(xs,cap*sizeof(float));ys=(float*)realloc(ys,cap*sizeof(float));sc=(float*)realloc(sc,cap*sizeof(float));}
                xs[cnt]=(float)x; ys[cnt]=(float)y; sc[cnt]=(float)best_consec; cnt++;
            }
        }
    }
    if(fg)vision_image_free(gray);
    VisionCorners *res=(VisionCorners*)malloc(sizeof(VisionCorners));
    if(!res){free(xs);free(ys);free(sc);return NULL;}
    res->x=xs;res->y=ys;res->score=sc;res->count=cnt;
    if(out_count)*out_count=cnt;
    return res;
}
