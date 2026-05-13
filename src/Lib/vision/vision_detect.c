#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static inline float clampf(float v,float lo,float hi){return v<lo?lo:(v>hi?hi:v);}

/* ------------------------------------------------------------------ IoU */

float vision_iou(const VisionBBox *a, const VisionBBox *b) {
    float ix0=a->x1>b->x1?a->x1:b->x1, iy0=a->y1>b->y1?a->y1:b->y1;
    float ix1=a->x2<b->x2?a->x2:b->x2, iy1=a->y2<b->y2?a->y2:b->y2;
    float iw=ix1-ix0, ih=iy1-iy0;
    if(iw<=0||ih<=0)return 0.f;
    float inter=iw*ih;
    float ua=(a->x2-a->x1)*(a->y2-a->y1);
    float ub=(b->x2-b->x1)*(b->y2-b->y1);
    return inter/(ua+ub-inter+1e-6f);
}

float vision_giou(const VisionBBox *a, const VisionBBox *b) {
    float iou=vision_iou(a,b);
    float cx0=a->x1<b->x1?a->x1:b->x1, cy0=a->y1<b->y1?a->y1:b->y1;
    float cx1=a->x2>b->x2?a->x2:b->x2, cy1=a->y2>b->y2?a->y2:b->y2;
    float C=(cx1-cx0)*(cy1-cy0);
    float ua=(a->x2-a->x1)*(a->y2-a->y1);
    float ub=(b->x2-b->x1)*(b->y2-b->y1);
    return iou-(C-(ua+ub-iou*((ua+ub)/(C+1e-6f))))/(C+1e-6f);
}

float vision_diou(const VisionBBox *a, const VisionBBox *b) {
    float iou=vision_iou(a,b);
    float cax=(a->x1+a->x2)*0.5f, cay=(a->y1+a->y2)*0.5f;
    float cbx=(b->x1+b->x2)*0.5f, cby=(b->y1+b->y2)*0.5f;
    float d2=(cax-cbx)*(cax-cbx)+(cay-cby)*(cay-cby);
    float cx0=a->x1<b->x1?a->x1:b->x1, cy0=a->y1<b->y1?a->y1:b->y1;
    float cx1=a->x2>b->x2?a->x2:b->x2, cy1=a->y2>b->y2?a->y2:b->y2;
    float diag=(cx1-cx0)*(cx1-cx0)+(cy1-cy0)*(cy1-cy0);
    return iou-d2/(diag+1e-6f);
}

/* ------------------------------------------------------------------ BBoxArray lifecycle */

VisionBBoxArray *vision_bbox_array_create(int capacity) {
    VisionBBoxArray *arr=(VisionBBoxArray*)malloc(sizeof(VisionBBoxArray));
    if(!arr)return NULL;
    arr->boxes=(VisionBBox*)malloc((size_t)capacity*sizeof(VisionBBox));
    if(!arr->boxes){free(arr);return NULL;}
    arr->count=0; arr->capacity=capacity;
    return arr;
}

void vision_bbox_array_free(VisionBBoxArray *arr) {
    if(arr){free(arr->boxes);free(arr);}
}

int vision_bbox_array_push(VisionBBoxArray *arr, const VisionBBox *box) {
    if(arr->count>=arr->capacity){
        int nc=arr->capacity*2;
        VisionBBox *nb=(VisionBBox*)realloc(arr->boxes,(size_t)nc*sizeof(VisionBBox));
        if(!nb)return -1;
        arr->boxes=nb; arr->capacity=nc;
    }
    arr->boxes[arr->count++]=*box;
    return 0;
}

/* ------------------------------------------------------------------ NMS */

static int cmp_score_desc(const void *a,const void *b){
    float fa=((const VisionBBox*)a)->score, fb=((const VisionBBox*)b)->score;
    return fa>fb?-1:fa<fb?1:0;
}

VisionBBoxArray *vision_nms(const VisionBBoxArray *boxes, float iou_thresh) {
    if(!boxes||boxes->count==0) return vision_bbox_array_create(0);
    int n=boxes->count;
    VisionBBox *sorted=(VisionBBox*)malloc((size_t)n*sizeof(VisionBBox));
    if(!sorted)return NULL;
    memcpy(sorted,boxes->boxes,(size_t)n*sizeof(VisionBBox));
    qsort(sorted,n,sizeof(VisionBBox),cmp_score_desc);

    uint8_t *suppressed=(uint8_t*)calloc(n,1);
    VisionBBoxArray *out=vision_bbox_array_create(n);
    if(!suppressed||!out){free(suppressed);free(sorted);vision_bbox_array_free(out);return NULL;}

    for(int i=0;i<n;i++){
        if(suppressed[i])continue;
        vision_bbox_array_push(out,&sorted[i]);
        for(int j=i+1;j<n;j++){
            if(!suppressed[j]&&vision_iou(&sorted[i],&sorted[j])>iou_thresh)
                suppressed[j]=1;
        }
    }
    free(suppressed); free(sorted);
    return out;
}

VisionBBoxArray *vision_soft_nms(const VisionBBoxArray *boxes,
                                 float sigma, float score_thresh) {
    if(!boxes||boxes->count==0) return vision_bbox_array_create(0);
    int n=boxes->count;
    VisionBBox *work=(VisionBBox*)malloc((size_t)n*sizeof(VisionBBox));
    if(!work)return NULL;
    memcpy(work,boxes->boxes,(size_t)n*sizeof(VisionBBox));

    VisionBBoxArray *out=vision_bbox_array_create(n);
    if(!out){free(work);return NULL;}

    /* Gaussian soft-NMS: iteratively pick max, decay others */
    for(int iter=0;iter<n;iter++){
        /* find max score */
        int mi=iter;
        for(int j=iter+1;j<n;j++) if(work[j].score>work[mi].score) mi=j;
        if(work[mi].score<score_thresh) break;
        /* swap to front */
        VisionBBox tmp=work[iter]; work[iter]=work[mi]; work[mi]=tmp;
        vision_bbox_array_push(out,&work[iter]);
        /* decay */
        for(int j=iter+1;j<n;j++){
            float iou=vision_iou(&work[iter],&work[j]);
            work[j].score*=expf(-iou*iou/sigma);
        }
    }
    free(work);
    return out;
}

/* ------------------------------------------------------------------ anchor generation */

VisionBBoxArray *vision_generate_anchors(int feat_w, int feat_h,
                                         int stride,
                                         const float *scales, int n_scales,
                                         const float *ratios, int n_ratios) {
    int total=feat_w*feat_h*n_scales*n_ratios;
    VisionBBoxArray *arr=vision_bbox_array_create(total);
    if(!arr)return NULL;

    for(int fy=0;fy<feat_h;fy++){
        for(int fx=0;fx<feat_w;fx++){
            float cx=(fx+0.5f)*stride, cy=(fy+0.5f)*stride;
            for(int si=0;si<n_scales;si++){
                float s=scales[si];
                for(int ri=0;ri<n_ratios;ri++){
                    float r=ratios[ri];
                    float w=s*sqrtf(1.f/r)*stride;
                    float h=s*sqrtf(r)*stride;
                    VisionBBox box={cx-w/2,cy-h/2,cx+w/2,cy+h/2,1.f,0};
                    vision_bbox_array_push(arr,&box);
                }
            }
        }
    }
    return arr;
}

/* ------------------------------------------------------------------ bbox encode / decode */

void vision_bbox_encode(const VisionBBox *anchor, const VisionBBox *target,
                        float *dx, float *dy, float *dw, float *dh) {
    float aw=anchor->x2-anchor->x1, ah=anchor->y2-anchor->y1;
    float ax=(anchor->x1+anchor->x2)*0.5f, ay=(anchor->y1+anchor->y2)*0.5f;
    float tw=target->x2-target->x1, th=target->y2-target->y1;
    float tx=(target->x1+target->x2)*0.5f, ty=(target->y1+target->y2)*0.5f;
    *dx=(tx-ax)/(aw+1e-6f);
    *dy=(ty-ay)/(ah+1e-6f);
    *dw=logf((tw+1e-6f)/(aw+1e-6f));
    *dh=logf((th+1e-6f)/(ah+1e-6f));
}

void vision_bbox_decode(const VisionBBox *anchor,
                        float dx, float dy, float dw, float dh,
                        VisionBBox *out) {
    float aw=anchor->x2-anchor->x1, ah=anchor->y2-anchor->y1;
    float ax=(anchor->x1+anchor->x2)*0.5f, ay=(anchor->y1+anchor->y2)*0.5f;
    float cx=dx*aw+ax, cy=dy*ah+ay;
    float w=expf(clampf(dw,-10,10))*aw, h=expf(clampf(dh,-10,10))*ah;
    out->x1=cx-w/2; out->y1=cy-h/2;
    out->x2=cx+w/2; out->y2=cy+h/2;
    out->score=anchor->score; out->class_id=anchor->class_id;
}
