#define VISION_INTERNAL
#include "vision.h"
#include <string.h>
#include <stdlib.h>

static inline int clampi(int v,int lo,int hi){return v<lo?lo:(v>hi?hi:v);}

/* ------------------------------------------------------------------ mask resize (nearest-neighbor, label-safe) */

VisionImage *vision_mask_resize(const VisionImage *mask, int new_width, int new_height) {
    if(!mask||mask->format!=VISION_FMT_UINT8||mask->channels!=1){
        VISION_ERR("mask_resize: needs uint8 1-channel"); return NULL;
    }
    VisionImage *dst=vision_image_create(new_width,new_height,1,VISION_FMT_UINT8,
                                          VISION_LAYOUT_HWC,VISION_CS_GRAY);
    if(!dst)return NULL;
    float sx=(float)mask->width/new_width;
    float sy=(float)mask->height/new_height;
    #pragma omp parallel for schedule(static)
    for(int oy=0;oy<new_height;oy++){
        int iy=clampi((int)(oy*sy),0,mask->height-1);
        uint8_t *dr=dst->data+oy*dst->stride;
        const uint8_t *sr=mask->data+iy*mask->stride;
        for(int ox=0;ox<new_width;ox++){
            int ix=clampi((int)(ox*sx),0,mask->width-1);
            dr[ox]=sr[ix];
        }
    }
    return dst;
}

/* ------------------------------------------------------------------ polygon rasterize (scanline) */

VisionImage *vision_polygon_rasterize(const float *pts, int n_pts,
                                      int width, int height, uint8_t fill_val) {
    if(!pts||n_pts<3||width<=0||height<=0){
        VISION_ERR("polygon_rasterize: invalid args"); return NULL;
    }
    VisionImage *mask=vision_image_create(width,height,1,VISION_FMT_UINT8,
                                          VISION_LAYOUT_HWC,VISION_CS_GRAY);
    if(!mask)return NULL;
    memset(mask->data,0,(size_t)height*mask->stride);

    /* scanline fill: for each row, find x intersections with polygon edges */
    int *xs=(int*)malloc((size_t)n_pts*sizeof(int));
    if(!xs){vision_image_free(mask);return NULL;}

    for(int y=0;y<height;y++){
        float fy=(float)y+0.5f;
        int cnt=0;
        for(int i=0;i<n_pts;i++){
            int j=(i+1)%n_pts;
            float y0=pts[i*2+1], y1=pts[j*2+1];
            if((y0<=fy&&y1>fy)||(y1<=fy&&y0>fy)){
                float x0=pts[i*2], x1=pts[j*2];
                float xi=x0+(fy-y0)*(x1-x0)/(y1-y0);
                xs[cnt++]=(int)xi;
            }
        }
        /* sort intersections */
        for(int a=0;a<cnt-1;a++)
            for(int b=a+1;b<cnt;b++)
                if(xs[b]<xs[a]){int t=xs[a];xs[a]=xs[b];xs[b]=t;}
        uint8_t *row=mask->data+y*mask->stride;
        for(int k=0;k+1<cnt;k+=2){
            int x0=clampi(xs[k],0,width-1);
            int x1=clampi(xs[k+1],0,width);
            memset(row+x0,fill_val,(size_t)(x1-x0));
        }
    }
    free(xs);
    return mask;
}

/* ------------------------------------------------------------------ connected components (two-pass union-find) */

static int cc_find(int *parent, int x) {
    while(parent[x]!=x){ parent[x]=parent[parent[x]]; x=parent[x]; }
    return x;
}
static void cc_union(int *parent, int a, int b) {
    a=cc_find(parent,a); b=cc_find(parent,b);
    if(a!=b) parent[b]=a;
}

VisionCC *vision_connected_components(const VisionImage *mask) {
    if(!mask||mask->format!=VISION_FMT_UINT8||mask->channels!=1){
        VISION_ERR("cc: needs uint8 1-channel"); return NULL;
    }
    int W=mask->width, H=mask->height;
    /* use int for intermediate labels, promote to uint16 at end */
    int *ilabels=(int*)calloc((size_t)H*W,sizeof(int));
    int *parent=(int*)malloc((size_t)(H*W+1)*sizeof(int));
    if(!ilabels||!parent){free(ilabels);free(parent);return NULL;}
    for(int i=0;i<=H*W;i++) parent[i]=i;

    /* fg = any non-zero value; 4-connectivity */
    int next_label=1;
    for(int y=0;y<H;y++){
        const uint8_t *row=mask->data+y*mask->stride;
        for(int x=0;x<W;x++){
            if(row[x]==0){ilabels[y*W+x]=0;continue;}
            int up  =(y>0&&mask->data[(y-1)*mask->stride+x]!=0)?ilabels[(y-1)*W+x]:0;
            int left=(x>0&&row[x-1]!=0)?ilabels[y*W+x-1]:0;
            int lab=0;
            if(up>0)  lab=up;
            if(left>0){if(lab==0)lab=left; else cc_union(parent,lab,left);}
            if(lab==0){lab=next_label++;parent[lab]=lab;}
            ilabels[y*W+x]=lab;
        }
    }
    /* second pass: relabel contiguous from 1 */
    int *remap=(int*)calloc((size_t)next_label,sizeof(int));
    if(!remap){free(ilabels);free(parent);return NULL;}
    int n_comp=0;
    for(int i=0;i<H*W;i++){
        if(ilabels[i]==0)continue;
        int root=cc_find(parent,ilabels[i]);
        if(remap[root]==0)remap[root]=++n_comp;
        ilabels[i]=remap[root];
    }
    free(parent); free(remap);

    /* allocate output arrays (1-indexed: index 0 unused) */
    uint16_t *labels16=(uint16_t*)malloc((size_t)H*W*sizeof(uint16_t));
    int *bx1=(int*)malloc((size_t)(n_comp+1)*sizeof(int));
    int *by1=(int*)malloc((size_t)(n_comp+1)*sizeof(int));
    int *bx2=(int*)malloc((size_t)(n_comp+1)*sizeof(int));
    int *by2=(int*)malloc((size_t)(n_comp+1)*sizeof(int));
    int *areas=(int*)calloc((size_t)(n_comp+1),sizeof(int));
    if(!labels16||!bx1||!by1||!bx2||!by2||!areas){
        free(ilabels);free(labels16);free(bx1);free(by1);free(bx2);free(by2);free(areas);
        return NULL;
    }
    for(int k=1;k<=n_comp;k++){bx1[k]=W;by1[k]=H;bx2[k]=-1;by2[k]=-1;}
    for(int y=0;y<H;y++){
        for(int x=0;x<W;x++){
            int lbl=ilabels[y*W+x];
            labels16[y*W+x]=(uint16_t)lbl;
            if(lbl==0)continue;
            areas[lbl]++;
            if(x<bx1[lbl])bx1[lbl]=x; if(x>bx2[lbl])bx2[lbl]=x;
            if(y<by1[lbl])by1[lbl]=y; if(y>by2[lbl])by2[lbl]=y;
        }
    }
    free(ilabels);

    VisionCC *cc=(VisionCC*)malloc(sizeof(VisionCC));
    if(!cc){free(labels16);free(bx1);free(by1);free(bx2);free(by2);free(areas);return NULL;}
    cc->labels=labels16;
    cc->n_components=n_comp;
    cc->width=W; cc->height=H;
    cc->bbox_x1=bx1; cc->bbox_y1=by1;
    cc->bbox_x2=bx2; cc->bbox_y2=by2;
    cc->areas=areas;
    return cc;
}

void vision_cc_free(VisionCC *cc) {
    if(cc){
        free(cc->labels);
        free(cc->bbox_x1);free(cc->bbox_y1);
        free(cc->bbox_x2);free(cc->bbox_y2);
        free(cc->areas);
        free(cc);
    }
}
