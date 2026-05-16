/* vision_model.c — Model-specific decode & utility functions
 *
 * Implements postprocessing for:
 *   SSDLite    — anchor-based multi-scale decode
 *   NanoDet    — FCOS-style GFL distribution decode
 *   PicoDet    — DFL distribution decode (aligned head)
 *   YOLO11n    — DFL + anchor-free decode
 *   FastSAM    — prototype × coefficient mask assembly
 *
 * Rules:
 *   • ALL allocation via vision_alloc() / posix_memalign
 *   • ZERO PHP-side math — all ops run here
 *   • OpenMP on outer loops where N*C > threshold
 */
#define VISION_INTERNAL
#include "vision.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

/* ── helpers ─────────────────────────────────────────────────────────────── */
static inline float _vm_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float _vm_clampf(float x, float lo, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
/* softmax over n values in src → dst (in-place safe) */
static void _vm_softmax(const float* src, float* dst, int n) {
    float mx = src[0];
    for (int i = 1; i < n; i++) if (src[i] > mx) mx = src[i];
    float s = 0.0f;
    for (int i = 0; i < n; i++) { dst[i] = expf(src[i] - mx); s += dst[i]; }
    float inv = 1.0f / s;
    for (int i = 0; i < n; i++) dst[i] *= inv;
}
/* DFL/GFL decode: distribution over reg_max bins → distance scalar */
static float _vm_dfl_decode(const float* logits, int reg_max) {
    float tmp[64]; /* reg_max <= 32 in practice */
    _vm_softmax(logits, tmp, reg_max);
    float d = 0.0f;
    for (int i = 0; i < reg_max; i++) d += tmp[i] * (float)i;
    return d;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 1. SSD PRIOR BOXES
 *
 * Generates all prior (anchor) boxes in normalised [0,1] xyxy format.
 *
 * feat_sizes  : {fH0,fW0, fH1,fW1, ...}  2*n_scales values
 * min_sizes   : min anchor size per scale  (n_scales values)
 * max_sizes   : max anchor size per scale  (n_scales values)
 * ratios      : extra aspect ratios shared across all scales (n_ratios)
 * img_size    : input image width = height (square assumed)
 * Returns     : VisionBBoxArray* in cx,cy,w,h normalised, or NULL on error
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionBBoxArray* vision_ssd_prior_boxes(const int*   feat_sizes,  int n_scales,
                                         const float* min_sizes,
                                         const float* max_sizes,
                                         const float* ratios,     int n_ratios,
                                         int img_size) {
    /* count total anchors */
    int total = 0;
    for (int s = 0; s < n_scales; s++) {
        int fH = feat_sizes[2*s], fW = feat_sizes[2*s+1];
        total += fH * fW * (2 + n_ratios * 2);   /* square-min + square-max + ratio pairs */
    }
    VisionBBoxArray* arr = vision_bbox_array_create(total);
    if (!arr) return NULL;

    float img_f = (float)img_size;
    for (int s = 0; s < n_scales; s++) {
        int fH = feat_sizes[2*s], fW = feat_sizes[2*s+1];
        float sk  = min_sizes[s] / img_f;
        float sk1 = max_sizes[s] / img_f;
        float sk_prime = sqrtf(sk * sk1);
        for (int row = 0; row < fH; row++) {
            float cy = ((float)row + 0.5f) / (float)fH;
            for (int col = 0; col < fW; col++) {
                float cx = ((float)col + 0.5f) / (float)fW;
                /* 1:1 at min_size */
                VisionBBox b; b.class_id = -1; b.score = 1.0f;
                b.x1 = cx - sk/2; b.y1 = cy - sk/2; b.x2 = cx + sk/2; b.y2 = cy + sk/2;
                vision_bbox_array_push(arr, &b);
                /* 1:1 at sqrt(min*max) */
                b.x1 = cx-sk_prime/2; b.y1 = cy-sk_prime/2;
                b.x2 = cx+sk_prime/2; b.y2 = cy+sk_prime/2;
                vision_bbox_array_push(arr, &b);
                /* aspect ratios */
                for (int r = 0; r < n_ratios; r++) {
                    float sqr = sqrtf(ratios[r]);
                    float w = sk * sqr, h = sk / sqr;
                    b.x1=cx-w/2; b.y1=cy-h/2; b.x2=cx+w/2; b.y2=cy+h/2;
                    vision_bbox_array_push(arr, &b);
                    /* inverted */
                    b.x1=cx-h/2; b.y1=cy-w/2; b.x2=cx+h/2; b.y2=cy+w/2;
                    vision_bbox_array_push(arr, &b);
                }
            }
        }
    }
    return arr;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 2. SSD DECODE
 *
 * loc_pred   : float[n_anchors * 4]  — (dcx, dcy, dw, dh) deltas
 * cls_pred   : float[n_anchors * (n_cls+1)] — raw logits (bg=0)
 * anchors    : VisionBBoxArray of prior boxes (xyxy, normalised)
 * n_cls      : foreground classes (background excluded)
 * conf_thr   : objectness threshold (after sigmoid/softmax)
 * var        : [var_xy, var_wh] SSD variance correction factors
 * Returns    : VisionBBoxArray* (xyxy, normalised); caller frees via vision_bbox_array_free
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionBBoxArray* vision_ssd_decode(const float* loc_pred,  const float* cls_pred,
                                    const VisionBBoxArray* anchors,
                                    int n_cls,    float conf_thr,
                                    float var_xy, float var_wh) {
    int n = anchors->count;
    VisionBBoxArray* out = vision_bbox_array_create(64);
    if (!out) return NULL;
    float* scores = (float*)malloc((size_t)(n_cls + 1) * sizeof(float));
    if (!scores) { vision_bbox_array_free(out); return NULL; }

    for (int i = 0; i < n; i++) {
        const VisionBBox* prior = &anchors->boxes[i];
        /* prior in xyxy → cx/cy/w/h */
        float pcx = (prior->x1 + prior->x2) * 0.5f;
        float pcy = (prior->y1 + prior->y2) * 0.5f;
        float pw  = prior->x2 - prior->x1;
        float ph  = prior->y2 - prior->y1;

        const float* d = loc_pred + i * 4;
        float cx = d[0] * var_xy * pw + pcx;
        float cy = d[1] * var_xy * ph + pcy;
        float w  = pw * expf(d[2] * var_wh);
        float h  = ph * expf(d[3] * var_wh);

        _vm_softmax(cls_pred + i*(n_cls+1), scores, n_cls+1);
        int best_cls = -1; float best_sc = conf_thr;
        for (int c = 1; c <= n_cls; c++) {
            if (scores[c] > best_sc) { best_sc = scores[c]; best_cls = c - 1; }
        }
        if (best_cls < 0) continue;

        VisionBBox b;
        b.x1 = _vm_clampf(cx - w*0.5f, 0.0f, 1.0f);
        b.y1 = _vm_clampf(cy - h*0.5f, 0.0f, 1.0f);
        b.x2 = _vm_clampf(cx + w*0.5f, 0.0f, 1.0f);
        b.y2 = _vm_clampf(cy + h*0.5f, 0.0f, 1.0f);
        b.score = best_sc; b.class_id = best_cls;
        vision_bbox_array_push(out, &b);
    }
    free(scores);
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 3. NANODET DECODE  (FCOS-style + GFL distribution)
 *
 * cls_pred   : float[feat_h * feat_w * n_cls]  — raw logits
 * reg_pred   : float[feat_h * feat_w * 4 * reg_max]  — distribution logits
 * stride     : feature map stride in pixels (e.g. 8, 16, 32)
 * img_w/h    : input image size
 * Returns    : VisionBBoxArray* (xyxy, pixel coords); caller frees
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionBBoxArray* vision_nanodet_decode(const float* cls_pred, const float* reg_pred,
                                        int feat_h, int feat_w, int stride,
                                        int n_cls, int reg_max,
                                        int img_w, int img_h, float conf_thr) {
    VisionBBoxArray* out = vision_bbox_array_create(64);
    if (!out) return NULL;
    int n_loc = feat_h * feat_w;
    for (int idx = 0; idx < n_loc; idx++) {
        int row = idx / feat_w, col = idx % feat_w;
        /* best class via sigmoid */
        const float* cp = cls_pred + idx * n_cls;
        int best_c = -1; float best_s = conf_thr;
        for (int c = 0; c < n_cls; c++) {
            float s = _vm_sigmoid(cp[c]);
            if (s > best_s) { best_s = s; best_c = c; }
        }
        if (best_c < 0) continue;
        /* DFL decode: 4 directions × reg_max bins */
        const float* rp = reg_pred + idx * 4 * reg_max;
        float lt = _vm_dfl_decode(rp,              reg_max) * (float)stride;
        float tt = _vm_dfl_decode(rp +   reg_max,  reg_max) * (float)stride;
        float rt = _vm_dfl_decode(rp + 2*reg_max,  reg_max) * (float)stride;
        float bt = _vm_dfl_decode(rp + 3*reg_max,  reg_max) * (float)stride;
        /* anchor point at cell centre */
        float cx = ((float)col + 0.5f) * (float)stride;
        float cy = ((float)row + 0.5f) * (float)stride;
        VisionBBox b;
        b.x1 = _vm_clampf(cx - lt, 0.0f, (float)img_w);
        b.y1 = _vm_clampf(cy - tt, 0.0f, (float)img_h);
        b.x2 = _vm_clampf(cx + rt, 0.0f, (float)img_w);
        b.y2 = _vm_clampf(cy + bt, 0.0f, (float)img_h);
        b.score = best_s; b.class_id = best_c;
        if ((b.x2 - b.x1) > 0 && (b.y2 - b.y1) > 0)
            vision_bbox_array_push(out, &b);
    }
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 4. PICODET DECODE  (aligned head + DFL — same math as NanoDet)
 *
 * PicoDet uses the same GFL/DFL distribution decode as NanoDet.
 * The difference is in the neck architecture (CSP-PAN) handled in PHP layers.
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionBBoxArray* vision_picodet_decode(const float* cls_pred, const float* reg_pred,
                                        int feat_h, int feat_w, int stride,
                                        int n_cls, int reg_max,
                                        int img_w, int img_h, float conf_thr) {
    /* Identical decode to NanoDet — both use DFL distribution */
    return vision_nanodet_decode(cls_pred, reg_pred, feat_h, feat_w, stride,
                                  n_cls, reg_max, img_w, img_h, conf_thr);
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 5. YOLO11 DECODE  (DFL + anchor-free, sigmoid cls)
 *
 * output   : float[feat_h * feat_w * (4*reg_max + n_cls)]
 *            first 4*reg_max values = box distribution, then n_cls = class logits
 * Returns  : VisionBBoxArray* (xyxy, pixel coords)
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionBBoxArray* vision_yolo11_decode(const float* output,
                                       int feat_h, int feat_w, int stride,
                                       int n_cls, int reg_max,
                                       int img_w, int img_h, float conf_thr) {
    VisionBBoxArray* out = vision_bbox_array_create(64);
    if (!out) return NULL;
    int box_dim = 4 * reg_max;
    int row_dim = box_dim + n_cls;
    for (int idx = 0; idx < feat_h * feat_w; idx++) {
        const float* row = output + idx * row_dim;
        /* class scores with sigmoid */
        int best_c = -1; float best_s = conf_thr;
        for (int c = 0; c < n_cls; c++) {
            float s = _vm_sigmoid(row[box_dim + c]);
            if (s > best_s) { best_s = s; best_c = c; }
        }
        if (best_c < 0) continue;
        /* DFL decode */
        float lt = _vm_dfl_decode(row,             reg_max) * (float)stride;
        float tt = _vm_dfl_decode(row +   reg_max, reg_max) * (float)stride;
        float rt = _vm_dfl_decode(row + 2*reg_max, reg_max) * (float)stride;
        float bt = _vm_dfl_decode(row + 3*reg_max, reg_max) * (float)stride;
        float cx = ((float)(idx % feat_w) + 0.5f) * (float)stride;
        float cy = ((float)(idx / feat_w) + 0.5f) * (float)stride;
        VisionBBox b;
        b.x1 = _vm_clampf(cx - lt, 0.0f, (float)img_w);
        b.y1 = _vm_clampf(cy - tt, 0.0f, (float)img_h);
        b.x2 = _vm_clampf(cx + rt, 0.0f, (float)img_w);
        b.y2 = _vm_clampf(cy + bt, 0.0f, (float)img_h);
        b.score = best_s; b.class_id = best_c;
        if ((b.x2 - b.x1) > 0 && (b.y2 - b.y1) > 0)
            vision_bbox_array_push(out, &b);
    }
    return out;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 6. FASTSAM MASK ASSEMBLY
 *
 * Assembles per-instance segmentation masks from prototype bank + coefficients.
 * Implements: mask_i = sigmoid(proto.T @ coeffs[i])  then threshold + resize.
 *
 * proto    : float[n_proto * proto_h * proto_w]  (CHW, row-major)
 * coeffs   : float[n_dets  * n_proto]
 * boxes    : VisionBBoxArray* for crop-to-bbox (optional, may be NULL)
 * out_h/w  : output mask size (typically = input image size)
 * mask_thr : sigmoid threshold for binary mask (typically 0.5)
 *
 * Returns  : VisionImage* shape [1, n_dets, out_h, out_w] UINT8 (0/255 per mask)
 *            Caller owns it and must call vision_image_free().
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionImage* vision_fastsam_assemble_masks(const float* proto, int n_proto,
                                            int proto_h, int proto_w,
                                            const float* coeffs, int n_dets,
                                            const VisionBBoxArray* boxes,
                                            int out_w, int out_h, float mask_thr) {
    /* Result: n_dets channels, each out_h×out_w, stored flat */
    size_t mask_pix = (size_t)out_h * out_w;
    size_t total    = (size_t)n_dets * mask_pix;
    uint8_t* data   = (uint8_t*)vision_alloc(total);
    if (!data) return NULL;

    /* pre-compute scale factors: proto space → output space */
    float sy = (float)out_h / (float)proto_h;
    float sx = (float)out_w / (float)proto_w;

    #pragma omp parallel for schedule(static) if(n_dets > 4)
    for (int d = 0; d < n_dets; d++) {
        const float* c = coeffs + d * n_proto;
        uint8_t* out_d = data + (size_t)d * mask_pix;

        /* bbox crop window in output coords */
        float bx1=0, by1=0, bx2=(float)out_w, by2=(float)out_h;
        if (boxes && d < boxes->count) {
            bx1 = boxes->boxes[d].x1; by1 = boxes->boxes[d].y1;
            bx2 = boxes->boxes[d].x2; by2 = boxes->boxes[d].y2;
        }

        for (int py = 0; py < proto_h; py++) {
            for (int px = 0; px < proto_w; px++) {
                /* dot: sum_j coeffs[d,j] * proto[j, py, px] */
                float dot = 0.0f;
                size_t pix_off = (size_t)py * proto_w + px;
                for (int j = 0; j < n_proto; j++)
                    dot += c[j] * proto[(size_t)j * proto_h * proto_w + pix_off];
                float val = _vm_sigmoid(dot);

                /* bilinear upscale: proto pixel → output pixels */
                int oy0 = (int)((float)py       * sy);
                int oy1 = (int)((float)(py + 1) * sy);
                int ox0 = (int)((float)px       * sx);
                int ox1 = (int)((float)(px + 1) * sx);
                if (oy1 > out_h) oy1 = out_h;
                if (ox1 > out_w) ox1 = out_w;

                uint8_t mval = (val >= mask_thr) ? 255 : 0;
                for (int oy = oy0; oy < oy1; oy++) {
                    float fy = (float)oy;
                    if (fy < by1 || fy >= by2) { /* outside bbox crop */
                        for (int ox = ox0; ox < ox1; ox++) out_d[oy*out_w + ox] = 0;
                        continue;
                    }
                    for (int ox = ox0; ox < ox1; ox++) {
                        float fx = (float)ox;
                        out_d[oy*out_w + ox] = (fx >= bx1 && fx < bx2) ? mval : 0;
                    }
                }
            }
        }
    }

    /* Wrap into VisionImage: treat as single-channel stack (n_dets channels) */
    VisionImage* img = vision_image_create(out_w, out_h, n_dets,
                                           VISION_FMT_UINT8, VISION_LAYOUT_CHW,
                                           VISION_COLOR_GRAY);
    if (!img) { vision_dealloc(data, total); return NULL; }
    /* replace internal buffer with our aligned allocation */
    if (img->owns_data && img->data) vision_dealloc(img->data, img->data_size);
    img->data      = data;
    img->data_size = total;
    img->owns_data = 1;
    return img;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * 7. MULTI-SCALE DECODE  (utility: run decode over multiple FPN strides)
 *
 * Merges detections from N feature map scales into one VisionBBoxArray,
 * then applies class-specific NMS.  Used by NanoDet/PicoDet/YOLO11.
 *
 * cls_preds   : float*[n_scales]  — per-scale cls logits
 * reg_preds   : float*[n_scales]  — per-scale reg logits
 * feat_hs/ws  : int[n_scales]     — feature map sizes
 * strides     : int[n_scales]     — stride per scale
 * decode_fn   : 0=nanodet, 1=yolo11
 * iou_thr     : NMS IoU threshold
 * ═══════════════════════════════════════════════════════════════════════════ */
VisionBBoxArray* vision_multiscale_decode(const float** cls_preds,
                                           const float** reg_preds,
                                           const int*    feat_hs,
                                           const int*    feat_ws,
                                           const int*    strides,
                                           int n_scales,
                                           int n_cls, int reg_max,
                                           int img_w, int img_h,
                                           float conf_thr, float iou_thr,
                                           int decode_fn) {
    /* accumulate all detections */
    VisionBBoxArray* all = vision_bbox_array_create(256);
    if (!all) return NULL;
    for (int s = 0; s < n_scales; s++) {
        VisionBBoxArray* det;
        if (decode_fn == 1)
            det = vision_yolo11_decode(cls_preds[s],
                                       feat_hs[s], feat_ws[s], strides[s],
                                       n_cls, reg_max, img_w, img_h, conf_thr);
        else
            det = vision_nanodet_decode(cls_preds[s], reg_preds[s],
                                        feat_hs[s], feat_ws[s], strides[s],
                                        n_cls, reg_max, img_w, img_h, conf_thr);
        if (!det) continue;
        for (int i = 0; i < det->count; i++)
            vision_bbox_array_push(all, &det->boxes[i]);
        vision_bbox_array_free(det);
    }
    VisionBBoxArray* result = vision_nms(all, iou_thr);
    vision_bbox_array_free(all);
    return result;
}
