#define _GNU_SOURCE
#include "quant.h"
#include <math.h>
#include <string.h>
#include <float.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* ─── helpers ──────────────────────────────────────────────────────────── */

static inline const float *cf32(const void *p) { return (const float *)p; }
static inline float       *f32(void *p)        { return (float *)p; }

#define IDX(t, n, N) ((t) * (N) + (n))

/* ─── log returns ──────────────────────────────────────────────────────── */

void quant_log_returns(const void *close, void *out, int T, int N)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    for (int n = 0; n < N; n++) o[n] = NAN; /* row 0 */

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 1; t < T; t++) {
            float prev = c[IDX(t-1, n, N)];
            float curr = c[IDX(t,   n, N)];
            o[IDX(t, n, N)] = (prev > 0.f && curr > 0.f) ? logf(curr / prev) : NAN;
        }
    }
}

/* ─── momentum ──────────────────────────────────────────────────────────── */

void quant_momentum(const void *close, void *out, int T, int N, int lookback)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < lookback) { o[IDX(t, n, N)] = NAN; continue; }
            float prev = c[IDX(t - lookback, n, N)];
            float curr = c[IDX(t,             n, N)];
            o[IDX(t, n, N)] = (prev > 0.f) ? (curr / prev - 1.f) : NAN;
        }
    }
}

/* ─── EMA ───────────────────────────────────────────────────────────────── */

void quant_ema(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);
    float alpha    = 2.f / (period + 1.f);
    float decay    = 1.f - alpha;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float ema = x[n]; /* seed with first bar */
        o[n] = ema;
        for (int t = 1; t < T; t++) {
            float v = x[IDX(t, n, N)];
            ema = isnan(v) ? ema : alpha * v + decay * ema;
            o[IDX(t, n, N)] = ema;
        }
    }
}

/* ─── rolling SMA ───────────────────────────────────────────────────────── */

void quant_rolling_mean(const void *X, void *out, int T, int N, int period)
{
    const float *x   = cf32(X);
    float       *o   = f32(out);
    float        inv = 1.f / period;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float sum = 0.f;
        for (int t = 0; t < T; t++) {
            sum += x[IDX(t, n, N)];
            if (t >= period) sum -= x[IDX(t - period, n, N)];
            o[IDX(t, n, N)] = (t < period - 1) ? NAN : sum * inv;
        }
    }
}

/* ─── rolling std (population) ─────────────────────────────────────────── */

void quant_rolling_std(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float sum = 0.f, sum2 = 0.f;
        for (int t = 0; t < T; t++) {
            float v = x[IDX(t, n, N)];
            sum  += v; sum2 += v * v;
            if (t >= period) {
                float old = x[IDX(t - period, n, N)];
                sum -= old; sum2 -= old * old;
            }
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float mean = sum / period;
            float var  = sum2 / period - mean * mean;
            o[IDX(t, n, N)] = var > 0.f ? sqrtf(var) : 0.f;
        }
    }
}

/* ─── rolling max ───────────────────────────────────────────────────────── */

void quant_rolling_max(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float mx = x[IDX(t - period + 1, n, N)];
            for (int k = t - period + 2; k <= t; k++) {
                float v = x[IDX(k, n, N)];
                if (v > mx) mx = v;
            }
            o[IDX(t, n, N)] = mx;
        }
    }
}

/* ─── rolling min ───────────────────────────────────────────────────────── */

void quant_rolling_min(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float mn = x[IDX(t - period + 1, n, N)];
            for (int k = t - period + 2; k <= t; k++) {
                float v = x[IDX(k, n, N)];
                if (v < mn) mn = v;
            }
            o[IDX(t, n, N)] = mn;
        }
    }
}

/* ─── ATR ───────────────────────────────────────────────────────────────── */

void quant_atr(const void *high, const void *low, const void *close,
               void *out, int T, int N, int period)
{
    const float *h = cf32(high), *l = cf32(low), *c = cf32(close);
    float       *o = f32(out);
    float alpha    = 2.f / (period + 1.f);
    float decay    = 1.f - alpha;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float atr = h[n] - l[n]; /* seed: first bar TR */
        o[n] = atr;
        for (int t = 1; t < T; t++) {
            float pc = c[IDX(t-1, n, N)];
            float hi = h[IDX(t,   n, N)];
            float lo = l[IDX(t,   n, N)];
            float tr = hi - lo;
            float a  = fabsf(hi - pc);
            float b  = fabsf(lo - pc);
            if (a > tr) tr = a;
            if (b > tr) tr = b;
            atr = alpha * tr + decay * atr;
            o[IDX(t, n, N)] = atr;
        }
    }
}

/* ─── RSI (Wilder) ──────────────────────────────────────────────────────── */

void quant_rsi(const void *close, void *out, int T, int N, int period)
{
    const float *c = cf32(close);
    float       *o = f32(out);
    float alpha    = 1.f / period; /* Wilder's smoothing factor */
    float decay    = 1.f - alpha;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        /* Seed averages from first `period` price changes */
        float avg_gain = 0.f, avg_loss = 0.f;
        int seed_end = period < T ? period : T - 1;
        for (int t = 1; t <= seed_end; t++) {
            float diff = c[IDX(t, n, N)] - c[IDX(t-1, n, N)];
            if (diff > 0.f) avg_gain += diff;
            else            avg_loss -= diff;
        }
        avg_gain /= period;
        avg_loss /= period;

        for (int t = 0; t < period; t++) o[IDX(t, n, N)] = NAN;

        if (period >= T) continue;

        float rs = (avg_loss > 0.f) ? avg_gain / avg_loss : 100.f;
        o[IDX(period, n, N)] = 100.f - 100.f / (1.f + rs);

        for (int t = period + 1; t < T; t++) {
            float diff  = c[IDX(t, n, N)] - c[IDX(t-1, n, N)];
            float gain  = diff > 0.f ? diff : 0.f;
            float loss  = diff < 0.f ? -diff : 0.f;
            avg_gain = alpha * gain + decay * avg_gain;
            avg_loss = alpha * loss + decay * avg_loss;
            rs = (avg_loss > 0.f) ? avg_gain / avg_loss : 100.f;
            o[IDX(t, n, N)] = 100.f - 100.f / (1.f + rs);
        }
    }
}

/* ─── efficiency ratio ──────────────────────────────────────────────────── */

void quant_efficiency_ratio(const void *close, void *out, int T, int N, int period)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period) { o[IDX(t, n, N)] = NAN; continue; }
            float direction = fabsf(c[IDX(t, n, N)] - c[IDX(t - period, n, N)]);
            float path = 0.f;
            for (int k = t - period + 1; k <= t; k++)
                path += fabsf(c[IDX(k, n, N)] - c[IDX(k-1, n, N)]);
            o[IDX(t, n, N)] = (path > 1e-8f) ? direction / path : 0.f;
        }
    }
}

/* ─── rolling beta vs benchmark ─────────────────────────────────────────── */

void quant_rolling_beta(const void *stock_ret, const void *bench_ret,
                        void *out, int T, int N, int period)
{
    const float *s = cf32(stock_ret);
    const float *b = cf32(bench_ret); /* [T] 1-D */
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float ss = 0.f, sb = 0.f, sb_cross = 0.f, b2 = 0.f;
            int   cnt = 0;
            for (int k = t - period + 1; k <= t; k++) {
                float sv = s[IDX(k, n, N)];
                float bv = b[k];
                if (isnan(sv) || isnan(bv)) continue;
                ss += sv; sb += bv;
                sb_cross += sv * bv;
                b2 += bv * bv;
                cnt++;
            }
            if (cnt < 2) { o[IDX(t, n, N)] = NAN; continue; }
            float ms   = ss / cnt, mb = sb / cnt;
            float cov  = sb_cross / cnt - ms * mb;
            float varb = b2 / cnt - mb * mb;
            o[IDX(t, n, N)] = (varb > 1e-12f) ? cov / varb : 0.f;
        }
    }
}

/* ─── rolling max drawdown ──────────────────────────────────────────────── */

void quant_rolling_max_drawdown(const void *close, void *out, int T, int N, int period)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float peak = c[IDX(t - period + 1, n, N)];
            float mdd  = 0.f;
            for (int k = t - period + 2; k <= t; k++) {
                float v = c[IDX(k, n, N)];
                if (v > peak) peak = v;
                float dd = (peak > 0.f) ? (peak - v) / peak : 0.f;
                if (dd > mdd) mdd = dd;
            }
            o[IDX(t, n, N)] = mdd;
        }
    }
}

/* ─── downside std ──────────────────────────────────────────────────────── */

void quant_downside_std(const void *returns, void *out, int T, int N, int period)
{
    const float *r = cf32(returns);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float sum2 = 0.f; int cnt = 0;
            for (int k = t - period + 1; k <= t; k++) {
                float v = r[IDX(k, n, N)];
                if (!isnan(v) && v < 0.f) { sum2 += v * v; cnt++; }
            }
            o[IDX(t, n, N)] = (cnt > 0) ? sqrtf(sum2 / cnt) : 0.f;
        }
    }
}

/* ─── cross-sectional rank normalize [0, 1] ─────────────────────────────── */

typedef struct { float val; int idx; } _vi;
static int _vi_cmp(const void *a, const void *b)
{
    float av = ((_vi *)a)->val, bv = ((_vi *)b)->val;
    return (av > bv) - (av < bv);
}

void quant_rank_normalize(const void *X, void *out, int T, int N)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel
    {
        _vi *buf = (_vi *)malloc((size_t)N * sizeof(_vi));

        #pragma omp for schedule(static)
        for (int t = 0; t < T; t++) {
            int valid = 0;
            for (int n = 0; n < N; n++) {
                float v = x[IDX(t, n, N)];
                if (!isnan(v)) { buf[valid].val = v; buf[valid].idx = n; valid++; }
            }
            for (int n = 0; n < N; n++) o[IDX(t, n, N)] = NAN;
            if (valid < 1) continue;
            if (valid == 1) { o[IDX(t, buf[0].idx, N)] = 0.5f; continue; }
            qsort(buf, (size_t)valid, sizeof(_vi), _vi_cmp);
            float inv = 1.f / (valid - 1);
            for (int r = 0; r < valid; r++)
                o[IDX(t, buf[r].idx, N)] = r * inv;
        }

        free(buf);
    }
}

/* ─── cross-sectional z-score (winsorized ±3σ) ──────────────────────────── */

void quant_zscore_normalize(const void *X, void *out, int T, int N)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float sum = 0.f, sum2 = 0.f; int cnt = 0;
        for (int n = 0; n < N; n++) {
            float v = x[IDX(t, n, N)];
            if (!isnan(v)) { sum += v; sum2 += v * v; cnt++; }
        }
        if (cnt < 2) {
            for (int n = 0; n < N; n++) o[IDX(t, n, N)] = 0.f;
            continue;
        }
        float mean = sum / cnt;
        float var  = sum2 / cnt - mean * mean;
        float inv  = (var > 0.f) ? 1.f / sqrtf(var) : 1.f;
        for (int n = 0; n < N; n++) {
            float v = x[IDX(t, n, N)];
            if (isnan(v)) { o[IDX(t, n, N)] = 0.f; continue; }
            float z = (v - mean) * inv;
            if (z >  3.f) z =  3.f;
            if (z < -3.f) z = -3.f;
            o[IDX(t, n, N)] = z;
        }
    }
}

/* ─── write one factor column into a pre-allocated [TN, F] interleaved buf ─ */
/*
 * Eliminates the PHP interleave loop in FactorEngine.
 * src: [T,N] = TN floats (row-major, contiguous)
 * dst: [TN, F] pre-allocated output (layout: dst[i*F + col])
 * NaN in src → nan_fill in dst.
 */
void quant_write_factor_col(const void *src_v, void *dst_v,
                             int col, int TN, int F, float nan_fill)
{
    const float *s = cf32(src_v);
    float       *d = f32(dst_v);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < TN; i++) {
        float v = s[i];
        d[i * F + col] = isnan(v) ? nan_fill : v;
    }
}

/* ─── weighted composite score across all factors ───────────────────────── */
/*
 * Replaces RankAggregator's T×N×F PHP loop.
 * rank_mat: [T*N, F]  — rank-normalised factor matrix
 * weights:  [F]       — factor weights; negative ⟹ invert (1−val) before weighting
 * out:      [T*N]     — composite score per (t,n), row-major same layout as [T,N]
 */
void quant_weighted_composite(const void *rank_mat_v, const float *weights,
                               void *out_v, int TN, int F)
{
    const float *rm = cf32(rank_mat_v);
    float       *o  = f32(out_v);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < TN; i++) {
        float score = 0.f;
        const float *row = rm + i * F;
        for (int f = 0; f < F; f++) {
            float w = weights[f];
            float v = row[f];
            if (isnan(v)) v = 0.5f;
            score += (w >= 0.f) ? (w * v) : (-w * (1.f - v));
        }
        o[i] = score;
    }
}

/* ─── row mean across N axis: breadth / fraction ────────────────────────── */
/*
 * mat: [T, N] — e.g. a 0/1 indicator (above-SMA)
 * out: [T]    — fraction of valid (non-NaN) entries per row
 */
void quant_row_mean(const void *mat_v, void *out_v, int T, int N)
{
    const float *m   = cf32(mat_v);
    float       *o   = f32(out_v);
    float        inv = (N > 0) ? 1.f / N : 1.f;

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float sum = 0.f;
        int   cnt = 0;
        for (int n = 0; n < N; n++) {
            float v = m[t * N + n];
            if (!isnan(v)) { sum += v; cnt++; }
        }
        o[t] = (cnt > 0) ? sum / cnt : 0.f;
    }
}

/* ─── ADX (Average Directional Index) ──────────────────────────────────── */

/* ADX = Wilder-smoothed DX over `period` bars.
   DX = 100 × |+DI − −DI| / (+DI + −DI).
   NaN for first (2×period + 1) bars while warm-up completes. */
void quant_adx(const void *high, const void *low, const void *close,
               void *out, int T, int N, int period)
{
    const float *h = cf32(high), *l = cf32(low), *c = cf32(close);
    float       *o = f32(out);
    float alpha    = 1.f / (float)period;
    int   warmup   = 2 * period + 2;

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float smTR = 0.f, smPDM = 0.f, smMDM = 0.f;
        float adx  = NAN;

        for (int t = 0; t < T; t++) {
            o[IDX(t, n, N)] = NAN;
            if (t < 1) continue;

            float hi  = h[IDX(t,   n, N)];
            float lo  = l[IDX(t,   n, N)];
            float ph  = h[IDX(t-1, n, N)];
            float pl  = l[IDX(t-1, n, N)];
            float pc  = c[IDX(t-1, n, N)];

            float tr  = fmaxf(hi - lo, fmaxf(fabsf(hi - pc), fabsf(lo - pc)));
            /* +DM: new high extends more upward; −DM: new low extends more downward */
            float pDM = (hi - ph > pl - lo && hi - ph > 0.f) ? hi - ph : 0.f;
            float mDM = (pl - lo > hi - ph && pl - lo > 0.f) ? pl - lo : 0.f;

            if (t == 1) {
                smTR = tr; smPDM = pDM; smMDM = mDM;
                continue;
            }

            /* Wilder's smoothing: S_t = S_{t-1} × (1 − 1/p) + new */
            smTR  = smTR  * (1.f - alpha) + tr;
            smPDM = smPDM * (1.f - alpha) + pDM;
            smMDM = smMDM * (1.f - alpha) + mDM;

            if (smTR < 1e-10f) continue;

            float pDI  = 100.f * smPDM / smTR;
            float mDI  = 100.f * smMDM / smTR;
            float dsum = pDI + mDI;
            float dx   = (dsum > 0.f) ? 100.f * fabsf(pDI - mDI) / dsum : 0.f;

            adx = isnan(adx) ? dx : adx * (1.f - alpha) + dx * alpha;

            if (t >= warmup) o[IDX(t, n, N)] = adx;
        }
    }
}

/* ─── MACD histogram ─────────────────────────────────────────────────────── */

/* MACD histogram = EMA(fast) − EMA(slow) − EMA(signal)[MACD line].
   Positive = momentum building upward; histogram peak-to-zero crossing = entry. */
void quant_macd_histogram(const void *close, void *out, int T, int N,
                           int fast, int slow, int signal_period)
{
    const float *c = cf32(close);
    float       *o = f32(out);
    float af   = 2.f / ((float)fast          + 1.f);
    float as_  = 2.f / ((float)slow          + 1.f);
    float asig = 2.f / ((float)signal_period + 1.f);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        float ef = 0.f, es = 0.f, sig = 0.f;
        int   init = 0;

        for (int t = 0; t < T; t++) {
            float price = c[IDX(t, n, N)];
            if (price <= 0.f || isnan(price)) { o[IDX(t, n, N)] = NAN; continue; }
            if (!init) {
                ef = es = price; sig = 0.f; init = 1;
            } else {
                ef  += af   * (price - ef);
                es  += as_  * (price - es);
            }
            float macd = ef - es;
            sig += asig * (macd - sig);
            o[IDX(t, n, N)] = macd - sig;
        }
    }
}

/* ─── forward-looking binary labels ─────────────────────────────────────── */

void quant_generate_labels(const void *close, void *out,
                            int T, int N, int horizon, float threshold)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t + horizon >= T) { o[IDX(t, n, N)] = NAN; continue; }
            float curr = c[IDX(t, n, N)];
            if (curr <= 0.f) { o[IDX(t, n, N)] = NAN; continue; }
            float target = curr * (1.f + threshold);
            float label  = 0.f;
            for (int k = 1; k <= horizon; k++) {
                if (c[IDX(t + k, n, N)] >= target) { label = 1.f; break; }
            }
            o[IDX(t, n, N)] = label;
        }
    }
}

/* Volatility-adjusted labels: threshold = k_sigma × annualised rolling vol.
   Uses rolling std of log-returns over vol_lookback bars to compute per-stock
   threshold at each time step. NaN for first vol_lookback bars and last horizon rows. */
/* ─── rolling argmax ────────────────────────────────────────────────────── */

/* Returns days-ago offset of the max within [t-period+1..t].
   0 = today is the max, period-1 = oldest bar is the max. NaN for warmup. */
void quant_rolling_argmax(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            int   best = t;
            float mx   = x[IDX(t, n, N)];
            for (int k = t - 1; k >= t - period + 1; k--) {
                float v = x[IDX(k, n, N)];
                if (v > mx) { mx = v; best = k; }
            }
            o[IDX(t, n, N)] = (float)(t - best);
        }
    }
}

/* ─── rolling argmin ────────────────────────────────────────────────────── */

/* Returns days-ago offset of the min within [t-period+1..t]. */
void quant_rolling_argmin(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            int   best = t;
            float mn   = x[IDX(t, n, N)];
            for (int k = t - 1; k >= t - period + 1; k--) {
                float v = x[IDX(k, n, N)];
                if (v < mn) { mn = v; best = k; }
            }
            o[IDX(t, n, N)] = (float)(t - best);
        }
    }
}

/* ─── rolling time-series rank ──────────────────────────────────────────── */

/* Percentile rank of X[t,n] within the rolling window [t-period+1..t].
   Result in [0, 1]: 0 = lowest value in window, 1 = highest. */
void quant_rolling_rank(const void *X, void *out, int T, int N, int period)
{
    const float *x = cf32(X);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float cur   = x[IDX(t, n, N)];
            int   below = 0;
            for (int k = t - period + 1; k <= t; k++) {
                if (x[IDX(k, n, N)] < cur) below++;
            }
            o[IDX(t, n, N)] = (period > 1) ? (float)below / (float)(period - 1) : 0.5f;
        }
    }
}

/* ─── higher-low score ──────────────────────────────────────────────────── */

/* Fraction of lows in [t-period+1..t-1] that are BELOW low[t].
   High score = current low is a new high → good consolidation / base quality. */
void quant_higher_low_score(const void *low, void *out, int T, int N, int period)
{
    const float *l = cf32(low);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period) { o[IDX(t, n, N)] = NAN; continue; }
            float cur_lo = l[IDX(t, n, N)];
            int   higher = 0, total = 0;
            for (int k = t - period + 1; k < t; k++) {
                float v = l[IDX(k, n, N)];
                if (!isnan(v)) {
                    if (v < cur_lo) higher++;
                    total++;
                }
            }
            o[IDX(t, n, N)] = total > 0 ? (float)higher / total : NAN;
        }
    }
}

/* ─── consolidation tightness ───────────────────────────────────────────── */

/* 1 - CV(close over period), clamped to [0, 1].
   High value = tight price action = quality base / consolidation. */
void quant_consolidation_tightness(const void *close, void *out,
                                    int T, int N, int period)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t < period - 1) { o[IDX(t, n, N)] = NAN; continue; }
            float sum = 0.f, sum2 = 0.f;
            int   cnt = 0;
            for (int k = t - period + 1; k <= t; k++) {
                float v = c[IDX(k, n, N)];
                if (!isnan(v) && v > 0.f) { sum += v; sum2 += v * v; cnt++; }
            }
            if (cnt < 2) { o[IDX(t, n, N)] = NAN; continue; }
            float mean = sum / cnt;
            float var  = sum2 / cnt - mean * mean;
            float std  = sqrtf(var > 0.f ? var : 0.f);
            float cv   = (mean > 0.f) ? std / mean : 1.f;
            /* Scale: CV=0 → 1.0 (perfectly tight), CV=0.1 → 0.0 (10%+ range = loose) */
            o[IDX(t, n, N)] = fmaxf(0.f, fminf(1.f, 1.f - cv * 10.f));
        }
    }
}

/* ─── vol-adj labels ────────────────────────────────────────────────────── */

void quant_generate_labels_vol_adj(const void *close, void *out,
                                    int T, int N, int horizon,
                                    float k_sigma, int vol_lookback)
{
    const float *c = cf32(close);
    float       *o = f32(out);

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 0; t < T; t++) {
            if (t + horizon >= T) { o[IDX(t, n, N)] = NAN; continue; }

            /* rolling vol: std of log-returns over [t-vol_lookback+1, t] */
            int win_start = t - vol_lookback + 1;
            if (win_start < 1) { o[IDX(t, n, N)] = NAN; continue; }

            float sum = 0.f, sum2 = 0.f;
            int   cnt = 0;
            for (int k = win_start; k <= t; k++) {
                float prev = c[IDX(k - 1, n, N)];
                float cur  = c[IDX(k,     n, N)];
                if (prev > 0.f && cur > 0.f) {
                    float r = logf(cur / prev);
                    sum  += r;
                    sum2 += r * r;
                    cnt++;
                }
            }
            if (cnt < 3) { o[IDX(t, n, N)] = NAN; continue; }

            float mean         = sum / cnt;
            float var          = sum2 / cnt - mean * mean;
            float vol_week     = sqrtf(var > 0.f ? var : 0.f);
            /* Scale to the holding horizon, not to a year.
               A 30%-annual-vol stock has ~9.8% 8-week vol → achievable target. */
            float vol_horizon  = vol_week * sqrtf((float)horizon);
            float threshold    = k_sigma * vol_horizon;
            if (threshold < 0.03f) threshold = 0.03f;   /* floor: 3% min move */
            if (threshold > 0.30f) threshold = 0.30f;   /* ceil:  30% max */

            float curr   = c[IDX(t, n, N)];
            if (curr <= 0.f) { o[IDX(t, n, N)] = NAN; continue; }
            float target = curr * (1.f + threshold);

            float label = 0.f;
            for (int k = 1; k <= horizon; k++) {
                if (c[IDX(t + k, n, N)] >= target) { label = 1.f; break; }
            }
            o[IDX(t, n, N)] = label;
        }
    }
}

/* ─── row fraction positive ─────────────────────────────────────────────── */

/* For each row t: fraction of non-NaN columns where mat[t,n] > 0.
   out is [T] float32. Used for cross-sectional breadth computation. */
void quant_row_fraction_positive(const void *mat_v, void *out_v, int T, int N)
{
    const float *mat = cf32(mat_v);
    float       *out = f32(out_v);

    #pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        int pos = 0, cnt = 0;
        const float *row = mat + t * N;
        for (int n = 0; n < N; n++) {
            float v = row[n];
            if (!isnan(v)) { cnt++; if (v > 0.f) pos++; }
        }
        out[t] = cnt > 0 ? (float)pos / cnt : 0.5f;
    }
}

/* ─── turtle strategy rules 1-4 ────────────────────────────────────────── */

/* Computes all four Turtle strategy rules in a single OpenMP-parallel pass.
 *
 * All inputs are [T, N] float32:
 *   close, high52off (argmax(high,252) offsets), low52off (argmin(low,252) offsets),
 *   min252 (rolling_min(low,252)), hh20 (rolling_max(high,20)), sma55h (rolling_mean(high,55))
 *
 * Rules 3 & 4 use LAG-1 values of hh20/sma55h (Donchian breakout definition).
 * Outputs (all [T, N] float32, values 0.0 or 1.0):
 *   rule1..rule4, all_rules (all 4 pass), rules34 (rules 3&4 only, for avg-down gate)
 */
void quant_turtle_rules(
    const void *close_v,   const void *high52off_v, const void *low52off_v,
    const void *min252_v,  const void *hh20_v,      const void *sma55h_v,
    void *rule1_v, void *rule2_v, void *rule3_v, void *rule4_v,
    void *all_v,   void *r34_v,
    int T, int N, float zone_min, float zone_max
)
{
    const float *close  = cf32(close_v);
    const float *h52off = cf32(high52off_v);
    const float *l52off = cf32(low52off_v);
    const float *min252 = cf32(min252_v);
    const float *hh20   = cf32(hh20_v);
    const float *sma55h = cf32(sma55h_v);
    float *r1 = f32(rule1_v), *r2 = f32(rule2_v);
    float *r3 = f32(rule3_v), *r4 = f32(rule4_v);
    float *ra = f32(all_v),   *r34 = f32(r34_v);

    memset(r1,  0, (size_t)T * N * sizeof(float));
    memset(r2,  0, (size_t)T * N * sizeof(float));
    memset(r3,  0, (size_t)T * N * sizeof(float));
    memset(r4,  0, (size_t)T * N * sizeof(float));
    memset(ra,  0, (size_t)T * N * sizeof(float));
    memset(r34, 0, (size_t)T * N * sizeof(float));

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        for (int t = 1; t < T; t++) {   /* t=0 skipped: needs t-1 for lag */
            int idx  = IDX(t,     n, N);
            int prev = IDX(t - 1, n, N);

            float c     = close[idx];
            float h52o  = h52off[idx];
            float l52o  = l52off[idx];
            float m252  = min252[idx];
            float hh20v = hh20[prev];   /* lag-1 Donchian breakout level */
            float s55hv = sma55h[prev]; /* lag-1 55-day SMA of high      */

            if (isnan(c) || isnan(h52o) || isnan(l52o) ||
                isnan(m252) || isnan(hh20v) || isnan(s55hv)) continue;

            /* Rule 1: 52w low date > 52w high date (low is more recent) */
            float r1v = (l52o < h52o) ? 1.f : 0.f;

            /* Rule 2: price 20%–30% above 52w low */
            float recov = (m252 > 0.f) ? (c - m252) / m252 : 0.f;
            float r2v   = (recov >= zone_min && recov <= zone_max) ? 1.f : 0.f;

            /* Rule 3: today's close >= yesterday's 20-day highest high */
            float r3v = (c >= hh20v) ? 1.f : 0.f;

            /* Rule 4: yesterday's 20-day high > yesterday's 55-day SMA of high */
            float r4v = (hh20v > s55hv) ? 1.f : 0.f;

            r1[idx]  = r1v;
            r2[idx]  = r2v;
            r3[idx]  = r3v;
            r4[idx]  = r4v;
            ra[idx]  = (r1v > .5f && r2v > .5f && r3v > .5f && r4v > .5f) ? 1.f : 0.f;
            r34[idx] = (r3v > .5f && r4v > .5f) ? 1.f : 0.f;
        }
    }
}
