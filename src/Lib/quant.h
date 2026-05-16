#pragma once
#ifdef __cplusplus
extern "C" {
#endif

/*
 * quant.h — Financial time-series C kernels for [T, N] row-major float32 layout
 * T = time steps, N = stocks
 * All functions operate on pre-allocated void* buffers (no malloc).
 * Parameters named `bench` are 1-D [T] float32 series.
 * NaN is written for warm-up periods where the window is incomplete.
 */

/* Log returns: out[t,n] = log(close[t,n] / close[t-1,n]). out[0,n]=NaN. */
void quant_log_returns(const void *close, void *out, int T, int N);

/* Simple momentum: out[t,n] = close[t,n]/close[t-lb,n] - 1. NaN for t<lb. */
void quant_momentum(const void *close, void *out, int T, int N, int lookback);

/* EMA with alpha = 2/(period+1). Runs from first bar; no NaN warm-up. */
void quant_ema(const void *X, void *out, int T, int N, int period);

/* Rolling SMA. NaN for first (period-1) bars. */
void quant_rolling_mean(const void *X, void *out, int T, int N, int period);

/* Rolling population std-dev. NaN for first (period-1) bars. */
void quant_rolling_std(const void *X, void *out, int T, int N, int period);

/* Rolling maximum over window. NaN for first (period-1) bars. */
void quant_rolling_max(const void *X, void *out, int T, int N, int period);

/* Rolling minimum over window. NaN for first (period-1) bars. */
void quant_rolling_min(const void *X, void *out, int T, int N, int period);

/* ATR (Wilder EMA of True Range). out[0,n]=high[0]-low[0] as seed. */
void quant_atr(const void *high, const void *low, const void *close,
               void *out, int T, int N, int period);

/* Wilder RSI. NaN for first `period` bars. */
void quant_rsi(const void *close, void *out, int T, int N, int period);

/* Kaufman Efficiency Ratio: direction/path over window. NaN for t<period. */
void quant_efficiency_ratio(const void *close, void *out, int T, int N, int period);

/* Rolling beta of each stock vs 1-D benchmark return series.
   bench_ret points to a float32[T] array. NaN for t<period-1. */
void quant_rolling_beta(const void *stock_ret, const void *bench_ret,
                        void *out, int T, int N, int period);

/* Rolling maximum drawdown within window. NaN for t<period-1. */
void quant_rolling_max_drawdown(const void *close, void *out,
                                int T, int N, int period);

/* Rolling downside std-dev (only negative returns). NaN for t<period-1. */
void quant_downside_std(const void *returns, void *out, int T, int N, int period);

/* Cross-sectional rank normalization to [0,1] per row (ignores NaN cells). */
void quant_rank_normalize(const void *X, void *out, int T, int N);

/* Cross-sectional z-score with ±3σ winsorization per row. NaN imputed as 0. */
void quant_zscore_normalize(const void *X, void *out, int T, int N);

/* Forward-looking binary labels: 1 if any close in (t, t+horizon] >= close[t]*(1+thr).
   NaN for last `horizon` rows. */
void quant_generate_labels(const void *close, void *out,
                            int T, int N, int horizon, float threshold);

/* Volatility-adjusted labels: threshold = k_sigma × annualised rolling vol per stock.
   NaN for first vol_lookback bars and last horizon rows. */
void quant_generate_labels_vol_adj(const void *close, void *out,
                                    int T, int N, int horizon,
                                    float k_sigma, int vol_lookback);

/* ADX trend strength [0,100]. NaN for first (2×period + 1) bars. */
void quant_adx(const void *high, const void *low, const void *close,
               void *out, int T, int N, int period);

/* MACD histogram: EMA(fast) − EMA(slow) − signal_EMA(MACD). */
void quant_macd_histogram(const void *close, void *out, int T, int N,
                           int fast, int slow, int signal_period);

/* Write a single factor's [TN] data into column `col` of a [TN, F] interleaved
   buffer. NaN in src is replaced with nan_fill. No PHP array work required. */
void quant_write_factor_col(const void *src, void *dst,
                             int col, int TN, int F, float nan_fill);

/* Weighted composite score: rank_mat [TN, F] × weights [F] → out [TN].
   Negative weight ⟹ score = |w| × (1 − val). NaN cells treated as 0.5. */
void quant_weighted_composite(const void *rank_mat, const float *weights,
                               void *out, int TN, int F);

/* Row-wise mean across N columns: mat [T, N] → out [T].
   NaN cells are skipped; fraction = sum(valid) / count(valid). */
void quant_row_mean(const void *mat, void *out, int T, int N);

/* Rolling argmax: days-ago offset of max in [t-period+1..t]. 0=today is max.
   NaN for first (period-1) bars. */
void quant_rolling_argmax(const void *X, void *out, int T, int N, int period);

/* Rolling argmin: days-ago offset of min in [t-period+1..t]. 0=today is min.
   NaN for first (period-1) bars. */
void quant_rolling_argmin(const void *X, void *out, int T, int N, int period);

/* Time-series percentile rank of X[t,n] within rolling window [t-period+1..t].
   Result in [0,1]. NaN for first (period-1) bars. */
void quant_rolling_rank(const void *X, void *out, int T, int N, int period);

/* Higher-low score: fraction of lows in window that are below low[t].
   High = current low is higher than most of the window = base quality.
   NaN for first period bars. */
void quant_higher_low_score(const void *low, void *out, int T, int N, int period);

/* Consolidation tightness: 1 - CV(close over period), clamped [0,1].
   High = tight price range = quality base. NaN for first (period-1) bars. */
void quant_consolidation_tightness(const void *close, void *out,
                                    int T, int N, int period);

/* Cross-sectional fraction of non-NaN positive values per row.
   mat [T,N] → out [T]. Used for market breadth (% above SMA). */
void quant_row_fraction_positive(const void *mat, void *out, int T, int N);

/* All 4 Turtle strategy rules in one OpenMP pass. Rules 3 & 4 use lag-1
   hh20/sma55h (Donchian breakout definition). Outputs: 0.0 or 1.0 per cell.
   rule1..rule4, all_rules (1 iff all 4 pass), rules34 (rules 3&4 only). */
void quant_turtle_rules(
    const void *close,     const void *high52off, const void *low52off,
    const void *min252low, const void *hh20,      const void *sma55h,
    void *rule1, void *rule2, void *rule3, void *rule4,
    void *all_rules, void *rules34,
    int T, int N, float zone_min, float zone_max);

#ifdef __cplusplus
}
#endif
