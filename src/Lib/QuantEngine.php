<?php
declare(strict_types=1);

namespace Pml\Lib;

use Pml\Tensor;

/**
 * FFI bridge for libquant.so — financial time-series C kernels.
 *
 * All kernels operate on [T, N] row-major float32 tensors where
 * T = time steps and N = stocks. Outputs are written into pre-allocated
 * Tensor objects; no allocation happens inside C.
 *
 * Usage:
 *   QuantEngine::rollingMean($close, $out, 20); // SMA-20
 */
final class QuantEngine
{
    private static ?\FFI $ffi = null;

    public static function get(): \FFI
    {
        if (self::$ffi !== null) return self::$ffi;

        $libPath = __DIR__ . '/libquant.so';
        $srcPath = __DIR__ . '/quant.c';

        // §37 — filemtime() is now inside the singleton guard (self::$ffi !== null check
        // is at the top of get()), so this block runs exactly once per PHP process.
        // §36/§7 — never auto-compile in production
        if (!file_exists($libPath)) {
            if (getenv('PML_ENV') === 'production') {
                throw new \RuntimeException(
                    '[QuantEngine] libquant.so not found. Pre-build it before deploying.'
                );
            }
            $result = shell_exec(
                'gcc -O3 -march=native -mtune=native -mfma -fno-math-errno'
                . ' -funsafe-math-optimizations -fopenmp -funroll-loops'
                . ' -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC'
                . ' -o ' . escapeshellarg($libPath)
                . ' ' . escapeshellarg($srcPath)
                . ' -lm 2>&1'
            );
            if (!file_exists($libPath)) {
                throw new \RuntimeException("[QuantEngine] Build failed:\n" . (string)$result);
            }
        } elseif (filemtime($srcPath) > filemtime($libPath)) {
            // quant.c is newer than the .so — rebuild only in dev
            if (getenv('PML_ENV') !== 'production') {
                shell_exec(
                    'gcc -O3 -march=native -mtune=native -mfma -fno-math-errno'
                    . ' -funsafe-math-optimizations -fopenmp -funroll-loops'
                    . ' -fomit-frame-pointer -D_GNU_SOURCE -shared -fPIC'
                    . ' -o ' . escapeshellarg($libPath)
                    . ' ' . escapeshellarg($srcPath)
                    . ' -lm 2>&1'
                );
            }
        }

        self::$ffi = \FFI::cdef('
            void quant_log_returns(const void *close, void *out, int T, int N);
            void quant_momentum(const void *close, void *out, int T, int N, int lookback);
            void quant_ema(const void *X, void *out, int T, int N, int period);
            void quant_rolling_mean(const void *X, void *out, int T, int N, int period);
            void quant_rolling_std(const void *X, void *out, int T, int N, int period);
            void quant_rolling_max(const void *X, void *out, int T, int N, int period);
            void quant_rolling_min(const void *X, void *out, int T, int N, int period);
            void quant_atr(const void *high, const void *low, const void *close,
                           void *out, int T, int N, int period);
            void quant_rsi(const void *close, void *out, int T, int N, int period);
            void quant_efficiency_ratio(const void *close, void *out, int T, int N, int period);
            void quant_rolling_beta(const void *stock_ret, const void *bench_ret,
                                    void *out, int T, int N, int period);
            void quant_rolling_max_drawdown(const void *close, void *out,
                                             int T, int N, int period);
            void quant_downside_std(const void *returns, void *out, int T, int N, int period);
            void quant_rank_normalize(const void *X, void *out, int T, int N);
            void quant_zscore_normalize(const void *X, void *out, int T, int N);
            void quant_adx(const void *high, const void *low, const void *close,
                            void *out, int T, int N, int period);
            void quant_macd_histogram(const void *close, void *out, int T, int N,
                                        int fast, int slow, int signal_period);
            void quant_generate_labels(const void *close, void *out,
                                        int T, int N, int horizon, float threshold);
            void quant_generate_labels_vol_adj(const void *close, void *out,
                                        int T, int N, int horizon,
                                        float k_sigma, int vol_lookback);
            void quant_write_factor_col(const void *src, void *dst,
                                         int col, int TN, int F, float nan_fill);
            void quant_weighted_composite(const void *rank_mat, const float *weights,
                                           void *out, int TN, int F);
            void quant_row_mean(const void *mat, void *out, int T, int N);
            void quant_rolling_argmax(const void *X, void *out, int T, int N, int period);
            void quant_rolling_argmin(const void *X, void *out, int T, int N, int period);
            void quant_rolling_rank(const void *X, void *out, int T, int N, int period);
            void quant_higher_low_score(const void *low, void *out, int T, int N, int period);
            void quant_consolidation_tightness(const void *close, void *out, int T, int N, int period);
            void quant_row_fraction_positive(const void *mat, void *out, int T, int N);
            void quant_turtle_rules(
                const void *close,     const void *high52off, const void *low52off,
                const void *min252low, const void *hh20,      const void *sma55h,
                void *rule1, void *rule2, void *rule3, void *rule4,
                void *all_rules, void *rules34,
                int T, int N, float zone_min, float zone_max);
        ', $libPath);

        return self::$ffi;
    }

    /* ── convenience wrappers ─────────────────────────────────────────── */

    private static function alloc2D(int $T, int $N): Tensor
    {
        return new Tensor([$T, $N]);
    }

    /**
     * Extract void* data pointer from a Tensor.
     * Cast using the QuantEngine FFI context so it is compatible with libquant.so.
     */
    private static function ptr(Tensor $t): \FFI\CData
    {
        return self::get()->cast('void*', $t->ptr->data);
    }

    public static function logReturns(Tensor $close): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_log_returns(self::ptr($close), self::ptr($out), $T, $N);
        return $out;
    }

    public static function momentum(Tensor $close, int $lookback): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_momentum(self::ptr($close), self::ptr($out), $T, $N, $lookback);
        return $out;
    }

    public static function ema(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_ema(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    public static function rollingMean(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_mean(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    public static function rollingStd(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_std(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    public static function rollingMax(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_max(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    public static function rollingMin(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_min(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    public static function atr(Tensor $high, Tensor $low, Tensor $close, int $period): Tensor
    {
        [$T, $N] = [$high->shape()[0], $high->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_atr(
            self::ptr($high), self::ptr($low), self::ptr($close),
            self::ptr($out), $T, $N, $period
        );
        return $out;
    }

    public static function rsi(Tensor $close, int $period): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rsi(self::ptr($close), self::ptr($out), $T, $N, $period);
        return $out;
    }

    public static function efficiencyRatio(Tensor $close, int $period): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_efficiency_ratio(
            self::ptr($close), self::ptr($out), $T, $N, $period
        );
        return $out;
    }

    /**
     * Rolling beta vs 1-D benchmark return series.
     * @param Tensor $benchRet shape [T] or [T,1] — flattened as [T]
     */
    public static function rollingBeta(Tensor $stockRet, Tensor $benchRet, int $period): Tensor
    {
        [$T, $N] = [$stockRet->shape()[0], $stockRet->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_beta(
            self::ptr($stockRet), self::ptr($benchRet),
            self::ptr($out), $T, $N, $period
        );
        return $out;
    }

    public static function rollingMaxDrawdown(Tensor $close, int $period): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_max_drawdown(
            self::ptr($close), self::ptr($out), $T, $N, $period
        );
        return $out;
    }

    public static function downsideStd(Tensor $returns, int $period): Tensor
    {
        [$T, $N] = [$returns->shape()[0], $returns->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_downside_std(
            self::ptr($returns), self::ptr($out), $T, $N, $period
        );
        return $out;
    }

    /** Cross-sectional rank normalization to [0,1] per row. */
    public static function rankNormalize(Tensor $X): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rank_normalize(self::ptr($X), self::ptr($out), $T, $N);
        return $out;
    }

    /** Cross-sectional z-score (winsorized ±3σ) per row. */
    public static function zscoreNormalize(Tensor $X): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_zscore_normalize(self::ptr($X), self::ptr($out), $T, $N);
        return $out;
    }

    /** ADX trend-strength indicator [0, 100]. NaN for first 2×period bars. */
    public static function adx(Tensor $high, Tensor $low, Tensor $close, int $period): Tensor
    {
        [$T, $N] = [$high->shape()[0], $high->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_adx(
            self::ptr($high), self::ptr($low), self::ptr($close),
            self::ptr($out), $T, $N, $period
        );
        return $out;
    }

    /** MACD histogram: EMA(fast) − EMA(slow) − signal_EMA(MACD). */
    public static function macdHistogram(
        Tensor $close, int $fast = 12, int $slow = 26, int $signal = 9
    ): Tensor {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_macd_histogram(
            self::ptr($close), self::ptr($out), $T, $N, $fast, $slow, $signal
        );
        return $out;
    }

    /** Generate forward-looking binary labels. */
    public static function generateLabels(Tensor $close, int $horizon, float $threshold): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_generate_labels(
            self::ptr($close), self::ptr($out), $T, $N, $horizon, $threshold
        );
        return $out;
    }

    /**
     * Volatility-adjusted labels: label=1 if stock gains > k_sigma × annualised vol
     * within horizon bars. Each stock gets its own threshold based on its rolling vol.
     */
    public static function generateLabelsVolAdj(
        Tensor $close, int $horizon, float $kSigma, int $volLookback
    ): Tensor {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_generate_labels_vol_adj(
            self::ptr($close), self::ptr($out), $T, $N, $horizon, $kSigma, $volLookback
        );
        return $out;
    }

    /**
     * Write one [T,N] factor into column `col` of a pre-allocated [TN, F] Tensor.
     * Replaces the PHP interleave loop in FactorEngine — no data crosses FFI boundary.
     */
    public static function writeFactorCol(
        Tensor $src, Tensor $dst, int $col, int $TN, int $F, float $nanFill = 0.0
    ): void {
        self::get()->quant_write_factor_col(
            self::ptr($src), self::ptr($dst), $col, $TN, $F, $nanFill
        );
    }

    /**
     * Weighted composite score: rank_mat [TN, F] × weights float[F] → [T, N] Tensor.
     * Negative weight inverts the factor score: |w| × (1 − val).
     * Replaces the T×N×F PHP loop in RankAggregator.
     */
    public static function weightedComposite(
        Tensor $rankMat, array $wArr, int $T, int $N, int $F
    ): Tensor {
        $ffi  = self::get();
        $wBuf = $ffi->new("float[$F]");
        for ($i = 0; $i < $F; $i++) $wBuf[$i] = (float)($wArr[$i] ?? 0.0);
        $out  = self::alloc2D($T, $N);
        $ffi->quant_weighted_composite(self::ptr($rankMat), $wBuf, self::ptr($out), $T * $N, $F);
        return $out;
    }

    /**
     * Row-wise mean across N stocks: mat [T, N] → [T] Tensor.
     * Used for breadth (fraction of stocks above SMA20).
     */
    public static function rowMean(Tensor $mat, int $T, int $N): Tensor
    {
        $out = new Tensor([$T]);
        self::get()->quant_row_mean(self::ptr($mat), self::ptr($out), $T, $N);
        return $out;
    }

    /**
     * Rolling argmax: days-ago offset of max within window. [T, N] → [T, N].
     * 0 = today is max, period-1 = oldest bar is max. NaN for warmup.
     */
    public static function rollingArgmax(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_argmax(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    /**
     * Rolling argmin: days-ago offset of min within window. [T, N] → [T, N].
     */
    public static function rollingArgmin(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_argmin(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    /**
     * Time-series percentile rank of each value within its rolling window. [0, 1].
     */
    public static function rollingRank(Tensor $X, int $period): Tensor
    {
        [$T, $N] = [$X->shape()[0], $X->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_rolling_rank(self::ptr($X), self::ptr($out), $T, $N, $period);
        return $out;
    }

    /**
     * Higher-low score: fraction of prior lows in window below current low.
     * High = base quality / accumulation.
     */
    public static function higherLowScore(Tensor $low, int $period): Tensor
    {
        [$T, $N] = [$low->shape()[0], $low->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_higher_low_score(self::ptr($low), self::ptr($out), $T, $N, $period);
        return $out;
    }

    /**
     * Consolidation tightness: 1 - CV(close, period) clamped [0, 1].
     * High = tight price action = quality base before breakout.
     */
    public static function consolidationTightness(Tensor $close, int $period): Tensor
    {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $out = self::alloc2D($T, $N);
        self::get()->quant_consolidation_tightness(self::ptr($close), self::ptr($out), $T, $N, $period);
        return $out;
    }

    /**
     * Cross-sectional fraction of non-NaN positive values per row.
     * mat [T, N] → out [T]. Used for market breadth (% stocks above SMA).
     */
    public static function rowFractionPositive(Tensor $mat, int $T, int $N): Tensor
    {
        $out = new Tensor([$T]);
        self::get()->quant_row_fraction_positive(self::ptr($mat), self::ptr($out), $T, $N);
        return $out;
    }

    /**
     * All 4 Turtle strategy rules in one OpenMP-parallel pass.
     * Rules 3 & 4 use lag-1 hh20/sma55h (Donchian breakout definition).
     *
     * @return array{rule1:Tensor, rule2:Tensor, rule3:Tensor, rule4:Tensor,
     *               all_rules:Tensor, rules34:Tensor}  each [T, N] float32 (0.0 or 1.0)
     */
    public static function turtleRules(
        Tensor $close, Tensor $high52off, Tensor $low52off,
        Tensor $min252, Tensor $hh20, Tensor $sma55h,
        float $zoneMin = 0.20, float $zoneMax = 0.30
    ): array {
        [$T, $N] = [$close->shape()[0], $close->shape()[1]];
        $r1 = self::alloc2D($T, $N); $r2 = self::alloc2D($T, $N);
        $r3 = self::alloc2D($T, $N); $r4 = self::alloc2D($T, $N);
        $ra = self::alloc2D($T, $N); $r34= self::alloc2D($T, $N);
        self::get()->quant_turtle_rules(
            self::ptr($close),    self::ptr($high52off), self::ptr($low52off),
            self::ptr($min252),   self::ptr($hh20),      self::ptr($sma55h),
            self::ptr($r1), self::ptr($r2), self::ptr($r3), self::ptr($r4),
            self::ptr($ra), self::ptr($r34),
            $T, $N, $zoneMin, $zoneMax
        );
        return ['rule1' => $r1, 'rule2' => $r2, 'rule3' => $r3, 'rule4' => $r4,
                'all_rules' => $ra, 'rules34' => $r34];
    }
}
