<?php

declare(strict_types=1);

namespace Pml\Tests\Core;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  TestRunner — Dependency-free CLI test orchestrator for Pml
//
//  Design goals:
//    - Zero external dependencies (no PHPUnit, no Pest)
//    - Explicit GC and memory tracking around each test suite to detect
//      PHP-heap AND C-heap (FFI) leaks
//    - ANSI-coloured terminal output: ✓ green, ✗ red, ⊘ yellow (skip)
//    - Final test_report.md summarising results, timings, and RAM deltas
//
//  ── Memory Tracking Strategy ─────────────────────────────────────────────
//
//  PHP's memory_get_usage() tracks the PHP heap only — FFI buffers allocated
//  via \FFI::new() live in the C heap and are invisible to it.
//
//  To catch FFI leaks we additionally read VmRSS from /proc/self/status
//  (Linux only), which is the actual Resident Set Size of the process:
//    PHP heap + FFI C buffers + shared library code + stack
//
//  Strategy per suite:
//    1. gc_collect_cycles()   — force PHP cyclic GC to clean up freed objects
//    2. Record $memBefore = memory_get_usage(true)  (PHP heap, system-granular)
//    3. Record $rssBefore  = /proc/self/status VmRSS (full process RSS)
//    4. Run all tests in the suite
//    5. gc_collect_cycles()   — force GC again to flush the suite's allocations
//    6. Record $memAfter, $rssAfter
//    7. Report delta: ($memAfter - $memBefore) and ($rssAfter - $rssBefore)
//
//  A large positive RSS delta after GC strongly indicates a C-heap (FFI) leak.
//  A large positive PHP-heap delta after GC indicates a reference cycle or
//  a static-property accumulation.
//
//  ── Suite/Test Structure ─────────────────────────────────────────────────
//
//  $runner->suite('Suite Name', function(TestRunner $r) {
//      $r->test('Test description', function() use ($r) {
//          // assertions here
//          $r->assertGreaterThan($acc, 0.90, 'accuracy > 90%');
//      });
//  });
//  $runner->finish();
//
//  Assertions throw AssertionError on failure.  All Throwables are caught
//  by test() so a single failure does not abort the suite.
// ═══════════════════════════════════════════════════════════════════════════

final class TestRunner
{
    // ── ANSI colour codes ─────────────────────────────────────────────────
    private const COL_GREEN  = "\033[32m";
    private const COL_RED    = "\033[31m";
    private const COL_YELLOW = "\033[33m";
    private const COL_CYAN   = "\033[36m";
    private const COL_BOLD   = "\033[1m";
    private const COL_DIM    = "\033[2m";
    private const COL_RESET  = "\033[0m";

    // ── State ─────────────────────────────────────────────────────────────
    private int   $totalPassed  = 0;
    private int   $totalFailed  = 0;
    private int   $totalSkipped = 0;
    private float $totalStartMs;

    // Accumulated log for test_report.md
    private string $reportBuf = '';

    // Current suite metadata (set at the start of each suite() call)
    private string $currentSuite  = '';
    private int    $suiteMemBefore = 0;
    private int    $suiteRssBefore = 0;

    // All individual test records for the report
    private array $records = [];   // [{suite, name, status, detail, ms, phpDeltaKB}, ...]

    private readonly bool $colorsEnabled;

    public function __construct(
        private readonly string $reportPath = 'test_report.md',
    ) {
        // Only emit ANSI codes when writing to a real terminal
        $this->colorsEnabled = function_exists('posix_isatty') && posix_isatty(STDOUT);
        $this->totalStartMs  = $this->nowMs();

        $this->printHeader();
    }

    // ── Suite ──────────────────────────────────────────────────────────────

    /**
     * Run a named test suite.
     *
     * Memory snapshot is taken before and after the suite (with forced GC),
     * giving a per-suite PHP-heap delta and process-RSS delta.
     *
     * @param callable $fn  function(TestRunner $runner): void
     */
    public function suite(string $name, callable $fn): void
    {
        $this->currentSuite = $name;

        // ── Pre-suite: force GC, then snapshot ────────────────────────
        gc_collect_cycles();
        $this->suiteMemBefore = memory_get_usage(true);   // PHP heap (page-granular)
        $this->suiteRssBefore = $this->readRssBytes();    // OS process RSS

        $this->println('');
        $this->println($this->bold($this->col(self::COL_CYAN, "Suite: {$name}")));

        $this->reportBuf .= "\n## Suite: {$name}\n\n";

        // ── Run suite ─────────────────────────────────────────────────
        $fn($this);

        // ── Post-suite: force GC, then snapshot ───────────────────────
        //
        // Why gc_collect_cycles() here?
        //   Closures registered as $_backward on Tensor objects form reference
        //   cycles (closure → Tensor → closure) that refcount-GC cannot break.
        //   cc() sweeps the cycle-detector, freeing those tensors and their FFI
        //   buffers before we take the final memory snapshot.
        gc_collect_cycles();
        $memAfter = memory_get_usage(true);
        $rssAfter = $this->readRssBytes();

        $phpDeltaMiB = ($memAfter - $this->suiteMemBefore) / 1048576;
        $rssDeltaMiB = ($rssAfter  - $this->suiteRssBefore) / 1048576;

        // Flag concerning leaks
        $phpWarn = $phpDeltaMiB > 10.0;
        $rssWarn = $rssDeltaMiB > 20.0;

        $phpStr = sprintf('%+.1f MiB', $phpDeltaMiB);
        $rssStr = $this->suiteRssBefore > 0
            ? sprintf(', RSS %+.1f MiB', $rssDeltaMiB)
            : '';

        $memLine = "  " . $this->col(self::COL_DIM, "Suite RAM delta: ") . $phpStr . $rssStr;
        if ($phpWarn || $rssWarn) {
            $memLine .= " " . $this->col(self::COL_YELLOW, "⚠ possible leak");
        }
        $this->println($memLine);

        $this->reportBuf .= sprintf(
            "\n> **Memory delta:** PHP heap %+.1f MiB%s\n",
            $phpDeltaMiB,
            $this->suiteRssBefore > 0 ? sprintf(', RSS %+.1f MiB', $rssDeltaMiB) : ''
        );
    }

    // ── Test ───────────────────────────────────────────────────────────────

    /**
     * Run a single named test.
     *
     * Captures wall-clock time and PHP-heap delta.
     * Catches all Throwables — a failing test does not abort the suite.
     *
     * @param callable $fn  function(): void — may call $runner->assert*()
     */
    public function test(string $name, callable $fn): void
    {
        $memBefore = memory_get_usage(true);
        $tStart    = $this->nowMs();
        $status    = 'PASS';
        $detail    = '';

        try {
            $fn();
        } catch (SkipException $e) {
            $status = 'SKIP';
            $detail = $e->getMessage();
        } catch (\Throwable $e) {
            $status = 'FAIL';
            // Include class and message; strip stack trace for readability
            $detail = get_class($e) . ': ' . $e->getMessage();
        }

        $ms        = round($this->nowMs() - $tStart, 1);
        $memAfter  = memory_get_usage(true);
        $phpDeltaKB = ($memAfter - $memBefore) / 1024;

        $memStr = sprintf('[%.0fms, %+.0f KB]', $ms, $phpDeltaKB);

        match ($status) {
            'PASS' => $this->printPass($name, $detail, $memStr),
            'FAIL' => $this->printFail($name, $detail, $memStr),
            'SKIP' => $this->printSkip($name, $detail, $memStr),
        };

        match ($status) {
            'PASS' => $this->totalPassed++,
            'FAIL' => $this->totalFailed++,
            'SKIP' => $this->totalSkipped++,
        };

        $this->records[] = [
            'suite'      => $this->currentSuite,
            'name'       => $name,
            'status'     => $status,
            'detail'     => $detail,
            'ms'         => $ms,
            'phpDeltaKB' => $phpDeltaKB,
        ];
    }

    // ── Assertions ─────────────────────────────────────────────────────────

    /**
     * Assert strict equality (===).
     */
    public function assertEq(mixed $actual, mixed $expected, string $msg = ''): void
    {
        if ($actual !== $expected) {
            $a = $this->repr($actual);
            $e = $this->repr($expected);
            $this->fail("assertEq failed{$this->context($msg)}: got {$a}, expected {$e}");
        }
    }

    /**
     * Assert |actual - expected| ≤ atol.
     *
     * Use this for floating-point results where exact equality is not meaningful.
     * For scores and coefficients, atol should reflect acceptable estimation error.
     */
    public function assertFloatClose(float $actual, float $expected, float $atol, string $msg = ''): void
    {
        $diff = abs($actual - $expected);
        if ($diff > $atol) {
            $this->fail(sprintf(
                "assertFloatClose failed%s: |%.6f - %.6f| = %.6f > atol=%.6f",
                $this->context($msg), $actual, $expected, $diff, $atol
            ));
        }
    }

    /**
     * Assert that $actual > $threshold.
     *
     * Typical usage: accuracy, R², or any score that must exceed a bar.
     */
    public function assertGreaterThan(float $actual, float $threshold, string $msg = ''): void
    {
        if ($actual <= $threshold) {
            $this->fail(sprintf(
                "assertGreaterThan failed%s: %.6f ≤ %.6f",
                $this->context($msg), $actual, $threshold
            ));
        }
    }

    /**
     * Assert that $actual < $threshold (e.g. a loss value is below a ceiling).
     */
    public function assertLessThan(float $actual, float $threshold, string $msg = ''): void
    {
        if ($actual >= $threshold) {
            $this->fail(sprintf(
                "assertLessThan failed%s: %.6f ≥ %.6f",
                $this->context($msg), $actual, $threshold
            ));
        }
    }

    /**
     * Assert that a Tensor has the expected shape.
     *
     * @param array<int> $expectedShape
     */
    public function assertShape(Tensor $t, array $expectedShape, string $msg = ''): void
    {
        if ($t->shape !== $expectedShape) {
            $got = '[' . implode(', ', $t->shape) . ']';
            $exp = '[' . implode(', ', $expectedShape) . ']';
            $this->fail("assertShape failed{$this->context($msg)}: got {$got}, expected {$exp}");
        }
    }

    /**
     * Assert that the loss strictly decreased from the first to the last epoch.
     *
     * This is the key correctness proof for the autograd + optimizer system:
     *   - A decreasing loss proves that backprop computed the correct gradient
     *     direction (loss goes DOWN when we step in -∇ direction).
     *   - It proves AdamW::step() correctly updated the weights.
     *   - It proves AdamW::zeroGrad() correctly cleared gradients (otherwise
     *     gradients would accumulate and the updates would be wrong).
     *   - It proves CrossEntropyLoss::backward() correctly computed dL/dz.
     *
     * We assert loss[0] > loss[-1], not strict monotonicity, because:
     *   - Early training can have noisy fluctuations on small batches.
     *   - The key requirement is that SOME progress was made, not that
     *     every step was productive.
     *
     * @param float[] $losses  Per-epoch loss values (e.g. MLPClassifier::$loss_curve_)
     */
    public function assertLossDecreases(array $losses, string $msg = ''): void
    {
        if (count($losses) < 2) {
            $this->fail("assertLossDecreases{$this->context($msg)}: need at least 2 loss values, got " . count($losses));
        }

        $first = $losses[0];
        $last  = $losses[count($losses) - 1];

        if ($last >= $first) {
            $this->fail(sprintf(
                "assertLossDecreases failed%s: loss did not decrease (loss[0]=%.4f, loss[-1]=%.4f)",
                $this->context($msg), $first, $last
            ));
        }

        // Optionally surface the improvement in the test output via a note
        // stored in a way that test() can access it — we use a hack here:
        // just let it pass silently; test() will mark it PASS.
    }

    /**
     * Assert that all float[] scores are in [0, 1] (or any valid score range).
     *
     * Useful for cross-validation score arrays where the scoring metric is
     * bounded (e.g. accuracy, R²).
     *
     * @param float[] $scores
     */
    public function assertAllFinite(array $scores, float $lo = 0.0, float $hi = 1.0, string $msg = ''): void
    {
        foreach ($scores as $i => $s) {
            if (!is_finite($s)) {
                $this->fail("assertAllFinite{$this->context($msg)}: scores[{$i}] = {$s} is not finite");
            }
            if ($s < $lo || $s > $hi) {
                $this->fail(sprintf(
                    "assertAllFinite%s: scores[%d] = %.4f outside [%.2f, %.2f]",
                    $this->context($msg), $i, $s, $lo, $hi
                ));
            }
        }
    }

    /**
     * Assert that two prediction float-arrays match exactly element-wise.
     *
     * Used for Joblib serialization tests: the predictions of the original
     * and resurrected model must agree to bit-level precision, because we
     * restored the exact C-memory bytes via \FFI::memcpy.
     *
     * @param float[] $a
     * @param float[] $b
     */
    public function assertArraysMatch(array $a, array $b, string $msg = ''): void
    {
        $la = count($a);
        $lb = count($b);
        if ($la !== $lb) {
            $this->fail("assertArraysMatch{$this->context($msg)}: lengths differ ({$la} vs {$lb})");
        }
        for ($i = 0; $i < $la; $i++) {
            if ($a[$i] !== $b[$i]) {
                $this->fail(sprintf(
                    "assertArraysMatch%s: mismatch at index %d: %.8f != %.8f",
                    $this->context($msg), $i, $a[$i], $b[$i]
                ));
            }
        }
    }

    /**
     * Skip the current test with an explanatory message.
     *
     * Throws SkipException which is caught by test() and logged as SKIP (yellow).
     * Use for tests that require optional libraries (libsvm, libxgboost) that
     * may not be present in every environment.
     */
    public function skip(string $reason): never
    {
        throw new SkipException($reason);
    }

    // ── Finish ─────────────────────────────────────────────────────────────

    /**
     * Print the final summary and write the Markdown report.
     *
     * Call once after all suites have been registered.
     */
    public function finish(): void
    {
        $totalMs = round($this->nowMs() - $this->totalStartMs, 0);
        $peakMiB = memory_get_peak_usage(true) / 1048576;

        $total  = $this->totalPassed + $this->totalFailed + $this->totalSkipped;
        $pLine  = $this->col(self::COL_GREEN,  "{$this->totalPassed} passed");
        $fLine  = $this->col(self::COL_RED,    "{$this->totalFailed} failed");
        $sLine  = $this->col(self::COL_YELLOW, "{$this->totalSkipped} skipped");

        $this->println('');
        $this->println(str_repeat('═', 62));

        if ($this->totalFailed === 0) {
            $summary = $this->col(self::COL_GREEN, $this->bold("ALL TESTS PASSED"));
        } else {
            $summary = $this->col(self::COL_RED, $this->bold("{$this->totalFailed} FAILURE(S)"));
        }

        $this->println("Results: {$summary}  ({$pLine}, {$fLine}, {$sLine})");
        $this->println(sprintf(
            "Total time: %.0f ms | Peak PHP RAM: %.1f MiB",
            $totalMs,
            $peakMiB
        ));
        $this->println(str_repeat('═', 62));

        $this->writeReport($totalMs, $peakMiB);
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /** @throws \RuntimeException — caught by test() and marked FAIL */
    private function fail(string $message): never
    {
        throw new \RuntimeException($message);
    }

    /** Format a message context label */
    private function context(string $msg): string
    {
        return $msg !== '' ? " ({$msg})" : '';
    }

    /** Human-readable repr of a value for error messages */
    private function repr(mixed $v): string
    {
        if (is_array($v)) {
            return '[' . implode(', ', array_map(fn($x) => $this->repr($x), $v)) . ']';
        }
        if (is_float($v)) {
            return number_format($v, 6);
        }
        if (is_bool($v)) {
            return $v ? 'true' : 'false';
        }
        return (string)$v;
    }

    private function printPass(string $name, string $detail, string $memStr): void
    {
        $tick  = $this->col(self::COL_GREEN, '  ✓');
        $dStr  = $detail !== '' ? ": {$detail}" : '';
        $mStr  = $this->col(self::COL_DIM, "  {$memStr}");
        $this->println("{$tick} {$name}{$dStr}{$mStr}");
        $this->reportBuf .= "- ✅ **{$name}**{$dStr}\n";
    }

    private function printFail(string $name, string $detail, string $memStr): void
    {
        $cross = $this->col(self::COL_RED, '  ✗');
        $dStr  = $detail !== '' ? "\n      " . $this->col(self::COL_RED, $detail) : '';
        $mStr  = $this->col(self::COL_DIM, "  {$memStr}");
        $this->println("{$cross} {$name}{$mStr}{$dStr}");
        $this->reportBuf .= "- ❌ **{$name}**: {$detail}\n";
    }

    private function printSkip(string $name, string $detail, string $memStr): void
    {
        $sym  = $this->col(self::COL_YELLOW, '  ⊘');
        $mStr = $this->col(self::COL_DIM, "  {$memStr}");
        $this->println("{$sym} {$name} (skipped: {$detail}){$mStr}");
        $this->reportBuf .= "- ⚠️  **{$name}** (skipped: {$detail})\n";
    }

    private function printHeader(): void
    {
        $line = str_repeat('═', 62);
        $this->println($line);
        $title = $this->bold($this->col(self::COL_CYAN, "  Pml Integration Test Suite"));
        $this->println($title);
        $date  = date('Y-m-d H:i:s');
        $php   = PHP_VERSION;
        $this->println("  PHP {$php} | {$date}");
        $this->println($line);

        $this->reportBuf .= "# Pml Integration Test Report\n\n";
        $this->reportBuf .= "> Generated: {$date} | PHP {$php}\n\n";
    }

    /** Write the final Markdown report to disk. */
    private function writeReport(float $totalMs, float $peakMiB): void
    {
        $passed  = $this->totalPassed;
        $failed  = $this->totalFailed;
        $skipped = $this->totalSkipped;
        $total   = $passed + $failed + $skipped;

        $status = $failed === 0 ? '✅ PASS' : '❌ FAIL';

        $summary = <<<MD

---

## Summary

| Metric | Value |
|--------|-------|
| Status | {$status} |
| Passed | {$passed} / {$total} |
| Failed | {$failed} |
| Skipped | {$skipped} |
| Total time | {$totalMs} ms |
| Peak PHP RAM | {$peakMiB} MiB |

### Per-test results

| Suite | Test | Status | Time (ms) | PHP Δ (KB) |
|-------|------|--------|-----------|------------|

MD;

        foreach ($this->records as $r) {
            $icon = match ($r['status']) {
                'PASS' => '✅', 'FAIL' => '❌', 'SKIP' => '⚠️', default => '?',
            };
            $detail = str_replace('|', '\\|', $r['detail']);
            $summary .= sprintf(
                "| %s | %s %s | %s | %.0f | %+.0f |\n",
                $r['suite'], $icon, $r['name'], $r['status'], $r['ms'], $r['phpDeltaKB']
            );
        }

        $fullReport = $this->reportBuf . $summary;

        if (file_put_contents($this->reportPath, $fullReport) === false) {
            fwrite(STDERR, "Warning: could not write report to {$this->reportPath}\n");
        } else {
            $this->println($this->col(self::COL_DIM, "Report written to {$this->reportPath}"));
        }
    }

    /** Current wall-clock time in milliseconds */
    private function nowMs(): float
    {
        return microtime(true) * 1000.0;
    }

    /**
     * Read the process Resident Set Size from /proc/self/status.
     *
     * VmRSS measures actual physical RAM used by the process, including:
     *   - PHP heap
     *   - FFI C-heap allocations (malloc'd via \FFI::new())
     *   - Shared library code pages
     *   - Stack
     *
     * By diffing VmRSS before and after a suite (with GC in between), we can
     * detect C-heap leaks that PHP's memory_get_usage() would never see.
     *
     * Returns 0 on non-Linux platforms where /proc/self/status is unavailable.
     */
    private function readRssBytes(): int
    {
        $path = '/proc/self/status';
        if (!file_exists($path)) {
            return 0;
        }

        $content = @file_get_contents($path);
        if ($content === false) {
            return 0;
        }

        // VmRSS:   12345 kB
        if (preg_match('/VmRSS:\s+(\d+)\s+kB/', $content, $m)) {
            return (int)$m[1] * 1024;  // kB → bytes
        }

        return 0;
    }

    /** Apply an ANSI colour code if running in a terminal, otherwise passthrough */
    private function col(string $code, string $text): string
    {
        return $this->colorsEnabled ? "{$code}{$text}" . self::COL_RESET : $text;
    }

    private function bold(string $text): string
    {
        return $this->colorsEnabled ? self::COL_BOLD . $text . self::COL_RESET : $text;
    }

    private function println(string $line): void
    {
        echo $line . "\n";
    }
}

// ── Sentinel exceptions (declared here to keep everything in one file) ────

/**
 * Thrown by TestRunner::skip() to mark a test as intentionally skipped.
 * Caught by test() — NOT by a failure handler.
 */
class SkipException extends \Exception {}
