<?php

/**
 * run_e2e.php — Pml Integration Test Entry Point
 *
 * Usage:
 *   php tests/run_e2e.php                        # run all suites
 *   php tests/run_e2e.php --suite classic        # Classic ML (Pipeline, CV, Ridge, Lasso)
 *   php tests/run_e2e.php --suite dl             # Deep Learning autograd engine
 *   php tests/run_e2e.php --suite joblib         # Serialisation integrity
 *   php tests/run_e2e.php --suite preprocessing  # Imputer, VarianceThreshold, OHE, GridSearch
 *   php tests/run_e2e.php --suite clustering     # KMeans, KNN
 *   php tests/run_e2e.php --suite advanced       # ElasticNet, AdaBoost, Bagging
 *   php tests/run_e2e.php --suite native        # SVC, SVR, XGBClassifier, DBSCAN
 *   php tests/run_e2e.php --report /path/to/report.md
 *
 * Exit codes:
 *   0 — all tests passed (or skipped)
 *   1 — one or more tests failed
 *
 * Requirements:
 *   - PHP 8.1+ with FFI extension enabled (extension=ffi in php.ini)
 *   - libopenblas.so on the library search path
 *   - Composer autoloader at vendor/autoload.php
 *   - Memory limit >= 256M (FFI BLAS matrices can be large)
 */

declare(strict_types=1);

// ── Runtime guards ─────────────────────────────────────────────────────────

if (PHP_MAJOR_VERSION < 8 || (PHP_MAJOR_VERSION === 8 && PHP_MINOR_VERSION < 1)) {
    fwrite(STDERR, "Pml requires PHP 8.1 or later. Running: " . PHP_VERSION . "\n");
    exit(1);
}

if (!extension_loaded('ffi')) {
    fwrite(STDERR, "Pml requires the FFI extension.\n"
        . "Enable it in php.ini: extension=ffi\n"
        . "Or run: php -d extension=ffi tests/run_e2e.php\n");
    exit(1);
}

// Raise memory limit — FFI allocates float[N²] distance matrices outside PHP's
// heap, but Tensor objects and PHP arrays still count toward memory_limit.
ini_set('memory_limit', '512M');

// ── Autoloading ────────────────────────────────────────────────────────────

$autoloadPaths = [
    __DIR__ . '/../vendor/autoload.php',   // when run from tests/
    __DIR__ . '/vendor/autoload.php',      // when run from project root
];

$autoloaderFound = false;
foreach ($autoloadPaths as $path) {
    if (file_exists($path)) {
        require_once $path;
        $autoloaderFound = true;
        break;
    }
}

if (!$autoloaderFound) {
    fwrite(STDERR, "Composer autoloader not found.\n"
        . "Run: composer install\n"
        . "Searched: " . implode(', ', $autoloadPaths) . "\n");
    exit(1);
}

// ── Require test files (not in Composer's PSR-4 autoload map) ─────────────

require_once __DIR__ . '/Core/TestRunner.php';
require_once __DIR__ . '/Datasets/DatasetLoader.php';
require_once __DIR__ . '/Suites/ClassicSuite.php';
require_once __DIR__ . '/Suites/DeepLearningSuite.php';
require_once __DIR__ . '/Suites/JoblibSuite.php';
require_once __DIR__ . '/Suites/PreprocessingSuite.php';
require_once __DIR__ . '/Suites/ClusteringSuite.php';
require_once __DIR__ . '/Suites/AdvancedEnsembleSuite.php';
require_once __DIR__ . '/Suites/NativeBindingsSuite.php';

use Pml\Tests\Core\TestRunner;
use Pml\Tests\Suites\{
    ClassicSuite,
    DeepLearningSuite,
    JoblibSuite,
    PreprocessingSuite,
    ClusteringSuite,
    AdvancedEnsembleSuite,
    NativeBindingsSuite,
};

// ── Argument parsing ───────────────────────────────────────────────────────

$opts = getopt('', ['suite:', 'report:']);

$suiteFilter  = strtolower($opts['suite'] ?? 'all');
$reportPath   = $opts['report'] ?? __DIR__ . '/../test_report.md';

// ── Run ────────────────────────────────────────────────────────────────────

$runner = new TestRunner(reportPath: $reportPath);

if (in_array($suiteFilter, ['all', 'classic'], true)) {
    ClassicSuite::run($runner);
}

if (in_array($suiteFilter, ['all', 'dl', 'deep', 'deeplearning'], true)) {
    DeepLearningSuite::run($runner);
}

if (in_array($suiteFilter, ['all', 'joblib', 'serial', 'serialization'], true)) {
    JoblibSuite::run($runner);
}

if (in_array($suiteFilter, ['all', 'preprocessing', 'preprocess'], true)) {
    PreprocessingSuite::run($runner);
}

if (in_array($suiteFilter, ['all', 'clustering', 'cluster'], true)) {
    ClusteringSuite::run($runner);
}

if (in_array($suiteFilter, ['all', 'advanced', 'ensemble'], true)) {
    AdvancedEnsembleSuite::run($runner);
}

if (in_array($suiteFilter, ['all', 'native', 'svm', 'xgboost'], true)) {
    NativeBindingsSuite::run($runner);
}

$runner->finish();

// ── Exit code ──────────────────────────────────────────────────────────────
// Read the total failure count from the runner.
// We use Reflection to access the private property rather than adding a public
// getter — the runner's public API is deliberately minimal.
$rc = new ReflectionClass($runner);
$p  = $rc->getProperty('totalFailed');
$p->setAccessible(true);
exit($p->getValue($runner) > 0 ? 1 : 0);
