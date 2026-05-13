<?php
declare(strict_types=1);

/**
 * PML Examples — shared bootstrap.
 * Include this at the top of every example script.
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Tensor;

// Use all available CPU cores for tensor ops.
$cores = max(1, (int) trim((string) shell_exec('nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 4')));
Tensor::configureThreading($cores, 1);

// ── Tiny helpers used across examples ────────────────────────────────────────

/**
 * Print a section banner.
 */
function section(string $title): void
{
    echo "\n" . str_repeat('─', 60) . "\n";
    echo "  {$title}\n";
    echo str_repeat('─', 60) . "\n";
}

/**
 * Print a key/value metric line.
 */
function metric(string $name, mixed $value, string $unit = ''): void
{
    if (is_float($value)) {
        printf("  %-30s %s%s\n", $name . ':', number_format($value, 4), $unit);
    } else {
        printf("  %-30s %s%s\n", $name . ':', $value, $unit);
    }
}

/**
 * Elapsed time since $start in ms.
 */
function elapsed(float $start): string
{
    $ms = (microtime(true) - $start) * 1000;
    return $ms < 1000 ? number_format($ms, 1) . ' ms' : number_format($ms / 1000, 2) . ' s';
}
