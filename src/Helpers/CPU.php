<?php
declare(strict_types=1);

namespace Pml\Helpers;

/**
 * CPU topology helpers for parallel backend configuration.
 */
final class CPU
{
    /**
     * Number of logical CPU cores available to this process.
     * Reads /proc/cpuinfo on Linux; falls back to 1 if unavailable.
     */
    public static function cores(): int
    {
        static $cores = null;

        if ($cores === null) {
            if (is_file('/proc/cpuinfo')) {
                $count = substr_count((string) file_get_contents('/proc/cpuinfo'), 'processor');
                $cores = max(1, $count);
            } elseif (function_exists('proc_open')) {
                $uname = php_uname('s');
                if (str_contains($uname, 'Darwin')) {
                    $cores = (int) shell_exec('sysctl -n hw.ncpu');
                } else {
                    $cores = (int) shell_exec('nproc --all 2>/dev/null');
                }
                $cores = max(1, $cores);
            } else {
                $cores = 1;
            }
        }

        return $cores;
    }

    /**
     * Optimal worker count for a batch of $n independent tasks.
     * Caps at physical cores to avoid context-switch overhead.
     */
    public static function optimalWorkers(int $n): int
    {
        return min($n, self::cores());
    }
}
