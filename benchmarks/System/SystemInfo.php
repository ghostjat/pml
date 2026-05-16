<?php
declare(strict_types=1);

namespace Pml\Benchmarks\System;

/**
 * Collects reproducible machine specification for benchmark result metadata.
 *
 * Usage:
 *   $spec = SystemInfo::collect();
 *   $spec->print();
 *   $spec->toArray();  // embed in result JSON
 *
 * All information comes from /proc, /sys, and standard CLI tools.
 * Falls back gracefully on non-Linux systems.
 */
final class SystemInfo
{
    private function __construct(
        public readonly string $hostname,
        public readonly string $os,
        public readonly string $kernelVersion,
        public readonly string $cpuModel,
        public readonly int    $cpuCores,
        public readonly int    $cpuThreads,
        public readonly string $cpuMhzBase,
        public readonly string $cpuMhzBoost,
        public readonly array  $cpuFlags,        // avx2, fma, avx512f, etc.
        public readonly string $ramTotal,
        public readonly int    $numaCpunodes,
        public readonly string $phpVersion,
        public readonly string $gccVersion,
        public readonly string $openBlasVersion,
        public readonly string $openMpVersion,
        public readonly int    $ompNumThreads,
        public readonly bool   $opcacheEnabled,
        public readonly bool   $jitEnabled,
        public readonly string $cpuGovernor,
        public readonly string $libtensorPath,
        public readonly string $libtensorSymbols,
        public readonly \DateTimeImmutable $collectedAt,
    ) {}

    public static function collect(): self
    {
        return new self(
            hostname:           gethostname() ?: 'unknown',
            os:                 self::readLine('/etc/os-release', 'PRETTY_NAME') ?? php_uname('s'),
            kernelVersion:      php_uname('r'),
            cpuModel:           self::cpuModel(),
            cpuCores:           self::cpuCount('core id'),
            cpuThreads:         self::cpuCount('processor'),
            cpuMhzBase:         self::readSysfs('/sys/devices/system/cpu/cpu0/cpufreq/base_frequency')
                                    ? number_format((int)self::readSysfs('/sys/devices/system/cpu/cpu0/cpufreq/base_frequency') / 1000) . ' MHz'
                                    : (self::cpuInfoField('cpu MHz') ?? 'unknown'),
            cpuMhzBoost:        self::readSysfs('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq')
                                    ? number_format((int)self::readSysfs('/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq') / 1000) . ' MHz'
                                    : 'unknown',
            cpuFlags:           self::cpuFlags(),
            ramTotal:           self::memTotal(),
            numaCpunodes:       self::numaNodes(),
            phpVersion:         PHP_VERSION,
            gccVersion:         self::gccVersion(),
            openBlasVersion:    self::openBlasVersion(),
            openMpVersion:      self::openMpVersion(),
            ompNumThreads:      (int)(getenv('OMP_NUM_THREADS') ?: self::cpuCount('processor')),
            opcacheEnabled:     (bool)ini_get('opcache.enable_cli'),
            jitEnabled:         (bool)ini_get('opcache.jit'),
            cpuGovernor:        self::readSysfs('/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor') ?? 'unknown',
            libtensorPath:      self::libtensorPath(),
            libtensorSymbols:   self::libtensorSymbolCount(),
            collectedAt:        new \DateTimeImmutable(),
        );
    }

    public function toArray(): array
    {
        return [
            'collected_at'       => $this->collectedAt->format(\DateTimeInterface::ATOM),
            'hostname'           => $this->hostname,
            'os'                 => $this->os,
            'kernel'             => $this->kernelVersion,
            'cpu_model'          => $this->cpuModel,
            'cpu_cores'          => $this->cpuCores,
            'cpu_threads'        => $this->cpuThreads,
            'cpu_mhz_base'       => $this->cpuMhzBase,
            'cpu_mhz_boost'      => $this->cpuMhzBoost,
            'cpu_flags'          => $this->cpuFlags,
            'ram_total'          => $this->ramTotal,
            'numa_nodes'         => $this->numaCpunodes,
            'php_version'        => $this->phpVersion,
            'gcc_version'        => $this->gccVersion,
            'openblas_version'   => $this->openBlasVersion,
            'openmp_version'     => $this->openMpVersion,
            'omp_num_threads'    => $this->ompNumThreads,
            'opcache_enabled'    => $this->opcacheEnabled,
            'jit_enabled'        => $this->jitEnabled,
            'cpu_governor'       => $this->cpuGovernor,
            'libtensor_path'     => $this->libtensorPath,
            'libtensor_symbols'  => $this->libtensorSymbols,
        ];
    }

    public function toJson(): string
    {
        return json_encode($this->toArray(), JSON_PRETTY_PRINT | JSON_UNESCAPED_SLASHES);
    }

    public function print(): void
    {
        $pad = fn(string $k, string $v) => printf("  %-22s %s\n", $k . ':', $v);

        echo "\n";
        echo "┌─────────────────────────────────────────────┐\n";
        echo "│  PML Benchmark System Specification          │\n";
        echo "└─────────────────────────────────────────────┘\n";
        $pad('Collected at', $this->collectedAt->format('Y-m-d H:i:s T'));
        $pad('Hostname', $this->hostname);
        $pad('OS', $this->os);
        $pad('Kernel', $this->kernelVersion);
        echo "\n";
        $pad('CPU model', $this->cpuModel);
        $pad('Cores / threads', "{$this->cpuCores} cores / {$this->cpuThreads} threads");
        $pad('Base / boost MHz', "{$this->cpuMhzBase} / {$this->cpuMhzBoost}");
        $pad('CPU governor', $this->cpuGovernor);
        $pad('SIMD flags', implode(' ', array_intersect(['avx2', 'avx512f', 'fma', 'sse4_2'], $this->cpuFlags)));
        $pad('RAM total', $this->ramTotal);
        $pad('NUMA nodes', (string)$this->numaCpunodes);
        echo "\n";
        $pad('PHP version', $this->phpVersion);
        $pad('GCC version', $this->gccVersion);
        $pad('OpenBLAS', $this->openBlasVersion);
        $pad('OMP_NUM_THREADS', (string)$this->ompNumThreads);
        $pad('OPcache', $this->opcacheEnabled ? 'enabled' : 'disabled');
        $pad('JIT', $this->jitEnabled ? 'enabled' : 'disabled');
        echo "\n";
        $pad('libtensor.so', $this->libtensorPath);
        $pad('Exported symbols', $this->libtensorSymbols);
        echo "\n";

        if ($this->cpuGovernor !== 'performance') {
            echo "  ⚠ WARNING: CPU governor is '{$this->cpuGovernor}', not 'performance'.\n";
            echo "    Timing noise will be higher. Run:\n";
            echo "    sudo cpupower frequency-set -g performance\n\n";
        }
    }

    // ─── Helpers ────────────────────────────────────────────────────────────

    /** Read VmRSS from /proc/self/status — total RSS including C heap */
    public static function vmRssKb(): int
    {
        if (!is_readable('/proc/self/status')) return 0;
        foreach (file('/proc/self/status') as $line) {
            if (str_starts_with($line, 'VmRSS:')) {
                return (int)preg_replace('/\D/', '', $line);
            }
        }
        return 0;
    }

    /** Measure VmRSS delta around a closure — measures actual C heap impact */
    public static function measureRssDelta(callable $fn): array
    {
        gc_collect_cycles();
        $before = self::vmRssKb();
        $fn();
        gc_collect_cycles();
        $after = self::vmRssKb();
        return ['before_kb' => $before, 'after_kb' => $after, 'delta_kb' => $after - $before];
    }

    private static function cpuModel(): string
    {
        return self::cpuInfoField('model name') ?? 'unknown';
    }

    private static function cpuCount(string $field): int
    {
        if (!is_readable('/proc/cpuinfo')) return 1;
        $ids = [];
        foreach (file('/proc/cpuinfo') as $line) {
            if (str_starts_with(trim($line), $field)) {
                $ids[] = trim(explode(':', $line, 2)[1] ?? '0');
            }
        }
        return count(array_unique($ids));
    }

    private static function cpuInfoField(string $field): ?string
    {
        if (!is_readable('/proc/cpuinfo')) return null;
        foreach (file('/proc/cpuinfo') as $line) {
            if (str_starts_with(trim($line), $field)) {
                return trim(explode(':', $line, 2)[1] ?? '');
            }
        }
        return null;
    }

    private static function cpuFlags(): array
    {
        $flags = self::cpuInfoField('flags') ?? '';
        return array_filter(explode(' ', $flags));
    }

    private static function memTotal(): string
    {
        if (!is_readable('/proc/meminfo')) return 'unknown';
        foreach (file('/proc/meminfo') as $line) {
            if (str_starts_with($line, 'MemTotal:')) {
                $kb = (int)preg_replace('/\D/', '', $line);
                return number_format(round($kb / 1024 / 1024, 1), 1) . ' GB';
            }
        }
        return 'unknown';
    }

    private static function numaNodes(): int
    {
        $nodes = glob('/sys/devices/system/node/node[0-9]*') ?: [];
        return max(1, count($nodes));
    }

    private static function gccVersion(): string
    {
        $out = shell_exec('gcc --version 2>/dev/null | head -1') ?? '';
        return trim($out) ?: 'not found';
    }

    private static function openBlasVersion(): string
    {
        // Try dpkg first
        $dpkg = shell_exec('dpkg -l libopenblas-dev 2>/dev/null | grep "^ii" | awk \'{print $3}\'') ?? '';
        $dpkg = trim($dpkg);
        if ($dpkg) return "libopenblas-dev {$dpkg}";

        // Try pkg-config
        $pc = shell_exec('pkg-config --modversion openblas 2>/dev/null') ?? '';
        $pc = trim($pc);
        if ($pc) return $pc;

        return 'unknown';
    }

    private static function openMpVersion(): string
    {
        // GCC ships libgomp; version matches GCC
        $out = shell_exec('gcc --version 2>/dev/null | head -1') ?? '';
        if (preg_match('/(\d+\.\d+\.\d+)/', $out, $m)) {
            return 'libgomp ' . $m[1];
        }
        return 'unknown';
    }

    private static function libtensorPath(): string
    {
        $candidates = [
            __DIR__ . '/../../src/Lib/libtensor.so',
            dirname(__DIR__, 2) . '/src/Lib/libtensor.so',
        ];
        foreach ($candidates as $p) {
            if (file_exists($p)) return realpath($p) ?: $p;
        }
        return 'not found';
    }

    private static function libtensorSymbolCount(): string
    {
        $path = self::libtensorPath();
        if ($path === 'not found') return 'N/A';
        $out = shell_exec("nm -D {$path} 2>/dev/null | grep -c '^[0-9a-f]'") ?? '';
        $count = (int)trim($out);
        return $count > 0 ? "{$count} exported" : 'unknown';
    }

    private static function readLine(string $file, string $key): ?string
    {
        if (!is_readable($file)) return null;
        foreach (file($file) as $line) {
            if (str_starts_with($line, $key . '=')) {
                return trim(str_replace([$key . '=', '"'], '', $line));
            }
        }
        return null;
    }

    private static function readSysfs(string $path): ?string
    {
        if (!is_readable($path)) return null;
        return trim(file_get_contents($path));
    }
}
