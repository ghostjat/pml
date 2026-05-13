<?php
declare(strict_types=1);
/**
 * TENSOR ENGINE SHOWCASE & BENCHMARK
 * ═══════════════════════════════════════════════════════════════════
 * Demonstrates the raw power of PML's C tensor engine:
 * AVX2 + OpenMP SIMD, OpenBLAS matrix multiplication,
 * fused kernels, and zero-copy mmap loading.
 *
 * Run on your machine to see what PHP + C can really do.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Tensor;
use Pml\Lib\SafeTensorsIO;

section('PML Tensor Engine Benchmark');

$cores = max(1, (int) trim((string) shell_exec('nproc 2>/dev/null || echo 4')));
metric('CPU cores', $cores);
metric('OMP threads', $cores);

// ── 1. Large matrix multiply (OpenBLAS SGEMM) ────────────────────────────────
section('Matrix Multiplication (OpenBLAS SGEMM)');

$sizes = [[512, 512], [1024, 1024], [2048, 512]];
foreach ($sizes as [$m, $n]) {
    $A = Tensor::randomNormal([$m, $n], 0.0, 0.01);
    $B = Tensor::randomNormal([$n, $m], 0.0, 0.01);

    $t0 = microtime(true);
    $C  = $A->matmul($B);
    $ms = (microtime(true) - $t0) * 1000;

    $gflops = 2.0 * $m * $n * $m / 1e9 / (($ms) / 1000);
    printf("  [%4d×%4d] @ [%4d×%4d]  → %.1f ms  %.1f GFLOP/s\n", $m, $n, $n, $m, $ms, $gflops);
}

// ── 2. Element-wise ops (AVX2 SIMD) ──────────────────────────────────────────
section('Element-Wise Operations (AVX2 SIMD)');

$n    = 4_000_000;
$data = Tensor::randomNormal([$n], 0.0, 1.0);

$ops = [
    'sigmoid' => fn($x) => $x->sigmoid(),
    'relu'    => fn($x) => $x->relu(),
    'gelu'    => fn($x) => $x->gelu(),
    'exp'     => fn($x) => $x->exp(),
    'tanh'    => fn($x) => $x->tanh(),
    'sqrt(|x|)'=> fn($x) => $x->abs()->sqrt(),
];

foreach ($ops as $name => $fn) {
    $t0  = microtime(true);
    $out = $fn($data);
    $ms  = (microtime(true) - $t0) * 1000;
    $throughput = $n / ($ms / 1000) / 1e6;
    printf("  %-12s  N=%s  %.2f ms  %.0f M elem/s\n",
           $name, number_format($n), $ms, $throughput);
}

// ── 3. Reduction operations ───────────────────────────────────────────────────
section('Reductions');

$mat = Tensor::randomNormal([10000, 1000], 0.0, 1.0);

$t0  = microtime(true); $s = $mat->sum();   $ms = (microtime(true) - $t0) * 1000;
printf("  sum  [10k×1k]  %.2f ms\n", $ms);

$t0  = microtime(true); $m = $mat->mean();  $ms = (microtime(true) - $t0) * 1000;
printf("  mean [10k×1k]  %.2f ms\n", $ms);

$t0  = microtime(true); $ax = $mat->sumAxis(1); $ms = (microtime(true) - $t0) * 1000;
printf("  sumAxis(1) [10k×1k]→[10k]  %.2f ms\n", $ms);

// ── 4. Adam fused kernel throughput ──────────────────────────────────────────
section('Fused Adam Step (GPU-class throughput)');

$nParams = 50_000_000;  // 50M parameter model
$param   = Tensor::randomNormal([$nParams], 0.0, 0.01);
$grad    = Tensor::randomNormal([$nParams], 0.0, 0.001);
$m       = Tensor::zeros($nParams);
$v       = Tensor::zeros($nParams);

$t0 = microtime(true);
Tensor::fusedAdamStep($param, $grad, $m, $v, 1e-3, 0.9, 0.999, 1e-8, 1);
$ms = (microtime(true) - $t0) * 1000;

printf("  50M parameters  %.2f ms  (%.0f M param/s)\n", $ms, $nParams / ($ms / 1000) / 1e6);

// ── 5. SafeTensors mmap round-trip ────────────────────────────────────────────
section('SafeTensors Save / mmap Load');

$tensors = [
    'weights' => Tensor::randomNormal([4096, 4096], 0.0, 0.01),
    'bias'    => Tensor::zeros(4096),
];

$path = sys_get_temp_dir() . '/bench_tensors.safetensors';

$t0 = microtime(true);
SafeTensorsIO::save($path, $tensors);
$saveMs = (microtime(true) - $t0) * 1000;

$t0 = microtime(true);
$loaded = SafeTensorsIO::load($path);
$loadMs = (microtime(true) - $t0) * 1000;

$sizeMb = filesize($path) / 1024 / 1024;
printf("  Saved  : %.1f MB in %.1f ms  (%.0f MB/s)\n", $sizeMb, $saveMs, $sizeMb / ($saveMs / 1000));
printf("  Loaded : %.1f MB in %.1f ms  (mmap zero-copy)\n", $sizeMb, $loadMs);

// ── 6. Softmax cross-entropy throughput ───────────────────────────────────────
section('Fused Cross-Entropy Loss (training hot-path)');

$batchSizes = [32, 128, 512];
$vocab      = 32000;

foreach ($batchSizes as $bs) {
    $logits  = Tensor::randomNormal([$bs, $vocab], 0.0, 1.0);
    $targets = Tensor::fromArray(array_map(fn() => (float)mt_rand(0, $vocab - 1),
                                           range(0, $bs - 1)), Tensor::DTYPE_INT32);

    $t0 = microtime(true);
    ['loss' => $loss] = $logits->fusedCrossEntropyLossAndGrad($targets);
    $ms = (microtime(true) - $t0) * 1000;

    printf("  batch=%3d  vocab=%d  %.2f ms  loss=%.4f\n", $bs, $vocab, $ms, $loss);
}

echo "\n✓ Benchmark complete\n";
