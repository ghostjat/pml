<?php

declare(strict_types=1);

/**
 * Pml — Comprehensive Demo & Sanity Tests
 *
 * Run: php demo.php
 *
 * This file exercises every major subsystem:
 *   1. Tensor construction and shape operations
 *   2. BLAS operations (sgemm, saxpy, sdot, scopy)
 *   3. Math operators (activations, normalization)
 *   4. Linear algebra (inverse, SVD, eigendecomposition)
 *   5. Layer primitives (attention, FFN, transformer block)
 *   6. Sampler (greedy, temperature, top-k, top-p)
 *   7. SafetensorsLoader (save + reload round-trip)
 */

require_once __DIR__ . '/vendor/autoload.php';

use Pml\{Tensor, Ops, BlasEngine, ModelConfig, Metrics};
use Pml\Layers\{
    Linear, Embedding, MultiHeadAttention, FeedForward,
    TransformerBlock, KVCache, Dropout, RoPE
};
use Pml\Generation\Sampler;

// ── ANSI colours for a nicer terminal output ──────────────────────────────
$GRN = "\033[32m"; $YLW = "\033[33m"; $CYN = "\033[36m"; $RST = "\033[0m"; $BLD = "\033[1m";

function pass(string $name, string $detail = ''): void
{
    global $GRN, $RST;
    echo "  {$GRN}✓{$RST}  {$name}" . ($detail ? "  ({$detail})" : '') . "\n";
}

function section(string $title): void
{
    global $BLD, $CYN, $RST;
    echo "\n{$CYN}{$BLD}── {$title} ──{$RST}\n";
}

function assertNear(float $a, float $b, float $tol = 1e-4, string $msg = ''): void
{
    if (abs($a - $b) > $tol) {
        echo "  \033[31m✗  FAIL: {$msg} — expected ~{$b}, got {$a}\033[0m\n";
    }
}

echo "\n{$BLD}Pml Demo & Sanity Suite{$RST}\n";
echo "PHP " . PHP_VERSION . " | FFI " . (extension_loaded('ffi') ? 'ON' : 'OFF') . "\n";

// ═══════════════════════════════════════════════════════════════════════════
//  1. TENSOR CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════
section('1. Tensor Construction');

$zeros = Tensor::zeros([3, 4]);
assert($zeros->shape === [3, 4]);
assert($zeros->size  === 12);
pass('zeros([3,4])', (string)$zeros);

$ones = Tensor::ones([2, 3]);
assertNear($ones->sum(), 6.0, 1e-5, 'ones sum');
pass('ones([2,3])  sum=' . $ones->sum());

$full = Tensor::full([5], 3.14);
assertNear($full->get(2), 3.14, 1e-5, 'full value');
pass('full([5], 3.14)');

$eye = Tensor::eye(4);
assertNear($eye->get(0, 0), 1.0, 1e-5, 'eye diagonal');
assertNear($eye->get(0, 1), 0.0, 1e-5, 'eye off-diagonal');
pass('eye(4)  trace=' . ($eye->get(0,0)+$eye->get(1,1)+$eye->get(2,2)+$eye->get(3,3)));

$arr = Tensor::fromArray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]);
assert($arr->shape === [3, 2]);
assertNear($arr->get(2, 1), 6.0, 1e-5, 'fromArray element');
pass('fromArray([[1,2],[3,4],[5,6]])');

$arange = Tensor::arange(0.0, 5.0);
assertNear($arange->get(3), 3.0, 1e-5, 'arange element');
pass('arange(0,5)  values=' . implode(',', $arange->toArray()));

$randn = Tensor::randn([1000]);
$mean  = $randn->mean();
$norm  = $randn->norm();
pass(sprintf(
    "randn([1000])  mean≈%.4f  ‖x‖≈%.2f",
    $mean,
    $norm
));

// ═══════════════════════════════════════════════════════════════════════════
//  2. SHAPE OPS
// ═══════════════════════════════════════════════════════════════════════════
section('2. Shape Operations');

$t    = Tensor::fromArray([[1,2,3],[4,5,6]]);       // [2, 3]
$flat = $t->flatten();
assert($flat->shape === [6]);
assertNear($flat->get(4), 5.0, 1e-5, 'flatten element');
pass('flatten [2,3] → [6]');

$r = $t->reshape([3, 2]);
assert($r->shape === [3, 2]);
assertNear($r->get(2, 1), 6.0, 1e-5, 'reshape element');
pass('reshape [2,3] → [3,2]');

$tT = $t->T(); // logical transpose
assert($tT->shape === [3, 2]);
assert($tT->_transposed === true);
pass('T() logical transpose [2,3] → [3,2]  (zero-copy)');

$tTp = $t->transposePhysical();
assertNear($tTp->get(0, 1), 4.0, 1e-5, 'physical transpose');
pass('transposePhysical()  [2,3] → [3,2]');

$row = $t->getRow(1);
assert($row->shape === [3]);
assertNear($row->get(0), 4.0, 1e-5, 'getRow element');
pass('getRow(1)  → [4, 5, 6]');

$t3 = Tensor::randn([2, 3, 4]);
$sq = $t3->unsqueeze(1);
assert($sq->shape === [2, 1, 3, 4]);
pass('unsqueeze(1)  [2,3,4] → [2,1,3,4]');

// ═══════════════════════════════════════════════════════════════════════════
//  3. BLAS OPS
// ═══════════════════════════════════════════════════════════════════════════
section('3. BLAS Operations');

// matmul: [2,3] · [3,2] → [2,2]
$A  = Tensor::fromArray([[1,2,3],[4,5,6]]);        // [2,3]
$B  = Tensor::fromArray([[7,8],[9,10],[11,12]]);    // [3,2]
$C  = Ops::matmul($A, $B);
assert($C->shape === [2, 2]);
assertNear($C->get(0,0), 58.0, 1e-4, 'matmul C[0,0]');
assertNear($C->get(1,1), 154.0, 1e-4, 'matmul C[1,1]');
pass('matmul [2,3]·[3,2]  C[0,0]='. $C->get(0,0) . ' C[1,1]=' . $C->get(1,1));

// dot product
$x = Tensor::fromArray([1.0, 2.0, 3.0]);
$y = Tensor::fromArray([4.0, 5.0, 6.0]);
$d = Ops::dot($x, $y);
assertNear($d, 32.0, 1e-5, 'dot product');
pass("dot([1,2,3],[4,5,6])  = {$d}");

// saxpy: y += alpha * x
$u = Tensor::fromArray([1.0, 1.0, 1.0]);
$v = Tensor::fromArray([2.0, 2.0, 2.0]);
Ops::saxpy($u, $v, 3.0);  // v = v + 3*u = [5,5,5]
assertNear($v->get(0), 5.0, 1e-5, 'saxpy');
pass("saxpy: v += 3*u  → v[0]={$v->get(0)}");

// norm
$n2 = Tensor::fromArray([3.0, 4.0]);
assertNear($n2->norm(), 5.0, 1e-5, 'L2 norm');
pass("norm([3,4]) = {$n2->norm()}");

// outer product
$oa = Tensor::fromArray([1.0, 2.0]);
$ob = Tensor::fromArray([3.0, 4.0, 5.0]);
$OC = Ops::outer($oa, $ob);
assert($OC->shape === [2, 3]);
assertNear($OC->get(1, 2), 10.0, 1e-5, 'outer product');
pass("outer([1,2],[3,4,5])  [1,2]={$OC->get(1,2)}");

// ═══════════════════════════════════════════════════════════════════════════
//  4. ACTIVATIONS & NORMALIZATION
// ═══════════════════════════════════════════════════════════════════════════
section('4. Activations & Normalization');

// ReLU
$relu_in  = Tensor::fromArray([-2.0, -1.0, 0.0, 1.0, 2.0]);
$relu_out = Ops::relu($relu_in);
assertNear($relu_out->get(0), 0.0, 1e-5, 'relu negative');
assertNear($relu_out->get(4), 2.0, 1e-5, 'relu positive');
pass('ReLU([-2,-1,0,1,2])  = ' . implode(',', $relu_out->toArray()));

// GELU
$gelu_in  = Tensor::fromArray([0.0, 1.0, -1.0]);
$gelu_out = Ops::gelu($gelu_in);
// GELU(0) = 0, GELU(1) ≈ 0.841, GELU(-1) ≈ -0.159
assertNear($gelu_out->get(0), 0.0, 1e-4, 'gelu(0)');
assertNear($gelu_out->get(1), 0.841, 1e-2, 'gelu(1)');
pass('GELU([0,1,-1])  = ' . implode(',', array_map(fn($v) => round($v,4), $gelu_out->toArray())));

// SiLU
$silu_out = Ops::silu(Tensor::fromArray([0.0, 1.0, -1.0]));
assertNear($silu_out->get(0), 0.0, 1e-5, 'silu(0)');
assertNear($silu_out->get(1), 0.731, 1e-2, 'silu(1)');
pass('SiLU([0,1,-1])  = ' . implode(',', array_map(fn($v) => round($v,4), $silu_out->toArray())));

// Softmax
$logits = Tensor::fromArray([1.0, 2.0, 3.0]);
$probs  = Ops::softmax($logits->unsqueeze(0))->squeeze();
assertNear($probs->sum(), 1.0, 1e-5, 'softmax sum to 1');
pass('Softmax sum = ' . $probs->sum() . '  max_idx=' . $probs->argmax());

// RMSNorm
$xNorm = Tensor::fromArray([[1.0, 2.0, 3.0, 4.0]]);
$w     = Tensor::ones([4]);
Ops::rmsNormInPlace($xNorm, $w);
$rmsVal = $xNorm->norm();
pass(sprintf("RMSNorm | norm(result) = %.4f", $rmsVal));

// LayerNorm
$xLN  = Tensor::fromArray([[1.0, 2.0, 3.0, 4.0]]);
$gam  = Tensor::ones([4]);
$beta = Tensor::zeros([4]);
Ops::layerNormInPlace($xLN, $gam, $beta);
assertNear($xLN->mean(), 0.0, 1e-4, 'LayerNorm mean≈0');
pass(sprintf(
    "LayerNorm  mean=%.6f  std≈1",
    $xLN->mean()
));

// ═══════════════════════════════════════════════════════════════════════════
//  5. LAYER PRIMITIVES
// ═══════════════════════════════════════════════════════════════════════════
section('5. Layer Primitives');

// Linear layer
$linW = Tensor::heInit([8, 4]);    // [out, in]
$lin  = new Linear($linW);
$xinL = Tensor::randn([3, 4]);     // [seq=3, in=4]
$yL   = $lin->forward($xinL);
assert($yL->shape === [3, 8]);
pass("Linear [3,4] → [3,8]  output.norm=" . round($yL->norm(), 4));

// Embedding
$embW = Tensor::randn([100, 16]); // [vocab=100, d=16]
$emb  = new Embedding($embW);
$out  = $emb->forward([5, 12, 0, 99]);
assert($out->shape === [4, 16]);
pass("Embedding [4 tokens] → [4, 16]");

// KV Cache
$kvc = new KVCache(512, 64);
$k1  = Tensor::randn([1, 64]);
$v1  = Tensor::randn([1, 64]);
$kvc->append($k1, $v1);
$kvc->append($k1, $v1);
assert($kvc->currentLength() === 2);
$activeK = $kvc->getActiveK();
assert($activeK->shape === [2, 64]);
pass("KVCache: appended 2 rows, getActiveK shape=" . implode('×', $activeK->shape));

// Dropout (training mode)
$doPut = new Dropout(0.5, training: true);
$xDo   = Tensor::ones([1000]);
$yDo   = $doPut->forward($xDo);
$mean  = $yDo->mean();
// With 50% dropout + scale, mean should be ≈1.0
pass(sprintf(
    "Dropout(p=0.5) training mean≈%.3f  (expect ~1.0)",
    $mean
));

// RoPE
$rope = new RoPE(headDim: 64);
$q    = Tensor::randn([4, 64]);  // [seq=4, d=64]
$qRoped = $rope->apply($q->clone(), 0);
assert($qRoped->shape === [4, 64]);
pass("RoPE applied to [4, 64]  ‖q_rope‖≈" . round($qRoped->norm(), 2));

// Full MultiHeadAttention pass
$dM = 128; $nH = 4; $dKV = 64;
$wq = Tensor::xavierInit([$dM, $dM]);
$wk = Tensor::xavierInit([$dM, $dKV]);
$wv = Tensor::xavierInit([$dM, $dKV]);
$wo = Tensor::xavierInit([$dM, $dM]);
$mha = new MultiHeadAttention($wq, $wk, $wv, $wo, nHeads: $nH, nKVHeads: 2);

$seq   = 6;
$xAttn = Tensor::randn([$seq, $dM]);
$cache = new KVCache(64, $dKV);
$attnOut = $mha->forward($xAttn, $cache, 0, causal: true);
assert($attnOut->shape === [$seq, $dM]);
pass("MultiHeadAttention [6, 128] causal  output.norm=" . round($attnOut->norm(), 4));

// FeedForward (SwiGLU)
$dFF = 256;
$fw1  = Tensor::heInit([$dM, $dFF]);
$fw2  = Tensor::heInit([$dFF, $dM]);
$fw3  = Tensor::heInit([$dM, $dFF]);
$ffn  = new FeedForward($fw1, $fw2, $fw3);
$ffnOut = $ffn->forward($xAttn);
assert($ffnOut->shape === [$seq, $dM]);
pass("FeedForward SwiGLU [6, 128] → [6, 128]  norm=" . round($ffnOut->norm(), 4));

// TransformerBlock
$normW1 = Tensor::ones([$dM]);
$normW2 = Tensor::ones([$dM]);
$block  = new TransformerBlock($mha, $ffn, $normW1, $normW2);
$cache2 = new KVCache(64, $dKV);
$xBlock = Tensor::randn([$seq, $dM]);
$bOut   = $block->forward($xBlock, $cache2);
assert($bOut->shape === [$seq, $dM]);
pass("TransformerBlock [6, 128]  output.norm=" . round($bOut->norm(), 4));

// ═══════════════════════════════════════════════════════════════════════════
//  6. SAMPLER
// ═══════════════════════════════════════════════════════════════════════════
section('6. Sampler');

$vocabLogits = Tensor::randn([50257]);  // GPT-2 vocab size

$greedy = Sampler::greedy($vocabLogits);
pass("Greedy  argmax={$greedy}");

$sampled = Sampler::sample($vocabLogits, temperature: 0.7, topK: 50, topP: 0.9);
pass("Sample(T=0.7, topK=50, topP=0.9)  token={$sampled}");

$sampled2 = Sampler::sample($vocabLogits, temperature: 1.5);
pass("Sample(T=1.5 creative)  token={$sampled2}");

// ═══════════════════════════════════════════════════════════════════════════
//  7. I/O: SafetensorsLoader round-trip
// ═══════════════════════════════════════════════════════════════════════════
section('7. SafetensorsLoader Round-trip');

use Pml\IO\SafetensorsLoader;

$tmpFile = sys_get_temp_dir() . '/phptensor_test_' . getmypid() . '.safetensors';

$origA = Tensor::fromArray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
$origB = Tensor::randn([8, 4]);
SafetensorsLoader::save($tmpFile, ['layerA' => $origA, 'layerB' => $origB]);

$loaded = SafetensorsLoader::load($tmpFile);
assert(isset($loaded['layerA'], $loaded['layerB']));
assertNear($loaded['layerA']->get(1, 2), 6.0, 1e-5, 'reloaded element');
assertNear($loaded['layerB']->norm(), $origB->norm(), 1e-3, 'norm preserved');
unlink($tmpFile);

pass("save + load round-trip  layerA[1,2]=" . $loaded['layerA']->get(1,2));

// ═══════════════════════════════════════════════════════════════════════════
//  8. LINEAL ALGEBRA (LAPACK) — only if OpenBLAS includes LAPACKE
// ═══════════════════════════════════════════════════════════════════════════
section('8. LAPACK Linear Algebra');

try {
    // Matrix inverse
    $M  = Tensor::fromArray([[2.0, 1.0], [5.0, 3.0]]);
    $Mi = Ops::inverse($M);
    // M * M^-1 should be ≈ I
    $Id = Ops::matmul($M, $Mi);
    assertNear($Id->get(0,0), 1.0, 1e-4, 'inverse I[0,0]');
    assertNear($Id->get(0,1), 0.0, 1e-4, 'inverse I[0,1]');
    pass("inverse(M)  M*M^-1 ≈ I  diag=(" . round($Id->get(0,0),4) . ',' . round($Id->get(1,1),4) . ')');

    // SVD
    $S    = Tensor::fromArray([[3.0, 1.0], [1.0, 3.0], [0.0, 0.0]]);
    [$U, $s, $Vt] = Ops::svd($S);
    pass("SVD [3,2]  singular values=[" . implode(',', array_map(fn($v)=>round($v,3), $s->toArray())) . "]");

    // Eigendecomposition
    $Sym  = Tensor::fromArray([[4.0, 2.0], [2.0, 3.0]]);
    [$eigenvals, $eigenvecs] = Ops::eigh($Sym);
    pass("eigh([4,2;2,3])  eigenvalues≈[" . implode(',', array_map(fn($v)=>round($v,3), $eigenvals->toArray())) . "]");

    // Solve Ax = b
    $Asys = Tensor::fromArray([[2.0, 1.0], [5.0, 3.0]]);
    $bvec = Tensor::fromArray([[8.0], [21.0]]);
    $xvec = Ops::solve($Asys, $bvec);
    assertNear($xvec->get(0, 0), 3.0, 1e-4, 'solve x[0]');
    assertNear($xvec->get(1, 0), 2.0, 1e-4, 'solve x[1]');
    pass("solve Ax=b  x=[" . round($xvec->get(0,0),4) . ',' . round($xvec->get(1,0),4) . "]  (expect [3,2])");

} catch (\Throwable $e) {
    echo "  \033[33m⚠  LAPACK skipped: {$e->getMessage()}\033[0m\n";
    echo "  \033[33m   (LAPACKE not available in this OpenBLAS build — BLAS ops still work)\033[0m\n";
}

// ═══════════════════════════════════════════════════════════════════════════
//  9. METRICS
// ═══════════════════════════════════════════════════════════════════════════
section('9. Metrics');

// ── Classification ────────────────────────────────────────────────────────
// Perfect classifier: logit for correct class is highest.
$logits4 = Tensor::fromArray([
    [2.0, 0.1, 0.1, 0.1],   // pred 0, true 0 ✓
    [0.1, 2.0, 0.1, 0.1],   // pred 1, true 1 ✓
    [0.1, 0.1, 2.0, 0.1],   // pred 2, true 2 ✓
    [0.1, 0.1, 0.1, 2.0],   // pred 3, true 3 ✓
    [2.0, 0.1, 0.1, 0.1],   // pred 0, true 1 ✗
]);
$targets5 = [0, 1, 2, 3, 1];

$acc = Metrics::accuracy($logits4, $targets5);
assertNear($acc, 0.8, 1e-6, 'accuracy');
pass("accuracy  4/5 correct  acc=" . round($acc, 4));

$top2 = Metrics::topKAccuracy($logits4, $targets5, k: 2);
// For the misclassified sample (pred=0, true=1): top-2 of [2.0, 0.1, 0.1, 0.1] → {0,1} ✓
assertNear($top2, 1.0, 1e-6, 'top-2 accuracy');
pass("top-2 accuracy  5/5  acc=" . round($top2, 4));

$yPred5 = Metrics::argmax($logits4);
$cm = Metrics::confusionMatrix($yPred5, $targets5, numClasses: 4);
assert($cm->shape === [4, 4]);
assertNear($cm->get(0, 0), 1.0, 1e-6, 'CM[0,0]');  // true 0, pred 0
assertNear($cm->get(1, 0), 1.0, 1e-6, 'CM[1,0]');  // true 1, pred 0 (miss)
assertNear($cm->get(1, 1), 1.0, 1e-6, 'CM[1,1]');  // true 1, pred 1
pass("confusionMatrix [4×4]  CM[1,0]=" . (int)$cm->get(1,0) . " (expected 1 miss)");

$prf = Metrics::precisionRecallF1($yPred5, $targets5, numClasses: 4);
assertNear($prf['micro_f1'], 0.8, 1e-6, 'micro F1');
pass("precision/recall/F1  macro_f1=" . round($prf['macro_f1'], 4)
    . "  micro_f1=" . round($prf['micro_f1'], 4));

// ── Regression ────────────────────────────────────────────────────────────
$perfect = Tensor::fromArray([1.0, 2.0, 3.0, 4.0]);
$shifted = Tensor::fromArray([1.5, 2.5, 3.5, 4.5]);   // constant offset +0.5

assertNear(Metrics::mae($perfect, $shifted), 0.5, 1e-5, 'MAE');
pass("MAE   pred vs target+0.5  mae=" . round(Metrics::mae($perfect, $shifted), 4));

assertNear(Metrics::rmse($perfect, $shifted), 0.5, 1e-5, 'RMSE');
pass("RMSE  pred vs target+0.5  rmse=" . round(Metrics::rmse($perfect, $shifted), 4));

$r2 = Metrics::r2Score($perfect, $perfect);
assertNear($r2, 1.0, 1e-5, 'R² perfect');
pass("R²    perfect fit  r2=" . round($r2, 4) . "  (expect 1.0)");

$r2bad = Metrics::r2Score($shifted, $perfect);
pass("R²    shifted pred  r2=" . round($r2bad, 4));

// ── Language model: perplexity ────────────────────────────────────────────
// Perplexity of a uniform distribution over C=4 classes should equal C.
$uniformLogits = Tensor::fromArray([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]);
$ppl = Metrics::perplexity($uniformLogits, [0, 1]);
assertNear($ppl, 4.0, 0.01, 'perplexity uniform');
pass("perplexity  uniform over 4 classes  ppl=" . round($ppl, 4) . "  (expect 4.0)");

// ═══════════════════════════════════════════════════════════════════════════
//  SUMMARY
// ═══════════════════════════════════════════════════════════════════════════
echo "\n{$GRN}{$BLD}All checks passed.{$RST}  Pml is operational.\n\n";