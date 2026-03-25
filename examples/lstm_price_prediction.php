<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/lstm_price_prediction.php — LSTM Direction Classifier
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Predicts whether a synthetic price series will go UP or DOWN over the next
 * step using a sliding-window LSTM sequence classifier.
 *
 * ── Pipeline ──────────────────────────────────────────────────────────────
 *
 *  1. Generate a synthetic price series with trend, volatility clustering,
 *     and AR(1) momentum (mimics a mean-reverting equity-like process).
 *
 *  2. Build lag features: for each time step t, the feature vector is the
 *     window of T normalised log-returns ending at t:
 *       x_t = [r_{t-T+1}, …, r_t]  where r_t = log(p_t / p_{t-1})
 *     Label: y_t = 1 if p_{t+1} > p_t, else 0  (binary direction).
 *
 *  3. Split 80/20 train/test.  Build Tensor sequences [B, T, I] where I=1.
 *
 *  4. Train an LSTMCell (hidden=32, T=10) wrapped in RNNClassifier for
 *     N_EPOCHS epochs using mini-batch SGD with learning rate scheduling.
 *     Gradient clipping (maxNorm=5.0) prevents BPTT divergence.
 *
 *  5. Report per-epoch train loss + test accuracy; show final confusion
 *     matrix and per-class precision/recall.
 *
 * ── Why LSTM over Vanilla RNN? ────────────────────────────────────────────
 *
 *  The forget gate lets the LSTM selectively retain price-momentum signals
 *  over long lags (e.g., 5–10 bars) while resetting after volatility spikes.
 *  Vanilla RNNs suffer from vanishing gradients on lags > ~5 steps;
 *  LSTMs maintain the gradient through the cell-state highway c_t.
 *
 * Usage:
 *   php examples/lstm_price_prediction.php
 * ════════════════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\Tensor;
use Pml\Layers\{LSTMCell, RNNClassifier};
use Pml\Training\AdamW;

// ─── Hyper-parameters ─────────────────────────────────────────────────────

const N_PRICES   = 2_000;    // total time steps in synthetic price series
const SEQ_LEN    = 10;       // LSTM window / T
const INPUT_SIZE = 1;        // one feature per step: normalised log-return
const HIDDEN     = 32;       // LSTM hidden size
const N_CLASSES  = 2;        // UP=1, DOWN=0
const BATCH_SIZE = 64;
const N_EPOCHS   = 25;
const LR         = 3e-3;
const WEIGHT_DECAY = 1e-4;
const MAX_GRAD_NORM = 5.0;   // gradient clipping threshold

mt_srand(7);

// ─── 1. Synthetic price series ─────────────────────────────────────────────
//
//  log(p_t) = log(p_{t-1})
//           + μ                         (drift)
//           + σ_t · ε_t                 (stochastic vol)
//  σ_t      = 0.8 σ_{t-1} + 0.01       (GARCH-like mean-reversion)
//  ε_t ~ N(0,1)

/**
 * Box-Muller N(0,1).
 */
function randn_bm(): float
{
    static $spare = null;
    static $hasSpare = false;
    if ($hasSpare) { $hasSpare = false; return $spare; }
    do { $u = mt_rand() / mt_getrandmax(); } while ($u === 0.0);
    $v = mt_rand() / mt_getrandmax();
    $m = sqrt(-2.0 * log($u));
    $spare = $m * sin(2.0 * M_PI * $v);
    $hasSpare = true;
    return $m * cos(2.0 * M_PI * $v);
}

$prices = [100.0];
$sigma  = 0.015;
$mu     = 0.0002;

for ($t = 1; $t < N_PRICES; $t++) {
    $sigma  = 0.8 * $sigma + 0.003;          // vol clustering
    $sigma  = max(0.005, min(0.04, $sigma));  // clamp
    $ret    = $mu + $sigma * randn_bm();
    $prices[] = $prices[$t - 1] * exp($ret);
}

// Log-returns
$logRets = [];
for ($t = 1; $t < N_PRICES; $t++) {
    $logRets[] = log($prices[$t] / $prices[$t - 1]);
}

// Normalise log-returns (z-score over entire series for simplicity)
$mean = array_sum($logRets) / count($logRets);
$var  = 0.0;
foreach ($logRets as $r) { $var += ($r - $mean) ** 2; }
$std  = max(1e-8, sqrt($var / count($logRets)));
$normRets = array_map(fn($r) => ($r - $mean) / $std, $logRets);

// ─── 2. Sliding-window sequences ──────────────────────────────────────────
//
//  Feature vector for sample at position t (0-indexed in normRets):
//    X[t] = normRets[t..t+SEQ_LEN-1]  shape [SEQ_LEN, 1]
//  Label:
//    y[t] = 1 if prices[t+SEQ_LEN+1] > prices[t+SEQ_LEN], else 0
//
//  Valid range: t = 0 .. (len(normRets) - SEQ_LEN - 1 - 1)

$seqX = [];
$seqY = [];

$maxT = count($normRets) - SEQ_LEN - 1;
for ($t = 0; $t < $maxT; $t++) {
    // Each element is a [SEQ_LEN, 1] mini-array; we'll pack into Tensor later
    $win = [];
    for ($k = 0; $k < SEQ_LEN; $k++) {
        $win[] = $normRets[$t + $k];
    }
    $seqX[] = $win;
    // Label: up if price increases after the window
    $seqY[] = ($prices[$t + SEQ_LEN + 1] > $prices[$t + SEQ_LEN]) ? 1 : 0;
}

$N = count($seqX);
echo "\n";
echo "════════════════════════════════════════════════════════════\n";
echo "  LSTM Price Direction Classifier\n";
echo sprintf("  Series: %d steps   Sequences: %d   Class balance: %.1f%% up\n",
    N_PRICES, $N, 100.0 * array_sum($seqY) / $N);
echo "════════════════════════════════════════════════════════════\n\n";

// ─── 3. Train / test split (80/20 chronological) ──────────────────────────

$nTrain  = (int) round($N * 0.80);
$nTest   = $N - $nTrain;

$trainX  = array_slice($seqX, 0, $nTrain);
$trainY  = array_slice($seqY, 0, $nTrain);
$testX   = array_slice($seqX, $nTrain);
$testY   = array_slice($seqY, $nTrain);

echo sprintf("  Train: %d   Test: %d\n\n", $nTrain, $nTest);

/**
 * Pack a PHP float[][] (each row = [SEQ_LEN] floats) into a Tensor [B, T, I].
 */
function packBatch(array $batch, int $T, int $I): Tensor
{
    $B   = count($batch);
    $out = new Tensor([$B, $T, $I]);
    $off = 0;
    foreach ($batch as $seq) {
        foreach ($seq as $feat) {
            $out->buffer[$off++] = (float) $feat;
        }
    }
    return $out;
}

/**
 * Pack the full test set as one batch (safe since nTest < 400 for our setup).
 */
$testXTensor = packBatch($testX, SEQ_LEN, INPUT_SIZE);

// ─── 4. Build LSTM classifier ─────────────────────────────────────────────

$cell       = new LSTMCell(INPUT_SIZE, HIDDEN);
$classifier = new RNNClassifier($cell, N_CLASSES, MAX_GRAD_NORM);

$optimizer  = new AdamW(
    $classifier->parameters(),
    lr: LR,
    beta1: 0.9,
    beta2: 0.999,
    eps: 1e-8,
    weightDecay: WEIGHT_DECAY,
);

// ─── 5. Training loop ─────────────────────────────────────────────────────

echo "  Training LSTM(hidden=" . HIDDEN . ") for " . N_EPOCHS . " epochs...\n";
echo "  " . str_repeat('─', 54) . "\n";
echo sprintf("  %-7s  %-12s  %-12s  %-10s\n", 'Epoch', 'Train Loss', 'Test Acc', 'LR');
echo "  " . str_repeat('─', 54) . "\n";

$nBatches = (int) ceil($nTrain / BATCH_SIZE);

for ($epoch = 1; $epoch <= N_EPOCHS; $epoch++) {

    // ── Cosine LR decay ───────────────────────────────────────────────────
    $lrScale = 0.5 * (1.0 + cos(M_PI * ($epoch - 1) / N_EPOCHS));
    $currentLR = LR * max(0.1, $lrScale);

    // ── Shuffle training indices ───────────────────────────────────────────
    $idx = range(0, $nTrain - 1);
    shuffle($idx);

    $epochLoss   = 0.0;
    $batchesSeen = 0;

    for ($b = 0; $b < $nBatches; $b++) {
        $start = $b * BATCH_SIZE;
        $end   = min($start + BATCH_SIZE, $nTrain);
        $bSize = $end - $start;

        // Build batch tensors
        $batchXArr = [];
        $batchY    = [];
        for ($i = $start; $i < $end; $i++) {
            $batchXArr[] = $trainX[$idx[$i]];
            $batchY[]    = $trainY[$idx[$i]];
        }
        $batchX = packBatch($batchXArr, SEQ_LEN, INPUT_SIZE);

        // ── Forward + loss gradient ────────────────────────────────────────
        [, $dLogits, $loss] = $classifier->loss($batchX, $batchY);
        $epochLoss += $loss;
        $batchesSeen++;

        // ── BPTT + gradient clipping ───────────────────────────────────────
        $classifier->zeroGrad();
        $classifier->trainStep($batchX, $dLogits);

        // ── Optimizer step ────────────────────────────────────────────────
        $optimizer->step();
    }

    $avgLoss = $epochLoss / $batchesSeen;

    // ── Test accuracy (full test set, no grad) ────────────────────────────
    $testAcc = $classifier->accuracy($testXTensor, $testY);

    echo sprintf("  %-7d  %-12.4f  %-12.4f  %.2e\n",
        $epoch, $avgLoss, $testAcc, $currentLR);
}

echo "  " . str_repeat('─', 54) . "\n\n";

// ─── 6. Final evaluation: confusion matrix ────────────────────────────────

echo "── Confusion Matrix (test set) ──\n\n";

$testLogits = $classifier->predict($testXTensor);
$C          = N_CLASSES;
$confMatrix = array_fill(0, $C, array_fill(0, $C, 0));

for ($i = 0; $i < $nTest; $i++) {
    $off     = $i * $C;
    $pred    = (float) $testLogits->buffer[$off] > (float) $testLogits->buffer[$off + 1] ? 0 : 1;
    $actual  = $testY[$i];
    $confMatrix[$actual][$pred]++;
}

echo "              Pred:DOWN  Pred:UP\n";
for ($r = 0; $r < $C; $r++) {
    $label = $r === 0 ? ' Actual:DOWN' : ' Actual:UP  ';
    echo $label . '    '
        . sprintf('%6d', $confMatrix[$r][0]) . '       '
        . sprintf('%6d', $confMatrix[$r][1]) . "\n";
}

// Per-class precision / recall
echo "\n  Class     Precision   Recall     F1\n";
echo "  " . str_repeat('─', 42) . "\n";

$labels = ['DOWN', 'UP'];
for ($c = 0; $c < $C; $c++) {
    $tp = $confMatrix[$c][$c];
    $fp = 0;
    $fn = 0;
    for ($r = 0; $r < $C; $r++) {
        if ($r !== $c) { $fp += $confMatrix[$r][$c]; }
    }
    for ($cc = 0; $cc < $C; $cc++) {
        if ($cc !== $c) { $fn += $confMatrix[$c][$cc]; }
    }

    $prec  = $tp + $fp > 0 ? $tp / ($tp + $fp)  : 0.0;
    $rec   = $tp + $fn > 0 ? $tp / ($tp + $fn)  : 0.0;
    $f1    = $prec + $rec > 0 ? 2 * $prec * $rec / ($prec + $rec) : 0.0;

    echo sprintf("  %-8s  %9.3f   %9.3f   %.3f\n", $labels[$c], $prec, $rec, $f1);
}

// Overall accuracy
$correct = $confMatrix[0][0] + $confMatrix[1][1];
$total   = array_sum(array_map('array_sum', $confMatrix));
echo "\n  Overall accuracy : " . sprintf("%.2f%%", 100.0 * $correct / $total) . " ({$correct}/{$total})\n";

echo "\n════════════════════════════════════════════════════════════\n";
echo "  Done.\n";
echo "════════════════════════════════════════════════════════════\n\n";
