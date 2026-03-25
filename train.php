<?php

/**
 * train.php — NanoGPT-style training script (v2: production-ready)
 * [QA Approved - Features Execution Guards and Tokenizer State Persistence]
 */

declare(strict_types=1);

require_once __DIR__ . '/vendor/autoload.php';

use Pml\{Tensor, Ops, BlasEngine};
use Pml\Training\{DataLoader, CrossEntropyLoss, AdamW};
use Pml\IO\{SafetensorsLoader, SafetensorsWriter};

// ── Model hyper-parameters ────────────────────────────────────────────────
const D_MODEL    = 256;
const N_LAYERS   = 6;
const N_HEADS    = 4;          
const D_FF       = 4 * D_MODEL; 
const SPLIT_RATIO = 0.9;        

if (D_MODEL % N_HEADS !== 0) {
    fwrite(STDERR, "Error: D_MODEL must be divisible by N_HEADS.\n");
    exit(1);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Helper Functions & Classes (Safe to require in other scripts)
// ═══════════════════════════════════════════════════════════════════════════

function buildSinusoidalPE(int $maxSeqLen, int $dModel): Tensor {
    $pe = Tensor::zeros([$maxSeqLen, $dModel]);
    for ($pos = 0; $pos < $maxSeqLen; $pos++) {
        for ($i = 0; $i < intdiv($dModel, 2); $i++) {
            $angle = $pos / (10000.0 ** (2.0 * $i / $dModel));
            $pe->buffer[$pos * $dModel + 2 * $i]     = (float) sin($angle);
            $pe->buffer[$pos * $dModel + 2 * $i + 1] = (float) cos($angle);
        }
    }
    return $pe->detach();
}

function embeddingWithGrad(Tensor $weight, array $ids): Tensor {
    $dModel = $weight->shape[1];
    $seqLen = count($ids);
    $out    = Tensor::zeros([$seqLen, $dModel]);
    $ffi    = BlasEngine::get()->ffi;

    for ($i = 0; $i < $seqLen; $i++) {
        $id  = $ids[$i];
        $src = \FFI::cast('float*', \FFI::addr($weight->buffer[$id * $dModel]));
        $dst = \FFI::cast('float*', \FFI::addr($out->buffer[$i * $dModel]));
        $ffi->cblas_scopy($dModel, $src, 1, $dst, 1);
    }

    if ($weight->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$weight];
        $capturedIds       = $ids;
        $out->_backward = static function () use ($weight, $out, $capturedIds, $dModel, $seqLen, $ffi): void {
            $weight->initGrad();
            for ($i = 0; $i < $seqLen; $i++) {
                $id   = $capturedIds[$i];
                $gSrc = \FFI::cast('float*', \FFI::addr($out->grad[$i * $dModel]));
                $gDst = \FFI::cast('float*', \FFI::addr($weight->grad[$id * $dModel]));
                $ffi->cblas_saxpy($dModel, 1.0, $gSrc, 1, $gDst, 1);
            }
        };
    }
    return $out;
}

function scaleWithGrad(Tensor $x, float $scale): Tensor {
    $out = $x->clone();
    BlasEngine::get()->ffi->cblas_sscal($out->size, $scale, $out->buffer, 1);
    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$x];
        $out->_backward = static function () use ($x, $out, $scale): void {
            $x->initGrad();
            BlasEngine::get()->ffi->cblas_saxpy($out->size, $scale, $out->grad, 1, $x->grad, 1);
        };
    }
    return $out;
}

function softmaxRowsWithGrad(Tensor $x): Tensor {
    [$M, $N] = $x->shape;
    $out     = Tensor::zeros([$M, $N]);
    $ffi     = BlasEngine::get()->ffi;

    for ($i = 0; $i < $M; $i++) {
        $off  = $i * $N;
        $maxV = (float) $x->buffer[$off];
        for ($j = 1; $j < $N; $j++) {
            $v = (float) $x->buffer[$off + $j];
            if ($v > $maxV) $maxV = $v;
        }
        $sum = 0.0;
        for ($j = 0; $j < $N; $j++) {
            $e = exp((float) $x->buffer[$off + $j] - $maxV);
            $out->buffer[$off + $j] = $e;
            $sum += $e;
        }
        $rowPtr = \FFI::cast('float*', \FFI::addr($out->buffer[$off]));
        $ffi->cblas_sscal($N, 1.0 / $sum, $rowPtr, 1);
    }

    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$x];
        $out->_backward = static function () use ($x, $out, $M, $N): void {
            $x->initGrad();
            for ($i = 0; $i < $M; $i++) {
                $off = $i * $N;
                $dot = 0.0;
                for ($j = 0; $j < $N; $j++) {
                    $dot += (float) $out->buffer[$off + $j] * (float) $out->grad[$off + $j];
                }
                for ($j = 0; $j < $N; $j++) {
                    $p  = (float) $out->buffer[$off + $j];
                    $dP = (float) $out->grad[$off + $j];
                    $x->grad[$off + $j] = (float) $x->grad[$off + $j] + $p * ($dP - $dot);
                }
            }
        };
    }
    return $out;
}

function reluWithGrad(Tensor $x): Tensor {
    $n   = $x->size;
    $out = Tensor::zeros($x->shape);
    for ($i = 0; $i < $n; $i++) {
        $v = (float) $x->buffer[$i];
        if ($v > 0.0) $out->buffer[$i] = $v;
    }
    if ($x->requiresGrad) {
        $out->requiresGrad = true;
        $out->_prev        = [$x];
        $out->_backward = static function () use ($x, $out, $n): void {
            $x->initGrad();
            for ($i = 0; $i < $n; $i++) {
                if ((float) $out->buffer[$i] > 0.0) {
                    $x->grad[$i] = (float) $x->grad[$i] + (float) $out->grad[$i];
                }
            }
        };
    }
    return $out;
}

function applyCausalMaskInPlace(Tensor $scores): void {
    $seqLen = $scores->shape[0];
    for ($i = 0; $i < $seqLen; $i++) {
        for ($j = $i + 1; $j < $seqLen; $j++) {
            $scores->buffer[$i * $seqLen + $j] = -1.0e9;
        }
    }
}

final class TinyGPT {
    public Tensor $tokenEmb;
    public array $layers = [];
    public Tensor $lmHead;
    public Tensor $finalNorm;
    private Tensor $posEnc;
    private readonly int $headDim;

    public function __construct(
        private readonly int $vocabSize,
        private readonly int $dModel   = D_MODEL,
        private readonly int $nLayers  = N_LAYERS,
        private readonly int $nHeads   = N_HEADS,
        private readonly int $dFF      = D_FF,
        int                  $maxSeqLen = 128,
    ) {
        $this->headDim = intdiv($dModel, $nHeads);
        $embStd         = sqrt(2.0 / $dModel);
        $this->tokenEmb = Tensor::randn([$vocabSize, $dModel], 0.0, $embStd);
        $this->tokenEmb->requiresGrad = true;
        $this->posEnc = buildSinusoidalPE($maxSeqLen, $dModel);

        for ($l = 0; $l < $nLayers; $l++) {
            $this->layers[$l] = [
                'wq' => $this->initWeight([$dModel, $dModel]),
                'wk' => $this->initWeight([$dModel, $dModel]),
                'wv' => $this->initWeight([$dModel, $dModel]),
                'wo' => $this->initWeight([$dModel, $dModel]),
                'w1' => $this->initWeight([$dFF, $dModel]),
                'w2' => $this->initWeight([$dModel, $dFF]),
                'rms1_w' => $this->initNormWeight($dModel),
                'rms2_w' => $this->initNormWeight($dModel),
            ];
        }

        $this->lmHead    = $this->initWeight([$vocabSize, $dModel]);
        $this->finalNorm = $this->initNormWeight($dModel);
    }

    public function forward(array $tokenIds): Tensor {
        $seqLen = count($tokenIds);
        $x        = embeddingWithGrad($this->tokenEmb, $tokenIds);
        $posSlice = new Tensor([$seqLen, $this->dModel], $this->posEnc->buffer);
        $x        = Ops::add($x, $posSlice);

        foreach ($this->layers as $layer) {
            $x = $this->block($x, $layer, $seqLen);
        }

        $x = Ops::rmsNorm($x, $this->finalNorm);
        return Ops::matmul($x, $this->lmHead, false, true); 
    }

    public function getParams(): array {
        $params = [$this->tokenEmb];
        foreach ($this->layers as $layer) {
            foreach ($layer as $w) {
                $params[] = $w;
            }
        }
        $params[] = $this->finalNorm;
        $params[] = $this->lmHead;
        return $params;
    }

    public function namedParams(): array {
        $named = ['token_emb' => $this->tokenEmb];
        foreach ($this->layers as $l => $layer) {
            foreach ($layer as $key => $w) {
                $named["layer.{$l}.{$key}"] = $w;
            }
        }
        $named['final_norm'] = $this->finalNorm;
        $named['lm_head']    = $this->lmHead;
        return $named;
    }

    private function initWeight(array $shape): Tensor {
        $fanIn = $shape[count($shape) - 1];
        $std   = sqrt(2.0 / $fanIn);
        $w     = Tensor::randn($shape, 0.0, $std);
        $w->requiresGrad = true;
        return $w;
    }

    private function initNormWeight(int $d): Tensor {
        $w = Tensor::ones([$d]);
        $w->requiresGrad = true;
        return $w;
    }

    private function block(Tensor $x, array $layer, int $seqLen): Tensor {
        $xNorm = Ops::rmsNorm($x, $layer['rms1_w']);
        $q = Ops::matmul($xNorm, $layer['wq'], false, true);
        $k = Ops::matmul($xNorm, $layer['wk'], false, true);
        $v = Ops::matmul($xNorm, $layer['wv'], false, true);

        $headOutputs = [];
        $scaleFactor = 1.0 / sqrt((float) $this->headDim);

        for ($h = 0; $h < $this->nHeads; $h++) {
            $colStart = $h * $this->headDim;
            $colEnd   = $colStart + $this->headDim;

            $q_h = Ops::sliceCols($q, $colStart, $colEnd); 
            $k_h = Ops::sliceCols($k, $colStart, $colEnd);
            $v_h = Ops::sliceCols($v, $colStart, $colEnd);

            $scores_h = Ops::matmul($q_h, $k_h, false, true); 
            $scores_h = scaleWithGrad($scores_h, $scaleFactor);
            applyCausalMaskInPlace($scores_h); 
            $attn_h = softmaxRowsWithGrad($scores_h);

            $headOutputs[] = Ops::matmul($attn_h, $v_h); 
        }

        $attnOut = Ops::concatCols($headOutputs);
        $o = Ops::matmul($attnOut, $layer['wo'], false, true); 
        $x = Ops::add($x, $o);                                 

        $xNorm = Ops::rmsNorm($x, $layer['rms2_w']);
        $h_ff = Ops::matmul($xNorm, $layer['w1'], false, true); 
        $h_ff = reluWithGrad($h_ff);
        $ff   = Ops::matmul($h_ff, $layer['w2'], false, true);  

        return Ops::add($x, $ff); 
    }
}

function evaluate(TinyGPT $model, DataLoader $loader, CrossEntropyLoss $criterion, int $nBatches = 20): array {
    $params = $model->getParams();
    foreach ($params as $p) { $p->requiresGrad = false; }

    $totalLoss    = 0.0;
    $totalCorrect = 0;
    $totalTokens  = 0;
    $totalSeqs    = 0;

    for ($b = 0; $b < $nBatches; $b++) {
        [$xBatch, $yBatch] = $loader->getBatch('val');
        foreach ($xBatch as $idx => $xSeq) {
            $ySeq   = $yBatch[$idx];
            $logits = $model->forward($xSeq);  

            $loss         = $criterion->forward($logits, $ySeq);
            $totalLoss   += (float) $loss->buffer[0];
            $totalSeqs++;

            $seqLen    = count($ySeq);
            $vocabSize = $logits->shape[1];

            for ($t = 0; $t < $seqLen; $t++) {
                $off    = $t * $vocabSize;
                $argmax = 0;
                $maxVal = (float) $logits->buffer[$off];
                for ($v = 1; $v < $vocabSize; $v++) {
                    $val = (float) $logits->buffer[$off + $v];
                    if ($val > $maxVal) {
                        $maxVal = $val;
                        $argmax = $v;
                    }
                }
                if ($argmax === $ySeq[$t]) { $totalCorrect++; }
                $totalTokens++;
            }
        }
    }

    foreach ($params as $p) { $p->requiresGrad = true; }
    return [
        'loss' => $totalSeqs > 0 ? $totalLoss / $totalSeqs : 0.0,
        'acc'  => $totalTokens > 0 ? $totalCorrect / $totalTokens : 0.0,
    ];
}


// ═══════════════════════════════════════════════════════════════════════════
//  EXECUTION GUARD: The Training Loop 
//  (Will ONLY run if train.php is executed directly via CLI)
// ═══════════════════════════════════════════════════════════════════════════

// Detect if this file was executed directly (php train.php) or required by another script.
$isMainScript = (php_sapi_name() === 'cli' && realpath($_SERVER['SCRIPT_FILENAME']) === realpath(__FILE__));

if ($isMainScript) {

    $opts = getopt('', ['steps::', 'seqlen::', 'lr::', 'data::', 'batchsize::', 'evalsteps::', 'checkpoint::']);

    $nSteps     = (int)   ($opts['steps']      ?? 5000);
    $seqLen     = (int)   ($opts['seqlen']     ?? 256);
    $lr         = (float) ($opts['lr']         ?? 1e-4);
    $dataFile   = (string)($opts['data']       ?? __DIR__ . '/datasets/career_counselling_10000.csv');
    $batchSize  = (int)   ($opts['batchsize']  ?? 4);
    $evalEvery  = (int)   ($opts['evalsteps']  ?? 50);  
    $ckptPath   = (string)($opts['checkpoint'] ?? __DIR__ . '/career-nano.safetensors');

    echo "[train] Loading dataset from: {$dataFile}\n";

    if (!file_exists($dataFile)) {
        echo "[train] Dataset not found. Generating synthetic corpus...\n";
        $topics  = ['engineering', 'data science', 'software', 'design', 'finance', 'law', 'medicine'];
        $lines   = [];
        for ($i = 0; $i < 200; $i++) {
            $t       = $topics[$i % count($topics)];
            $lines[] = "Q: What career suits me if I like {$t}?\nA: Consider {$t} as a strong career path. Build skills step by step.\n";
        }
        $corpus = implode('', $lines);
    } else {
        $fp = fopen($dataFile, 'r');
        fgetcsv($fp);
        $rows = [];
        while (($row = fgetcsv($fp)) !== false) {
            if (count($row) >= 2) {
                // QA FIX: Inject a strict End-of-Sequence marker
                $q = trim($row[0]);
                $a = trim($row[1]);
                $rows[] = "Q: {$q}\nA: {$a}<|end|>";
            }
        }
        fclose($fp);
        //$rows = array_slice($rows, 0, 5000); // Increase data usage!
        $corpus = implode('', $rows);
        echo '[train] Loaded ' . count($rows) . " Q&A pairs (" . strlen($corpus) . " chars).\n";
    }

    $rawBytes  = array_values(unpack('C*', $corpus));
    $byteToId  = [];
    $idToByte  = [];
    foreach ($rawBytes as $b) {
        if (!isset($byteToId[$b])) {
            $id            = count($byteToId);
            $byteToId[$b]  = $id;
            $idToByte[$id] = $b;
        }
    }
    ksort($byteToId);
    $vocabSize = count($byteToId);
    $tokens    = array_map(fn(int $b) => $byteToId[$b], $rawBytes);

    echo "[train] Vocabulary size: {$vocabSize} unique bytes.\n";
    echo "[train] Corpus tokens:   " . count($tokens) . "\n";

    // ── QA FIX: Export Tokenizer to JSON so chat scripts do not rebuild it ──
    $tokenizerPath = __DIR__ . '/tokenizer.json';
    file_put_contents($tokenizerPath, json_encode([
        'vocabSize' => $vocabSize,
        'byteToId'  => $byteToId,
        'idToByte'  => $idToByte
    ], JSON_PRETTY_PRINT));
    echo "[train] Tokenizer mapping strictly saved to: {$tokenizerPath}\n";

    echo "\n[train] Initialising TinyGPT v2...\n";

    $model = new TinyGPT(vocabSize: $vocabSize, maxSeqLen: $seqLen + 16);

    $totalParams = 0;
    foreach ($model->getParams() as $p) { $totalParams += $p->size; }
    echo "[train] Total parameters: " . number_format($totalParams) . "\n";

    $loader    = new DataLoader($tokens, $seqLen, $batchSize, SPLIT_RATIO);
    $criterion = new CrossEntropyLoss();
    $optimizer = new AdamW(
        params:      $model->getParams(),
        lr:          $lr,
        beta1:       0.9,
        beta2:       0.999,
        eps:         1e-8,
        weightDecay: 0.1,
    );

    $resumeStep = 0;
    if (file_exists($ckptPath)) {
        echo "\n[train] Found checkpoint at {$ckptPath} — resuming.\n";
        $loadedTensors = SafetensorsLoader::load($ckptPath, false);
        $namedParams   = $model->namedParams();
        $ffi           = BlasEngine::get()->ffi;

        $loadedCount = 0;
        foreach ($namedParams as $name => $param) {
            if (isset($loadedTensors[$name])) {
                $ffi->cblas_scopy($param->size, $loadedTensors[$name]->buffer, 1, $param->buffer, 1);
                $loadedCount++;
            }
        }
        echo "[train] Restored {$loadedCount}/" . count($namedParams) . " weight tensors.\n";
        $optimizer->loadNamedState($loadedTensors, $namedParams);

        if (isset($loadedTensors['__opt_step__'])) {
            $resumeStep = (int) (float) $loadedTensors['__opt_step__']->buffer[0];
            $optimizer->setStep($resumeStep);
        }
    } else {
        echo "\n[train] No checkpoint found — training from scratch.\n";
    }

    echo "[train] Starting training for {$nSteps} steps...\n\n";

    $startTime    = microtime(true);
    $logFrequency = 10;
    $avgLoss      = 0.0; 

    for ($step = 1; $step <= $nSteps; $step++) {
        $optimizer->zeroGrad();
        [$xBatch, $yBatch] = $loader->getBatch('train');

        $totalLoss = 0.0;
        foreach ($xBatch as $b => $xSeq) {
            $ySeq = $yBatch[$b];
            $logits = $model->forward($xSeq);
            $loss = $criterion->forward($logits, $ySeq);
            $loss->backward();
            $totalLoss += (float) $loss->buffer[0];
        }

        $avgLoss = $totalLoss / $batchSize;
        $gradNorm = AdamW::clipGradNorm($model->getParams(), 1.0);
        $optimizer->step();

        if ($step % $logFrequency === 0 || $step === 1) {
            $elapsed   = microtime(true) - $startTime;
            $stepsPerS = $step / max($elapsed, 1e-9);
            $eta       = ($nSteps - $step) / max($stepsPerS, 1e-9);
            printf("step %4d/%d  loss=%.4f  gnorm=%.3f  %.1f steps/s  ETA %.0fs\n", $step, $nSteps, $avgLoss, $gradNorm, $stepsPerS, $eta);
        }

        if ($loader->valSize() > 0 && $step % $evalEvery === 0) {
            $metrics = evaluate($model, $loader, $criterion, 20);
            printf("  [eval]  step %d  val_loss=%.4f  val_acc=%.2f%%\n", $step, $metrics['loss'], $metrics['acc'] * 100.0);
        }
        // ── Periodic Auto-Save (The Upgrade) ──────────────────────────────────
        $saveEvery = 100; // Save to disk every 500 steps
        
        if ($step % $saveEvery === 0) {
            $currentTotalStep = $resumeStep + $step;
            echo "  [save] Auto-saving intermediate checkpoint at step {$currentTotalStep}...\n";
            
            $namedParams   = $model->namedParams();
            $optState      = $optimizer->getNamedState($namedParams);
            $optStepTensor = Tensor::full([1], (float) $currentTotalStep);

            $allTensors = array_merge($namedParams, $optState, ['__opt_step__' => $optStepTensor]);

            SafetensorsWriter::write($ckptPath, $allTensors, [
                'arch'       => 'TinyGPT-v2',
                'dModel'     => (string) D_MODEL,
                'nLayers'    => (string) N_LAYERS,
                'nHeads'     => (string) N_HEADS,
                'dFF'        => (string) D_FF,
                'vocabSize'  => (string) $vocabSize,
                'seqLen'     => (string) $seqLen,
                'totalSteps' => (string) $currentTotalStep,
                'lr'         => (string) $lr,
            ]);
        }
        // ──────────────────────────────────────────────────────────────────────
    }

    $elapsed = microtime(true) - $startTime;
    echo "\n[train] Training complete. {$nSteps} steps in " . number_format($elapsed, 1) . "s.\n";

    if ($loader->valSize() > 0) {
        $finalMetrics = evaluate($model, $loader, $criterion, 40);
        printf("[eval]  FINAL  val_loss=%.4f  val_acc=%.2f%%\n", $finalMetrics['loss'], $finalMetrics['acc'] * 100.0);
    } else {
        $finalMetrics = ['loss' => 0.0, 'acc' => 0.0];
    }

    echo "\n[train] Saving checkpoint to: {$ckptPath}\n";

    $namedParams  = $model->namedParams();
    $optState     = $optimizer->getNamedState($namedParams);
    $optStepTensor = Tensor::full([1], (float) ($resumeStep + $nSteps));

    $allTensors = array_merge($namedParams, $optState, ['__opt_step__' => $optStepTensor]);

    SafetensorsWriter::write($ckptPath, $allTensors, [
        'arch'       => 'TinyGPT-v2',
        'dModel'     => (string) D_MODEL,
        'nLayers'    => (string) N_LAYERS,
        'nHeads'     => (string) N_HEADS,
        'dFF'        => (string) D_FF,
        'vocabSize'  => (string) $vocabSize,
        'seqLen'     => (string) $seqLen,
        'totalSteps' => (string) ($resumeStep + $nSteps),
        'lr'         => (string) $lr,
        'finalLoss'  => number_format($avgLoss, 6),
        'valLoss'    => number_format($finalMetrics['loss'], 6),
        'valAcc'     => number_format($finalMetrics['acc'], 6),
    ]);

    $ckptBytes = file_exists($ckptPath) ? filesize($ckptPath) : 0;
    $ckptMiB   = number_format($ckptBytes / 1_048_576, 2);
    echo "[train] Checkpoint saved ({$ckptMiB} MiB).  Done.\n";
}