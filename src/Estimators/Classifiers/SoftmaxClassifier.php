<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Softmax Classifier (Multinomial Logistic Regression).
 * Trains a linear model with cross-entropy loss + softmax output for K classes.
 *
 * JIT & Memory Optimized:
 * - Full forward/backward pass is a chain of in-place BLAS calls.
 * - Class probabilities stay in C memory; PHP only reads the argmax scalar.
 * - Weight matrix is [features × K]; one matmul covers all classes in parallel.
 */
final class SoftmaxClassifier implements Learner, Probabilistic, Persistable
{
    private ?Tensor $weights     = null;
    private ?Tensor $bias        = null;
    /** @var int[] label → class index */
    private array $classMap      = [];
    /** @var int[] class index → label */
    private array $indexMap      = [];
    private ?Tensor $labelToIdx  = null;  // [maxLabel+1]: label → class index gather table
    private ?Tensor $idxToLabel  = null;  // [K]:          class index → label gather table

    public function __construct(
        private readonly int   $epochs       = 100,
        private readonly float $learningRate = 0.01,
        private readonly float $l2           = 0.0,
        private readonly int   $batchSize    = 32
    ) {}

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("SoftmaxClassifier requires labeled data.");
        }

        // Build class index maps from integer labels (single C read)
        $flat = array_map('intval', $labels->toFlatArray());
        $unique = array_values(array_unique($flat));
        sort($unique);
        $this->classMap = array_flip($unique);
        $this->indexMap = $unique;
        $this->buildLookupTensors();
        $k = count($unique);
        $d = $dataset->numColumns();

        // Weight: [D × K], Bias: [1 × K]
        $this->weights = Tensor::randomNormal([$d, $k], 0.0, 0.01);
        $this->bias    = Tensor::zeros(1, $k);

        // Pre-allocate gradient buffers once — reused every batch (zero-alloc hot path)
        $dW = Tensor::zeros($d, $k);
        $db = Tensor::zeros(1, $k);

        // For small batch sizes, disable OpenMP thread spawning overhead;
        // BLAS handles its own threading for the matmul dominants.
        if ($this->batchSize < 2048) {
            Tensor::configureThreading(1, 16);
        }

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x = $batch->samples();                             // [N × D]
                $n = (float) $x->shape()[0];

                // Gather class indices in C — eliminates PHP array_map over batchSize
                $classIdx = Tensor::gatherIndices($batch->labels(), $this->labelToIdx); // [N]

                // One-hot in C: single FFI call, zero PHP loops  [N × K]
                $yOH = Tensor::onehot($classIdx, $k);

                // Forward: logits = X*W + b  [N × K] — in-place on new alloc
                $logits = $x->matmul($this->weights)->addInplace($this->bias);

                // Numerically-stable softmax in-place (no copy needed)
                $logits->rowSoftmaxInplace();

                // dLogits = (P − Y) / N — reuse logits buffer (sub + scale in-place)
                $logits->subInplace($yOH)->mulScalarInplace(1.0 / $n);

                // Gradients into pre-allocated buffers
                $dW->matmulInto($x, $logits, true, false);   // X^T @ dLogits → [D,K]
                $db->sumAxisInto($logits, 0);                // sum rows → [1,K]

                // L2 on weights (fused: dW += l2 * W)
                if ($this->l2 > 0.0) {
                    $dW->addInplace($this->weights->mulScalar($this->l2));
                }

                // In-place parameter update
                $this->weights->subInplace($dW->mulScalarInplace($this->learningRate));
                $this->bias->subInplace($db->mulScalarInplace($this->learningRate));
            }
        }

        // Restore threading to full parallel for inference and other estimators
        if ($this->batchSize < 2048) {
            Tensor::configureThreading(16, 16);
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("SoftmaxClassifier is not trained.");
        }
        $logits = $dataset->samples()->matmul($this->weights)->addInplace($this->bias);
        return $logits->rowSoftmaxInplace();                        // [N × K] in-place
    }

    public function predict(Dataset $dataset): Tensor
    {
        // argmaxAxis(1) → [N] class indices; gather labels in C — zero PHP per-sample
        return Tensor::gatherIndices($this->proba($dataset)->argmaxAxis(1), $this->idxToLabel);
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }

    private function buildLookupTensors(): void
    {
        // [K] float32: class index → original label
        $this->idxToLabel = Tensor::fromArray(array_map('floatval', $this->indexMap));
        // [maxLabel+1] float32: label → class index (dense lookup table)
        $maxLabel = max(array_map('intval', array_keys($this->classMap)));
        $table = array_fill(0, $maxLabel + 1, 0.0);
        foreach ($this->classMap as $label => $idx) {
            $table[(int)$label] = (float)$idx;
        }
        $this->labelToIdx = Tensor::fromArray($table);
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['epochs'=>$this->epochs,'learningRate'=>$this->learningRate,'l2'=>$this->l2,'batchSize'=>$this->batchSize,'classMap'=>$this->classMap,'indexMap'=>$this->indexMap]));
        if ($this->weights !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['weights' => $this->weights, 'bias' => $this->bias]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['epochs'], (float)$c['learningRate'], (float)$c['l2'], (int)$c['batchSize']);
        $i->classMap = $c['classMap'] ?? []; $i->indexMap = $c['indexMap'] ?? [];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->weights = $t['weights'] ?? null; $i->bias = $t['bias'] ?? null; }
        $i->buildLookupTensors();
        return $i;
    }
}
