<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
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
final class SoftmaxClassifier implements Learner, Probabilistic
{
    private ?Tensor $weights   = null;
    private ?Tensor $bias      = null;
    /** @var int[] label → class index */
    private array $classMap    = [];
    /** @var int[] class index → label */
    private array $indexMap    = [];

    public function __construct(
        private readonly int   $epochs       = 100,
        private readonly float $learningRate = 0.01,
        private readonly float $l2           = 0.0,
        private readonly int   $batchSize    = 32
    ) {}

    public function train(Dataset $dataset): void
    {
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("SoftmaxClassifier requires labeled data.");
        }

        // Build class index maps from integer labels (single C read)
        $flat = $labels->toFlatArray();
        $unique = array_values(array_unique($flat));
        sort($unique);
        $this->classMap = array_flip($unique);
        $this->indexMap = $unique;
        $k = count($unique);
        $d = $dataset->numColumns();

        // Weight: [D × K], Bias: [1 × K]
        $this->weights = Tensor::randomNormal([$d, $k], 0.0, 0.01);
        $this->bias    = Tensor::zeros(1, $k);

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x    = $batch->samples();                          // [N × D]
                $n    = (float) $x->shape()[0];

                // One-hot encode labels into [N × K] target matrix
                $yInts  = $batch->labels()->toFlatArray();
                $yData  = array_fill(0, (int) $n * $k, 0.0);
                foreach ($yInts as $i => $lbl) {
                    $yData[$i * $k + ($this->classMap[$lbl] ?? 0)] = 1.0;
                }
                $yOH = Tensor::fromArray(
                    array_chunk($yData, $k)
                );                                                   // [N × K]

                // Forward: logits = X*W + b  [N × K]
                $logits = $x->matmul($this->weights)->addInplace($this->bias);

                // Softmax in C: exp(x_i) / sum(exp(x_j))
                $expL   = $logits->exp();
                $sumExp = $expL->sumAxis(1)->expandDims(1);         // [N × 1]
                $proba  = $expL->div($sumExp);                      // [N × K]

                // dL/dLogits = (P - Y) / N
                $dLogits = $proba->sub($yOH)->mulScalarInplace(1.0 / $n);

                // Gradients
                $dW = $x->transpose()->matmul($dLogits);            // [D × K]
                $db = $dLogits->sumAxis(0);                         // [1 × K]

                // L2 regularization on weights
                if ($this->l2 > 0.0) {
                    $dW->addInplace($this->weights->mulScalar($this->l2));
                }

                // In-place parameter update
                $this->weights->subInplace($dW->mulScalarInplace($this->learningRate));
                $this->bias->subInplace($db->mulScalarInplace($this->learningRate));
            }
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("SoftmaxClassifier is not trained.");
        }
        $logits = $dataset->samples()->matmul($this->weights)->addInplace($this->bias);
        $expL   = $logits->exp();
        $sumExp = $expL->sumAxis(1)->expandDims(1);
        return $expL->divInplace($sumExp);                          // [N × K]
    }

    public function predict(Dataset $dataset): Tensor
    {
        $proba   = $this->proba($dataset);                          // [N × K]
        $indices = $proba->argsort(1);                              // sort along K axis
        $n       = $dataset->numRows();
        // argmax per row: take last column after argsort ascending
        $k       = count($this->indexMap);
        $argmaxT = $indices->col($k - 1);                          // [N] — index of max class
        $intIdx  = $argmaxT->toFlatArray();

        $preds = [];
        foreach ($intIdx as $idx) {
            $preds[] = $this->indexMap[(int) $idx] ?? 0;
        }
        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }
}
