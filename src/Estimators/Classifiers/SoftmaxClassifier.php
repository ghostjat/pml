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
        $flat = array_map('intval', $labels->toFlatArray());
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

                // Map label values → class indices (pure PHP map, no FFI)
                $rawLabels = $batch->labels()->toFlatArray();
                $idxArr    = array_map(fn($lbl) => (float)($this->classMap[(int)$lbl] ?? 0), $rawLabels);
                // One-hot via broadcast equal: classIdx[N,1] == arange[1,K] → [N,K]
                $classIdx = Tensor::fromArray($idxArr)->expandDims(1);          // [N,1]
                $arange   = Tensor::linspace(0.0, (float)($k - 1), $k);        // [K]
                $yOH      = $classIdx->equal($arange);                          // [N,K]

                // Forward: logits = X*W + b  [N × K]
                $logits = $x->matmul($this->weights)->addInplace($this->bias);

                // Numerically stable softmax in-place
                $proba  = $logits->copy()->rowSoftmaxInplace();                  // [N × K]

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
        return $logits->rowSoftmaxInplace();                        // [N × K] in-place
    }

    public function predict(Dataset $dataset): Tensor
    {
        // argmaxAxis(1) → [N] class indices; map through indexMap (O(N) pure PHP)
        $classIdx = $this->proba($dataset)->argmaxAxis(1)->toFlatArray();
        $preds    = array_map(fn($i) => (float)($this->indexMap[(int)$i] ?? 0), $classIdx);
        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->weights !== null;
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
        return $i;
    }
}
