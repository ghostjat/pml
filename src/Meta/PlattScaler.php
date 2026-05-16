<?php
declare(strict_types=1);

namespace Pml\Meta;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Platt Scaling — calibrates the probability outputs of a binary classifier.
 *
 * After fitting the wrapped estimator, trains a logistic regression (A, B scalars)
 * on its raw decision scores to produce calibrated probabilities:
 *   P(y=1|f) = sigmoid(A*f + B)
 *
 * Training uses 100 epochs of gradient descent on cross-entropy loss.
 * Zero PHP arrays; all math via Tensor ops.
 */
final class PlattScaler implements Learner, Probabilistic, Persistable
{
    private ?Tensor $scaleA = null;  // [1] scalar A
    private ?Tensor $scaleB = null;  // [1] scalar B

    public function __construct(
        private readonly Probabilistic&Learner $estimator,
        private readonly int   $epochs = 100,
        private readonly float $lr     = 0.01
    ) {}

    public function train(Dataset $dataset, mixed ...$options): void
    {
        // 1. Train the wrapped estimator
        $this->estimator->train($dataset);

        // 2. Get raw scores (column 1 = P(positive) from proba [N,2], or [N] logits)
        $proba = $this->estimator->proba($dataset);
        $shape = $proba->shape();
        $scores = ($proba->ndim() === 2 && $shape[1] >= 2)
            ? $proba->col(1)->copy()       // [N] P(pos) from [N,2]
            : $proba->copy();              // [N] raw scores

        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("PlattScaler requires labeled data.");
        }

        $N = (float)$scores->size();

        // 3. Fit logistic regression: P = sigmoid(A*f + B) via gradient descent
        $this->scaleA = Tensor::fromArray([1.0]);
        $this->scaleB = Tensor::fromArray([0.0]);

        for ($e = 0; $e < $this->epochs; $e++) {
            // logit = A*f + B → [N]
            $logit = $scores->mulScalar((float)$this->scaleA->buffer()[0])
                            ->addScalarInplace((float)$this->scaleB->buffer()[0]);
            $p     = $logit->copy()->sigmoidInplace();               // [N]

            // grad_A = sum((p-y)*f) / N,  grad_B = sum(p-y) / N
            $diff   = $p->sub($y);
            $gradA  = $diff->mul($scores)->sum() / $N;
            $gradB  = $diff->sum() / $N;

            $this->scaleA->buffer()[0] -= $this->lr * $gradA;
            $this->scaleB->buffer()[0] -= $this->lr * $gradB;
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("PlattScaler has not been trained.");
        }
        $proba  = $this->estimator->proba($dataset);
        $shape  = $proba->shape();
        $scores = ($proba->ndim() === 2 && $shape[1] >= 2)
            ? $proba->col(1)->copy()
            : $proba->copy();

        $logit = $scores->mulScalar((float)$this->scaleA->buffer()[0])
                        ->addScalarInplace((float)$this->scaleB->buffer()[0]);
        $p1    = $logit->sigmoidInplace();                           // [N] calibrated P(pos)

        $N  = $scores->size();
        $p0 = Tensor::ones($N)->sub($p1);
        return Tensor::concat([$p0->expandDims(1), $p1->expandDims(1)], 1); // [N,2]
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->estimator->predict($dataset);
    }

    public function trained(): bool
    {
        return $this->scaleA !== null;
    }

    public function getWrapped(): Probabilistic&Learner
    {
        return $this->estimator;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        if (!($this->estimator instanceof Persistable)) {
            throw new RuntimeException("PlattScaler::save() requires the wrapped estimator to implement Persistable.");
        }
        $this->estimator->save($dir . '/estimator');
        file_put_contents($dir . '/config.json', json_encode(['epochs' => $this->epochs, 'lr' => $this->lr, 'estimatorClass' => get_class($this->estimator)]));
        if ($this->scaleA !== null) {
            SafeTensorsIO::save($dir . '/model.safetensors', ['scale_a' => $this->scaleA, 'scale_b' => $this->scaleB]);
        }
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $estimatorClass = $c['estimatorClass'];
        $estimator = $estimatorClass::load($dir . '/estimator');
        $i = new self($estimator, (int) $c['epochs'], (float) $c['lr']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $t = SafeTensorsIO::load($stPath);
            $i->scaleA = $t['scale_a'] ?? null;
            $i->scaleB = $t['scale_b'] ?? null;
        }
        return $i;
    }
}
