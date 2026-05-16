<?php
declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * ADALINE — Adaptive Linear Neuron (Widrow-Hoff LMS rule).
 * A single linear unit trained with batch gradient descent on MSE loss.
 *
 * JIT & Memory Optimized:
 * - All arithmetic is pure in-place BLAS (matmul + scalar ops).
 * - No intermediate arrays cross the FFI boundary during training.
 */
final class Adaline implements Learner, Persistable
{
    private ?Tensor $weights = null;
    private float   $bias    = 0.0;

    public function __construct(
        private readonly int   $epochs       = 100,
        private readonly float $learningRate = 0.01,
        private readonly int   $batchSize    = 32
    ) {}

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $d = $dataset->numColumns();
        $this->weights = Tensor::randomNormal([$d, 1], 0.0, 0.001);
        $this->bias    = 0.0;

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x   = $batch->samples();
                $y   = $batch->labels();
                $y   = $y->ndim() === 1 ? $y->expandDims(1) : $y;
                $n   = (float) $x->shape()[0];

                // Net input: Z = X*W + b
                $z   = $x->matmul($this->weights)->addScalarInplace($this->bias);

                // Error: dZ = Z - Y  (MSE gradient)
                $dz  = $z->sub($y);

                // Weight gradient: dW = X^T * dZ / N
                $dw  = $x->transpose()->matmul($dz)->mulScalarInplace(1.0 / $n);
                $db  = $dz->mean();

                $this->weights->subInplace($dw->mulScalarInplace($this->learningRate));
                $this->bias -= $db * $this->learningRate;
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Adaline is not trained.");
        }
        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['epochs'=>$this->epochs,'learningRate'=>$this->learningRate,'batchSize'=>$this->batchSize,'bias'=>$this->bias]));
        if ($this->weights !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['weights' => $this->weights]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((int)$c['epochs'], (float)$c['learningRate'], (int)$c['batchSize']);
        $i->bias = (float)$c['bias'];
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->weights = $t['weights'] ?? null; }
        return $i;
    }
}
