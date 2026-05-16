<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Support Vector Classifier (Linear SVM).
 * Maximizes the geometric margin separating two classes using Hinge Loss.
 * * JIT & Memory Optimized:
 * - Maps labels dynamically to {-1, 1} for ultra-fast C-level subgradient vectorization.
 * - Modifies gradients via boolean masking (`less`) to bypass iteration.
 */
final class SVC implements Learner, Persistable
{
    private float $c;
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private float $bias = 0.0;

    /**
     * @param float $c The penalty parameter for margin violations. Lower C enforces a wider margin.
     */
    public function __construct(float $c = 1.0, int $epochs = 100, float $learningRate = 0.01, int $batchSize = 32)
    {
        $this->c = $c;
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
        $this->batchSize = $batchSize;
    }

    public function train(Dataset $dataset, mixed ...$options): void
    {
        $features = $dataset->numColumns();
        $this->weights = Tensor::randomNormal([$features, 1], 0.0, 0.01);
        $this->bias = 0.0;

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x = $batch->samples();
                $y = $batch->labels();
                
                // Map Labels from {0, 1} to {-1, 1} in C for Hinge Loss formulation
                $yMapped = $y->mulScalar(2.0)->addScalarInplace(-1.0);
                $yMapped = $yMapped->ndim() === 1 ? $yMapped->expandDims(1) : $yMapped;
                $n = (float) $x->shape()[0];

                // Z = X * W + b
                $z = $x->matmul($this->weights)->addScalarInplace($this->bias);
                
                // Geometric Margin: Y * Z
                $margin = $yMapped->mul($z);

                // Margin Violation Mask: margin < 1.0
                $one = Tensor::zeros(1)->addScalarInplace(1.0);
                $violationMask = $margin->less($one);

                // If violation, grad is -Y. Else 0.
                $dZ = $yMapped->mulScalar(-1.0)->mulInplace($violationMask);

                // Subgradient: dW = C * (X^T * dZ / N) + W
                $dw = $x->transpose()->matmul($dZ)->mulScalarInplace($this->c / $n)->addInplace($this->weights);
                $db = $dZ->mean() * $this->c;

                // Update In-Place
                $dw->mulScalarInplace($this->learningRate);
                $this->weights->subInplace($dw);
                $this->bias -= $db * $this->learningRate;
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("SVC has not been trained.");
        }

        // Output = sign(X * W + b)
        // If > 0, outputs 1.0. If <= 0, outputs 0.0 (converted back from -1, 1 space natively)
        $z = $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias);
        $zero = Tensor::zeros(1);
        
        return $z->greater($zero)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode([
                'class'        => self::class,
                'c'            => $this->c,
                'epochs'       => $this->epochs,
                'learningRate' => $this->learningRate,
                'batchSize'    => $this->batchSize,
                'bias'         => $this->bias,
            ], \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );

        if ($this->weights !== null) {
            SafeTensorsIO::save(
                $dir . \DIRECTORY_SEPARATOR . 'model.safetensors',
                ['weights' => $this->weights]
            );
        }
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("SVC::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
            (float) $config['c'],
            (int)   $config['epochs'],
            (float) $config['learningRate'],
            (int)   $config['batchSize']
        );
        $instance->bias = (float) $config['bias'];

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->weights = $tensors['weights'] ?? null;
        }

        return $instance;
    }
}