<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Probabilistic;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Binary Logistic Regression.
 * Solves the classification problem using Stochastic Gradient Descent.
 * * JIT & Memory Optimized:
 * - 100% vector-matrix arithmetic via OpenBLAS.
 * - Memory-flat training loop updates weights purely In-Place.
 */
final class LogisticRegression implements Learner, Probabilistic, Persistable
{
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private float $bias = 0.0;

    public function __construct(int $epochs = 100, float $learningRate = 0.01, int $batchSize = 32)
    {
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
        $this->batchSize = $batchSize;
    }

    public function train(Dataset $dataset): void
    {
        $features = $dataset->numColumns();
        
        // Initialize weights to tiny random values [D, 1]
        $this->weights = Tensor::randomNormal([$features, 1], 0.0, 0.01);
        $this->bias = 0.0;

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x = $batch->samples();
                
                // Y must be shape [N, 1]
                $y = $batch->labels();
                $y = $y->ndim() === 1 ? $y->expandDims(1) : $y;
                $n = (float) $x->shape()[0];

                // 1. Forward Pass: Z = X * W + b
                $z = $x->matmul($this->weights)->addScalarInplace($this->bias);
                $predictions = $z->sigmoid();

                // 2. Compute Gradients: dZ = A - Y
                $dz = $predictions->sub($y);
                
                // dW = (X^T * dZ) / N
                $dw = $x->transpose()->matmul($dz)->mulScalarInplace(1.0 / $n);
                $db = $dz->mean();

                // 3. Update Weights In-Place: W -= dW * lr
                $dw->mulScalarInplace($this->learningRate);
                $this->weights->subInplace($dw);
                $this->bias -= $db * $this->learningRate;
            }
        }
    }

    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Logistic Regression has not been trained.");
        }

        // Returns continuous probabilities [0.0 - 1.0]
        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->sigmoid()->squeeze();
    }

    public function predict(Dataset $dataset): Tensor
    {
        // Round probabilities to strict 0.0 or 1.0 classes natively in C
        return $this->proba($dataset)->round();
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
            throw new \RuntimeException("LogisticRegression::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
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