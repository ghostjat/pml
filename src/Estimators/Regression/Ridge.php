<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Ridge Regression (Linear Regression with L2 Regularization).
 * Prevents coefficient explosion by shrinking weights toward zero.
 * * JIT & Memory Optimized:
 * - Executes Gradient Descent rapidly using hardware BLAS.
 */
final class Ridge implements Learner, Persistable
{
    private float $alpha;
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private float $bias = 0.0;

    /**
     * @param float $alpha The L2 penalty multiplier. Larger values shrink weights more aggressively.
     */
    public function __construct(float $alpha = 1.0, int $epochs = 100, float $learningRate = 0.01, int $batchSize = 32)
    {
        $this->alpha = $alpha;
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
        $this->batchSize = $batchSize;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $y = $dataset->labels();
        if ($y === null) {
            throw new \InvalidArgumentException("Ridge requires labeled data.");
        }

        // Augment X with a bias column of ones so closed-form solves bias jointly.
        // X_aug [N, D+1]; this keeps $this->bias = 0 (absorbed into last weight).
        $ones  = Tensor::ones($x->shape()[0])->expandDims(1);          // [N,1]
        $xAug  = Tensor::concat([$x, $ones], 1);                        // [N, D+1]

        // Closed-form: W_aug = (X_aug^T X_aug + λI)^{-1} X_aug^T y — single LAPACKE call
        $wAug         = Tensor::ridgeSolve($xAug, $y, $this->alpha);    // [D+1, 1]
        $d            = $x->shape()[1];
        $this->weights = $wAug->slice(0, 0, $d);                        // [D, 1] feature weights
        $this->bias    = $wAug->slice(0, $d, 1)->toFlatArray()[0];      // scalar bias
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("Ridge Regression has not been trained.");
        }
        return $dataset->samples()->matmul($this->weights)->addScalarInplace($this->bias)->squeeze();
    }

    public function trained(): bool
    {
        return $this->weights !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) { mkdir($dir, 0755, true); }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode([
                'class'        => self::class,
                'alpha'        => $this->alpha,
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
            throw new \RuntimeException("Ridge::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
            (float) $config['alpha'],
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