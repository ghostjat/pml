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
 * ElasticNet Regression.
 * Combines L1 (Lasso) and L2 (Ridge) Regularization to blend sparsity with stability.
 * * JIT & Memory Optimized:
 * - Resolves compound penalty formulation simultaneously natively in C.
 */
final class ElasticNet implements Learner, Persistable
{
    private float $alpha;
    private float $l1Ratio;
    private int $epochs;
    private float $learningRate;
    private int $batchSize;
    
    private ?Tensor $weights = null;
    private ?Tensor $bias    = null;

    /**
     * @param float $alpha The total penalty multiplier.
     * @param float $l1Ratio The ratio of L1 penalty to L2 penalty (0.0 = Ridge, 1.0 = Lasso).
     */
    public function __construct(float $alpha = 1.0, float $l1Ratio = 0.5, int $epochs = 100, float $learningRate = 0.01, int $batchSize = 32)
    {
        $this->alpha = $alpha;
        $this->l1Ratio = $l1Ratio;
        $this->epochs = $epochs;
        $this->learningRate = $learningRate;
        $this->batchSize = $batchSize;
    }

    public function train(Dataset $dataset): void
    {
        $features      = $dataset->numColumns();
        $this->weights = Tensor::randomNormal([$features, 1], 0.0, 0.01);
        $this->bias    = Tensor::zeros(1);

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();
            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x = $batch->samples();
                $y = $batch->labels();
                $y = $y->ndim() === 1 ? $y->expandDims(1) : $y;
                // Fused BLAS step: predictions, residuals, gradient, L1+L2 penalties, update — 1 C call
                Tensor::lassoSgdStep($x, $y, $this->weights, $this->bias,
                                     $this->alpha, $this->learningRate, $this->l1Ratio);
            }
        }
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("ElasticNet Regression has not been trained.");
        }
        return $dataset->samples()->matmul($this->weights)->addInplace($this->bias)->squeeze();
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
                'l1Ratio'      => $this->l1Ratio,
                'epochs'       => $this->epochs,
                'learningRate' => $this->learningRate,
                'batchSize'    => $this->batchSize,
            ], \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );

        $tensors = [];
        if ($this->weights !== null) $tensors['weights'] = $this->weights;
        if ($this->bias    !== null) $tensors['bias']    = $this->bias;
        if ($tensors) SafeTensorsIO::save($dir . \DIRECTORY_SEPARATOR . 'model.safetensors', $tensors);
    }

    public static function load(string $dir): self
    {
        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("ElasticNet::load — config.json missing in '$dir'.");
        }
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $instance = new self(
            (float) $config['alpha'],
            (float) $config['l1Ratio'],
            (int)   $config['epochs'],
            (float) $config['learningRate'],
            (int)   $config['batchSize']
        );

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->weights = $tensors['weights'] ?? null;
            $instance->bias    = $tensors['bias']    ?? Tensor::zeros(1);
        }

        return $instance;
    }
}