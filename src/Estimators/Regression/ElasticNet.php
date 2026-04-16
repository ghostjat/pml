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
    private float $bias = 0.0;

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
        $features = $dataset->numColumns();
        $this->weights = Tensor::randomNormal([$features, 1], 0.0, 0.01);
        $this->bias = 0.0;

        for ($e = 0; $e < $this->epochs; $e++) {
            $dataset->randomize();

            foreach ($dataset->batches($this->batchSize) as $batch) {
                $x = $batch->samples();
                $y = $batch->labels();
                $y = $y->ndim() === 1 ? $y->expandDims(1) : $y;
                $n = (float) $x->shape()[0];

                // Y_pred = X * W + b
                $predictions = $x->matmul($this->weights)->addScalarInplace($this->bias);

                // dZ = Y_pred - Y
                $dz = $predictions->sub($y);
                
                // dW = (X^T * dZ) / N
                $dw = $x->transpose()->matmul($dz)->mulScalarInplace(1.0 / $n);
                
                // Compound Penalty: L1 + L2
                $l1Penalty = $this->weights->sign()->mulScalarInplace($this->alpha * $this->l1Ratio);
                $l2Penalty = $this->weights->mulScalar($this->alpha * (1.0 - $this->l1Ratio));
                
                $dw->addInplace($l1Penalty)->addInplace($l2Penalty);
                
                $db = $dz->mean();

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
            throw new RuntimeException("ElasticNet Regression has not been trained.");
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
                'l1Ratio'      => $this->l1Ratio,
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
        $instance->bias = (float) $config['bias'];

        $stPath = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->weights = $tensors['weights'] ?? null;
        }

        return $instance;
    }
}