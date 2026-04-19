<?php

declare(strict_types=1);

namespace Pml\Estimators\Regression;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Dummy Regressor.
 * Acts as a baseline sanity-check by always predicting the continuous Mean.
 */
final class DummyRegressor implements Learner, Persistable
{
    private ?float $mean = null;

    public function train(Dataset $dataset): void
    {
        // Extract the continuous mean natively in C
        $this->mean = $dataset->labels()->mean();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("DummyRegressor is not trained.");
        }

        // Return a tensor filled with the mean
        return Tensor::zeros($dataset->numRows())->addScalarInplace($this->mean);
    }

    public function trained(): bool { return $this->mean !== null; }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['mean' => $this->mean]));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self(); $i->mean = (float) $c['mean']; return $i;
    }
}