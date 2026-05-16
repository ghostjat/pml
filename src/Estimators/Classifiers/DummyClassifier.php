<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;

final class DummyClassifier implements Learner, Persistable
{
    private ?float $mode = null;

    public function train(Dataset $dataset, mixed ...$options): void
    {
        // Extract the most frequent class natively in C using bincount -> argmax
        $this->mode = (float) $dataset->labels()->bincount()->argmax();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("DummyClassifier is not trained.");
        }

        // Return a tensor filled with the mode
        return Tensor::zeros($dataset->numRows())->addScalarInplace($this->mode);
    }

    public function trained(): bool { return $this->mode !== null; }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['class' => self::class, 'mode' => $this->mode]));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self(); $i->mode = (float) $c['mode']; return $i;
    }
}