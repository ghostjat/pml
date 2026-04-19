<?php

declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Radius Neighbors Classifier.
 * Classifies a sample based on a majority vote of all neighbors within a given physical radius, 
 * rather than a strict count of K neighbors.
 * * JIT & Memory Optimized:
 * - Uses AVX2 Boolean Masking (`less()`) to extract points within the radius natively in C.
 */
final class RadiusNeighborsClassifier implements Learner, Persistable
{
    private float $radius;
    private ?Tensor $fitSamples = null;
    private ?Tensor $fitLabels = null;

    public function __construct(float $radius = 1.0)
    {
        $this->radius = $radius;
    }

    public function train(Dataset $dataset): void
    {
        $this->fitSamples = $dataset->samples();
        $this->fitLabels = $dataset->labels();
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("RadiusNeighbors is not trained.");

        $testX = $dataset->samples();
        $nTest = $testX->shape()[0];
        $preds = [];
        
        $radiusSq = Tensor::zeros(1)->addScalarInplace($this->radius * $this->radius);

        for ($i = 0; $i < $nTest; $i++) {
            $x = $testX->row($i);
            
            // Broadcast squared Euclidean distance
            $sqDist = $this->fitSamples->sub($x)->square()->sumAxis(1);
            
            // Mask points STRICTLY within the radius natively in C
            $inRadiusMask = $sqDist->less($radiusSq);
            $neighbors = $this->fitLabels->booleanIndex($inRadiusMask);
            
            if ($neighbors->size() === 0) {
                // Outlier fallback: Predict 0 or handle globally
                $preds[] = 0.0;
                continue;
            }
            
            // Majority vote natively in C
            $preds[] = $neighbors->bincount()->argmax();
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->fitSamples !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['radius' => $this->radius]));
        if ($this->fitSamples !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['fit_samples' => $this->fitSamples, 'fit_labels' => $this->fitLabels]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((float)$c['radius']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->fitSamples = $t['fit_samples'] ?? null; $i->fitLabels = $t['fit_labels'] ?? null; }
        return $i;
    }
}
