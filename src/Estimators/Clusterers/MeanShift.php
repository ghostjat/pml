<?php

declare(strict_types=1);

namespace Pml\Estimators\Clusterers;

use Pml\Interfaces\Learner;
use Pml\Lib\SafeTensorsIO;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Mean Shift Clusterer.
 * A sliding-window density algorithm that shifts points towards the mode (highest density) 
 * of the data distribution. Does not require setting K clusters beforehand!
 * * JIT & Memory Optimized:
 * - Operates fully via C-Level masking and vector accumulation.
 */
final class MeanShift implements Learner, Persistable
{
    private float $bandwidth;
    private int $maxIter;
    private float $tolerance;
    
    private ?Tensor $centroids = null;

    public function __construct(float $bandwidth = 1.0, int $maxIter = 100, float $tolerance = 1e-4)
    {
        $this->bandwidth = $bandwidth;
        $this->maxIter = $maxIter;
        $this->tolerance = $tolerance;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $n = $x->shape()[0];
        
        // 1. Initialize centroids at every single data point
        $centroids = $x->copy();
        $bwSq = Tensor::zeros(1)->addScalarInplace($this->bandwidth * $this->bandwidth);

        // 2. Iteratively shift centroids to the mean of their local neighborhood
        for ($iter = 0; $iter < $this->maxIter; $iter++) {
            $maxShift = 0.0;
            $newCentroids = [];

            for ($i = 0; $i < $n; $i++) {
                $c = $centroids->row($i);
                
                // Find all points within bandwidth
                $distSq = $x->sub($c)->square()->sumAxis(1);
                $mask = $distSq->less($bwSq);
                
                $neighbors = $x->booleanIndex($mask);
                
                // Shift centroid to the mean of these neighbors
                $newC = $neighbors->meanAxis(0);
                $newCentroids[] = $newC;
                
                // Track convergence
                $shift = $c->sub($newC)->abs()->max();
                if ($shift > $maxShift) $maxShift = $shift;
            }
            
            $centroids = Tensor::concat($newCentroids, 0);
            if ($maxShift < $this->tolerance) break;
        }

        // 3. Merge identical/overlapping centroids natively
        $unique = [];
        $mergedMap = [];
        
        for ($i = 0; $i < $n; $i++) {
            if (isset($mergedMap[$i])) continue;
            
            $c1 = $centroids->row($i);
            $unique[] = $c1;
            
            for ($j = $i + 1; $j < $n; $j++) {
                if (isset($mergedMap[$j])) continue;
                $c2 = $centroids->row($j);
                
                if ($c1->sub($c2)->abs()->max() < $this->tolerance) {
                    $mergedMap[$j] = true;
                }
            }
        }

        $this->centroids = Tensor::concat($unique, 0);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("MeanShift is not trained.");

        $testX = $dataset->samples();
        $nTest = $testX->shape()[0];
        $preds = [];

        // Assign to nearest converged centroid
        for ($i = 0; $i < $nTest; $i++) {
            $x = $testX->row($i);
            $preds[] = $this->centroids->sub($x)->square()->sumAxis(1)->argmin();
        }

        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return $this->centroids !== null;
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        file_put_contents($dir . '/config.json', json_encode(['bandwidth'=>$this->bandwidth,'maxIter'=>$this->maxIter,'tolerance'=>$this->tolerance]));
        if ($this->centroids !== null) SafeTensorsIO::save($dir . '/model.safetensors', ['centroids' => $this->centroids]);
    }
    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        $i = new self((float)$c['bandwidth'], (int)$c['maxIter'], (float)$c['tolerance']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) { $t = SafeTensorsIO::load($stPath); $i->centroids = $t['centroids'] ?? null; }
        return $i;
    }
}
