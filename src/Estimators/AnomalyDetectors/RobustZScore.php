<?php

declare(strict_types=1);

namespace Pml\Estimators\AnomalyDetectors;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Lib\SafeTensorsIO;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * Robust Z-Score.
 * Detects anomalies by measuring standard deviations from the median (using Median Absolute Deviation).
 * Much less sensitive to extreme outliers than standard Gaussian MLE.
 */
final class RobustZScore implements Learner, Persistable
{
    private float $threshold;
    
    private ?Tensor $medians = null;
    private ?Tensor $mads = null;

    public function __construct(float $threshold = 3.5)
    {
        $this->threshold = $threshold;
    }

    public function train(Dataset $dataset): void
    {
        $x = $dataset->samples();
        $cols = $x->shape()[1];
        
        $medians = [];
        $mads = [];

        // Median and MAD are resilient to outliers but require sorting.
        // We use fast PHP 1D array sorts.
        for ($i = 0; $i < $cols; $i++) {
            $colData = $x->col($i)->toFlatArray();
            sort($colData);
            
            $median = $colData[(int) floor(count($colData) / 2)];
            $medians[] = $median;
            
            $absDeviations = array_map(fn($v) => abs($v - $median), $colData);
            sort($absDeviations);
            
            $mad = $absDeviations[(int) floor(count($absDeviations) / 2)];
            // Scale MAD to approximate Standard Deviation (1.4826 multiplier)
            $mads[] = $mad * 1.4826;
        }

        $this->medians = Tensor::fromArray($medians);
        $this->mads = Tensor::fromArray($mads)->clip(1e-8, INF);
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) throw new RuntimeException("RobustZScore is not trained.");

        $x = $dataset->samples();
        
        // Z-Score = |X - Median| / MAD
        // Evaluated simultaneously for all features in C-Memory
        $zScores = $x->sub($this->medians)->abs()->divInplace($this->mads);
        
        // An anomaly is flagged if ANY feature's Z-Score exceeds the threshold
        $thresholdT = Tensor::zeros(1)->addScalarInplace($this->threshold);
        $anomaliesMask = $zScores->greater($thresholdT)->maxAxis(1);

        return $anomaliesMask;
    }

    public function trained(): bool
    {
        return $this->medians !== null;
    }

    public function save(string $dir): void
    {
        if (!is_dir($dir)) { mkdir($dir, 0755, true); }
        file_put_contents($dir . '/config.json', json_encode(['threshold' => $this->threshold], JSON_PRETTY_PRINT));
        if ($this->medians !== null && $this->mads !== null) {
            SafeTensorsIO::save($dir . '/model.safetensors', ['medians' => $this->medians, 'mads' => $this->mads]);
        }
    }

    public static function load(string $dir): self
    {
        $cfg = json_decode(file_get_contents($dir . '/config.json'), true);
        $instance = new self((float) $cfg['threshold']);
        $stPath = $dir . '/model.safetensors';
        if (is_file($stPath)) {
            $tensors = SafeTensorsIO::load($stPath);
            $instance->medians = $tensors['medians'] ?? null;
            $instance->mads = $tensors['mads'] ?? null;
        }
        return $instance;
    }
}