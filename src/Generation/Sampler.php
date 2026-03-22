<?php
declare(strict_types=1);

namespace Pml\Generation;

use Pml\Tensor;

class Sampler
{
    /**
     * @param Tensor $logits 1D Tensor of vocabulary scores
     * @param float $temperature 0.0 = greedy (argmax), > 0.0 = creative
     */
    public static function sample(Tensor $logits, float $temperature = 0.7): int
    {
        $vocabSize = $logits->size;
        $buf = $logits->buffer;

        // 1. Greedy Decoding (Temperature = 0)
        if ($temperature <= 0.0) {
            $maxIndex = 0;
            $maxVal = $buf[0];
            for ($i = 1; $i < $vocabSize; $i++) {
                if ($buf[$i] > $maxVal) {
                    $maxVal = $buf[$i];
                    $maxIndex = $i;
                }
            }
            return $maxIndex;
        }

        // 2. Temperature Scaling & Softmax
        $maxVal = -INF;
        for ($i = 0; $i < $vocabSize; $i++) {
            $buf[$i] /= $temperature;
            if ($buf[$i] > $maxVal) $maxVal = $buf[$i]; // for numerical stability
        }

        $sum = 0.0;
        for ($i = 0; $i < $vocabSize; $i++) {
            $val = exp($buf[$i] - $maxVal);
            $buf[$i] = $val;
            $sum += $val;
        }

        // 3. Cumulative Distribution Function (CDF) Sampling
        // Generate a random float between 0 and 1
        $r = (mt_rand() / mt_getrandmax()) * $sum;
        
        $cumulative = 0.0;
        for ($i = 0; $i < $vocabSize; $i++) {
            $cumulative += $buf[$i];
            if ($r <= $cumulative) {
                return $i;
            }
        }

        return $vocabSize - 1; // Fallback
    }
}