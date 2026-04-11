<?php
declare(strict_types=1);

namespace Pml\Strategies;

use Pml\Tensor;

/**
 * Imputes by sampling uniformly from the K most frequent integer values.
 * Uses C-level bincount to find frequencies, PHP array sort for top-K.
 */
final class KMostFrequent implements Strategy
{
    /** @var int[] */
    private array $topK = [];

    public function __construct(private readonly int $k = 1) {}

    public function fit(Tensor $values): void
    {
        // bincount requires non-negative integer data
        $counts = $values->abs()->round()->bincount()->toFlatArray();
        arsort($counts);
        $this->topK = array_keys(array_slice($counts, 0, $this->k, true));
    }

    public function guess(): float
    {
        if (empty($this->topK)) {
            return 0.0;
        }
        return (float) $this->topK[array_rand($this->topK)];
    }
}
