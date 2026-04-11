<?php
declare(strict_types=1);

namespace Pml\Helpers;

/**
 * Hyperparameter grid generation for GridSearch / RandomSearch.
 * All grid expansion is pure PHP (no tensors needed — param values are scalars).
 */
final class Params
{
    /**
     * Generate a linearly spaced grid of float values.
     *
     * @return float[]
     */
    public static function floats(float $min, float $max, int $n): array
    {
        if ($n < 2) {
            throw new \InvalidArgumentException("Grid must contain at least 2 values.");
        }
        $step   = ($max - $min) / ($n - 1);
        $values = [];
        for ($i = 0; $i < $n; $i++) {
            $values[] = $min + $i * $step;
        }
        return $values;
    }

    /**
     * Generate a logarithmically spaced grid (useful for learning rates, alpha).
     *
     * @return float[]
     */
    public static function logspace(float $min, float $max, int $n): array
    {
        if ($n < 2) {
            throw new \InvalidArgumentException("Grid must contain at least 2 values.");
        }
        $logMin = log10($min);
        $logMax = log10($max);
        $step   = ($logMax - $logMin) / ($n - 1);
        $values = [];
        for ($i = 0; $i < $n; $i++) {
            $values[] = 10 ** ($logMin + $i * $step);
        }
        return $values;
    }

    /**
     * Generate an integer grid.
     *
     * @return int[]
     */
    public static function ints(int $min, int $max, int $n): array
    {
        $floats = self::floats((float) $min, (float) $max, $n);
        return array_map('intval', $floats);
    }

    /**
     * Cartesian product of all parameter grids — feeds into GridSearch.
     *
     * @param array<string, array> $grid
     * @return array<array<string, mixed>>
     */
    public static function grid(array $grid): array
    {
        $combinations = [[]];
        foreach ($grid as $param => $values) {
            $expanded = [];
            foreach ($combinations as $combination) {
                foreach ($values as $value) {
                    $expanded[] = $combination + [$param => $value];
                }
            }
            $combinations = $expanded;
        }
        return $combinations;
    }
}
