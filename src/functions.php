<?php
declare(strict_types=1);

namespace Pml;

/**
 * Compute the softmax of a float array — pure PHP, for scalar use cases.
 * For Tensor-level softmax use Tensor::exp() / Tensor::sumAxis().
 *
 * @param  float[] $logits
 * @return float[]
 */
function softmax(array $logits): array
{
    $max    = max($logits);
    $exps   = array_map(fn($x) => exp($x - $max), $logits);
    $sum    = array_sum($exps);
    return array_map(fn($e) => $e / $sum, $exps);
}

/**
 * Compute log-sum-exp stably: log( sum( exp(x_i) ) ).
 *
 * @param float[] $values
 */
function logSumExp(array $values): float
{
    $max = max($values);
    return $max + log(array_sum(array_map(fn($x) => exp($x - $max), $values)));
}

/**
 * Clamp a float value into [min, max].
 */
function clamp(float $value, float $min, float $max): float
{
    return max($min, min($max, $value));
}

/**
 * Sigmoid function: 1 / (1 + exp(-x)).
 */
function sigmoid(float $x): float
{
    return 1.0 / (1.0 + exp(-$x));
}

/**
 * Return the argmax index of a float array.
 *
 * @param float[] $values
 */
function argmax(array $values): int
{
    $maxIdx = 0;
    $maxVal = -INF;
    foreach ($values as $i => $v) {
        if ($v > $maxVal) { $maxVal = $v; $maxIdx = $i; }
    }
    return $maxIdx;
}

/**
 * One-hot encode an integer class index into a float array.
 *
 * @return float[]
 */
function oneHot(int $class, int $numClasses): array
{
    $vec         = array_fill(0, $numClasses, 0.0);
    $vec[$class] = 1.0;
    return $vec;
}

/**
 * Compute entropy of a probability distribution: -sum(p * log(p)).
 *
 * @param float[] $probs
 */
function entropy(array $probs): float
{
    $h = 0.0;
    foreach ($probs as $p) {
        if ($p > 0.0) $h -= $p * log($p);
    }
    return $h;
}

/**
 * Convert a nested PHP array of floats into a 2-D Tensor.
 * Alias for Tensor::fromArray() for ergonomic pipeline construction.
 *
 * @param float[][] $data
 */
function tensor(array $data): Tensor
{
    return Tensor::fromArray($data);
}
