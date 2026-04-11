<?php
declare(strict_types=1);

namespace Pml\Transformers;

use Pml\Interfaces\Transformer;
use Pml\Tensor;
use Pml\Dataset;

/**
 * Lambda Function Transformer — applies a user-supplied closure to the sample Tensor.
 * Enables arbitrary C-level operations inline in a Pipeline without subclassing.
 *
 * JIT & Memory Optimized: the closure should return a Tensor to avoid PHP↔C copies.
 *
 * Example:
 *   new LambdaFunction(fn(Tensor $x) => $x->log1p())
 */
final class LambdaFunction implements Transformer
{
    private bool $fitted = false;
    private ?\Closure $fitFn = null;

    /**
     * @param \Closure $transformFn  fn(Tensor $x): Tensor
     * @param \Closure|null $fitFn   fn(Dataset $d): void  (optional stateful fitting)
     */
    public function __construct(
        private readonly \Closure $transformFn,
        ?\Closure $fitFn = null
    ) {
        $this->fitFn = $fitFn;
    }

    public function fit(Dataset $dataset): void
    {
        if ($this->fitFn !== null) {
            ($this->fitFn)($dataset);
        }
        $this->fitted = true;
    }

    public function transform(Dataset $dataset): Dataset
    {
        $result = ($this->transformFn)($dataset->samples());
        return new Dataset($result, $dataset->labels());
    }

    public function fitted(): bool { return $this->fitted; }
}
