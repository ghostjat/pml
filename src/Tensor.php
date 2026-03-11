<?phph
declare(strict_types=1);
namespace Pml;
// ═══════════════════════════════════════════════════════════════════════════
//  TENSOR INTERFACE
// ═══════════════════════════════════════════════════════════════════════════

interface Tensor extends \Countable, \IteratorAggregate, \JsonSerializable
{
    // dtype constants (match RubixML)
    public const FLOAT = 1;
    public const INT   = 2;

    public function shape(): array;
    public function shapeString(): string;
    public function isSquare(): bool;
    public function size(): int;
    public function ndim(): int;
    public function dtype(): int;
    public function asArray(): array;
    public function flatten(): Vector;

    // reductions
    public function sum(): int|float;
    public function product(): int|float;
    public function min(): int|float;
    public function max(): int|float;
    public function mean(): float;
    public function median(): float;
    public function variance(?Tensor $mean = null): float;

    // element-wise ops (return same type)
    public function round(int $precision = 0): static;
    public function floor(): static;
    public function ceil(): static;
    public function clip(float $min, float $max): static;
    public function clipLower(float $min): static;
    public function clipUpper(float $max): static;
    public function abs(): static;
    public function square(): static;
    public function sqrt(): static;
    public function exp(): static;
    public function expm1(): static;
    public function log(float $base = M_E): static;
    public function log1p(): static;
    public function sin(): static;
    public function cos(): static;
    public function tan(): static;
    public function asin(): static;
    public function acos(): static;
    public function atan(): static;
    public function sinh(): static;
    public function cosh(): static;
    public function tanh(): static;
    public function negate(): static;
    public function reciprocal(): static;

    // arithmetic (scalar or Tensor argument)
    public function add(int|float|Tensor $b): static;
    public function subtract(int|float|Tensor $b): static;
    public function multiply(int|float|Tensor $b): static;
    public function divide(int|float|Tensor $b): static;
    public function pow(int|float|Tensor $b): static;
    public function mod(int|float|Tensor $b): static;

    // comparisons (return same shape, 1.0 / 0.0)
    public function equal(int|float|Tensor $b): static;
    public function notEqual(int|float|Tensor $b): static;
    public function greater(int|float|Tensor $b): static;
    public function greaterEqual(int|float|Tensor $b): static;
    public function less(int|float|Tensor $b): static;
    public function lessEqual(int|float|Tensor $b): static;

    // functional
    public function map(callable $fn): static;
    public function reduce(callable $fn, int|float $initial = 0): int|float;
}
