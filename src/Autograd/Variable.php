<?php
declare(strict_types=1);

namespace Pml\Autograd;

use Pml\Tensor;

/**
 * PHP wrapper around a C VarNode* on a Tape.
 *
 * All variables belong to the tape that created them.  VarNodes are
 * pool-allocated inside the tape's fixed-capacity arrays, so this PHP object
 * holds a stable pointer — no GC finaliser needed (the tape owns memory).
 *
 * Fluent API:
 *   $z = $a->mul($b)->add($c)->relu();
 *   $tape->backward($z);
 *   $gradA = $a->grad();        // returns Pml\Tensor or null
 *   $fwd   = $a->data();        // returns Pml\Tensor
 */
final class Variable
{
    /** @var \FFI\CData  VarNode* */
    private \FFI\CData $node;

    private Tape $tape;

    private function __construct(Tape $tape, \FFI\CData $node)
    {
        $this->tape = $tape;
        $this->node = $node;
    }

    // ── Factory ───────────────────────────────────────────────────────────────

    /**
     * Wrap an existing Pml\Tensor as a leaf variable.
     * The tape does NOT take ownership of $tensor's data.
     */
    public static function leaf(Tape $tape, Tensor $tensor, bool $requiresGrad = false): self
    {
        $ffi  = $tape->ffi();
        $node = $ffi->ag_var($tape->ptr(), $tensor->ptr, $requiresGrad);
        if (\FFI::isNull($node)) {
            self::throwLastError($ffi, 'ag_var');
        }
        return new self($tape, $node);
    }

    // ── Differentiable ops ────────────────────────────────────────────────────

    /** Element-wise add: z = $this + $other */
    public function add(Variable $other): self
    {
        $ffi  = $this->tape->ffi();
        $node = $ffi->ag_add($this->tape->ptr(), $this->node, $other->node);
        if (\FFI::isNull($node)) {
            self::throwLastError($ffi, 'ag_add');
        }
        return new self($this->tape, $node);
    }

    /** Element-wise multiply: z = $this ⊙ $other */
    public function mul(Variable $other): self
    {
        $ffi  = $this->tape->ffi();
        $node = $ffi->ag_mul($this->tape->ptr(), $this->node, $other->node);
        if (\FFI::isNull($node)) {
            self::throwLastError($ffi, 'ag_mul');
        }
        return new self($this->tape, $node);
    }

    /** Matrix multiply: Z = $this @ $other  ([M,K] @ [K,N]) */
    public function matmul(Variable $other): self
    {
        $ffi  = $this->tape->ffi();
        $node = $ffi->ag_matmul($this->tape->ptr(), $this->node, $other->node);
        if (\FFI::isNull($node)) {
            self::throwLastError($ffi, 'ag_matmul');
        }
        return new self($this->tape, $node);
    }

    /** ReLU: z = max(0, $this) */
    public function relu(): self
    {
        $ffi  = $this->tape->ffi();
        $node = $ffi->ag_relu($this->tape->ptr(), $this->node);
        if (\FFI::isNull($node)) {
            self::throwLastError($ffi, 'ag_relu');
        }
        return new self($this->tape, $node);
    }

    /**
     * Shorthand: run tape->backward($this).
     * Equivalent to $tape->backward($var) — kept for method-chain ergonomics.
     */
    public function backward(): void
    {
        $this->tape->backward($this);
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /**
     * Forward value as a Pml\Tensor (zero-copy view of the C tensor).
     */
    public function data(): Tensor
    {
        return Tensor::wrap($this->node->data);
    }

    /**
     * Accumulated gradient as a Pml\Tensor, or null if backward hasn't run
     * or this variable does not require grad.
     */
    public function grad(): ?Tensor
    {
        if (!$this->node->requires_grad) {
            return null;
        }
        if (\FFI::isNull($this->node->grad)) {
            return null;
        }
        return Tensor::wrap($this->node->grad);
    }

    public function requiresGrad(): bool
    {
        return (bool)$this->node->requires_grad;
    }

    /** @internal used by Tape and other Variables */
    public function cdata(): \FFI\CData
    {
        return $this->node;
    }

    // ── Internal helpers ──────────────────────────────────────────────────────

    private static function throwLastError(\FFI $ffi, string $op): never
    {
        $msg = $ffi->tensor_check_error()
            ? \FFI::string($ffi->tensor_get_last_error())
            : 'unknown error';
        $ffi->tensor_clear_error();
        throw new \RuntimeException("[Variable::{$op}] {$msg}");
    }
}
