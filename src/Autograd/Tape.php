<?php
declare(strict_types=1);

namespace Pml\Autograd;

use Pml\Lib\TensorEngine;

/**
 * PHP wrapper around the C Tape* computation graph.
 *
 * Usage:
 *   $tape = new Tape();
 *   $a = Variable::leaf($tape, $tensorA, requiresGrad: true);
 *   $b = Variable::leaf($tape, $tensorB, requiresGrad: true);
 *   $z = $a->mul($b)->add($c)->relu();
 *   $tape->backward($z);
 *   $gradA = $a->grad();   // Pml\Tensor
 *
 * The tape is destroyed (C memory freed) when this object is GC'd.
 * Call reset() to reuse the same tape for a new forward pass.
 */
final class Tape
{
    /** @var \FFI\CData  Tape* */
    private \FFI\CData $ptr;

    private \FFI $ffi;

    /**
     * @param int $opsCap  max OpNodes  (0 = library default 4096)
     * @param int $varsCap max VarNodes (0 = library default 8192)
     */
    public function __construct(int $opsCap = 0, int $varsCap = 0)
    {
        $this->ffi = TensorEngine::get();
        $tape = $this->ffi->tape_create($opsCap, $varsCap);
        if (\FFI::isNull($tape)) {
            throw new \RuntimeException('[Tape] tape_create() returned NULL — out of memory.');
        }
        $this->ptr = $tape;
    }

    public function __destruct()
    {
        $this->ffi->tape_destroy($this->ptr);
    }

    /**
     * Free grad tensors and owned data tensors; reset op/var counts.
     * After reset() the tape can be reused for a new forward pass.
     */
    public function reset(): void
    {
        $this->ffi->tape_reset($this->ptr);
    }

    /**
     * Zero all allocated grad buffers without freeing them.
     * Cheaper than reset() when re-running backward on the same graph.
     */
    public function clearGrads(): void
    {
        $this->ffi->tape_clear_grads($this->ptr);
    }

    /**
     * Run reverse-mode autodiff from $root.
     * After this call every requires_grad Variable's grad() is populated.
     */
    public function backward(Variable $root): void
    {
        $this->ffi->tape_backward($this->ptr, $root->cdata());
        $this->checkError();
    }

    /** @internal used by Variable */
    public function ffi(): \FFI
    {
        return $this->ffi;
    }

    /** @internal used by Variable */
    public function ptr(): \FFI\CData
    {
        return $this->ptr;
    }

    // ── helpers ──────────────────────────────────────────────────────────────

    private function checkError(): void
    {
        if ($this->ffi->tensor_check_error()) {
            $msg = \FFI::string($this->ffi->tensor_get_last_error());
            $this->ffi->tensor_clear_error();
            throw new \RuntimeException('[Tape] ' . $msg);
        }
    }
}
