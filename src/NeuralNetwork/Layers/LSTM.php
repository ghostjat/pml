<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Interfaces\Stateful;
use Pml\Tensor;

/**
 * Long Short-Term Memory (LSTM) Layer.
 *
 * Forward:  single C call — no PHP time-step loop (§25).
 * Backward: single fused C BPTT kernel — no per-step FFI crossings, no OOM.
 *
 * In training mode, forward caches gate activations [B,T,4H] and cell
 * states [B,T,H] for the backward pass.  In inference mode, no cache
 * is allocated.
 */
final class LSTM implements Layer, Stateful, HasTrainingMode
{
    private int $hiddenSize;

    private Tensor $W_ih;   /* [D, 4H] */
    private Tensor $W_hh;   /* [H, 4H] */
    private Tensor $b_ih;   /* [4H]    */
    private Tensor $b_hh;   /* [4H]    */

    /* Weight gradients — populated by backward(), consumed by optimizer */
    private ?Tensor $dW_ih = null;
    private ?Tensor $dW_hh = null;
    private ?Tensor $db_ih = null;
    private ?Tensor $db_hh = null;

    /* BPTT cache — allocated during train-mode forward, freed after backward */
    private ?Tensor $cachedInput  = null;  /* [B, T, D] */
    private ?Tensor $cachedOutput = null;  /* [B, T, H] */
    private ?Tensor $cachedActs   = null;  /* [B, T, 4H] gate activations */
    private ?Tensor $cachedCell   = null;  /* [B, T, H]  cell states      */

    private bool $training = false;

    public function __construct(int $inputSize, int $hiddenSize)
    {
        $this->hiddenSize = $hiddenSize;
        $k = 1.0 / sqrt($hiddenSize);

        $this->W_ih = Tensor::randomUniform([$inputSize,    4 * $hiddenSize], -$k, $k);
        $this->W_hh = Tensor::randomUniform([$hiddenSize,   4 * $hiddenSize], -$k, $k);
        $this->b_ih = Tensor::randomUniform([4 * $hiddenSize], -$k, $k);
        $this->b_hh = Tensor::randomUniform([4 * $hiddenSize], -$k, $k);
    }

    public function setTraining(bool $mode): void
    {
        $this->training = $mode;
    }

    public function forward(Tensor $input): Tensor
    {
        if (!$this->training) {
            return Tensor::lstmForward($input, $this->W_ih, $this->W_hh,
                                       $this->b_ih, $this->b_hh);
        }

        /* Training mode: allocate caches and run the cache-writing kernel */
        [$B, $T] = [$input->shape()[0], $input->shape()[1]];
        $H4      = $this->W_ih->shape()[1];   /* 4 * hiddenSize */
        $H       = $this->hiddenSize;

        $this->cachedActs  = new Tensor([$B, $T, $H4]);
        $this->cachedCell  = new Tensor([$B, $T, $H]);
        $this->cachedInput = $input;

        $output = Tensor::lstmForwardTrain(
            $input, $this->W_ih, $this->W_hh, $this->b_ih, $this->b_hh,
            $this->cachedActs, $this->cachedCell
        );

        $this->cachedOutput = $output;
        return $output;
    }

    public function backward(Tensor $dY): Tensor
    {
        if ($this->cachedInput === null || $this->cachedActs === null) {
            throw new \RuntimeException(
                "LSTM::backward() called without a preceding train-mode forward(). " .
                "Call setTraining(true) before forward()."
            );
        }

        /* Single fused C call — full BPTT, zero per-step FFI crossings */
        [$dX, $dWih, $dWhh, $dbIh, $dbHh] = Tensor::lstmBackward(
            $dY,
            $this->cachedInput,
            $this->cachedOutput,
            $this->W_ih, $this->W_hh,
            $this->cachedActs, $this->cachedCell
        );

        $this->dW_ih = $dWih;
        $this->dW_hh = $dWhh;
        $this->db_ih = $dbIh;
        $this->db_hh = $dbHh;

        /* Release caches immediately — no longer needed */
        $this->cachedInput  = null;
        $this->cachedOutput = null;
        $this->cachedActs   = null;
        $this->cachedCell   = null;

        return $dX;
    }

    public function getParameters(): array
    {
        return [
            'W_ih' => $this->W_ih, 'W_hh' => $this->W_hh,
            'b_ih' => $this->b_ih, 'b_hh' => $this->b_hh,
        ];
    }

    public function getGradients(): array
    {
        if ($this->dW_ih === null) {
            return [];
        }
        return [
            'W_ih' => $this->dW_ih, 'W_hh' => $this->dW_hh,
            'b_ih' => $this->db_ih, 'b_hh' => $this->db_hh,
        ];
    }

    public function getConfig(): array
    {
        return [
            'inputSize'  => $this->W_ih->shape()[0],
            'hiddenSize' => $this->hiddenSize,
        ];
    }

    public static function fromConfig(array $config): static
    {
        return new static((int) $config['inputSize'], (int) $config['hiddenSize']);
    }

    public function getStateDict(string $prefix = ''): array
    {
        return [
            $prefix . 'W_ih' => $this->W_ih,
            $prefix . 'W_hh' => $this->W_hh,
            $prefix . 'b_ih' => $this->b_ih,
            $prefix . 'b_hh' => $this->b_hh,
        ];
    }

    public function loadStateDict(array $dict, string $prefix = ''): void
    {
        $this->W_ih = $dict[$prefix . 'W_ih'];
        $this->W_hh = $dict[$prefix . 'W_hh'];
        $this->b_ih = $dict[$prefix . 'b_ih'];
        $this->b_hh = $dict[$prefix . 'b_hh'];
    }
}
