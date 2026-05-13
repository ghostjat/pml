<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

use Pml\Tensor;
use Pml\Interfaces\Stateful;
use Pml\Interfaces\Verbose;
use Psr\Log\LoggerInterface;

/**
 * Multi-Modal Hardware-Accelerated Mamba (Selective State Space Model) Layer.
 * Supports dynamic sequence lengths for Text (GPT-style), Audio, and Vision tasks.
 * Strictly uses high-level Tensor.php APIs for Zero-Copy memory execution.
 */
final class Mamba implements Layer, Stateful, HasTrainingMode, Verbose {

    private int $dModel;
    private int $dState;
    private bool $isTraining = true;
    private ?LoggerInterface $logger = null;
    // --- Trainable Parameters ---
    private Tensor $A_log;
    private Tensor $D_skip;
    private Tensor $W_B;
    private Tensor $b_B;
    private Tensor $W_C;
    private Tensor $b_C;
    private Tensor $W_delta;
    private Tensor $b_delta;
    // --- Gradients ---
    private Tensor $dA_log;
    private Tensor $dD_skip;
    private Tensor $dW_B;
    private Tensor $db_B;
    private Tensor $dW_C;
    private Tensor $db_C;
    private Tensor $dW_delta;
    private Tensor $db_delta;
    // --- Cached Tensors for Backprop (Zero-Copy References) ---
    private ?Tensor $cacheX = null;
    private ?Tensor $cacheB = null;
    private ?Tensor $cacheC = null;
    private ?Tensor $cacheDelta = null;
    private ?Tensor $h0 = null;
    private ?Tensor $mambaCache = null;

    /**
     * @param int $dModel Hidden dimension size (D).
     * @param int $dState State dimension size (N, usually 16).
     */
    public function __construct(int $dModel, int $dState = 16) {
        $this->dModel = $dModel;
        $this->dState = $dState;

        // Allocate trainable parameters using Tensor object factory
        $this->A_log = new Tensor([$dModel, $dState]);
        $this->D_skip = new Tensor([$dModel]);

        $this->W_B = new Tensor([$dState, $dModel]);
        $this->b_B = new Tensor([$dState]);

        $this->W_C = new Tensor([$dState, $dModel]);
        $this->b_C = new Tensor([$dState]);

        $this->W_delta = new Tensor([$dModel, $dModel]);
        $this->b_delta = new Tensor([$dModel]);

        // Pre-allocate gradient accumulators
        $this->dA_log = new Tensor([$dModel, $dState]);
        $this->dD_skip = new Tensor([$dModel]);

        $this->initializeWeights();
    }

    private function initializeWeights(): void {
        // A_log must be <= 0 for SSM stability
        $this->A_log->fill(-1.0);
        $this->D_skip->fill(1.0);

        $stddev = sqrt(2.0 / $this->dModel);
        $this->W_B     = Tensor::randomNormal([$this->dState, $this->dModel], 0.0, $stddev);
        $this->W_C     = Tensor::randomNormal([$this->dState, $this->dModel], 0.0, $stddev);
        // Tiny stddev keeps delta near b_delta. Xavier stddev ≈ 0.072 produces
        // σ(delta) ≈ 1.4 after RMSNorm, pushing almost all values outside the
        // C-kernel clamp range [1e-4, 1.0] where ∂clamp/∂delta = 0, zeroing
        // every W_delta gradient from step 1.
        $this->W_delta = Tensor::randomNormal([$this->dModel, $this->dModel], 0.0, 0.01);

        $this->b_B->fill(0.0);
        $this->b_C->fill(0.0);
        // 0.3 puts initial delta in the middle of [1e-4, 1.0] → Ā = exp(-0.3) ≈ 0.74,
        // giving real state decay and non-trivial input contribution from step 1.
        $this->b_delta->fill(0.3);
    }

    /**
     * 
     * @param LoggerInterface $logger
     * @return void
     */
    public function setLogger(LoggerInterface $logger): void {
        $this->logger = $logger;
        $this->logger->info(sprintf("Mamba Layer Initialized (dModel: %d, dState: %d)", $this->dModel, $this->dState));
    }

    /**
     * 
     * @param bool $mode
     * @return void
     */
    /**
     * Project A_log back into (-20, 0] after each optimizer step.
     *
     * Ā = exp(delta × A_log).  If Adam pushes A_log positive, Ā > 1 and the
     * SSM state grows exponentially over T timesteps.  The backward gradient
     * accumulates Ā^(T-t) factors → gradient norm explodes to 10^16+.
     * Clamping to [-20, 0] keeps Ā ∈ (2×10⁻⁹, 1] — always a stable decay.
     */
    public function enforceStability(): void
    {
        $this->A_log->clampInplace(-20.0, 0.0);
    }

    public function setTraining(bool $mode): void {
        $this->isTraining = $mode;
        if ($this->logger) {
            $this->logger->debug("Mamba Layer switched to " . ($mode ? "Training" : "Inference") . " mode.");
        }
    }

    /**
     * 
     * @param Tensor $input
     * @return Tensor
     */
    public function forward(Tensor $input): Tensor {
        $this->cacheX = $input; // Retain reference for backward pass

        $shape = $input->shape();
        $B = $shape[0];
        $T = $shape[1]; // Sequence Length (Auto-adapts for Audio/Vision/Text)
        $D = $this->dModel;
        $N = $this->dState;

        // 1. Flatten sequence to [B*T, D] for fused linear projections
        $flatX = $input->reshape($B * $T, $D);

        // 2. Fused Linear Projections (Using Tensor.php native `linear` method)
        $flatB = $flatX->linear($this->W_B, $this->b_B);
        $flatC = $flatX->linear($this->W_C, $this->b_C);
        $flatDelta = $flatX->linear($this->W_delta, $this->b_delta);

        // 3. Reshape back to sequence tensors [B, T, ...]
        $this->cacheB = $flatB->reshape($B, $T, $N);
        $this->cacheC = $flatC->reshape($B, $T, $N);
        $this->cacheDelta = $flatDelta->reshape($B, $T, $D);

        // 4. Allocate State & Output via Tensor factory methods
        $state = Tensor::mambaAllocState($B, $D, $N);
        $out = Tensor::zeros($B, $T, $D);
        $this->h0 = Tensor::mambaAllocState($B, $D, $N); // Clean state for backprop

        $this->mambaCache = $this->isTraining ? Tensor::mambaAllocCache($B, $T, $D, $N) : null;

        if ($this->logger && $this->isTraining) {
            $this->logger->debug(sprintf("Mamba Forward: Processing Batch=%d, SeqLen=%d", $B, $T));
        }

        // 5. Execute Hardware Core via Tensor.php
        $input->mambaForward(
                $this->A_log,
                $this->cacheB,
                $this->cacheC,
                $this->D_skip,
                $this->cacheDelta,
                $state,
                $out,
                $this->mambaCache,
                $this->isTraining
        );

        return $out;
    }

    /**
     * 
     * @param Tensor $dY
     * @return Tensor
     */
    public function backward(Tensor $dY): Tensor {
        $shape = $dY->shape();
        $B = $shape[0];
        $T = $shape[1];
        $D = $this->dModel;
        $N = $this->dState;

        // 1. Pre-allocate gradient containers using Tensor object
        $dX_mamba = Tensor::zeros($B, $T, $D);
        $dB_proj = Tensor::zeros($B, $T, $N);
        $dC_proj = Tensor::zeros($B, $T, $N);
        $dDelta = Tensor::zeros($B, $T, $D);

        // A_log and D_skip accumulate inside the kernel, zero them out first
        $this->dA_log->fill(0.0);
        $this->dD_skip->fill(0.0);

        if ($this->logger) {
            $this->logger->debug("Mamba Backward: Computing Hardware Gradients...");
        }

        // 2. Hardware Fused Mamba Backward Core
        $dY->mambaBackward(
                $this->cacheX,
                $this->A_log,
                $this->cacheB,
                $this->cacheC,
                $this->D_skip,
                $this->cacheDelta,
                $this->h0,
                $this->mambaCache,
                $dX_mamba,
                $this->dA_log,
                $dB_proj,
                $dC_proj,
                $this->dD_skip,
                $dDelta
        );

        // 3. Backpropagate through Linear Projections (Zero-copy reshaping)
        $flatX = $this->cacheX->reshape($B * $T, $D);
        $flat_dB = $dB_proj->reshape($B * $T, $N);
        $flat_dC = $dC_proj->reshape($B * $T, $N);
        $flat_dDelta = $dDelta->reshape($B * $T, $D);

        // Calculate Weights Gradients: dW = dY^T @ X
        // true, false equates to: transA=true, transB=false -> (dY)^T @ X
        $this->dW_B = $flat_dB->matmul($flatX, true, false);
        $this->dW_C = $flat_dC->matmul($flatX, true, false);
        $this->dW_delta = $flat_dDelta->matmul($flatX, true, false);

        // Calculate Bias Gradients: db = sum(dY, axis=0)
        $this->db_B = $flat_dB->sumAxis(0);
        $this->db_C = $flat_dC->sumAxis(0);
        $this->db_delta = $flat_dDelta->sumAxis(0);

        // Calculate Input Gradients from projections: dX_proj = dY @ W
        $dX_B = $flat_dB->matmul($this->W_B, false, false);
        $dX_C = $flat_dC->matmul($this->W_C, false, false);
        $dX_Delta = $flat_dDelta->matmul($this->W_delta, false, false);

        // Total input gradient: dX = dX_mamba + dX_B + dX_C + dX_Delta
        $flat_dX_mamba = $dX_mamba->reshape($B * $T, $D);
        $flat_dX_mamba->addInplace($dX_B)->addInplace($dX_C)->addInplace($dX_Delta);

        // Cleanup caches (Handled automatically by Tensor->__destruct when nulled)
        $this->cacheB = $this->cacheC = $this->cacheDelta = null;

        // Output shape matches original input shape automatically
        return $dX_mamba;
    }

    public function getParameters(): array {
        return [
            'A_log' => $this->A_log, 'D_skip' => $this->D_skip,
            'W_B' => $this->W_B, 'b_B' => $this->b_B,
            'W_C' => $this->W_C, 'b_C' => $this->b_C,
            'W_delta' => $this->W_delta, 'b_delta' => $this->b_delta,
        ];
    }

    public function getGradients(): array {
        return [
            'A_log' => $this->dA_log, 'D_skip' => $this->dD_skip,
            'W_B' => $this->dW_B, 'b_B' => $this->db_B,
            'W_C' => $this->dW_C, 'b_C' => $this->db_C,
            'W_delta' => $this->dW_delta, 'b_delta' => $this->db_delta,
        ];
    }

    // --- Stateful Interface Implementation ---

    /**
     * 
     * @param string $prefix
     * @return array
     */
    public function getStateDict(string $prefix = ''): array {
        if ($this->logger) {
            $this->logger->info("Saving Mamba Layer parameters [{$prefix}] to SafeTensors.");
        }

        $dict = [];
        foreach ($this->getParameters() as $key => $tensor) {
            $dict[$prefix . $key] = $tensor;
        }
        return $dict;
    }

    /**
     * 
     * @param array $dict
     * @param string $prefix
     * @return void
     */
    public function loadStateDict(array $dict, string $prefix = ''): void {
        foreach ($this->getParameters() as $key => $tensor) {
            if (isset($dict[$prefix . $key])) {
                $tensor->copyFrom($dict[$prefix . $key]);
            }
        }

        if ($this->logger) {
            $this->logger->info("Successfully loaded Mamba Layer parameters [{$prefix}] from SafeTensors.");
        }
    }

    public function getConfig(): array {
        return ['dModel' => $this->dModel, 'dState' => $this->dState];
    }

    /**
     * 
     * @param array $config
     * @return self
     */
    public static function fromConfig(array $config): self {
        return new self($config['dModel'], $config['dState']);
    }
}
