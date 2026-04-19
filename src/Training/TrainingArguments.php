<?php

declare(strict_types=1);

namespace Pml\Training;

use Pml\Dataset;

/**
 * Hyperparameter container for a Trainer run.
 *
 * All fields have sensible defaults so callers only need to override
 * what they care about.  Immutable after construction — clone and
 * override to create variations.
 */
final class TrainingArguments
{
    public function __construct(
        // ── Core ──────────────────────────────────────────────────────────────
        public readonly int $epochs = 10,
        public readonly int $batchSize = 32,

        // ── Regularisation / Early Stopping ──────────────────────────────────
        public readonly int $patience = 0,          // 0 = disabled
        public readonly float $minDelta = 1e-4,

        // ── Learning Rate ─────────────────────────────────────────────────────
        public readonly float $learningRate = 0.001,
        public readonly string $lrSchedule = 'none', // 'none' | 'cosine' | 'step' | 'linear'
        public readonly float $lrDecay = 0.1,        // factor for 'step' schedule
        public readonly int $lrStepSize = 5,          // epochs between LR drops (step schedule)
        public readonly int $warmupEpochs = 0,        // linear warm-up from 0 → learningRate

        // ── Mixed Precision ───────────────────────────────────────────────────
        public readonly bool $mixedPrecision = false, // scaffold; actual AMP requires CUDA

        // ── Checkpointing ─────────────────────────────────────────────────────
        public readonly ?string $outputDir = null,    // where to write checkpoints
        public readonly int $saveEvery = 0,           // 0 = only save at end; N = every N epochs
        public readonly bool $saveBest = true,        // keep the best-val-loss checkpoint

        // ── Logging ───────────────────────────────────────────────────────────
        public readonly int $logEvery = 1,            // log every N epochs (0 = silent)
    ) {}

    /**
     * Forward all deep-learning args to Sequential::train() as a named array.
     * The spread operator `...$args->toTrainOptions()` works for variadic calls.
     *
     * @return array<string,mixed>
     */
    public function toTrainOptions(?Dataset $validation = null): array
    {
        return [
            'epochs'     => $this->epochs,
            'batchSize'  => $this->batchSize,
            'validation' => $validation,
            'patience'   => $this->patience,
            'minDelta'   => $this->minDelta,
        ];
    }
}
