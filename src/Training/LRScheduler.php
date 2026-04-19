<?php

declare(strict_types=1);

namespace Pml\Training;

use Pml\NeuralNetwork\Optimizers\LearningRateAware;
use Pml\NeuralNetwork\Sequential;

/**
 * Learning-rate scheduler for Sequential neural networks.
 *
 * Computes and applies a new LR at the end of each epoch.
 * Requires the model's optimizer to implement LearningRateAware.
 *
 * Supported schedules (set via TrainingArguments::$lrSchedule):
 *   'none'    — constant LR, no changes.
 *   'step'    — multiply by $decay every $stepSize epochs.
 *   'cosine'  — cosine annealing from baseLr → 0 over $totalEpochs.
 *   'linear'  — linear decay from baseLr → 0 over $totalEpochs.
 *   'warmup'  — linear ramp from 0 → baseLr over $warmupEpochs,
 *               then switches to the chosen post-warmup schedule.
 */
final class LRScheduler
{
    private float $baseLr;
    private ?LearningRateAware $optimizer;

    public function __construct(
        private readonly Sequential $model,
        private readonly TrainingArguments $args
    ) {
        $opt = $model->getOptimizer();
        $this->optimizer = $opt instanceof LearningRateAware ? $opt : null;
        $this->baseLr    = $args->learningRate;

        // Sync the optimizer's initial LR with the requested value.
        $this->optimizer?->setLearningRate($this->baseLr);
    }

    /**
     * Compute and apply the LR for $epoch (1-based).
     * Called by Trainer at the end of each epoch.
     */
    public function step(int $epoch, int $totalEpochs): void
    {
        if ($this->optimizer === null) {
            return; // Optimizer doesn't support LR updates.
        }

        $lr = $this->computeLr($epoch, $totalEpochs);
        $this->optimizer->setLearningRate($lr);
    }

    public function currentLr(): float
    {
        return $this->optimizer?->getLearningRate() ?? $this->baseLr;
    }

    // -------------------------------------------------------------------------

    private function computeLr(int $epoch, int $totalEpochs): float
    {
        // ── Warm-up phase ─────────────────────────────────────────────────────
        $warmup = $this->args->warmupEpochs;
        if ($warmup > 0 && $epoch <= $warmup) {
            return $this->baseLr * ($epoch / $warmup);
        }

        // Adjusted epoch for post-warmup schedules.
        $t     = max(1, $epoch - $warmup);
        $total = max(1, $totalEpochs - $warmup);

        return match ($this->args->lrSchedule) {
            'step'   => $this->stepLr($t),
            'cosine' => $this->cosineLr($t, $total),
            'linear' => $this->linearLr($t, $total),
            default  => $this->baseLr,   // 'none'
        };
    }

    private function stepLr(int $t): float
    {
        $steps = (int) ($t / max(1, $this->args->lrStepSize));
        return $this->baseLr * pow($this->args->lrDecay, $steps);
    }

    private function cosineLr(int $t, int $total): float
    {
        return $this->baseLr * 0.5 * (1.0 + cos(M_PI * $t / $total));
    }

    private function linearLr(int $t, int $total): float
    {
        return $this->baseLr * max(0.0, 1.0 - $t / $total);
    }
}
