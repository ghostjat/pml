<?php

declare(strict_types=1);

namespace Pml\Training;

/**
 * Immutable summary of a completed training run.
 *
 * Returned by Trainer::train() and passed to TrainerCallback::onTrainEnd().
 */
final class TrainingResult
{
    /**
     * @param int          $epochsRun       Actual epochs executed (may be < planned if early-stopped).
     * @param float[]      $trainLossHistory Per-epoch average training loss (index 0 = epoch 1).
     * @param float[]      $valLossHistory   Per-epoch average validation loss (empty if no validation).
     * @param float|null   $bestValLoss      Best validation loss achieved, or null if no validation.
     * @param bool         $earlyStopped     Whether early stopping triggered.
     * @param float        $elapsedSeconds   Wall-clock seconds for the whole run.
     */
    public function __construct(
        public readonly int $epochsRun,
        public readonly array $trainLossHistory,
        public readonly array $valLossHistory,
        public readonly ?float $bestValLoss,
        public readonly bool $earlyStopped,
        public readonly float $elapsedSeconds,
    ) {}

    /** Final (last-epoch) training loss. */
    public function finalTrainLoss(): float
    {
        return !empty($this->trainLossHistory)
            ? $this->trainLossHistory[\count($this->trainLossHistory) - 1]
            : 0.0;
    }

    /** Final (last-epoch) validation loss, or null. */
    public function finalValLoss(): ?float
    {
        return !empty($this->valLossHistory)
            ? $this->valLossHistory[\count($this->valLossHistory) - 1]
            : null;
    }
}
