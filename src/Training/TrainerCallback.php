<?php

declare(strict_types=1);

namespace Pml\Training;

/**
 * Hook interface for training lifecycle events.
 *
 * Implement this to add custom logic at key points in the training loop
 * (logging to external dashboards, LR warmup, gradient monitoring, etc.).
 * All methods have empty default implementations — only override what you need.
 */
interface TrainerCallback
{
    /**
     * Called once before the first epoch begins.
     *
     * @param TrainingArguments $args   The resolved training configuration.
     * @param int               $steps  Batches per epoch (from DataLoader::steps()).
     */
    public function onTrainBegin(TrainingArguments $args, int $steps): void;

    /**
     * Called at the start of each epoch (before any batch processing).
     *
     * @param int   $epoch   1-based epoch number.
     * @param int   $epochs  Total planned epochs.
     */
    public function onEpochBegin(int $epoch, int $epochs): void;

    /**
     * Called after each batch is processed.
     *
     * @param int   $step      0-based batch index within the epoch.
     * @param float $batchLoss Raw loss for this batch.
     */
    public function onBatchEnd(int $step, float $batchLoss): void;

    /**
     * Called at the end of each epoch with aggregate metrics.
     *
     * @param int        $epoch      1-based epoch number.
     * @param float      $trainLoss  Average training loss over this epoch.
     * @param float|null $valLoss    Average validation loss, or null if no validation set.
     */
    public function onEpochEnd(int $epoch, float $trainLoss, ?float $valLoss): void;

    /**
     * Called once after training completes (or is stopped early).
     *
     * @param TrainingResult $result  Final training outcome summary.
     */
    public function onTrainEnd(TrainingResult $result): void;
}
