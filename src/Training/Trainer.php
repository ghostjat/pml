<?php

declare(strict_types=1);

namespace Pml\Training;

use Pml\Backends\TorchBackend;
use Pml\Data\DataLoader;
use Pml\Dataset;
use Pml\Interfaces\MLBackend;
use Pml\Losses\Loss;
use Pml\NeuralNetwork\Layers\HasTrainingMode;
use Pml\NeuralNetwork\Sequential;
use Psr\Log\LoggerInterface;
use Psr\Log\NullLogger;

/**
 * High-level training orchestrator.
 *
 * Drives a full training loop with:
 *  - DataLoader-based batching with optional shuffle and drop_last.
 *  - Per-epoch LR scheduling via LRScheduler.
 *  - Mixed-precision scaffolding via GradScaler (CPU no-op).
 *  - Pluggable TrainerCallback hooks at every lifecycle event.
 *  - Optional checkpoint saving every N epochs (and always at end).
 *  - PSR-3 logging for human-readable progress output.
 *
 * Designed for TorchBackend (Sequential) but also accepts any MLBackend
 * for classic ML backends (though LR scheduling won't apply there).
 *
 * Usage:
 *   $trainer = new Trainer($backend, $args, logger: $myLogger);
 *   $result  = $trainer->train($trainDataset, $valDataset);
 */
final class Trainer
{
    private readonly LoggerInterface $logger;

    /** @var TrainerCallback[] */
    private array $callbacks = [];

    public function __construct(
        private readonly MLBackend $backend,
        private readonly TrainingArguments $args,
        ?LoggerInterface $logger = null
    ) {
        $this->logger = $logger ?? new NullLogger();
    }

    public function addCallback(TrainerCallback $callback): void
    {
        $this->callbacks[] = $callback;
    }

    // -------------------------------------------------------------------------

    /**
     * Run the full training loop.
     *
     * For TorchBackend: drives a manual epoch × batch loop via DataLoader
     * so callbacks and LR scheduling work at granular resolution.
     *
     * For other backends: delegates to MLBackend::fit() and wraps the
     * result in a TrainingResult.
     *
     * @param Dataset      $dataset     Training dataset.
     * @param Dataset|null $validation  Optional validation dataset.
     * @return TrainingResult
     */
    public function train(Dataset $dataset, ?Dataset $validation = null): TrainingResult
    {
        // Non-torch backends: delegate and return a minimal result.
        if (!$this->backend instanceof TorchBackend) {
            $t0 = microtime(true);
            $this->backend->fit($dataset, ...$this->args->toTrainOptions($validation));
            return new TrainingResult(
                epochsRun: $this->args->epochs,
                trainLossHistory: [],
                valLossHistory: [],
                bestValLoss: null,
                earlyStopped: false,
                elapsedSeconds: microtime(true) - $t0,
            );
        }

        return $this->trainSequential($this->backend->model(), $dataset, $validation);
    }

    // -------------------------------------------------------------------------
    // Internal: manual training loop for Sequential
    // -------------------------------------------------------------------------

    private function trainSequential(
        Sequential $model,
        Dataset $dataset,
        ?Dataset $validation
    ): TrainingResult {
        $args       = $this->args;
        $loader     = new DataLoader($dataset, $args->batchSize, shuffle: true);
        $scheduler  = new LRScheduler($model, $args);
        $scaler     = new GradScaler(enabled: $args->mixedPrecision);
        $lossFn     = $model->getLoss();

        $trainHistory = [];
        $valHistory   = [];
        $bestValLoss  = PHP_FLOAT_MAX;
        $bestState    = [];
        $earlyStopped = false;
        $staleEpochs  = 0;
        $t0           = microtime(true);

        $this->fireCallback('onTrainBegin', $args, $loader->steps());

        for ($epoch = 1; $epoch <= $args->epochs; $epoch++) {
            $this->fireCallback('onEpochBegin', $epoch, $args->epochs);

            // ── Training pass ─────────────────────────────────────────────────
            $this->setTrainingMode($model, true);
            $trainLoss  = 0.0;
            $trainSteps = 0;

            foreach ($loader->batches() as $step => $batch) {
                $x    = $batch->inputs();
                $y    = $batch->labels();

                $preds    = $model->forward($x);
                $rawLoss  = $lossFn->compute($preds, $y);
                $grad     = $lossFn->differentiate($preds, $y);

                $scaledGrad = $scaler->scale($grad);
                $model->backward($scaledGrad);
                $scaler->unscaleAndStep($model->getOptimizer(), $model->getLayers());
                $scaler->update();

                $trainLoss += $rawLoss;
                $trainSteps++;

                $this->fireCallback('onBatchEnd', $step, $rawLoss);
            }

            $avgTrainLoss = $trainSteps > 0 ? $trainLoss / $trainSteps : 0.0;
            $trainHistory[] = $avgTrainLoss;

            // ── Validation pass ───────────────────────────────────────────────
            $avgValLoss = null;

            if ($validation !== null) {
                $this->setTrainingMode($model, false);
                $valLoader = new DataLoader($validation, $args->batchSize, shuffle: false);
                $valLoss   = 0.0;
                $valSteps  = 0;

                foreach ($valLoader->batches() as $valBatch) {
                    $valPreds = $model->forward($valBatch->inputs());
                    $valLoss += $lossFn->compute($valPreds, $valBatch->labels());
                    $valSteps++;
                }

                $avgValLoss   = $valSteps > 0 ? $valLoss / $valSteps : 0.0;
                $valHistory[] = $avgValLoss;

                // ── Early Stopping ────────────────────────────────────────────
                if ($avgValLoss < $bestValLoss - $args->minDelta) {
                    $bestValLoss = $avgValLoss;
                    $staleEpochs = 0;
                    if ($args->saveBest && $args->outputDir !== null) {
                        $this->saveCheckpoint($epoch, 'best');
                    }
                } else {
                    $staleEpochs++;
                    if ($args->patience > 0 && $staleEpochs >= $args->patience) {
                        $earlyStopped = true;
                        $this->fireCallback('onEpochEnd', $epoch, $avgTrainLoss, $avgValLoss);
                        $this->log($epoch, $args->epochs, $avgTrainLoss, $avgValLoss, $scheduler);
                        break;
                    }
                }
            }

            // ── LR Scheduling ─────────────────────────────────────────────────
            $scheduler->step($epoch, $args->epochs);

            // ── Periodic checkpoint ───────────────────────────────────────────
            if ($args->saveEvery > 0 && $epoch % $args->saveEvery === 0 && $args->outputDir !== null) {
                $this->saveCheckpoint($epoch, "epoch_{$epoch}");
            }

            // ── Logging & callbacks ───────────────────────────────────────────
            if ($args->logEvery > 0 && $epoch % $args->logEvery === 0) {
                $this->log($epoch, $args->epochs, $avgTrainLoss, $avgValLoss, $scheduler);
            }

            $this->fireCallback('onEpochEnd', $epoch, $avgTrainLoss, $avgValLoss);
        }

        // Restore training mode and mark the model as trained.
        $this->setTrainingMode($model, true);

        // Final checkpoint save.
        if ($args->outputDir !== null) {
            $this->saveCheckpoint(null, 'final');
        }

        $result = new TrainingResult(
            epochsRun: \count($trainHistory),
            trainLossHistory: $trainHistory,
            valLossHistory: $valHistory,
            bestValLoss: $bestValLoss < PHP_FLOAT_MAX ? $bestValLoss : null,
            earlyStopped: $earlyStopped,
            elapsedSeconds: microtime(true) - $t0,
        );

        $this->fireCallback('onTrainEnd', $result);

        return $result;
    }

    // -------------------------------------------------------------------------

    private function setTrainingMode(Sequential $model, bool $mode): void
    {
        foreach ($model->getLayers() as $layer) {
            if ($layer instanceof HasTrainingMode) {
                $layer->setTraining($mode);
            }
        }
    }

    private function saveCheckpoint(?int $epoch, string $tag): void
    {
        $dir = rtrim($this->args->outputDir, '/\\')
            . \DIRECTORY_SEPARATOR . $tag;
        $this->backend->save($dir);
        $this->logger->info("Checkpoint saved → {$dir}");
    }

    private function log(
        int $epoch,
        int $total,
        float $trainLoss,
        ?float $valLoss,
        LRScheduler $scheduler
    ): void {
        $lr  = \sprintf('%.2e', $scheduler->currentLr());
        $msg = \sprintf('Epoch %d/%d — train_loss=%.6f', $epoch, $total, $trainLoss);
        if ($valLoss !== null) {
            $msg .= \sprintf('  val_loss=%.6f', $valLoss);
        }
        $msg .= "  lr={$lr}";
        $this->logger->info($msg);
    }

    /** @param mixed ...$args */
    private function fireCallback(string $method, mixed ...$args): void
    {
        foreach ($this->callbacks as $cb) {
            $cb->$method(...$args);
        }
    }
}
