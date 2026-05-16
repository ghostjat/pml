<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork;

use Pml\Tensor;
use Pml\Dataset;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Quantizable;
use Pml\Interfaces\Stateful;
use Pml\Interfaces\TrainableWithOptions;
use Pml\Interfaces\Verbose;
use Pml\Lib\ModelStore;
use Pml\Lib\SafeTensorsIO;
use Pml\Losses\Loss;
use Pml\NeuralNetwork\Optimizers\Optimizer;
use Psr\Log\LoggerInterface;

/**
 * A Sequential Neural Network Model Container.
 * Implements TrainableWithOptions so Pipeline can forward epochs, validation,
 * patience, and other deep-learning-specific args without widening Learner.
 */
final class Sequential implements TrainableWithOptions, Persistable, Verbose
{
    /** @var \Pml\NeuralNetwork\Layers\Layer[] */
    private array $layers = [];
    private Loss $lossFn;
    private Optimizer $optimizer;
    private bool $isTrained = false;
    
    private ?LoggerInterface $logger = null;

    public function __construct(array $layers, Loss $lossFn, Optimizer $optimizer)
    {
        foreach ($layers as $layer) {
            $this->add($layer);
        }
        $this->lossFn = $lossFn;
        $this->optimizer = $optimizer;
    }

    public function setLogger(LoggerInterface $logger): void
    {
        $this->logger = $logger;
    }

    public function add(Layers\Layer $layer): void
    {
        $this->layers[] = $layer;
    }

    // ---- Read-only accessors used by Trainer / LRScheduler ---------------

    /** @return Layers\Layer[] */
    public function getLayers(): array { return $this->layers; }

    public function getOptimizer(): Optimizer { return $this->optimizer; }

    public function getLoss(): Loss { return $this->lossFn; }

    // ----------------------------------------------------------------------

    public function forward(Tensor $input): Tensor
    {
        $current = $input;
        foreach ($this->layers as $layer) {
            $current = $layer->forward($current);
        }
        return $current;
    }

    public function backward(Tensor $lossGradient): void
    {
        $currentGradient = $lossGradient;
        for ($i = \count($this->layers) - 1; $i >= 0; $i--) {
            $currentGradient = $this->layers[$i]->backward($currentGradient);
        }
    }

    /**
     * Toggles the behavior of regularization layers (like Dropout/BatchNorm).
     */
    private function setTrainingMode(bool $mode): void
    {
        foreach ($this->layers as $layer) {
            if ($layer instanceof Layers\HasTrainingMode) {
                $layer->setTraining($mode);
            }
        }
    }

    /**
     * Global gradient-norm clipping (L2).
     * Computes the total norm across all gradients, then scales every gradient
     * tensor in-place so the global norm equals $maxNorm.
     * Single pass: O(P) where P = total parameters. No extra allocations.
     */
    private function clipGradients(float $maxNorm): void
    {
        // Pass 1: accumulate sum of squared gradient elements across all layers
        $totalSqNorm = 0.0;
        foreach ($this->layers as $layer) {
            foreach ($layer->getGradients() as $grad) {
                $totalSqNorm += $grad->square()->sum();
            }
        }

        $globalNorm = \sqrt($totalSqNorm);
        if ($globalNorm <= $maxNorm || $globalNorm === 0.0) {
            return;
        }

        // Pass 2: scale all gradients in-place by (maxNorm / globalNorm)
        $scale = $maxNorm / $globalNorm;
        foreach ($this->layers as $layer) {
            foreach ($layer->getGradients() as $grad) {
                $grad->mulScalarInplace($scale);
            }
        }
    }

    /**
     * Train the model with optional Early Stopping and Validation monitoring.
     *
     * Accepts options as named variadic args (PHP 8 named-arg spread) so the
     * signature satisfies TrainableWithOptions while still being callable with
     * named parameters: $model->train($ds, epochs: 20, batchSize: 64).
     *
     * @param Dataset $dataset    Training dataset.
     * @param mixed   ...$options Supported keys (all optional):
     *   int        epochs      (default 10)
     *   int        batchSize   (default 32)
     *   Dataset    validation  (default null)
     *   int        patience    (default 0 — disabled)
     *   float      minDelta    (default 1e-4)
     *   float      clipGradNorm Global gradient-norm clip threshold (default 0.0 = disabled).
     *                           Clips all parameter gradients so ||g||₂ ≤ clipGradNorm.
     *                           Recommended ≈ 1.0–5.0 for RNN/LSTM to prevent exploding gradients.
     */
    public function train(Dataset $dataset, mixed ...$options): void
    {
        $epochs       = (int)   ($options['epochs']       ?? 10);
        $batchSize    = (int)   ($options['batchSize']    ?? 32);
        $validation   = isset($options['validation']) && $options['validation'] instanceof Dataset
            ? $options['validation'] : null;
        $patience     = (int)   ($options['patience']     ?? 0);
        $minDelta     = (float) ($options['minDelta']     ?? 1e-4);
        $clipGradNorm = (float) ($options['clipGradNorm'] ?? 0.0);

        $es        = $patience > 0 ? new EarlyStopping($patience, 'min', $minDelta) : null;
        $bestState = [];

        for ($epoch = 1; $epoch <= $epochs; $epoch++) {
            
            // --- 1. TRAINING PHASE ---
            $this->setTrainingMode(true); // Ensure all layers are actively learning
            $dataset->randomize();
            $trainLoss = 0.0;
            $trainSteps = 0;

            foreach ($dataset->batches($batchSize) as $batch) {
                $x = $batch->samples();
                $y = $batch->labels();

                $predictions = $this->forward($x);

                if ($this->logger !== null) {
                    $trainLoss += $this->lossFn->compute($predictions, $y);
                }

                $lossGradient = $this->lossFn->differentiate($predictions, $y);
                $this->backward($lossGradient);

                if ($clipGradNorm > 0.0) {
                    $this->clipGradients($clipGradNorm);
                }

                $this->optimizer->step($this->layers);
                $trainSteps++;
            }

            $avgTrainLoss = $trainSteps > 0 ? $trainLoss / $trainSteps : 0.0;

            // --- 2. VALIDATION & EARLY STOPPING PHASE ---
            if ($validation !== null) {
                
                $this->setTrainingMode(false); // Freeze layers for objective validation evaluation
                
                $valLoss = 0.0;
                $valSteps = 0;
                
                // Zero-copy evaluation pass (No backprop)
                foreach ($validation->batches($batchSize) as $valBatch) {
                    $valPreds = $this->forward($valBatch->samples());
                    $valLoss += $this->lossFn->compute($valPreds, $valBatch->labels());
                    $valSteps++;
                }
                
                $avgValLoss = $valSteps > 0 ? $valLoss / $valSteps : 0.0;

                if ($this->logger !== null) {
                    $this->logger->info(\sprintf("Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f", $epoch, $epochs, $avgTrainLoss, $avgValLoss));
                }

                // Check Early Stopping Criteria
                if ($es !== null) {
                    $signal = $es->update($avgValLoss);
                    if ($signal === EarlyStopping::IMPROVED) {
                        $bestState = $this->cloneState();
                    } elseif ($signal === EarlyStopping::STOP) {
                        if ($this->logger !== null) {
                            $this->logger->info(\sprintf("Early stopping triggered at Epoch %d (Best Val Loss: %.6f)", $epoch, $es->getBestMetric()));
                        }
                        $this->restoreState($bestState);
                        break;
                    }
                }
            } else {
                // Basic Logging (No Validation)
                if ($this->logger !== null) {
                    $this->logger->info(\sprintf("Epoch %d/%d - Train Loss: %.6f", $epoch, $epochs, $avgTrainLoss));
                }
            }
        }
        
        $this->isTrained = true;
    }

    /**
     * Single forward+backward+optimizer step on one pre-transformed batch.
     * Returns the scalar cross-entropy loss for that batch.
     * Used by streaming training loops that manage their own epoch logic.
     */
    public function stepOnBatch(Dataset $batch, float $clipGradNorm = 0.0): float
    {
        $this->setTrainingMode(true);
        $pred = $this->forward($batch->samples());
        $loss = $this->lossFn->compute($pred, $batch->labels());
        $this->backward($this->lossFn->differentiate($pred, $batch->labels()));
        if ($clipGradNorm > 0.0) $this->clipGradients($clipGradNorm);
        $this->optimizer->step($this->layers);
        return $loss;
    }

    /**
     * Mark the model as trained without going through train().
     * Call after a streaming training loop finishes.
     */
    public function markTrained(): void
    {
        $this->isTrained = true;
    }

    public function predict(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new \RuntimeException("Model is not trained.");
        }
        
        $this->setTrainingMode(false); // Turn off Dropout & freeze BatchNorm
        $predictions = $this->forward($dataset->samples());
        $this->setTrainingMode(true);  // Restore default state
        
        return $predictions;
    }

    public function trained(): bool
    {
        return $this->isTrained;
    }

    /**
     * Quantize all Quantizable layers (Dense) in the model to INT8.
     *
     * After this call, forward() runs the AVX2 fused int8→fp32 kernel for
     * every quantized layer.  The fp32 weight matrices are freed immediately,
     * reducing peak RAM by ~4× for weight storage.
     *
     * @param int $groupSize  Elements per quantization group (32 = Q8_0-class).
     */
    public function quantize(int $groupSize = 32): void
    {
        foreach ($this->layers as $layer) {
            if ($layer instanceof Quantizable) {
                $layer->quantize($groupSize);
            }
        }
    }

    // ========================================================================
    // STATE MANAGEMENT (For Early Stopping & Backups)
    // ========================================================================

    /**
     * Safely snapshots the C-memory state of the entire network.
     */
    private function cloneState(): array
    {
        $state = [];
        foreach ($this->layers as $i => $layer) {
            foreach ($layer->getParameters() as $name => $tensor) {
                // Allocates new C-memory for the backup. PHP GC handles the old ones.
                $state["{$i}_{$name}"] = $tensor->copy();
            }
        }
        return $state;
    }

    /**
     * Overwrites the active network with a previously saved snapshot.
     */
    private function restoreState(array $state): void
    {
        if (empty($state)) return;
        
        foreach ($this->layers as $i => $layer) {
            foreach ($layer->getParameters() as $name => $tensor) {
                $key = "{$i}_{$name}";
                if (isset($state[$key])) {
                    // ZERO-ALLOCATION RESTORE TRICK:
                    // Zeros out the active memory, then adds the backup bytes directly over it.
                    // This prevents re-instantiating FFI pointers and maintains reference integrity!
                    $tensor->mulScalarInplace(0.0)->addInplace($state[$key]);
                }
            }
        }
    }

    // ========================================================================
    // PERSISTENCE — ModelStore + SafeTensors bundle (zero serialize())
    //
    // Saved layout:
    //   $dir/config.json         — layers, loss, optimizer as JSON (ModelStore)
    //   $dir/model.safetensors  — all Stateful Tensor weights (HF-compatible)
    //
    // serialize() is NEVER called.  All PHP objects (Loss, Optimizer, layers)
    // are encoded via ModelStore::toArray() which uses Saveable or Reflection
    // and skips Tensor / FFI\CData values entirely.
    // Tensor C-memory travels exclusively through SafeTensors (zero-copy).
    // ========================================================================

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $config = [
            'class'     => self::class,
            'isTrained' => $this->isTrained,
            'lossFn'    => ModelStore::toArray($this->lossFn),
            'optimizer' => ModelStore::toArray($this->optimizer),
            'layers'    => [],
        ];

        $tensorDict = [];

        foreach ($this->layers as $i => $layer) {
            $prefix = "layer_{$i}.";

            if ($layer instanceof Stateful) {
                foreach ($layer->getStateDict($prefix) as $key => $tensor) {
                    $tensorDict[$key] = $tensor;
                }
                $config['layers'][] = [
                    'type'   => 'stateful',
                    'prefix' => $prefix,
                    'data'   => ModelStore::toArray($layer),
                ];
            } else {
                $config['layers'][] = [
                    'type' => 'plain',
                    'data' => ModelStore::toArray($layer),
                ];
            }
        }

        if (!empty($tensorDict)) {
            SafeTensorsIO::save($dir . \DIRECTORY_SEPARATOR . 'model.safetensors', $tensorDict);
        }

        file_put_contents(
            $dir . \DIRECTORY_SEPARATOR . 'config.json',
            json_encode($config, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );
    }

    public static function load(string $dir): self
    {
        if (!is_dir($dir)) {
            throw new \RuntimeException("Sequential::load — directory not found: '$dir'.");
        }

        $raw = file_get_contents($dir . \DIRECTORY_SEPARATOR . 'config.json');
        if ($raw === false) {
            throw new \RuntimeException("Sequential::load — config.json missing in '$dir'.");
        }

        /** @var array<string,mixed> $config */
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $lossFn    = ModelStore::fromArray($config['lossFn']);
        $optimizer = ModelStore::fromArray($config['optimizer']);

        // All Tensor weights mmap'd in one call (zero-copy).
        $stPath  = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        $weights = is_file($stPath) ? SafeTensorsIO::load($stPath) : [];

        $layers = [];
        foreach ($config['layers'] as $layerCfg) {
            $layer = ModelStore::fromArray($layerCfg['data']);

            if ($layerCfg['type'] === 'stateful' && $layer instanceof Stateful) {
                $layer->loadStateDict($weights, $layerCfg['prefix']);
            }

            $layers[] = $layer;
        }

        $model            = new self($layers, $lossFn, $optimizer);
        $model->isTrained = (bool) $config['isTrained'];

        return $model;
    }
}