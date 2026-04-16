<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork;

use Pml\Tensor;
use Pml\Dataset;
use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Stateful;
use Pml\Interfaces\Verbose;
use Pml\Lib\SafeTensorsIO;
use Pml\Losses\Loss;
use Pml\NeuralNetwork\Optimizers\Optimizer;
use Psr\Log\LoggerInterface;

/**
 * A Sequential Neural Network Model Container.
 * * Upgraded with Zero-Copy Validation & Early Stopping.
 * * Auto-toggles layer training states for Dropout and Batch Normalization.
 */
final class Sequential implements Learner, Persistable, Verbose
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
     * Train the model with optional Early Stopping and Validation monitoring.
     * * @param Dataset $dataset Training dataset.
     * @param int $epochs Maximum epochs to train.
     * @param int $batchSize Size of the zero-copy mini-batches.
     * @param Dataset|null $validation Validation dataset to monitor for overfitting.
     * @param int $patience Number of epochs to wait for improvement before early stopping (0 = disabled).
     * @param float $minDelta Minimum change in validation loss to qualify as an improvement.
     */
    public function train(
        Dataset $dataset, 
        int $epochs = 10, 
        int $batchSize = 32,
        ?Dataset $validation = null,
        int $patience = 0,
        float $minDelta = 1e-4
    ): void {
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

                if ($this->logger !== null || $validation === null) {
                    $trainLoss += $this->lossFn->compute($predictions, $y);
                }

                $lossGradient = $this->lossFn->differentiate($predictions, $y);
                $this->backward($lossGradient);
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
    // PERSISTENCE — SafeTensors + JSON bundle (zero Tensor serialisation)
    //
    // Saved layout:
    //   $dir/config.json          — PHP class names, hyperparams, no C-data
    //   $dir/model.safetensors   — all Stateful tensor weights (HF-compatible)
    //
    // PHP serialize() is used ONLY for pure-PHP objects (Loss, Optimizer,
    // and layer shells that have had every Tensor property stripped out).
    // C-memory is never passed through PHP serialize().
    // ========================================================================

    public function save(string $dir): void
    {
        if (!is_dir($dir)) {
            mkdir($dir, 0755, true);
        }

        $config = [
            'class'     => self::class,
            'isTrained' => $this->isTrained,
            // Loss / Optimizer are pure PHP — no C-pointers, safe to serialize.
            'lossFn'    => base64_encode(serialize($this->lossFn)),
            'optimizer' => base64_encode(serialize($this->optimizer)),
            'layers'    => [],
        ];

        $tensorDict = [];   // name → Tensor, fed to SafeTensorsIO::save()

        foreach ($this->layers as $i => $layer) {
            $prefix = "layer_{$i}.";

            if ($layer instanceof Stateful) {
                // Collect live C-memory tensors — zero-copy, just PHP references.
                foreach ($layer->getStateDict($prefix) as $key => $tensor) {
                    $tensorDict[$key] = $tensor;
                }

                if (method_exists($layer, 'getConfig')) {
                    // Clean path: pure-JSON descriptor, no object serialisation.
                    $config['layers'][] = [
                        'type'   => 'stateful_config',
                        'class'  => \get_class($layer),
                        'prefix' => $prefix,
                        'config' => $layer->getConfig(),
                    ];
                } else {
                    // Fallback: serialise a Tensor-stripped shell.
                    $config['layers'][] = [
                        'type'   => 'stateful_shell',
                        'prefix' => $prefix,
                        'shell'  => base64_encode(serialize(self::stripTensors($layer))),
                    ];
                }
            } else {
                // Activation / dropout / etc. — only nullable ?Tensor caches,
                // all reset to null before serialisation by stripTensors().
                $config['layers'][] = [
                    'type'  => 'plain',
                    'shell' => base64_encode(serialize(self::stripTensors($layer))),
                ];
            }
        }

        // Write tensor weights as a single HF-compatible file.
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

        $configPath = $dir . \DIRECTORY_SEPARATOR . 'config.json';
        $raw = file_get_contents($configPath);
        if ($raw === false) {
            throw new \RuntimeException("Sequential::load — config.json missing in '$dir'.");
        }

        /** @var array<string,mixed> $config */
        $config = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);

        $lossFn    = unserialize(base64_decode($config['lossFn']));
        $optimizer = unserialize(base64_decode($config['optimizer']));

        // Load SafeTensors once — all tensor weights are mmap'd (zero-copy).
        $stPath  = $dir . \DIRECTORY_SEPARATOR . 'model.safetensors';
        $weights = is_file($stPath) ? SafeTensorsIO::load($stPath) : [];

        $layers = [];
        foreach ($config['layers'] as $layerCfg) {
            switch ($layerCfg['type']) {
                case 'stateful_config':
                    // Reconstruct from pure-JSON descriptor; no unserialize of C-data.
                    $class = $layerCfg['class'];
                    /** @var \Pml\NeuralNetwork\Layers\Layer&Stateful $layer */
                    $layer = $class::fromConfig($layerCfg['config']);
                    $layer->loadStateDict($weights, $layerCfg['prefix']);
                    break;

                case 'stateful_shell':
                    $layer = unserialize(base64_decode($layerCfg['shell']));
                    if ($layer instanceof Stateful) {
                        $layer->loadStateDict($weights, $layerCfg['prefix']);
                    }
                    break;

                default: // 'plain'
                    $layer = unserialize(base64_decode($layerCfg['shell']));
                    break;
            }
            $layers[] = $layer;
        }

        $model            = new self($layers, $lossFn, $optimizer);
        $model->isTrained = (bool) $config['isTrained'];

        return $model;
    }

    // -------------------------------------------------------------------------
    // Helper: produce a clone with all Tensor properties nulled/zeroed so that
    // PHP serialize() never touches a C-pointer.
    // -------------------------------------------------------------------------

    /**
     * Clone $obj and neutralise every Tensor-typed property so that PHP's
     * serialize() never encounters an \FFI\CData value.
     *
     * - Nullable ?Tensor props → set to null via ReflectionProperty::setValue().
     * - Non-nullable Tensor props → unset() via Closure::bind(), leaving the
     *   property "uninitialized" (PHP serialises these cleanly; unserialize
     *   restores them as uninitialized, ready for loadStateDict() to fill in).
     */
    private static function stripTensors(object $obj): object
    {
        $clone = clone $obj;
        $class = \get_class($clone);

        foreach ((new \ReflectionClass($clone))->getProperties() as $prop) {
            $type = $prop->getType();
            if (!$type instanceof \ReflectionNamedType) {
                continue;
            }
            $typeName = $type->getName();
            if ($typeName !== Tensor::class && !is_subclass_of($typeName, Tensor::class)) {
                continue;
            }

            $prop->setAccessible(true);

            if ($type->allowsNull()) {
                $prop->setValue($clone, null);
            } else {
                // Non-nullable: bypass type enforcement by unsetting via a
                // closure bound to the object's private scope.
                $name = $prop->getName();
                \Closure::bind(
                    static function (object $o) use ($name): void { unset($o->$name); },
                    null,
                    $class
                )($clone);
            }
        }

        return $clone;
    }
}