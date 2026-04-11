<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork;

use Pml\Tensor;
use Pml\Dataset;
use Pml\Interfaces\Learner;
use Pml\Interfaces\Persistable;
use Pml\Interfaces\Verbose;
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
        for ($i = count($this->layers) - 1; $i >= 0; $i--) {
            $currentGradient = $this->layers[$i]->backward($currentGradient);
        }
    }

    /**
     * Toggles the behavior of regularization layers (like Dropout/BatchNorm).
     */
    private function setTrainingMode(bool $mode): void
    {
        foreach ($this->layers as $layer) {
            if (property_exists($layer, 'training')) {
                $layer->training = $mode;
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
        $bestLoss = INF;
        $patienceCounter = 0;
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
                    $this->logger->info(sprintf("Epoch %d/%d - Train Loss: %.6f, Val Loss: %.6f", $epoch, $epochs, $avgTrainLoss, $avgValLoss));
                }

                // Check Early Stopping Criteria
                if ($patience > 0) {
                    if ($avgValLoss < $bestLoss - $minDelta) {
                        // Improvement found! Reset patience and snapshot weights.
                        $bestLoss = $avgValLoss;
                        $patienceCounter = 0;
                        $bestState = $this->cloneState();
                    } else {
                        // No improvement.
                        $patienceCounter++;
                        if ($patienceCounter >= $patience) {
                            if ($this->logger !== null) {
                                $this->logger->info(sprintf("Early stopping triggered at Epoch %d (Best Val Loss: %.6f)", $epoch, $bestLoss));
                            }
                            $this->restoreState($bestState);
                            break;
                        }
                    }
                }
            } else {
                // Basic Logging (No Validation)
                if ($this->logger !== null) {
                    $this->logger->info(sprintf("Epoch %d/%d - Train Loss: %.6f", $epoch, $epochs, $avgTrainLoss));
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
    // PERSISTENCE (SSD Serialization)
    // ========================================================================

    public function save(string $filepath): void
    {
        if (!is_dir($filepath)) {
            mkdir($filepath, 0777, true);
        }

        $manifest = [
            'isTrained' => $this->isTrained,
            'lossFn'    => serialize($this->lossFn),
            'optimizer' => serialize($this->optimizer),
            'layers'    => []
        ];

        foreach ($this->layers as $i => $layer) {
            $layerClone = clone $layer;
            $params = $layer->getParameters();
            
            foreach ($params as $name => $tensor) {
                $tensorFile = "layer_{$i}_{$name}.tns";
                $tensor->save($filepath . DIRECTORY_SEPARATOR . $tensorFile);
            }

            $refClass = new \ReflectionClass($layerClone);
            foreach ($refClass->getProperties() as $prop) {
                $type = $prop->getType();
                if ($type instanceof \ReflectionNamedType && $type->getName() === Tensor::class) {
                    $prop->setAccessible(true);
                    $prop->setValue($layerClone, null); 
                }
            }

            $manifest['layers'][] = serialize($layerClone);
        }

        file_put_contents($filepath . DIRECTORY_SEPARATOR . 'manifest.json', json_encode($manifest));
    }

    public static function load(string $filepath): self
    {
        if (!is_dir($filepath)) {
            throw new \RuntimeException("Model directory not found: {$filepath}");
        }

        $manifestJson = file_get_contents($filepath . DIRECTORY_SEPARATOR . 'manifest.json');
        if (!$manifestJson) {
            throw new \RuntimeException("Model manifest is corrupt or missing.");
        }

        $manifest = json_decode($manifestJson, true);
        $lossFn = unserialize($manifest['lossFn']);
        $optimizer = unserialize($manifest['optimizer']);
        $layers = [];

        foreach ($manifest['layers'] as $i => $serializedLayer) {
            $layer = unserialize($serializedLayer);
            $refClass = new \ReflectionClass($layer);
            
            foreach (glob($filepath . DIRECTORY_SEPARATOR . "layer_{$i}_*.tns") as $tensorFile) {
                preg_match("/layer_{$i}_(.*)\.tns/", basename($tensorFile), $matches);
                if ($matches) {
                    $name = $matches[1];
                    if ($refClass->hasProperty($name)) {
                        $prop = $refClass->getProperty($name);
                        $prop->setAccessible(true);
                        $prop->setValue($layer, Tensor::load($tensorFile));
                    }
                }
            }
            $layers[] = $layer;
        }

        $model = new self($layers, $lossFn, $optimizer);
        $model->isTrained = $manifest['isTrained'];

        return $model;
    }
}