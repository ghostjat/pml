<?php

declare(strict_types=1);

namespace Pml\Backends;

use Pml\Dataset;
use Pml\Interfaces\MLBackend;
use Pml\Interfaces\Verbose;
use Pml\NeuralNetwork\Sequential;
use Pml\Tensor;
use Psr\Log\LoggerInterface;

/**
 * Deep-Learning backend wrapping a Sequential neural network.
 *
 * Delegates all computation to the C-native TensorEngine via Sequential.
 * Persistence uses the SafeTensors + JSON bundle format (HF-compatible).
 * LR scheduling, early stopping, and validation are forwarded through
 * Sequential's TrainableWithOptions variadic train() signature.
 */
final class TorchBackend implements MLBackend
{
    public function __construct(private Sequential $model) {}

    // ---- MLBackend -----------------------------------------------------------

    /**
     * @param mixed ...$options  Forwarded to Sequential::train():
     *   int        $epochs      (default 10)
     *   int        $batchSize   (default 32)
     *   Dataset    $validation  (optional)
     *   int        $patience    (default 0 = disabled)
     *   float      $minDelta    (default 1e-4)
     */
    public function fit(Dataset $dataset, mixed ...$options): void
    {
        $this->model->train($dataset, ...$options);
    }

    public function predict(Dataset $dataset): Tensor
    {
        return $this->model->predict($dataset);
    }

    public function isTrained(): bool
    {
        return $this->model->trained();
    }

    public function save(string $path): void
    {
        $this->model->save($path);
    }

    public static function load(string $path): self
    {
        return new self(Sequential::load($path));
    }

    public function backendName(): string { return 'torch'; }

    // ---- Extras --------------------------------------------------------------

    /** Expose the inner model for advanced use (LRScheduler, layer inspection). */
    public function model(): Sequential { return $this->model; }

    /** Attach a PSR-3 logger to the underlying Sequential model. */
    public function setLogger(LoggerInterface $logger): void
    {
        if ($this->model instanceof Verbose) {
            $this->model->setLogger($logger);
        }
    }
}
