<?php

declare(strict_types=1);

namespace Pml\NeuralNetwork\Layers;

/**
 * Marker interface for layers that have distinct training / inference behaviour
 * (e.g. Dropout, BatchNormalization).
 *
 * Sequential::setTrainingMode() uses this for a type-safe dispatch instead of
 * the duck-typed `property_exists()` pattern.
 */
interface HasTrainingMode
{
    public function setTraining(bool $mode): void;
}
