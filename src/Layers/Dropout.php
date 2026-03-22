<?php


declare(strict_types=1);

namespace Pml\Layers;

use Pml\{Tensor};

// ═══════════════════════════════════════════════════════════════════════════
//  DROPOUT (inference mode: identity, training mode: random zero-mask)
// ═══════════════════════════════════════════════════════════════════════════

class Dropout
{
    public function __construct(
        private readonly float $p       = 0.1,
        private bool           $training = false,
    ) {}

    public function setTraining(bool $training): void { $this->training = $training; }

    public function forward(Tensor $x): Tensor
    {
        if (!$this->training || $this->p === 0.0) return $x; // No-op at inference

        $scale = 1.0 / (1.0 - $this->p);
        $out   = $x->clone();
        for ($i = 0; $i < $out->size; $i++) {
            if ((mt_rand() / mt_getrandmax()) < $this->p) {
                $out->buffer[$i] = 0.0;
            } else {
                $out->buffer[$i] = (float)$out->buffer[$i] * $scale;
            }
        }
        return $out;
    }
}
