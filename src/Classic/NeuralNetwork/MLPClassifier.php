<?php

declare(strict_types=1);

namespace Pml\Classic\NeuralNetwork;

use Pml\{Tensor, BlasEngine, Ops};
use Pml\Training\{CrossEntropyLoss, AdamW};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  MLPClassifier — sklearn.neural_network.MLPClassifier
//
//  Multi-Layer Perceptron classifier built on the existing Pml autograd
//  engine (Ops::matmul, Ops::add) plus a local reluWithGrad and addBias
//  helper that follow the same computational-graph pattern.
//
//  ── Architecture ────────────────────────────────────────────────────────
//
//  Input → [Lin₀ → ReLU] → [Lin₁ → ReLU] → … → [LinL] → logits
//
//  Each linear layer: Z = X @ W + b
//    W[in, out] — weight matrix, He-initialised (good for ReLU)
//    b[out]     — bias vector,  zero-initialised
//
//  No activation is applied after the final linear layer (raw logits are
//  fed into CrossEntropyLoss which fuses softmax + NLL internally).
//
//  ── Autograd Ops Used ────────────────────────────────────────────────────
//
//  Ops::matmul(X, W)      — tracked matrix multiply [m,k] @ [k,n] → [m,n]
//  addBias(Z, b)           — tracked broadcast bias add [m,n] + [n] → [m,n]
//  reluWithGrad(Z)         — tracked element-wise ReLU
//
//  Both addBias() and reluWithGrad() are implemented below following the
//  same $_backward closure pattern as Ops::matmul and Ops::add.
//
//  ── Training ─────────────────────────────────────────────────────────────
//
//  For each epoch:
//    1. Forward pass → logits [n_samples, n_classes]
//    2. CrossEntropyLoss::forward(logits, int_targets) → scalar loss
//    3. loss->backward()  — populates $param->grad for every W and b
//    4. AdamW::step()     — weight update with decoupled weight decay
//    5. AdamW::zeroGrad() — clear grad buffers for next iteration
//
//  ── Inference ─────────────────────────────────────────────────────────────
//
//  predict() temporarily disables requiresGrad on all weight/bias tensors
//  before the forward pass so no computational graph is built — this avoids
//  allocating backward closures and parent lists that would be GC'd anyway.
//  requiresGrad is restored after the forward pass.
// ═══════════════════════════════════════════════════════════════════════════

final class MLPClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /** @var Tensor[]  Weight matrices [W_0, W_1, ..., W_L] */
    public array $weights_ = [];

    /** @var Tensor[]  Bias vectors   [b_0, b_1, ..., b_L] */
    public array $biases_  = [];

    /** @var int[]    Distinct class labels seen during fit(), sorted. */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    /** Per-epoch training loss (useful for diagnostics). */
    public array $loss_curve_ = [];

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int[]  $hidden_layer_sizes  Sizes of hidden layers. e.g. [100, 50].
     * @param string $activation          Hidden layer activation: 'relu', 'tanh', 'logistic'.
     * @param float  $learning_rate_init  Initial learning rate for AdamW.
     * @param int    $max_iter            Maximum number of training epochs.
     * @param float  $weight_decay        L2 weight decay (decoupled, AdamW style).
     * @param int    $random_state        RNG seed for weight initialisation.
     */
    public function __construct(
        private readonly array  $hidden_layer_sizes = [100],
        private readonly string $activation         = 'relu',
        private readonly float  $learning_rate_init = 1e-3,
        private readonly int    $max_iter           = 200,
        private readonly float  $weight_decay       = 1e-4,
        private readonly int    $random_state       = 0,
    ) {
        if (!in_array($activation, ['relu', 'tanh', 'logistic'], true)) {
            throw new \InvalidArgumentException(
                "MLPClassifier: unknown activation '{$activation}'. Use 'relu', 'tanh', or 'logistic'."
            );
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Build the network, then train for max_iter epochs using AdamW.
     *
     * @param Tensor      $X  Feature matrix [n_samples, n_features]
     * @param Tensor|null $y  Label vector   [n_samples] — required for classification
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('MLPClassifier: y must be provided for classification.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('MLPClassifier: X must be 2-D [n_samples, n_features].');
        }

        [$n, $d]              = $X->shape;
        $this->n_features_in_ = $d;

        // ── Discover classes ──────────────────────────────────────────
        $labelSet = [];
        for ($i = 0; $i < $n; $i++) {
            $lbl = (int)(float)$y->buffer[$i];
            $labelSet[$lbl] = true;
        }
        $classArr = array_keys($labelSet);
        sort($classArr);
        $this->classes_   = $classArr;
        $this->n_classes_ = count($classArr);
        $classToIdx       = array_flip($classArr);   // label → network output column

        // ── Build integer target array for CrossEntropyLoss ────────────
        $targets = [];
        for ($i = 0; $i < $n; $i++) {
            $targets[$i] = $classToIdx[(int)(float)$y->buffer[$i]];
        }

        // ── Initialise weights ─────────────────────────────────────────
        //
        // Layer sizes: [n_features, h_0, h_1, ..., h_L, n_classes]
        mt_srand($this->random_state);
        $layerSizes   = array_merge([$d], $this->hidden_layer_sizes, [$this->n_classes_]);
        $nLayers      = count($layerSizes) - 1;
        $this->weights_ = [];
        $this->biases_  = [];

        for ($l = 0; $l < $nLayers; $l++) {
            $fanIn  = $layerSizes[$l];
            $fanOut = $layerSizes[$l + 1];

            // He initialisation: std = sqrt(2 / fan_in)  — optimal for ReLU
            $w = Tensor::randn([$fanIn, $fanOut], 0.0, sqrt(2.0 / $fanIn));
            $w->requiresGrad = true;

            $b = Tensor::zeros([$fanOut]);
            $b->requiresGrad = true;

            $this->weights_[$l] = $w;
            $this->biases_[$l]  = $b;
        }

        // ── Collect all parameters for AdamW ──────────────────────────
        $allParams = array_merge($this->weights_, $this->biases_);
        $optimizer = new AdamW(
            params:      $allParams,
            lr:          $this->learning_rate_init,
            weightDecay: $this->weight_decay,
        );
        $criterion = new CrossEntropyLoss();

        // ── Training loop ──────────────────────────────────────────────
        $this->loss_curve_ = [];

        for ($epoch = 0; $epoch < $this->max_iter; $epoch++) {
            // ── Forward pass ───────────────────────────────────────────
            $logits = $this->forwardPass($X, buildGraph: true);

            // ── Loss (fused softmax + cross-entropy with autograd) ─────
            $loss = $criterion->forward($logits, $targets);

            $this->loss_curve_[] = (float) $loss->buffer[0];

            // ── Backward pass ──────────────────────────────────────────
            $loss->backward();

            // ── Optimizer step + gradient zero ─────────────────────────
            $optimizer->step();
            $optimizer->zeroGrad();
        }

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels for samples in X.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples]  integer class labels (stored as float32)
     */
    public function predict(Tensor $X): Tensor
    {
        $this->checkFitted();
        $proba = $this->predict_proba($X);
        [$m, $nC] = $proba->shape;
        $out = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $bestC = 0;
            $bestV = (float) $proba->buffer[$i * $nC];
            for ($c = 1; $c < $nC; $c++) {
                $v = (float) $proba->buffer[$i * $nC + $c];
                if ($v > $bestV) { $bestV = $v; $bestC = $c; }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestC];
        }

        return $out;
    }

    /**
     * Return class probability estimates via softmax of the output logits.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_classes]
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        // Disable grad tracking for inference — no graph is built
        foreach ($this->weights_ as $w) { $w->requiresGrad = false; }
        foreach ($this->biases_  as $b) { $b->requiresGrad = false; }

        $logits = $this->forwardPass($X, buildGraph: false);

        // Re-enable grad for future training calls
        foreach ($this->weights_ as $w) { $w->requiresGrad = true; }
        foreach ($this->biases_  as $b) { $b->requiresGrad = true; }

        // Apply row-wise softmax to convert logits → probabilities
        $proba = $logits->clone();
        Ops::softmaxInPlace($proba);
        return $proba;
    }

    // ── Internal helpers ───────────────────────────────────────────────────

    /**
     * Run the feedforward pass through all layers.
     *
     * @param Tensor $X          [n_samples, n_features]
     * @param bool   $buildGraph true during training (requiresGrad on params),
     *                           false during inference.
     * @return Tensor            [n_samples, n_classes]  raw logits
     */
    private function forwardPass(Tensor $X, bool $buildGraph = true): Tensor
    {
        $nLayers = count($this->weights_);
        $current = $X;

        for ($l = 0; $l < $nLayers; $l++) {
            // ── Linear: Z = X @ W + b ──────────────────────────────────
            //
            // Ops::matmul builds the backward closure only when at least one
            // operand has requiresGrad=true.  During inference both W and b
            // have requiresGrad=false, so no graph is constructed.
            $Z = Ops::matmul($current, $this->weights_[$l]);
            $Z = self::addBias($Z, $this->biases_[$l]);

            // ── Activation (all layers except last) ────────────────────
            if ($l < $nLayers - 1) {
                $Z = $this->applyActivation($Z);
            }

            $current = $Z;
        }

        return $current;  // raw logits [n_samples, n_classes]
    }

    /**
     * Apply the configured activation function to a tensor.
     *
     * Uses the autograd-tracked version of ReLU when the input has
     * requiresGrad=true, the plain version otherwise.
     */
    private function applyActivation(Tensor $Z): Tensor
    {
        return match ($this->activation) {
            'relu'    => self::reluWithGrad($Z),
            'tanh'    => self::tanhWithGrad($Z),
            'logistic'=> self::sigmoidWithGrad($Z),
        };
    }

    // ── Differentiable ops (autograd-tracked) ─────────────────────────────

    /**
     * Broadcast bias addition: Z[m,n] + b[n] → C[m,n], with backward.
     *
     * Forward:  C[i,j] = Z[i,j] + b[j]     ∀ i ∈ [0,m), j ∈ [0,n)
     *
     * Backward (chain rule):
     *   dZ[i,j] += dC[i,j]                  (identity, accumulated via saxpy)
     *   db[j]   += Σ_i dC[i,j]             (sum over batch — saxpy per row)
     *
     * The bias gradient accumulates across the batch dimension:
     *   db = dC^T @ 1_m   where 1_m is a vector of ones of length m.
     * Equivalently: saxpy(n, 1.0, dC_row_i, 1, db, 1)  for each row i.
     */
    private static function addBias(Tensor $Z, Tensor $b): Tensor
    {
        [$m, $n] = $Z->shape;
        $blas = BlasEngine::get()->ffi;

        // Copy Z → C, then add b to each row using saxpy
        $C = $Z->clone();
        for ($i = 0; $i < $m; $i++) {
            $rowPtr = \FFI::cast('float*', \FFI::addr($C->buffer[$i * $n]));
            $blas->cblas_saxpy($n, 1.0, $b->buffer, 1, $rowPtr, 1);
        }

        // ── Build autograd graph ──────────────────────────────────────
        if ($Z->requiresGrad || $b->requiresGrad) {
            $C->requiresGrad = true;
            $C->_prev        = [$Z, $b];

            $C->_backward = static function() use ($Z, $b, $C, $m, $n): void {
                $ffi = BlasEngine::get()->ffi;

                // dZ += dC  (identity pass-through, same shape)
                if ($Z->requiresGrad) {
                    $Z->initGrad();
                    $ffi->cblas_saxpy($C->size, 1.0, $C->grad, 1, $Z->grad, 1);
                }

                // db[j] += Σ_i dC[i,j]  → one saxpy per batch row
                if ($b->requiresGrad) {
                    $b->initGrad();
                    for ($i = 0; $i < $m; $i++) {
                        $rowGradPtr = \FFI::cast('float*', \FFI::addr($C->grad[$i * $n]));
                        $ffi->cblas_saxpy($n, 1.0, $rowGradPtr, 1, $b->grad, 1);
                    }
                }
            };
        }

        return $C;
    }

    /**
     * ReLU with autograd: out = max(0, x), backward: dout * (x > 0).
     *
     * Forward:  out[i] = max(0, x[i])
     * Backward: dx[i] += dout[i]  if out[i] > 0  else 0
     *
     * The ReLU mask is re-derived from out[i] > 0 (equivalent to x[i] > 0)
     * without storing a separate mask buffer — out is already in memory.
     */
    private static function reluWithGrad(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $v = (float) $x->buffer[$i];
            $out->buffer[$i] = $v > 0.0 ? $v : 0.0;
        }

        if ($x->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x];

            $out->_backward = static function() use ($x, $out): void {
                if (!$x->requiresGrad) { return; }
                $x->initGrad();
                for ($i = 0; $i < $x->size; $i++) {
                    // Gate gradient: pass through only where output > 0 (i.e. x > 0)
                    if ((float) $out->buffer[$i] > 0.0) {
                        $x->grad[$i] = (float) $x->grad[$i] + (float) $out->grad[$i];
                    }
                }
            };
        }

        return $out;
    }

    /**
     * Tanh with autograd.
     *
     * Forward:  out[i] = tanh(x[i])
     * Backward: dx[i] += dout[i] * (1 − out[i]²)   [tanh derivative]
     */
    private static function tanhWithGrad(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $out->buffer[$i] = tanh((float) $x->buffer[$i]);
        }

        if ($x->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x];

            $out->_backward = static function() use ($x, $out): void {
                if (!$x->requiresGrad) { return; }
                $x->initGrad();
                for ($i = 0; $i < $x->size; $i++) {
                    $t = (float) $out->buffer[$i];
                    $x->grad[$i] = (float) $x->grad[$i] + (float) $out->grad[$i] * (1.0 - $t * $t);
                }
            };
        }

        return $out;
    }

    /**
     * Logistic (sigmoid) with autograd.
     *
     * Forward:  out[i] = σ(x[i]) = 1 / (1 + exp(-x[i]))
     * Backward: dx[i] += dout[i] * out[i] * (1 − out[i])
     */
    private static function sigmoidWithGrad(Tensor $x): Tensor
    {
        $out = new Tensor($x->shape);
        for ($i = 0; $i < $x->size; $i++) {
            $out->buffer[$i] = 1.0 / (1.0 + exp(-(float) $x->buffer[$i]));
        }

        if ($x->requiresGrad) {
            $out->requiresGrad = true;
            $out->_prev        = [$x];

            $out->_backward = static function() use ($x, $out): void {
                if (!$x->requiresGrad) { return; }
                $x->initGrad();
                for ($i = 0; $i < $x->size; $i++) {
                    $s = (float) $out->buffer[$i];
                    $x->grad[$i] = (float) $x->grad[$i] + (float) $out->grad[$i] * $s * (1.0 - $s);
                }
            };
        }

        return $out;
    }

    private function checkFitted(): void
    {
        if (empty($this->weights_)) {
            throw new \RuntimeException('MLPClassifier is not fitted. Call fit() first.');
        }
    }
}
