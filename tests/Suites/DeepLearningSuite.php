<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\{Tensor, Ops};
use Pml\Training\{AdamW, CrossEntropyLoss};
use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\NeuralNetwork\MLPClassifier;

// ═══════════════════════════════════════════════════════════════════════════
//  DeepLearningSuite — Autograd / backprop correctness tests
//
//  ── Critical Assertion: Loss Monotonically Decreases ─────────────────────
//
//  The single most important property to verify in a differentiable neural
//  network system is that gradient descent actually makes the loss go down.
//  If any link in the chain is broken, the loss will NOT decrease:
//
//    ① Broken forward pass      → logits are garbage → random loss, no decrease
//    ② Broken loss function     → wrong dL/dz        → wrong gradient direction
//    ③ Broken backward pass     → wrong ∂L/∂W        → updates point the wrong way
//    ④ Accumulating gradients   → if zeroGrad() fails, gradients grow without
//       (AdamW::zeroGrad() bug)   bound → update diverges
//    ⑤ Broken optimizer step    → if AdamW::step() doesn't update W, no decrease
//
//  A single "loss[0] > loss[-1]" assertion catches ALL of these failure modes.
//
//  ── Why NOT require strict monotonicity? ────────────────────────────────
//
//  On small, noisy datasets without mini-batching, individual steps can
//  temporarily increase the loss (the optimizer overshoots).  We assert that
//  the loss at the END of training is less than at the START — a much weaker
//  but universally reliable signal that the system is working.
//
//  ── Tests ────────────────────────────────────────────────────────────────
//
//    1. MLPClassifier (high-level API):
//       10-epoch training on Iris; assert loss_curve_[0] > loss_curve_[9].
//
//    2. Low-level autograd graph (optional verification):
//       Manually build W, X, CrossEntropyLoss; run one AdamW step;
//       verify loss decreases after the step.
// ═══════════════════════════════════════════════════════════════════════════

final class DeepLearningSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Deep Learning Autograd Engine', function(TestRunner $r) {

            $iris = DatasetLoader::iris();

            // ── Test 1: MLPClassifier loss_curve_ decreases ───────────
            $r->test('MLPClassifier(hidden=[32], 20 epochs): loss decreases end-to-end', function() use ($r, $iris) {

                // ── Tiny network for fast testing ──────────────────────
                //
                // Architecture:
                //   Input(4) → Linear(4→32) → ReLU → Linear(32→3) → softmax
                //
                // 20 epochs on full Iris (n=150) — small enough to run in < 2 s.
                // learning_rate_init=0.01 is slightly aggressive for a quick
                // visible decrease; weight_decay keeps weights from exploding.
                $mlp = new MLPClassifier(
                    hidden_layer_sizes: [32],
                    activation:         'relu',
                    learning_rate_init: 0.01,
                    max_iter:           20,
                    weight_decay:       1e-4,
                    random_state:       42,
                );

                $mlp->fit($iris['X'], $iris['y']);

                // ── loss_curve_ is populated during fit(): one entry per epoch ─
                //
                // The assertion proves the full forward → backward → AdamW::step
                // → AdamW::zeroGrad → next-forward chain is working correctly:
                //
                //   loss[0]  = loss BEFORE any weight update (first epoch)
                //   loss[19] = loss AFTER 20 gradient steps
                //
                // For a 3-class random-init classifier on Iris:
                //   loss[0] ≈ ln(3) ≈ 1.099   (random uniform prediction)
                //   loss[19] should be noticeably less (< 0.9 typically)
                $r->assertEq(
                    count($mlp->loss_curve_), 20,
                    'loss_curve_ must have one entry per epoch'
                );

                $r->assertLossDecreases(
                    $mlp->loss_curve_,
                    sprintf(
                        'MLPClassifier loss: epoch[0]=%.4f → epoch[19]=%.4f',
                        $mlp->loss_curve_[0],
                        $mlp->loss_curve_[19]
                    )
                );
            });

            // ── Test 2: Low-level autograd manual loop ─────────────────
            $r->test('Low-level autograd: 10 AdamW steps decrease cross-entropy loss', function() use ($r, $iris) {

                // ── Build a minimal 1-layer network manually ───────────
                //
                // Forward: logits = X @ W  (no bias, no activation — simplest possible)
                // Loss:    CrossEntropyLoss(logits, y_int[])
                //
                // This directly tests the primitives:
                //   Ops::matmul()      — tracked matmul with $_backward closure
                //   CrossEntropyLoss   — fused softmax+NLL with analytical dL/dz
                //   AdamW::step()      — m/v moment update + weight decay
                //   AdamW::zeroGrad()  — zero all $param->grad buffers
                //
                $n = $iris['X']->shape[0];   // 150
                $d = $iris['X']->shape[1];   // 4
                $K = 3;                       // 3 classes

                // He initialisation: W ~ N(0, sqrt(2/d))
                mt_srand(99);
                $heStd = sqrt(2.0 / $d);
                $W = new Tensor([$d, $K]);
                $W->requiresGrad = true;
                $W->initGrad();
                for ($i = 0; $i < $d * $K; $i++) {
                    // Box-Muller for normal sample
                    $u1 = max(mt_rand() / mt_getrandmax(), 1e-10);
                    $u2 = mt_rand() / mt_getrandmax();
                    $z  = sqrt(-2.0 * log($u1)) * cos(2.0 * M_PI * $u2);
                    $W->buffer[$i] = (float)($z * $heStd);
                }

                // ── Extract integer targets ────────────────────────────
                $targets = [];
                for ($i = 0; $i < $n; $i++) {
                    $targets[$i] = (int)(float)$iris['y']->buffer[$i];
                }

                // ── Optimizer ─────────────────────────────────────────
                $opt = new AdamW([$W], lr: 0.01, weightDecay: 1e-4);

                // ── Training loop (10 steps) ───────────────────────────
                $lossFirst = null;
                $lossLast  = null;

                for ($step = 0; $step < 10; $step++) {
                    // Forward: logits = X @ W  [n, K]
                    $logits = Ops::matmul($iris['X'], $W);

                    // Loss: fused softmax + NLL cross-entropy
                    // Returns a scalar Tensor with backward closure
                    $criterion = new CrossEntropyLoss();
                    $lossT = $criterion->forward($logits, $targets);

                    $lossVal = (float)$lossT->buffer[0];

                    if ($step === 0) { $lossFirst = $lossVal; }
                    $lossLast = $lossVal;

                    // Backward: populate $W->grad via autograd chain
                    $lossT->backward();

                    // Optimizer step: update W using accumulated grad
                    $opt->step();

                    // Zero gradients: CRITICAL — without this, grads accumulate
                    // and the effective learning rate grows unboundedly
                    $opt->zeroGrad();
                }

                // ── The critical assertion ─────────────────────────────
                //
                // After 10 gradient steps, the cross-entropy loss MUST be lower
                // than at initialisation.  If this fails, one of ①–⑤ is broken.
                $r->assertLossDecreases(
                    [$lossFirst, $lossLast],
                    sprintf('low-level autograd: step[0]=%.4f → step[9]=%.4f', $lossFirst, $lossLast)
                );
            });

            // ── Test 3: Gradient direction sanity ─────────────────────
            $r->test('Single backward step: parameter gradient is non-zero', function() use ($r, $iris) {

                // Verifies that:
                //   (a) backward() actually writes to $W->grad
                //   (b) The gradient is not all-zero (which would mean no signal)
                //
                // A gradient of exactly zero on all elements would indicate
                // that either the forward pass produced a constant output (no
                // gradient signal flows back) or grad buffers weren't allocated.

                $d = $iris['X']->shape[1];  // 4
                $K = 3;

                mt_srand(0);
                $W = new Tensor([$d, $K]);
                $W->requiresGrad = true;
                $W->initGrad();

                // Small non-zero random initialisation
                for ($i = 0; $i < $d * $K; $i++) {
                    $W->buffer[$i] = (float)((mt_rand() / mt_getrandmax() - 0.5) * 0.1);
                }

                $targets = [];
                for ($i = 0; $i < $iris['X']->shape[0]; $i++) {
                    $targets[$i] = (int)(float)$iris['y']->buffer[$i];
                }

                $logits = Ops::matmul($iris['X'], $W);
                $criterion = new CrossEntropyLoss();
                $lossT = $criterion->forward($logits, $targets);
                $lossT->backward();

                // Check that at least one gradient element is non-zero
                $gradNonZero = false;
                for ($i = 0; $i < $d * $K; $i++) {
                    if (abs((float)$W->grad[$i]) > 1e-8) {
                        $gradNonZero = true;
                        break;
                    }
                }

                if (!$gradNonZero) {
                    throw new \RuntimeException(
                        'All gradient elements are zero after backward() — backpropagation is broken.'
                    );
                }
                // If we get here without throwing, the test passes silently.
            });

        });
    }
}
