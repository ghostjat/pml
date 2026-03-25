<?php

declare(strict_types=1);

namespace Pml\RL;

use Pml\{Tensor, BlasEngine};
use Pml\Training\AdamW;

// ═══════════════════════════════════════════════════════════════════════════
//  DQNAgent — Deep Q-Network (Mnih et al. 2015)
//
//  ── Architecture ─────────────────────────────────────────────────────────
//
//  Two identical fully-connected networks (online + target):
//
//    Linear(stateDim → hiddenSize)  →  ReLU
//    Linear(hiddenSize → hiddenSize)  →  ReLU
//    Linear(hiddenSize → actionDim)   →  Q-values  [no activation]
//
//  Kaiming He initialisation for all weight matrices (appropriate for ReLU).
//
//  ── Training Update ──────────────────────────────────────────────────────
//
//  1. Sample a mini-batch {(s, a, r, s', done)} from ReplayBuffer.
//
//  2. Compute TD targets using the TARGET network (no gradient):
//       y_b = r_b + γ · max_{a'} Q_target(s'_b, a') · (1 − done_b)
//
//  3. Compute online Q-values:  Q_online(s_b, :)  [full action vector]
//
//  4. MSE loss over taken actions only:
//       L = (1/B) Σ_b (Q_online(s_b, a_b) − y_b)²
//
//     Gradient (sparse — only action a_b is non-zero):
//       dQ[b, a] = { 2(Q_pred − y_b)/B   if a == a_b
//                  { 0                    otherwise
//
//  5. Backpropagate through the online network using explicit BLAS BPTT.
//
//  6. AdamW optimizer step on online parameters.
//
//  7. Decay ε: ε ← max(ε_min, ε · ε_decay)  (multiplicative schedule).
//
//  8. Every `targetUpdateFreq` training steps: hard-copy online → target
//     using cblas_scopy (Pascanu et al. 2013; Mnih et al. 2015 §4).
//
//  ── ε-Greedy Exploration ──────────────────────────────────────────────────
//
//  selectAction(s):
//    • With probability ε:  return random action  ∈ [0, actionDim)
//    • Else:                return argmax_a Q_online(s, a)
//
//  ── BLAS Matrix Dimensions ───────────────────────────────────────────────
//
//  Forward (batch B, state S, hidden H, actions A):
//
//    h1 [B,H] = x[B,S] @ W1^T[S,H] + b1  — sgemm(NoTrans, Trans, B, H, S)
//    r1 [B,H] = ReLU(h1)
//    h2 [B,H] = r1[B,H] @ W2^T[H,H] + b2 — sgemm(NoTrans, Trans, B, H, H)
//    r2 [B,H] = ReLU(h2)
//    q  [B,A] = r2[B,H] @ W3^T[H,A] + b3 — sgemm(NoTrans, Trans, B, A, H)
//
//  Backward (only online network):
//
//    dW3 [A,H] += dq^T  @ r2  — sgemm(Trans, NoTrans, A, H, B)
//    db3 [A]   += Σ_b dq      — sgemv(Trans, B, A)
//    dr2 [B,H]  = dq   @ W3   — sgemm(NoTrans, NoTrans, B, H, A)
//
//    dh2 = dr2 ⊙ mask2       — element-wise PHP loop
//
//    dW2 [H,H] += dh2^T @ r1 — sgemm(Trans, NoTrans, H, H, B)
//    db2 [H]   += Σ_b dh2    — sgemv(Trans, B, H)
//    dr1 [B,H]  = dh2  @ W2  — sgemm(NoTrans, NoTrans, B, H, H)
//
//    dh1 = dr1 ⊙ mask1       — element-wise PHP loop
//
//    dW1 [H,S] += dh1^T @ x  — sgemm(Trans, NoTrans, H, S, B)
//    db1 [H]   += Σ_b dh1    — sgemv(Trans, B, H)
//    (dx not needed — input states have no learnable parameters)
// ═══════════════════════════════════════════════════════════════════════════

final class DQNAgent
{
    // ── Dimensions ───────────────────────────────────────────────────────

    public readonly int $stateDim;
    public readonly int $actionDim;
    public readonly int $hiddenSize;

    // ── Online network weights ────────────────────────────────────────────

    public readonly Tensor $W1;
    public readonly Tensor $b1;
    public readonly Tensor $W2;
    public readonly Tensor $b2;
    public readonly Tensor $W3;
    public readonly Tensor $b3;

    // ── Target network weights (no requiresGrad) ──────────────────────────

    private readonly Tensor $tW1;
    private readonly Tensor $tb1;
    private readonly Tensor $tW2;
    private readonly Tensor $tb2;
    private readonly Tensor $tW3;
    private readonly Tensor $tb3;

    // ── Exploration state ─────────────────────────────────────────────────

    private float $epsilon;

    // ── Replay buffer ─────────────────────────────────────────────────────

    private readonly ReplayBuffer $buffer;

    // ── Training hyper-parameters ──────────────────────────────────────────

    private readonly float $epsilonMin;
    private readonly float $epsilonDecay;
    private readonly float $gamma;
    private readonly int   $batchSize;
    private readonly int   $targetUpdateFreq;

    // ── Optimizer ─────────────────────────────────────────────────────────

    private readonly AdamW $optimizer;

    // ── Step counter (drives target update schedule) ──────────────────────

    private int $trainSteps = 0;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int          $stateDim         Observation vector dimension.
     * @param int          $actionDim        Number of discrete actions.
     * @param ReplayBuffer $replay           Shared experience replay buffer.
     * @param int          $hiddenSize       Width of both hidden layers (default 64).
     * @param float        $epsilonStart     Initial exploration rate (default 1.0).
     * @param float        $epsilonMin       Minimum exploration rate (default 0.01).
     * @param float        $epsilonDecay     Multiplicative decay per train step (default 0.995).
     * @param float        $gamma            Discount factor γ (default 0.99).
     * @param int          $batchSize        Mini-batch size for each training step.
     * @param int          $targetUpdateFreq Hard-copy online→target every N train steps.
     * @param float        $lr               AdamW learning rate.
     * @param float        $weightDecay      AdamW weight decay (L2 regularisation).
     */
    public function __construct(
        int          $stateDim,
        int          $actionDim,
        ReplayBuffer $replay,
        int          $hiddenSize       = 64,
        float        $epsilonStart     = 1.0,
        float        $epsilonMin       = 0.01,
        float        $epsilonDecay     = 0.995,
        float        $gamma            = 0.99,
        int          $batchSize        = 64,
        int          $targetUpdateFreq = 100,
        float        $lr               = 1e-3,
        float        $weightDecay      = 1e-4,
    ) {
        $this->stateDim         = $stateDim;
        $this->actionDim        = $actionDim;
        $this->hiddenSize       = $hiddenSize;
        $this->epsilon          = $epsilonStart;
        $this->epsilonMin       = $epsilonMin;
        $this->epsilonDecay     = $epsilonDecay;
        $this->gamma            = $gamma;
        $this->batchSize        = $batchSize;
        $this->targetUpdateFreq = $targetUpdateFreq;
        $this->buffer           = $replay;

        // Kaiming He init: std = sqrt(2 / fan_in) for ReLU networks
        $std1 = sqrt(2.0 / max(1, $stateDim));
        $std2 = sqrt(2.0 / max(1, $hiddenSize));

        // Online network
        $this->W1 = Tensor::randn([$hiddenSize, $stateDim],   0.0, $std1);
        $this->b1 = Tensor::zeros([$hiddenSize]);
        $this->W2 = Tensor::randn([$hiddenSize, $hiddenSize], 0.0, $std2);
        $this->b2 = Tensor::zeros([$hiddenSize]);
        $this->W3 = Tensor::randn([$actionDim,  $hiddenSize], 0.0, $std2);
        $this->b3 = Tensor::zeros([$actionDim]);

        foreach ($this->parameters() as $p) {
            $p->requiresGrad = true;
        }

        // Target network — same shape, no grad, synced below
        $this->tW1 = Tensor::randn([$hiddenSize, $stateDim],   0.0, $std1);
        $this->tb1 = Tensor::zeros([$hiddenSize]);
        $this->tW2 = Tensor::randn([$hiddenSize, $hiddenSize], 0.0, $std2);
        $this->tb2 = Tensor::zeros([$hiddenSize]);
        $this->tW3 = Tensor::randn([$actionDim,  $hiddenSize], 0.0, $std2);
        $this->tb3 = Tensor::zeros([$actionDim]);

        // Initialise target = online
        $this->updateTarget();

        $this->optimizer = new AdamW(
            $this->parameters(),
            lr: $lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weightDecay: $weightDecay,
        );
    }

    // ── Public API ────────────────────────────────────────────────────────

    /**
     * Select an action using the ε-greedy policy.
     *
     * @param  float[] $state  Current observation vector (length = stateDim).
     * @return int             Action index ∈ [0, actionDim).
     */
    public function selectAction(array $state): int
    {
        // Exploration: random action with probability ε
        if ((mt_rand() / mt_getrandmax()) < $this->epsilon) {
            return mt_rand(0, $this->actionDim - 1);
        }

        // Exploitation: argmax Q_online(state, :)
        $x = $this->stateToTensor($state);
        [$q,] = $this->mlpForward(
            $x, $this->W1, $this->b1, $this->W2, $this->b2, $this->W3, $this->b3
        );

        $best  = 0;
        $bestV = (float) $q->buffer[0];
        for ($a = 1; $a < $this->actionDim; $a++) {
            $v = (float) $q->buffer[$a];
            if ($v > $bestV) { $bestV = $v; $best = $a; }
        }
        return $best;
    }

    /**
     * Store a transition in the replay buffer.
     *
     * @param float[] $state
     * @param int     $action
     * @param float   $reward
     * @param float[] $nextState
     * @param bool    $done
     */
    public function remember(
        array $state, int $action, float $reward, array $nextState, bool $done
    ): void {
        $this->buffer->push($state, $action, $reward, $nextState, $done);
    }

    /**
     * Sample a mini-batch and perform one gradient update.
     *
     * Returns null (skips update) if the buffer is not yet warm (< batchSize).
     *
     * @return float|null  Mean squared TD error for the batch, or null.
     */
    public function train(): ?float
    {
        if ($this->buffer->size() < $this->batchSize) {
            return null;
        }

        $batch = $this->buffer->sample($this->batchSize);
        $B     = $this->batchSize;
        $A     = $this->actionDim;

        // ── Pack states into Tensors ───────────────────────────────────────
        $stateTensor     = $this->packStates($batch['states']);
        $nextStateTensor = $this->packStates($batch['nextStates']);

        // ── TD targets via target network (no gradient) ───────────────────
        [$qNext,] = $this->mlpForward(
            $nextStateTensor,
            $this->tW1, $this->tb1, $this->tW2, $this->tb2, $this->tW3, $this->tb3
        );

        $targets = [];
        for ($b = 0; $b < $B; $b++) {
            $off  = $b * $A;
            $maxQ = (float) $qNext->buffer[$off];
            for ($a = 1; $a < $A; $a++) {
                $v = (float) $qNext->buffer[$off + $a];
                if ($v > $maxQ) $maxQ = $v;
            }
            $notDone  = $batch['dones'][$b] ? 0.0 : 1.0;
            $targets[] = (float) $batch['rewards'][$b] + $this->gamma * $maxQ * $notDone;
        }

        // ── Online Q-values ───────────────────────────────────────────────
        [$qOnline, $fwdCache] = $this->mlpForward(
            $stateTensor,
            $this->W1, $this->b1, $this->W2, $this->b2, $this->W3, $this->b3
        );

        // ── MSE loss + sparse gradient (only taken action per sample) ─────
        $dq  = new Tensor([$B, $A]);   // zero-initialised
        $mse = 0.0;

        for ($b = 0; $b < $B; $b++) {
            $a      = $batch['actions'][$b];
            $qPred  = (float) $qOnline->buffer[$b * $A + $a];
            $err    = $qPred - $targets[$b];
            $mse   += $err * $err;
            $dq->buffer[$b * $A + $a] = 2.0 * $err / $B;
        }
        $mse /= $B;

        // ── Backward pass + optimizer ──────────────────────────────────────
        $this->zeroGrad();
        $this->mlpBackward($dq, $fwdCache);
        $this->optimizer->step();

        // ── Decay ε ───────────────────────────────────────────────────────
        $this->epsilon = max($this->epsilonMin, $this->epsilon * $this->epsilonDecay);

        // ── Periodic target network hard-update ───────────────────────────
        $this->trainSteps++;
        if ($this->trainSteps % $this->targetUpdateFreq === 0) {
            $this->updateTarget();
        }

        return $mse;
    }

    /**
     * Hard-copy online network weights to target network via cblas_scopy.
     *
     * cblas_scopy(n, src, incx=1, dst, incy=1) copies n float32 elements
     * from `src` to `dst` at unit stride.  This is the DQN "hard update"
     * (Mnih et al. 2015).  A "soft update" (Polyak averaging) would use
     * cblas_sscal + cblas_saxpy, but hard updates are standard for CartPole.
     */
    public function updateTarget(): void
    {
        $blas = BlasEngine::get()->ffi;

        $pairs = [
            [$this->W1, $this->tW1],
            [$this->b1, $this->tb1],
            [$this->W2, $this->tW2],
            [$this->b2, $this->tb2],
            [$this->W3, $this->tW3],
            [$this->b3, $this->tb3],
        ];

        foreach ($pairs as [$src, $dst]) {
            // cblas_scopy(n, src_buf, incx, dst_buf, incy)
            $blas->cblas_scopy($src->size, $src->buffer, 1, $dst->buffer, 1);
        }
    }

    /**
     * Current exploration rate ε.
     */
    public function getEpsilon(): float
    {
        return $this->epsilon;
    }

    /**
     * Number of completed training steps.
     */
    public function getTrainSteps(): int
    {
        return $this->trainSteps;
    }

    /**
     * All learnable (online network) parameters.
     * @return Tensor[]
     */
    public function parameters(): array
    {
        return [$this->W1, $this->b1, $this->W2, $this->b2, $this->W3, $this->b3];
    }

    // ── Private helpers ───────────────────────────────────────────────────

    /**
     * MLP forward pass (shared by online and target networks).
     *
     * Uses fully explicit BLAS sgemm calls; bias addition is a PHP loop
     * (O(B·H) — not on the hot path relative to sgemm).
     *
     * @param  Tensor $x   Input [B, stateDim].
     * @param  Tensor $W1  [hiddenSize, stateDim]
     * @param  Tensor $b1  [hiddenSize]
     * @param  Tensor $W2  [hiddenSize, hiddenSize]
     * @param  Tensor $b2  [hiddenSize]
     * @param  Tensor $W3  [actionDim, hiddenSize]
     * @param  Tensor $b3  [actionDim]
     * @return array{Tensor, array}  [q [B,A], cache for backward]
     */
    private function mlpForward(
        Tensor $x,
        Tensor $W1, Tensor $b1,
        Tensor $W2, Tensor $b2,
        Tensor $W3, Tensor $b3,
    ): array {
        $blas = BlasEngine::get()->ffi;
        $B    = $x->shape[0];
        $S    = $this->stateDim;
        $H    = $this->hiddenSize;
        $A    = $this->actionDim;

        // ── Layer 1: h1 [B,H] = x[B,S] @ W1^T + b1 ───────────────────────
        //   sgemm(RowMajor, NoTrans, Trans, M=B, N=H, K=S,
        //          A=x[B×S], lda=S,   B=W1[H×S], ldb=S,   C=h1[B×H], ldc=H)
        $h1 = new Tensor([$B, $H]);
        $blas->cblas_sgemm(
            101, 111, 112, $B, $H, $S,
            1.0, $x->buffer,  $S,
                 $W1->buffer, $S,
            0.0, $h1->buffer, $H
        );
        for ($bi = 0; $bi < $B; $bi++) {
            for ($hi = 0; $hi < $H; $hi++) {
                $h1->buffer[$bi * $H + $hi] =
                    (float) $h1->buffer[$bi * $H + $hi] + (float) $b1->buffer[$hi];
            }
        }

        // ── ReLU1 ──────────────────────────────────────────────────────────
        $r1    = new Tensor([$B, $H]);
        $mask1 = new Tensor([$B, $H]);
        for ($i = 0; $i < $B * $H; $i++) {
            $v = (float) $h1->buffer[$i];
            if ($v > 0.0) { $r1->buffer[$i] = $v; $mask1->buffer[$i] = 1.0; }
        }

        // ── Layer 2: h2 [B,H] = r1[B,H] @ W2^T + b2 ──────────────────────
        $h2 = new Tensor([$B, $H]);
        $blas->cblas_sgemm(
            101, 111, 112, $B, $H, $H,
            1.0, $r1->buffer, $H,
                 $W2->buffer, $H,
            0.0, $h2->buffer, $H
        );
        for ($bi = 0; $bi < $B; $bi++) {
            for ($hi = 0; $hi < $H; $hi++) {
                $h2->buffer[$bi * $H + $hi] =
                    (float) $h2->buffer[$bi * $H + $hi] + (float) $b2->buffer[$hi];
            }
        }

        // ── ReLU2 ──────────────────────────────────────────────────────────
        $r2    = new Tensor([$B, $H]);
        $mask2 = new Tensor([$B, $H]);
        for ($i = 0; $i < $B * $H; $i++) {
            $v = (float) $h2->buffer[$i];
            if ($v > 0.0) { $r2->buffer[$i] = $v; $mask2->buffer[$i] = 1.0; }
        }

        // ── Layer 3: q [B,A] = r2[B,H] @ W3^T + b3 ───────────────────────
        //   sgemm(RowMajor, NoTrans, Trans, M=B, N=A, K=H)
        $q = new Tensor([$B, $A]);
        $blas->cblas_sgemm(
            101, 111, 112, $B, $A, $H,
            1.0, $r2->buffer, $H,
                 $W3->buffer, $H,
            0.0, $q->buffer,  $A
        );
        for ($bi = 0; $bi < $B; $bi++) {
            for ($ai = 0; $ai < $A; $ai++) {
                $q->buffer[$bi * $A + $ai] =
                    (float) $q->buffer[$bi * $A + $ai] + (float) $b3->buffer[$ai];
            }
        }

        return [$q, compact('x', 'r1', 'mask1', 'r2', 'mask2')];
    }

    /**
     * MLP backward pass — accumulates gradients into online network params.
     *
     * @param Tensor $dq     Gradient w.r.t. Q-values [B, actionDim].
     * @param array  $cache  Cache returned by mlpForward().
     */
    private function mlpBackward(Tensor $dq, array $cache): void
    {
        $blas = BlasEngine::get()->ffi;

        $x     = $cache['x'];
        $r1    = $cache['r1'];
        $mask1 = $cache['mask1'];
        $r2    = $cache['r2'];
        $mask2 = $cache['mask2'];

        $B = $x->shape[0];
        $S = $this->stateDim;
        $H = $this->hiddenSize;
        $A = $this->actionDim;

        // Ensure gradient buffers are allocated
        $this->W3->initGrad();
        $this->b3->initGrad();
        $this->W2->initGrad();
        $this->b2->initGrad();
        $this->W1->initGrad();
        $this->b1->initGrad();

        $onesB = Tensor::ones([$B]);

        // ── Layer 3 backward ──────────────────────────────────────────────

        // dW3 [A,H] += dq^T [A,B] @ r2 [B,H]
        //   sgemm(RowMajor, Trans, NoTrans, M=A, N=H, K=B,
        //          A=dq[B×A], lda=A,  B=r2[B×H], ldb=H,  C=dW3[A×H], ldc=H)
        $blas->cblas_sgemm(
            101, 112, 111, $A, $H, $B,
            1.0, $dq->buffer,       $A,
                 $r2->buffer,       $H,
            1.0, $this->W3->grad,   $H
        );
        // db3 [A] += Σ_b dq[b,:] — sgemv(RowMajor, Trans, M=B, N=A)
        $blas->cblas_sgemv(
            101, 112, $B, $A,
            1.0, $dq->buffer, $A, $onesB->buffer, 1,
            1.0, $this->b3->grad, 1
        );
        // dr2 [B,H] = dq [B,A] @ W3 [A,H]
        //   sgemm(RowMajor, NoTrans, NoTrans, M=B, N=H, K=A)
        $dr2 = new Tensor([$B, $H]);
        $blas->cblas_sgemm(
            101, 111, 111, $B, $H, $A,
            1.0, $dq->buffer,        $A,
                 $this->W3->buffer,  $H,
            0.0, $dr2->buffer,       $H
        );

        // ── ReLU2 backward ────────────────────────────────────────────────
        $dh2 = new Tensor([$B, $H]);
        for ($i = 0; $i < $B * $H; $i++) {
            $dh2->buffer[$i] = (float) $dr2->buffer[$i] * (float) $mask2->buffer[$i];
        }

        // ── Layer 2 backward ──────────────────────────────────────────────

        // dW2 [H,H] += dh2^T [H,B] @ r1 [B,H]
        $blas->cblas_sgemm(
            101, 112, 111, $H, $H, $B,
            1.0, $dh2->buffer,      $H,
                 $r1->buffer,       $H,
            1.0, $this->W2->grad,   $H
        );
        // db2 [H] += Σ_b dh2[b,:]
        $blas->cblas_sgemv(
            101, 112, $B, $H,
            1.0, $dh2->buffer, $H, $onesB->buffer, 1,
            1.0, $this->b2->grad, 1
        );
        // dr1 [B,H] = dh2 [B,H] @ W2 [H,H]
        $dr1 = new Tensor([$B, $H]);
        $blas->cblas_sgemm(
            101, 111, 111, $B, $H, $H,
            1.0, $dh2->buffer,       $H,
                 $this->W2->buffer,  $H,
            0.0, $dr1->buffer,       $H
        );

        // ── ReLU1 backward ────────────────────────────────────────────────
        $dh1 = new Tensor([$B, $H]);
        for ($i = 0; $i < $B * $H; $i++) {
            $dh1->buffer[$i] = (float) $dr1->buffer[$i] * (float) $mask1->buffer[$i];
        }

        // ── Layer 1 backward ──────────────────────────────────────────────

        // dW1 [H,S] += dh1^T [H,B] @ x [B,S]
        $blas->cblas_sgemm(
            101, 112, 111, $H, $S, $B,
            1.0, $dh1->buffer,      $H,
                 $x->buffer,        $S,
            1.0, $this->W1->grad,   $S
        );
        // db1 [H] += Σ_b dh1[b,:]
        $blas->cblas_sgemv(
            101, 112, $B, $H,
            1.0, $dh1->buffer, $H, $onesB->buffer, 1,
            1.0, $this->b1->grad, 1
        );
        // dx not computed — environment observations are not differentiable parameters
    }

    /**
     * Zero gradients for all online network parameters.
     */
    private function zeroGrad(): void
    {
        foreach ($this->parameters() as $p) {
            $p->zeroGrad();
        }
    }

    /**
     * Wrap a single state vector as a [1, stateDim] Tensor.
     *
     * @param float[] $state
     */
    private function stateToTensor(array $state): Tensor
    {
        $t = new Tensor([1, $this->stateDim]);
        for ($i = 0; $i < $this->stateDim; $i++) {
            $t->buffer[$i] = (float) $state[$i];
        }
        return $t;
    }

    /**
     * Pack a list of state vectors into a [B, stateDim] Tensor.
     *
     * @param  float[][] $states  List of B state vectors.
     */
    private function packStates(array $states): Tensor
    {
        $B   = count($states);
        $S   = $this->stateDim;
        $t   = new Tensor([$B, $S]);
        $off = 0;
        foreach ($states as $s) {
            for ($i = 0; $i < $S; $i++) {
                $t->buffer[$off++] = (float) $s[$i];
            }
        }
        return $t;
    }
}
