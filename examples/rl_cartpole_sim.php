<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/rl_cartpole_sim.php — DQN on the CartPole-v1 Simulator
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Trains a Deep Q-Network (DQN, Mnih et al. 2015) to balance a pole on a
 * simulated cart using the classic CartPole physics environment.
 *
 * ── Environment ───────────────────────────────────────────────────────────
 *
 *   State  : [x, ẋ, θ, θ̇]  (cart position, cart velocity,
 *                              pole angle, pole angular velocity)
 *   Actions: 0 = push left  (force = −10 N)
 *            1 = push right (force = +10 N)
 *   Reward : +1.0 for every step the pole stays upright
 *   Done   : |θ| > 12°  OR  |x| > 2.4  OR  step ≥ 500
 *
 *   Physics: Euler integration of the nonlinear rigid-body CartPole ODEs
 *   (Barto et al. 1983 / OpenAI Gym reference equations).
 *
 * ── DQN Setup ─────────────────────────────────────────────────────────────
 *
 *   Network : Linear(4→64) → ReLU → Linear(64→64) → ReLU → Linear(64→2)
 *   Target  : Hard-copied from online network every 100 training steps
 *             via cblas_scopy.
 *   Replay  : Circular buffer, capacity 10 000.
 *   Loss    : MSE on Q(s,a) vs Bellman target, sparse gradient (taken action).
 *   Optim.  : AdamW(lr=1e-3, wd=1e-4).
 *   ε-greedy: ε = 1.0 → 0.01, multiplicative decay 0.995 per train step.
 *
 * ── Convergence expectation ───────────────────────────────────────────────
 *
 *   The agent is considered "solved" when the mean episode reward over the
 *   last 50 episodes exceeds 475 (95% of the max 500 steps).  On a typical
 *   run this occurs around episode 250–400.
 *
 * Usage:
 *   php examples/rl_cartpole_sim.php
 * ════════════════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../vendor/autoload.php';

use Pml\RL\{ReplayBuffer, DQNAgent};

// ─── Hyper-parameters ─────────────────────────────────────────────────────

const N_EPISODES      = 500;
const MAX_STEPS       = 500;    // per episode
const REPLAY_CAPACITY = 10_000;
const BATCH_SIZE_RL   = 64;
const HIDDEN_RL       = 64;
const LR_RL           = 1e-3;
const WD_RL           = 1e-4;
const GAMMA_RL        = 0.99;
const EPS_START       = 1.0;
const EPS_MIN         = 0.01;
const EPS_DECAY       = 0.995;
const TARGET_UPDATE   = 100;    // train steps between hard target copies
const WARMUP_STEPS    = 500;    // fill replay before first gradient update
const SOLVE_THRESHOLD = 475.0;  // mean reward over last 50 episodes
const REPORT_EVERY    = 50;     // print a progress row every N episodes

mt_srand(1337);

// ─── CartPole environment ─────────────────────────────────────────────────
//
//  Rigid-body CartPole physics (Barto et al. 1983; OpenAI Gym equations).
//
//    gravity          g  = 9.8 m/s²
//    cart mass       mc  = 1.0 kg
//    pole mass       mp  = 0.1 kg
//    half-pole len    l  = 0.5 m
//    applied force    F  = ±10 N
//    time step        τ  = 0.02 s (Euler)
//
//  Update equations (Euler):
//
//    temp      = (F + mp·l·θ̇²·sin θ) / (mc + mp)
//    θ̈        = (g·sin θ  −  cos θ·temp) / (l·(4/3 − mp·cos²θ/(mc+mp)))
//    ẍ         = temp − mp·l·θ̈·cos θ / (mc + mp)
//
//    x  ← x  + τ·ẋ
//    ẋ  ← ẋ  + τ·ẍ
//    θ  ← θ  + τ·θ̇
//    θ̇  ← θ̇  + τ·θ̈

final class CartPole
{
    // ── Physics constants ──────────────────────────────────────────────────
    private const G              = 9.8;
    private const MASS_CART      = 1.0;
    private const MASS_POLE      = 0.1;
    private const TOTAL_MASS     = 1.1;          // MASS_CART + MASS_POLE
    private const HALF_POLE      = 0.5;          // half-length of pole
    private const POLE_M_L       = 0.05;         // MASS_POLE * HALF_POLE
    private const FORCE_MAG      = 10.0;
    private const TAU            = 0.02;

    // ── Termination thresholds ─────────────────────────────────────────────
    private const X_LIMIT        = 2.4;
    private const THETA_LIMIT    = 12.0 * M_PI / 180.0;   // ≈ 0.2094 rad

    // ── State ─────────────────────────────────────────────────────────────
    private float $x        = 0.0;
    private float $xDot     = 0.0;
    private float $theta    = 0.0;
    private float $thetaDot = 0.0;
    private int   $step     = 0;

    /**
     * Reset to a random state in [−0.05, 0.05]⁴.
     * @return float[]  Initial observation [x, ẋ, θ, θ̇].
     */
    public function reset(): array
    {
        $this->x        = (mt_rand() / mt_getrandmax() * 0.1) - 0.05;
        $this->xDot     = (mt_rand() / mt_getrandmax() * 0.1) - 0.05;
        $this->theta    = (mt_rand() / mt_getrandmax() * 0.1) - 0.05;
        $this->thetaDot = (mt_rand() / mt_getrandmax() * 0.1) - 0.05;
        $this->step     = 0;
        return $this->observation();
    }

    /**
     * Apply action and return (nextObservation, reward, done).
     *
     * @param  int   $action  0 = left, 1 = right
     * @return array{float[], float, bool}
     */
    public function step(int $action): array
    {
        $force     = $action === 1 ? self::FORCE_MAG : -self::FORCE_MAG;
        $cosTheta  = cos($this->theta);
        $sinTheta  = sin($this->theta);

        $temp      = ($force + self::POLE_M_L * $this->thetaDot ** 2 * $sinTheta)
                   / self::TOTAL_MASS;
        $thetaAcc  = (self::G * $sinTheta - $cosTheta * $temp)
                   / (self::HALF_POLE * (4.0 / 3.0
                      - self::MASS_POLE * $cosTheta ** 2 / self::TOTAL_MASS));
        $xAcc      = $temp - self::POLE_M_L * $thetaAcc * $cosTheta / self::TOTAL_MASS;

        // Euler integration
        $this->x        += self::TAU * $this->xDot;
        $this->xDot     += self::TAU * $xAcc;
        $this->theta    += self::TAU * $this->thetaDot;
        $this->thetaDot += self::TAU * $thetaAcc;
        $this->step++;

        $done = abs($this->x)     > self::X_LIMIT
             || abs($this->theta) > self::THETA_LIMIT
             || $this->step       >= MAX_STEPS;

        return [$this->observation(), 1.0, $done];
    }

    private function observation(): array
    {
        return [$this->x, $this->xDot, $this->theta, $this->thetaDot];
    }
}

// ─── Build agent ──────────────────────────────────────────────────────────

$replay = new ReplayBuffer(REPLAY_CAPACITY);
$agent  = new DQNAgent(
    stateDim:         4,
    actionDim:        2,
    replay:           $replay,
    hiddenSize:       HIDDEN_RL,
    epsilonStart:     EPS_START,
    epsilonMin:       EPS_MIN,
    epsilonDecay:     EPS_DECAY,
    gamma:            GAMMA_RL,
    batchSize:        BATCH_SIZE_RL,
    targetUpdateFreq: TARGET_UPDATE,
    lr:               LR_RL,
    weightDecay:      WD_RL,
);

$env = new CartPole();

// ─── Print header ─────────────────────────────────────────────────────────

echo "\n";
echo "════════════════════════════════════════════════════════════\n";
echo "  DQN CartPole Simulator\n";
echo sprintf(
    "  Episodes: %d   Max steps: %d   Hidden: %d   Buffer: %d\n",
    N_EPISODES, MAX_STEPS, HIDDEN_RL, REPLAY_CAPACITY
);
echo sprintf(
    "  ε: %.2f→%.2f (×%.3f)   γ=%.2f   Target update: %d steps\n",
    EPS_START, EPS_MIN, EPS_DECAY, GAMMA_RL, TARGET_UPDATE
);
echo "════════════════════════════════════════════════════════════\n\n";
echo sprintf(
    "  %-9s  %-10s  %-12s  %-10s  %-8s\n",
    'Episode', 'Reward', 'Mean50', 'Avg Loss', 'ε'
);
echo "  " . str_repeat('─', 58) . "\n";

// ─── Training loop ────────────────────────────────────────────────────────

$rewardHistory   = [];
$solvedAt        = null;
$warmingUp       = true;   // true until replay has WARMUP_STEPS transitions
$totalTrainSteps = 0;

for ($episode = 1; $episode <= N_EPISODES; $episode++) {

    $state       = $env->reset();
    $episodeReward = 0.0;
    $episodeLoss   = 0.0;
    $episodeTrain  = 0;

    while (true) {
        // ── Select action ─────────────────────────────────────────────────
        $action = $agent->selectAction($state);

        // ── Environment step ──────────────────────────────────────────────
        [$nextState, $reward, $done] = $env->step($action);
        $episodeReward += $reward;

        // ── Store transition ──────────────────────────────────────────────
        $agent->remember($state, $action, $reward, $nextState, $done);
        $state = $nextState;

        // ── Train if warm ─────────────────────────────────────────────────
        if ($replay->size() >= WARMUP_STEPS) {
            $loss = $agent->train();
            if ($loss !== null) {
                $episodeLoss  += $loss;
                $episodeTrain++;
                $totalTrainSteps++;
            }
        }

        if ($done) break;
    }

    $rewardHistory[] = $episodeReward;

    // ── Compute mean reward over last 50 episodes ─────────────────────────
    $window   = array_slice($rewardHistory, -50);
    $mean50   = array_sum($window) / count($window);
    $avgLoss  = $episodeTrain > 0 ? $episodeLoss / $episodeTrain : NAN;

    // ── Check solved ──────────────────────────────────────────────────────
    if ($solvedAt === null && count($window) === 50 && $mean50 >= SOLVE_THRESHOLD) {
        $solvedAt = $episode;
    }

    // ── Periodic reporting ────────────────────────────────────────────────
    if ($episode % REPORT_EVERY === 0 || $episode === 1) {
        $lossStr = is_nan($avgLoss) ? '     n/a' : sprintf('%8.4f', $avgLoss);
        echo sprintf(
            "  %-9d  %-10.1f  %-12.2f  %s  %.4f\n",
            $episode, $episodeReward, $mean50, $lossStr, $agent->getEpsilon()
        );
    }
}

echo "  " . str_repeat('─', 58) . "\n\n";

// ─── Final summary ────────────────────────────────────────────────────────

$lastWindow = array_slice($rewardHistory, -50);
$finalMean  = array_sum($lastWindow) / count($lastWindow);
$maxReward  = max($rewardHistory);
$minReward  = min($rewardHistory);

echo "── Results ──────────────────────────────────────────────────\n\n";
echo sprintf("  Episodes run       : %d\n",     N_EPISODES);
echo sprintf("  Final mean50       : %.2f\n",   $finalMean);
echo sprintf("  Max episode reward : %.1f\n",   $maxReward);
echo sprintf("  Min episode reward : %.1f\n",   $minReward);
echo sprintf("  Total train steps  : %d\n",     $totalTrainSteps);
echo sprintf("  Final ε            : %.4f\n",   $agent->getEpsilon());

if ($solvedAt !== null) {
    echo sprintf("\n  ✓ Solved at episode %d  (mean50 ≥ %.0f)\n", $solvedAt, SOLVE_THRESHOLD);
} else {
    echo sprintf(
        "\n  ✗ Not yet solved (best mean50 = %.2f < %.0f)\n",
        $finalMean, SOLVE_THRESHOLD
    );
}

// ── Reward trend (ASCII bar chart, last 100 episodes) ─────────────────────

echo "\n── Episode rewards (last 100, bucketed) ─────────────────────\n\n";

$last100 = array_slice($rewardHistory, -100);
$maxR    = max($last100) ?: 1.0;
$nBars   = min(20, count($last100));  // one bar per ~5 episodes
$bucket  = (int) ceil(count($last100) / $nBars);

for ($b = 0; $b < $nBars; $b++) {
    $slice = array_slice($last100, $b * $bucket, $bucket);
    if (empty($slice)) break;
    $avg    = array_sum($slice) / count($slice);
    $barLen = (int) round(40 * $avg / max(MAX_STEPS, $maxR));
    $ep     = (N_EPISODES - 100) + $b * $bucket + 1;
    echo sprintf("  Ep%-4d  %s  %.0f\n",
        $ep, str_repeat('█', $barLen) . str_repeat('░', 40 - $barLen), $avg);
}

echo "\n════════════════════════════════════════════════════════════\n";
echo "  Done.\n";
echo "════════════════════════════════════════════════════════════\n\n";
