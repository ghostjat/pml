<?php

declare(strict_types=1);

namespace Pml\RL;

// ═══════════════════════════════════════════════════════════════════════════
//  ReplayBuffer — Fixed-capacity experience replay for DQN
//
//  Stores transitions (s, a, r, s', done) in a circular buffer of size
//  `capacity`.  When the buffer is full, the oldest transition is silently
//  overwritten (FIFO eviction).  Random mini-batch sampling without
//  replacement is provided via array_rand.
//
//  ── Complexity ────────────────────────────────────────────────────────────
//  push   : O(1)  — slot assignment, pointer increment
//  sample : O(k)  — PHP array_rand over current fill, then k lookups
//  size   : O(1)
// ═══════════════════════════════════════════════════════════════════════════

final class ReplayBuffer
{
    // ── Storage arrays (each index 0..capacity-1 is one slot) ────────────

    /** @var float[][]  state vectors */
    private array $states     = [];

    /** @var int[]  actions taken */
    private array $actions    = [];

    /** @var float[]  rewards received */
    private array $rewards    = [];

    /** @var float[][]  next-state vectors */
    private array $nextStates = [];

    /** @var bool[]  terminal flags */
    private array $dones      = [];

    /** Write-head position in the circular buffer. */
    private int $head  = 0;

    /** Number of transitions currently stored (≤ capacity). */
    private int $count = 0;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int $capacity  Maximum number of transitions to retain.
     *                       Oldest entries are overwritten once full.
     */
    public function __construct(private readonly int $capacity) {}

    // ── API ───────────────────────────────────────────────────────────────

    /**
     * Store a transition.
     *
     * @param float[] $state      Current observation vector.
     * @param int     $action     Action index taken.
     * @param float   $reward     Scalar reward received.
     * @param float[] $nextState  Observation after the action.
     * @param bool    $done       Whether the episode terminated.
     */
    public function push(
        array $state, int $action, float $reward, array $nextState, bool $done
    ): void {
        $this->states[$this->head]     = $state;
        $this->actions[$this->head]    = $action;
        $this->rewards[$this->head]    = $reward;
        $this->nextStates[$this->head] = $nextState;
        $this->dones[$this->head]      = $done;

        $this->head = ($this->head + 1) % $this->capacity;
        if ($this->count < $this->capacity) {
            $this->count++;
        }
    }

    /**
     * Draw a random mini-batch of transitions (without replacement).
     *
     * @param  int   $batchSize  Number of transitions to sample.
     *                           Must be ≤ $this->size().
     * @return array{
     *   states:     float[][],
     *   actions:    int[],
     *   rewards:    float[],
     *   nextStates: float[][],
     *   dones:      bool[]
     * }
     * @throws \UnderflowException if the buffer has fewer than $batchSize entries.
     */
    public function sample(int $batchSize): array
    {
        if ($batchSize > $this->count) {
            throw new \UnderflowException(
                "ReplayBuffer::sample(): requested {$batchSize} transitions "
                . "but only {$this->count} are stored."
            );
        }

        // array_rand returns a scalar when batchSize=1, else an array
        $keys = array_rand(array_fill(0, $this->count, null), $batchSize);
        if (!is_array($keys)) {
            $keys = [$keys];
        }

        $states     = [];
        $actions    = [];
        $rewards    = [];
        $nextStates = [];
        $dones      = [];

        foreach ($keys as $i) {
            $states[]     = $this->states[$i];
            $actions[]    = $this->actions[$i];
            $rewards[]    = $this->rewards[$i];
            $nextStates[] = $this->nextStates[$i];
            $dones[]      = $this->dones[$i];
        }

        return compact('states', 'actions', 'rewards', 'nextStates', 'dones');
    }

    /**
     * Current number of stored transitions (≤ capacity).
     */
    public function size(): int
    {
        return $this->count;
    }
}
