<?php

declare(strict_types=1);

namespace Pml\Serialization;

use Pml\Interfaces\MLBackend;
use Pml\Training\TrainingResult;
use RuntimeException;

/**
 * Manages a rolling window of named checkpoints under a root directory.
 *
 * Features:
 * - Saves a checkpoint via ModelHub with a timestamped or epoch-tagged name.
 * - Tracks the "best" checkpoint by validation loss.
 * - Prunes old checkpoints to keep at most $keepLast on disk.
 * - Writes a `checkpoints.json` index so the state survives process restarts.
 *
 * Directory layout:
 *   $root/checkpoints.json           — index of all managed checkpoints
 *   $root/epoch_005/hub_meta.json    — individual checkpoint bundle
 *   $root/epoch_010/…
 *   $root/best/hub_meta.json         — symlink OR copy of best checkpoint
 *
 * Usage:
 *   $mgr = new CheckpointManager('/runs/exp1', keepLast: 3);
 *   $mgr->save($backend, epoch: 5, valLoss: 0.241);
 *   $mgr->saveBest($backend, valLoss: 0.198);
 *   $backend = $mgr->loadBest();
 */
final class CheckpointManager
{
    private const INDEX_FILE = 'checkpoints.json';
    private const BEST_TAG   = 'best';

    /** @var array<int, array{tag: string, valLoss: float|null, savedAt: string}> */
    private array $index = [];

    private ?float $bestValLoss = null;

    public function __construct(
        private readonly string $root,
        private readonly int $keepLast = 5
    ) {
        if (!is_dir($root)) {
            mkdir($root, 0755, true);
        }
        $this->loadIndex();
    }

    /**
     * Save a checkpoint tagged by epoch number.
     * Prunes oldest checkpoints if the rolling window is exceeded.
     *
     * @param MLBackend    $backend
     * @param int          $epoch
     * @param float|null   $valLoss  Used for best-checkpoint tracking.
     */
    public function save(MLBackend $backend, int $epoch, ?float $valLoss = null): string
    {
        $tag = \sprintf('epoch_%05d', $epoch);
        $dir = $this->root . \DIRECTORY_SEPARATOR . $tag;

        ModelHub::save($backend, $dir);

        $this->index[] = [
            'tag'     => $tag,
            'valLoss' => $valLoss,
            'savedAt' => date(\DATE_ATOM),
        ];

        $this->prune();
        $this->writeIndex();

        return $dir;
    }

    /**
     * Save the "best" checkpoint (overwrites the previous best).
     * Skips saving if $valLoss is not an improvement.
     *
     * @return bool  true if saved, false if skipped.
     */
    public function saveBest(MLBackend $backend, float $valLoss): bool
    {
        if ($this->bestValLoss !== null && $valLoss >= $this->bestValLoss) {
            return false;
        }

        $this->bestValLoss = $valLoss;
        $dir = $this->root . \DIRECTORY_SEPARATOR . self::BEST_TAG;

        // Wipe the old best directory before overwriting.
        if (is_dir($dir)) {
            $this->rmdirRecursive($dir);
        }

        ModelHub::save($backend, $dir);
        $this->writeIndex();

        return true;
    }

    /**
     * Load the latest checkpoint by epoch.
     */
    public function loadLatest(): MLBackend
    {
        if (empty($this->index)) {
            throw new RuntimeException("CheckpointManager: no checkpoints saved yet.");
        }
        $last = $this->index[\count($this->index) - 1];
        return ModelHub::load($this->root . \DIRECTORY_SEPARATOR . $last['tag']);
    }

    /**
     * Load the best (lowest val-loss) checkpoint.
     */
    public function loadBest(): MLBackend
    {
        $dir = $this->root . \DIRECTORY_SEPARATOR . self::BEST_TAG;
        if (!is_dir($dir)) {
            throw new RuntimeException("CheckpointManager: no 'best' checkpoint saved yet.");
        }
        return ModelHub::load($dir);
    }

    /**
     * Return all checkpoint metadata from the index.
     *
     * @return array<int, array{tag: string, valLoss: float|null, savedAt: string}>
     */
    public function listCheckpoints(): array
    {
        return $this->index;
    }

    public function bestValLoss(): ?float { return $this->bestValLoss; }

    // -------------------------------------------------------------------------

    private function prune(): void
    {
        if ($this->keepLast <= 0) {
            return;
        }

        while (\count($this->index) > $this->keepLast) {
            $oldest = array_shift($this->index);
            $dir    = $this->root . \DIRECTORY_SEPARATOR . $oldest['tag'];
            if (is_dir($dir)) {
                $this->rmdirRecursive($dir);
            }
        }
    }

    private function loadIndex(): void
    {
        $path = $this->root . \DIRECTORY_SEPARATOR . self::INDEX_FILE;
        if (!is_file($path)) {
            return;
        }
        $raw = file_get_contents($path);
        if ($raw === false) return;

        $data = json_decode($raw, true, 512, \JSON_THROW_ON_ERROR);
        $this->index       = $data['checkpoints'] ?? [];
        $this->bestValLoss = isset($data['bestValLoss']) ? (float) $data['bestValLoss'] : null;
    }

    private function writeIndex(): void
    {
        $data = [
            'checkpoints' => $this->index,
            'bestValLoss' => $this->bestValLoss,
            'updatedAt'   => date(\DATE_ATOM),
        ];
        file_put_contents(
            $this->root . \DIRECTORY_SEPARATOR . self::INDEX_FILE,
            json_encode($data, \JSON_PRETTY_PRINT | \JSON_UNESCAPED_SLASHES)
        );
    }

    private function rmdirRecursive(string $dir): void
    {
        $items = scandir($dir);
        if ($items === false) return;

        foreach ($items as $item) {
            if ($item === '.' || $item === '..') continue;
            $path = $dir . \DIRECTORY_SEPARATOR . $item;
            is_dir($path) ? $this->rmdirRecursive($path) : unlink($path);
        }
        rmdir($dir);
    }
}
