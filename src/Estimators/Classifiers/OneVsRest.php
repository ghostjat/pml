<?php
declare(strict_types=1);

namespace Pml\Estimators\Classifiers;

use Pml\Interfaces\Learner;
use Pml\Interfaces\Probabilistic;
use Pml\Interfaces\Persistable;
use Pml\Tensor;
use Pml\Dataset;
use RuntimeException;

/**
 * One-vs-Rest (OvR) meta-classifier.
 * Trains one binary classifier per class; predicts via argmax of probabilities.
 *
 * JIT & Memory Optimized:
 * - Each binary dataset view is a zero-copy boolean-indexed Tensor slice.
 * - Probability aggregation stays in C; PHP only reads a single argmax integer.
 *
 * @template T of (Learner&Probabilistic)
 */
final class OneVsRest implements Learner, Probabilistic, Persistable
{
    /** @var array<int, Learner&Probabilistic> */
    private array $classifiers = [];
    /** @var int[] class index → original label */
    private array $classes     = [];

    /**
     * @param Learner&Probabilistic $prototype  A fitted-prototype or fresh binary classifier.
     *        It is cloned once per class.
     */
    public function __construct(private readonly Learner $prototype) {}

    public function train(Dataset $dataset): void
    {
        $labels = $dataset->labels();
        if ($labels === null) {
            throw new \InvalidArgumentException("OneVsRest requires labeled data.");
        }

        $flat         = $labels->toFlatArray();
        $unique       = array_values(array_unique($flat));
        sort($unique);
        $this->classes     = $unique;
        $this->classifiers = [];

        foreach ($unique as $class) {
            // Binary label: 1 if current class, 0 otherwise — stays in C
            $binaryLabels = $labels->equal(
                Tensor::zeros($dataset->numRows())->addScalarInplace((float) $class)
            );                                                         // [N] 0/1 float

            $binaryDataset = new Dataset($dataset->samples(), $binaryLabels);

            $clf = clone $this->prototype;
            $clf->train($binaryDataset);
            $this->classifiers[] = $clf;
        }
    }

    /**
     * Returns [N × K] probability matrix.
     */
    public function proba(Dataset $dataset): Tensor
    {
        if (!$this->trained()) {
            throw new RuntimeException("OneVsRest is not trained.");
        }

        $cols = [];
        foreach ($this->classifiers as $clf) {
            /** @var Probabilistic $clf */
            $p      = $clf->proba($dataset);                            // [N]
            $p1d    = $p->ndim() > 1 ? $p->flatten() : $p;
            $cols[] = $p1d->expandDims(1);                             // [N × 1]
        }

        return Tensor::concat($cols, 1);                               // [N × K]
    }

    public function predict(Dataset $dataset): Tensor
    {
        $proba   = $this->proba($dataset);                             // [N × K]
        $k       = count($this->classes);
        $indices = $proba->argsort(1)->col($k - 1)->toFlatArray();    // argmax per row

        $preds = [];
        foreach ($indices as $idx) {
            $preds[] = $this->classes[(int) $idx] ?? 0;
        }
        return Tensor::fromArray($preds);
    }

    public function trained(): bool
    {
        return !empty($this->classifiers);
    }

    public function save(string $dir): void
    {
        is_dir($dir) || mkdir($dir, 0755, true);
        $manifest = [];
        foreach ($this->classifiers as $idx => $clf) {
            if (!($clf instanceof Persistable)) {
                throw new RuntimeException("OneVsRest::save() requires all classifiers to implement Persistable.");
            }
            $clf->save($dir . '/clf_' . $idx);
            $manifest[] = get_class($clf);
        }
        file_put_contents($dir . '/config.json', json_encode(['classes' => $this->classes, 'clfClasses' => $manifest]));
    }

    public static function load(string $dir): self
    {
        $c = json_decode(file_get_contents($dir . '/config.json'), true);
        // Reconstruct using the first classifier as prototype (already trained; clone not needed)
        $clfs = [];
        foreach ($c['clfClasses'] as $idx => $class) {
            $clfs[] = $class::load($dir . '/clf_' . $idx);
        }
        // prototype is only used at train time; pass first clf as a stand-in
        $i = new self($clfs[0]);
        $i->classifiers = $clfs;
        $i->classes = $c['classes'] ?? [];
        return $i;
    }
}
