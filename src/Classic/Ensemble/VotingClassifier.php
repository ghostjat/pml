<?php

declare(strict_types=1);

namespace Pml\Classic\Ensemble;

use Pml\{Tensor, BlasEngine};
use Pml\Classic\{Estimator, Predictor};

// ═══════════════════════════════════════════════════════════════════════════
//  VotingClassifier — sklearn.ensemble.VotingClassifier
//
//  Meta-estimator that fits a set of heterogeneous classifiers and combines
//  their predictions via majority vote (hard) or averaged probability (soft).
//
//  ── Hard Voting ────────────────────────────────────────────────────────────
//
//  Each estimator emits a hard class label via predict().  For each sample,
//  the label with the most votes is returned.  Ties are broken by the lower
//  label value (consistent with the sorted $this->classes_ order).
//
//  ── Soft Voting ────────────────────────────────────────────────────────────
//
//  Each estimator emits a probability distribution via predict_proba().
//  Probabilities are accumulated (with class alignment) across all estimators,
//  then divided by n_estimators.  The class with the highest mean probability
//  is predicted.
//
//  ── Class alignment ────────────────────────────────────────────────────────
//
//  Different estimators may have been trained on data containing different
//  subsets of classes (e.g. when a base estimator was pre-fitted externally).
//  The VotingClassifier discovers the global union of all classes seen across
//  ALL estimators' classes_ at fit() time and stores it in $this->classes_.
//  When accumulating probabilities (soft) or votes (hard), each estimator's
//  output is aligned to the global class space via $classPos map, exactly as
//  RandomForestClassifier handles its bootstrap tree subset-classes.
//
//  ── BLAS in soft voting ────────────────────────────────────────────────────
//
//  When an estimator's classes_ exactly matches the forest-level classes_:
//    cblas_saxpy(m*K, 1.0, proba, 1, acc, 1)    — accumulate in one C call
//
//  Normalise: cblas_sscal(m*K, 1/T, acc, 1)     — divide in one C call
//
//  ── Estimator contract ────────────────────────────────────────────────────
//
//  Each value in the $estimators constructor array must implement both
//  Estimator and Predictor.  For soft voting they must additionally expose
//  a predict_proba(Tensor): Tensor method returning [n_samples, n_classes].
//
//  Pre-fitted estimators are accepted — fit() skips re-fitting if the
//  estimator was already fitted (detected via a public `classes_` property).
//  Set $fitEstimators=false to suppress fitting entirely.
// ═══════════════════════════════════════════════════════════════════════════

final class VotingClassifier implements Estimator, Predictor
{
    // ── Fitted attributes ─────────────────────────────────────────────────

    /**
     * Fitted estimator objects, keyed by their name from the constructor.
     * @var array<string, Estimator&Predictor>
     */
    public readonly array $estimators_;

    /**
     * Globally merged, sorted unique class labels across all estimators.
     * @var int[]
     */
    public readonly array $classes_;

    public readonly int $n_classes_;
    public readonly int $n_features_in_;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param array<string, Estimator&Predictor> $estimators
     *        Named estimators: ['rf' => new RandomForestClassifier(), 'svc' => new SVC(), …].
     *        Each must implement Estimator and Predictor (fit + predict).
     *        For soft voting they must also have predict_proba().
     *
     * @param string $voting     'hard' (majority class vote) or
     *                           'soft' (argmax of averaged class probabilities).
     *
     * @param bool   $fitEstimators  If true (default), fit() re-trains every estimator.
     *                               Set false if passing pre-fitted estimators.
     */
    public function __construct(
        private readonly array  $estimators,
        private readonly string $voting        = 'hard',
        private readonly bool   $fitEstimators = true,
    ) {
        if (!in_array($voting, ['hard', 'soft'], true)) {
            throw new \InvalidArgumentException(
                "VotingClassifier: voting must be 'hard' or 'soft'; got '{$voting}'."
            );
        }
        if (count($estimators) < 2) {
            throw new \InvalidArgumentException(
                'VotingClassifier: at least 2 estimators are required.'
            );
        }
        foreach ($estimators as $name => $est) {
            if (!($est instanceof Estimator && $est instanceof Predictor)) {
                throw new \InvalidArgumentException(
                    "VotingClassifier: estimator '{$name}' must implement Estimator and Predictor."
                );
            }
        }
    }

    // ── Estimator ──────────────────────────────────────────────────────────

    /**
     * Fit all base estimators on (X, y) and discover the global class set.
     *
     * If $fitEstimators=false (passed at construction), each estimator is
     * assumed to be pre-fitted and fit() simply collects their classes_.
     *
     * @param Tensor      $X  [n_samples, n_features]
     * @param Tensor|null $y  Class labels [n_samples]
     */
    public function fit(Tensor $X, ?Tensor $y = null): static
    {
        if ($y === null) {
            throw new \InvalidArgumentException('VotingClassifier: y must be provided.');
        }
        if (count($X->shape) !== 2) {
            throw new \InvalidArgumentException('VotingClassifier: X must be 2-D [n_samples, n_features].');
        }

        $fittedEstimators = [];
        $globalClassSet   = [];

        foreach ($this->estimators as $name => $est) {
            if ($this->fitEstimators) {
                $est->fit($X, $y);
            }
            $fittedEstimators[$name] = $est;

            // Collect classes from the fitted estimator (requires classes_ property)
            if (property_exists($est, 'classes_')) {
                foreach ($est->classes_ as $cls) {
                    $globalClassSet[$cls] = true;
                }
            }
        }

        // If no estimator exposed classes_, derive from y
        if ($globalClassSet === []) {
            $n = $X->shape[0];
            for ($i = 0; $i < $n; $i++) {
                $globalClassSet[(int) round((float) $y->buffer[$i])] = true;
            }
        }

        ksort($globalClassSet);
        $allClasses = array_keys($globalClassSet);

        $this->estimators_    = $fittedEstimators;
        $this->classes_       = $allClasses;
        $this->n_classes_     = count($allClasses);
        $this->n_features_in_ = $X->shape[1];

        return $this;
    }

    // ── Predictor ──────────────────────────────────────────────────────────

    /**
     * Predict class labels.
     *
     * Hard voting: majority class across all estimators' predict() outputs.
     * Soft voting:  argmax of averaged predict_proba() across all estimators.
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    Integer class labels [n_samples]
     */
    public function predict(Tensor $X): Tensor
    {
        if ($this->voting === 'soft') {
            return $this->predictSoft($X);
        }
        return $this->predictHard($X);
    }

    /**
     * Predict averaged class probabilities.
     * Only valid when voting='soft'; each estimator must implement predict_proba().
     *
     * @param Tensor $X  [n_samples, n_features]
     * @return Tensor    [n_samples, n_classes] averaged probabilities
     */
    public function predict_proba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if ($this->voting !== 'soft') {
            throw new \RuntimeException(
                "VotingClassifier::predict_proba() is only available when voting='soft'."
            );
        }

        return $this->accumulatedProba($X);
    }

    /**
     * Accuracy score on test data.
     */
    public function score(Tensor $X, Tensor $y): float
    {
        $pred = $this->predict($X);
        $n    = $y->size;
        $ok   = 0;
        for ($i = 0; $i < $n; $i++) {
            if ((int) round((float) $y->buffer[$i]) === (int) round((float) $pred->buffer[$i])) {
                $ok++;
            }
        }
        return $ok / $n;
    }

    // ── Private prediction helpers ────────────────────────────────────────

    /**
     * Hard voting: collect each estimator's predict() output, then take the
     * majority class for each sample.
     *
     * Class alignment is handled via a per-sample vote counter array indexed
     * by position in $this->classes_ (the global class order).
     */
    private function predictHard(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "VotingClassifier::predict() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m          = $X->shape[0];
        $K          = $this->n_classes_;
        $classPos   = array_flip($this->classes_);
        $nEst       = count($this->estimators_);

        // votes[i][c] = number of estimators that voted class $this->classes_[c] for sample i
        // Use flat PHP int[] for speed: votes[$i * $K + $c]
        $votes = array_fill(0, $m * $K, 0);

        foreach ($this->estimators_ as $est) {
            $pred = $est->predict($X);   // [m]

            for ($i = 0; $i < $m; $i++) {
                $label = (int) round((float) $pred->buffer[$i]);
                $pos   = $classPos[$label] ?? null;
                if ($pos !== null) {
                    $votes[$i * $K + $pos]++;
                }
            }
        }

        // Output: argmax over votes for each sample
        $out = new Tensor([$m]);
        for ($i = 0; $i < $m; $i++) {
            $base    = $i * $K;
            $bestPos = 0;
            $bestV   = $votes[$base];
            for ($c = 1; $c < $K; $c++) {
                if ($votes[$base + $c] > $bestV) {
                    $bestV   = $votes[$base + $c];
                    $bestPos = $c;
                }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Soft voting: accumulate predict_proba() matrices with class alignment,
     * normalise, then return argmax as the predicted label.
     */
    private function predictSoft(Tensor $X): Tensor
    {
        $proba = $this->accumulatedProba($X);
        $m     = $X->shape[0];
        $K     = $this->n_classes_;
        $out   = new Tensor([$m]);

        for ($i = 0; $i < $m; $i++) {
            $base    = $i * $K;
            $bestPos = 0;
            $bestVal = (float) $proba->buffer[$base];
            for ($c = 1; $c < $K; $c++) {
                $v = (float) $proba->buffer[$base + $c];
                if ($v > $bestVal) { $bestVal = $v; $bestPos = $c; }
            }
            $out->buffer[$i] = (float) $this->classes_[$bestPos];
        }

        return $out;
    }

    /**
     * Accumulate predict_proba() across all soft-voting estimators with class
     * alignment, then normalise by dividing by n_estimators.
     *
     * ── Class alignment ───────────────────────────────────────────────────
     *
     * An estimator's predict_proba() returns [n_samples, est_n_classes].
     * Its column order corresponds to est->classes_, which may be a subset
     * or different ordering from $this->classes_.
     *
     * For each estimator, we build a scatter map:
     *   estClassToForestPos[est_col] = $this->classes_ position
     * and use it to place each probability column into the correct position
     * in the global accumulator.
     *
     * Fast path (same class layout as forest): cblas_saxpy on full [m*K] vector.
     *
     * @return Tensor [n_samples, n_classes] averaged probabilities
     */
    private function accumulatedProba(Tensor $X): Tensor
    {
        $this->checkFitted();

        if (count($X->shape) !== 2 || $X->shape[1] !== $this->n_features_in_) {
            throw new \InvalidArgumentException(
                "VotingClassifier::predict_proba() expected [*, {$this->n_features_in_}], "
                . 'got [' . implode(', ', $X->shape) . '].'
            );
        }

        $m             = $X->shape[0];
        $K             = $this->n_classes_;
        $blas          = BlasEngine::get()->ffi;
        $acc           = Tensor::zeros([$m, $K]);
        $forestClassPos = array_flip($this->classes_);
        $nContrib      = 0;

        foreach ($this->estimators_ as $name => $est) {
            if (!method_exists($est, 'predict_proba')) {
                throw new \RuntimeException(
                    "VotingClassifier (soft): estimator '{$name}' does not implement predict_proba()."
                );
            }

            $proba   = $est->predict_proba($X);   // [m, est_K]
            $estCls  = property_exists($est, 'classes_') ? $est->classes_ : $this->classes_;
            $estK    = count($estCls);

            // Fast path: same class layout as global
            if ($estK === $K && $estCls === $this->classes_) {
                $blas->cblas_saxpy($m * $K, 1.0, $proba->buffer, 1, $acc->buffer, 1);
            } else {
                // Scatter: for each estimator class, find its global position
                for ($i = 0; $i < $m; $i++) {
                    $tBase      = $i * $estK;
                    $forestBase = $i * $K;
                    for ($ec = 0; $ec < $estK; $ec++) {
                        $fp = $forestClassPos[$estCls[$ec]] ?? null;
                        if ($fp !== null) {
                            $acc->buffer[$forestBase + $fp] =
                                (float) $acc->buffer[$forestBase + $fp]
                                + (float) $proba->buffer[$tBase + $ec];
                        }
                    }
                }
            }
            $nContrib++;
        }

        // Normalise: divide by number of contributing estimators
        if ($nContrib > 0) {
            $blas->cblas_sscal($m * $K, 1.0 / $nContrib, $acc->buffer, 1);
        }

        return $acc;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    private function checkFitted(): void
    {
        if (!isset($this->estimators_)) {
            throw new \RuntimeException(
                'VotingClassifier is not fitted. Call fit() first.'
            );
        }
    }
}
