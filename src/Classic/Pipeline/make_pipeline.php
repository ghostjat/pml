<?php

declare(strict_types=1);

namespace Pml\Classic\Pipeline;

// ═══════════════════════════════════════════════════════════════════════════
//  make_pipeline — sklearn.pipeline.make_pipeline
//
//  Convenience factory that builds a Pipeline from a variadic list of
//  estimator/transformer objects, auto-generating a lowercase step name
//  from each object's short class name.
//
//  Duplicate class names get a numeric suffix appended: 'minmaxscaler',
//  'minmaxscaler-1', 'minmaxscaler-2' … (matching sklearn's convention).
//
//  Usage:
//    use function Pml\Classic\Pipeline\make_pipeline;
//
//    $pipe = make_pipeline(
//        new MinMaxScaler(),
//        new PCA(n_components: 10),
//        new LogisticRegression(),
//    );
//    // equivalent to:
//    $pipe = new Pipeline([
//        ['minmaxscaler', new MinMaxScaler()],
//        ['pca',          new PCA(n_components: 10)],
//        ['logisticregression', new LogisticRegression()],
//    ]);
//
//  Importable as a global function via:
//    use function Pml\Classic\Pipeline\make_pipeline;
//
//  Or callable statically via Pipeline::make(...$steps).
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Build a Pipeline, auto-naming each step from its short class name.
 *
 * Duplicate short names are resolved by appending -1, -2, … to the second
 * and subsequent occurrences, matching sklearn's make_pipeline() convention.
 *
 * @param object ...$steps  Ordered list of transformers + final estimator.
 * @return Pipeline
 */
function make_pipeline(object ...$steps): Pipeline
{
    if (count($steps) === 0) {
        throw new \InvalidArgumentException('make_pipeline() requires at least one step.');
    }

    // ── Auto-generate names ────────────────────────────────────────────────
    //
    // 1. Derive a base name: strtolower of the short class name (no namespace).
    //    e.g. Pml\Classic\Preprocess\MinMaxScaler → 'minmaxscaler'
    //
    // 2. Track occurrence counts; append '-N' to the Nth duplicate starting
    //    at N=1.  The first occurrence keeps the bare name (no suffix).
    $counts  = [];   // base_name → how many times seen so far
    $namedSteps = [];

    foreach ($steps as $step) {
        // ReflectionClass::getShortName() strips the namespace prefix
        $baseName = strtolower((new \ReflectionClass($step))->getShortName());

        if (!isset($counts[$baseName])) {
            // First occurrence: use bare name, record count = 0
            $counts[$baseName] = 0;
            $name = $baseName;
        } else {
            // Duplicate: increment counter and append suffix
            $counts[$baseName]++;
            $name = $baseName . '-' . $counts[$baseName];
        }

        $namedSteps[] = [$name, $step];
    }

    return new Pipeline($namedSteps);
}
