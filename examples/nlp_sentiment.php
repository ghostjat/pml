<?php

declare(strict_types=1);

/**
 * ════════════════════════════════════════════════════════════════════════════
 *  examples/nlp_sentiment.php — Binary Sentiment Classification with TF-IDF
 * ════════════════════════════════════════════════════════════════════════════
 *
 * Demonstrates the full NLP feature-extraction pipeline on a small, hard-coded
 * corpus of movie reviews.  After training, we predict the sentiment of a new
 * unseen sentence.
 *
 * ── What you will learn ──────────────────────────────────────────────────────
 *  Step 1  Tokenise raw strings with CountVectorizer (unigrams + bigrams)
 *  Step 2  Weight term frequencies with TfidfTransformer (IDF + L2 norm)
 *  Step 3  Train XGBClassifier on the resulting float32 Tensor
 *  Step 4  Preprocess a new sentence through the same fitted transformers
 *  Step 5  Predict and interpret the class-probability output
 *
 * ── Why CountVectorizer sits outside Pipeline ─────────────────────────────────
 *  Pml\Classic\Pipeline\Pipeline passes Tensors between steps.
 *  CountVectorizer's input is array<string>, so it lives *before* the Tensor
 *  bridge.  Once the text is converted to a Tensor by CountVectorizer, the
 *  remaining steps (TfidfTransformer, classifier) fit cleanly into Pipeline.
 *
 * ── Usage ────────────────────────────────────────────────────────────────────
 *   php examples/nlp_sentiment.php
 */

// ── Autoloader ────────────────────────────────────────────────────────────────

foreach ([__DIR__ . '/../vendor/autoload.php', __DIR__ . '/vendor/autoload.php'] as $al) {
    if (file_exists($al)) { require_once $al; break; }
}

use Pml\Tensor;
use Pml\Classic\FeatureExtraction\Text\{CountVectorizer, TfidfTransformer};
use Pml\Classic\Ensemble\XGBClassifier;

// ── Helper ────────────────────────────────────────────────────────────────────

function printSection(string $title): void
{
    $line = str_repeat('─', 70);
    echo "\n{$line}\n  {$title}\n{$line}\n";
}

// ════════════════════════════════════════════════════════════════════════════
//  1. Corpus
// ════════════════════════════════════════════════════════════════════════════

printSection('1. Training corpus');

$trainDocs = [
    // ── Positive (label 1) ──────────────────────────────────────────────
    'The movie was fantastic and absolutely great',
    'A wonderful film with brilliant performances',
    'I loved every minute of this masterpiece',
    'Outstanding direction and a gripping story',
    'Superb acting and a heartwarming narrative',
    'One of the best films I have ever seen',
    'A delightful and emotionally rich experience',
    'Incredible visuals paired with a stellar cast',

    // ── Negative (label 0) ──────────────────────────────────────────────
    'Terrible film, worst movie I have ever seen',
    'A complete waste of time and money',
    'Dull, boring, and painfully predictable',
    'The acting was awful and the plot made no sense',
    'I fell asleep halfway through this disaster',
    'Poorly written script with zero character development',
    'Absolutely dreadful from start to finish',
    'A forgettable mess that insults the audience',
];

$trainLabels = [1, 1, 1, 1, 1, 1, 1, 1,   // positive
                0, 0, 0, 0, 0, 0, 0, 0];   // negative

echo 'Documents : ' . count($trainDocs) . "\n";
echo 'Positives : ' . array_sum($trainLabels) . '   Negatives : '
                    . (count($trainLabels) - array_sum($trainLabels)) . "\n";

// ════════════════════════════════════════════════════════════════════════════
//  2. CountVectorizer  (unigrams + bigrams)
// ════════════════════════════════════════════════════════════════════════════

printSection('2. CountVectorizer  (ngram_range=[1,2])');

$cv = new CountVectorizer(ngram_range: [1, 2], lowercase: true);
$X_counts = $cv->fit_transform($trainDocs);

printf("Vocabulary size : %d N-grams\n", $cv->n_features_);
printf("Count matrix    : %d × %d\n", ...$X_counts->shape);

// Print the 20 most frequent unigrams to illustrate the vocabulary
$freq = array_fill(0, $cv->n_features_, 0.0);
for ($i = 0; $i < $X_counts->shape[0]; $i++) {
    $base = $i * $cv->n_features_;
    for ($j = 0; $j < $cv->n_features_; $j++) {
        $freq[$j] += (float) $X_counts->buffer[$base + $j];
    }
}
arsort($freq);
$top20 = array_slice($freq, 0, 20, preserve_keys: true);
$invVocab = array_flip($cv->vocabulary_);
echo "\nTop-20 N-grams by corpus frequency:\n";
foreach ($top20 as $idx => $count) {
    printf("  %-30s  %.0f\n", $invVocab[$idx], $count);
}

// ════════════════════════════════════════════════════════════════════════════
//  3. TfidfTransformer  (IDF weighting + L2 row normalisation)
// ════════════════════════════════════════════════════════════════════════════

printSection('3. TfidfTransformer  (use_idf=true, norm=\'l2\')');

$tfidf = new TfidfTransformer(use_idf: true, norm: 'l2', smooth_idf: true);
$X_tfidf = $tfidf->fit_transform($X_counts);

printf("TF-IDF matrix : %d × %d  (float32 Tensor)\n", ...$X_tfidf->shape);

// Sanity: every row should have L2-norm ≈ 1 (or 0 for all-zero rows)
$rowNorms = [];
for ($i = 0; $i < $X_tfidf->shape[0]; $i++) {
    $sum = 0.0;
    $base = $i * $tfidf->n_features_in_;
    for ($j = 0; $j < $tfidf->n_features_in_; $j++) {
        $v = (float) $X_tfidf->buffer[$base + $j];
        $sum += $v * $v;
    }
    $rowNorms[] = sqrt($sum);
}
printf("Row L2 norms  : min=%.4f  max=%.4f  (should be ≈ 1.0)\n",
       min($rowNorms), max($rowNorms));

// ════════════════════════════════════════════════════════════════════════════
//  4. XGBClassifier — train on TF-IDF features
// ════════════════════════════════════════════════════════════════════════════

printSection('4. XGBClassifier  (binary:logistic)');

// Wrap integer labels into a float32 Tensor [n_samples]
$y = Tensor::fromArray(array_map('floatval', $trainLabels));

$clf = new XGBClassifier(
    n_estimators:  60,
    max_depth:      4,
    learning_rate:  0.2,
    subsample:      0.8,
    random_state:   42,
);
$clf->fit($X_tfidf, $y);

// Training accuracy
$predTrain = $clf->predict($X_tfidf);
$correct   = 0;
for ($i = 0; $i < count($trainLabels); $i++) {
    if ((int) $predTrain->buffer[$i] === $trainLabels[$i]) {
        $correct++;
    }
}
printf("Training accuracy : %.1f%%  (%d / %d)\n",
       100.0 * $correct / count($trainLabels),
       $correct, count($trainLabels));

// ════════════════════════════════════════════════════════════════════════════
//  5. Inference on new sentences
// ════════════════════════════════════════════════════════════════════════════

printSection('5. Inference on unseen sentences');

$testSentences = [
    'This was an amazing and brilliant movie',
    'Horrible experience, completely awful and boring',
    'Decent film with some great moments',
    'Total disaster, a dreadful waste of time',
];

// Apply the SAME fitted transformers (no re-fitting)
$X_test_counts = $cv->transform($testSentences);
$X_test_tfidf  = $tfidf->transform($X_test_counts);

$predictions = $clf->predict($X_test_tfidf);
$probas      = $clf->predict_proba($X_test_tfidf);

// predict_proba returns [n_samples, 2] for binary classification
$nClasses = $clf->n_classes_;
printf("%-44s  %-8s  %s\n", 'Sentence', 'Pred', 'P(positive)');
printf("%s\n", str_repeat('─', 70));
foreach ($testSentences as $i => $sent) {
    $label   = (int) $predictions->buffer[$i];
    $tag     = $label === 1 ? 'POSITIVE' : 'NEGATIVE';
    // Shape-driven extraction: [n, 2] → column 1 is P(class=1);
    // fallback for any 1D output (e.g. multi:softmax, or future objectives)
    $is2D = count($probas->shape) === 2 && $probas->shape[1] === 2;
    $pPos = $is2D ? (float) $probas->buffer[$i * 2 + 1] : (float) $probas->buffer[$i];
    printf("%-44s  %-8s  %.3f\n", '"' . substr($sent, 0, 42) . '"', $tag, $pPos);
}

// ════════════════════════════════════════════════════════════════════════════
//  6. TF-IDF Pipeline pattern  (TfidfTransformer + classifier, Tensor-native)
// ════════════════════════════════════════════════════════════════════════════

printSection('6. Equivalent Pipeline pattern  (Tensor steps only)');

echo <<<'EOT'
// CountVectorizer sits BEFORE the Pipeline because it accepts array<string>,
// not Tensor.  Once documents are vectorised, the remaining Tensor-compatible
// steps chain via make_pipeline():
//
//   $cv    = new CountVectorizer(ngram_range: [1, 2]);
//   $X_raw = $cv->fit_transform($trainDocs);          // array → Tensor
//
//   $pipeline = make_pipeline(
//       new TfidfTransformer(norm: 'l2'),             // Tensor → Tensor
//       new XGBClassifier(n_estimators: 60),          // Tensor → labels
//   );
//   $pipeline->fit($X_raw, $y);
//   $pred = $pipeline->predict($cv->transform($testDocs));

EOT;

echo "\nDone.\n";
