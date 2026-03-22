<?php

declare(strict_types=1);

/**
 * career_slm.php — Tiny Career-Counselling Micro-SLM
 *
 * A 2-layer MLP trained from scratch on the Pml tensor library.
 *
 *   Architecture : BoW[90] → Dense[128] → ReLU → Dense[8] → Softmax
 *   Training     : full-batch SGD, manual backpropagation, cross-entropy loss
 *   Evaluation   : Accuracy, Top-2, Macro-F1, Confusion Matrix (Pml\Metrics)
 *   Inference    : interactive REPL — type a career question, get advice
 *
 * Run:  php -d ffi.enable=true career_slm.php
 */

require_once __DIR__ . '/vendor/autoload.php';

use Pml\{Tensor, Ops, BlasEngine, Metrics};

// ── ANSI helpers ──────────────────────────────────────────────────────────────
[$GRN, $YLW, $CYN, $RED, $BLD, $RST] =
    ["\033[32m", "\033[33m", "\033[36m", "\033[31m", "\033[1m", "\033[0m"];

function hdr(string $s): void  { global $CYN,$BLD,$RST; echo "\n{$CYN}{$BLD}── {$s} ──{$RST}\n"; }
function ok(string $s): void   { global $GRN,$RST;      echo "  {$GRN}✓{$RST}  {$s}\n"; }
function info(string $s): void { global $YLW,$RST;      echo "  {$YLW}→{$RST}  {$s}\n"; }
function err(string $s): void  { global $RED,$RST;      echo "  {$RED}✗{$RST}  {$s}\n"; }

// ── VOCABULARY (90 tokens) ────────────────────────────────────────────────────
const VOCAB = [
    // Engineering (0-8)
    'build'=>0,'mechanical'=>1,'electrical'=>2,'civil'=>3,'structural'=>4,
    'machines'=>5,'robots'=>6,'engineer'=>7,'engineering'=>8,
    // Medicine (9-17)
    'doctor'=>9,'health'=>10,'hospital'=>11,'patient'=>12,'surgery'=>13,
    'medical'=>14,'disease'=>15,'nurse'=>16,'medicine'=>17,
    // Law (18-26)
    'lawyer'=>18,'court'=>19,'justice'=>20,'legal'=>21,'rights'=>22,
    'attorney'=>23,'judge'=>24,'policy'=>25,'law'=>26,
    // Business (27-35)
    'money'=>27,'entrepreneur'=>28,'startup'=>29,'management'=>30,
    'finance'=>31,'marketing'=>32,'sales'=>33,'business'=>34,'profit'=>35,
    // Technology (36-45)
    'coding'=>36,'software'=>37,'programming'=>38,'computer'=>39,'ai'=>40,
    'data'=>41,'apps'=>42,'web'=>43,'developer'=>44,'technology'=>45,
    // Arts (46-55)
    'creative'=>46,'painting'=>47,'music'=>48,'film'=>49,'writing'=>50,
    'drama'=>51,'photography'=>52,'artist'=>53,'design'=>54,'arts'=>55,
    // Science (56-64)
    'research'=>56,'experiment'=>57,'chemistry'=>58,'physics'=>59,'biology'=>60,
    'lab'=>61,'discover'=>62,'scientist'=>63,'science'=>64,
    // Teaching (65-73)
    'students'=>65,'classroom'=>66,'education'=>67,'teach'=>68,'mentor'=>69,
    'school'=>70,'university'=>71,'professor'=>72,'teaching'=>73,
    // General intent words (74-89)
    'like'=>74,'want'=>75,'love'=>76,'interested'=>77,'good'=>78,
    'become'=>79,'career'=>80,'job'=>81,'work'=>82,'future'=>83,
    'study'=>84,'how'=>85,'what'=>86,'should'=>87,'help'=>88,'passion'=>89,
];
const VOCAB_SIZE  = 90;
const INTENT_NAMES = ['Engineering','Medicine','Law','Business',
                       'Technology','Arts','Science','Teaching'];
const NUM_INTENTS  = 8;

// ── DATASET ───────────────────────────────────────────────────────────────────
// 10 labelled samples per intent; last 2 of each are held out for testing.
const DATASET = [
    // 0 · Engineering
    ['i want to build machines and design robots', 0],
    ['i love mechanical engineering and structural design', 0],
    ['how do i become an electrical engineer', 0],
    ['i am interested in civil engineering and building bridges', 0],
    ['engineering robots and machines is my passion', 0],
    ['i want to work as an engineer building things', 0],
    ['how should i study to become an engineer', 0],
    ['building structural systems and electrical machines excites me', 0],
    ['i love the challenge of civil engineering and design', 0],      // test
    ['i want to design and build complex mechanical machines', 0],    // test

    // 1 · Medicine
    ['i want to become a doctor', 1],
    ['i am interested in medicine and healthcare', 1],
    ['how do i become a nurse working in hospitals', 1],
    ['i love taking care of patients in medical settings', 1],
    ['i want to study medicine and become a surgeon', 1],
    ['working in hospitals helping patients is my passion', 1],
    ['disease research and medical care interest me deeply', 1],
    ['i want to help patients as a doctor in a hospital', 1],
    ['how do i study to do surgery and become a medical doctor', 1],  // test
    ['health and patient care in hospitals drives me', 1],            // test

    // 2 · Law
    ['i want to become a lawyer', 2],
    ['i am interested in legal studies and court cases', 2],
    ['how do i become an attorney working for justice', 2],
    ['i want to become a judge in court', 2],
    ['law and legal rights are important to me', 2],
    ['i want to work on policy and legal issues', 2],
    ['i am interested in justice and human rights', 2],
    ['i love legal research and policy work', 2],
    ['defending rights in court as a lawyer is my dream', 2],         // test
    ['how do i study law to become an attorney for justice', 2],      // test

    // 3 · Business
    ['i want to start a business and become an entrepreneur', 3],
    ['i am interested in finance and money management', 3],
    ['how do i get into marketing and sales', 3],
    ['i love startups and building businesses', 3],
    ['business management and profit are my goals', 3],
    ['i want to work in finance and marketing', 3],
    ['i love entrepreneurship and startup culture', 3],
    ['growing sales and managing money interests me', 3],
    ['how should i study for a career in business management', 3],    // test
    ['i want to lead a startup and make it profitable', 3],           // test

    // 4 · Technology
    ['i love coding and building software apps', 4],
    ['i want to become a software developer', 4],
    ['programming and computer science is my passion', 4],
    ['i am interested in ai and data science', 4],
    ['how do i get into web development and technology', 4],
    ['i want to work with ai and build apps', 4],
    ['software programming and data interest me', 4],
    ['i love coding and technology and want to become a developer', 4],
    ['building web apps and ai systems is what i love', 4],           // test
    ['how do i become a computer software developer', 4],             // test

    // 5 · Arts
    ['i love painting and creative design', 5],
    ['i want to become an artist or musician', 5],
    ['film and drama are my greatest passions', 5],
    ['i am interested in photography and creative arts', 5],
    ['how do i pursue a career in music or writing', 5],
    ['i want to work in creative design and arts', 5],
    ['photography film and drama fascinate me', 5],
    ['writing music and drama are what i live for', 5],
    ['i love being creative with painting and arts design', 5],       // test
    ['i want to become a creative artist or photographer', 5],        // test

    // 6 · Science
    ['i love doing research and experiments in the lab', 6],
    ['chemistry and physics are my favourite subjects', 6],
    ['i want to become a scientist and discover new things', 6],
    ['biology and science research interest me deeply', 6],
    ['how do i become a research scientist', 6],
    ['i am passionate about lab experiments and biology', 6],
    ['discovering new things through science research excites me', 6],
    ['i want to work in a lab doing chemistry and physics', 6],
    ['science experiments in biology and chemistry inspire me', 6],   // test
    ['i want to study biology and become a research scientist', 6],   // test

    // 7 · Teaching
    ['i love working with students in the classroom', 7],
    ['i want to become a teacher or professor', 7],
    ['education and mentoring students is my passion', 7],
    ['how do i get into teaching at a university', 7],
    ['i want to teach and mentor young students', 7],
    ['working in education at schools interests me', 7],
    ['i love being a mentor in educational settings', 7],
    ['i want to dedicate my career to education and teaching', 7],
    ['classroom teaching and student mentorship are what i love', 7], // test
    ['how should i study to become a professor at university', 7],    // test
];

// ── RESPONSE BANK ─────────────────────────────────────────────────────────────
const RESPONSES = [
    0 => [
        "Engineering is where creativity meets precision — a great choice!",
        "Explore Mechanical, Electrical, Civil, or Software Engineering.",
        "Strong foundations: mathematics, physics, and CAD/simulation tools.",
        "Seek internships at automotive, aerospace, or construction firms early on.",
    ],
    1 => [
        "Medicine is one of the most rewarding and impactful careers.",
        "Excel in biology, chemistry, and physics at school.",
        "Consider a pre-med undergraduate, then medical school or nursing college.",
        "Specialisations: surgery, paediatrics, general practice, or psychiatry.",
    ],
    2 => [
        "Law is an excellent path for those passionate about justice and society.",
        "Build strong reading, writing, and critical-thinking skills.",
        "Specialise in criminal law, corporate law, or human-rights advocacy.",
        "Moot-court competitions and law-firm internships give you a real edge.",
    ],
    3 => [
        "Business and entrepreneurship open doors across every industry!",
        "Study commerce, economics, or an MBA to build strong foundations.",
        "Core skills: marketing, finance, leadership, and strategic thinking.",
        "Start a small project or intern at a startup to learn by doing.",
    ],
    4 => [
        "Tech careers are in high demand and growing faster than ever!",
        "Begin with programming fundamentals: Python, JavaScript, or C++.",
        "Specialise in AI/ML, cybersecurity, cloud computing, or full-stack dev.",
        "Build real projects, contribute to open-source, and grow your portfolio.",
    ],
    5 => [
        "Creative careers are deeply fulfilling — the world needs artists!",
        "Develop your portfolio and practise your craft every single day.",
        "Explore graphic design, music production, film-making, or creative writing.",
        "Arts school, online courses, and freelancing all build real experience.",
    ],
    6 => [
        "Science puts you at the frontier of human knowledge — exciting!",
        "Excel in mathematics, chemistry, physics, and biology.",
        "Aim for a research-focused university; consider a PhD for deep work.",
        "Lab internships, scholarships, and science olympiads set you apart.",
    ],
    7 => [
        "Teaching shapes the future — it's one of the most impactful careers.",
        "You'll need a subject degree plus a teaching certification (PGCE etc.).",
        "Decide: primary, secondary, or tertiary (university / college) education.",
        "Strong communication, patience, and mentorship skills are essential.",
    ],
];

// ── ENCODE ────────────────────────────────────────────────────────────────────
function encode(string $sentence): Tensor
{
    $vec = Tensor::zeros([VOCAB_SIZE]);
    foreach (explode(' ', strtolower(trim($sentence))) as $word) {
        $word = preg_replace('/[^a-z]/', '', $word);
        if ($word !== '' && isset(VOCAB[$word])) {
            $vec->buffer[VOCAB[$word]] = 1.0;
        }
    }
    return $vec;
}

/** Build a batch Tensor [N, D] and label array from a list of [sentence, label] pairs. */
function buildBatch(array $rows): array
{
    $N      = count($rows);
    $batch  = Tensor::zeros([$N, VOCAB_SIZE]);
    $labels = [];
    $ffi    = BlasEngine::get()->ffi;

    foreach ($rows as $i => [$sentence, $label]) {
        $vec = encode($sentence);
        $dst = \FFI::cast('float*', \FFI::addr($batch->buffer[$i * VOCAB_SIZE]));
        $ffi->cblas_scopy(VOCAB_SIZE, $vec->buffer, 1, $dst, 1);
        $labels[] = $label;
    }
    return [$batch, $labels];
}

// ── TINY MLP ─────────────────────────────────────────────────────────────────
/**
 * Two-layer MLP with manual SGD backpropagation.
 *
 *   Forward:  X[N,D] → hPre[N,H] = X@W1+b1 → h[N,H] = ReLU(hPre)
 *                    → logits[N,C] = h@W2+b2
 *   Backward: analytic gradients → SGD weight update
 */
final class TinyMLP
{
    public Tensor $W1; // [D, H]
    public Tensor $b1; // [H]
    public Tensor $W2; // [H, C]
    public Tensor $b2; // [C]

    public function __construct(int $D, int $H, int $C)
    {
        $this->W1 = Tensor::xavierInit([$D, $H]);
        $this->b1 = Tensor::zeros([$H]);
        $this->W2 = Tensor::xavierInit([$H, $C]);
        $this->b2 = Tensor::zeros([$C]);
    }

    /** Returns [logits[N,C], hPre[N,H], h[N,H]]. */
    public function forward(Tensor $X): array
    {
        // hPre = X @ W1 + b1
        $hPre = Ops::matmul($X, $this->W1);
        Ops::addBiasInPlace($hPre, $this->b1);

        // h = ReLU(hPre)
        $h = $hPre->clone();
        Ops::reluInPlace($h);

        // logits = h @ W2 + b2
        $logits = Ops::matmul($h, $this->W2);
        Ops::addBiasInPlace($logits, $this->b2);

        return [$logits, $hPre, $h];
    }

    /** Backprop + SGD update. Returns the scalar cross-entropy loss. */
    public function step(Tensor $X, array $targets, float $lr): float
    {
        $N = $X->shape[0];
        $H = $this->W1->shape[1];
        $C = $this->W2->shape[1];

        [$logits, $hPre, $h] = $this->forward($X);

        $loss = Ops::crossEntropyLoss($logits, $targets);

        // ── dL/dLogits = softmax(logits) − one_hot(y),  scaled by 1/N ───────
        $dL = $logits->clone();
        Ops::softmaxInPlace($dL);
        for ($i = 0; $i < $N; $i++) {
            $dL->buffer[$i * $C + $targets[$i]] -= 1.0;
        }
        for ($i = 0; $i < $dL->size; $i++) {
            $dL->buffer[$i] /= $N;
        }

        // ── Layer-2 gradients ─────────────────────────────────────────────────
        $dW2 = Ops::matmul($h, $dL, transA: true);           // [H, C]

        $dB2 = Tensor::zeros([$C]);
        $ffi = BlasEngine::get()->ffi;
        for ($i = 0; $i < $N; $i++) {
            $row = \FFI::cast('float*', \FFI::addr($dL->buffer[$i * $C]));
            $ffi->cblas_saxpy($C, 1.0, $row, 1, $dB2->buffer, 1);
        }

        // ── Backprop through ReLU ─────────────────────────────────────────────
        $dH = Ops::matmul($dL, $this->W2, transB: true);     // [N, H]
        for ($i = 0; $i < $dH->size; $i++) {
            if ($hPre->buffer[$i] <= 0.0) $dH->buffer[$i] = 0.0;
        }

        // ── Layer-1 gradients ─────────────────────────────────────────────────
        $dW1 = Ops::matmul($X, $dH, transA: true);           // [D, H]

        $dB1 = Tensor::zeros([$H]);
        for ($i = 0; $i < $N; $i++) {
            $row = \FFI::cast('float*', \FFI::addr($dH->buffer[$i * $H]));
            $ffi->cblas_saxpy($H, 1.0, $row, 1, $dB1->buffer, 1);
        }

        // ── SGD: θ ← θ − lr · ∇θ ────────────────────────────────────────────
        Ops::saxpy($dW1, $this->W1, -$lr);
        Ops::saxpy($dB1, $this->b1, -$lr);
        Ops::saxpy($dW2, $this->W2, -$lr);
        Ops::saxpy($dB2, $this->b2, -$lr);

        return $loss;
    }

    /** Classify a single encoded sentence vector [D] → intent index. */
    public function predict(Tensor $x): int
    {
        $batch = Tensor::zeros([1, VOCAB_SIZE]);
        BlasEngine::get()->ffi->cblas_scopy(VOCAB_SIZE, $x->buffer, 1, $batch->buffer, 1);
        [$logits,,] = $this->forward($batch);
        return Metrics::argmax($logits)[0];
    }

    /** Confidence scores (softmax) for a single input [D] → float[C]. */
    public function probabilities(Tensor $x): array
    {
        $batch = Tensor::zeros([1, VOCAB_SIZE]);
        BlasEngine::get()->ffi->cblas_scopy(VOCAB_SIZE, $x->buffer, 1, $batch->buffer, 1);
        [$logits,,] = $this->forward($batch);
        Ops::softmaxInPlace($logits);
        $probs = [];
        for ($i = 0; $i < NUM_INTENTS; $i++) {
            $probs[$i] = round((float)$logits->buffer[$i] * 100, 1);
        }
        return $probs;
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
//  MAIN
// ═══════════════════════════════════════════════════════════════════════════════
echo "\n{$BLD}Career Counselling Micro-SLM{$RST}  (Pml · PHP 2-layer MLP)\n";
echo "Architecture: BoW[" . VOCAB_SIZE . "] → Dense[128] → ReLU → Dense[" . NUM_INTENTS . "] → Softmax\n";

// ── Split dataset ─────────────────────────────────────────────────────────────
// Each intent has 10 samples; last 2 are test, first 8 are train.
$trainRows = [];
$testRows  = [];
$perIntent = 10;
foreach (array_chunk(DATASET, $perIntent) as $chunk) {
    $trainRows = array_merge($trainRows, array_slice($chunk, 0, 8));
    $testRows  = array_merge($testRows,  array_slice($chunk, 8));
}
// Shuffle training set
shuffle($trainRows);

[$Xtrain, $Ytrain] = buildBatch($trainRows);
[$Xtest,  $Ytest]  = buildBatch($testRows);

$nTrain = count($trainRows);  // 64
$nTest  = count($testRows);   // 16

echo "Dataset: {$nTrain} train / {$nTest} test samples · " . NUM_INTENTS . " career intents\n";

// ── Training ──────────────────────────────────────────────────────────────────
hdr('Training  (SGD · full-batch · 400 epochs)');

$model  = new TinyMLP(VOCAB_SIZE, 128, NUM_INTENTS);
$epochs = 400;
$lr     = 0.5;

$lossLog = [];
for ($epoch = 1; $epoch <= $epochs; $epoch++) {
    $loss = $model->step($Xtrain, $Ytrain, $lr);
    $lossLog[$epoch] = $loss;

    // Linear LR decay
    if ($epoch % 100 === 0) {
        $lr *= 0.5;
        info(sprintf("epoch %4d  loss=%.5f  lr=%.5f", $epoch, $loss, $lr));
    }
}

$finalLoss = end($lossLog);
ok(sprintf("Training complete  final loss=%.5f", $finalLoss));

// ── Evaluation ────────────────────────────────────────────────────────────────
hdr('Evaluation');

// Train set
[$trainLogits,,] = $model->forward($Xtrain);
$trainAcc  = Metrics::accuracy($trainLogits, $Ytrain);
$trainTop2 = Metrics::topKAccuracy($trainLogits, $Ytrain, k: 2);
$trainPPL  = Metrics::perplexity($trainLogits, $Ytrain);
ok(sprintf("Train  acc=%.1f%%  top-2=%.1f%%  perplexity=%.3f",
    $trainAcc * 100, $trainTop2 * 100, $trainPPL));

// Test set
[$testLogits,,] = $model->forward($Xtest);
$testAcc  = Metrics::accuracy($testLogits, $Ytest);
$testTop2 = Metrics::topKAccuracy($testLogits, $Ytest, k: 2);
$testPPL  = Metrics::perplexity($testLogits, $Ytest);
ok(sprintf("Test   acc=%.1f%%  top-2=%.1f%%  perplexity=%.3f",
    $testAcc * 100, $testTop2 * 100, $testPPL));

// Precision / Recall / F1
$yPredTest = Metrics::argmax($testLogits);
$prf       = Metrics::precisionRecallF1($yPredTest, $Ytest, NUM_INTENTS);
ok(sprintf("Test   macro-precision=%.3f  macro-recall=%.3f  macro-F1=%.3f",
    $prf['macro_precision'], $prf['macro_recall'], $prf['macro_f1']));

// Per-intent F1
hdr('Per-intent F1 (test set)');
foreach (INTENT_NAMES as $i => $name) {
    $bar = str_repeat('█', (int)round($prf['f1'][$i] * 20));
    printf("  %-14s  F1=%.2f  %s\n", $name, $prf['f1'][$i], $bar);
}

// Confusion matrix
hdr('Confusion Matrix (test set — rows=true, cols=pred)');
$yPredTest = Metrics::argmax($testLogits);
$cm        = Metrics::confusionMatrix($yPredTest, $Ytest, NUM_INTENTS);

$header = sprintf("  %14s", '');
foreach (INTENT_NAMES as $name) {
    $header .= sprintf(" %5s", substr($name, 0, 5));
}
echo $header . "\n";

foreach (INTENT_NAMES as $i => $rowName) {
    $line = sprintf("  %-14s", $rowName);
    for ($j = 0; $j < NUM_INTENTS; $j++) {
        $v     = (int)$cm->get($i, $j);
        $mark  = $i === $j ? ($v > 0 ? "\033[32m" : "\033[31m") : ($v > 0 ? "\033[33m" : '');
        $line .= sprintf("  %s%4d%s ", $mark, $v, $v > 0 ? "\033[0m" : '');
    }
    echo $line . "\n";
}

// ── Loss curve summary ────────────────────────────────────────────────────────
hdr('Loss Curve (every 50 epochs)');
$prev = null;
foreach ($lossLog as $ep => $l) {
    if ($ep % 50 !== 0) continue;
    $bar   = str_repeat('▓', max(1, (int)($l * 30)));
    $delta = $prev !== null ? sprintf("  Δ=%.5f", $prev - $l) : '';
    printf("  epoch %3d  %.5f  %s%s\n", $ep, $l, $bar, $delta);
    $prev = $l;
}

// ── Interactive REPL ──────────────────────────────────────────────────────────
hdr('Interactive Career Counsellor');
echo "  Type a career question and press Enter. Type 'quit' to exit.\n\n";

while (true) {
    echo "{$BLD}  You:{$RST} ";
    $line = fgets(STDIN);
    if ($line === false) break;
    $line = trim($line);
    if ($line === '' ) continue;
    if (in_array(strtolower($line), ['quit', 'exit', 'q'], true)) break;

    $vec    = encode($line);
    $intent = $model->predict($vec);
    $probs  = $model->probabilities($vec);

    // Top-2 intents
    arsort($probs);
    $top2Keys = array_slice(array_keys($probs), 0, 2);

    echo "\n{$CYN}  Detected intent:{$RST} {$BLD}" . INTENT_NAMES[$intent] . "{$RST}";
    echo "  (confidence: {$probs[$intent]}%)\n";

    // Show top-2 if second is close
    if (count($top2Keys) > 1 && $top2Keys[0] !== $top2Keys[1]) {
        $second = $top2Keys[1];
        if ($probs[$second] > 10.0) {
            echo "  {$YLW}Also considering:{$RST} " . INTENT_NAMES[$second]
                 . " ({$probs[$second]}%)\n";
        }
    }

    echo "\n";
    foreach (RESPONSES[$intent] as $line) {
        echo "  {$GRN}»{$RST} {$line}\n";
    }
    echo "\n";
}

echo "\n{$GRN}{$BLD}Good luck with your career!{$RST}\n\n";
