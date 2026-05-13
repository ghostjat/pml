<?php
declare(strict_types=1);
/**
 * VIDEO MOTION DETECTION — Retail Loss Prevention
 * ═══════════════════════════════════════════════════════════════════
 * Problem  : Detect anomalous motion events in a CCTV video stream:
 *              • Loitering (person stationary in restricted zone)
 *              • Fast movement (grab-and-run)
 *              • Unusual trajectory (erratic path)
 *            Normal motion: shoppers walking at typical pace.
 *
 * Method   : Per-frame feature extraction from synthetic video frames
 *            (frame differencing, optical-flow proxies, region stats)
 *            → IsolationForest for unsupervised anomaly detection.
 *
 *            No labelled anomaly data is required — we train only on
 *            normal shopping traffic, so any deviation is flagged.
 *
 * Business : Retail shrinkage costs $100 B/year globally.  AI loss
 *            prevention cuts shrinkage by 25–40 % without increasing
 *            store security headcount.
 * ═══════════════════════════════════════════════════════════════════
 */

require_once __DIR__ . '/../bootstrap.php';

use Pml\Dataset;
use Pml\Estimators\AnomalyDetectors\IsolationForest;
use Pml\Estimators\AnomalyDetectors\RobustZScore;

section('Video Motion Detection — Retail Loss Prevention');

mt_srand(55);
$rng = fn(float $lo, float $hi) => $lo + (mt_rand() / mt_getrandmax()) * ($hi - $lo);

// ── Constants ─────────────────────────────────────────────────────────────────
const VID_H = 32;
const VID_W = 32;
const FPS   = 10;   // simulated frames per second

// ── 1. Synthetic video frame generator ───────────────────────────────────────
/**
 * Each "person" in the scene has position, velocity, and motion_type.
 * Returns [VID_H × VID_W] occupancy map (0.0=empty, 1.0=person detected).
 */
class ShoppingScene
{
    private array $persons = [];  // [cx, cy, vx, vy, type]

    public function __construct(callable $rng, int $nPersons = 3)
    {
        for ($i = 0; $i < $nPersons; $i++) {
            $this->persons[] = [
                'cx' => $rng(4, VID_W - 4),
                'cy' => $rng(4, VID_H - 4),
                'vx' => $rng(-1.5, 1.5),
                'vy' => $rng(-1.5, 1.5),
                'type' => 'normal',
            ];
        }
    }

    public function addAnomalousPerson(callable $rng, string $type): void
    {
        $this->persons[] = match ($type) {
            'loitering' => [
                'cx' => $rng(10, 22), 'cy' => $rng(10, 22),
                'vx' => $rng(-0.1, 0.1), 'vy' => $rng(-0.1, 0.1),
                'type' => 'loitering',
            ],
            'fast_move' => [
                'cx' => $rng(2, VID_W - 2), 'cy' => $rng(2, VID_H - 2),
                'vx' => $rng(5, 9) * (mt_rand(0, 1) ? 1 : -1),
                'vy' => $rng(2, 5) * (mt_rand(0, 1) ? 1 : -1),
                'type' => 'fast_move',
            ],
            'erratic' => [
                'cx' => $rng(4, VID_W - 4), 'cy' => $rng(4, VID_H - 4),
                'vx' => $rng(-6, 6), 'vy' => $rng(-6, 6),
                'type' => 'erratic',
            ],
            default => ['cx' => 16, 'cy' => 16, 'vx' => 0, 'vy' => 0, 'type' => 'normal'],
        };
    }

    public function step(callable $rng): void
    {
        foreach ($this->persons as &$p) {
            if ($p['type'] === 'erratic') {
                $p['vx'] += $rng(-3, 3);
                $p['vy'] += $rng(-3, 3);
                $p['vx'] = max(-8, min(8, $p['vx']));
                $p['vy'] = max(-8, min(8, $p['vy']));
            }
            $p['cx'] += $p['vx'] / FPS;
            $p['cy'] += $p['vy'] / FPS;
            // Bounce off walls
            if ($p['cx'] < 2 || $p['cx'] > VID_W - 2) { $p['vx'] *= -1; $p['cx'] = max(2, min(VID_W-2, $p['cx'])); }
            if ($p['cy'] < 2 || $p['cy'] > VID_H - 2) { $p['vy'] *= -1; $p['cy'] = max(2, min(VID_H-2, $p['cy'])); }
        }
        unset($p);
    }

    public function frame(): array
    {
        $f = array_fill(0, VID_H * VID_W, 0.0);
        foreach ($this->persons as $p) {
            $r = (int)round($p['cy']); $c = (int)round($p['cx']);
            for ($dr = -1; $dr <= 1; $dr++) {
                for ($dc = -1; $dc <= 1; $dc++) {
                    $rr = $r + $dr; $cc = $c + $dc;
                    if ($rr >= 0 && $rr < VID_H && $cc >= 0 && $cc < VID_W) {
                        $f[$rr * VID_W + $cc] = min(1.0, $f[$rr * VID_W + $cc] + 0.5);
                    }
                }
            }
        }
        return $f;
    }
}

// ── 2. Feature extraction from consecutive frames ─────────────────────────────
/**
 * Extracts motion features from 3 consecutive frames (t-1, t, t+1):
 *  - Frame diff magnitude (mean absolute change)
 *  - Number of active pixels (motion blobs)
 *  - Motion centroid (where is movement happening)
 *  - Motion spread (standard deviation of active region)
 *  - Max motion magnitude
 *  - Temporal consistency (diff between consecutive diffs)
 *  - Region stats: activity in 4 quadrants
 */
function motionFeatures(array $prev, array $curr, array $next): array
{
    $n = count($curr);
    $H = VID_H; $W = VID_W;

    // Frame differences
    $diff1 = []; $diff2 = [];
    $totalMag = 0.0; $maxMag = 0.0; $activePx = 0;
    $sumR = 0.0; $sumC = 0.0; $sumR2 = 0.0; $sumC2 = 0.0;
    $q = [0.0, 0.0, 0.0, 0.0];  // quadrant activity

    for ($i = 0; $i < $n; $i++) {
        $d1 = abs($curr[$i] - $prev[$i]);
        $d2 = abs($next[$i] - $curr[$i]);
        $diff1[] = $d1; $diff2[] = $d2;
        $mag = ($d1 + $d2) / 2;
        $totalMag += $mag; $maxMag = max($maxMag, $mag);
        if ($mag > 0.05) {
            $activePx++;
            $r = (int)($i / $W); $c = $i % $W;
            $sumR += $r; $sumC += $c;
            $sumR2 += $r * $r; $sumC2 += $c * $c;
            $q[($r < $H/2 ? 0 : 2) + ($c < $W/2 ? 0 : 1)] += $mag;
        }
    }

    $meanMag = $totalMag / $n;
    $centR   = $activePx > 0 ? $sumR / $activePx / $H : 0.5;
    $centC   = $activePx > 0 ? $sumC / $activePx / $W : 0.5;
    $varR    = $activePx > 0 ? max(0, $sumR2 / $activePx - ($sumR / $activePx) ** 2) / ($H * $H) : 0;
    $varC    = $activePx > 0 ? max(0, $sumC2 / $activePx - ($sumC / $activePx) ** 2) / ($W * $W) : 0;
    $spread  = sqrt($varR + $varC);

    // Temporal consistency: how much do the two diffs agree
    $consistency = 0.0;
    for ($i = 0; $i < $n; $i++) {
        $consistency += abs($diff2[$i] - $diff1[$i]);
    }
    $consistency /= $n;

    return [
        $meanMag,
        $maxMag,
        $activePx / $n,
        $centR,
        $centC,
        $spread,
        $consistency,
        $q[0] / max(1e-6, $totalMag),  // top-left quadrant share
        $q[1] / max(1e-6, $totalMag),  // top-right
        $q[2] / max(1e-6, $totalMag),  // bottom-left
        $q[3] / max(1e-6, $totalMag),  // bottom-right
    ];
}

// ── 3. Generate normal traffic (training data) ────────────────────────────────
section('Simulating Normal Shopping Traffic (Training)');

$normalFeatures = [];
$nNormalClips   = 4000;  // 14-min equivalent of 10 FPS

for ($clip = 0; $clip < $nNormalClips; $clip++) {
    $scene = new ShoppingScene($rng, mt_rand(1, 4));
    $frames = [];
    for ($f = 0; $f < 3; $f++) {
        $scene->step($rng);
        $frames[] = $scene->frame();
    }
    $normalFeatures[] = motionFeatures($frames[0], $frames[1], $frames[2]);
}

metric('Normal frames (training)', $nNormalClips);
metric('Motion features per frame', count($normalFeatures[0]));

// ── 4. Train anomaly detectors ────────────────────────────────────────────────
section('Training Anomaly Detectors');
$t0 = microtime(true);

$trainDs   = Dataset::fromArray($normalFeatures);
$isoForest = new IsolationForest(nEstimators: 100, sampleSize: 256, contamination: 0.01);
$isoForest->train($trainDs);

$robustZ   = new RobustZScore(threshold: 4.0);
$robustZ->train($trainDs);

metric('Training time', elapsed($t0));

// ── 5. Simulate live CCTV feed with injected incidents ───────────────────────
section('Live CCTV Feed Simulation');

$scenarios = [
    ['type' => 'normal',    'anomaly_type' => null,       'expected' => false],
    ['type' => 'normal',    'anomaly_type' => null,       'expected' => false],
    ['type' => 'incident',  'anomaly_type' => 'loitering','expected' => true],
    ['type' => 'normal',    'anomaly_type' => null,       'expected' => false],
    ['type' => 'incident',  'anomaly_type' => 'fast_move','expected' => true],
    ['type' => 'normal',    'anomaly_type' => null,       'expected' => false],
    ['type' => 'incident',  'anomaly_type' => 'erratic',  'expected' => true],
    ['type' => 'normal',    'anomaly_type' => null,       'expected' => false],
];

printf("\n  %-12s | %-12s | %-8s | %-8s | %s\n",
       'Time', 'Event', 'IsoScore', 'ZScore', 'Alert');
printf("  %s\n", str_repeat('-', 70));

$detected = 0; $falsePos = 0;
foreach ($scenarios as $ti => $sc) {
    $scene = new ShoppingScene($rng, mt_rand(1, 3));
    if ($sc['anomaly_type'] !== null) {
        $scene->addAnomalousPerson($rng, $sc['anomaly_type']);
    }
    $frames = [];
    for ($f = 0; $f < 3; $f++) { $scene->step($rng); $frames[] = $scene->frame(); }
    $feats = motionFeatures($frames[0], $frames[1], $frames[2]);

    $ds       = Dataset::fromArray([$feats]);
    $isoScore = $isoForest->predict($ds)->toFlatArray()[0] ?? 0.0;
    $zScore   = $robustZ->predict($ds)->toFlatArray()[0]   ?? 0.0;
    $alarm    = $isoScore > 0.5 || $zScore > 0.5;

    if ($alarm && $sc['expected'])  $detected++;
    if ($alarm && !$sc['expected']) $falsePos++;

    $timeStr = sprintf('%02d:%02d', (int)($ti * 3.5), ($ti * 35) % 60);
    $evtStr  = $sc['anomaly_type'] ?? 'normal';
    $icon    = match(true) {
        $alarm && $sc['expected']  => '🚨 ALERT',
        $alarm && !$sc['expected'] => '⚠️  FP',
        !$alarm && $sc['expected'] => '❌ MISSED',
        default                    => '✅ OK',
    };
    printf("  %-12s | %-12s | %-8.4f | %-8.4f | %s\n",
           $timeStr, $evtStr, $isoScore, $zScore, $icon);
}

$nIncidents = count(array_filter($scenarios, fn($s) => $s['expected']));
printf("\n  Detection rate : %d/%d incidents\n", $detected, $nIncidents);
printf("  False positives: %d\n", $falsePos);

// ── 6. Deployment pattern ─────────────────────────────────────────────────────
section('Deployment Pattern');
echo <<<TXT
  // Per-frame pipeline (runs at camera FPS):
  function analyseFrame(array \$prevFrame, array \$currFrame, array \$nextFrame,
                        IsolationForest \$detector): bool {
      \$features = motionFeatures(\$prevFrame, \$currFrame, \$nextFrame);
      \$score = \$detector->predict(Dataset::fromArray([\$features]))->toFlatArray()[0];
      if (\$score > 0.65) {
          \$this->alerts->fire(type: 'motion_anomaly', score: \$score, ts: time());
      }
      return \$score > 0.65;
  }
TXT;

echo "\n✓ Done\n";
