<?php
declare(strict_types=1);

namespace Pml\Generation;

use Pml\{Tensor,Ops};


// ═══════════════════════════════════════════════════════════════════════════
//  SAMPLER
//  Converts raw logits into a discrete token ID.
// ═══════════════════════════════════════════════════════════════════════════

final class Sampler
{
    /**
     * Greedy decoding: always pick the most likely token.
     */
    public static function greedy(Tensor $logits): int
    {
        return $logits->argmax();
    }

    /**
     * Temperature sampling + optional top-k + top-p (nucleus) filtering.
     *
     * @param float $temperature  0.0 = greedy, >0.0 = stochastic
     * @param int   $topK         0 = disabled, >0 = keep top-k tokens
     * @param float $topP         1.0 = disabled, <1.0 = nucleus sampling
     */
    public static function sample(
        Tensor $logits,
        float  $temperature = 0.7,
        int    $topK        = 0,
        float  $topP        = 1.0
    ): int {
        if ($temperature <= 0.0) {
            return self::greedy($logits);
        }

        $vocabSize = $logits->size;
        $raw       = $logits->toArray(); // must cross boundary once for sorting

        // ── 1. Temperature scaling ─────────────────────────────────────────
        $invTemp = 1.0 / $temperature;
        $scaled  = array_map(fn($v) => $v * $invTemp, $raw);

        // ── 2. Numerical stability: subtract max ───────────────────────────
        $maxVal = max($scaled);
        $probs  = array_map(fn($v) => exp($v - $maxVal), $scaled);

        // ── 3. Top-K filtering ─────────────────────────────────────────────
        if ($topK > 0 && $topK < $vocabSize) {
            $sorted = $probs;
            arsort($sorted);
            $cutoff = array_keys($sorted)[$topK - 1];
            $thresh = $sorted[$cutoff];
            foreach ($probs as $i => &$p) {
                if ($p < $thresh) $p = 0.0;
            }
            unset($p);
        }

        // ── 4. Top-P (nucleus) filtering ──────────────────────────────────
        if ($topP < 1.0) {
            $sorted = $probs;
            arsort($sorted);
            $cumulative = 0.0;
            $pivotIdx   = count($sorted) - 1;
            foreach ($sorted as $idx => $p) {
                $cumulative += $p;
                if ($cumulative >= $topP) {
                    $pivotIdx = $idx;
                    break;
                }
            }
            $thresh = $sorted[$pivotIdx];
            foreach ($probs as $i => &$p) {
                if ($p < $thresh) $p = 0.0;
            }
            unset($p);
        }

        // ── 5. Normalize → CDF sampling ───────────────────────────────────
        $sum = array_sum($probs);
        if ($sum <= 0.0) return self::greedy($logits); // Fallback

        $r          = (mt_rand() / mt_getrandmax()) * $sum;
        $cumulative = 0.0;
        foreach ($probs as $i => $p) {
            $cumulative += $p;
            if ($r <= $cumulative) return $i;
        }

        return $vocabSize - 1;
    }

    /**
     * Beam search: maintains $beamWidth candidate sequences.
     * Returns the top sequence as an int[].
     * This is a simplified single-step beam for architecture demonstration.
     *
     * @param callable $forwardFn  fn(int[] $tokens) → Tensor logits[vocab_size]
     */
    public static function beamSearch(
        array    $inputTokens,
        callable $forwardFn,
        int      $maxNewTokens,
        int      $beamWidth    = 4,
        int      $eosTokenId   = 2
    ): array {
        // Each beam: [tokens[], cumLogProb]
        $beams = [[$inputTokens, 0.0]];

        for ($step = 0; $step < $maxNewTokens; $step++) {
            $candidates = [];

            foreach ($beams as [$tokens, $logProb]) {
                $logits   = $forwardFn($tokens);
                $logSoftmax = Ops::logSoftmax($logits->unsqueeze(0))->squeeze()->toArray();

                // Expand top-k candidates from this beam
                arsort($logSoftmax);
                $topK = array_slice(array_keys($logSoftmax), 0, $beamWidth, true);

                foreach ($topK as $tokenId) {
                    $newTokens = array_merge($tokens, [$tokenId]);
                    $newScore  = $logProb + $logSoftmax[$tokenId];
                    $candidates[] = [$newTokens, $newScore, $tokenId === $eosTokenId];
                }
            }

            // Sort by log-prob (descending) and prune to beam width
            usort($candidates, fn($a, $b) => $b[1] <=> $a[1]);
            $beams = [];
            foreach (array_slice($candidates, 0, $beamWidth) as [$tokens, $score, $done]) {
                $beams[] = [$tokens, $score];
                if ($done) break 2;
            }
        }

        return $beams[0][0]; // Best sequence
    }
}
