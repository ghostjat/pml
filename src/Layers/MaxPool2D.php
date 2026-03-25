<?php

declare(strict_types=1);

namespace Pml\Layers;

use Pml\Tensor;

// ═══════════════════════════════════════════════════════════════════════════
//  MaxPool2D — 2-D Max-Pooling Layer
//
//  ── Tensor Format: NCHW ──────────────────────────────────────────────────
//
//  Input:  [B, C, H_in,  W_in ]
//  Output: [B, C, H_out, W_out]
//
//  H_out = floor((H_in - kH) / stride_h) + 1
//  W_out = floor((W_in - kW) / stride_w) + 1
//
//  ── Forward Pass ─────────────────────────────────────────────────────────
//
//  For each output position (b, c, h, w):
//
//    out[b, c, h, w] = max_{kh, kw}  input[b, c, h·sH + kh, w·sW + kw]
//
//  The argmax index (flat within the input spatial slice) is stored so the
//  backward pass can route the gradient exclusively to the winning element.
//
//  ── Backward Pass ─────────────────────────────────────────────────────────
//
//  Given dout [B, C, H_out, W_out]:
//
//  For each output position (b, c, h, w):
//
//    dinput[b, c, argmax_h, argmax_w] += dout[b, c, h, w]
//
//  All non-winning input positions in each pool window receive zero gradient.
//  This is the "switch" backward (Springenberg et al. 2014 / standard MaxPool).
//
//  ── Complexity ────────────────────────────────────────────────────────────
//
//  Forward  : O(B · C · kH · kW · H_out · W_out)  — PHP loop (comparison)
//  Backward : O(B · C · H_out · W_out)            — scatter-add PHP loop
// ═══════════════════════════════════════════════════════════════════════════

final class MaxPool2D
{
    // ── Hyper-parameters ──────────────────────────────────────────────────

    public readonly int $kernel_h;
    public readonly int $kernel_w;
    public readonly int $stride_h;
    public readonly int $stride_w;

    // ── Constructor ───────────────────────────────────────────────────────

    /**
     * @param int|int[] $kernel_size  Pooling window size (scalar or [kH, kW]).
     * @param int|int[] $stride       Stride (default = kernel_size for non-overlapping pools).
     */
    public function __construct(
        int|array $kernel_size,
        int|array $stride = 0,   // 0 sentinel → same as kernel_size
    ) {
        [$this->kernel_h, $this->kernel_w] = self::pair($kernel_size);

        // If stride is not specified, default to kernel_size (standard behaviour)
        if ($stride === 0 || $stride === [0, 0]) {
            $this->stride_h = $this->kernel_h;
            $this->stride_w = $this->kernel_w;
        } else {
            [$this->stride_h, $this->stride_w] = self::pair($stride);
        }
    }

    // ── Forward pass ──────────────────────────────────────────────────────

    /**
     * Apply max-pooling and track argmax indices for the backward pass.
     *
     * @param  Tensor $input  [B, C, H_in, W_in]
     * @return array{Tensor, array}  [output [B, C, H_out, W_out], cache]
     */
    public function forward(Tensor $input): array
    {
        // ── Shape assertion ────────────────────────────────────────────────
        if (count($input->shape) !== 4) {
            throw new \InvalidArgumentException(
                'MaxPool2D::forward(): input must be 4-D [B, C, H, W], got rank '
                . count($input->shape) . '.'
            );
        }

        [$B, $C, $H, $W] = $input->shape;

        $kH = $this->kernel_h;
        $kW = $this->kernel_w;
        $sH = $this->stride_h;
        $sW = $this->stride_w;

        $H_out = intdiv($H - $kH, $sH) + 1;
        $W_out = intdiv($W - $kW, $sW) + 1;

        if ($H_out <= 0 || $W_out <= 0) {
            throw new \InvalidArgumentException(
                "MaxPool2D: kernel ({$kH}×{$kW}) with stride ({$sH},{$sW}) produces "
                . "non-positive output size for input ({$H}×{$W})."
            );
        }

        $output  = new Tensor([$B, $C, $H_out, $W_out]);
        // argmax stores the flat index within input[b, c, :, :] for each output position
        // Shape: [$B, $C, $H_out, $W_out] — same as output
        $argmax  = new \SplFixedArray($B * $C * $H_out * $W_out);

        $outIdx = 0;

        for ($b = 0; $b < $B; $b++) {
            $bBase = $b * $C * $H * $W;
            for ($c = 0; $c < $C; $c++) {
                $cBase = $bBase + $c * $H * $W;
                for ($h = 0; $h < $H_out; $h++) {
                    $hStart = $h * $sH;
                    for ($w = 0; $w < $W_out; $w++) {
                        $wStart = $w * $sW;

                        // Find the maximum within the kH×kW window
                        $maxVal   = -INF;
                        $maxFlatH = $hStart;   // flat row within [H, W] slice
                        $maxFlatW = $wStart;

                        for ($kh = 0; $kh < $kH; $kh++) {
                            $h_in = $hStart + $kh;
                            for ($kw = 0; $kw < $kW; $kw++) {
                                $w_in = $wStart + $kw;
                                $val  = (float) $input->buffer[$cBase + $h_in * $W + $w_in];
                                if ($val > $maxVal) {
                                    $maxVal   = $val;
                                    $maxFlatH = $h_in;
                                    $maxFlatW = $w_in;
                                }
                            }
                        }

                        $output->buffer[$outIdx] = $maxVal;
                        // Store flat offset within the FULL input buffer for direct use in backward
                        $argmax[$outIdx]          = $cBase + $maxFlatH * $W + $maxFlatW;
                        $outIdx++;
                    }
                }
            }
        }

        $cache = [
            'argmax' => $argmax,
            'B'      => $B,
            'C'      => $C,
            'H'      => $H,
            'W'      => $W,
            'H_out'  => $H_out,
            'W_out'  => $W_out,
        ];

        return [$output, $cache];
    }

    // ── Backward pass ─────────────────────────────────────────────────────

    /**
     * Route gradients back through the argmax positions.
     *
     * @param Tensor $dout   Upstream gradient [B, C, H_out, W_out].
     * @param array  $cache  Cache returned by forward().
     * @return Tensor  dinput [B, C, H_in, W_in]
     */
    public function backward(Tensor $dout, array $cache): Tensor
    {
        $argmax = $cache['argmax'];
        $B      = $cache['B'];
        $C      = $cache['C'];
        $H      = $cache['H'];
        $W      = $cache['W'];
        $H_out  = $cache['H_out'];
        $W_out  = $cache['W_out'];

        // ── Dimension assertion ────────────────────────────────────────────
        if ($dout->shape !== [$B, $C, $H_out, $W_out]) {
            throw new \InvalidArgumentException(
                'MaxPool2D::backward(): dout shape mismatch. Expected ['
                . implode(',', [$B, $C, $H_out, $W_out]) . '], got ['
                . implode(',', $dout->shape) . '].'
            );
        }

        $dinput = new Tensor([$B, $C, $H, $W]);   // zero-initialised

        $n = $B * $C * $H_out * $W_out;
        for ($i = 0; $i < $n; $i++) {
            // argmax[$i] is the flat index within the FULL input buffer
            $flatIdx = $argmax[$i];
            $dinput->buffer[$flatIdx] =
                (float) $dinput->buffer[$flatIdx] + (float) $dout->buffer[$i];
        }

        return $dinput;
    }

    // ── Helpers ───────────────────────────────────────────────────────────

    /**
     * Convert int or [int, int] to a [height, width] pair.
     */
    private static function pair(int|array $v): array
    {
        return is_array($v) ? [$v[0], $v[1]] : [$v, $v];
    }
}
