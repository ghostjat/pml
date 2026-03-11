<?php 
declare(strict_types=1);
namespace Pml;

// ═══════════════════════════════════════════════════════════════════════════
//  PURE-PHP LINEAR ALGEBRA UTILITIES
//  (Used as fallback when FFI/LAPACK unavailable)
// ═══════════════════════════════════════════════════════════════════════════

final class LinAlg
{
    private const BLOCK = 32;

    // ── matrix multiply (tiled, 4× unrolled) ─────────────────────────────

    /** Multiply A(m×k) · B(k×n), both as array-of-arrays. Returns array-of-arrays. */
    public static function matmul(array $A, array $B, int $m, int $k, int $n): array
    {
        // Transpose B for cache-friendly access
        $BT = self::transposeRaw($B, $k, $n);
        $C  = array_fill(0, $m, null);
        for ($i = 0; $i < $m; ++$i) $C[$i] = array_fill(0, $n, 0.0);

        $bl = self::BLOCK;
        for ($ii = 0; $ii < $m; $ii += $bl) {
            $iMx = min($ii + $bl, $m);
            for ($jj = 0; $jj < $n; $jj += $bl) {
                $jMx = min($jj + $bl, $n);
                for ($kk = 0; $kk < $k; $kk += $bl) {
                    $kMx  = min($kk + $bl, $k);
                    $kMx4 = $kk + (int)(($kMx - $kk) / 4) * 4;
                    for ($i = $ii; $i < $iMx; ++$i) {
                        $Ar = $A[$i];
                        $Cr = &$C[$i];
                        for ($j = $jj; $j < $jMx; ++$j) {
                            $BTr = $BT[$j];
                            $s0 = $s1 = $s2 = $s3 = 0.0;
                            for ($kp = $kk; $kp < $kMx4; $kp += 4) {
                                $s0 += $Ar[$kp]   * $BTr[$kp];
                                $s1 += $Ar[$kp+1] * $BTr[$kp+1];
                                $s2 += $Ar[$kp+2] * $BTr[$kp+2];
                                $s3 += $Ar[$kp+3] * $BTr[$kp+3];
                            }
                            $Cr[$j] += $s0 + $s1 + $s2 + $s3;
                            for ($kp = $kMx4; $kp < $kMx; ++$kp)
                                $Cr[$j] += $Ar[$kp] * $BTr[$kp];
                        }
                    }
                }
            }
        }
        return $C;
    }

    /** Matrix × vector: A(m×n) · x(n) → y(m) flat array */
    public static function matvec(array $A, array $x, int $m, int $n): array
    {
        $y = array_fill(0, $m, 0.0);
        for ($i = 0; $i < $m; ++$i) {
            $s = 0.0;
            $r = $A[$i];
            for ($j = 0; $j < $n; ++$j) $s += $r[$j] * $x[$j];
            $y[$i] = $s;
        }
        return $y;
    }

    // ── transpose ─────────────────────────────────────────────────────────

    /** Transpose m×n array-of-arrays → n×m */
    public static function transposeRaw(array $M, int $m, int $n): array
    {
        $T = [];
        for ($j = 0; $j < $n; ++$j) $T[$j] = array_fill(0, $m, 0.0);
        for ($i = 0; $i < $m; ++$i)
            for ($j = 0; $j < $n; ++$j)
                $T[$j][$i] = $M[$i][$j];
        return $T;
    }

    // ── LU decomposition (in-place, partial pivoting) ─────────────────────

    /**
     * Returns [L_and_U_combined, pivot_indices]
     * L is unit lower-triangular (diagonal 1s stored implicitly),
     * U is upper-triangular.
     * Both stored in the same n×n array (standard LAPACK layout).
     */
    public static function lu(array $A, int $n): array
    {
        $LU   = $A;  // copy
        $ipiv = array_fill(0, $n, 0);

        for ($k = 0; $k < $n; ++$k) {
            // find pivot
            $maxVal = abs($LU[$k][$k]);
            $maxRow = $k;
            for ($i = $k+1; $i < $n; ++$i) {
                if (($v = abs($LU[$i][$k])) > $maxVal) { $maxVal = $v; $maxRow = $i; }
            }
            $ipiv[$k] = $maxRow;
            if ($maxRow !== $k) {  // swap rows
                [$LU[$k], $LU[$maxRow]] = [$LU[$maxRow], $LU[$k]];
            }
            if (abs($LU[$k][$k]) < 1e-15) continue;  // singular
            $invPivot = 1.0 / $LU[$k][$k];
            for ($i = $k+1; $i < $n; ++$i) {
                $LU[$i][$k] *= $invPivot;
                $lik = $LU[$i][$k];
                for ($j = $k+1; $j < $n; ++$j)
                    $LU[$i][$j] -= $lik * $LU[$k][$j];
            }
        }
        return [$LU, $ipiv];
    }

    /** Determinant from LU factorization */
    public static function det(array $A, int $n): float
    {
        [$LU, $ipiv] = self::lu($A, $n);
        $det  = 1.0;
        $swaps = 0;
        for ($i = 0; $i < $n; ++$i) {
            $det *= $LU[$i][$i];
            if ($ipiv[$i] !== $i) ++$swaps;
        }
        return ($swaps % 2 === 0) ? $det : -$det;
    }

    /** Solve A*x = b using pre-computed LU. b is a flat array. */
    public static function luSolve(array $LU, array $ipiv, array $b, int $n): array
    {
        $x = $b;
        // apply pivots
        for ($i = 0; $i < $n; ++$i)
            if ($ipiv[$i] !== $i) [$x[$i], $x[$ipiv[$i]]] = [$x[$ipiv[$i]], $x[$i]];
        // forward substitution (L is unit lower-triangular)
        for ($i = 1; $i < $n; ++$i) {
            for ($j = 0; $j < $i; ++$j)
                $x[$i] -= $LU[$i][$j] * $x[$j];
        }
        // back substitution (U)
        for ($i = $n-1; $i >= 0; --$i) {
            for ($j = $i+1; $j < $n; ++$j)
                $x[$i] -= $LU[$i][$j] * $x[$j];
            $x[$i] /= $LU[$i][$i];
        }
        return $x;
    }

    /** Inverse via LU */
    public static function inverse(array $A, int $n): array
    {
        [$LU, $ipiv] = self::lu($A, $n);
        $inv = [];
        for ($j = 0; $j < $n; ++$j) {
            $e = array_fill(0, $n, 0.0);
            $e[$j] = 1.0;
            $col = self::luSolve($LU, $ipiv, $e, $n);
            for ($i = 0; $i < $n; ++$i) $inv[$i][$j] = $col[$i];
        }
        return $inv;
    }

    /** Solve A*X = B where B is array-of-arrays (n×nrhs) */
    public static function solve(array $A, array $B, int $n, int $nrhs): array
    {
        [$LU, $ipiv] = self::lu($A, $n);
        $X = [];
        for ($j = 0; $j < $nrhs; ++$j) {
            $b = array_column($B, $j);
            $x = self::luSolve($LU, $ipiv, $b, $n);
            for ($i = 0; $i < $n; ++$i) $X[$i][$j] = $x[$i];
        }
        return $X;
    }

    // ── Cholesky decomposition ─────────────────────────────────────────────

    /** Cholesky-Banachiewicz: A = L*L^T, returns L (lower triangular) */
    public static function cholesky(array $A, int $n): array
    {
        $L = array_fill(0, $n, null);
        for ($i = 0; $i < $n; ++$i) $L[$i] = array_fill(0, $n, 0.0);

        for ($i = 0; $i < $n; ++$i) {
            for ($j = 0; $j <= $i; ++$j) {
                $s = $A[$i][$j];
                for ($k = 0; $k < $j; ++$k) $s -= $L[$i][$k] * $L[$j][$k];
                if ($i === $j) {
                    if ($s < 0.0) throw new \RuntimeException("Matrix is not positive definite");
                    $L[$i][$j] = sqrt($s);
                } else {
                    $L[$i][$j] = $s / $L[$j][$j];
                }
            }
        }
        return $L;
    }

    // ── QR decomposition (Gram-Schmidt) ────────────────────────────────────

    /** Returns [Q(m×n), R(n×n)] via modified Gram-Schmidt */
    public static function qr(array $A, int $m, int $n): array
    {
        $Q = $A;
        $R = array_fill(0, $n, null);
        for ($i = 0; $i < $n; ++$i) $R[$i] = array_fill(0, $n, 0.0);

        for ($j = 0; $j < $n; ++$j) {
            // extract column j of Q
            $v = [];
            for ($i = 0; $i < $m; ++$i) $v[$i] = $Q[$i][$j];
            // orthogonalise against previous columns
            for ($k = 0; $k < $j; ++$k) {
                $q = [];
                for ($i = 0; $i < $m; ++$i) $q[$i] = $Q[$i][$k];
                $dot = 0.0;
                for ($i = 0; $i < $m; ++$i) $dot += $q[$i] * $v[$i];
                $R[$k][$j] = $dot;
                for ($i = 0; $i < $m; ++$i) $v[$i] -= $dot * $q[$i];
            }
            $norm = 0.0;
            for ($i = 0; $i < $m; ++$i) $norm += $v[$i] * $v[$i];
            $norm = sqrt($norm);
            $R[$j][$j] = $norm;
            if ($norm > 1e-14)
                for ($i = 0; $i < $m; ++$i) $Q[$i][$j] = $v[$i] / $norm;
            else
                for ($i = 0; $i < $m; ++$i) $Q[$i][$j] = 0.0;
        }
        return [$Q, $R];
    }

    // ── SVD (Golub-Kahan bidiagonalization + QR) ───────────────────────────

    /**
     * Thin SVD of A(m×n), m >= n.
     * Returns [U(m×n), s(n), VT(n×n)].
     * For m < n: transpose, compute, swap U/VT.
     * Pure-PHP fallback — for large matrices, use FFI/LAPACK.
     */
    public static function svd(array $A, int $m, int $n): array
    {
        $transposed = false;
        if ($m < $n) {
            $A = self::transposeRaw($A, $m, $n);
            [$m, $n] = [$n, $m];
            $transposed = true;
        }

        // Bidiagonalization via Householder reflections
        $B  = $A;  // working copy, will become bidiagonal
        $U  = self::eye($m);
        $V  = self::eye($n);
        $d  = array_fill(0, $n, 0.0);  // diagonal
        $e  = array_fill(0, $n-1, 0.0); // super-diagonal

        for ($k = 0; $k < $n; ++$k) {
            // Left Householder to zero below B[k][k]
            $x = [];
            for ($i = $k; $i < $m; ++$i) $x[] = $B[$i][$k];
            [$v, $beta] = self::householder($x);
            if ($beta != 0.0) {
                // B = (I - beta*v*v^T) * B  (rows k..m, cols k..n)
                for ($j = $k; $j < $n; ++$j) {
                    $s = 0.0;
                    for ($i = 0; $i < count($v); ++$i) $s += $v[$i] * $B[$k+$i][$j];
                    $s *= $beta;
                    for ($i = 0; $i < count($v); ++$i) $B[$k+$i][$j] -= $s * $v[$i];
                }
                // Accumulate U
                for ($j = 0; $j < $m; ++$j) {
                    $s = 0.0;
                    for ($i = 0; $i < count($v); ++$i) $s += $v[$i] * $U[$j][$k+$i];
                    $s *= $beta;
                    for ($i = 0; $i < count($v); ++$i) $U[$j][$k+$i] -= $s * $v[$i];
                }
            }
            $d[$k] = $B[$k][$k];

            if ($k < $n-1) {
                // Right Householder to zero right of B[k][k+1]
                $x = [];
                for ($j = $k+1; $j < $n; ++$j) $x[] = $B[$k][$j];
                [$v, $beta] = self::householder($x);
                if ($beta != 0.0) {
                    for ($i = $k; $i < $m; ++$i) {
                        $s = 0.0;
                        for ($j = 0; $j < count($v); ++$j) $s += $v[$j] * $B[$i][$k+1+$j];
                        $s *= $beta;
                        for ($j = 0; $j < count($v); ++$j) $B[$i][$k+1+$j] -= $s * $v[$j];
                    }
                    for ($i = 0; $i < $n; ++$i) {
                        $s = 0.0;
                        for ($j = 0; $j < count($v); ++$j) $s += $v[$j] * $V[$i][$k+1+$j];
                        $s *= $beta;
                        for ($j = 0; $j < count($v); ++$j) $V[$i][$k+1+$j] -= $s * $v[$j];
                    }
                }
                $e[$k] = $B[$k][$k+1];
            }
        }

        // QR iteration on bidiagonal matrix to extract singular values
        $maxIter = 1000 * $n;
        for ($iter = 0; $iter < $maxIter && $n > 1; ++$iter) {
            // check convergence of last super-diagonal element
            if (abs($e[$n-2]) <= 1e-14 * (abs($d[$n-2]) + abs($d[$n-1]))) {
                --$n; continue;
            }
            // Golub-Kahan SVD step (simplified: just deflate)
            // Use Givens rotations on the bidiagonal
            $mu = self::wilkinsonShift($d[$n-2], $e[$n-2], $d[$n-1]);
            $y  = $d[0]*$d[0] - $mu;
            $z  = $d[0] * $e[0];
            for ($k = 0; $k < $n-1; ++$k) {
                [$c, $s] = self::givens($y, $z);
                // Apply right Givens G(k,k+1,theta) to B columns k and k+1
                for ($i = 0; $i <= min($k+1, count($d)-1); ++$i) {
                    if ($i === $k) {
                        [$d[$k], $e[$k]] = [$c*$d[$k] + $s*$e[$k], -$s*$d[$k] + $c*$e[$k]];
                    }
                }
                // accumulate in V
                for ($i = 0; $i < count($V); ++$i) {
                    $vk  = $V[$i][$k];
                    $vk1 = $V[$i][$k+1];
                    $V[$i][$k]   = $c*$vk  + $s*$vk1;
                    $V[$i][$k+1] = -$s*$vk + $c*$vk1;
                }
                $y = $d[$k]; $z = $s * ($k+1 < $n-1 ? $e[$k] : 0.0);
                // Left Givens
                [$c, $s] = self::givens($y, $z);
                $d[$k] = $c*$d[$k] + $s*($k < count($e) ? $e[$k] : 0.0);
                if ($k < count($e)) $e[$k] = $c*$e[$k] - $s*$d[$k];
                if ($k+1 < count($d)) {
                    $y = $e[$k] ?? 0.0; $z = $s * $d[$k+1];
                }
                for ($i = 0; $i < count($U[0]); ++$i) {
                    $uk  = $U[$k][$i];
                    $uk1 = $U[$k+1][$i];
                    $U[$k][$i]   = $c*$uk  + $s*$uk1;
                    $U[$k+1][$i] = -$s*$uk + $c*$uk1;
                }
            }
        }
        // ensure non-negative singular values
        $mm = count($d);
        for ($i = 0; $i < $mm; ++$i) {
            if ($d[$i] < 0) {
                $d[$i] = -$d[$i];
                for ($j = 0; $j < count($U); ++$j) $U[$j][$i] = -$U[$j][$i];
            }
        }
        // sort descending
        arsort($d);
        $ord = array_keys($d);
        $d   = array_values($d);
        $Uo  = $U; $Vo = $V;
        foreach ($ord as $newIdx => $oldIdx) {
            for ($i = 0; $i < count($U); ++$i) $Uo[$i][$newIdx] = $U[$i][$oldIdx];
            for ($i = 0; $i < count($V); ++$i) $Vo[$i][$newIdx] = $V[$i][$oldIdx];
        }
        $VT = self::transposeRaw($Vo, count($Vo), count($Vo[0]));
        if ($transposed) return [$Vo, $d, self::transposeRaw($Uo, count($Uo), count($Uo[0]))];
        return [$Uo, $d, $VT];
    }

    private static function householder(array $x): array
    {
        $n    = count($x);
        $norm = 0.0;
        foreach ($x as $v) $norm += $v*$v;
        $norm = sqrt($norm);
        if ($norm == 0.0) return [$x, 0.0];
        $v    = $x;
        $v[0] += ($x[0] >= 0 ? 1 : -1) * $norm;
        $vNorm = 0.0;
        foreach ($v as $vi) $vNorm += $vi*$vi;
        $beta = 2.0 / $vNorm;
        return [$v, $beta];
    }

    private static function wilkinsonShift(float $a, float $b, float $c): float
    {
        $d   = ($a*$a + $c*$c - $b*$b - $a*$a) / 2.0;  // simplified
        $mu  = $c*$c + $d - ($d >= 0 ? 1 : -1) * sqrt($d*$d + $b*$b*$c*$c / max(1e-300, $c*$c));
        return $mu;
    }

    private static function givens(float $a, float $b): array
    {
        if ($b == 0.0) return [1.0, 0.0];
        if (abs($b) > abs($a)) { $t = -$a/$b; $s = 1/sqrt(1+$t*$t); return [$s*$t, $s]; }
        $t = -$b/$a; $c = 1/sqrt(1+$t*$t); return [$c, $c*$t];
    }

    /** Identity matrix n×n as array-of-arrays */
    public static function eye(int $n): array
    {
        $I = [];
        for ($i = 0; $i < $n; ++$i) {
            $row = array_fill(0, $n, 0.0);
            $row[$i] = 1.0;
            $I[$i] = $row;
        }
        return $I;
    }

    // ── Symmetric eigendecomposition (Jacobi iteration) ───────────────────

    /**
     * Jacobi eigendecomposition for real symmetric matrices.
     * Returns [eigenvalues (sorted ascending), eigenvectors (columns)].
     */
    public static function eigSymmetric(array $A, int $n): array
    {
        $V  = self::eye($n);
        $D  = $A;  // will become diagonal

        for ($sweep = 0; $sweep < 100 * $n * $n; ++$sweep) {
            // find largest off-diagonal element
            $maxVal = 0.0; $p = 0; $q = 1;
            for ($i = 0; $i < $n; ++$i)
                for ($j = $i+1; $j < $n; ++$j)
                    if (($v = abs($D[$i][$j])) > $maxVal) { $maxVal = $v; $p = $i; $q = $j; }
            if ($maxVal < 1e-14) break;

            // Jacobi rotation
            $theta = ($D[$q][$q] - $D[$p][$p]) / (2.0 * $D[$p][$q]);
            $t     = ($theta >= 0 ? 1 : -1) / (abs($theta) + sqrt(1 + $theta*$theta));
            $c     = 1.0 / sqrt(1 + $t*$t);
            $s     = $t * $c;

            // rotate D
            $Dpp = $D[$p][$p]; $Dqq = $D[$q][$q]; $Dpq = $D[$p][$q];
            $D[$p][$p] = $Dpp - $t*$Dpq;
            $D[$q][$q] = $Dqq + $t*$Dpq;
            $D[$p][$q] = $D[$q][$p] = 0.0;
            for ($r = 0; $r < $n; ++$r) {
                if ($r !== $p && $r !== $q) {
                    $Drp = $D[$r][$p]; $Drq = $D[$r][$q];
                    $D[$r][$p] = $D[$p][$r] = $c*$Drp - $s*$Drq;
                    $D[$r][$q] = $D[$q][$r] = $s*$Drp + $c*$Drq;
                }
                // accumulate eigenvectors
                $Vrp = $V[$r][$p]; $Vrq = $V[$r][$q];
                $V[$r][$p] = $c*$Vrp - $s*$Vrq;
                $V[$r][$q] = $s*$Vrp + $c*$Vrq;
            }
        }
        // extract eigenvalues and sort ascending
        $w = [];
        for ($i = 0; $i < $n; ++$i) $w[$i] = $D[$i][$i];
        asort($w);
        $ord = array_keys($w);
        $w   = array_values($w);
        $Vo  = $V;
        foreach ($ord as $newIdx => $oldIdx)
            for ($i = 0; $i < $n; ++$i)
                $Vo[$i][$newIdx] = $V[$i][$oldIdx];
        return [$w, $Vo];
    }

    // ── Row Echelon Form ──────────────────────────────────────────────────

    public static function ref(array $A, int $m, int $n): array
    {
        $R     = $A;
        $swaps = 0;
        $row   = 0;
        for ($col = 0; $col < $n && $row < $m; ++$col) {
            // find pivot
            $maxVal = abs($R[$row][$col]); $maxRow = $row;
            for ($i = $row+1; $i < $m; ++$i)
                if (($v = abs($R[$i][$col])) > $maxVal) { $maxVal = $v; $maxRow = $i; }
            if ($maxVal < 1e-14) continue;
            if ($maxRow !== $row) { [$R[$row], $R[$maxRow]] = [$R[$maxRow], $R[$row]]; ++$swaps; }
            $inv = 1.0 / $R[$row][$col];
            for ($i = $row+1; $i < $m; ++$i) {
                $f = $R[$i][$col] * $inv;
                for ($j = $col; $j < $n; ++$j)
                    $R[$i][$j] -= $f * $R[$row][$j];
                $R[$i][$col] = 0.0;
            }
            ++$row;
        }
        return [$R, $swaps];
    }

    public static function rref(array $A, int $m, int $n): array
    {
        [$R, ] = self::ref($A, $m, $n);
        $lead = 0;
        for ($r = 0; $r < $m; ++$r) {
            if ($lead >= $n) break;
            $i = $r;
            while ($i < $m && abs($R[$i][$lead]) < 1e-14) {
                ++$i;
                if ($i === $m) { $i = $r; ++$lead; if ($lead === $n) goto done; }
            }
            [$R[$i], $R[$r]] = [$R[$r], $R[$i]];
            $lv = $R[$r][$lead];
            if (abs($lv) > 1e-14)
                for ($j = 0; $j < $n; ++$j) $R[$r][$j] /= $lv;
            for ($i = 0; $i < $m; ++$i) {
                if ($i !== $r) {
                    $f = $R[$i][$lead];
                    for ($j = 0; $j < $n; ++$j) $R[$i][$j] -= $f * $R[$r][$j];
                }
            }
            ++$lead;
        }
        done:
        return $R;
    }

    /** Matrix rank via REF */
    public static function rank(array $A, int $m, int $n): int
    {
        [$R, ] = self::ref($A, $m, $n);
        $rank = 0;
        for ($i = 0; $i < $m; ++$i) {
            $nonzero = false;
            for ($j = 0; $j < $n; ++$j) if (abs($R[$i][$j]) > 1e-10) { $nonzero = true; break; }
            if ($nonzero) ++$rank;
        }
        return $rank;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  MATRIX CLASS
// ═══════════════════════════════════════════════════════════════════════════