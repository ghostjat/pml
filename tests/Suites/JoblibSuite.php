<?php

declare(strict_types=1);

namespace Pml\Tests\Suites;

use Pml\Tensor;
use Pml\Tests\Core\TestRunner;
use Pml\Tests\Datasets\DatasetLoader;
use Pml\Classic\NaiveBayes\GaussianNB;
use Pml\Classic\Utils\Joblib;
use Pml\Classic\ModelSelection\DataSplit;

// ═══════════════════════════════════════════════════════════════════════════
//  JoblibSuite — FFI-safe serialization roundtrip tests
//
//  Goal: Prove that Joblib::dump() + Joblib::load() perfectly restores a
//  fitted model's C-memory state so that it produces IDENTICAL predictions.
//
//  ── Why is this hard? ────────────────────────────────────────────────────
//
//  Tensor objects wrap \FFI\CData buffers.  PHP's serialize() cannot handle
//  CData — it throws a fatal error at the CData boundary.  Joblib sidesteps
//  this by:
//
//    dump(): Reflecting over all properties → extracting CData as binary PHP
//            strings via \FFI::string($buf, byteCount) → serialize() on the
//            safe surrogate tree.
//
//    load(): unserialize() → for each tensor surrogate:
//              allocFloat(n) → \FFI::memcpy(buf, binary_str, byteCount)
//              → new Tensor(shape, pre-populated buf)
//            → inject all properties via ReflectionProperty::setValue().
//
//  ── The \FFI::memcpy Correctness Test ────────────────────────────────────
//
//  The critical property being tested:
//
//    \FFI::memcpy(dst, $phpBinaryString, N*4)
//
//  must restore the EXACT bit pattern of the original float32 array.
//  Even a single bit error would change a learned parameter (θ, σ²) and
//  therefore shift a prediction.
//
//  We verify this by:
//    1. Predicting with the original model → hash the float array
//    2. Joblib::dump() + Joblib::load() → resurrected model
//    3. Predicting with the resurrected model → hash the float array
//    4. Assert hashes match  (= all predictions agree bit-for-bit)
//
//  ── Why GaussianNB? ──────────────────────────────────────────────────────
//
//  GaussianNB stores all its state in Tensor arrays (theta_, var_, class_prior_,
//  class_count_).  A single float32 perturbation in theta_ or var_ will almost
//  certainly shift the argmax of the log-posterior for at least one sample,
//  producing a different class prediction and a hash mismatch.
//
//  It is also fast to train (single pass) and has no non-determinism, making
//  it the ideal serialization test subject.
// ═══════════════════════════════════════════════════════════════════════════

final class JoblibSuite
{
    public static function run(TestRunner $r): void
    {
        $r->suite('Serialization Integrity (Joblib)', function(TestRunner $r) {

            $iris = DatasetLoader::iris();

            // ── Test 1: GaussianNB dump/load roundtrip ─────────────────
            $r->test('GaussianNB: Joblib dump→load predictions match bit-for-bit', function() use ($r, $iris) {

                // ── Split: train on 80%, test on 20% ──────────────────
                // Same random_state → same split every run.
                [$Xtrain, $Xtest, $ytrain, $ytest] = DataSplit::train_test_split(
                    $iris['X'], $iris['y'],
                    test_size:    0.2,
                    random_state: 7,
                );

                // ── Fit original model ─────────────────────────────────
                $gnb = new GaussianNB(var_smoothing: 1e-9);
                $gnb->fit($Xtrain, $ytrain);

                // ── Predict with original model ────────────────────────
                $predOrig = $gnb->predict($Xtest);

                // Convert Tensor predictions to a plain PHP float array.
                // We must copy NOW — after Joblib::dump, the Tensor's buffer
                // is still valid, but we want a snapshot before any possible
                // state mutation.
                $origArr = self::tensorToArray($predOrig);

                // ── Compute SHA-256 hash of original predictions ───────
                //
                // Hash the packed binary representation of the float array
                // for a compact, collision-resistant fingerprint.
                //
                // We use pack('f*', ...) to serialize the floats in the same
                // IEEE 754 binary format that \FFI::memcpy uses, ensuring that
                // "identical" means bit-identical (not just PHP ==).
                $origHash = hash('sha256', pack('f*', ...$origArr));

                // ── Dump to temporary file ─────────────────────────────
                $tmpFile = sys_get_temp_dir() . '/pml_test_gnb_' . uniqid() . '.joblib';

                Joblib::dump($gnb, $tmpFile);

                if (!file_exists($tmpFile)) {
                    throw new \RuntimeException("Joblib::dump() did not create file: {$tmpFile}");
                }

                $fileSize = filesize($tmpFile);

                // ── Load from file ─────────────────────────────────────
                $gnb2 = Joblib::load($tmpFile);

                // The resurrected model must be a GaussianNB instance
                if (!($gnb2 instanceof GaussianNB)) {
                    throw new \RuntimeException(
                        'Joblib::load() returned ' . get_class($gnb2) . ' instead of GaussianNB'
                    );
                }

                // ── Predict with resurrected model ─────────────────────
                //
                // We pass the SAME $Xtest (same C-buffer, unchanged).
                // Any prediction difference must come from the model parameters,
                // not from the input data.
                $predResurrected = $gnb2->predict($Xtest);
                $resurrArr       = self::tensorToArray($predResurrected);

                // ── Hash resurrected predictions ───────────────────────
                $resurrHash = hash('sha256', pack('f*', ...$resurrArr));

                // ── THE CRITICAL ASSERTION ─────────────────────────────
                //
                // Both hashes must match.  A mismatch would mean that at least
                // one prediction changed after serialization — indicating that
                // \FFI::memcpy failed to restore the exact bit pattern of one
                // or more learned parameters (theta_, var_, or class_prior_).
                $r->assertEq(
                    $resurrHash,
                    $origHash,
                    sprintf(
                        'SHA-256 mismatch (file=%d bytes). Original: %s… Resurrected: %s…',
                        $fileSize,
                        substr($origHash, 0, 16),
                        substr($resurrHash, 0, 16),
                    )
                );

                // Also verify element-wise for a more informative failure message
                $r->assertArraysMatch(
                    $origArr,
                    $resurrArr,
                    'element-wise prediction parity'
                );

                // ── Cleanup ────────────────────────────────────────────
                @unlink($tmpFile);
            });

            // ── Test 2: Serialized file integrity ─────────────────────
            $r->test('Joblib file is valid PHP-serialized surrogate (no CData)', function() use ($r, $iris) {

                // Verify that the .joblib file contains only safe PHP-serializable
                // data and that unserialize() succeeds without CData contamination.
                //
                // This tests the dump() phase independently of load().

                $gnb = new GaussianNB();
                $gnb->fit($iris['X'], $iris['y']);

                $tmpFile = sys_get_temp_dir() . '/pml_test_gnb_integrity_' . uniqid() . '.joblib';
                Joblib::dump($gnb, $tmpFile);

                $raw = file_get_contents($tmpFile);

                // unserialize with allowed_classes=false: if the file contains
                // any PHP object references (including CData), unserialize()
                // would return false or throw.
                // A clean surrogate tree has NO class instances — only arrays and scalars.
                $surrogate = unserialize($raw, ['allowed_classes' => false]);

                if ($surrogate === false) {
                    throw new \RuntimeException('Joblib file failed to unserialize — possible CData contamination');
                }

                // The top-level surrogate must be an array (the __pml_object__ wrapper)
                if (!is_array($surrogate)) {
                    throw new \RuntimeException('Top-level surrogate is not an array — unexpected format');
                }

                if (!isset($surrogate['__pml_object__'])) {
                    throw new \RuntimeException('Missing __pml_object__ tag in surrogate — dump() may be broken');
                }

                // Passed: the file is clean PHP-serialized data with no CData
                @unlink($tmpFile);
            });

            // ── Test 3: Predict shape preserved after roundtrip ───────
            $r->test('GaussianNB predictions tensor shape preserved after Joblib roundtrip', function() use ($r, $iris) {

                $gnb = new GaussianNB();
                $gnb->fit($iris['X'], $iris['y']);

                $predBefore = $gnb->predict($iris['X']);

                $tmpFile = sys_get_temp_dir() . '/pml_test_gnb_shape_' . uniqid() . '.joblib';
                Joblib::dump($gnb, $tmpFile);
                $gnb2       = Joblib::load($tmpFile);
                $predAfter  = $gnb2->predict($iris['X']);

                $r->assertShape($predAfter, $predBefore->shape, 'prediction shape after roundtrip');
                @unlink($tmpFile);
            });

        });
    }

    // ── Helpers ────────────────────────────────────────────────────────────

    /**
     * Extract all float values from a Tensor into a PHP float[].
     *
     * Reads from the FFI float32 buffer element by element.
     * The result is a plain PHP array — serializable, hashable, and comparable
     * without any FFI considerations.
     *
     * @return float[]
     */
    private static function tensorToArray(Tensor $t): array
    {
        $arr = [];
        for ($i = 0; $i < $t->size; $i++) {
            $arr[] = (float)$t->buffer[$i];
        }
        return $arr;
    }
}
