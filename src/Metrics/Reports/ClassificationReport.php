<?php

declare(strict_types=1);

namespace Pml\Metrics\Reports;

use Pml\Tensor;

/**
 * Classification Report Generator.
 * Consumes the zero-copy Confusion Matrix to output a detailed Precision/Recall/F1 text report.
 */
final class ClassificationReport
{
    public static function generate(Tensor $predictions, Tensor $labels, ?array $classNames = null): string
    {
        $matrix = ConfusionMatrix::generate($predictions, $labels);
        $numClasses = count($matrix);

        if ($numClasses === 0) return "Empty Classification Report\n";

        $output = sprintf("%-15s %-10s %-10s %-10s %-10s\n", "Class", "Precision", "Recall", "F1 Score", "Support");
        $output .= str_repeat("-", 60) . "\n";

        $macroPrecision = 0.0;
        $macroRecall = 0.0;
        $macroF1 = 0.0;
        $totalSupport = 0;

        for ($i = 0; $i < $numClasses; $i++) {
            $tp = $matrix[$i][$i];
            
            $fn = 0;
            $fp = 0;
            $support = 0;

            for ($j = 0; $j < $numClasses; $j++) {
                $support += $matrix[$i][$j]; // Sum of the true row
                if ($i !== $j) {
                    $fn += $matrix[$i][$j];
                    $fp += $matrix[$j][$i]; // Sum of the predicted column
                }
            }

            $precision = ($tp + $fp) > 0 ? $tp / ($tp + $fp) : 0.0;
            $recall = ($tp + $fn) > 0 ? $tp / ($tp + $fn) : 0.0;
            $f1 = ($precision + $recall) > 0 ? 2 * ($precision * $recall) / ($precision + $recall) : 0.0;

            $macroPrecision += $precision;
            $macroRecall += $recall;
            $macroF1 += $f1;
            $totalSupport += $support;

            $label = $classNames ? ($classNames[$i] ?? (string)$i) : (string)$i;
            
            $output .= sprintf(
                "%-15s %-10.4f %-10.4f %-10.4f %-10d\n",
                $label, $precision, $recall, $f1, $support
            );
        }

        $output .= str_repeat("-", 60) . "\n";
        $output .= sprintf(
            "%-15s %-10.4f %-10.4f %-10.4f %-10d\n",
            "Macro Avg", 
            $macroPrecision / $numClasses, 
            $macroRecall / $numClasses, 
            $macroF1 / $numClasses, 
            $totalSupport
        );

        return $output;
    }
}