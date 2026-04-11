<?php
declare(strict_types=1);

namespace Pml\Helpers;

/**
 * Generates Graphviz DOT-language representations of tree estimators.
 */
final class Graphviz
{
    /**
     * Render a PHP node-tree (as produced by DecisionTreeClassifier) to a DOT string.
     *
     * @param array $tree    Root node of the decision tree
     * @param string $name   Graph name
     */
    public static function treeToDot(array $tree, string $name = 'DecisionTree'): string
    {
        $nodes = [];
        $edges = [];
        $counter = 0;

        self::walk($tree, $counter, $nodes, $edges);

        $lines = ["digraph {$name} {", '    node [shape=box fontname="Helvetica"];'];
        foreach ($nodes as $id => $label) {
            $escaped = addslashes($label);
            $lines[] = "    n{$id} [label=\"{$escaped}\"];";
        }
        foreach ($edges as [$from, $to, $label]) {
            $lines[] = "    n{$from} -> n{$to} [label=\"{$label}\"];";
        }
        $lines[] = '}';

        return implode("\n", $lines);
    }

    private static function walk(array $node, int &$counter, array &$nodes, array &$edges): int
    {
        $id = $counter++;

        if (isset($node['feature'])) {
            $nodes[$id] = "Feature #{$node['feature']} <= {$node['threshold']}";
            $leftId     = self::walk($node['left'],  $counter, $nodes, $edges);
            $rightId    = self::walk($node['right'], $counter, $nodes, $edges);
            $edges[]    = [$id, $leftId,  'true'];
            $edges[]    = [$id, $rightId, 'false'];
        } else {
            $nodes[$id] = "class: {$node['class']}";
        }

        return $id;
    }
}
