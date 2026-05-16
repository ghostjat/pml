---
name: Feature Request
about: Propose a new operation, model, transformer, API change, or infrastructure improvement
title: "[FEAT] "
labels: ["enhancement", "needs-triage"]
assignees: ghostjat
---

## Summary

<!-- One sentence: what are you proposing? -->

## Motivation

<!-- Why does PML need this? What problem does it solve? What use case does it enable? -->

## Proposed API

<!-- If this is a PHP-facing change, sketch the API you'd want: -->

```php
<?php
// example usage of the proposed feature
```

<!-- If this is a C-level addition, sketch the function signature: -->

```c
// tensor_my_op(TensorC* a, TensorC* b) -> TensorC*
```

## Technical Approach

<!-- How would you implement this? Which C files and PHP files are involved?
     If you don't know, leave this blank — the maintainer will fill it in. -->

## Performance Expectations

<!-- Is this expected to be faster/slower/same as existing alternatives?
     Any known benchmarks or reference implementations to compare against? -->

## Alternatives Considered

<!-- What existing workarounds exist? Why are they insufficient? -->

## Layer

- [ ] C tensor kernel (new `tensor_*` function)
- [ ] PHP Tensor API (new method on `Tensor`)
- [ ] Estimator (new classifier / regressor / anomaly detector)
- [ ] Transformer (new feature transformer)
- [ ] Neural network layer
- [ ] Optimizer / loss function
- [ ] Inference / LLM
- [ ] Vision module
- [ ] Pipeline / cross-validation
- [ ] Training infrastructure
- [ ] Tooling / CLI
- [ ] Documentation

## Priority

- [ ] Blocks a real project I'm working on (describe below)
- [ ] Would significantly improve my workflow
- [ ] Nice to have

<!-- If this blocks a real project, describe what you're building and why this is needed. -->
