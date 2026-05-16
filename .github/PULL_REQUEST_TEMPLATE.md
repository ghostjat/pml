## Summary

<!-- One paragraph: what does this PR do and why? -->

## Type of Change

- [ ] Bug fix (corrects a defect, no API changes)
- [ ] Performance improvement (same behavior, measurably faster or lower memory)
- [ ] New C kernel (new function in `tensor.c`, `dataset_io.c`, `inference.c`, etc.)
- [ ] New PHP API (new method, class, or interface)
- [ ] New estimator / transformer / layer
- [ ] Refactor (no behavior change)
- [ ] Documentation / examples
- [ ] Build / CI / tooling
- [ ] Breaking change (describe migration path below)

---

## Changes

<!-- List all files changed and what changed in each. Be specific. -->

- `src/Lib/tensor.c` — added `tensor_my_op()` with OpenMP loop
- `src/Lib/tensor.h` — declared `tensor_my_op`
- `src/Lib/TensorEngine.php` — added `tensor_my_op` cdef + `myOp()` method
- `src/Tensor.php` — added `myOp()` proxy method
- `tests/TensorMyOpTest.php` — 4 unit tests

---

## C Backend Changes

<!-- If you changed any .c or .h files, complete this section. -->

### New symbols exported

```bash
# run this after building and paste output
nm -D src/Lib/libtensor.so | grep my_op
```

```
# paste output
```

### Memory contract

- [ ] All new `TensorC*` return values are owned by the caller (caller must `tensor_free()`)
- [ ] No new global mutable state introduced
- [ ] No dynamic allocations inside OpenMP parallel regions
- [ ] Tested with `valgrind --leak-check=full` — no new leaks

### Build verified

```bash
cd src/Lib
gcc -O3 -march=native -mfma -fopenmp -funroll-loops -fomit-frame-pointer \
    -D_GNU_SOURCE -shared -fPIC -funsafe-math-optimizations \
    -o libtensor.so.7 tensor.c dataset_io.c inference.c autograd.c graph.c tokenizer.c \
    -lopenblas -llapacke -lm
# result:
```

---

## PHP API Changes

### Interface compliance

- [ ] New classes implement all required interfaces for their type
- [ ] No ML math added to PHP (all numeric operations go through FFI)
- [ ] No PHP-level loops over tensor data

### Backward compatibility

- [ ] No existing method signatures changed
- [ ] No existing behavior changed
- [ ] If breaking: migration path documented below

---

## Tests

- [ ] PHPUnit tests added / updated
- [ ] All existing tests pass: `vendor/bin/phpunit --colors=always`

```
# paste test output summary
```

---

## Benchmarks

<!-- Required for any change to a hot path (tensor ops, estimator training, inference) -->

### Before

```bash
vendor/bin/phpbench run benchmarks/... --report=aggregate
```

```
# paste before results
```

### After

```bash
vendor/bin/phpbench run benchmarks/... --report=aggregate
```

```
# paste after results
```

### Performance delta

| Benchmark | Before | After | Change |
|---|---|---|---|
| `benchMyOp` | X ms | Y ms | +/- Z% |

---

## Documentation

- [ ] `docs/` HTML updated if this adds or changes public API
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`
- [ ] If new C functions: added to relevant `docs/*.html` API block

---

## Checklist

- [ ] I have read [CONTRIBUTING.md](../CONTRIBUTING.md)
- [ ] PHP orchestrates — no ML math added in PHP
- [ ] Zero-copy invariant maintained — no PHP-side tensor buffer copies in hot paths
- [ ] Memory leak free — `tensor_free()` called on all owned `TensorC*` pointers
- [ ] C backend built and `nm -D` verified
- [ ] `vendor/bin/phpunit` passes
- [ ] Benchmark results included (if hot path changed)

---

## Migration Guide (if breaking change)

<!-- If this is a breaking change, describe exactly what callers need to update: -->

```php
// Before:
$model->oldMethod($arg);

// After:
$model->newMethod($arg);
```
