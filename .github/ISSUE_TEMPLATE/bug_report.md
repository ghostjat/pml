---
name: Bug Report
about: Report a crash, incorrect output, memory error, or unexpected behavior
title: "[BUG] "
labels: ["bug", "needs-triage"]
assignees: ghostjat
---

## Description

<!-- Clear, one-paragraph description of the bug. What did you expect? What happened instead? -->

## Reproduction

<!-- Minimal PHP snippet that reproduces the problem. Trim to the essential few lines. -->

```php
<?php
require 'vendor/autoload.php';

// your reproduction here
```

## Environment

| Field | Value |
|---|---|
| PHP version | <!-- php --version --> |
| PML version | <!-- composer show ghostjat/pml --> |
| OS | <!-- uname -a --> |
| GCC version | <!-- gcc --version --> |
| OpenBLAS version | <!-- dpkg -l libopenblas-dev --> |
| CPU | <!-- lscpu \| grep 'Model name' --> |
| ffi.enable | <!-- php -r "echo ini_get('ffi.enable');" --> |

## C Backend Version

```bash
# Output of:
nm -D src/Lib/libtensor.so | head -5
```

<!-- paste output -->

## Error Output

<!-- Full PHP error, stack trace, segfault output, or valgrind report -->

```
paste here
```

## Valgrind Output (if segfault or memory error)

```bash
valgrind --leak-check=full --error-exitcode=1 php your_script.php 2>&1
```

```
paste here
```

## Severity

- [ ] Crash / segfault
- [ ] Memory leak
- [ ] Wrong numerical output
- [ ] PHP exception / fatal error
- [ ] Performance regression
- [ ] Other

## Additional Context

<!-- Any other context, links to related issues, attempted workarounds. -->
