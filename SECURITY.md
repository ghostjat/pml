# Security Policy

## Supported Versions

| Version | Security Support |
|---|---|
| 1.x (current) | ✅ Active |
| < 1.0 | ✗ No support |

---

## Reporting a Vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Report security issues by emailing: **edspireconsultancy@gmail.com**

Include in your report:
- Description of the vulnerability
- Steps to reproduce
- Affected component (`tensor.c`, `TensorEngine.php`, `NotebookController`, etc.)
- Potential impact assessment
- If available, a proof-of-concept

You will receive an acknowledgment within **48 hours** and a full response within **7 days**.

---

## Scope

### In Scope

- Memory safety bugs in the C backend (`tensor.c`, `dataset_io.c`, `inference.c`, etc.)
  - Buffer overflows
  - Use-after-free
  - Integer overflow in tensor shape/stride calculations
  - Heap corruption in CSV mmap loader
- PHP injection vulnerabilities in public-facing API surfaces
- FFI type confusion that could allow memory corruption from PHP
- Authentication or authorization bypass in the REST API (`core/api/`)
- Command injection in `VisionEngine` or any `exec()`-based code path
- Path traversal in dataset loading, model persistence, or storage manager
- JWT implementation weaknesses (`core/api/JWTAuth.php`)

### Out of Scope

- Vulnerabilities requiring physical access to the server
- Social engineering attacks
- Issues in dependencies (report these to the upstream project)
- Issues requiring an authenticated admin to exploit (low severity, report via normal issues)
- Performance degradation without security impact
- `NotebookController` sandbox escapes — the notebook executor is intentionally documented as requiring container isolation in production

---

## Known Security Considerations

### Notebook Code Execution

`NotebookController` executes arbitrary PHP code in a `pcntl_fork`ed child process with a 30-second timeout. This is **intentionally unsafe** without container-level isolation. In production:

```bash
# Wrap PHP-FPM in a seccomp profile or gVisor container
# Or disable the notebook endpoint entirely:
# nginx: deny POST /api/notebook/execute;
```

Do not expose the notebook endpoint to untrusted users without container isolation.

### FFI Memory Safety

The FFI bridge relies on C memory discipline. If you write custom C extensions that interact with `TensorC*` structs, you are responsible for:
- Not calling `tensor_free()` on tensors you don't own
- Not holding `TensorC*` pointers past the lifetime of the PHP `Tensor` wrapper
- Not passing malformed shape arrays to C functions

### CSV mmap

The `tensor_dataset_from_csv()` function mmaps the file directly. A maliciously crafted CSV can trigger edge cases in the type-inference and column-parsing code. Do not ingest CSV files from untrusted sources without validation.

### JWT Secret

The JWT secret (`JWT_SECRET` environment variable) must be set to a random value of at least 32 bytes in production. The default in the example configuration is a placeholder and must be changed.

---

## Disclosure Policy

Once a fix is ready and released:

1. A security advisory is published on the GitHub repository
2. The fix is tagged in a patch release
3. The reporter is credited (unless they request anonymity)
4. CVE is requested if the severity warrants it (CVSS ≥ 4.0)

Coordinated disclosure timeline: reporter notified of fix 5 days before public disclosure.
