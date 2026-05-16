# Sponsors

PML is an independent open-source project built and maintained by a single engineer. It has no VC backing, no corporate owner, and no commercial product behind it. Everything shipped so far — the C tensor engine, the LLM inference runtime, the vision module, the quantization system — was built in private time.

If PML saves you engineering hours, enables products you couldn't otherwise build, or advances the state of AI in the PHP ecosystem, please consider sponsoring.

**[❤ Sponsor on GitHub Sponsors](https://github.com/sponsors/ghostjat)**

---

## Why Sponsor?

### The Technical Case

PML is not a toy library. It is:

- A **native C tensor engine** with 500+ exported functions, OpenBLAS SGEMM, AVX2 SIMD, and OpenMP parallelism
- A **production LLM inference runtime** running LLaMA-3 8B at 12 tokens/sec in INT8 on a single CPU
- A **zero-copy PHP↔C memory architecture** — tensors live in C, PHP holds pointers
- A **vision system** with 106 C functions: detection, segmentation, augmentation, MobileNetV3
- A **complete ML stack** with 40+ estimators, 35+ transformers, 6 CV strategies, autograd, and more

### The Ecosystem Case

PHP runs on hundreds of millions of servers. Most of them have no path to ML without Python subprocesses, external APIs, or full stack rewrites. PML changes that. Every improvement to PML multiplies across that deployment surface.

### What Sponsorship Funds

| Priority | Work |
|---|---|
| Immediate | Vulkan GPU backend — cross-vendor, PHP API unchanged |
| Near-term | ONNX model import, fp16 tensors, Flash Attention |
| Near-term | GGUF format reader (run llama.cpp models directly) |
| Medium-term | Agent runtime, tool-call system, RAG pipeline |
| Long-term | Distributed training, ARM NEON SIMD, edge deployment |

---

## Sponsor Tiers

### ☕ Community — $5/month

For individuals who use PML in personal projects or learning.

**Perks:**
- Name in `SPONSORS.md`
- Community sponsor badge

---

### 🔧 Developer — $25/month

For individual engineers who ship PML-powered products or use it daily.

**Perks:**
- Everything in Community
- Priority issue responses (within 48 hours)
- Early access to release notes and changelogs
- Name + GitHub link in `SPONSORS.md`

---

### 🏗 Builder — $100/month

For teams and small companies building products on PML.

**Perks:**
- Everything in Developer
- Logo in `SPONSORS.md` and `README.md`
- Private discussion channel access
- Priority bug fixes for your reported issues
- Architecture consultation (1 session/month, 30 min)

---

### 🚀 Infrastructure — $500/month

For companies whose products depend on PML at scale or who want to directly accelerate development.

**Perks:**
- Everything in Builder
- Featured logo placement in README and documentation
- Dedicated issue queue for your organization's needs
- Architecture consultation (2 sessions/month, 60 min each)
- Option to sponsor a specific roadmap item (GPU kernel, model format, etc.)
- Advance notice of breaking changes

---

### 🏛 Foundation — $2,000/month

For organizations that want to be primary stewards of the PML ecosystem.

**Perks:**
- Everything in Infrastructure
- Top billing in README, documentation, and GitHub profile
- Co-design access for major roadmap decisions
- Technical advisory relationship
- Named acknowledgment in all major release announcements
- Direct line to maintainer for architectural guidance

---

## Funding Goals

| Monthly Goal | Unlocks |
|---|---|
| $500/month | Full-time weekend work on PML (16 hrs/month dedicated) |
| $1,500/month | Vulkan GPU backend development (Phase 2.0) |
| $3,000/month | ONNX import + fp16 + Flash Attention (Phase 2.1) |
| $6,000/month | Half-time equivalent on PML core infrastructure |
| $12,000/month | Full-time equivalent — accelerate all roadmap phases |

---

## Current Sponsors

*Be the first to sponsor PML and have your name here.*

### Foundation Sponsors
*— open —*

### Infrastructure Sponsors
*— open —*

### Builder Sponsors
*— open —*

### Developer Sponsors
*— open —*

### Community Sponsors
*— open —*

---

## One-Time Contributions

If recurring sponsorship isn't right for you, one-time contributions are also accepted via GitHub Sponsors. Suggestions:

| Amount | What it represents |
|---|---|
| $25 | One benchmark session — profiling and optimization of one hot path |
| $100 | One GPU kernel — GLSL compute shader for one operator |
| $500 | One model format — ONNX operator set or GGUF reader |
| $1,000 | One framework feature — full implementation, tests, docs |

---

## For Companies

If your company builds PHP products that benefit from ML capabilities or you're evaluating PML for production use, reach out at **edspireconsultancy@gmail.com**.

Topics for company engagement:
- Integration support
- Custom model format adapters
- Performance profiling for your specific workload
- Architecture review for ML-in-PHP deployments
- Roadmap input for features your product needs

---

*Thank you for supporting independent systems programming in the PHP ecosystem.*
