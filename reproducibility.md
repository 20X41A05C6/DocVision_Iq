# Reproducibility Statement

This document describes the assumptions, constraints, and known sources of
variation affecting the reproducibility of the DocVision system.

---

## Hardware Assumptions

- CPU-only execution environment
- No GPU required
- Minimum 8 GB RAM recommended
- Stable internet connectivity required for external API access
- Tested on cloud-based Linux and local desktop environments

---

## Runtime Estimates

Approximate end-to-end runtime per document (single page):

| Pipeline Stage            | Avg Time (seconds) |
|---------------------------|--------------------|
| OCR (LlamaParse)          | 3.0 – 5.0          |
| Vision LLM Inference      | 1.5 – 3.5          |
| Logo Detection (CPU)      | < 10               |
| **Total (Full Pipeline)** | **14.5 – 18.0**    |

Multi-page PDFs process only the first page by default, keeping runtime bounded.

Actual latency may vary depending on API load and network conditions.

---

## Random Seed Handling

- Temperature for Vision LLM inference is fixed at `0.1`
- Experiments declare a fixed seed value (`seed: 50`) for documentation purposes
- No stochastic sampling is intentionally introduced in the pipeline
- Deterministic preprocessing steps (PDF rendering, resizing, hashing)

---

## Known Sources of Nondeterminism

Despite fixed configuration, some nondeterminism remains due to:

- Cloud-hosted Vision LLM inference
- OCR service variability across identical inputs
- Network latency fluctuations
- Concurrent execution order of asynchronous tasks

These factors may cause minor variations in extracted fields or reasoning text,
but document classification remains stable in most cases.

---

## Cost Considerations

- OCR and Vision inference rely on external APIs (LlamaParse, OpenRouter)
- Cost scales linearly with number of processed documents
- No GPU compute cost is incurred
- Logo detection and preprocessing run locally on CPU
- Caching mechanisms reduce redundant API calls for repeated inputs

Users should be aware of API usage limits and associated costs when processing
large batches of documents.

---

## Summary

DocVision prioritizes reproducibility through deterministic preprocessing,
fixed inference parameters, and declarative experiment definitions. Remaining
sources of nondeterminism stem primarily from external AI services rather than
internal system design.
