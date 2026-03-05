# Problem Definition: Financial Statement Page-to-JSON Extraction

Date: 2026-03-05

## 1) Core Problem (project-specific)
Given a **single financial-statement page image** (often Hebrew, table-heavy, multi-column), produce a **strictly valid `PageExtraction` JSON** with:
- page metadata (`meta`)
- a set of grounded financial facts (`facts`)
- per-fact geometry (`bbox`) and business semantics (`path`, `date`, `currency`, `scale`, `value_type`, `beur_num`, `refference`, etc.)

This is not only OCR or table reading. It is a **grounded structured information extraction** task with hard schema constraints.

## 2) Widely Accepted Terms You Can Use
- Document AI / Document Understanding
- Key Information Extraction (KIE)
- Document Information Extraction (DIE)
- Table Understanding / Table IE
- Visual Information Extraction (VIE)
- Multimodal Information Extraction
- OCR-free Document Parsing (for end-to-end VLM settings)
- Schema-constrained generation
- Structured output generation
- Grounded extraction (text + location alignment)

## 3) Recommended Canonical Task Name
Use this as the primary label in docs/experiments:

**Schema-Constrained Grounded Financial Fact Extraction (SC-GFFE)**  

Why this name:
- `Schema-constrained`: strict JSON validity is mandatory.
- `Grounded`: each fact is tied to a `bbox`.
- `Financial fact extraction`: target is accounting facts, not free-form summarization.

## 4) Formal Scope
### In scope
- Page-level extraction from images.
- Financial statement page typing (`balance_sheet`, `income_statement`, etc.).
- Numeric facts including placeholders (`-`, `—`, `–`) and percentages.
- Hierarchical semantic mapping via `path`.
- Context normalization (`date`, `currency`, `scale`, `value_type`).
- Valid strict JSON with no extra keys.

### Out of scope (for now)
- Cross-page reconciliation (e.g., checking totals across statements).
- Entity-level consolidation across an entire report.
- Auditing/accounting correctness beyond extraction fidelity.
- PDF-native text parsing as a required dependency (if running OCR-free VLM mode).

## 5) Practical Success Criteria
- Schema validity rate (strict parse to `PageExtraction`).
- Fact recall/precision at value level.
- Grounding quality (bbox IoU / center-hit on value tokens).
- Semantic accuracy for `path`, `date`, `currency`, `scale`, `value_type`.
- Duplicate-fact rate (important data-quality failure mode observed in current annotations).

## 6) Key Difficulty Axes
- Dense tables with weak separators and mixed reading order.
- Hebrew RTL layout + numeric LTR patterns.
- Implicit context (scale/currency/date in header, not per-cell).
- Reference fields (`beur_num` vs `refference`) requiring document semantics.
- Strict formatting requirements (null vs empty string, enum restrictions).

## 7) Better Ways to Frame the Same Project
### Framing A: End-to-End Transduction
**Image -> strict JSON** with constrained decoding/validation loop.  
Best when you want minimal pipeline complexity and fast iteration.

### Framing B: Grounded Set Prediction
Predict an unordered set of facts with geometry + attributes, then canonicalize order.  
Best when recall and deduplication are priority.

### Framing C: Layout Graph Extraction
Build a graph (cells/headers/notes/refs), then map graph -> schema facts.  
Best when `path`, note linking, and reference reasoning are bottlenecks.

### Framing D: Hybrid Two-Stage System
Stage 1: detect/segment values + headers; Stage 2: VLM semantic normalization to JSON.  
Best when model size/latency budgets matter and controllability is needed.

## 8) Suggested Primary Framing for This Repo (near-term)
Use **Framing D (Hybrid Two-Stage)** as engineering default, while retaining end-to-end VLM as baseline:
1. Keep current single-model extraction path as baseline.
2. Add explicit post-generation validator + repair pass (schema + business rules).
3. Add deduplication and consistency checks in training data and eval.
4. Evaluate both with the same metric suite, not only token loss.

This gives immediate quality gains without blocking on major architecture changes.
