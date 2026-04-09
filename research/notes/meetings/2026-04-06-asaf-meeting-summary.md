# Meeting Summary: Asaf

## Meeting Details

- **Date:** April 6, 2026
- **Time:** 12:00-12:45
- **Participants:** Asaf, GPT-5.4
- **Purpose:** Define FineTree's product direction and clarify the right representation layer for financial documents.

## Summary

This meeting focused on defining FineTree's broader product direction and distinguishing between the internal extraction layer and the product-facing representation layer. The core conclusion was that `pdf2facts` remains a strong internal name for the general extraction layer, but it should not define the full product concept. FineTree should instead be framed around transforming financial statements into machine-usable structured representations for enterprise systems and AI workflows.

Two product directions were clarified. The first is structured financial data for enterprise systems, where extracted facts from financial statements are mapped into an organization's own schema, business fields, or internal data model. The second is agent-ready financial document context, where financial statements are converted into structured JSON or similar representations that AI systems can use for reasoning, enrichment, and downstream workflows.

The discussion also clarified that raw flat fact lists remain useful as a canonical extraction layer, but they are not the best end format for reasoning-heavy AI use cases. For long-context and agentic workflows, a more compact semantic representation is preferable, especially tree-like or typed block-based page structures that preserve order, hierarchy, grouping, and note boundaries.

## Key Discussion Points

- `pdf2facts` should remain the internal extraction-layer name, but it should not be treated as the product's external framing.
- FineTree should be positioned more broadly than "PDF to JSON" and instead focus on machine-usable representations of financial statements.
- One product direction is extracting financial statement facts and mapping them into organization-specific structured data models.
- A second product direction is producing agent-ready structured financial document context for AI systems.
- Flat fact lists are still useful as a canonical intermediate layer because they provide a general, reusable extraction output.
- Flat fact lists are not ideal as the final representation for model reasoning, especially in long-context or agentic settings.
- Tree-like or typed block-based page representations are stronger downstream formats because they preserve grouping, order, hierarchy, and note structure while staying compact.
- The general `pdf2facts` layer should remain domain-agnostic so it can be reused across document types beyond financial statements.
- Richer financial trees, note-aware structures, and other financial-specific representations should be downstream projections built on top of the generic extraction layer.
- The preferred AI product direction is agentic, schema-aware retrieval of exact pages, notes, statements, or blocks rather than semantic-search-first RAG.
- Semantic search may still be useful later as a fallback layer for fuzzier queries, but it should not be the foundation of the system.
- OCR-derived text is important for preserving narrative context such as accounting policies, note introductions, explanations, and qualifiers.
- The recommended representation approach is an ordered page structure that combines OCR text blocks with FineTree's structured fact blocks.

## Conclusions

- Keep `pdf2facts` as the internal name for the canonical extraction layer.
- Frame FineTree more broadly as infrastructure for structured financial document representations, not as a generic PDF-to-JSON product.
- Treat the enterprise structured-data product and the agent-ready context product as the two primary product directions.
- Use flat fact lists as the canonical upstream extraction output, but prefer more semantic downstream structures for reasoning and agent workflows.
- Keep the base extraction layer domain-agnostic and implement financial-specific trees or note-level structures as downstream projections.
- Prioritize agentic, typed retrieval over classic semantic-search-first RAG for the AI-facing product direction.
- Integrate textual OCR as a core complementary input, especially for the agent-ready context product.

## Next Steps

Document this product framing in future research and architecture notes so the distinction between the generic extraction layer and downstream financial representations remains explicit.
