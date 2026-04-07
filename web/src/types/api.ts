/** API request/response types for the FineTree backend. */

import type { PageState } from "./schema";

// ── Workspace ──────────────────────────────────────────────────────

export interface WorkspaceDocument {
  doc_id: string;
  source_pdf: string | null;
  images_dir: string;
  annotations_path: string | null;
  thumbnail_path: string | null;
  thumbnail_name: string | null;
  page_count: number;
  annotated_page_count: number;
  approved_page_count: number;
  progress_pct: number;
  status: string;
  updated_at: number | null;
  fact_count: number;
  annotated_token_count: number;
  reg_flag_count: number;
  warning_count: number;
  pages_with_reg_flags: number;
  pages_with_warnings: number;
  checked: boolean;
  reviewed: boolean;
  // Legacy compat
  has_annotations?: boolean;
}

// ── Annotations ────────────────────────────────────────────────────

/** Raw shape returned by GET /api/annotations/{doc_id} */
export interface DocumentPayload {
  images_dir: string;
  page_images: string[];
  document_meta: Record<string, unknown>;
  page_states: Record<string, PageState>;
}

export interface SaveDocumentRequest {
  raw_payload?: Record<string, unknown>;
  page_states?: Record<string, PageState>;
}

export interface ValidationIssue {
  code: string;
  message: string;
  fact_index: number | null;
}

export interface ValidationResult {
  pages: Record<
    string,
    {
      reg_flags: ValidationIssue[];
      warnings: ValidationIssue[];
    }
  >;
}

// ── AI ─────────────────────────────────────────────────────────────

export interface ExtractRequest {
  doc_id: string;
  page_name: string;
  provider: "gemini" | "qwen";
  action?: string;
  config?: Record<string, unknown>;
}

export interface FillRequest {
  doc_id: string;
  page_name: string;
  facts: Record<string, unknown>[];
  fields: string[];
  provider?: string;
  config?: Record<string, unknown>;
}

export interface DetectBboxRequest {
  doc_id: string;
  page_name: string;
  backend?: string;
}

// ── Schema options ─────────────────────────────────────────────────

export interface SchemaOptions {
  canonical_fact_keys: string[];
  currency_values: string[];
  scale_values: number[];
  enums: Record<string, string[]>;
}

// ── SSE event types ────────────────────────────────────────────────

export type SSEEventType = "chunk" | "bbox" | "done" | "error" | "cancelled";

export interface SSEEvent {
  type: SSEEventType;
  text?: string;
  data?: unknown;
  message?: string;
}
