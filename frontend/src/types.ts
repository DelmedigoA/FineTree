export type ApiBBox = {
  x: number;
  y: number;
  w: number;
  h: number;
};

export type ApiFact = {
  bbox: ApiBBox;
  value: string;
  comment: string | null;
  note_flag: boolean;
  note_name: string | null;
  note_num: number | null;
  note_reference: string | null;
  date: string | null;
  path: string[];
  currency: string | null;
  scale: number | null;
  value_type: string | null;
};

export type ApiPageMeta = {
  entity_name: string | null;
  page_num: string | null;
  type: string;
  title: string | null;
};

export type ApiDocumentMeta = {
  language: string | null;
  reading_direction: string | null;
  company_name: string | null;
  company_id: string | null;
  report_year: number | null;
};

export type ApiIssue = {
  severity: string;
  code: string;
  message: string;
  page_image: string;
  fact_index: number | null;
  field_name: string | null;
};

export type ApiPageIssueSummary = {
  page_image: string;
  reg_flag_count: number;
  warning_count: number;
  issues: ApiIssue[];
};

export type ApiDocumentIssueSummary = {
  reg_flag_count: number;
  warning_count: number;
  pages_with_reg_flags: number;
  pages_with_warnings: number;
  page_summaries: Record<string, ApiPageIssueSummary>;
};

export type ApiFormatFinding = {
  page: string | null;
  fact_index: number | null;
  issue_codes: string[];
  message: string | null;
};

export type ApiDocumentPage = {
  image: string;
  image_path: string;
  width: number;
  height: number;
  meta: ApiPageMeta;
  facts: ApiFact[];
  annotated: boolean;
  issue_summary: ApiPageIssueSummary;
};

export type ApiDocumentDetail = {
  doc_id: string;
  source_pdf: string | null;
  images_dir: string;
  annotations_path: string;
  document_meta: ApiDocumentMeta;
  pages: ApiDocumentPage[];
  issue_summary: ApiDocumentIssueSummary;
  checked: boolean;
  reviewed: boolean;
  status: string;
  annotated_page_count: number;
  page_count: number;
  progress_pct: number;
  updated_at: number | null;
  save_warnings: ApiFormatFinding[];
};

export type ApiWorkspaceDocumentSummary = {
  doc_id: string;
  source_pdf: string | null;
  images_dir: string;
  annotations_path: string;
  thumbnail_path: string | null;
  page_count: number;
  annotated_page_count: number;
  progress_pct: number;
  status: string;
  updated_at: number | null;
  annotated_token_count: number;
  reg_flag_count: number;
  warning_count: number;
  pages_with_reg_flags: number;
  pages_with_warnings: number;
  checked: boolean;
  reviewed: boolean;
};

export type ApiDocumentSaveRequest = {
  document_meta: ApiDocumentMeta;
  pages: Array<{
    image: string;
    meta: ApiPageMeta;
    facts: ApiFact[];
  }>;
};

export type ApiDocumentSaveResponse = {
  document: ApiDocumentDetail;
  changed: boolean;
  save_warnings: ApiFormatFinding[];
};

export type ApiDocumentValidateResponse = {
  issue_summary: ApiDocumentIssueSummary;
  save_warnings: ApiFormatFinding[];
};

export type ApiExtractionResponse = {
  provider: "gemini" | "qwen";
  model: string;
  page_image: string;
  prompt: string;
  extraction: {
    meta: ApiPageMeta;
    facts: ApiFact[];
  };
};

export type ApiAppConfig = {
  startup_doc_id: string | null;
  frontend_dist: string | null;
  data_root: string;
  schema: {
    page_types: string[];
    value_types: string[];
    currencies: string[];
    scales: number[];
  };
};

export type FactDraft = ApiFact & { id: string };
export type PageDraft = Omit<ApiDocumentPage, "facts"> & { facts: FactDraft[] };
export type DocumentDraft = Omit<ApiDocumentDetail, "pages"> & { pages: PageDraft[] };
