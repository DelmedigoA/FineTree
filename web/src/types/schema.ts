/** TypeScript mirrors of Python schemas.py Pydantic models. */

// ── Enums ──────────────────────────────────────────────────────────

export type PageType =
  | "title"
  | "contents"
  | "declaration"
  | "statements"
  | "other";

export type StatementType =
  | "balance_sheet"
  | "income_statement"
  | "cash_flow_statement"
  | "statement_of_changes_in_equity"
  | "notes_to_financial_statements"
  | "board_of_directors_report"
  | "auditors_report"
  | "statement_of_activities"
  | "other_declaration";

export type ValueType = "amount" | "percent" | "ratio" | "count" | "points";

export type ValueContext = "textual" | "tabular" | "mixed";

export type NaturalSign = "positive" | "negative";

export type RowRole = "detail" | "total";

export type Currency = "ILS" | "USD" | "EUR" | "GBP";

export type Scale = 1 | 1000 | 1000000;

export type DocLanguage = "he" | "en";

export type ReadingDirection = "rtl" | "ltr";

export type EntityType =
  | "state_owned_enterprise"
  | "public_company"
  | "private_company"
  | "registered_nonprofit"
  | "nonprofit_npo"
  | "public_benefit_company"
  | "partnership"
  | "limited_partnership"
  | "limited_liability_company"
  | "other";

export type ReportScope =
  | "separate"
  | "consolidated"
  | "combined"
  | "pro_forma"
  | "other";

export type PeriodType = "instant" | "duration" | "expected";

export type DurationType = "recurrent";

export type RecurringPeriod = "daily" | "quarterly" | "monthly" | "yearly";

export type AnnotationStatus = "approved" | "flagged";

export type PathSource = "observed" | "inferred";

// ── Models ─────────────────────────────────────────────────────────

export interface BBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface EquationVariant {
  equation: string;
  fact_equation: string | null;
}

export interface PageMeta {
  entity_name: string | null;
  page_num: string | null;
  page_type: PageType;
  statement_type: StatementType | null;
  title: string | null;
  annotation_note: string | null;
  annotation_status: AnnotationStatus | null;
}

export interface Fact {
  value: string;
  fact_num: number | null;
  equations: EquationVariant[] | null;
  natural_sign: NaturalSign | null;
  row_role: RowRole;
  comment_ref: string | null;
  note_flag: boolean;
  note_name: string | null;
  note_num: string | null;
  note_ref: string | null;
  date: string | null;
  period_type: PeriodType | null;
  period_start: string | null;
  period_end: string | null;
  duration_type: DurationType | null;
  recurring_period: RecurringPeriod | null;
  path: string[];
  path_source: PathSource | null;
  currency: Currency | null;
  scale: Scale | null;
  value_type: ValueType | null;
  value_context: ValueContext | null;
}

export interface ExtractedFact extends Fact {
  bbox: BBox;
}

export interface Page {
  image: string | null;
  meta: PageMeta;
  facts: ExtractedFact[];
}

export interface Metadata {
  language: DocLanguage | null;
  reading_direction: ReadingDirection | null;
  company_name: string | null;
  company_id: string | null;
  report_year: number | null;
  report_scope: ReportScope | null;
  entity_type: EntityType | null;
}

export interface Document {
  schema_version: number;
  images_dir: string | null;
  metadata: Metadata | null;
  pages: Page[];
}

// ── Internal working types ─────────────────────────────────────────

/** Client-side box record matching annotation_core.BoxRecord. */
export interface BoxRecord {
  bbox: BBox;
  fact: Record<string, unknown>;
}

/** Client-side page state matching annotation_core.PageState. */
export interface PageState {
  meta: Record<string, unknown>;
  facts: BoxRecord[];
}

/** Default fact data matching annotation_core.default_fact_data(). */
export function defaultFactData(): Record<string, unknown> {
  return {
    value: "",
    fact_num: null,
    equations: null,
    natural_sign: null,
    row_role: "detail",
    comment_ref: null,
    note_flag: false,
    note_name: null,
    note_num: null,
    note_ref: null,
    date: null,
    period_type: null,
    period_start: null,
    period_end: null,
    duration_type: null,
    recurring_period: null,
    path: [],
    path_source: null,
    currency: null,
    scale: null,
    value_type: null,
    value_context: null,
  };
}

export function defaultPageMeta(): PageMeta {
  return {
    entity_name: null,
    page_num: null,
    page_type: "other",
    statement_type: null,
    title: null,
    annotation_note: null,
    annotation_status: null,
  };
}

export const CURRENT_SCHEMA_VERSION = 4;
