import type {
  ApiBBox,
  ApiDocumentDetail,
  ApiDocumentMeta,
  ApiDocumentSaveRequest,
  ApiFact,
  DocumentDraft,
  FactDraft,
  PageDraft
} from "./types";

const PAGE_TYPES = [
  "balance_sheet",
  "income_statement",
  "cash_flow",
  "activities",
  "notes",
  "contents_page",
  "title",
  "declaration",
  "profits",
  "other"
] as const;
const CURRENCIES = new Set(["ILS", "USD", "EUR", "GBP"]);
const SCALES = new Set([1, 1000, 1000000]);
const VALUE_TYPES = new Set(["amount", "%"]);

let factCounter = 0;

function nextFactId(prefix: string): string {
  factCounter += 1;
  return `${prefix}-${factCounter}`;
}

export function cloneDocument(draft: DocumentDraft): DocumentDraft {
  return structuredClone(draft);
}

export function hydrateDocument(document: ApiDocumentDetail): DocumentDraft {
  return {
    ...structuredClone(document),
    pages: document.pages.map((page) => ({
      ...structuredClone(page),
      facts: sortFactsForPage(
        page.facts.map((fact, index) => ({
          ...structuredClone(fact),
          id: nextFactId(`${page.image}-${index}`)
        })),
        document.document_meta,
        page.meta
      )
    }))
  };
}

export function dehydrateDocument(draft: DocumentDraft): ApiDocumentSaveRequest {
  return {
    document_meta: structuredClone(draft.document_meta),
    pages: draft.pages.map((page) => ({
      image: page.image,
      meta: structuredClone(page.meta),
      facts: sortFactsForPage(page.facts, draft.document_meta, page.meta).map(({ id, ...fact }) =>
        structuredClone(fact)
      )
    }))
  };
}

export function pageHasAnnotation(
  pageMeta: { entity_name: string | null; page_num: string | null; type: string; title: string | null },
  facts: FactDraft[]
): boolean {
  if (facts.length > 0) {
    return true;
  }
  if (normalizeOptionalText(pageMeta.page_num)) {
    return true;
  }
  return normalizePageType(pageMeta.type) !== "other";
}

export function createEmptyFact(bbox: ApiBBox): FactDraft {
  return {
    id: nextFactId("fact"),
    bbox: normalizeBBox(bbox),
    value: "",
    comment: null,
    note_flag: false,
    note_name: null,
    note_num: null,
    note_reference: null,
    date: null,
    path: [],
    currency: null,
    scale: null,
    value_type: null
  };
}

export function normalizeBBox(raw: Partial<ApiBBox> | number[] | null | undefined): ApiBBox {
  let xRaw = 0;
  let yRaw = 0;
  let wRaw = 1;
  let hRaw = 1;
  if (Array.isArray(raw)) {
    [xRaw, yRaw, wRaw, hRaw] = raw;
  } else if (raw) {
    xRaw = Number(raw.x ?? 0);
    yRaw = Number(raw.y ?? 0);
    wRaw = Number(raw.w ?? 1);
    hRaw = Number(raw.h ?? 1);
  }
  return {
    x: round2(xRaw),
    y: round2(yRaw),
    w: round2(Math.max(wRaw, 1)),
    h: round2(Math.max(hRaw, 1))
  };
}

export function clampBBox(bbox: ApiBBox, width: number, height: number): ApiBBox {
  const normalized = normalizeBBox(bbox);
  const x = Math.max(0, Math.min(normalized.x, Math.max(0, width - normalized.w)));
  const y = Math.max(0, Math.min(normalized.y, Math.max(0, height - normalized.h)));
  const w = Math.min(normalized.w, Math.max(1, width - x));
  const h = Math.min(normalized.h, Math.max(1, height - y));
  return {
    x: round2(x),
    y: round2(y),
    w: round2(Math.max(1, w)),
    h: round2(Math.max(1, h))
  };
}

export function normalizeDocumentMeta(meta: Partial<ApiDocumentMeta>): ApiDocumentMeta {
  const language = meta.language === "he" || meta.language === "en" ? meta.language : null;
  const direction = meta.reading_direction === "rtl" || meta.reading_direction === "ltr" ? meta.reading_direction : null;
  const reportYear =
    typeof meta.report_year === "number" && Number.isInteger(meta.report_year)
      ? meta.report_year
      : typeof meta.report_year === "string" && /^\d+$/.test(meta.report_year)
        ? Number(meta.report_year)
        : null;
  return {
    language,
    reading_direction: direction,
    company_name: normalizeOptionalText(meta.company_name),
    company_id: normalizeOptionalText(meta.company_id),
    report_year: reportYear
  };
}

export function normalizeFact(fact: Partial<ApiFact> & { bbox?: ApiBBox | number[] }): ApiFact {
  const currencyText = normalizeOptionalText(fact.currency)?.toUpperCase() ?? null;
  const valueTypeText = normalizeOptionalText(fact.value_type);
  const scaleValue = fact.scale == null ? null : Number(fact.scale);
  return {
    bbox: normalizeBBox(fact.bbox),
    value: String(fact.value ?? "").trim(),
    comment: normalizeOptionalText(fact.comment),
    note_flag: normalizeBoolean(fact.note_flag),
    note_name: normalizeOptionalText(fact.note_name),
    note_num: normalizeInteger(fact.note_num),
    note_reference: normalizeOptionalText(fact.note_reference),
    date: normalizeOptionalText(fact.date),
    path: normalizePath(fact.path),
    currency: currencyText && CURRENCIES.has(currencyText) ? currencyText : null,
    scale: scaleValue != null && SCALES.has(scaleValue) ? scaleValue : null,
    value_type: valueTypeText && VALUE_TYPES.has(valueTypeText) ? valueTypeText : null
  };
}

function normalizePageType(value: unknown): string {
  const text = String(value ?? "").trim();
  return PAGE_TYPES.includes(text as (typeof PAGE_TYPES)[number]) ? text : "other";
}

export function sortFactsForPage(
  facts: FactDraft[],
  documentMeta: ApiDocumentMeta,
  pageMeta: { entity_name: string | null; page_num: string | null; type: string; title: string | null }
): FactDraft[] {
  if (facts.length <= 1) {
    return [...facts];
  }
  const direction = resolveReadingDirection(documentMeta, pageMeta, facts);
  const tolerance = rowTolerance(facts.map((fact) => fact.bbox.h));
  const sortedByY = [...facts].sort((left, right) => {
    if (left.bbox.y === right.bbox.y) {
      return left.bbox.x - right.bbox.x;
    }
    return left.bbox.y - right.bbox.y;
  });

  const rows: FactDraft[][] = [];
  let currentRow: FactDraft[] = [];
  let anchor = 0;
  sortedByY.forEach((fact) => {
    const cy = fact.bbox.y + fact.bbox.h / 2;
    if (currentRow.length === 0) {
      currentRow = [fact];
      anchor = cy;
      return;
    }
    if (Math.abs(cy - anchor) <= tolerance) {
      currentRow.push(fact);
      anchor = currentRow.reduce((sum, item) => sum + item.bbox.y + item.bbox.h / 2, 0) / currentRow.length;
      return;
    }
    rows.push(currentRow);
    currentRow = [fact];
    anchor = cy;
  });
  if (currentRow.length > 0) {
    rows.push(currentRow);
  }

  return rows.flatMap((row) =>
    [...row].sort((left, right) => {
      const leftCenter = left.bbox.x + left.bbox.w / 2;
      const rightCenter = right.bbox.x + right.bbox.w / 2;
      if (direction === "ltr") {
        return leftCenter - rightCenter;
      }
      return rightCenter - leftCenter;
    })
  );
}

function resolveReadingDirection(
  documentMeta: ApiDocumentMeta,
  pageMeta: { entity_name: string | null; page_num: string | null; type: string; title: string | null },
  facts: FactDraft[]
): "rtl" | "ltr" {
  if (documentMeta.reading_direction === "rtl" || documentMeta.reading_direction === "ltr") {
    return documentMeta.reading_direction;
  }
  if (documentMeta.language === "he") {
    return "rtl";
  }
  if (documentMeta.language === "en") {
    return "ltr";
  }

  let hebrew = 0;
  let latin = 0;
  const texts = [
    pageMeta.entity_name,
    pageMeta.page_num,
    pageMeta.title,
    pageMeta.type,
    ...facts.flatMap((fact) => [
      fact.value,
      fact.comment,
      fact.note_name,
      fact.note_reference,
      fact.date,
      fact.note_num == null ? null : String(fact.note_num),
      ...fact.path
    ])
  ];
  texts.forEach((value) => {
    const text = String(value ?? "");
    for (const ch of text) {
      const code = ch.charCodeAt(0);
      if (code >= 0x0590 && code <= 0x05ff) {
        hebrew += 1;
      } else if ((ch >= "A" && ch <= "Z") || (ch >= "a" && ch <= "z")) {
        latin += 1;
      }
    }
  });
  if (hebrew === 0 && latin > 0) {
    return "ltr";
  }
  if (latin === 0 && hebrew > 0) {
    return "rtl";
  }
  if (latin > 0 && hebrew / latin >= 1.2) {
    return "rtl";
  }
  if (hebrew > 0 && latin / hebrew >= 1.2) {
    return "ltr";
  }
  return "rtl";
}

function rowTolerance(heights: number[]): number {
  const valid = heights.filter((height) => height > 0).sort((left, right) => left - right);
  if (valid.length === 0) {
    return 6;
  }
  const mid = Math.floor(valid.length / 2);
  const median = valid.length % 2 === 0 ? (valid[mid - 1] + valid[mid]) / 2 : valid[mid];
  return Math.max(6, median * 0.35);
}

export function applyEntityNameAcrossPages(draft: DocumentDraft, overwriteExisting: boolean): DocumentDraft {
  const next = cloneDocument(draft);
  const currentPage = next.pages.find((page) => page.image === next.pages[0]?.image);
  const sourceEntity =
    normalizeOptionalText(
      next.pages.find((page) => page.image === currentPage?.image)?.meta.entity_name ?? next.pages[0]?.meta.entity_name
    ) ?? null;
  if (!sourceEntity) {
    return next;
  }
  next.pages = next.pages.map((page) => {
    if (page.meta.entity_name && !overwriteExisting) {
      return page;
    }
    return {
      ...page,
      meta: {
        ...page.meta,
        entity_name: sourceEntity
      }
    };
  });
  return next;
}

export function applyEntityNameUsingValue(
  draft: DocumentDraft,
  entityName: string,
  overwriteExisting: boolean
): DocumentDraft {
  const normalizedEntity = normalizeOptionalText(entityName);
  if (!normalizedEntity) {
    return cloneDocument(draft);
  }
  const next = cloneDocument(draft);
  next.pages = next.pages.map((page) => {
    if (page.meta.entity_name && !overwriteExisting) {
      return page;
    }
    return {
      ...page,
      meta: {
        ...page.meta,
        entity_name: normalizedEntity
      }
    };
  });
  return next;
}

export function mergeImportedPayload(
  draft: DocumentDraft,
  rawPayload: unknown,
  defaultPageImage: string,
  normalized1000: boolean
): DocumentDraft {
  const next = cloneDocument(draft);
  const pageMap = new Map(next.pages.map((page) => [page.image, page]));
  const payload =
    Array.isArray(rawPayload) ? { meta: {}, facts: rawPayload } : typeof rawPayload === "object" && rawPayload ? rawPayload : {};
  const pages = extractImportedPages(payload, defaultPageImage);
  pages.forEach((rawPage) => {
    const image = String(rawPage.image ?? defaultPageImage).trim();
    const targetPage = pageMap.get(image);
    if (!targetPage) {
      return;
    }
    const metaInput = rawPage.meta && typeof rawPage.meta === "object" ? rawPage.meta : {};
    const factsInput = Array.isArray(rawPage.facts) ? rawPage.facts : [];
    const nextFacts = factsInput
      .map((fact, index) => {
        if (!fact || typeof fact !== "object") {
          return null;
        }
        const rawFact = "fact" in fact && fact.fact && typeof fact.fact === "object" ? fact.fact : fact;
        let bbox = normalizeBBox((fact as { bbox?: ApiBBox | number[] }).bbox);
        if (normalized1000 && looksNormalized1000(bbox)) {
          bbox = {
            x: round2((bbox.x * targetPage.width) / 1000),
            y: round2((bbox.y * targetPage.height) / 1000),
            w: round2((bbox.w * targetPage.width) / 1000),
            h: round2((bbox.h * targetPage.height) / 1000)
          };
        }
        return {
          ...normalizeFact({ ...(rawFact as Partial<ApiFact>), bbox }),
          id: nextFactId(`${image}-import-${index}`)
        } satisfies FactDraft;
      })
      .filter((fact): fact is FactDraft => fact != null);
    pageMap.set(image, {
      ...targetPage,
      meta: {
        entity_name: normalizeOptionalText((metaInput as Record<string, unknown>).entity_name),
        page_num: normalizeOptionalText((metaInput as Record<string, unknown>).page_num),
        type: normalizePageType((metaInput as Record<string, unknown>).type),
        title: normalizeOptionalText((metaInput as Record<string, unknown>).title)
      },
      facts: nextFacts,
      annotated: pageHasAnnotation(
        {
          entity_name: normalizeOptionalText((metaInput as Record<string, unknown>).entity_name),
          page_num: normalizeOptionalText((metaInput as Record<string, unknown>).page_num),
          type: normalizePageType((metaInput as Record<string, unknown>).type),
          title: normalizeOptionalText((metaInput as Record<string, unknown>).title)
        },
        nextFacts
      )
    });
  });

  const documentMeta = extractDocumentMeta(payload);
  if (Object.values(documentMeta).some((value) => value != null && value !== "")) {
    next.document_meta = documentMeta;
  }
  next.pages = next.pages.map((page) => {
    const updated = pageMap.get(page.image) ?? page;
    return {
      ...updated,
      facts: sortFactsForPage(updated.facts, next.document_meta, updated.meta)
    };
  });
  return next;
}

function extractImportedPages(
  payload: unknown,
  defaultPageImage: string
): Array<{ image?: unknown; meta?: unknown; facts?: unknown }> {
  if (payload && typeof payload === "object" && Array.isArray((payload as { pages?: unknown }).pages)) {
    return (payload as { pages: Array<{ image?: unknown; meta?: unknown; facts?: unknown }> }).pages;
  }
  if (payload && typeof payload === "object") {
    return [
      {
        image: (payload as { image?: unknown }).image ?? defaultPageImage,
        meta: (payload as { meta?: unknown }).meta,
        facts: (payload as { facts?: unknown }).facts
      }
    ];
  }
  return [];
}

function extractDocumentMeta(payload: unknown): ApiDocumentMeta {
  if (!payload || typeof payload !== "object" || !("document_meta" in payload)) {
    return normalizeDocumentMeta({});
  }
  return normalizeDocumentMeta((payload as { document_meta?: Partial<ApiDocumentMeta> }).document_meta ?? {});
}

function looksNormalized1000(bbox: ApiBBox): boolean {
  return (
    bbox.x >= 0 &&
    bbox.y >= 0 &&
    bbox.w >= 0 &&
    bbox.h >= 0 &&
    bbox.x + bbox.w <= 1000 &&
    bbox.y + bbox.h <= 1000
  );
}

export function duplicateSelectedFacts(
  page: PageDraft,
  selectedIds: string[],
  documentMeta: ApiDocumentMeta
): PageDraft {
  if (selectedIds.length === 0) {
    return page;
  }
  const selected = page.facts.filter((fact) => selectedIds.includes(fact.id));
  const duplicates = selected.map((fact) => ({
    ...structuredClone(fact),
    id: nextFactId(`${page.image}-dup`),
    bbox: clampBBox(
      {
        ...fact.bbox,
        x: fact.bbox.x + 12,
        y: fact.bbox.y + 12
      },
      page.width,
      page.height
    )
  }));
  return {
    ...page,
    facts: sortFactsForPage([...page.facts, ...duplicates], documentMeta, page.meta),
    annotated: pageHasAnnotation(page.meta, [...page.facts, ...duplicates])
  };
}

function normalizeOptionalText(value: unknown): string | null {
  const text = String(value ?? "").trim();
  return text ? text : null;
}

function normalizePath(value: unknown): string[] {
  if (Array.isArray(value)) {
    return value.map((item) => String(item).trim()).filter(Boolean);
  }
  if (typeof value === "string") {
    return value
      .split("\n")
      .map((item) => item.trim())
      .filter(Boolean);
  }
  return [];
}

function normalizeBoolean(value: unknown): boolean {
  if (typeof value === "boolean") {
    return value;
  }
  const text = String(value ?? "").trim().toLowerCase();
  return ["1", "true", "yes", "y"].includes(text);
}

function normalizeInteger(value: unknown): number | null {
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }
  const text = String(value ?? "").trim();
  return /^\d+$/.test(text) ? Number(text) : null;
}

function round2(value: number): number {
  return Math.round((Number(value) || 0) * 100) / 100;
}
