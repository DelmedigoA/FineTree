/**
 * Equation workflow hook.
 *
 * State machine:
 *   IDLE -> Alt held (1 bbox selected) -> READY
 *   READY -> click bbox -> toggle term -> READY
 *   READY -> drag > 4px -> DRAGGING (sweep-select terms)
 *   DRAGGING -> mouseup -> READY (commit swept terms)
 *   READY -> Alt+Shift -> APPLY (save equation to target)
 *   READY -> Alt released -> IDLE
 *
 * Terms are ordered by drag vector (project bbox centers onto drag unit vector).
 */

import { useDocumentStore } from "../stores/documentStore";
import { useSelectionStore } from "../stores/selectionStore";
import { useCanvasStore } from "../stores/canvasStore";
import type { BoxRecord } from "../types/schema";

export interface EquationPreviewTerm {
  index: number;
  factNum: number | null;
  operator: "+" | "-";
  rawValue: string;
  status: "ok" | "normalized_dash" | "invalid";
}

export interface EquationPreview {
  equation: string | null;
  factEquation: string | null;
  result: number | null;
  terms: EquationPreviewTerm[];
}

function normalizeEquationOperator(value: unknown): "+" | "-" {
  const text = String(value ?? "").trim().toLowerCase();
  if (["-", "minus", "subtractive"].includes(text)) return "-";
  return "+";
}

function deriveNaturalSignFromValueText(value: unknown): "positive" | "negative" | null {
  const text = String(value ?? "").trim();
  if (!text || text === "-") return null;
  const normalized = normalizeAngleBracketedNumericText(text);
  if (!normalized || normalized === "-") return null;
  if ((normalized.startsWith("(") && normalized.endsWith(")")) || normalized.startsWith("-")) {
    return "negative";
  }
  return "positive";
}

function naturalSignMultiplier(value: unknown): 1 | -1 {
  return deriveNaturalSignFromValueText(value) === "negative" || String(value ?? "").trim().toLowerCase() === "negative"
    ? -1
    : 1;
}

function operatorMultiplier(value: unknown): 1 | -1 {
  return normalizeEquationOperator(value) === "-" ? -1 : 1;
}

function normalizeAngleBracketedNumericText(value: unknown): string {
  const raw = String(value ?? "").trim();
  if (raw.length < 3 || !raw.startsWith("<") || !raw.endsWith(">")) {
    return raw;
  }
  const inner = raw.slice(1, -1).trim();
  if ((inner.startsWith("(") && inner.endsWith(")")) || inner.startsWith("-")) {
    return inner;
  }
  return raw;
}

function parseFactValueForEquation(value: unknown): {
  parsed: number | null;
  display: string | null;
  status: "ok" | "normalized_dash" | "invalid";
} {
  const raw = normalizeAngleBracketedNumericText(value);
  if (!raw) {
    return { parsed: null, display: null, status: "invalid" };
  }
  if (raw === "-") {
    return { parsed: 0, display: "0", status: "normalized_dash" };
  }

  let negative = false;
  let text = raw;
  if (text.startsWith("(") && text.endsWith(")")) {
    negative = true;
    text = text.slice(1, -1).trim();
  }
  if (text.startsWith("+")) {
    text = text.slice(1).trim();
  } else if (text.startsWith("-")) {
    negative = true;
    text = text.slice(1).trim();
  }

  if (!/^\d[\d,]*(?:\.\d+)?$/.test(text)) {
    return { parsed: null, display: null, status: "invalid" };
  }

  const parsed = Number.parseFloat(text.replace(/,/g, ""));
  if (Number.isNaN(parsed)) {
    return { parsed: null, display: null, status: "invalid" };
  }
  return {
    parsed: negative ? -parsed : parsed,
    display: text,
    status: "ok",
  };
}

/** Evaluate an equation string like "123 - 254" or "- 131 + 50" → number. */
export function evaluateEquationString(equation: string): number | null {
  let text = equation.trim();
  if (!text) return null;
  if (text.includes("=")) {
    text = text.split("=", 1)[0]!.trim();
  }
  if (!text) return null;

  const tokens = text.match(/[+-]?\s*\d[\d,]*(?:\.\d+)?/g);
  if (!tokens || tokens.join("").replace(/\s+/g, "") !== text.replace(/\s+/g, "")) {
    return null;
  }

  let total = 0;
  for (const token of tokens) {
    const normalized = token.replace(/\s+/g, "").replace(/,/g, "");
    const value = Number.parseFloat(normalized);
    if (Number.isNaN(value)) return null;
    total += value;
  }
  return total;
}

/** Order bbox indices by projecting their centers onto a drag direction vector. */
export function orderByDragVector(
  indices: Set<number>,
  bboxes: BoxRecord[],
  dragStartX: number,
  dragStartY: number,
  dragEndX: number,
  dragEndY: number,
): number[] {
  const dx = dragEndX - dragStartX;
  const dy = dragEndY - dragStartY;
  const len = Math.sqrt(dx * dx + dy * dy);
  if (len < 1) return [...indices];

  const ux = dx / len;
  const uy = dy / len;

  const projected = [...indices].map((idx) => {
    const bbox = bboxes[idx]?.bbox;
    if (!bbox) return { idx, proj: 0 };
    const cx = bbox.x + bbox.w / 2;
    const cy = bbox.y + bbox.h / 2;
    return { idx, proj: cx * ux + cy * uy };
  });

  projected.sort((a, b) => a.proj - b.proj);
  return projected.map((p) => p.idx);
}

/**
 * Build equation text from ordered term facts.
 *
 * Sign derivation (matches PyQt5 logic):
 *   1. Parentheses in value "(254)" → intrinsic negative (accounting convention).
 *   2. `natural_sign === "negative"` → additional sign flip.
 *   3. User double-click operator override (Map entry) → further flip.
 *   Effective sign = parens_sign × natural_sign_mult × operator_mult.
 *
 * Display: shows stripped value (no parens/brackets), prefix = effective sign.
 */
export function buildEquationText(
  termIndices: number[],
  facts: BoxRecord[],
  operators: Map<number, "+" | "-">,
): { equation: string; factEquation: string; result: number | null } {
  const preview = buildEquationPreview(termIndices, facts, operators);
  return {
    equation: preview.equation ?? "",
    factEquation: preview.factEquation ?? "",
    result: preview.result,
  };
}

export function buildEquationPreview(
  termIndices: number[],
  facts: BoxRecord[],
  operators: Map<number, "+" | "-">,
): EquationPreview {
  const parts: string[] = [];
  const factParts: string[] = [];
  const terms: EquationPreviewTerm[] = [];
  let result: number | null = 0;
  let validCount = 0;

  for (const idx of termIndices) {
    const fact = facts[idx];
    if (!fact) continue;

    const manualOp = normalizeEquationOperator(operators.get(idx));
    const factNum = fact.fact.fact_num as number | null;
    const rawValue = fact.fact.value;
    const naturalSign = (fact.fact as Record<string, unknown>).natural_sign;
    const { parsed, display, status } = parseFactValueForEquation(rawValue);
    const magnitude = parsed === null ? null : Math.abs(parsed);
    const contributionSign = naturalSignMultiplier(naturalSign ?? rawValue) * operatorMultiplier(manualOp);
    const effectiveValue = magnitude === null ? null : magnitude * contributionSign;
    const rawValueText = String(rawValue ?? "").trim() || "<empty>";

    terms.push({
      index: idx,
      factNum,
      operator: manualOp,
      rawValue: rawValueText,
      status,
    });

    if (parsed === null || display === null || factNum === null) {
      result = null;
      continue;
    }

    const forceOperatorSign = status === "normalized_dash";
    let prefix = "";
    if (validCount > 0) {
      prefix = forceOperatorSign
        ? contributionSign < 0 ? "- " : "+ "
        : (effectiveValue ?? 0) < 0 ? "- " : "+ ";
    } else if (forceOperatorSign) {
      prefix = contributionSign < 0 ? "- " : "";
    } else if ((effectiveValue ?? 0) < 0) {
      prefix = "- ";
    }

    parts.push(prefix ? `${prefix}${display}` : display);

    const factPrefix =
      validCount > 0 ? (manualOp === "-" ? "- " : "+ ")
      : manualOp === "-" ? "- "
      : "";
    factParts.push(factPrefix ? `${factPrefix}f${factNum}` : `f${factNum}`);

    if (result !== null) {
      result += effectiveValue ?? 0;
    }

    validCount++;
  }

  return {
    equation: validCount > 0 ? parts.join(" ") : null,
    factEquation: validCount > 0 ? factParts.join(" ") : null,
    result: validCount > 0 ? result : null,
    terms,
  };
}

/** Apply equation to the target fact. */
export function applyEquation(
  targetIndex: number,
  termIndices: number[],
  operators: Map<number, "+" | "-">,
) {
  const { pageStates, pageNames, currentPageIndex, updatePageState } =
    useDocumentStore.getState();
  const pageName = pageNames[currentPageIndex];
  if (!pageName) return;
  const page = pageStates.get(pageName);
  if (!page) return;

  const { equation, factEquation } = buildEquationText(
    termIndices,
    page.facts,
    operators,
  );

  const newFacts = [...page.facts];
  const target = newFacts[targetIndex];
  if (!target) return;

  const newVariant = { equation, fact_equation: factEquation };

  newFacts[targetIndex] = {
    ...target,
    fact: {
      ...target.fact,
      equations: [newVariant],
      row_role: termIndices.length > 0 ? "total" : target.fact.row_role,
    },
  };

  updatePageState(pageName, { ...page, facts: newFacts });

  // Clear equation terms + sweep rect, then mark dirty so re-render sees no yellow.
  const sel = useSelectionStore.getState();
  sel.clearEquationTerms();
  sel.setSelectionRect(null);
  useCanvasStore.getState().markDirty("bbox");
  useCanvasStore.getState().markDirty("interaction");
}

/** Check if equation result matches target value. Returns "ok" | "mismatch" | "unknown". */
export function equationMatchState(
  result: number | null,
  targetValue: string,
  targetNaturalSign?: string | null,
): "ok" | "mismatch" | "unknown" {
  if (result === null) return "unknown";
  const { parsed } = parseFactValueForEquation(targetValue);
  if (parsed === null) return "unknown";
  const signSource = targetNaturalSign ?? deriveNaturalSignFromValueText(targetValue);
  const targetNum = Math.abs(parsed) * naturalSignMultiplier(signSource);
  return Math.abs(result - targetNum) < 0.01 ? "ok" : "mismatch";
}
