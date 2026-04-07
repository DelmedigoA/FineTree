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

/** Evaluate an equation string like "123 - 254" or "- 131 + 50" → number.
 *  Handles leading "- " prefix, blank/dash tokens as 0. */
export function evaluateEquationString(equation: string): number | null {
  const trimmed = equation.trim();
  if (!trimmed) return null;
  // Split on whitespace before a +/- operator token.
  const tokens = trimmed.split(/\s+(?=[+-])/);
  let result: number | null = null;
  for (const token of tokens) {
    const t = token.trim();
    if (!t) continue;
    let sign = 1;
    let valueStr = t;
    if (t.startsWith("-")) { sign = -1; valueStr = t.slice(1).trim(); }
    else if (t.startsWith("+")) { sign = 1; valueStr = t.slice(1).trim(); }
    // Strip any remaining brackets/commas.
    const cleaned = valueStr.replace(/[<>()[\],]/g, "").replace(/\s/g, "").trim();
    const num = (cleaned === "" || cleaned === "-") ? 0 : parseFloat(cleaned);
    if (isNaN(num)) return null;
    result = result === null ? sign * num : result + sign * num;
  }
  return result;
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
  const parts: string[] = [];
  const factParts: string[] = [];
  let result: number | null = 0;
  let validCount = 0;

  for (const idx of termIndices) {
    const fact = facts[idx];
    if (!fact) continue;

    const manualOp = operators.get(idx) ?? "+";
    const factNum = fact.fact.fact_num as number | null;
    const rawValue = String(fact.fact.value ?? "");

    // 1. Detect parentheses negative: "(254)" → isParenNeg=true, workValue="254"
    let isParenNeg = false;
    let workValue = rawValue.trim();
    if (workValue.startsWith("(") && workValue.endsWith(")")) {
      isParenNeg = true;
      workValue = workValue.slice(1, -1).trim();
    }

    // 2. Strip angle brackets, commas for numeric parsing.
    const cleaned = workValue.replace(/[<>[\]]/g, "").replace(/,/g, "").trim();

    // 3. Handle blank / dash as zero.
    const isBlankOrDash = cleaned === "" || cleaned === "-";
    const absNum = isBlankOrDash ? 0 : parseFloat(cleaned);

    // 4. Intrinsic sign = parens × natural_sign field.
    const naturalSign = (fact.fact as Record<string, unknown>).natural_sign as string | null;
    const naturalSignMult = naturalSign === "negative" ? -1 : 1;
    const intrinsicSign = (isParenNeg ? -1 : 1) * naturalSignMult; // ±1

    // 5. Effective sign = intrinsic × manual operator override.
    const opMult = manualOp === "-" ? -1 : 1;
    const effectiveSign = intrinsicSign * opMult; // ±1

    // 6. Display value — stripped number, no parens.
    const displayValue = isBlankOrDash ? "0" : cleaned;

    // 7. Display prefix — based on effective sign.
    //    First term: no prefix if positive, "- " if negative.
    //    Subsequent terms: always explicit "+ " or "- ".
    const prefix = validCount === 0
      ? (effectiveSign < 0 ? "- " : "")
      : (effectiveSign < 0 ? "- " : "+ ");

    parts.push(`${prefix}${displayValue}`);

    // 8. Fact reference uses same prefix.
    const factLabel = factNum ? `f${factNum}` : `[${idx + 1}]`;
    factParts.push(`${prefix}${factLabel}`);

    // 9. Contribution to result.
    if (isNaN(absNum)) {
      result = null;
    } else if (result !== null) {
      result += absNum * effectiveSign;
    }

    validCount++;
  }

  return {
    equation: parts.join(" "),
    factEquation: factParts.join(" "),
    result,
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
): "ok" | "mismatch" | "unknown" {
  if (result === null) return "unknown";
  const cleaned = targetValue
    .replace(/[<>(),]/g, "")
    .replace(/\s/g, "")
    .trim();
  const targetNum = parseFloat(cleaned);
  if (isNaN(targetNum)) return "unknown";
  return Math.abs(result - targetNum) < 0.01 ? "ok" : "mismatch";
}
