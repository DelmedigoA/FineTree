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

/** Build equation text from ordered term facts. */
export function buildEquationText(
  termIndices: number[],
  facts: BoxRecord[],
  operators: Map<number, "+" | "-">,
): { equation: string; factEquation: string; result: number | null } {
  const parts: string[] = [];
  const factParts: string[] = [];
  let result: number | null = 0;

  for (let i = 0; i < termIndices.length; i++) {
    const idx = termIndices[i]!;
    const fact = facts[idx];
    if (!fact) continue;

    const op = operators.get(idx) ?? "+";
    const value = String(fact.fact.value ?? "");
    const factNum = fact.fact.fact_num as number | null;

    // Parse numeric value (strip angle brackets, commas, parens).
    const cleaned = value
      .replace(/[<>(),]/g, "")
      .replace(/\s/g, "")
      .trim();
    const num = parseFloat(cleaned);

    if (i === 0) {
      parts.push(value);
      factParts.push(factNum ? `f${factNum}` : `[${idx + 1}]`);
      result = isNaN(num) ? null : num * (op === "-" ? -1 : 1);
    } else {
      parts.push(`${op} ${value}`);
      factParts.push(`${op} ${factNum ? `f${factNum}` : `[${idx + 1}]`}`);
      if (result !== null && !isNaN(num)) {
        result += op === "-" ? -num : num;
      } else {
        result = null;
      }
    }
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

  const existingEquations = (target.fact.equations as unknown[]) ?? [];
  const newVariant = { equation, fact_equation: factEquation };

  newFacts[targetIndex] = {
    ...target,
    fact: {
      ...target.fact,
      equations: [newVariant, ...existingEquations],
      row_role: termIndices.length > 0 ? "total" : target.fact.row_role,
    },
  };

  updatePageState(pageName, { ...page, facts: newFacts });
  useCanvasStore.getState().markDirty("bbox");

  // Clear equation state.
  useSelectionStore.getState().clearEquationTerms();
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
