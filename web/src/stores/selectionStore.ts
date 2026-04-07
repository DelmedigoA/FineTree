import { create } from "zustand";
import type { SelectionRect } from "../types/canvas";

export interface SelectionStoreState {
  /** Indices of currently selected facts on the current page. */
  selectedIndices: Set<number>;
  /** Index of the fact under the cursor (for hover highlight). */
  hoveredIndex: number | null;
  /** Active rubber-band selection rectangle (world coords). */
  selectionRect: SelectionRect | null;
  /** Indices of bboxes selected as equation terms (for fast lookup). */
  equationTermIndices: Set<number>;
  /** Ordered list of equation term indices (insertion order preserved). */
  equationTermOrder: number[];
  /** Operator per term index — defaults to "+" if absent. */
  equationTermOperators: Map<number, "+" | "-">;
  /** Whether equation reference selection mode is active (Alt held). */
  equationModeActive: boolean;
}

export interface SelectionStoreActions {
  select(index: number): void;
  toggleSelect(index: number): void;
  addToSelection(indices: Iterable<number>): void;
  selectAll(count: number): void;
  clearSelection(): void;
  setHovered(index: number | null): void;
  setSelectionRect(rect: SelectionRect | null): void;
  setEquationModeActive(active: boolean): void;
  toggleEquationTerm(index: number): void;
  setEquationTerms(indices: Set<number>): void;
  /** Set equation terms with explicit ordering (preserves sweep order). */
  setEquationTermsOrdered(ordered: number[]): void;
  /** Toggle the operator (+/-) of an existing equation term. */
  toggleEquationTermOperator(index: number): void;
  clearEquationTerms(): void;
}

export const useSelectionStore = create<
  SelectionStoreState & SelectionStoreActions
>((set) => ({
  selectedIndices: new Set(),
  hoveredIndex: null,
  selectionRect: null,
  equationTermIndices: new Set(),
  equationTermOrder: [],
  equationTermOperators: new Map(),
  equationModeActive: false,

  select(index) {
    set({ selectedIndices: new Set([index]) });
  },

  toggleSelect(index) {
    set((s) => {
      const next = new Set(s.selectedIndices);
      if (next.has(index)) {
        next.delete(index);
      } else {
        next.add(index);
      }
      return { selectedIndices: next };
    });
  },

  addToSelection(indices) {
    set((s) => {
      const next = new Set(s.selectedIndices);
      for (const i of indices) next.add(i);
      return { selectedIndices: next };
    });
  },

  selectAll(count) {
    const all = new Set<number>();
    for (let i = 0; i < count; i++) all.add(i);
    set({ selectedIndices: all });
  },

  clearSelection() {
    set({ selectedIndices: new Set() });
  },

  setHovered(index) {
    set({ hoveredIndex: index });
  },

  setSelectionRect(rect) {
    set({ selectionRect: rect });
  },

  setEquationModeActive(active) {
    set({ equationModeActive: active });
  },

  toggleEquationTerm(index) {
    set((s) => {
      const next = new Set(s.equationTermIndices);
      let nextOrder: number[];
      if (next.has(index)) {
        next.delete(index);
        nextOrder = s.equationTermOrder.filter((i) => i !== index);
      } else {
        next.add(index);
        nextOrder = [...s.equationTermOrder, index];
      }
      return { equationTermIndices: next, equationTermOrder: nextOrder };
    });
  },

  setEquationTerms(indices) {
    const arr = [...indices];
    set({ equationTermIndices: new Set(arr), equationTermOrder: arr });
  },

  setEquationTermsOrdered(ordered) {
    set({ equationTermIndices: new Set(ordered), equationTermOrder: ordered });
  },

  toggleEquationTermOperator(index) {
    set((s) => {
      const next = new Map(s.equationTermOperators);
      const current = next.get(index) ?? "+";
      next.set(index, current === "+" ? "-" : "+");
      return { equationTermOperators: next };
    });
  },

  clearEquationTerms() {
    set({ equationTermIndices: new Set(), equationTermOrder: [], equationTermOperators: new Map() });
  },
}));
