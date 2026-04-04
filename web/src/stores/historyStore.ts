/** Undo/redo history store. Snapshots full page state before each edit. */

import { create } from "zustand";
import type { PageState } from "../types/schema";

interface Snapshot {
  pageStates: Map<string, PageState>;
  documentMeta: Record<string, unknown>;
}

export interface HistoryStoreState {
  undoStack: Snapshot[];
  redoStack: Snapshot[];
  maxSize: number;
}

export interface HistoryStoreActions {
  /** Take a snapshot of the current document state before an edit. */
  pushSnapshot(
    pageStates: Map<string, PageState>,
    documentMeta: Record<string, unknown>,
  ): void;
  /** Undo: pop from undo stack, push current to redo, return the snapshot. */
  undo(
    currentPageStates: Map<string, PageState>,
    currentMeta: Record<string, unknown>,
  ): Snapshot | null;
  /** Redo: pop from redo stack, push current to undo, return the snapshot. */
  redo(
    currentPageStates: Map<string, PageState>,
    currentMeta: Record<string, unknown>,
  ): Snapshot | null;
  /** Clear all history. */
  clear(): void;
}

function cloneSnapshot(
  pageStates: Map<string, PageState>,
  documentMeta: Record<string, unknown>,
): Snapshot {
  const clonedPages = new Map<string, PageState>();
  for (const [key, val] of pageStates) {
    clonedPages.set(key, JSON.parse(JSON.stringify(val)) as PageState);
  }
  return {
    pageStates: clonedPages,
    documentMeta: JSON.parse(JSON.stringify(documentMeta)) as Record<
      string,
      unknown
    >,
  };
}

export const useHistoryStore = create<HistoryStoreState & HistoryStoreActions>(
  (set, get) => ({
    undoStack: [],
    redoStack: [],
    maxSize: 200,

    pushSnapshot(pageStates, documentMeta) {
      const snapshot = cloneSnapshot(pageStates, documentMeta);
      set((s) => {
        const stack = [...s.undoStack, snapshot];
        if (stack.length > s.maxSize) stack.shift();
        return { undoStack: stack, redoStack: [] };
      });
    },

    undo(currentPageStates, currentMeta) {
      const { undoStack } = get();
      if (undoStack.length === 0) return null;
      const snapshot = undoStack[undoStack.length - 1]!;
      const currentSnapshot = cloneSnapshot(currentPageStates, currentMeta);
      set((s) => ({
        undoStack: s.undoStack.slice(0, -1),
        redoStack: [...s.redoStack, currentSnapshot],
      }));
      return snapshot;
    },

    redo(currentPageStates, currentMeta) {
      const { redoStack } = get();
      if (redoStack.length === 0) return null;
      const snapshot = redoStack[redoStack.length - 1]!;
      const currentSnapshot = cloneSnapshot(currentPageStates, currentMeta);
      set((s) => ({
        redoStack: s.redoStack.slice(0, -1),
        undoStack: [...s.undoStack, currentSnapshot],
      }));
      return snapshot;
    },

    clear() {
      set({ undoStack: [], redoStack: [] });
    },
  }),
);
