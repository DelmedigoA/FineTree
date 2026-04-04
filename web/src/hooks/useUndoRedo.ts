/** Undo/Redo hook — Ctrl+Z / Ctrl+Shift+Z. */

import { useEffect, useCallback } from "react";
import { useDocumentStore } from "../stores/documentStore";
import { useHistoryStore } from "../stores/historyStore";
import { useCanvasStore } from "../stores/canvasStore";
import type { PageState } from "../types/schema";

export function useUndoRedo() {
  const undo = useCallback(() => {
    const doc = useDocumentStore.getState();
    const snapshot = useHistoryStore
      .getState()
      .undo(doc.pageStates, doc.documentMeta);
    if (!snapshot) return;
    applySnapshot(snapshot.pageStates, snapshot.documentMeta);
  }, []);

  const redo = useCallback(() => {
    const doc = useDocumentStore.getState();
    const snapshot = useHistoryStore
      .getState()
      .redo(doc.pageStates, doc.documentMeta);
    if (!snapshot) return;
    applySnapshot(snapshot.pageStates, snapshot.documentMeta);
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "z") {
        e.preventDefault();
        if (e.shiftKey) {
          redo();
        } else {
          undo();
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [undo, redo]);

  return { undo, redo };
}

function applySnapshot(
  pageStates: Map<string, PageState>,
  documentMeta: Record<string, unknown>,
) {
  const store = useDocumentStore.getState();
  // Apply all page states from snapshot.
  for (const [key, val] of pageStates) {
    store.updatePageState(key, val);
  }
  store.updateDocumentMeta(documentMeta);
  useCanvasStore.getState().markDirty("all");
}

/** Call this before any edit to push a snapshot onto the undo stack. */
export function pushUndoSnapshot() {
  const { pageStates, documentMeta } = useDocumentStore.getState();
  useHistoryStore.getState().pushSnapshot(pageStates, documentMeta);
}
