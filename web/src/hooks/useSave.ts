/** Save hook — Ctrl+S to save document to backend. */

import { useEffect, useCallback } from "react";
import { useDocumentStore } from "../stores/documentStore";
import { put } from "../api/client";

export function useSave() {
  const save = useCallback(async () => {
    const { docId, pageStates, documentMeta, isDirty, markSaved } =
      useDocumentStore.getState();
    if (!docId || !isDirty) return;

    // Convert Map to plain object for JSON serialization.
    const pageStatesObj: Record<string, unknown> = {};
    for (const [key, val] of pageStates) {
      pageStatesObj[key] = val;
    }

    try {
      await put(`/annotations/${docId}`, {
        page_states: pageStatesObj,
        document_meta: documentMeta,
      });
      markSaved();
    } catch (err) {
      console.error("Save failed:", err);
    }
  }, []);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "s") {
        e.preventDefault();
        save();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [save]);

  return { save };
}
