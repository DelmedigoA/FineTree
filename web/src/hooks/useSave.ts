/** Save hook — Ctrl+S to save document to backend. */

import { useEffect, useCallback } from "react";
import { useDocumentStore } from "../stores/documentStore";
import { useUIStore } from "../stores/uiStore";
import { put } from "../api/client";

export function useSave() {
  const save = useCallback(async () => {
    const { docId, pageStates, documentMeta, isDirty, markSaved } =
      useDocumentStore.getState();
    if (!docId || !isDirty) return;

    const { setSaveStatus } = useUIStore.getState();

    // Convert Map to plain object for JSON serialization.
    const pageStatesObj: Record<string, unknown> = {};
    for (const [key, val] of pageStates) {
      pageStatesObj[key] = val;
    }

    setSaveStatus("saving");
    try {
      await put(`/annotations/${docId}`, {
        page_states: pageStatesObj,
        document_meta: documentMeta,
      });
      markSaved();
      setSaveStatus("saved");
      // Fade back to idle after 3 s
      setTimeout(() => {
        if (useUIStore.getState().saveStatus === "saved") {
          setSaveStatus("idle");
        }
      }, 3000);
    } catch (err) {
      console.error("Save failed:", err);
      setSaveStatus("error");
      setTimeout(() => {
        if (useUIStore.getState().saveStatus === "error") {
          setSaveStatus("idle");
        }
      }, 5000);
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
