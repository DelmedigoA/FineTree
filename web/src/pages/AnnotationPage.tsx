/** Annotation page: canvas + toolbar + thumbnail strip + inspector. */

import { useEffect } from "react";
import { useParams, useSearchParams } from "react-router-dom";
import { get } from "../api/client";
import { useDocumentStore } from "../stores/documentStore";
import { useCanvasStore } from "../stores/canvasStore";
import { CanvasContainer } from "../canvas/CanvasContainer";
import { AnnotationToolbar } from "../components/toolbar/AnnotationToolbar";
import { PageThumbnailStrip } from "../components/thumbnail/PageThumbnailStrip";
import { InspectorPanel } from "../components/inspector/InspectorPanel";
import { useSave } from "../hooks/useSave";
import { useUndoRedo } from "../hooks/useUndoRedo";
import { useAIStore } from "../stores/aiStore";
import { useSelectionStore } from "../stores/selectionStore";
import { useUIStore } from "../stores/uiStore";
import { AIDialog } from "../components/dialogs/AIDialog";
import type { DocumentPayload } from "../types/api";
import { recordView } from "./DashboardPage";

export function AnnotationPage() {
  const { docId } = useParams<{ docId: string }>();
  const [searchParams] = useSearchParams();
  const loadDocument = useDocumentStore((s) => s.loadDocument);
  const beginLoadingDocument = useDocumentStore((s) => s.beginLoadingDocument);
  const setCurrentPageIndex = useDocumentStore((s) => s.setCurrentPageIndex);
  const currentPageIndex = useDocumentStore((s) => s.currentPageIndex);
  const isLoading = useDocumentStore((s) => s.isLoading);
  const fitToView = useCanvasStore((s) => s.fitToView);
  const clearPageSelection = useSelectionStore((s) => s.clearPageSelection);

  const { save } = useSave();
  useUndoRedo();
  const openDialog = useAIStore((s) => s.openDialog);
  const openAI = () => openDialog();

  useEffect(() => {
    if (!docId) return;
    const abort = new AbortController();
    const initialPage = parseInt(searchParams.get("page") ?? "0", 10);
    beginLoadingDocument(docId);
    get<DocumentPayload>(`/annotations/${docId}`, { signal: abort.signal })
      .then((payload) => {
        if (abort.signal.aborted) return;
        recordView(docId);
        loadDocument(docId, {
          raw_payload: {},
          page_states: payload.page_states,
          document_meta: payload.document_meta,
          page_names: payload.page_images,
        });
        if (!isNaN(initialPage) && initialPage > 0) {
          setCurrentPageIndex(initialPage);
        }
        requestAnimationFrame(() => {
          requestAnimationFrame(fitToView);
        });
      })
      .catch((error) => {
        if ((error as Error).name === "AbortError") return;
        console.error(error);
      });
    return () => abort.abort();
  }, [docId, beginLoadingDocument, loadDocument, setCurrentPageIndex, fitToView, searchParams]);

  useEffect(() => {
    clearPageSelection();
  }, [clearPageSelection, docId, currentPageIndex]);

  if (isLoading) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100%",
          color: "var(--text-soft)",
          fontSize: 14,
        }}
      >
        Loading document...
      </div>
    );
  }

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        height: "100%",
        overflow: "hidden",
      }}
    >
      <AnnotationToolbar
        onSave={save}
        onAI={openAI}
      />
      <AIDialog />
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <PageThumbnailStrip />
        <div style={{ flex: 1, overflow: "hidden" }}>
          <CanvasContainer />
        </div>
        <InspectorPanel />
      </div>
      <StatusBar />
    </div>
  );
}

function StatusBar() {
  const saveStatus = useUIStore((s) => s.saveStatus);
  const savedAt = useUIStore((s) => s.savedAt);
  const aiStatus = useUIStore((s) => s.aiStatus);
  const aiMessage = useUIStore((s) => s.aiMessage);

  const label =
    aiStatus !== "idle" ? aiMessage
    : saveStatus === "saving" ? "Saving…"
    : saveStatus === "saved" ? `Saved ✓${savedAt ? "  " + new Date(savedAt).toLocaleTimeString() : ""}`
    : saveStatus === "error" ? "Save failed ✗"
    : null;

  const color =
    aiStatus === "running" ? "var(--text-muted)"
    : aiStatus === "success" ? "var(--accent)"
    : aiStatus === "error" ? "var(--warn)"
    : saveStatus === "saving" ? "var(--text-muted)"
    : saveStatus === "saved" ? "var(--accent)"
    : saveStatus === "error" ? "var(--warn)"
    : "transparent";

  return (
    <div
      style={{
        height: 24,
        padding: "0 16px",
        display: "flex",
        alignItems: "center",
        justifyContent: "flex-end",
        background: "var(--surface)",
        borderTop: "1px solid var(--surface-border)",
        flexShrink: 0,
      }}
    >
      <span
        style={{
          fontSize: 11,
          fontWeight: 500,
          color,
          fontFamily: "var(--font-mono)",
          transition: "opacity 0.3s ease",
          opacity: label ? 1 : 0,
        }}
      >
        {label ?? ""}
      </span>
    </div>
  );
}
