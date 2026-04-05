/** Annotation page: canvas + toolbar + thumbnail strip + inspector. */

import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { get } from "../api/client";
import { useDocumentStore } from "../stores/documentStore";
import { useCanvasStore } from "../stores/canvasStore";
import { CanvasContainer } from "../canvas/CanvasContainer";
import { AnnotationToolbar } from "../components/toolbar/AnnotationToolbar";
import { PageThumbnailStrip } from "../components/thumbnail/PageThumbnailStrip";
import { InspectorPanel } from "../components/inspector/InspectorPanel";
import { BatchInferDialog } from "../components/dialogs/BatchInferDialog";
import { useSave } from "../hooks/useSave";
import { useUndoRedo } from "../hooks/useUndoRedo";
import { useAIStore } from "../stores/aiStore";
import { AIDialog } from "../components/dialogs/AIDialog";
import type { DocumentPayload } from "../types/api";

export function AnnotationPage() {
  const { docId } = useParams<{ docId: string }>();
  const loadDocument = useDocumentStore((s) => s.loadDocument);
  const isLoading = useDocumentStore((s) => s.isLoading);
  const currentDocId = useDocumentStore((s) => s.docId);
  const fitToView = useCanvasStore((s) => s.fitToView);

  const { save } = useSave();
  useUndoRedo();
  const openDialog = useAIStore((s) => s.openDialog);
  const openAI = () => openDialog();
  const openBatch = () => openDialog("batch");
  const [batchInferOpen, setBatchInferOpen] = useState(false);

  useEffect(() => {
    if (!docId || docId === currentDocId) return;
    get<DocumentPayload>(`/annotations/${docId}`)
      .then((payload) => {
        loadDocument(docId, {
          raw_payload: {},
          page_states: payload.page_states,
          document_meta: payload.document_meta,
          page_names: payload.page_images,
        });
        requestAnimationFrame(() => {
          requestAnimationFrame(fitToView);
        });
      })
      .catch(console.error);
  }, [docId, currentDocId, loadDocument, fitToView]);

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
        onBatch={openBatch}
        onBatchInfer={() => setBatchInferOpen(true)}
      />
      <AIDialog />
      {batchInferOpen && docId && (
        <BatchInferDialog
          docIds={[docId]}
          onClose={() => setBatchInferOpen(false)}
        />
      )}
      <div style={{ display: "flex", flex: 1, overflow: "hidden" }}>
        <PageThumbnailStrip />
        <div style={{ flex: 1, overflow: "hidden" }}>
          <CanvasContainer />
        </div>
        <InspectorPanel />
      </div>
    </div>
  );
}
