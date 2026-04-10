/** AI inference dialog — extract, fill, detect, batch pages, and infer PDFs. */

import { useState, useRef, useEffect } from "react";
import { useAIStore } from "../../stores/aiStore";
import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { useSSEStream } from "../../hooks/useSSEStream";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import { post } from "../../api/client";
import type { BoxRecord } from "../../types/schema";
import { BatchInferSection } from "./BatchInferSection";

type ActionTab = "extract" | "fill" | "fix" | "detect" | "batch" | "infer";

const GEMINI_MODELS = [
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview" },
  { id: "gemini-pro", label: "Gemini Pro" },
];

type ExtractEngine = "gemini-3-flash-preview" | "gemini-pro" | "custom-endpoint";

const EXTRACT_ENGINES: { id: ExtractEngine; label: string }[] = [
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview" },
  { id: "gemini-pro", label: "Gemini Pro" },
  { id: "custom-endpoint", label: "Custom Endpoint" },
];

const FILL_FIELDS = [
  "value", "currency", "scale", "date", "period_type",
  "period_start", "period_end", "path", "value_type",
  "value_context", "natural_sign", "row_role",
];

const DOCTR_BACKENDS = [
  { id: "merged", label: "Merged (DocTR + heuristic)" },
  { id: "doctr", label: "DocTR only" },
];

const DEFAULT_CUSTOM_ENDPOINT_URL = "https://your-endpoint.ngrok-free.app/v1";
const DEFAULT_CUSTOM_ENDPOINT_MODEL = "asafd60/FineTree-27B-v2.9-merged";

const EMPTY_FACT: BoxRecord["fact"] = {
  value: "", fact_num: null, equations: null, natural_sign: null,
  row_role: "detail", comment_ref: null, note_flag: false, note_name: null,
  note_num: null, note_ref: null, date: null, period_type: null,
  period_start: null, period_end: null, duration_type: null,
  recurring_period: null, path: [], path_source: null,
  currency: null, scale: null, value_type: null, value_context: null,
};

export function AIDialog() {
  const { dialogOpen, closeDialog } = useAIStore();
  const { docId, pageNames, currentPageIndex } = useDocumentStore();
  const stream = useSSEStream();

  // Extract options
  const [action, setAction] = useState<string>("gt");
  const [extractEngine, setExtractEngine] = useState<ExtractEngine>("gemini-3-flash-preview");
  const [fillGeminiModel, setFillGeminiModel] = useState(GEMINI_MODELS[0]!.id);
  const [customEndpointUrl, setCustomEndpointUrl] = useState(DEFAULT_CUSTOM_ENDPOINT_URL);
  const [customEndpointModel, setCustomEndpointModel] = useState(DEFAULT_CUSTOM_ENDPOINT_MODEL);

  // Fill options
  const [fillFields, setFillFields] = useState<Set<string>>(new Set(["value", "currency", "scale"]));

  // Detect options
  const [doctrBackend, setDoctrBackend] = useState("merged");

  // Batch options
  const [batchFrom, setBatchFrom] = useState(1);
  const [batchTo, setBatchTo] = useState(pageNames.length || 1);
  const [batchAction, setBatchAction] = useState("gt");
  const [batchStatus, setBatchStatus] = useState<string | null>(null);
  const [batchRunning, setBatchRunning] = useState(false);

  const { initialTab } = useAIStore();
  const [tab, setTab] = useState<ActionTab>((initialTab as ActionTab) ?? "extract");
  const [applyStatus, setApplyStatus] = useState<string | null>(null);
  const accumulatedRef = useRef("");
  // Must be declared unconditionally (before conditional return) to satisfy rules of hooks.
  const useSSEStreamRef = useRef(stream);

  // Sync tab when dialog opens with a specific initialTab.
  useEffect(() => {
    if (dialogOpen && initialTab) setTab(initialTab as ActionTab);
  }, [dialogOpen, initialTab]);

  useEffect(() => {
    stream.cancel();
    setBatchRunning(false);
    setBatchStatus(null);
  }, [docId, stream.cancel]);

  useEffect(() => () => stream.cancel(), [stream.cancel]);

  if (!dialogOpen) return null;

  const currentPage = pageNames[currentPageIndex] ?? "";
  const isCustomEndpoint = extractEngine === "custom-endpoint";
  const extractRequest =
    isCustomEndpoint
      ? {
          provider: "custom_endpoint" as const,
          config: {
            base_url: customEndpointUrl.trim(),
            model_id: customEndpointModel.trim(),
          },
        }
      : {
          provider: "gemini" as const,
          config: { model: extractEngine },
        };

  // ── helpers ───────────────────────────────────────────────────────

  const applyExtractedFacts = (
    jsonText: string,
    isGT: boolean,
    expectedDocId: string,
    pageName: string,
  ): number => {
    const { docId: currentDocId, pageStates, updatePageStateForDocument } = useDocumentStore.getState();
    if (currentDocId !== expectedDocId || !pageName) return 0;
    const page = pageStates.get(pageName);
    if (!page) return 0;

    const cleaned = jsonText
      .replace(/^```[a-z]*\n?/gm, "")
      .replace(/\n?```$/gm, "")
      .trim();

    let parsed: unknown;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      const match = cleaned.match(/\[[\s\S]*\]/);
      if (!match) return 0;
      try { parsed = JSON.parse(match[0]); } catch { return 0; }
    }

    const incoming = Array.isArray(parsed)
      ? parsed
      : (parsed as Record<string, unknown>)?.facts;
    if (!Array.isArray(incoming) || incoming.length === 0) return 0;

    pushUndoSnapshot();
    const newFacts: BoxRecord[] = incoming.map((item: Record<string, unknown>) => ({
      bbox: (item.bbox as BoxRecord["bbox"]) ?? { x: 0, y: 0, w: 50, h: 20 },
      fact: (item.fact as BoxRecord["fact"]) ?? (item as BoxRecord["fact"]),
    }));

    const updated = updatePageStateForDocument(expectedDocId, pageName, {
      ...page,
      facts: isGT ? newFacts : [...page.facts, ...newFacts],
    });
    if (!updated) return 0;
    useCanvasStore.getState().markDirty("bbox");
    return newFacts.length;
  };

  // ── actions ───────────────────────────────────────────────────────

  const handleExtract = () => {
    if (!docId || !currentPage) return;
    const activeDocId = docId;
    const activePage = currentPage;
    setApplyStatus(null);
    accumulatedRef.current = "";
    stream.start(
      "/ai/extract",
      {
        doc_id: activeDocId,
        page_name: activePage,
        provider: extractRequest.provider,
        action,
        config: extractRequest.config,
      },
      (ev) => {
        const obj = ev.data as Record<string, unknown>;
        if (obj?.type === "chunk" && typeof obj.text === "string") {
          accumulatedRef.current += obj.text;
        } else if (obj?.type === "done") {
          const n = applyExtractedFacts(accumulatedRef.current, action === "gt", activeDocId, activePage);
          setApplyStatus(n > 0 ? `✓ Applied ${n} facts` : "No facts parsed");
        }
      },
    );
  };

  const handleFill = async () => {
    if (!docId || !currentPage) return;
    const activeDocId = docId;
    const activePage = currentPage;
    const { pageStates } = useDocumentStore.getState();
    const page = pageStates.get(activePage);
    if (!page) return;

    const selectedIds = useSelectionStore.getState().selectedIndices;
    const targetFacts = selectedIds.size > 0
      ? [...selectedIds].map((i) => page.facts[i]).filter(Boolean)
      : page.facts;

    if (targetFacts.length === 0) { setApplyStatus("No facts to fill"); return; }

    setApplyStatus(null);
    try {
      const result = await post<{ patch: Record<string, unknown>[] }>(
        "/ai/fill",
        {
          doc_id: activeDocId,
          page_name: activePage,
          provider: "gemini",
          facts: targetFacts.map((f) => f!.fact),
          fields: [...fillFields],
          config: { model: fillGeminiModel },
        },
      );

      if (!result.patch?.length) { setApplyStatus("No patch returned"); return; }
      const { docId: currentDocId, pageStates: currentStates, updatePageStateForDocument } = useDocumentStore.getState();
      if (currentDocId !== activeDocId) {
        setApplyStatus("Skipped stale fill result after document switch");
        return;
      }
      const latestPage = currentStates.get(activePage);
      if (!latestPage) {
        setApplyStatus("Skipped fill result: page no longer active");
        return;
      }
      pushUndoSnapshot();

      const newFacts = [...latestPage.facts];
      const targetIndices = selectedIds.size > 0
        ? [...selectedIds]
        : latestPage.facts.map((_, i) => i);

      result.patch.forEach((patch, pi) => {
        const idx = targetIndices[pi];
        if (idx == null) return;
        const f = newFacts[idx];
        if (!f) return;
        newFacts[idx] = { ...f, fact: { ...f.fact, ...patch } };
      });

      updatePageStateForDocument(activeDocId, activePage, { ...latestPage, facts: newFacts });
      useCanvasStore.getState().markDirty("bbox");
      setApplyStatus(`✓ Filled ${result.patch.length} facts`);
    } catch (err) {
      setApplyStatus(`Error: ${String(err)}`);
    }
  };

  const handleFixSpelling = async () => {
    if (!docId || !currentPage) return;
    const activeDocId = docId;
    const activePage = currentPage;
    setApplyStatus(null);
    try {
      await post("/ai/fix-spelling", { doc_id: activeDocId, page_name: activePage });
      setApplyStatus("✓ Fix spelling request sent (reload to see changes)");
    } catch (err) {
      setApplyStatus(`Error: ${String(err)}`);
    }
  };

  const handleDetect = () => {
    if (!docId || !currentPage) return;
    const activeDocId = docId;
    const activePage = currentPage;
    setApplyStatus(null);
    accumulatedRef.current = "";
    let count = 0;
    stream.start(
      "/ai/detect-bbox",
      { doc_id: activeDocId, page_name: activePage, backend: doctrBackend },
      (ev) => {
        const obj = ev.data as Record<string, unknown>;
        if (obj?.type === "bbox" && obj.data) {
          const { docId: currentDocId, pageStates, updatePageStateForDocument } = useDocumentStore.getState();
          if (currentDocId !== activeDocId) return;
          const page = pageStates.get(activePage);
          if (!page) return;
          const bbox = obj.data as BoxRecord["bbox"];
          const updated = updatePageStateForDocument(activeDocId, activePage, {
            ...page,
            facts: [...page.facts, { bbox, fact: { ...EMPTY_FACT } }],
          });
          if (!updated) return;
          useCanvasStore.getState().markDirty("bbox");
          count++;
          setApplyStatus(`Detecting... ${count} bboxes`);
        } else if (obj?.type === "done") {
          setApplyStatus(`✓ Detected ${count} bboxes`);
        }
      },
    );
  };

  const handleBatch = async () => {
    if (!docId) return;
    const activeDocId = docId;
    setBatchRunning(true);
    setBatchStatus(null);

    const from = Math.max(1, batchFrom) - 1;
    const to = Math.min(pageNames.length, batchTo);
    const pages = pageNames.slice(from, to);

    let done = 0;
    for (const pageName of pages) {
      if (useDocumentStore.getState().docId !== activeDocId) {
        setBatchStatus("Batch stopped after document switch");
        break;
      }
      setBatchStatus(`Processing ${pageName} (${done + 1}/${pages.length})…`);
      try {
        await new Promise<void>((resolve) => {
          let acc = "";
          useSSEStreamRef.current?.cancel();
          stream.start(
            "/ai/extract",
            {
              doc_id: activeDocId,
              page_name: pageName,
              provider: extractRequest.provider,
              action: batchAction,
              config: extractRequest.config,
            },
            (ev) => {
              const obj = ev.data as Record<string, unknown>;
              if (obj?.type === "chunk" && typeof obj.text === "string") acc += obj.text;
              else if (obj?.type === "done" || obj?.type === "error" || obj?.type === "cancelled") {
                if (obj.type === "done") applyExtractedFacts(acc, batchAction === "gt", activeDocId, pageName);
                resolve();
              }
            },
          );
        });
      } catch { /* continue on error */ }
      done++;
    }

    setBatchStatus(`✓ Batch complete — ${done}/${pages.length} pages processed`);
    setBatchRunning(false);
  };

  useSSEStreamRef.current = stream;

  // ── UI ────────────────────────────────────────────────────────────

  const TABS: { id: ActionTab; label: string }[] = [
    { id: "extract", label: "Extract" },
    { id: "fill", label: "Fill Fields" },
    { id: "fix", label: "Fix Spelling" },
    { id: "detect", label: "Detect BBoxes" },
    { id: "batch", label: "Batch Pages" },
    { id: "infer", label: "Infer PDF" },
  ];

  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 1000,
        display: "flex", alignItems: "center", justifyContent: "center",
        background: "rgba(0,0,0,0.6)", backdropFilter: "blur(4px)",
      }}
      onClick={(e) => { if (e.target === e.currentTarget) closeDialog(); }}
    >
      <div
        style={{
          width: 580, maxHeight: "88vh",
          background: "var(--surface-raised)",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--surface-border)",
          boxShadow: "0 24px 48px rgba(0,0,0,0.4)",
          display: "flex", flexDirection: "column", overflow: "hidden",
        }}
      >
        {/* Header */}
        <div style={{ padding: "16px 20px", borderBottom: "1px solid var(--surface-border)", display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <h2 style={{ fontFamily: "var(--font-heading)", fontSize: 16, fontWeight: 700 }}>AI Inference</h2>
          <button onClick={closeDialog} style={{ width: 28, height: 28, display: "flex", alignItems: "center", justifyContent: "center", borderRadius: "var(--radius-xs)", fontSize: 18, color: "var(--text-muted)", cursor: "pointer", background: "transparent" }}>×</button>
        </div>

        <div style={{ padding: "16px 20px", flex: 1, overflowY: "auto", display: "flex", flexDirection: "column", gap: 14 }}>

          {/* Action tabs */}
          <div style={{ display: "flex", gap: 4, borderBottom: "1px solid var(--surface-border)", paddingBottom: 12 }}>
            {TABS.map((t) => (
              <button
                key={t.id}
                onClick={() => { setTab(t.id); setApplyStatus(null); }}
                style={{
                  padding: "5px 12px",
                  fontSize: 12, fontWeight: 600,
                  borderRadius: "var(--radius-xs)",
                  background: tab === t.id ? "var(--accent)" : "transparent",
                  color: tab === t.id ? "#fff" : "var(--text-muted)",
                  border: tab === t.id ? "none" : "1px solid var(--surface-border)",
                  cursor: "pointer",
                  transition: "var(--transition-fast)",
                }}
              >
                {t.label}
              </button>
            ))}
          </div>

          {/* Tab body */}
          {tab === "extract" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <Row label="Model">
                <select value={extractEngine} onChange={(e) => setExtractEngine(e.target.value as ExtractEngine)} style={selectStyle}>
                  {EXTRACT_ENGINES.map((engine) => (
                    <option key={engine.id} value={engine.id}>{engine.label}</option>
                  ))}
                </select>
              </Row>
              {isCustomEndpoint && (
                <>
                  <Row label="Endpoint">
                    <input
                      value={customEndpointUrl}
                      onChange={(e) => setCustomEndpointUrl(e.target.value)}
                      placeholder="https://your-endpoint/v1"
                      style={selectStyle}
                    />
                  </Row>
                  <Row label="Model ID">
                    <input
                      value={customEndpointModel}
                      onChange={(e) => setCustomEndpointModel(e.target.value)}
                      placeholder="asafd60/FineTree-27B-v2.9-merged"
                      style={selectStyle}
                    />
                  </Row>
                </>
              )}
              <Row label="Action">
                <select value={action} onChange={(e) => setAction(e.target.value)} style={selectStyle}>
                  <option value="gt">Ground Truth (replace all)</option>
                  <option value="autocomplete">Autocomplete (append)</option>
                </select>
              </Row>
              <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                <ActionButton
                  onClick={handleExtract}
                  disabled={stream.isStreaming || (isCustomEndpoint && (!customEndpointUrl.trim() || !customEndpointModel.trim()))}
                  primary
                >
                  {stream.isStreaming ? "Extracting…" : "Extract"}
                </ActionButton>
                {stream.isStreaming && <ActionButton onClick={stream.cancel} danger>Cancel</ActionButton>}
              </div>
            </div>
          )}

          {tab === "fill" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <Row label="Gemini">
                <select value={fillGeminiModel} onChange={(e) => setFillGeminiModel(e.target.value)} style={selectStyle}>
                  {GEMINI_MODELS.map((model) => (
                    <option key={model.id} value={model.id}>{model.label}</option>
                  ))}
                </select>
              </Row>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Fields to fill (applies to selected facts, or all if none selected):
              </div>
              <div style={{ display: "flex", gap: 8 }}>
                <ActionButton onClick={() => setFillFields(new Set(FILL_FIELDS))}>
                  Select All
                </ActionButton>
                <ActionButton onClick={() => setFillFields(new Set())}>
                  Deselect All
                </ActionButton>
              </div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: 6 }}>
                {FILL_FIELDS.map((f) => (
                  <label key={f} style={{ ...checkLabel, fontSize: 12, gap: 4 }}>
                    <input
                      type="checkbox"
                      checked={fillFields.has(f)}
                      onChange={(e) => {
                        const next = new Set(fillFields);
                        e.target.checked ? next.add(f) : next.delete(f);
                        setFillFields(next);
                      }}
                      style={{ accentColor: "var(--accent)" }}
                    />
                    {f}
                  </label>
                ))}
              </div>
              <ActionButton onClick={handleFill} disabled={stream.isStreaming || fillFields.size === 0} primary>
                Fill Fields
              </ActionButton>
            </div>
          )}

          {tab === "fix" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.5 }}>
                Correct Hebrew spelling in fact values on the current page using Gemini vision.
              </p>
              <ActionButton onClick={handleFixSpelling} primary>Fix Spelling</ActionButton>
            </div>
          )}

          {tab === "detect" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <Row label="Backend">
                <select value={doctrBackend} onChange={(e) => setDoctrBackend(e.target.value)} style={selectStyle}>
                  {DOCTR_BACKENDS.map((b) => (
                    <option key={b.id} value={b.id}>{b.label}</option>
                  ))}
                </select>
              </Row>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Detected bboxes are added to the current page in real-time.
              </p>
              <div style={{ display: "flex", gap: 8 }}>
                <ActionButton onClick={handleDetect} disabled={stream.isStreaming} primary>
                  {stream.isStreaming ? "Detecting…" : "Detect BBoxes"}
                </ActionButton>
                {stream.isStreaming && <ActionButton onClick={stream.cancel} danger>Cancel</ActionButton>}
              </div>
            </div>
          )}

          {tab === "batch" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Run extraction across a range of pages sequentially.
              </p>
              <Row label="Model">
                <select value={extractEngine} onChange={(e) => setExtractEngine(e.target.value as ExtractEngine)} style={selectStyle}>
                  {EXTRACT_ENGINES.map((engine) => (
                    <option key={engine.id} value={engine.id}>{engine.label}</option>
                  ))}
                </select>
              </Row>
              {isCustomEndpoint && (
                <>
                  <Row label="Endpoint">
                    <input
                      value={customEndpointUrl}
                      onChange={(e) => setCustomEndpointUrl(e.target.value)}
                      placeholder="https://your-endpoint/v1"
                      style={selectStyle}
                    />
                  </Row>
                  <Row label="Model ID">
                    <input
                      value={customEndpointModel}
                      onChange={(e) => setCustomEndpointModel(e.target.value)}
                      placeholder="asafd60/FineTree-27B-v2.9-merged"
                      style={selectStyle}
                    />
                  </Row>
                </>
              )}
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <Row label="Pages">
                  <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                    <input
                      type="number" min={1} max={pageNames.length}
                      value={batchFrom}
                      onChange={(e) => setBatchFrom(Number(e.target.value))}
                      style={{ ...selectStyle, width: 60 }}
                    />
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>to</span>
                    <input
                      type="number" min={1} max={pageNames.length}
                      value={batchTo}
                      onChange={(e) => setBatchTo(Number(e.target.value))}
                      style={{ ...selectStyle, width: 60 }}
                    />
                    <span style={{ color: "var(--text-muted)", fontSize: 12 }}>of {pageNames.length}</span>
                  </div>
                </Row>
              </div>
              <Row label="Action">
                <select value={batchAction} onChange={(e) => setBatchAction(e.target.value)} style={selectStyle}>
                  <option value="gt">Ground Truth (replace)</option>
                  <option value="autocomplete">Autocomplete (append)</option>
                </select>
              </Row>
              <div style={{ display: "flex", gap: 8 }}>
                <ActionButton
                  onClick={handleBatch}
                  disabled={batchRunning || (isCustomEndpoint && (!customEndpointUrl.trim() || !customEndpointModel.trim()))}
                  primary
                >
                  {batchRunning ? "Running batch…" : `Run Batch (${Math.min(pageNames.length, batchTo) - (batchFrom - 1)} pages)`}
                </ActionButton>
                {batchRunning && <ActionButton onClick={() => { stream.cancel(); setBatchRunning(false); }} danger>Stop</ActionButton>}
              </div>
              {batchStatus && (
                <div style={{ fontSize: 12, fontWeight: 600, color: batchStatus.startsWith("✓") ? "var(--ok)" : "var(--accent)" }}>
                  {batchStatus}
                </div>
              )}
            </div>
          )}

          {tab === "infer" && (
            <BatchInferSection docIds={docId ? [docId] : []} />
          )}

          {/* Status */}
          {applyStatus && tab !== "batch" && tab !== "infer" && (
            <div style={{ fontSize: 12, fontWeight: 600, color: applyStatus.startsWith("✓") ? "var(--ok)" : "var(--warn)" }}>
              {applyStatus}
            </div>
          )}

          {/* Stream output */}
          {(stream.fullText || stream.error) && tab !== "batch" && tab !== "infer" && (
            <div style={{ background: "var(--surface-alt)", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-sm)", padding: 12, maxHeight: 260, overflowY: "auto" }}>
              {stream.error ? (
                <div style={{ color: "var(--danger)", fontSize: 13 }}>Error: {stream.error}</div>
              ) : (
                <pre style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", whiteSpace: "pre-wrap", wordBreak: "break-word", margin: 0 }}>
                  {stream.fullText}
                </pre>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// ── small helpers ─────────────────────────────────────────────────

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13, color: "var(--text-muted)" }}>
      <span style={{ minWidth: 60, fontWeight: 500 }}>{label}</span>
      {children}
    </label>
  );
}

function ActionButton({ children, onClick, disabled, primary, danger }: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 20px", fontSize: 13, fontWeight: 600,
        borderRadius: "var(--radius-sm)",
        background: danger ? "var(--danger)" : primary ? "var(--accent)" : "transparent",
        color: danger || primary ? "#fff" : "var(--text-muted)",
        border: danger || primary ? "none" : "1px solid var(--surface-border)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
  );
}

const selectStyle: React.CSSProperties = {
  flex: 1, padding: "6px 10px", fontSize: 13,
  background: "var(--surface-alt)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-xs)",
  color: "var(--text)", outline: "none",
};

const checkLabel: React.CSSProperties = {
  display: "flex", alignItems: "center", gap: 8,
  fontSize: 13, color: "var(--text-muted)", cursor: "pointer",
};
