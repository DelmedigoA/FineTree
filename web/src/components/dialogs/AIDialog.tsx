/** AI inference dialog — all actions: Extract, Fill, Fix Spelling, Detect, Align, Batch. */

import { useState, useRef, useEffect } from "react";
import { useAIStore } from "../../stores/aiStore";
import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { useSSEStream } from "../../hooks/useSSEStream";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import { post } from "../../api/client";
import type { BoxRecord } from "../../types/schema";

type ActionTab = "extract" | "fill" | "fix" | "detect" | "align" | "batch";

const GEMINI_MODELS = [
  { id: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
  { id: "gemini-2.0-flash-thinking-exp", label: "Gemini 2.0 Flash Thinking" },
  { id: "gemini-2.5-pro-preview-03-25", label: "Gemini 2.5 Pro" },
  { id: "gemini-1.5-pro", label: "Gemini 1.5 Pro" },
  { id: "gemini-1.5-flash", label: "Gemini 1.5 Flash" },
];

const QWEN_MODELS = [
  { id: "Qwen/Qwen2-VL-72B-Instruct", label: "Qwen2-VL 72B" },
  { id: "Qwen/Qwen2-VL-7B-Instruct", label: "Qwen2-VL 7B" },
  { id: "Qwen/Qwen2.5-VL-72B-Instruct", label: "Qwen2.5-VL 72B" },
  { id: "Qwen/QVQ-72B-Preview", label: "QVQ 72B" },
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

const EMPTY_FACT: BoxRecord["fact"] = {
  value: "", fact_num: null, equations: null, natural_sign: null,
  row_role: "detail", comment_ref: null, note_flag: false, note_name: null,
  note_num: null, note_ref: null, date: null, period_type: null,
  period_start: null, period_end: null, duration_type: null,
  recurring_period: null, path: [], path_source: null,
  currency: null, scale: null, value_type: null, value_context: null,
};

export function AIDialog() {
  const { dialogOpen, closeDialog, activeProvider, setProvider } = useAIStore();
  const { docId, pageNames, currentPageIndex } = useDocumentStore();
  const stream = useSSEStream();

  // Extract options
  const [action, setAction] = useState<string>("gt");
  const [enableThinking, setEnableThinking] = useState(false);
  const [geminiModel, setGeminiModel] = useState(GEMINI_MODELS[0]!.id);
  const [qwenModel, setQwenModel] = useState(QWEN_MODELS[0]!.id);

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

  // Sync tab when dialog opens with a specific initialTab.
  useEffect(() => {
    if (dialogOpen && initialTab) setTab(initialTab as ActionTab);
  }, [dialogOpen, initialTab]);

  if (!dialogOpen) return null;

  const currentPage = pageNames[currentPageIndex] ?? "";
  const modelConfig =
    activeProvider === "gemini"
      ? { model: geminiModel, enable_thinking: enableThinking }
      : { model_id: qwenModel, enable_thinking: enableThinking };

  // ── helpers ───────────────────────────────────────────────────────

  const applyExtractedFacts = (jsonText: string, isGT: boolean): number => {
    const { pageStates, updatePageState } = useDocumentStore.getState();
    const pageName = pageNames[currentPageIndex];
    if (!pageName) return 0;
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

    updatePageState(pageName, {
      ...page,
      facts: isGT ? newFacts : [...page.facts, ...newFacts],
    });
    useCanvasStore.getState().markDirty("bbox");
    return newFacts.length;
  };

  // ── actions ───────────────────────────────────────────────────────

  const handleExtract = () => {
    if (!docId || !currentPage) return;
    setApplyStatus(null);
    accumulatedRef.current = "";
    stream.start(
      "/ai/extract",
      { doc_id: docId, page_name: currentPage, provider: activeProvider, action, config: modelConfig },
      (ev) => {
        const obj = ev.data as Record<string, unknown>;
        if (obj?.type === "chunk" && typeof obj.text === "string") {
          accumulatedRef.current += obj.text;
        } else if (obj?.type === "done") {
          const n = applyExtractedFacts(accumulatedRef.current, action === "gt");
          setApplyStatus(n > 0 ? `✓ Applied ${n} facts` : "No facts parsed");
        }
      },
    );
  };

  const handleFill = async () => {
    if (!docId || !currentPage) return;
    const { pageStates, updatePageState } = useDocumentStore.getState();
    const page = pageStates.get(currentPage);
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
          doc_id: docId,
          page_name: currentPage,
          provider: activeProvider,
          facts: targetFacts.map((f) => f!.fact),
          fields: [...fillFields],
          config: modelConfig,
        },
      );

      if (!result.patch?.length) { setApplyStatus("No patch returned"); return; }
      pushUndoSnapshot();

      const newFacts = [...page.facts];
      const targetIndices = selectedIds.size > 0
        ? [...selectedIds]
        : page.facts.map((_, i) => i);

      result.patch.forEach((patch, pi) => {
        const idx = targetIndices[pi];
        if (idx == null) return;
        const f = newFacts[idx];
        if (!f) return;
        newFacts[idx] = { ...f, fact: { ...f.fact, ...patch } };
      });

      updatePageState(currentPage, { ...page, facts: newFacts });
      useCanvasStore.getState().markDirty("bbox");
      setApplyStatus(`✓ Filled ${result.patch.length} facts`);
    } catch (err) {
      setApplyStatus(`Error: ${String(err)}`);
    }
  };

  const handleFixSpelling = async () => {
    if (!docId || !currentPage) return;
    setApplyStatus(null);
    try {
      await post("/ai/fix-spelling", { doc_id: docId, page_name: currentPage });
      setApplyStatus("✓ Fix spelling request sent (reload to see changes)");
    } catch (err) {
      setApplyStatus(`Error: ${String(err)}`);
    }
  };

  const handleDetect = () => {
    if (!docId || !currentPage) return;
    setApplyStatus(null);
    accumulatedRef.current = "";
    let count = 0;
    stream.start(
      "/ai/detect-bbox",
      { doc_id: docId, page_name: currentPage, backend: doctrBackend },
      (ev) => {
        const obj = ev.data as Record<string, unknown>;
        if (obj?.type === "bbox" && obj.data) {
          const { pageStates, updatePageState } = useDocumentStore.getState();
          const page = pageStates.get(currentPage);
          if (!page) return;
          const bbox = obj.data as BoxRecord["bbox"];
          updatePageState(currentPage, {
            ...page,
            facts: [...page.facts, { bbox, fact: { ...EMPTY_FACT } }],
          });
          useCanvasStore.getState().markDirty("bbox");
          count++;
          setApplyStatus(`Detecting... ${count} bboxes`);
        } else if (obj?.type === "done") {
          setApplyStatus(`✓ Detected ${count} bboxes`);
        }
      },
    );
  };

  const handleAlign = async () => {
    if (!docId || !currentPage) return;
    setApplyStatus(null);
    const { pageStates, updatePageState } = useDocumentStore.getState();
    const page = pageStates.get(currentPage);
    if (!page || page.facts.length === 0) { setApplyStatus("No facts to align"); return; }

    try {
      const result = await post<{ aligned_facts: BoxRecord[] }>(
        "/ai/align-bboxes",
        { doc_id: docId, page_name: currentPage, facts: page.facts.map((f) => f.fact) },
      );
      if (!result.aligned_facts?.length) { setApplyStatus("No aligned facts returned"); return; }
      pushUndoSnapshot();
      updatePageState(currentPage, { ...page, facts: result.aligned_facts });
      useCanvasStore.getState().markDirty("bbox");
      setApplyStatus(`✓ Aligned ${result.aligned_facts.length} facts`);
    } catch (err) {
      setApplyStatus(`Error: ${String(err)}`);
    }
  };

  const handleBatch = async () => {
    if (!docId) return;
    setBatchRunning(true);
    setBatchStatus(null);

    const from = Math.max(1, batchFrom) - 1;
    const to = Math.min(pageNames.length, batchTo);
    const pages = pageNames.slice(from, to);

    let done = 0;
    for (const pageName of pages) {
      setBatchStatus(`Processing ${pageName} (${done + 1}/${pages.length})…`);
      try {
        await new Promise<void>((resolve) => {
          let acc = "";
          useSSEStreamRef.current?.cancel();
          stream.start(
            "/ai/extract",
            { doc_id: docId, page_name: pageName, provider: activeProvider, action: batchAction, config: modelConfig },
            (ev) => {
              const obj = ev.data as Record<string, unknown>;
              if (obj?.type === "chunk" && typeof obj.text === "string") acc += obj.text;
              else if (obj?.type === "done" || obj?.type === "error" || obj?.type === "cancelled") {
                if (obj.type === "done") applyExtractedFacts(acc, batchAction === "gt");
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

  const useSSEStreamRef = useRef(stream);
  useSSEStreamRef.current = stream;

  // ── UI ────────────────────────────────────────────────────────────

  const TABS: { id: ActionTab; label: string }[] = [
    { id: "extract", label: "Extract" },
    { id: "fill", label: "Fill Fields" },
    { id: "fix", label: "Fix Spelling" },
    { id: "detect", label: "Detect BBoxes" },
    { id: "align", label: "Align BBoxes" },
    { id: "batch", label: "Batch" },
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

          {/* Provider + model row */}
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            {(["gemini", "qwen"] as const).map((p) => (
              <ProviderPill key={p} active={activeProvider === p} onClick={() => setProvider(p)}>
                {p === "gemini" ? "Gemini" : "Qwen"}
              </ProviderPill>
            ))}
            <select
              value={activeProvider === "gemini" ? geminiModel : qwenModel}
              onChange={(e) => activeProvider === "gemini" ? setGeminiModel(e.target.value) : setQwenModel(e.target.value)}
              style={selectStyle}
            >
              {(activeProvider === "gemini" ? GEMINI_MODELS : QWEN_MODELS).map((m) => (
                <option key={m.id} value={m.id}>{m.label}</option>
              ))}
            </select>
          </div>

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
              <Row label="Action">
                <select value={action} onChange={(e) => setAction(e.target.value)} style={selectStyle}>
                  <option value="gt">Ground Truth (replace all)</option>
                  <option value="autocomplete">Autocomplete (append)</option>
                </select>
              </Row>
              <label style={checkLabel}>
                <input type="checkbox" checked={enableThinking} onChange={(e) => setEnableThinking(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
                Enable extended thinking
              </label>
              <div style={{ display: "flex", gap: 8, marginTop: 4 }}>
                <ActionButton onClick={handleExtract} disabled={stream.isStreaming} primary>
                  {stream.isStreaming ? "Extracting…" : "Extract"}
                </ActionButton>
                {stream.isStreaming && <ActionButton onClick={stream.cancel} danger>Cancel</ActionButton>}
              </div>
            </div>
          )}

          {tab === "fill" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Fields to fill (applies to selected facts, or all if none selected):
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
              <label style={checkLabel}>
                <input type="checkbox" checked={enableThinking} onChange={(e) => setEnableThinking(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
                Enable extended thinking
              </label>
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

          {tab === "align" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <p style={{ fontSize: 13, color: "var(--text-muted)", lineHeight: 1.5 }}>
                Re-align existing fact bboxes using the Qwen import matcher. Matches visual regions to annotated facts.
              </p>
              <ActionButton onClick={handleAlign} primary>Align BBoxes</ActionButton>
            </div>
          )}

          {tab === "batch" && (
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
                Run extraction across a range of pages sequentially.
              </p>
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
              <label style={checkLabel}>
                <input type="checkbox" checked={enableThinking} onChange={(e) => setEnableThinking(e.target.checked)} style={{ accentColor: "var(--accent)" }} />
                Enable extended thinking
              </label>
              <div style={{ display: "flex", gap: 8 }}>
                <ActionButton onClick={handleBatch} disabled={batchRunning} primary>
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

          {/* Status */}
          {applyStatus && tab !== "batch" && (
            <div style={{ fontSize: 12, fontWeight: 600, color: applyStatus.startsWith("✓") ? "var(--ok)" : "var(--warn)" }}>
              {applyStatus}
            </div>
          )}

          {/* Stream output */}
          {(stream.fullText || stream.error) && tab !== "batch" && (
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

function ProviderPill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "5px 14px", fontSize: 13, fontWeight: 600,
        borderRadius: "var(--radius-pill)",
        border: active ? "1px solid var(--accent)" : "1px solid var(--surface-border)",
        background: active ? "var(--accent-soft)" : "transparent",
        color: active ? "var(--accent)" : "var(--text-muted)",
        cursor: "pointer", transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
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
