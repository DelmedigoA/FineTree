/** AI inference dialog — provider/action selection + streaming output. */

import { useState, useRef } from "react";
import { useAIStore } from "../../stores/aiStore";
import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { useSSEStream } from "../../hooks/useSSEStream";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import type { BoxRecord } from "../../types/schema";

export function AIDialog() {
  const { dialogOpen, closeDialog, activeProvider, setProvider } = useAIStore();
  const { docId, pageNames, currentPageIndex } = useDocumentStore();
  const stream = useSSEStream();
  const [applyStatus, setApplyStatus] = useState<string | null>(null);
  const accumulatedRef = useRef("");

  const [action, setAction] = useState<string>("gt");
  const [enableThinking, setEnableThinking] = useState(false);

  if (!dialogOpen) return null;

  const currentPage = pageNames[currentPageIndex] ?? "";

  const applyExtractedFacts = (jsonText: string, isGT: boolean) => {
    const { pageStates, updatePageState } = useDocumentStore.getState();
    const pageName = pageNames[currentPageIndex];
    if (!pageName) return 0;
    const page = pageStates.get(pageName);
    if (!page) return 0;

    // Strip markdown code fences if present.
    const cleaned = jsonText
      .replace(/^```[a-z]*\n?/m, "")
      .replace(/\n?```$/m, "")
      .trim();

    let parsed: unknown;
    try {
      parsed = JSON.parse(cleaned);
    } catch {
      // Try to extract JSON array from text.
      const match = cleaned.match(/\[[\s\S]*\]/);
      if (!match) return 0;
      try {
        parsed = JSON.parse(match[0]);
      } catch {
        return 0;
      }
    }

    const incoming = Array.isArray(parsed) ? parsed : (parsed as Record<string, unknown>)?.facts;
    if (!Array.isArray(incoming) || incoming.length === 0) return 0;

    pushUndoSnapshot();

    const newFacts: BoxRecord[] = incoming.map((item: Record<string, unknown>) => ({
      bbox: (item.bbox as BoxRecord["bbox"]) ?? { x: 0, y: 0, w: 50, h: 20 },
      fact: item.fact as BoxRecord["fact"] ?? item,
    }));

    const updatedFacts = isGT ? newFacts : [...page.facts, ...newFacts];
    updatePageState(pageName, { ...page, facts: updatedFacts });
    useCanvasStore.getState().markDirty("bbox");
    return newFacts.length;
  };

  const handleExtract = () => {
    if (!docId || !currentPage) return;
    setApplyStatus(null);
    accumulatedRef.current = "";
    stream.start(
      "/ai/extract",
      {
        doc_id: docId,
        page_name: currentPage,
        provider: activeProvider,
        action,
        config: { enable_thinking: enableThinking },
      },
      (ev) => {
        const obj = ev.data as Record<string, unknown>;
        if (obj?.type === "chunk" && typeof obj.text === "string") {
          accumulatedRef.current += obj.text;
        } else if (obj?.type === "done") {
          const count = applyExtractedFacts(accumulatedRef.current, action === "gt");
          setApplyStatus(count > 0 ? `Applied ${count} facts` : "No facts parsed from response");
        }
      },
    );
  };

  const handleDetect = () => {
    if (!docId || !currentPage) return;
    setApplyStatus(null);
    accumulatedRef.current = "";
    stream.start(
      "/ai/detect-bbox",
      { doc_id: docId, page_name: currentPage },
      (ev) => {
        const obj = ev.data as Record<string, unknown>;
        if (obj?.type === "bbox" && obj.data) {
          // Apply bbox immediately as it arrives.
          const { pageStates, updatePageState } = useDocumentStore.getState();
          const pageName = pageNames[currentPageIndex];
          if (!pageName) return;
          const page = pageStates.get(pageName);
          if (!page) return;
          const bboxData = obj.data as BoxRecord["bbox"];
          const newFact: BoxRecord = {
            bbox: bboxData,
            fact: { value: "", fact_num: null, equations: null, natural_sign: null, row_role: "detail", comment_ref: null, note_flag: false, note_name: null, note_num: null, note_ref: null, date: null, period_type: null, period_start: null, period_end: null, duration_type: null, recurring_period: null, path: [], path_source: null, currency: null, scale: null, value_type: null, value_context: null },
          };
          updatePageState(pageName, { ...page, facts: [...page.facts, newFact] });
          useCanvasStore.getState().markDirty("bbox");
        }
      },
    );
  };

  return (
    <div
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.6)",
        backdropFilter: "blur(4px)",
      }}
      onClick={(e) => {
        if (e.target === e.currentTarget) closeDialog();
      }}
    >
      <div
        style={{
          width: 520,
          maxHeight: "80vh",
          background: "var(--surface-raised)",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--surface-border)",
          boxShadow: "0 24px 48px rgba(0,0,0,0.3)",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "16px 20px",
            borderBottom: "1px solid var(--surface-border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <h2
            style={{
              fontFamily: "var(--font-heading)",
              fontSize: 16,
              fontWeight: 700,
            }}
          >
            AI Inference
          </h2>
          <button
            onClick={closeDialog}
            style={{
              width: 28,
              height: 28,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "var(--radius-xs)",
              fontSize: 16,
              color: "var(--text-muted)",
              cursor: "pointer",
              background: "transparent",
            }}
          >
            {"\u00D7"}
          </button>
        </div>

        {/* Body */}
        <div style={{ padding: "16px 20px", flex: 1, overflowY: "auto" }}>
          {/* Provider selector */}
          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <ProviderPill
              active={activeProvider === "gemini"}
              onClick={() => setProvider("gemini")}
            >
              Gemini
            </ProviderPill>
            <ProviderPill
              active={activeProvider === "qwen"}
              onClick={() => setProvider("qwen")}
            >
              Qwen
            </ProviderPill>
          </div>

          {/* Options */}
          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: 10,
              marginBottom: 16,
            }}
          >
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
                color: "var(--text-muted)",
              }}
            >
              <span style={{ minWidth: 56 }}>Action</span>
              <select
                value={action}
                onChange={(e) => setAction(e.target.value)}
                style={{
                  flex: 1,
                  padding: "6px 10px",
                  fontSize: 13,
                  background: "var(--surface-alt)",
                  border: "1px solid var(--surface-border)",
                  borderRadius: "var(--radius-xs)",
                  color: "var(--text)",
                  outline: "none",
                }}
              >
                <option value="gt">Ground Truth</option>
                <option value="autocomplete">Autocomplete</option>
              </select>
            </label>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                fontSize: 13,
                color: "var(--text-muted)",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={enableThinking}
                onChange={(e) => setEnableThinking(e.target.checked)}
                style={{ accentColor: "var(--accent)" }}
              />
              Enable thinking
            </label>
          </div>

          {/* Actions */}
          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <ActionButton
              onClick={handleExtract}
              disabled={stream.isStreaming}
              primary
            >
              {stream.isStreaming ? "Extracting..." : "Extract"}
            </ActionButton>
            <ActionButton onClick={handleDetect} disabled={stream.isStreaming}>
              Detect BBoxes
            </ActionButton>
            {stream.isStreaming && (
              <ActionButton onClick={stream.cancel} danger>
                Cancel
              </ActionButton>
            )}
          </div>

          {/* Apply status */}
          {applyStatus && (
            <div
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: applyStatus.startsWith("Applied") ? "var(--ok)" : "var(--warn)",
                marginBottom: 8,
              }}
            >
              {applyStatus}
            </div>
          )}

          {/* Streaming output */}
          {(stream.fullText || stream.error) && (
            <div
              style={{
                background: "var(--surface-alt)",
                border: "1px solid var(--surface-border)",
                borderRadius: "var(--radius-sm)",
                padding: 12,
                maxHeight: 300,
                overflowY: "auto",
              }}
            >
              {stream.error ? (
                <div style={{ color: "var(--danger)", fontSize: 13 }}>
                  Error: {stream.error}
                </div>
              ) : (
                <pre
                  style={{
                    fontSize: 12,
                    fontFamily: "var(--font-mono)",
                    color: "var(--text)",
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    margin: 0,
                  }}
                >
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

function ProviderPill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "6px 16px",
        fontSize: 13,
        fontWeight: 600,
        borderRadius: "var(--radius-pill)",
        border: active
          ? "1px solid var(--accent)"
          : "1px solid var(--surface-border)",
        background: active ? "var(--accent-soft)" : "transparent",
        color: active ? "var(--accent)" : "var(--text-muted)",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
  );
}

function ActionButton({
  children,
  onClick,
  disabled,
  primary,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  primary?: boolean;
  danger?: boolean;
}) {
  const bg = danger
    ? "var(--danger)"
    : primary
      ? "var(--accent)"
      : "transparent";
  const color = danger || primary ? "#fff" : "var(--text-muted)";

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 18px",
        fontSize: 13,
        fontWeight: 600,
        borderRadius: "var(--radius-sm)",
        background: bg,
        color,
        border:
          danger || primary
            ? "none"
            : "1px solid var(--surface-border)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
  );
}
