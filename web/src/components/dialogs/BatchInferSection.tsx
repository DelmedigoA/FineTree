import { useCallback, useEffect, useRef, useState } from "react";

interface DocProgress {
  doc_id: string;
  total_pages: number;
  completed_pages: number;
  failed_pages: number;
  received_tokens: number;
  fact_count: number;
  current_page: string | null;
  status: "pending" | "running" | "done" | "error";
  failures: { page: string; error: string }[];
  save_error: string | null;
}

type SSEEvent =
  | { type: "start"; doc_id: string; total_pages: number }
  | {
      type: "progress";
      doc_id: string;
      page_name: string;
      page_index: number;
      total_pages: number;
      completed_pages: number;
      failed_pages: number;
      received_tokens: number;
      fact_count: number;
    }
  | {
      type: "doc_done";
      doc_id: string;
      completed_pages: number;
      failed_pages: number;
      fact_count: number;
      received_tokens: number;
      save_error: string | null;
      failures: { page: string; error: string }[];
    }
  | { type: "done"; total_docs: number; total_facts: number }
  | { type: "error"; message: string };

const DEFAULT_BASE_URL = "https://your-endpoint.ngrok-free.app/v1";
const DEFAULT_MODEL = "asafd60/FineTree-27B-v2.9-merged";

function buildDocProgress(docIds: string[]): Record<string, DocProgress> {
  return Object.fromEntries(
    docIds.map((id) => [
      id,
      {
        doc_id: id,
        total_pages: 0,
        completed_pages: 0,
        failed_pages: 0,
        received_tokens: 0,
        fact_count: 0,
        current_page: null,
        status: "pending" as const,
        failures: [],
        save_error: null,
      },
    ]),
  );
}

export function BatchInferSection({
  docIds,
}: {
  docIds: string[];
}) {
  const [baseUrl, setBaseUrl] = useState(DEFAULT_BASE_URL);
  const [modelId, setModelId] = useState(DEFAULT_MODEL);
  const [action, setAction] = useState<"gt" | "autocomplete">("gt");

  const [running, setRunning] = useState(false);
  const [finished, setFinished] = useState(false);
  const [globalErr, setGlobalErr] = useState<string | null>(null);
  const [summary, setSummary] = useState<{ docs: number; facts: number } | null>(
    null,
  );
  const [docs, setDocs] = useState<Record<string, DocProgress>>(() =>
    buildDocProgress(docIds),
  );

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setDocs(buildDocProgress(docIds));
    setRunning(false);
    setFinished(false);
    setGlobalErr(null);
    setSummary(null);
  }, [docIds]);

  const updateDoc = useCallback((doc_id: string, patch: Partial<DocProgress>) => {
    setDocs((prev) => ({
      ...prev,
      [doc_id]: { ...prev[doc_id]!, ...patch },
    }));
  }, []);

  const handleStart = async () => {
    if (!baseUrl.trim() || !modelId.trim() || docIds.length === 0) return;
    setRunning(true);
    setFinished(false);
    setGlobalErr(null);
    setSummary(null);
    setDocs(buildDocProgress(docIds));

    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const resp = await fetch("/api/ai/batch-infer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          doc_ids: docIds,
          base_url: baseUrl.trim(),
          model_id: modelId.trim(),
          action,
        }),
        signal: abort.signal,
      });

      if (!resp.ok || !resp.body) {
        throw new Error(`Server error ${resp.status}`);
      }

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;

          let ev: SSEEvent;
          try {
            ev = JSON.parse(raw);
          } catch {
            continue;
          }

          if (ev.type === "start") {
            updateDoc(ev.doc_id, { total_pages: ev.total_pages, status: "running" });
          } else if (ev.type === "progress") {
            updateDoc(ev.doc_id, {
              total_pages: ev.total_pages,
              completed_pages: ev.completed_pages,
              failed_pages: ev.failed_pages,
              received_tokens: ev.received_tokens,
              fact_count: ev.fact_count,
              current_page: ev.page_name,
              status: "running",
            });
          } else if (ev.type === "doc_done") {
            updateDoc(ev.doc_id, {
              completed_pages: ev.completed_pages,
              failed_pages: ev.failed_pages,
              fact_count: ev.fact_count,
              received_tokens: ev.received_tokens,
              save_error: ev.save_error,
              failures: ev.failures,
              current_page: null,
              status:
                ev.failed_pages === ev.completed_pages && ev.completed_pages > 0
                  ? "error"
                  : "done",
            });
          } else if (ev.type === "done") {
            setSummary({ docs: ev.total_docs, facts: ev.total_facts });
            setFinished(true);
          } else if (ev.type === "error") {
            setGlobalErr(ev.message);
            setFinished(true);
          }
        }
      }
    } catch (err: unknown) {
      if ((err as Error)?.name !== "AbortError") {
        setGlobalErr(String(err));
      }
      setFinished(true);
    } finally {
      setRunning(false);
      abortRef.current = null;
    }
  };

  const handleStop = () => {
    abortRef.current?.abort();
    setRunning(false);
  };

  const totalPages = Object.values(docs).reduce((sum, doc) => sum + doc.total_pages, 0);
  const donePages = Object.values(docs).reduce(
    (sum, doc) => sum + doc.completed_pages,
    0,
  );
  const totalTokens = Object.values(docs).reduce(
    (sum, doc) => sum + doc.received_tokens,
    0,
  );
  const overallPct = totalPages > 0 ? Math.round((donePages / totalPages) * 100) : 0;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <p style={{ fontSize: 12, color: "var(--text-muted)", margin: 0 }}>
        Run the current PDF through the document-level VLM inference endpoint.
      </p>

      {!running && !finished && (
        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <ConfigRow label="Endpoint URL">
            <input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              placeholder="https://your-endpoint/v1"
              style={inputStyle}
            />
          </ConfigRow>
          <ConfigRow label="Model ID">
            <input
              value={modelId}
              onChange={(e) => setModelId(e.target.value)}
              placeholder="asafd60/FineTree-27B-v2.9-merged"
              style={inputStyle}
            />
          </ConfigRow>
          <ConfigRow label="Action">
            <select
              value={action}
              onChange={(e) =>
                setAction(e.target.value as "gt" | "autocomplete")
              }
              style={inputStyle}
            >
              <option value="gt">Ground Truth (replace all)</option>
              <option value="autocomplete">Autocomplete (merge)</option>
            </select>
          </ConfigRow>
        </div>
      )}

      {(running || finished) && totalPages > 0 && (
        <div
          style={{
            background: "var(--surface-alt)",
            borderRadius: "var(--radius-sm)",
            padding: "10px 14px",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: 6,
            }}
          >
            <span style={{ fontSize: 13, fontWeight: 600 }}>
              Overall - {donePages}/{totalPages} pages
            </span>
            <span
              style={{
                fontSize: 12,
                fontFamily: "var(--font-mono)",
                color: "var(--text-muted)",
              }}
            >
              {(totalTokens / 1000).toFixed(1)}k tokens
            </span>
          </div>
          <ProgressBar pct={overallPct} />
        </div>
      )}

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        {Object.values(docs).map((doc) => (
          <DocRow key={doc.doc_id} doc={doc} />
        ))}
      </div>

      {globalErr && (
        <div
          style={{
            padding: "10px 14px",
            background: "rgba(239,68,68,0.1)",
            borderRadius: "var(--radius-sm)",
            fontSize: 13,
            color: "var(--danger)",
            border: "1px solid rgba(239,68,68,0.3)",
          }}
        >
          Error: {globalErr}
        </div>
      )}

      {summary && (
        <div
          style={{
            padding: "12px 14px",
            background: "var(--accent-soft)",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--accent)",
            fontSize: 13,
            color: "var(--accent)",
            fontWeight: 600,
          }}
        >
          ✓ Done - {summary.docs} doc{summary.docs !== 1 ? "s" : ""} ·{" "}
          {summary.facts.toLocaleString()} facts saved
        </div>
      )}

      <div style={{ display: "flex", gap: 8, justifyContent: "flex-end" }}>
        {!running && !finished && (
          <FooterBtn
            primary
            disabled={!baseUrl.trim() || !modelId.trim() || docIds.length === 0}
            onClick={handleStart}
          >
            Run Inference
          </FooterBtn>
        )}
        {running && (
          <FooterBtn danger onClick={handleStop}>
            Stop
          </FooterBtn>
        )}
      </div>
    </div>
  );
}

function DocRow({ doc }: { doc: DocProgress }) {
  const [expanded, setExpanded] = useState(false);
  const pct =
    doc.total_pages > 0
      ? Math.round((doc.completed_pages / doc.total_pages) * 100)
      : 0;
  const statusColor =
    doc.status === "done"
      ? "var(--ok)"
      : doc.status === "error"
        ? "var(--danger)"
        : doc.status === "running"
          ? "var(--accent)"
          : "var(--text-soft)";

  return (
    <div
      style={{
        background: "var(--surface-alt)",
        borderRadius: "var(--radius-sm)",
        border: "1px solid var(--surface-border)",
        overflow: "hidden",
      }}
    >
      <div
        style={{
          padding: "10px 14px",
          cursor: doc.failures.length > 0 ? "pointer" : "default",
        }}
        onClick={() => doc.failures.length > 0 && setExpanded(!expanded)}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            marginBottom: doc.status === "pending" ? 0 : 6,
          }}
        >
          <span
            style={{
              fontSize: 13,
              fontWeight: 600,
              flex: 1,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {doc.doc_id}
          </span>
          <span style={{ fontSize: 11, fontWeight: 700, color: statusColor }}>
            {doc.status === "pending"
              ? "Pending"
              : doc.status === "running"
                ? `${doc.completed_pages}/${doc.total_pages}`
                : doc.status === "done"
                  ? `✓ ${doc.fact_count} facts`
                  : `✗ ${doc.failed_pages} failed`}
          </span>
          {doc.received_tokens > 0 && (
            <span
              style={{
                fontSize: 11,
                color: "var(--text-soft)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {(doc.received_tokens / 1000).toFixed(1)}k tok
            </span>
          )}
        </div>

        {doc.status !== "pending" && (
          <ProgressBar pct={pct} danger={doc.failed_pages > 0} />
        )}

        {doc.current_page && doc.status === "running" && (
          <div
            style={{
              marginTop: 4,
              fontSize: 11,
              color: "var(--text-soft)",
              fontFamily: "var(--font-mono)",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {doc.current_page}
          </div>
        )}

        {doc.save_error && (
          <div style={{ marginTop: 4, fontSize: 11, color: "var(--danger)" }}>
            Save error: {doc.save_error}
          </div>
        )}

        {doc.failures.length > 0 && (
          <div style={{ marginTop: 4, fontSize: 11, color: "var(--warn)" }}>
            {doc.failed_pages} page{doc.failed_pages !== 1 ? "s" : ""} failed{" "}
            {expanded ? "▴" : "▾"}
          </div>
        )}
      </div>

      {expanded && doc.failures.length > 0 && (
        <div
          style={{
            borderTop: "1px solid var(--surface-border)",
            padding: "8px 14px",
            display: "flex",
            flexDirection: "column",
            gap: 3,
          }}
        >
          {doc.failures.map((failure, index) => (
            <div
              key={index}
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--danger)",
              }}
            >
              {failure.page}: {failure.error}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function ProgressBar({ pct, danger }: { pct: number; danger?: boolean }) {
  return (
    <div
      style={{
        height: 4,
        background: "var(--surface-border)",
        borderRadius: 2,
        overflow: "hidden",
      }}
    >
      <div
        style={{
          height: "100%",
          width: `${pct}%`,
          background: danger
            ? "var(--warn)"
            : pct >= 100
              ? "var(--ok)"
              : "var(--accent)",
          borderRadius: 2,
          transition: "width 0.3s ease",
        }}
      />
    </div>
  );
}

function ConfigRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        fontSize: 13,
        color: "var(--text-muted)",
      }}
    >
      <span style={{ minWidth: 90, fontWeight: 500 }}>{label}</span>
      {children}
    </label>
  );
}

function FooterBtn({
  children,
  onClick,
  primary,
  danger,
  disabled,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  primary?: boolean;
  danger?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 20px",
        fontSize: 13,
        fontWeight: 600,
        borderRadius: "var(--radius-sm)",
        background: danger
          ? "var(--danger)"
          : primary
            ? "var(--accent)"
            : "transparent",
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

const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: "6px 10px",
  fontSize: 13,
  background: "var(--surface)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-xs)",
  color: "var(--text)",
  outline: "none",
};
