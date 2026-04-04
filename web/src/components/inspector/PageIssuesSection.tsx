/** Page validation issues — reg flags + warnings. */

import { useState, useEffect } from "react";
import { post } from "../../api/client";
import { useDocumentStore } from "../../stores/documentStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { useCanvasStore } from "../../stores/canvasStore";
import type { ValidationResult } from "../../types/api";

export function PageIssuesSection() {
  const { docId, pageNames, currentPageIndex } = useDocumentStore();
  const [issues, setIssues] = useState<{ flags: string[]; warnings: string[] } | null>(null);
  const [loading, setLoading] = useState(false);

  const pageName = pageNames[currentPageIndex] ?? null;

  const validate = async () => {
    if (!docId) return;
    setLoading(true);
    try {
      const result = await post<ValidationResult>(`/annotations/${docId}/validate`, {});
      const pageResult = pageName ? result.pages[pageName] : null;
      setIssues(
        pageResult
          ? { flags: pageResult.reg_flags, warnings: pageResult.warnings }
          : { flags: [], warnings: [] },
      );
    } catch {
      setIssues(null);
    } finally {
      setLoading(false);
    }
  };

  // Re-validate when page changes.
  useEffect(() => {
    setIssues(null);
  }, [pageName]);

  if (!docId) return null;

  const totalCount = (issues?.flags.length ?? 0) + (issues?.warnings.length ?? 0);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        {totalCount > 0 && (
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: (issues?.flags.length ?? 0) > 0 ? "var(--danger)" : "var(--warn)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {issues!.flags.length > 0 && `${issues!.flags.length} flag${issues!.flags.length !== 1 ? "s" : ""}`}
            {issues!.flags.length > 0 && issues!.warnings.length > 0 && " · "}
            {issues!.warnings.length > 0 && `${issues!.warnings.length} warning${issues!.warnings.length !== 1 ? "s" : ""}`}
          </span>
        )}
        {issues && totalCount === 0 && (
          <span style={{ fontSize: 11, color: "var(--ok)", fontWeight: 600 }}>
            No issues
          </span>
        )}
        <button
          onClick={validate}
          disabled={loading}
          style={{
            fontSize: 11,
            fontWeight: 600,
            color: "var(--accent)",
            background: "transparent",
            border: "1px solid var(--accent)",
            borderRadius: "var(--radius-xs)",
            padding: "3px 10px",
            cursor: loading ? "default" : "pointer",
            opacity: loading ? 0.5 : 1,
            marginLeft: "auto",
            transition: "var(--transition-fast)",
          }}
        >
          {loading ? "Validating..." : "Validate"}
        </button>
      </div>

      {issues && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {issues.flags.map((flag, i) => (
            <IssueItem key={`f${i}`} text={flag} type="flag" />
          ))}
          {issues.warnings.map((warn, i) => (
            <IssueItem key={`w${i}`} text={warn} type="warning" />
          ))}
        </div>
      )}
    </div>
  );
}

function IssueItem({ text, type }: { text: string; type: "flag" | "warning" }) {
  const select = useSelectionStore((s) => s.select);
  const markDirty = useCanvasStore((s) => s.markDirty);

  // Parse fact index from issue text if present (e.g. "fact[3]: ...")
  const match = text.match(/fact\[(\d+)\]/);
  const factIdx = match ? parseInt(match[1]!, 10) - 1 : null;

  const handleClick = () => {
    if (factIdx !== null && factIdx >= 0) {
      select(factIdx);
      markDirty("bbox");
    }
  };

  return (
    <div
      onClick={factIdx !== null ? handleClick : undefined}
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 6,
        padding: "6px 8px",
        background: type === "flag" ? "rgba(185,73,73,0.08)" : "rgba(183,121,31,0.08)",
        borderRadius: "var(--radius-xs)",
        borderLeft: `3px solid ${type === "flag" ? "var(--danger)" : "var(--warn)"}`,
        cursor: factIdx !== null ? "pointer" : "default",
      }}
    >
      <span
        style={{
          fontSize: 10,
          fontWeight: 700,
          color: type === "flag" ? "var(--danger)" : "var(--warn)",
          textTransform: "uppercase",
          minWidth: 36,
          marginTop: 1,
        }}
      >
        {type === "flag" ? "Flag" : "Warn"}
      </span>
      <span
        style={{
          fontSize: 11,
          color: "var(--text-muted)",
          lineHeight: 1.4,
        }}
      >
        {text}
      </span>
    </div>
  );
}
