/** Page validation issues — reg flags + warnings. */

import { useState, useEffect, useCallback, useRef } from "react";
import { post } from "../../api/client";
import { useDocumentStore } from "../../stores/documentStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { useCanvasStore } from "../../stores/canvasStore";
import type { ValidationResult, ValidationIssue } from "../../types/api";

export function PageIssuesSection({
  onIssueCount,
}: {
  onIssueCount?: (count: number | null) => void;
}) {
  const { docId, pageNames, currentPageIndex } = useDocumentStore();
  const [issues, setIssues] = useState<{ flags: ValidationIssue[]; warnings: ValidationIssue[] } | null>(null);
  const [loading, setLoading] = useState(false);
  // Locally acknowledged issue keys (code + fact_index), reset on page change.
  const [acknowledged, setAcknowledged] = useState<Set<string>>(new Set());

  const pageName = pageNames[currentPageIndex] ?? null;

  const onIssueCountRef = useRef(onIssueCount);
  onIssueCountRef.current = onIssueCount;

  const validate = useCallback(async () => {
    const { docId: currentDocId, pageNames: names, currentPageIndex: idx } =
      useDocumentStore.getState();
    const currentPageName = names[idx] ?? null;
    if (!currentDocId) return;
    setLoading(true);
    try {
      const result = await post<ValidationResult>(
        `/annotations/${currentDocId}/validate`,
        {},
      );
      const pageResult = currentPageName ? result.pages[currentPageName] : null;
      const resolved = pageResult
        ? { flags: pageResult.reg_flags ?? [], warnings: pageResult.warnings ?? [] }
        : { flags: [], warnings: [] };
      setIssues(resolved);
      onIssueCountRef.current?.(resolved.flags.length + resolved.warnings.length);
    } catch {
      setIssues(null);
      onIssueCountRef.current?.(null);
    } finally {
      setLoading(false);
    }
  }, []);

  const validateRef = useRef(validate);
  validateRef.current = validate;

  // Auto-validate when switching pages (with a short debounce).
  useEffect(() => {
    setIssues(null);
    setAcknowledged(new Set());
    onIssueCountRef.current?.(null);
    if (!docId || !pageName) return;
    const timer = setTimeout(() => validateRef.current(), 400);
    return () => clearTimeout(timer);
  }, [pageName, docId]);

  if (!docId) return null;

  const issueKey = (issue: ValidationIssue) => `${issue.code}:${issue.fact_index ?? ""}`;
  const visibleFlags = (issues?.flags ?? []).filter((f) => !acknowledged.has(issueKey(f)));
  const visibleWarnings = (issues?.warnings ?? []).filter((w) => !acknowledged.has(issueKey(w)));
  const totalVisible = visibleFlags.length + visibleWarnings.length;
  const totalCount = (issues?.flags.length ?? 0) + (issues?.warnings.length ?? 0);

  const handleResolve = (issue: ValidationIssue) => {
    setAcknowledged((prev) => new Set([...prev, issueKey(issue)]));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        {totalVisible > 0 && (
          <span
            style={{
              fontSize: 11,
              fontWeight: 700,
              color: visibleFlags.length > 0 ? "var(--danger)" : "var(--warn)",
              fontFamily: "var(--font-mono)",
            }}
          >
            {visibleFlags.length > 0 && `${visibleFlags.length} flag${visibleFlags.length !== 1 ? "s" : ""}`}
            {visibleFlags.length > 0 && visibleWarnings.length > 0 && " · "}
            {visibleWarnings.length > 0 && `${visibleWarnings.length} warning${visibleWarnings.length !== 1 ? "s" : ""}`}
          </span>
        )}
        {issues && totalVisible === 0 && (
          <span style={{ fontSize: 11, color: "var(--ok)", fontWeight: 600 }}>
            {totalCount > 0 ? `All ${totalCount} resolved` : "No issues"}
          </span>
        )}
        <button
          onClick={() => { setAcknowledged(new Set()); validate(); }}
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

      {issues && totalVisible > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
          {visibleFlags.map((flag, i) => (
            <IssueItem key={`f${i}`} issue={flag} type="flag" onResolve={() => handleResolve(flag)} />
          ))}
          {visibleWarnings.map((warn, i) => (
            <IssueItem key={`w${i}`} issue={warn} type="warning" onResolve={() => handleResolve(warn)} />
          ))}
        </div>
      )}
    </div>
  );
}

function IssueItem({
  issue,
  type,
  onResolve,
}: {
  issue: ValidationIssue;
  type: "flag" | "warning";
  onResolve: () => void;
}) {
  const select = useSelectionStore((s) => s.select);
  const markDirty = useCanvasStore((s) => s.markDirty);
  const lastClickRef = useRef<number>(0);

  const handleClick = () => {
    // Single click: navigate to the fact.
    if (issue.fact_index !== null && issue.fact_index >= 0) {
      select(issue.fact_index);
      markDirty("bbox");
    }
  };

  const handleDoubleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    onResolve();
  };

  // Use manual double-click tracking to avoid firing single-click after double-click.
  const handlePointerDown = () => {
    const now = Date.now();
    if (now - lastClickRef.current < 350) {
      // Double click detected.
      onResolve();
      lastClickRef.current = 0;
    } else {
      lastClickRef.current = now;
      // Single click: navigate after a short delay (cancel if double-click follows).
      setTimeout(() => {
        if (lastClickRef.current !== 0 && Date.now() - lastClickRef.current >= 340) {
          handleClick();
        }
      }, 360);
    }
  };

  return (
    <div
      onPointerDown={handlePointerDown}
      onDoubleClick={handleDoubleClick}
      title="Click to navigate · Double-click to resolve"
      style={{
        display: "flex",
        alignItems: "flex-start",
        gap: 6,
        padding: "6px 8px",
        background: type === "flag" ? "rgba(185,73,73,0.08)" : "rgba(183,121,31,0.08)",
        borderRadius: "var(--radius-xs)",
        borderLeft: `3px solid ${type === "flag" ? "var(--danger)" : "var(--warn)"}`,
        cursor: "pointer",
        userSelect: "none",
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
          flex: 1,
        }}
      >
        {issue.message}
      </span>
      {issue.fact_index !== null && (
        <span style={{ fontSize: 10, color: "var(--text-soft)", fontFamily: "var(--font-mono)", marginTop: 1 }}>
          f{issue.fact_index + 1}
        </span>
      )}
    </div>
  );
}
