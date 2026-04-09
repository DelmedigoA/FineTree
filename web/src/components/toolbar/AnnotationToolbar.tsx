/** Top toolbar for the annotation view — modern SaaS style. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { useAlignBBoxes } from "../../hooks/useAlignBBoxes";

export function AnnotationToolbar({
  onSave,
  onAI,
}: {
  onSave?: () => void;
  onAI?: () => void;
}) {
  const { pageNames, currentPageIndex, setCurrentPageIndex, isDirty, docId } =
    useDocumentStore();
  const { fitToView, zoomBy, zoom } = useCanvasStore();
  const { alignBBoxes, canAlign, isAligning } = useAlignBBoxes();

  const pageCount = pageNames.length;
  const currentPage = pageNames[currentPageIndex] ?? "";

  const goPrev = () => {
    if (currentPageIndex > 0) setCurrentPageIndex(currentPageIndex - 1);
  };
  const goNext = () => {
    if (currentPageIndex < pageCount - 1)
      setCurrentPageIndex(currentPageIndex + 1);
  };

  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        padding: "8px 16px",
        background: "var(--surface)",
        borderBottom: "1px solid var(--surface-border)",
        flexShrink: 0,
        minHeight: 48,
      }}
    >
      {/* Document title + status */}
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span
          style={{
            fontFamily: "var(--font-heading)",
            fontWeight: 600,
            fontSize: 14,
            color: "var(--text)",
          }}
        >
          {docId ?? "No document"}
        </span>
        {isDirty && (
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: "var(--warn)",
              padding: "2px 8px",
              borderRadius: "var(--radius-pill)",
              border: "1px solid var(--warn)",
              lineHeight: 1.4,
            }}
          >
            Unsaved
          </span>
        )}
      </div>

      <Divider />

      {/* Page navigation */}
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <IconBtn onClick={goPrev} disabled={currentPageIndex <= 0}>
          {"\u2039"}
        </IconBtn>
        <span
          style={{
            fontSize: 13,
            color: "var(--text-muted)",
            minWidth: 64,
            textAlign: "center",
            fontFamily: "var(--font-mono)",
            fontWeight: 500,
          }}
        >
          {currentPage ? `${currentPageIndex + 1} / ${pageCount}` : "\u2014"}
        </span>
        <IconBtn
          onClick={goNext}
          disabled={currentPageIndex >= pageCount - 1}
        >
          {"\u203A"}
        </IconBtn>
      </div>

      <Divider />

      {/* Zoom */}
      <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
        <IconBtn onClick={() => zoomBy(-100)}>{"\u2212"}</IconBtn>
        <span
          style={{
            fontSize: 12,
            color: "var(--text-muted)",
            minWidth: 44,
            textAlign: "center",
            fontFamily: "var(--font-mono)",
            fontWeight: 500,
          }}
        >
          {Math.round(zoom * 100)}%
        </span>
        <IconBtn onClick={() => zoomBy(100)}>+</IconBtn>
        <IconBtn onClick={fitToView} title="Fit to view">
          {"\u2922"}
        </IconBtn>
      </div>

      <div style={{ flex: 1 }} />

      {/* Right actions */}
      <ActionBtn
        onClick={alignBBoxes}
        disabled={!canAlign || isAligning}
      >
        {isAligning ? "Aligning…" : "Align BBoxes"}
      </ActionBtn>
      <ActionBtn onClick={onAI}>
        AI
      </ActionBtn>
      <ActionBtn variant="primary" onClick={onSave}>
        Save
      </ActionBtn>
    </div>
  );
}

function Divider() {
  return (
    <div
      style={{
        width: 1,
        height: 20,
        background: "var(--surface-border)",
        margin: "0 4px",
      }}
    />
  );
}

function IconBtn({
  children,
  onClick,
  disabled,
  title,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={title}
      style={{
        width: 32,
        height: 32,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        fontSize: 16,
        fontWeight: 600,
        color: disabled ? "var(--text-soft)" : "var(--text-muted)",
        background: "transparent",
        border: "none",
        borderRadius: "var(--radius-xs)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: "var(--transition-fast)",
      }}
      onMouseEnter={(e) => {
        if (!disabled)
          e.currentTarget.style.background = "var(--surface-alt)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
      }}
    >
      {children}
    </button>
  );
}

function ActionBtn({
  children,
  onClick,
  variant = "ghost",
  disabled = false,
}: {
  children: React.ReactNode;
  onClick?: () => void;
  variant?: "ghost" | "primary";
  disabled?: boolean;
}) {
  const isPrimary = variant === "primary";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "6px 14px",
        fontSize: 13,
        fontWeight: 600,
        color: disabled ? "var(--text-soft)" : isPrimary ? "#fff" : "var(--text-muted)",
        background: disabled ? "transparent" : isPrimary ? "var(--accent)" : "transparent",
        border: isPrimary ? "none" : "1px solid var(--surface-border)",
        borderRadius: "var(--radius-xs)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.5 : 1,
        transition: "var(--transition-fast)",
      }}
      onMouseEnter={(e) => {
        if (disabled) return;
        if (isPrimary) {
          e.currentTarget.style.background = "var(--accent-strong)";
        } else {
          e.currentTarget.style.background = "var(--surface-alt)";
        }
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = disabled
          ? "transparent"
          : isPrimary
          ? "var(--accent)"
          : "transparent";
      }}
    >
      {children}
    </button>
  );
}
