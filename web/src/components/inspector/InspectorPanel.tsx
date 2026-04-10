/** Right sidebar inspector panel — accordion sections, resizable, RTL. */

import { useRef, useEffect, useState, useCallback } from "react";
import { useDocumentStore } from "../../stores/documentStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { DocumentMetaSection } from "./DocumentMetaSection";
import { PageMetaSection } from "./PageMetaSection";
import { FactsListSection } from "./FactsListSection";
import { FactEditorSection } from "./FactEditorSection";
import { BatchEditSection } from "./BatchEditSection";
import { PageIssuesSection } from "./PageIssuesSection";
import type { IssueSummaryCounts } from "./PageIssuesSection";
import type { PageState, BoxRecord } from "../../types/schema";

const DEFAULT_WIDTH = 500;
const MIN_WIDTH = 300;
const MAX_WIDTH = 900;

// Which sections start open
const DEFAULT_OPEN: Record<string, boolean> = {
  Document: false,
  Page: true,
  Issues: false,
  Facts: false,
  "Edit Fact": true,
  "Batch Edit": true,
};

export function InspectorPanel() {
  const { docId, pageStates, pageNames, currentPageIndex, documentMeta } =
    useDocumentStore();
  const { selectedIndices } = useSelectionStore();
  const editorRef = useRef<HTMLDivElement>(null);

  const [panelWidth, setPanelWidth] = useState(DEFAULT_WIDTH);
  const [open, setOpen] = useState<Record<string, boolean>>(DEFAULT_OPEN);
  const [issueSummary, setIssueSummary] = useState<IssueSummaryCounts | null>(null);
  const dragRef = useRef<{ startX: number; startWidth: number } | null>(null);

  // Auto-scroll to fact editor when selection changes.
  useEffect(() => {
    if (selectedIndices.size === 0) return;
    const timer = setTimeout(() => {
      editorRef.current?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }, 50);
    return () => clearTimeout(timer);
  }, [selectedIndices]);

  // Resize drag
  const onResizeStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    dragRef.current = { startX: e.clientX, startWidth: panelWidth };

    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return;
      const delta = dragRef.current.startX - ev.clientX;
      setPanelWidth(
        Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, dragRef.current.startWidth + delta)),
      );
    };
    const onUp = () => {
      dragRef.current = null;
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
  }, [panelWidth]);

  const toggle = (title: string) =>
    setOpen((prev) => ({ ...prev, [title]: !prev[title] }));

  const pageName = pageNames[currentPageIndex] ?? null;
  const page: PageState | undefined = pageName
    ? pageStates.get(pageName)
    : undefined;

  const selectedFacts: { index: number; record: BoxRecord }[] = [];
  if (page) {
    for (const idx of selectedIndices) {
      const record = page.facts[idx];
      if (record) selectedFacts.push({ index: idx, record });
    }
  }

  if (!docId) {
    return (
      <div style={{ ...panelBaseStyle(panelWidth) }}>
        <ResizeHandle onMouseDown={onResizeStart} />
        <EmptyMsg>Open a document to inspect.</EmptyMsg>
      </div>
    );
  }

  return (
    <div style={panelBaseStyle(panelWidth)}>
      <ResizeHandle onMouseDown={onResizeStart} />

      <div style={{ overflowY: "auto", flex: 1 }}>

        <AccordionSection
          title="Document"
          isOpen={open["Document"] ?? false}
          onToggle={() => toggle("Document")}
        >
          <DocumentMetaSection meta={documentMeta} />
        </AccordionSection>

        {page && (
          <AccordionSection
            title="Page"
            badge={pageName ? `${currentPageIndex + 1}` : undefined}
            isOpen={open["Page"] ?? true}
            onToggle={() => toggle("Page")}
          >
            <PageMetaSection meta={page.meta} pageName={pageName} />
          </AccordionSection>
        )}

        {page && (
          <>
            <div style={{ display: "none" }} aria-hidden="true">
              <PageIssuesSection onIssueCount={setIssueSummary} />
            </div>
            <AccordionSection
              title="Issues"
              badge={issueSummary && issueSummary.visibleCount > 0 ? `${issueSummary.visibleCount}` : undefined}
              badgeTone={
                issueSummary && issueSummary.visibleCount > 0
                  ? issueSummary.visibleFlagCount > 0
                    ? "danger"
                    : issueSummary.visibleWarningCount > 0
                      ? "warn"
                      : undefined
                  : undefined
              }
              isOpen={open["Issues"] ?? false}
              onToggle={() => toggle("Issues")}
            >
              <PageIssuesSection />
            </AccordionSection>
          </>
        )}

        {page && (
          <AccordionSection
            title="Facts"
            badge={`${page.facts.length}`}
            isOpen={open["Facts"] ?? false}
            onToggle={() => toggle("Facts")}
          >
            <FactsListSection facts={page.facts} pageName={pageName} />
          </AccordionSection>
        )}

        {selectedFacts.length > 0 && (
          <div ref={editorRef}>
            <AccordionSection
              title={selectedFacts.length === 1 ? "Edit Fact" : `Edit ${selectedFacts.length} Facts`}
              isOpen={open["Edit Fact"] ?? true}
              onToggle={() => toggle("Edit Fact")}
              highlight
            >
              <FactEditorSection
                selectedFacts={selectedFacts}
                pageName={pageName}
              />
            </AccordionSection>
          </div>
        )}

        {selectedFacts.length > 1 && (
          <AccordionSection
            title="Batch Edit"
            isOpen={open["Batch Edit"] ?? true}
            onToggle={() => toggle("Batch Edit")}
          >
            <BatchEditSection
              selectedFacts={selectedFacts}
              pageName={pageName}
            />
          </AccordionSection>
        )}

      </div>
    </div>
  );
}

// ── sub-components ────────────────────────────────────────────────

function panelBaseStyle(width: number): React.CSSProperties {
  return {
    width,
    minWidth: MIN_WIDTH,
    maxWidth: MAX_WIDTH,
    background: "var(--surface)",
    borderLeft: "1px solid var(--surface-border)",
    display: "flex",
    flexDirection: "column",
    position: "relative",
    direction: "rtl",
    flexShrink: 0,
  };
}

function ResizeHandle({ onMouseDown }: { onMouseDown: (e: React.MouseEvent) => void }) {
  return (
    <div
      onMouseDown={onMouseDown}
      style={{
        position: "absolute",
        left: 0,
        top: 0,
        bottom: 0,
        width: 5,
        cursor: "ew-resize",
        zIndex: 10,
        background: "transparent",
        transition: "background 0.15s",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = "var(--accent)";
        e.currentTarget.style.opacity = "0.4";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
        e.currentTarget.style.opacity = "1";
      }}
    />
  );
}

function AccordionSection({
  title,
  badge,
  badgeTone,
  isOpen,
  onToggle,
  highlight,
  children,
}: {
  title: string;
  badge?: string;
  badgeTone?: "danger" | "warn";
  isOpen: boolean;
  onToggle: () => void;
  highlight?: boolean;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        borderBottom: "1px solid var(--surface-border)",
      }}
    >
      {/* Header */}
      <button
        onClick={onToggle}
        style={{
          width: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "10px 16px",
          background: highlight && isOpen ? "var(--accent-soft)" : "transparent",
          border: "none",
          cursor: "pointer",
          textAlign: "right",
          direction: "rtl",
          transition: "background 0.15s",
        }}
        onMouseEnter={(e) => {
          if (!(highlight && isOpen))
            e.currentTarget.style.background = "var(--surface-alt)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.background =
            highlight && isOpen ? "var(--accent-soft)" : "transparent";
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              fontFamily: "var(--font-heading)",
              fontSize: 11,
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.06em",
              color: highlight ? "var(--accent)" : "var(--text-soft)",
            }}
          >
            {title}
          </span>
          {badge && (
            <span
              style={{
                fontSize: 10,
                fontWeight: 700,
                fontFamily: "var(--font-mono)",
                color: badgeTone === "danger"
                  ? "var(--danger)"
                  : badgeTone === "warn"
                    ? "var(--warn)"
                    : "var(--text-soft)",
                background: "var(--surface-alt)",
                padding: "1px 6px",
                borderRadius: "var(--radius-pill)",
              }}
            >
              {badge}
            </span>
          )}
        </div>
        {/* Chevron */}
        <span
          style={{
            fontSize: 11,
            color: "var(--text-soft)",
            transform: isOpen ? "rotate(0deg)" : "rotate(-90deg)",
            transition: "transform 0.2s ease",
            display: "inline-block",
            lineHeight: 1,
          }}
        >
          ▾
        </span>
      </button>

      {/* Body */}
      {isOpen && (
        <div
          style={{
            padding: "0 16px 14px",
            direction: "rtl",
          }}
        >
          {children}
        </div>
      )}
    </div>
  );
}

function EmptyMsg({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        padding: 24,
        color: "var(--text-soft)",
        fontSize: 13,
        textAlign: "center",
      }}
    >
      {children}
    </div>
  );
}
