/** Right sidebar inspector panel — document/page meta, facts list, fact editor. */

import { useDocumentStore } from "../../stores/documentStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { DocumentMetaSection } from "./DocumentMetaSection";
import { PageMetaSection } from "./PageMetaSection";
import { FactsListSection } from "./FactsListSection";
import { FactEditorSection } from "./FactEditorSection";
import { BatchEditSection } from "./BatchEditSection";
import { PageIssuesSection } from "./PageIssuesSection";
import type { PageState, BoxRecord } from "../../types/schema";

export function InspectorPanel() {
  const { docId, pageStates, pageNames, currentPageIndex, documentMeta } =
    useDocumentStore();
  const { selectedIndices } = useSelectionStore();

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
      <div style={panelStyle}>
        <EmptyMsg>Open a document to inspect.</EmptyMsg>
      </div>
    );
  }

  return (
    <div style={panelStyle}>
      {/* Document metadata */}
      <Section title="Document">
        <DocumentMetaSection meta={documentMeta} />
      </Section>

      {/* Page metadata */}
      {page && (
        <Section
          title="Page"
          badge={pageName ? `${currentPageIndex + 1}` : undefined}
        >
          <PageMetaSection meta={page.meta} pageName={pageName} />
        </Section>
      )}

      {/* Page issues */}
      {page && (
        <Section title="Issues">
          <PageIssuesSection />
        </Section>
      )}

      {/* Facts list */}
      {page && (
        <Section title="Facts" badge={`${page.facts.length}`}>
          <FactsListSection facts={page.facts} pageName={pageName} />
        </Section>
      )}

      {/* Fact editor (when selected) */}
      {selectedFacts.length > 0 && (
        <Section
          title={
            selectedFacts.length === 1
              ? "Edit Fact"
              : `Edit ${selectedFacts.length} Facts`
          }
        >
          <FactEditorSection
            selectedFacts={selectedFacts}
            pageName={pageName}
          />
        </Section>
      )}

      {/* Batch edit (multi-select only) */}
      {selectedFacts.length > 1 && (
        <Section title="Batch Edit">
          <BatchEditSection
            selectedFacts={selectedFacts}
            pageName={pageName}
          />
        </Section>
      )}
    </div>
  );
}

const panelStyle: React.CSSProperties = {
  width: 340,
  minWidth: 340,
  background: "var(--surface)",
  borderLeft: "1px solid var(--surface-border)",
  overflowY: "auto",
  padding: "12px 0",
  display: "flex",
  flexDirection: "column",
  gap: 0,
};

function Section({
  title,
  badge,
  children,
}: {
  title: string;
  badge?: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        borderBottom: "1px solid var(--surface-border)",
        padding: "12px 16px",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 8,
          marginBottom: 12,
        }}
      >
        <h3
          style={{
            fontFamily: "var(--font-heading)",
            fontSize: 11,
            fontWeight: 700,
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            color: "var(--text-soft)",
          }}
        >
          {title}
        </h3>
        {badge && (
          <span
            style={{
              fontSize: 10,
              fontWeight: 700,
              fontFamily: "var(--font-mono)",
              color: "var(--text-soft)",
              background: "var(--surface-alt)",
              padding: "1px 6px",
              borderRadius: "var(--radius-pill)",
            }}
          >
            {badge}
          </span>
        )}
      </div>
      {children}
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
