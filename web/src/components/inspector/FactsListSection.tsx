/** Scrollable facts list with extended selection. */

import { useSelectionStore } from "../../stores/selectionStore";
import { useCanvasStore } from "../../stores/canvasStore";
import type { BoxRecord } from "../../types/schema";

interface Props {
  facts: BoxRecord[];
  pageName: string | null;
}

export function FactsListSection({ facts }: Props) {
  const { selectedIndices, select, toggleSelect, setHovered } =
    useSelectionStore();
  const markDirty = useCanvasStore((s) => s.markDirty);

  if (facts.length === 0) {
    return (
      <div
        style={{
          padding: "16px 0",
          color: "var(--text-soft)",
          fontSize: 12,
          textAlign: "center",
        }}
      >
        No facts on this page.
      </div>
    );
  }

  return (
    <div
      style={{
        maxHeight: 220,
        overflowY: "auto",
        display: "flex",
        flexDirection: "column",
        gap: 2,
      }}
    >
      {facts.map((record, idx) => {
        const isSelected = selectedIndices.has(idx);
        const value = String(record.fact.value ?? "");
        const path = (record.fact.path as string[]) ?? [];
        const factNum = record.fact.fact_num as number | null;

        return (
          <button
            key={idx}
            onClick={(e) => {
              if (e.shiftKey || e.metaKey || e.ctrlKey) {
                toggleSelect(idx);
              } else {
                select(idx);
              }
              markDirty("bbox");
            }}
            onMouseEnter={() => {
              setHovered(idx);
              markDirty("interaction");
            }}
            onMouseLeave={() => {
              setHovered(null);
              markDirty("interaction");
            }}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "7px 10px",
              background: isSelected
                ? "var(--accent-soft)"
                : "transparent",
              border: isSelected
                ? "1px solid var(--accent)"
                : "1px solid transparent",
              borderRadius: "var(--radius-xs)",
              cursor: "pointer",
              textAlign: "left",
              transition: "var(--transition-fast)",
              width: "100%",
            }}
            onMouseDown={(e) => {
              // Don't focus-steal from editor.
              if (!isSelected) e.preventDefault();
            }}
          >
            {/* Fact number */}
            <span
              style={{
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                fontWeight: 600,
                color: "var(--text-soft)",
                minWidth: 20,
                textAlign: "right",
              }}
            >
              {factNum ?? idx + 1}
            </span>

            {/* Value + path preview */}
            <div style={{ flex: 1, overflow: "hidden" }}>
              <div
                style={{
                  fontSize: 13,
                  fontWeight: 500,
                  color: "var(--text)",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap",
                }}
              >
                {value || "\u2014"}
              </div>
              {path.length > 0 && (
                <div
                  style={{
                    fontSize: 10,
                    color: "var(--text-soft)",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                    whiteSpace: "nowrap",
                    marginTop: 1,
                  }}
                >
                  {path.join(" \u203A ")}
                </div>
              )}
            </div>
          </button>
        );
      })}
    </div>
  );
}
