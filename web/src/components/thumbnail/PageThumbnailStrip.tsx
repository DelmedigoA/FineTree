/** Vertical page thumbnail strip — clean modern style. */

import { useDocumentStore } from "../../stores/documentStore";

export function PageThumbnailStrip() {
  const { docId, pageNames, currentPageIndex, setCurrentPageIndex } =
    useDocumentStore();

  if (!docId || pageNames.length === 0) return null;

  return (
    <div
      style={{
        width: 88,
        minWidth: 88,
        background: "var(--surface)",
        borderRight: "1px solid var(--surface-border)",
        overflowY: "auto",
        padding: "8px 6px",
        display: "flex",
        flexDirection: "column",
        gap: 4,
      }}
    >
      {pageNames.map((name, idx) => {
        const active = idx === currentPageIndex;
        return (
          <button
            key={name}
            onClick={() => setCurrentPageIndex(idx)}
            style={{
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              gap: 3,
              padding: 3,
              background: active ? "var(--accent-soft)" : "transparent",
              border: active
                ? "2px solid var(--accent)"
                : "1px solid transparent",
              borderRadius: "var(--radius-xs)",
              cursor: "pointer",
              transition: "var(--transition-fast)",
            }}
            onMouseEnter={(e) => {
              if (!active)
                e.currentTarget.style.background = "var(--surface-alt)";
            }}
            onMouseLeave={(e) => {
              if (!active)
                e.currentTarget.style.background = "transparent";
            }}
          >
            <img
              src={`/api/images/${docId}/thumbnails/${name}`}
              alt={`Page ${idx + 1}`}
              style={{
                width: "100%",
                borderRadius: 4,
                objectFit: "contain",
                opacity: active ? 1 : 0.7,
              }}
              loading="lazy"
            />
            <span
              style={{
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                fontWeight: active ? 700 : 400,
                color: active ? "var(--accent)" : "var(--text-soft)",
              }}
            >
              {idx + 1}
            </span>
          </button>
        );
      })}
    </div>
  );
}
