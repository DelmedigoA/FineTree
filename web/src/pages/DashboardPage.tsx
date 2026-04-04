/** Dashboard — workspace documents grid, Roboflow-inspired. */

import { useState, useEffect, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { get, post, del } from "../api/client";
import type { WorkspaceDocument } from "../types/api";

export function DashboardPage() {
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<"all" | "new" | "annotated" | "checked" | "reviewed">("all");
  const navigate = useNavigate();

  const refresh = useCallback(() => {
    setLoading(true);
    get<WorkspaceDocument[]>("/workspace/documents")
      .then(setDocuments)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const filteredDocs = documents.filter((d) => {
    switch (filter) {
      case "new": return !d.has_annotations;
      case "annotated": return d.has_annotations && !d.checked;
      case "checked": return d.checked && !d.reviewed;
      case "reviewed": return d.reviewed;
      default: return true;
    }
  });

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => {
    if (selectedIds.size === filteredDocs.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredDocs.map((d) => d.doc_id)));
    }
  };

  const handleDelete = async (docId: string) => {
    try {
      await del(`/workspace/documents/${docId}`);
      refresh();
      setSelectedIds((prev) => {
        const next = new Set(prev);
        next.delete(docId);
        return next;
      });
    } catch (err) {
      console.error("Delete failed:", err);
    }
  };

  const handleMarkChecked = async (docId: string) => {
    try {
      await post(`/workspace/documents/${docId}/checked`, {});
      refresh();
    } catch (err) {
      console.error("Mark checked failed:", err);
    }
  };

  const handleMarkReviewed = async (docId: string) => {
    try {
      await post(`/workspace/documents/${docId}/reviewed`, {});
      refresh();
    } catch (err) {
      console.error("Mark reviewed failed:", err);
    }
  };

  const handleBatchMarkChecked = async () => {
    for (const id of selectedIds) {
      await post(`/workspace/documents/${id}/checked`, {}).catch(console.error);
    }
    refresh();
  };

  const handleBatchMarkReviewed = async () => {
    for (const id of selectedIds) {
      await post(`/workspace/documents/${id}/reviewed`, {}).catch(console.error);
    }
    refresh();
  };

  const handleImportPdf = async () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf";
    input.multiple = true;
    input.onchange = async () => {
      if (!input.files) return;
      for (const file of input.files) {
        const formData = new FormData();
        formData.append("file", file);
        try {
          await fetch("/api/workspace/import-pdf", {
            method: "POST",
            body: formData,
          });
        } catch (err) {
          console.error("Import failed:", err);
        }
      }
      refresh();
    };
    input.click();
  };

  const allSelected = filteredDocs.length > 0 && selectedIds.size === filteredDocs.length;

  return (
    <div style={{ height: "100%", overflowY: "auto" }}>
      {/* Header */}
      <div
        style={{
          padding: "32px 40px 24px",
          borderBottom: "1px solid var(--surface-border)",
          background: "var(--surface)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <div>
            <h1
              style={{
                fontFamily: "var(--font-heading)",
                fontSize: 24,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                marginBottom: 4,
              }}
            >
              Projects
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 14 }}>
              {loading
                ? "Loading..."
                : `${documents.length} document${documents.length !== 1 ? "s" : ""} in workspace`}
            </p>
          </div>
          <button
            onClick={handleImportPdf}
            style={{
              padding: "10px 20px",
              background: "var(--accent)",
              color: "#fff",
              borderRadius: "var(--radius-sm)",
              fontSize: 13,
              fontWeight: 600,
              border: "none",
              cursor: "pointer",
              transition: "var(--transition-fast)",
            }}
            onMouseEnter={(e) =>
              (e.currentTarget.style.background = "var(--accent-strong)")
            }
            onMouseLeave={(e) =>
              (e.currentTarget.style.background = "var(--accent)")
            }
          >
            + Import PDF
          </button>
        </div>

        {/* Filter bar */}
        <div style={{ display: "flex", gap: 6, marginTop: 16 }}>
          {(["all", "new", "annotated", "checked", "reviewed"] as const).map(
            (f) => (
              <FilterPill
                key={f}
                active={filter === f}
                onClick={() => setFilter(f)}
              >
                {f.charAt(0).toUpperCase() + f.slice(1)}
              </FilterPill>
            ),
          )}
        </div>
      </div>

      {/* Batch actions bar */}
      {selectedIds.size > 0 && (
        <div
          style={{
            padding: "10px 40px",
            background: "var(--accent-soft)",
            borderBottom: "1px solid var(--surface-border)",
            display: "flex",
            alignItems: "center",
            gap: 12,
            fontSize: 13,
          }}
        >
          <span style={{ fontWeight: 600, color: "var(--accent)" }}>
            {selectedIds.size} selected
          </span>
          <BatchActionBtn onClick={handleBatchMarkChecked}>
            Mark Checked
          </BatchActionBtn>
          <BatchActionBtn onClick={handleBatchMarkReviewed}>
            Mark Reviewed
          </BatchActionBtn>
          <button
            onClick={() => setSelectedIds(new Set())}
            style={{
              marginLeft: "auto",
              fontSize: 12,
              color: "var(--text-muted)",
              background: "transparent",
              border: "none",
              cursor: "pointer",
              textDecoration: "underline",
            }}
          >
            Clear selection
          </button>
        </div>
      )}

      {/* Content */}
      <div style={{ padding: "24px 40px 40px" }}>
        {loading ? (
          <LoadingGrid />
        ) : filteredDocs.length === 0 ? (
          documents.length === 0 ? (
            <EmptyState />
          ) : (
            <div
              style={{
                padding: 40,
                textAlign: "center",
                color: "var(--text-muted)",
                fontSize: 14,
              }}
            >
              No documents match the current filter.
            </div>
          )
        ) : (
          <>
            {/* Select all toggle */}
            <div
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                marginBottom: 16,
                fontSize: 12,
                color: "var(--text-muted)",
              }}
            >
              <input
                type="checkbox"
                checked={allSelected}
                onChange={selectAll}
                style={{ accentColor: "var(--accent)" }}
              />
              <span>
                {allSelected ? "Deselect all" : "Select all"} ({filteredDocs.length})
              </span>
            </div>

            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
                gap: 20,
              }}
            >
              {filteredDocs.map((doc) => (
                <DocumentCard
                  key={doc.doc_id}
                  doc={doc}
                  selected={selectedIds.has(doc.doc_id)}
                  onOpen={() => navigate(`/annotate/${doc.doc_id}`)}
                  onToggleSelect={() => toggleSelect(doc.doc_id)}
                  onDelete={() => handleDelete(doc.doc_id)}
                  onMarkChecked={() => handleMarkChecked(doc.doc_id)}
                  onMarkReviewed={() => handleMarkReviewed(doc.doc_id)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        justifyContent: "center",
        padding: "80px 40px",
        background: "var(--surface)",
        borderRadius: "var(--radius-lg)",
        border: "1px dashed var(--surface-border)",
      }}
    >
      <div
        style={{
          width: 56,
          height: 56,
          borderRadius: "var(--radius-md)",
          background: "var(--accent-soft)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: 24,
          marginBottom: 16,
        }}
      >
        +
      </div>
      <h3
        style={{
          fontFamily: "var(--font-heading)",
          fontSize: 16,
          fontWeight: 600,
          marginBottom: 8,
        }}
      >
        No documents yet
      </h3>
      <p
        style={{
          color: "var(--text-muted)",
          fontSize: 14,
          textAlign: "center",
          maxWidth: 320,
        }}
      >
        Import a PDF to start annotating financial documents with bounding boxes
        and structured data extraction.
      </p>
    </div>
  );
}

function LoadingGrid() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
        gap: 20,
      }}
    >
      {Array.from({ length: 6 }).map((_, i) => (
        <div
          key={i}
          style={{
            height: 240,
            borderRadius: "var(--radius-lg)",
            background: "var(--surface-alt)",
            border: "1px solid var(--surface-border)",
            animation: "pulse 1.5s ease infinite",
          }}
        />
      ))}
    </div>
  );
}

function DocumentCard({
  doc,
  selected,
  onOpen,
  onToggleSelect,
  onDelete,
  onMarkChecked,
  onMarkReviewed,
}: {
  doc: WorkspaceDocument;
  selected: boolean;
  onOpen: () => void;
  onToggleSelect: () => void;
  onDelete: () => void;
  onMarkChecked: () => void;
  onMarkReviewed: () => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);

  const statusColor = doc.reviewed
    ? "var(--ok)"
    : doc.checked
      ? "var(--accent)"
      : doc.has_annotations
        ? "var(--warn)"
        : "var(--text-soft)";

  const statusLabel = doc.reviewed
    ? "Reviewed"
    : doc.checked
      ? "Checked"
      : doc.has_annotations
        ? "Annotated"
        : "New";

  return (
    <div
      style={{
        borderRadius: "var(--radius-lg)",
        background: "var(--surface)",
        border: selected
          ? "2px solid var(--accent)"
          : "1px solid var(--surface-border)",
        overflow: "hidden",
        transition: "var(--transition-normal)",
        boxShadow: selected
          ? "0 0 0 3px var(--accent-soft)"
          : "0 1px 3px var(--shadow)",
        position: "relative",
      }}
      onMouseEnter={(e) => {
        if (!selected) {
          e.currentTarget.style.borderColor = "var(--accent)";
          e.currentTarget.style.transform = "translateY(-2px)";
          e.currentTarget.style.boxShadow = "0 4px 12px var(--shadow-lg)";
        }
      }}
      onMouseLeave={(e) => {
        if (!selected) {
          e.currentTarget.style.borderColor = "var(--surface-border)";
          e.currentTarget.style.transform = "translateY(0)";
          e.currentTarget.style.boxShadow = "0 1px 3px var(--shadow)";
        }
        setMenuOpen(false);
      }}
    >
      {/* Selection checkbox */}
      <div
        style={{
          position: "absolute",
          top: 10,
          left: 10,
          zIndex: 2,
        }}
      >
        <input
          type="checkbox"
          checked={selected}
          onChange={onToggleSelect}
          onClick={(e) => e.stopPropagation()}
          style={{ accentColor: "var(--accent)", width: 16, height: 16, cursor: "pointer" }}
        />
      </div>

      {/* Menu button */}
      <div
        style={{
          position: "absolute",
          top: 10,
          right: 10,
          zIndex: 2,
        }}
      >
        <button
          onClick={(e) => {
            e.stopPropagation();
            setMenuOpen(!menuOpen);
          }}
          style={{
            width: 28,
            height: 28,
            borderRadius: "var(--radius-xs)",
            background: "rgba(0,0,0,0.4)",
            backdropFilter: "blur(4px)",
            border: "none",
            color: "#fff",
            fontSize: 16,
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
          }}
        >
          {"\u22EE"}
        </button>

        {/* Dropdown menu */}
        {menuOpen && (
          <div
            style={{
              position: "absolute",
              top: 32,
              right: 0,
              width: 160,
              background: "var(--surface-raised)",
              border: "1px solid var(--surface-border)",
              borderRadius: "var(--radius-sm)",
              boxShadow: "0 8px 24px rgba(0,0,0,0.3)",
              zIndex: 10,
              overflow: "hidden",
            }}
          >
            <MenuItem
              onClick={() => {
                onOpen();
                setMenuOpen(false);
              }}
            >
              Open
            </MenuItem>
            {!doc.checked && (
              <MenuItem
                onClick={() => {
                  onMarkChecked();
                  setMenuOpen(false);
                }}
              >
                Mark Checked
              </MenuItem>
            )}
            {!doc.reviewed && (
              <MenuItem
                onClick={() => {
                  onMarkReviewed();
                  setMenuOpen(false);
                }}
              >
                Mark Reviewed
              </MenuItem>
            )}
            <MenuItem
              danger
              onClick={() => {
                onDelete();
                setMenuOpen(false);
              }}
            >
              Delete
            </MenuItem>
          </div>
        )}
      </div>

      {/* Thumbnail */}
      <div
        onClick={onOpen}
        style={{
          height: 140,
          background: "var(--surface-alt)",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          overflow: "hidden",
          cursor: "pointer",
        }}
      >
        {doc.thumbnail ? (
          <img
            src={`/api/images/${doc.doc_id}/thumbnails/${doc.thumbnail}`}
            alt={doc.doc_id}
            style={{
              width: "100%",
              height: "100%",
              objectFit: "cover",
            }}
            loading="lazy"
          />
        ) : (
          <span style={{ fontSize: 32, color: "var(--text-soft)" }}>
            {"\u2630"}
          </span>
        )}
      </div>

      {/* Info */}
      <div style={{ padding: "14px 16px" }}>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            marginBottom: 6,
          }}
        >
          <h3
            onClick={onOpen}
            style={{
              fontFamily: "var(--font-heading)",
              fontSize: 14,
              fontWeight: 600,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
              flex: 1,
              cursor: "pointer",
            }}
          >
            {doc.doc_id}
          </h3>
          <span
            style={{
              fontSize: 11,
              fontWeight: 600,
              color: statusColor,
              padding: "2px 8px",
              borderRadius: "var(--radius-pill)",
              border: `1px solid ${statusColor}`,
              marginLeft: 8,
              whiteSpace: "nowrap",
            }}
          >
            {statusLabel}
          </span>
        </div>
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <p style={{ fontSize: 12, color: "var(--text-muted)" }}>
            {doc.page_count} page{doc.page_count !== 1 ? "s" : ""}
          </p>
          {doc.fact_count != null && doc.fact_count > 0 && (
            <p
              style={{
                fontSize: 11,
                color: "var(--text-soft)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {doc.fact_count} facts
            </p>
          )}
        </div>
      </div>
    </div>
  );
}

function FilterPill({
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
        padding: "5px 14px",
        fontSize: 12,
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

function BatchActionBtn({
  onClick,
  children,
}: {
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        padding: "4px 12px",
        fontSize: 12,
        fontWeight: 600,
        borderRadius: "var(--radius-xs)",
        background: "var(--accent)",
        color: "#fff",
        border: "none",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
  );
}

function MenuItem({
  onClick,
  children,
  danger,
}: {
  onClick: () => void;
  children: React.ReactNode;
  danger?: boolean;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      style={{
        display: "block",
        width: "100%",
        padding: "8px 14px",
        fontSize: 13,
        fontWeight: 500,
        color: danger ? "var(--danger)" : "var(--text)",
        background: "transparent",
        border: "none",
        textAlign: "left",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
      onMouseEnter={(e) => {
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
