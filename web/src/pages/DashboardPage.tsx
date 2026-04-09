/** Dashboard — workspace documents grid with PDF preview, search, sort, and full stats. */

import { useState, useEffect, useCallback, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { get, post, del } from "../api/client";
import { BatchInferDialog } from "../components/dialogs/BatchInferDialog";
import type { WorkspaceDocument } from "../types/api";

type FilterKey = "all" | "new" | "annotated" | "checked" | "reviewed";
type SortKey = "recent" | "name" | "issues" | "progress";

const STATUS_COLOR: Record<string, string> = {
  "Complete":           "var(--ok)",
  "Ready":              "var(--ok)",
  "Checked":            "var(--accent)",
  "Reviewed":           "var(--ok)",
  "Annotated":          "var(--warn)",
  "In progress":        "var(--warn)",
  "Batch Complete":     "var(--accent)",
  "Needs extraction":   "var(--text-soft)",
  "Missing pages":      "var(--danger)",
  "Auto-Annotating":    "var(--accent)",
  "Batch Inferring":    "var(--accent)",
  "New":                "var(--text-soft)",
};

function statusLabel(doc: WorkspaceDocument): string {
  if (doc.reviewed) return "Reviewed";
  if (doc.checked)  return "Checked";
  return doc.status || (doc.has_annotations ? "Annotated" : "New");
}

function statusColor(doc: WorkspaceDocument): string {
  return STATUS_COLOR[statusLabel(doc)] ?? "var(--text-soft)";
}

const RECENT_VIEWS_KEY = "finetree:recent_views";

export function getRecentViews(): Record<string, number> {
  try {
    return JSON.parse(localStorage.getItem(RECENT_VIEWS_KEY) || "{}");
  } catch { return {}; }
}

export function recordView(docId: string): void {
  const views = getRecentViews();
  views[docId] = Date.now() / 1000;
  localStorage.setItem(RECENT_VIEWS_KEY, JSON.stringify(views));
}

function formatUpdated(ts: number | null): string {
  if (!ts) return "";
  const d = new Date(ts * 1000);
  const now = Date.now();
  const diff = (now - d.getTime()) / 1000;
  if (diff < 60)  return "Just now";
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
  return d.toLocaleDateString();
}

export function DashboardPage() {
  const [documents, setDocuments] = useState<WorkspaceDocument[]>([]);
  const [loading, setLoading] = useState(true);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<FilterKey>("all");
  const [sort, setSort] = useState<SortKey>("recent");
  const [search, setSearch] = useState("");
  const [batchInferOpen, setBatchInferOpen] = useState(false);
  const navigate = useNavigate();

  const refresh = useCallback(() => {
    setLoading(true);
    get<WorkspaceDocument[]>("/workspace/documents")
      .then(setDocuments)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => { refresh(); }, [refresh]);

  const filteredDocs = useMemo(() => {
    let docs = documents.filter((d) => {
      if (search) {
        const q = search.toLowerCase();
        if (!d.doc_id.toLowerCase().includes(q)) return false;
      }
      switch (filter) {
        case "new":       return !d.has_annotations && (d.annotated_page_count ?? 0) === 0;
        case "annotated": return (d.annotated_page_count ?? 0) > 0 && !d.checked;
        case "checked":   return d.checked && !d.reviewed;
        case "reviewed":  return d.reviewed;
        default:          return true;
      }
    });

    const recentViews = getRecentViews();
    docs = [...docs].sort((a, b) => {
      switch (sort) {
        case "name":     return a.doc_id.localeCompare(b.doc_id);
        case "issues":   return (b.reg_flag_count + b.warning_count) - (a.reg_flag_count + a.warning_count);
        case "progress": return b.progress_pct - a.progress_pct;
        case "recent":
        default: {
          const aRecent = Math.max(a.updated_at ?? 0, recentViews[a.doc_id] ?? 0);
          const bRecent = Math.max(b.updated_at ?? 0, recentViews[b.doc_id] ?? 0);
          return bRecent - aRecent;
        }
      }
    });
    return docs;
  }, [documents, filter, sort, search]);

  // Workspace stats
  const stats = useMemo(() => {
    const total = documents.length;
    const totalPages = documents.reduce((s, d) => s + d.page_count, 0);
    const annotatedPages = documents.reduce((s, d) => s + (d.annotated_page_count ?? 0), 0);
    const approvedPages = documents.reduce((s, d) => s + (d.approved_page_count ?? 0), 0);
    const totalFacts = documents.reduce((s, d) => s + (d.fact_count ?? 0), 0);
    const coveragePct = totalPages > 0 ? Math.round((annotatedPages / totalPages) * 100) : 0;
    const approvedPct = totalPages > 0 ? Math.round((approvedPages / totalPages) * 100) : 0;
    const totalTokens = documents.reduce((s, d) => s + (d.annotated_token_count ?? 0), 0);
    return { total, totalPages, annotatedPages, approvedPages, totalFacts, coveragePct, approvedPct, totalTokens };
  }, [documents]);

  const toggleSelect = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id); else next.add(id);
      return next;
    });
  };
  const selectAll = () => {
    if (selectedIds.size === filteredDocs.length) setSelectedIds(new Set());
    else setSelectedIds(new Set(filteredDocs.map((d) => d.doc_id)));
  };

  const handleDelete = async (docId: string) => {
    await del(`/workspace/documents/${docId}`).catch(console.error);
    refresh();
    setSelectedIds((prev) => { const n = new Set(prev); n.delete(docId); return n; });
  };
  const handleMarkChecked   = async (docId: string) => { await post(`/workspace/documents/${docId}/checked`, {}).catch(console.error); refresh(); };
  const handleMarkReviewed  = async (docId: string) => { await post(`/workspace/documents/${docId}/reviewed`, {}).catch(console.error); refresh(); };
  const handleResetApproved = async (docId: string) => { await post(`/workspace/documents/${docId}/reset-approved`, {}).catch(console.error); refresh(); };

  const handleBatchMarkChecked  = async () => { for (const id of selectedIds) await post(`/workspace/documents/${id}/checked`, {}).catch(console.error); refresh(); };
  const handleBatchMarkReviewed = async () => { for (const id of selectedIds) await post(`/workspace/documents/${id}/reviewed`, {}).catch(console.error); refresh(); };

  const handleImportPdf = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".pdf";
    input.multiple = true;
    input.onchange = async () => {
      if (!input.files) return;
      for (const file of input.files) {
        const fd = new FormData();
        fd.append("file", file);
        await fetch("/api/workspace/import-pdf", { method: "POST", body: fd }).catch(console.error);
      }
      refresh();
    };
    input.click();
  };

  const allSelected = filteredDocs.length > 0 && selectedIds.size === filteredDocs.length;

  return (
    <div style={{ height: "100%", overflowY: "auto", background: "var(--bg)" }}>

      {/* Header */}
      <div style={{ padding: "28px 40px 20px", borderBottom: "1px solid var(--surface-border)", background: "var(--surface)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 16 }}>
          <div>
            <h1 style={{ fontFamily: "var(--font-heading)", fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 3 }}>
              Projects
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              {loading ? "Loading..." : `${stats.total} document${stats.total !== 1 ? "s" : ""} · ${stats.annotatedPages}/${stats.totalPages} pages annotated · ${stats.approvedPages}/${stats.totalPages} pages approved`}
            </p>
          </div>
          <button onClick={handleImportPdf} style={primaryBtnStyle}>
            + Import PDF
          </button>
        </div>

        {/* Stats row */}
        {!loading && stats.total > 0 && (
          <div style={{ display: "flex", gap: 24, marginBottom: 16, flexWrap: "wrap" }}>
            <Stat label="Coverage" value={`${stats.coveragePct}%`} color="var(--accent)" />
            <Stat label="% Approved Pages" value={`${stats.approvedPct}%`} color="var(--ok)" />
            <Stat label="Facts"    value={stats.totalFacts.toLocaleString()} />
            <Stat label="Tokens"   value={`${(stats.totalTokens / 1000).toFixed(1)}k`} />
            <Stat label="Approved Pages" value={`${stats.approvedPages} / ${stats.totalPages}`} />
          </div>
        )}

        {/* Controls row */}
        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          {/* Filter pills */}
          {(["all", "new", "annotated", "checked", "reviewed"] as const).map((f) => (
            <FilterPill key={f} active={filter === f} onClick={() => setFilter(f)}>
              {f.charAt(0).toUpperCase() + f.slice(1)}
            </FilterPill>
          ))}

          <div style={{ flex: 1, minWidth: 24 }} />

          {/* Search */}
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search documents…"
            style={{
              padding: "6px 12px", fontSize: 13,
              background: "var(--surface-alt)",
              border: "1px solid var(--surface-border)",
              borderRadius: "var(--radius-xs)",
              color: "var(--text)", outline: "none", width: 200,
            }}
            onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
            onBlur={(e)  => (e.currentTarget.style.borderColor = "var(--surface-border)")}
          />

          {/* Sort */}
          <select
            value={sort}
            onChange={(e) => setSort(e.target.value as SortKey)}
            style={{
              padding: "6px 10px", fontSize: 12, fontWeight: 600,
              background: "var(--surface-alt)",
              border: "1px solid var(--surface-border)",
              borderRadius: "var(--radius-xs)",
              color: "var(--text)", outline: "none", cursor: "pointer",
            }}
          >
            <option value="recent">Recent</option>
            <option value="name">Name</option>
            <option value="issues">Issues ↓</option>
            <option value="progress">Progress ↓</option>
          </select>
        </div>
      </div>

      {/* Batch action bar */}
      {selectedIds.size > 0 && (
        <div style={{ padding: "10px 40px", background: "var(--accent-soft)", borderBottom: "1px solid var(--surface-border)", display: "flex", alignItems: "center", gap: 12, fontSize: 13 }}>
          <span style={{ fontWeight: 600, color: "var(--accent)" }}>{selectedIds.size} selected</span>
          <BatchBtn primary onClick={() => setBatchInferOpen(true)}>
            Batch Infer ({selectedIds.size})
          </BatchBtn>
          <BatchBtn onClick={handleBatchMarkChecked}>Mark Checked</BatchBtn>
          <BatchBtn onClick={handleBatchMarkReviewed}>Mark Reviewed</BatchBtn>
          <button onClick={() => setSelectedIds(new Set())} style={{ marginLeft: "auto", fontSize: 12, color: "var(--text-muted)", background: "transparent", border: "none", cursor: "pointer", textDecoration: "underline" }}>
            Clear
          </button>
        </div>
      )}

      {batchInferOpen && (
        <BatchInferDialog
          docIds={[...selectedIds]}
          onClose={(didRun) => {
            setBatchInferOpen(false);
            if (didRun) refresh();
          }}
        />
      )}

      {/* Content */}
      <div style={{ padding: "24px 40px 40px" }}>
        {loading ? (
          <LoadingGrid />
        ) : filteredDocs.length === 0 ? (
          documents.length === 0
            ? <EmptyState onImport={handleImportPdf} />
            : <div style={{ padding: 40, textAlign: "center", color: "var(--text-muted)", fontSize: 14 }}>No documents match the current filter.</div>
        ) : (
          <>
            <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 16, fontSize: 12, color: "var(--text-muted)" }}>
              <input type="checkbox" checked={allSelected} onChange={selectAll} style={{ accentColor: "var(--accent)" }} />
              <span>{allSelected ? "Deselect all" : "Select all"} ({filteredDocs.length})</span>
            </div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 16 }}>
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
                  onResetApproved={() => handleResetApproved(doc.doc_id)}
                />
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  );
}

// ── DocumentCard ──────────────────────────────────────────────────

function DocumentCard({
  doc, selected, onOpen, onToggleSelect, onDelete, onMarkChecked, onMarkReviewed, onResetApproved,
}: {
  doc: WorkspaceDocument;
  selected: boolean;
  onOpen: () => void;
  onToggleSelect: () => void;
  onDelete: () => void;
  onMarkChecked: () => void;
  onMarkReviewed: () => void;
  onResetApproved: () => void;
}) {
  const [menuOpen, setMenuOpen] = useState(false);
  const sColor = statusColor(doc);
  const sLabel = statusLabel(doc);
  const hasIssues = doc.reg_flag_count > 0 || doc.warning_count > 0;
  const thumbSrc = doc.thumbnail_name
    ? `/api/images/${doc.doc_id}/thumbnails/${doc.thumbnail_name}`
    : null;
  const approvedPct = doc.page_count > 0
    ? Math.round(((doc.approved_page_count ?? 0) / doc.page_count) * 100)
    : 0;

  return (
    <div
      style={{
        borderRadius: "var(--radius-lg)",
        background: "var(--surface)",
        border: selected ? "2px solid var(--accent)" : "1px solid var(--surface-border)",
        overflow: "hidden",
        transition: "var(--transition-normal)",
        boxShadow: selected ? "0 0 0 3px var(--accent-soft)" : "0 1px 3px var(--shadow)",
        position: "relative",
        display: "flex",
        flexDirection: "column",
      }}
      onMouseEnter={(e) => {
        if (!selected) {
          e.currentTarget.style.borderColor = "var(--accent)";
          e.currentTarget.style.boxShadow = "0 4px 14px var(--shadow-lg)";
        }
      }}
      onMouseLeave={(e) => {
        if (!selected) {
          e.currentTarget.style.borderColor = "var(--surface-border)";
          e.currentTarget.style.boxShadow = "0 1px 3px var(--shadow)";
        }
        setMenuOpen(false);
      }}
    >
      {/* Checkbox */}
      <div style={{ position: "absolute", top: 10, left: 10, zIndex: 2 }}>
        <input type="checkbox" checked={selected} onChange={onToggleSelect} onClick={(e) => e.stopPropagation()}
          style={{ accentColor: "var(--accent)", width: 16, height: 16, cursor: "pointer" }} />
      </div>

      {/* Menu */}
      <div style={{ position: "absolute", top: 10, right: 10, zIndex: 2 }}>
        <button
          onClick={(e) => { e.stopPropagation(); setMenuOpen(!menuOpen); }}
          style={{ width: 28, height: 28, borderRadius: "var(--radius-xs)", background: "rgba(0,0,0,0.45)", backdropFilter: "blur(4px)", border: "none", color: "#fff", fontSize: 16, cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center" }}
        >
          {"\u22EE"}
        </button>
        {menuOpen && (
          <div style={{ position: "absolute", top: 32, right: 0, width: 172, background: "var(--surface-raised)", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-sm)", boxShadow: "0 8px 24px rgba(0,0,0,0.3)", zIndex: 10, overflow: "hidden" }}>
            <MenuItem onClick={() => { onOpen(); setMenuOpen(false); }}>Open</MenuItem>
            {!doc.checked   && <MenuItem onClick={() => { onMarkChecked();  setMenuOpen(false); }}>Mark Checked</MenuItem>}
            {!doc.reviewed  && <MenuItem onClick={() => { onMarkReviewed(); setMenuOpen(false); }}>Mark Reviewed</MenuItem>}
            <MenuItem onClick={() => { onResetApproved(); setMenuOpen(false); }}>Reset Approved</MenuItem>
            <MenuItem danger onClick={() => { onDelete(); setMenuOpen(false); }}>Delete</MenuItem>
          </div>
        )}
      </div>

      {/* Thumbnail */}
      <div
        onClick={onOpen}
        style={{ height: 130, background: "#111", overflow: "hidden", cursor: "pointer", flexShrink: 0, display: "flex", alignItems: "center", justifyContent: "center", position: "relative" }}
      >
        {thumbSrc ? (
          <img src={thumbSrc} alt={doc.doc_id} style={{ width: "100%", height: "100%", objectFit: "contain", padding: "4px", boxSizing: "border-box" }} loading="lazy" />
        ) : (
          <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, color: "var(--text-soft)" }}>
            <span style={{ fontSize: 36 }}>{"\uD83D\uDCC4"}</span>
            <span style={{ fontSize: 11, fontFamily: "var(--font-mono)" }}>PDF</span>
          </div>
        )}
        {/* Progress bar overlay at bottom of thumbnail */}
        {doc.page_count > 0 && (
          <div style={{ position: "absolute", bottom: 0, left: 0, right: 0, height: 3, background: "rgba(0,0,0,0.2)" }}>
            <div style={{ height: "100%", width: `${doc.progress_pct}%`, background: doc.progress_pct >= 100 ? "var(--ok)" : "var(--accent)", transition: "width 0.3s ease" }} />
          </div>
        )}
      </div>

      {/* Info body */}
      <div style={{ padding: "10px 10px", display: "flex", flexDirection: "column", gap: 5, flex: 1 }}>

        {/* Title + status */}
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 6 }}>
          <h3
            onClick={onOpen}
            title={doc.doc_id}
            style={{ fontFamily: "var(--font-heading)", fontSize: 13, fontWeight: 600, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap", flex: 1, cursor: "pointer" }}
          >
            {doc.doc_id}
          </h3>
          <span style={{ fontSize: 10, fontWeight: 700, color: sColor, padding: "2px 7px", borderRadius: "var(--radius-pill)", border: `1px solid ${sColor}`, whiteSpace: "nowrap", flexShrink: 0 }}>
            {sLabel}
          </span>
        </div>

        {/* Approval / fact counts */}
        <div style={{ display: "flex", gap: 10, fontSize: 12, color: "var(--text-muted)" }}>
          <span>Approved {doc.approved_page_count ?? 0}/{doc.page_count} pages</span>
          <span>{approvedPct}% approved</span>
          {doc.fact_count > 0 && <span>{doc.fact_count.toLocaleString()} facts</span>}
          {doc.annotated_token_count > 0 && (
            <span style={{ fontFamily: "var(--font-mono)", fontSize: 11, color: "var(--text-soft)" }}>
              {(doc.annotated_token_count / 1000).toFixed(1)}k tok
            </span>
          )}
        </div>

        {/* Issue badges */}
        {hasIssues && (
          <div style={{ display: "flex", gap: 6 }}>
            {doc.reg_flag_count > 0 && (
              <span style={{ fontSize: 10, fontWeight: 700, color: "var(--danger)", background: "rgba(239,68,68,0.1)", padding: "1px 7px", borderRadius: "var(--radius-pill)" }}>
                {doc.reg_flag_count} flag{doc.reg_flag_count !== 1 ? "s" : ""}
              </span>
            )}
            {doc.warning_count > 0 && (
              <span style={{ fontSize: 10, fontWeight: 700, color: "var(--warn)", background: "rgba(245,158,11,0.1)", padding: "1px 7px", borderRadius: "var(--radius-pill)" }}>
                {doc.warning_count} warn{doc.warning_count !== 1 ? "s" : ""}
              </span>
            )}
          </div>
        )}

        {/* Footer: last updated + open button */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginTop: 2 }}>
          <span style={{ fontSize: 11, color: "var(--text-soft)" }}>
            {formatUpdated(Math.max(doc.updated_at ?? 0, getRecentViews()[doc.doc_id] ?? 0) || null)}
          </span>
          <button
            onClick={onOpen}
            style={{ fontSize: 12, fontWeight: 600, color: "var(--accent)", background: "var(--accent-soft)", border: "none", borderRadius: "var(--radius-xs)", padding: "4px 12px", cursor: "pointer" }}
          >
            Open →
          </button>
        </div>
      </div>
    </div>
  );
}

// ── helpers ───────────────────────────────────────────────────────

function Stat({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <span style={{ fontSize: 18, fontWeight: 700, fontFamily: "var(--font-mono)", color: color ?? "var(--text)" }}>{value}</span>
      <span style={{ fontSize: 11, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.05em" }}>{label}</span>
    </div>
  );
}

function EmptyState({ onImport }: { onImport: () => void }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", padding: "80px 40px", background: "var(--surface)", borderRadius: "var(--radius-lg)", border: "1px dashed var(--surface-border)" }}>
      <div style={{ width: 56, height: 56, borderRadius: "var(--radius-md)", background: "var(--accent-soft)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 24, marginBottom: 16 }}>+</div>
      <h3 style={{ fontFamily: "var(--font-heading)", fontSize: 16, fontWeight: 600, marginBottom: 8 }}>No documents yet</h3>
      <p style={{ color: "var(--text-muted)", fontSize: 14, textAlign: "center", maxWidth: 320, marginBottom: 20 }}>
        Import a PDF to start annotating financial documents with bounding boxes and structured data extraction.
      </p>
      <button onClick={onImport} style={primaryBtnStyle}>Import PDF</button>
    </div>
  );
}

function LoadingGrid() {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(6, 1fr)", gap: 16 }}>
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} style={{ height: 260, borderRadius: "var(--radius-lg)", background: "var(--surface-alt)", border: "1px solid var(--surface-border)", animation: "pulse 1.5s ease infinite" }} />
      ))}
    </div>
  );
}

function FilterPill({ active, onClick, children }: { active: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <button onClick={onClick} style={{ padding: "5px 14px", fontSize: 12, fontWeight: 600, borderRadius: "var(--radius-pill)", border: active ? "1px solid var(--accent)" : "1px solid var(--surface-border)", background: active ? "var(--accent-soft)" : "transparent", color: active ? "var(--accent)" : "var(--text-muted)", cursor: "pointer", transition: "var(--transition-fast)" }}>
      {children}
    </button>
  );
}

function BatchBtn({ onClick, children, primary }: { onClick: () => void; children: React.ReactNode; primary?: boolean }) {
  return (
    <button onClick={onClick} style={{
      padding: "4px 12px", fontSize: 12, fontWeight: 600,
      borderRadius: "var(--radius-xs)",
      background: primary ? "var(--accent-strong, var(--accent))" : "var(--accent)",
      color: "#fff", border: primary ? "2px solid var(--accent)" : "none",
      cursor: "pointer",
    }}>
      {children}
    </button>
  );
}

function MenuItem({ onClick, children, danger }: { onClick: () => void; children: React.ReactNode; danger?: boolean }) {
  return (
    <button
      onClick={(e) => { e.stopPropagation(); onClick(); }}
      style={{ display: "block", width: "100%", padding: "8px 14px", fontSize: 13, fontWeight: 500, color: danger ? "var(--danger)" : "var(--text)", background: "transparent", border: "none", textAlign: "left", cursor: "pointer" }}
      onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-alt)"; }}
      onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
    >
      {children}
    </button>
  );
}

const primaryBtnStyle: React.CSSProperties = {
  padding: "9px 20px", background: "var(--accent)", color: "#fff",
  borderRadius: "var(--radius-sm)", fontSize: 13, fontWeight: 600,
  border: "none", cursor: "pointer",
};
