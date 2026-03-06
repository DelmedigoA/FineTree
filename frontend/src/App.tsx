import { startTransition, useDeferredValue, useEffect, useRef, useState, type ReactNode } from "react";
import { BrowserRouter, Route, Routes, useNavigate, useParams } from "react-router-dom";

import { AnnotationCanvas } from "./AnnotationCanvas";
import {
  cloneDocument,
  dehydrateDocument,
  duplicateSelectedFacts,
  hydrateDocument,
  mergeImportedPayload,
  pageHasAnnotation,
  normalizeDocumentMeta,
  normalizeFact,
  sortFactsForPage
} from "./documentState";
import {
  extractPage,
  getAppConfig,
  getDocument,
  getDocuments,
  prepareDocument,
  saveDocument,
  toggleChecked,
  toggleReviewed,
  uploadPdf,
  validateDocument
} from "./api";
import type {
  ApiAppConfig,
  ApiDocumentDetail,
  ApiDocumentMeta,
  ApiDocumentValidateResponse,
  ApiFact,
  ApiWorkspaceDocumentSummary,
  DocumentDraft,
  FactDraft,
  PageDraft
} from "./types";

type ExtractionDialogState = {
  provider: "gemini" | "qwen";
  prompt: string;
  model: string;
  fewShotEnabled: boolean;
  fewShotPreset: string;
  enableThinking: boolean;
};

function readFileAsBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(reader.error);
    reader.onload = () => {
      const value = String(reader.result ?? "");
      const [, base64] = value.split(",", 2);
      resolve(base64 ?? "");
    };
    reader.readAsDataURL(file);
  });
}

function AppShell() {
  const [config, setConfig] = useState<ApiAppConfig | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    void getAppConfig().then((nextConfig) => {
      setConfig(nextConfig);
      if (nextConfig.startup_doc_id) {
        startTransition(() => navigate(`/documents/${nextConfig.startup_doc_id}`, { replace: true }));
      }
    });
  }, [navigate]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div>
          <p className="eyebrow">FineTree</p>
          <h1>Financial Annotation Workspace</h1>
        </div>
        <div className="topbar__note">
          <span>React UI</span>
          <span>Python API</span>
          <span>{config?.data_root ? `Data ${config.data_root}` : "Workspace"}</span>
        </div>
      </header>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/documents/:docId" element={<DocumentPage />} />
      </Routes>
    </div>
  );
}

function DashboardPage() {
  const navigate = useNavigate();
  const [documents, setDocuments] = useState<ApiWorkspaceDocumentSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("All");
  const [uploading, setUploading] = useState(false);
  const deferredSearch = useDeferredValue(search);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  useEffect(() => {
    void loadDocuments();
  }, []);

  async function loadDocuments() {
    setLoading(true);
    setError(null);
    try {
      setDocuments(await getDocuments());
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(file: File) {
    setUploading(true);
    setError(null);
    try {
      const content = await readFileAsBase64(file);
      const detail = await uploadPdf(file.name, content);
      await loadDocuments();
      startTransition(() => navigate(`/documents/${detail.doc_id}`));
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  const filtered = documents.filter((document) => {
    const query = deferredSearch.trim().toLowerCase();
    if (statusFilter !== "All" && document.status !== statusFilter) {
      return false;
    }
    if (!query) {
      return true;
    }
    return (
      document.doc_id.toLowerCase().includes(query) ||
      String(document.source_pdf ?? "").toLowerCase().includes(query)
    );
  });

  const stats = {
    total: documents.length,
    complete: documents.filter((document) => document.status === "Complete").length,
    attention: documents.filter((document) => document.reg_flag_count > 0).length,
    reviewed: documents.filter((document) => document.reviewed).length
  };

  return (
    <section className="dashboard">
      <div className="hero-card">
        <div>
          <p className="eyebrow">Workspace Dashboard</p>
          <h2>Open, prepare, review, and continue annotating without the Qt shell.</h2>
          <p className="hero-card__copy">
            The file contract stays in Python. The browser now talks to the same schema and workspace services the desktop app used.
          </p>
        </div>
        <div className="hero-card__actions">
          <button className="primary-button" onClick={() => fileInputRef.current?.click()} disabled={uploading}>
            {uploading ? "Importing..." : "Import PDF"}
          </button>
          <button className="ghost-button" onClick={() => void loadDocuments()} disabled={loading}>
            Refresh
          </button>
          <input
            ref={fileInputRef}
            hidden
            type="file"
            accept="application/pdf"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) {
                void handleUpload(file);
              }
            }}
          />
        </div>
      </div>

      <div className="stats-grid">
        <StatCard label="Documents" value={String(stats.total)} tone="cool" />
        <StatCard label="Complete" value={String(stats.complete)} tone="mint" />
        <StatCard label="With Flags" value={String(stats.attention)} tone="amber" />
        <StatCard label="Reviewed" value={String(stats.reviewed)} tone="slate" />
      </div>

      <div className="filters-bar">
        <input
          className="surface-input"
          placeholder="Search document id or source PDF"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select className="surface-input" value={statusFilter} onChange={(event) => setStatusFilter(event.target.value)}>
          {["All", "Ready", "In progress", "Complete", "Needs extraction", "Missing pages"].map((status) => (
            <option key={status}>{status}</option>
          ))}
        </select>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}
      {loading ? <div className="empty-state">Loading workspace…</div> : null}
      {!loading && filtered.length === 0 ? <div className="empty-state">No documents match the current filter.</div> : null}

      <div className="document-grid">
        {filtered.map((document) => (
          <article key={document.doc_id} className="document-card">
            <div className="document-card__head">
              <div>
                <p className="eyebrow">Managed document</p>
                <h3>{document.doc_id}</h3>
              </div>
              <span className={`status-pill status-pill--${toneForStatus(document.status)}`}>{document.status}</span>
            </div>
            <p className="document-card__path">{document.source_pdf ?? document.images_dir}</p>
            <div className="document-card__stats">
              <span>{document.annotated_page_count}/{document.page_count} pages</span>
              <span>{document.reg_flag_count} flags</span>
              <span>{document.warning_count} warnings</span>
            </div>
            <div className="progress-track">
              <div className="progress-track__fill" style={{ width: `${document.progress_pct}%` }} />
            </div>
            <div className="document-card__actions">
              <button
                className="primary-button"
                onClick={async () => {
                  if (document.status === "Needs extraction") {
                    const detail = await prepareDocument(document.doc_id);
                    await loadDocuments();
                    startTransition(() => navigate(`/documents/${detail.doc_id}`));
                    return;
                  }
                  startTransition(() => navigate(`/documents/${document.doc_id}`));
                }}
              >
                {document.status === "Needs extraction" ? "Prepare" : document.progress_pct === 100 ? "Review" : "Open"}
              </button>
              <button
                className="ghost-button"
                onClick={async () => {
                  await toggleChecked(document.doc_id, !document.checked);
                  await loadDocuments();
                }}
              >
                {document.checked ? "Checked" : "Mark Checked"}
              </button>
              <button
                className="ghost-button"
                disabled={!document.checked || document.reg_flag_count > 0}
                onClick={async () => {
                  await toggleReviewed(document.doc_id, !document.reviewed);
                  await loadDocuments();
                }}
              >
                {document.reviewed ? "Reviewed" : "Mark Reviewed"}
              </button>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

function DocumentPage() {
  const navigate = useNavigate();
  const { docId = "" } = useParams();
  const [documentDetail, setDocumentDetail] = useState<ApiDocumentDetail | null>(null);
  const [draft, setDraft] = useState<DocumentDraft | null>(null);
  const [validation, setValidation] = useState<ApiDocumentValidateResponse | null>(null);
  const [selectedPageImage, setSelectedPageImage] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [mode, setMode] = useState<"select" | "draw">("select");
  const [zoom, setZoom] = useState(1);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showImportDialog, setShowImportDialog] = useState(false);
  const [showExtractionDialog, setShowExtractionDialog] = useState(false);
  const [importText, setImportText] = useState("");
  const [importNormalized, setImportNormalized] = useState(true);
  const [extracting, setExtracting] = useState(false);
  const [extractionConfig, setExtractionConfig] = useState<ExtractionDialogState>({
    provider: "gemini",
    prompt: "",
    model: "",
    fewShotEnabled: false,
    fewShotPreset: "classic_4",
    enableThinking: true
  });
  const historyRef = useRef<DocumentDraft[]>([]);
  const historyIndexRef = useRef(-1);
  const [historyVersion, setHistoryVersion] = useState(0);
  const savedSnapshotRef = useRef("");

  useEffect(() => {
    void load();
  }, [docId]);

  useEffect(() => {
    if (!draft) {
      return undefined;
    }
    const serialized = JSON.stringify(dehydrateDocument(draft));
    const timeout = window.setTimeout(() => {
      void validateDocument(docId, JSON.parse(serialized) as ReturnType<typeof dehydrateDocument>).then(setValidation).catch(() => {
        // Validation should not block editing. Keep the last good response.
      });
    }, 280);
    return () => window.clearTimeout(timeout);
  }, [docId, draft]);

  useEffect(() => {
    const dirty = isDirty();
    const handler = (event: BeforeUnloadEvent) => {
      if (!dirty) {
        return;
      }
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", handler);
    return () => window.removeEventListener("beforeunload", handler);
  });

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (!draft || !selectedPage) {
        return;
      }
      const active = window.document.activeElement;
      const isEditable =
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement ||
        active instanceof HTMLSelectElement ||
        Boolean(active?.getAttribute("contenteditable"));
      const mod = event.metaKey || event.ctrlKey;
      if (mod && event.key.toLowerCase() === "s") {
        event.preventDefault();
        void handleSave();
        return;
      }
      if (mod && event.key.toLowerCase() === "d") {
        event.preventDefault();
        duplicateSelection();
        return;
      }
      if (mod && event.key.toLowerCase() === "z" && !event.shiftKey) {
        event.preventDefault();
        undo();
        return;
      }
      if ((mod && event.key.toLowerCase() === "y") || (mod && event.shiftKey && event.key.toLowerCase() === "z")) {
        event.preventDefault();
        redo();
        return;
      }
      if ((event.key === "Delete" || event.key === "Backspace") && !isEditable) {
        event.preventDefault();
        deleteSelection();
        return;
      }
      if (event.key === "ArrowLeft" || event.key === "ArrowRight" || event.key === "ArrowUp" || event.key === "ArrowDown") {
        if (isEditable || selectedIds.length === 0) {
          return;
        }
        event.preventDefault();
        const step = event.shiftKey ? 10 : 1;
        const delta =
          event.key === "ArrowLeft"
            ? { dx: -step, dy: 0 }
            : event.key === "ArrowRight"
              ? { dx: step, dy: 0 }
              : event.key === "ArrowUp"
                ? { dx: 0, dy: -step }
                : { dx: 0, dy: step };
        applyToCurrentPage((page) => ({
          ...page,
          facts: page.facts.map((fact) =>
            selectedIds.includes(fact.id)
              ? {
                  ...fact,
                  bbox: {
                    ...fact.bbox,
                    x: Math.max(0, Math.min(page.width - fact.bbox.w, fact.bbox.x + delta.dx)),
                    y: Math.max(0, Math.min(page.height - fact.bbox.h, fact.bbox.y + delta.dy))
                  }
                }
              : fact
          )
        }));
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [docId, draft, selectedIds]);

  async function load() {
    setLoading(true);
    setError(null);
    try {
      const detail = await getDocument(docId);
      const hydrated = hydrateDocument(detail);
      setDocumentDetail(detail);
      setDraft(hydrated);
      setValidation({
        issue_summary: detail.issue_summary,
        save_warnings: detail.save_warnings
      });
      setSelectedPageImage((current) => current ?? detail.pages[0]?.image ?? null);
      setSelectedIds([]);
      setMode("select");
      setZoom(Math.min(1, 960 / Math.max(detail.pages[0]?.width ?? 960, 1)));
      savedSnapshotRef.current = JSON.stringify(dehydrateDocument(hydrated));
      historyRef.current = [cloneDocument(hydrated)];
      historyIndexRef.current = 0;
      setHistoryVersion((version) => version + 1);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setLoading(false);
    }
  }

  function updateDraft(nextDraft: DocumentDraft, recordHistory: boolean) {
    setDraft(nextDraft);
    if (!selectedPageImage && nextDraft.pages[0]) {
      setSelectedPageImage(nextDraft.pages[0].image);
    }
    if (recordHistory) {
      const trimmed = historyRef.current.slice(0, historyIndexRef.current + 1);
      trimmed.push(cloneDocument(nextDraft));
      historyRef.current = trimmed;
      historyIndexRef.current = trimmed.length - 1;
      setHistoryVersion((version) => version + 1);
    }
  }

  function applyToCurrentPage(updater: (page: PageDraft) => PageDraft, recordHistory = true) {
    if (!draft || !selectedPageImage) {
      return;
    }
      const nextDraft = cloneDocument(draft);
      nextDraft.pages = nextDraft.pages.map((page) => {
        if (page.image !== selectedPageImage) {
          return page;
        }
        const updated = updater(page);
        return {
          ...updated,
          annotated: pageHasAnnotation(updated.meta, updated.facts),
          facts: sortFactsForPage(updated.facts, nextDraft.document_meta, updated.meta)
        };
      });
    updateDraft(nextDraft, recordHistory);
  }

  function applyToSelectedFacts(updater: (fact: FactDraft) => FactDraft) {
    applyToCurrentPage((page) => ({
      ...page,
      facts: page.facts.map((fact) => (selectedIds.includes(fact.id) ? updater(fact) : fact))
    }));
  }

  function undo() {
    if (historyIndexRef.current <= 0) {
      return;
    }
    historyIndexRef.current -= 1;
    const nextDraft = cloneDocument(historyRef.current[historyIndexRef.current]);
    setDraft(nextDraft);
    setHistoryVersion((version) => version + 1);
  }

  function redo() {
    if (historyIndexRef.current >= historyRef.current.length - 1) {
      return;
    }
    historyIndexRef.current += 1;
    const nextDraft = cloneDocument(historyRef.current[historyIndexRef.current]);
    setDraft(nextDraft);
    setHistoryVersion((version) => version + 1);
  }

  function isDirty(): boolean {
    if (!draft) {
      return false;
    }
    return JSON.stringify(dehydrateDocument(draft)) !== savedSnapshotRef.current;
  }

  async function handleSave() {
    if (!draft) {
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const response = await saveDocument(docId, dehydrateDocument(draft));
      const hydrated = hydrateDocument(response.document);
      setDocumentDetail(response.document);
      setDraft(hydrated);
      setValidation({
        issue_summary: response.document.issue_summary,
        save_warnings: response.save_warnings
      });
      savedSnapshotRef.current = JSON.stringify(dehydrateDocument(hydrated));
      historyRef.current = [cloneDocument(hydrated)];
      historyIndexRef.current = 0;
      setHistoryVersion((version) => version + 1);
    } catch (nextError) {
      setError(nextError instanceof Error ? nextError.message : String(nextError));
    } finally {
      setSaving(false);
    }
  }

  function deleteSelection() {
    applyToCurrentPage((page) => ({
      ...page,
      facts: page.facts.filter((fact) => !selectedIds.includes(fact.id))
    }));
    setSelectedIds([]);
  }

  function duplicateSelection() {
    if (!selectedPage || !draft) {
      return;
    }
    const nextPage = duplicateSelectedFacts(selectedPage, selectedIds, draft.document_meta);
    applyToCurrentPage(() => nextPage);
    const newIds = nextPage.facts.slice(-selectedIds.length).map((fact) => fact.id);
    setSelectedIds(newIds);
  }

  const selectedPage = draft?.pages.find((page) => page.image === selectedPageImage) ?? draft?.pages[0] ?? null;
  const selectedFacts = selectedPage?.facts.filter((fact) => selectedIds.includes(fact.id)) ?? [];
  const currentIssues = selectedPage ? validation?.issue_summary.page_summaries[selectedPage.image] ?? selectedPage.issue_summary : null;
  const sharedPath =
    selectedFacts.length === 0
      ? ""
      : selectedFacts.every((fact) => fact.path.join("\n") === selectedFacts[0].path.join("\n"))
        ? selectedFacts[0].path.join("\n")
        : "";

  const factFields = selectedFacts.length === 1 ? selectedFacts[0] : selectedFacts[0] ?? null;

  if (loading) {
    return <div className="empty-state">Loading document…</div>;
  }

  if (!draft || !documentDetail || !selectedPage) {
    return (
      <div className="empty-state">
        <p>{error ?? "Document unavailable."}</p>
        <button className="ghost-button" onClick={() => navigate("/")}>
          Back to dashboard
        </button>
      </div>
    );
  }

  return (
    <section className="document-layout">
      <div className="document-toolbar">
        <div>
          <p className="eyebrow">Annotator</p>
          <h2>{draft.doc_id}</h2>
          <p className="document-toolbar__meta">{draft.annotations_path}</p>
        </div>
        <div className="document-toolbar__actions">
          <button
            className="ghost-button"
            onClick={() => {
              if (isDirty() && !window.confirm("Discard unsaved changes and go back to the dashboard?")) {
                return;
              }
              navigate("/");
            }}
          >
            Dashboard
          </button>
          <button className="ghost-button" onClick={() => setShowImportDialog(true)}>
            Import JSON
          </button>
          <button className="ghost-button" onClick={() => setShowExtractionDialog(true)}>
            Extract
          </button>
          <button
            className="ghost-button"
            onClick={() => {
              const entityName = selectedPage.meta.entity_name ?? "";
              if (!entityName) {
                setError("Enter an entity_name on the current page before applying it to other pages.");
                return;
              }
              const overwrite = window.confirm("Apply entity_name to every page instead of only missing pages?");
              const nextDraft = cloneDocument(draft);
              updateDraft(
                {
                  ...nextDraft,
                  pages: nextDraft.pages.map((page) =>
                    page.meta.entity_name && !overwrite
                      ? page
                      : {
                          ...page,
                          meta: {
                            ...page.meta,
                            entity_name: entityName
                          }
                        }
                  )
                },
                true
              );
            }}
          >
            Apply Entity
          </button>
          <button className="ghost-button" onClick={() => setMode(mode === "draw" ? "select" : "draw")}>
            {mode === "draw" ? "Select Mode" : "Draw Mode"}
          </button>
          <button className="ghost-button" onClick={undo} disabled={historyIndexRef.current <= 0}>
            Undo
          </button>
          <button
            className="ghost-button"
            onClick={redo}
            disabled={historyIndexRef.current >= historyRef.current.length - 1}
          >
            Redo
          </button>
          <button className="ghost-button" onClick={duplicateSelection} disabled={selectedIds.length === 0}>
            Duplicate
          </button>
          <button className="ghost-button" onClick={deleteSelection} disabled={selectedIds.length === 0}>
            Delete
          </button>
          <div className="zoom-cluster">
            <button className="ghost-button" onClick={() => setZoom((value) => Math.max(0.35, value - 0.1))}>
              -
            </button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="ghost-button" onClick={() => setZoom((value) => Math.min(2.8, value + 0.1))}>
              +
            </button>
          </div>
          <button className="primary-button" onClick={() => void handleSave()} disabled={saving}>
            {saving ? "Saving…" : isDirty() ? "Save Changes" : "Saved"}
          </button>
        </div>
      </div>

      {error ? <div className="error-banner">{error}</div> : null}

      <div className="document-columns">
        <aside className="page-rail">
          {draft.pages.map((page) => {
            const issues = validation?.issue_summary.page_summaries[page.image] ?? page.issue_summary;
            return (
              <button
                key={page.image}
                className={`page-thumb${selectedPage.image === page.image ? " page-thumb--active" : ""}`}
                onClick={() => {
                  setSelectedPageImage(page.image);
                  setSelectedIds([]);
                }}
              >
                <img src={`/api/documents/${docId}/pages/${page.image}/image`} alt={page.image} />
                <div className="page-thumb__meta">
                  <strong>{page.image.replace("page_", "Page ").replace(".png", "")}</strong>
                  <span>{page.meta.type}</span>
                </div>
                <div className="page-thumb__badges">
                  {issues.reg_flag_count > 0 ? <span className="status-pill status-pill--danger">{issues.reg_flag_count} flags</span> : null}
                  {issues.warning_count > 0 ? <span className="status-pill status-pill--warn">{issues.warning_count} warnings</span> : null}
                  {issues.reg_flag_count === 0 && issues.warning_count === 0 && page.annotated ? (
                    <span className="status-pill status-pill--ok">OK</span>
                  ) : null}
                </div>
              </button>
            );
          })}
        </aside>

        <main className="annotation-column">
          <div className="annotation-stage-head">
            <div>
              <h3>{selectedPage.image}</h3>
              <p>{selectedPage.meta.title ?? selectedPage.meta.type}</p>
            </div>
            <div className="annotation-stage-head__stats">
              <span>{selectedPage.facts.length} facts</span>
              <span>{currentIssues?.reg_flag_count ?? 0} flags</span>
              <span>{currentIssues?.warning_count ?? 0} warnings</span>
            </div>
          </div>
          <AnnotationCanvas
            docId={docId}
            page={selectedPage}
            selectedIds={selectedIds}
            mode={mode}
            zoom={zoom}
            onSelect={setSelectedIds}
            onCommitFacts={(facts) => {
              applyToCurrentPage((page) => ({ ...page, facts }));
              setMode("select");
            }}
          />
        </main>

        <aside className="inspector">
          <section className="surface-panel">
            <p className="eyebrow">Document Meta</p>
            <div className="form-grid">
              <label>
                <span>Language</span>
                <select
                  className="surface-input"
                  value={draft.document_meta.language ?? ""}
                  onChange={(event) => {
                    const nextDraft = cloneDocument(draft);
                    nextDraft.document_meta = normalizeDocumentMeta({
                      ...nextDraft.document_meta,
                      language: event.target.value || null
                    });
                    updateDraft(nextDraft, true);
                  }}
                >
                  <option value="">Auto</option>
                  <option value="he">Hebrew</option>
                  <option value="en">English</option>
                </select>
              </label>
              <label>
                <span>Direction</span>
                <select
                  className="surface-input"
                  value={draft.document_meta.reading_direction ?? ""}
                  onChange={(event) => {
                    const nextDraft = cloneDocument(draft);
                    nextDraft.document_meta = normalizeDocumentMeta({
                      ...nextDraft.document_meta,
                      reading_direction: event.target.value || null
                    });
                    updateDraft(nextDraft, true);
                  }}
                >
                  <option value="">Auto</option>
                  <option value="rtl">RTL</option>
                  <option value="ltr">LTR</option>
                </select>
              </label>
              <label>
                <span>Company Name</span>
                <input
                  className="surface-input"
                  value={draft.document_meta.company_name ?? ""}
                  onChange={(event) => {
                    const nextDraft = cloneDocument(draft);
                    nextDraft.document_meta.company_name = event.target.value || null;
                    updateDraft(nextDraft, true);
                  }}
                />
              </label>
              <label>
                <span>Company ID</span>
                <input
                  className="surface-input"
                  value={draft.document_meta.company_id ?? ""}
                  onChange={(event) => {
                    const nextDraft = cloneDocument(draft);
                    nextDraft.document_meta.company_id = event.target.value || null;
                    updateDraft(nextDraft, true);
                  }}
                />
              </label>
              <label>
                <span>Report Year</span>
                <input
                  className="surface-input"
                  value={draft.document_meta.report_year ?? ""}
                  onChange={(event) => {
                    const nextDraft = cloneDocument(draft);
                    nextDraft.document_meta.report_year = event.target.value ? Number(event.target.value) : null;
                    updateDraft(nextDraft, true);
                  }}
                />
              </label>
            </div>
          </section>

          <section className="surface-panel">
            <p className="eyebrow">Page Meta</p>
            <div className="form-grid">
              <label>
                <span>Entity Name</span>
                <input
                  className="surface-input"
                  value={selectedPage.meta.entity_name ?? ""}
                  onChange={(event) =>
                    applyToCurrentPage((page) => ({
                      ...page,
                      meta: { ...page.meta, entity_name: event.target.value || null }
                    }))
                  }
                />
              </label>
              <label>
                <span>Page Num</span>
                <input
                  className="surface-input"
                  value={selectedPage.meta.page_num ?? ""}
                  onChange={(event) =>
                    applyToCurrentPage((page) => ({
                      ...page,
                      meta: { ...page.meta, page_num: event.target.value || null }
                    }))
                  }
                />
              </label>
              <label>
                <span>Type</span>
                <select
                  className="surface-input"
                  value={selectedPage.meta.type}
                  onChange={(event) =>
                    applyToCurrentPage((page) => ({
                      ...page,
                      meta: { ...page.meta, type: event.target.value }
                    }))
                  }
                >
                  {["balance_sheet", "income_statement", "cash_flow", "activities", "notes", "contents_page", "title", "declaration", "profits", "other"].map((pageType) => (
                    <option key={pageType}>{pageType}</option>
                  ))}
                </select>
              </label>
              <label>
                <span>Title</span>
                <input
                  className="surface-input"
                  value={selectedPage.meta.title ?? ""}
                  onChange={(event) =>
                    applyToCurrentPage((page) => ({
                      ...page,
                      meta: { ...page.meta, title: event.target.value || null }
                    }))
                  }
                />
              </label>
            </div>
          </section>

          <section className="surface-panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Facts</p>
                <h3>{selectedPage.facts.length} on this page</h3>
              </div>
              <span className={`status-pill status-pill--${selectedIds.length > 0 ? "cool" : "slate"}`}>{selectedIds.length} selected</span>
            </div>
            <div className="facts-list">
              {selectedPage.facts.map((fact, index) => (
                <button
                  key={fact.id}
                  className={`fact-row${selectedIds.includes(fact.id) ? " fact-row--selected" : ""}`}
                  onClick={() => setSelectedIds((current) => (current.includes(fact.id) ? current.filter((id) => id !== fact.id) : [fact.id]))}
                >
                  <span className="fact-row__index">#{index + 1}</span>
                  <span className="fact-row__value">{fact.value || "Empty value"}</span>
                  <span className="fact-row__path">{fact.path.join(" > ") || "No path"}</span>
                </button>
              ))}
            </div>
          </section>

          <section className="surface-panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Selection Editor</p>
                <h3>{selectedFacts.length === 1 ? "Single fact" : `${selectedFacts.length} facts`}</h3>
              </div>
            </div>
            {selectedFacts.length === 0 ? (
              <p className="helper-copy">Select one or more boxes to edit their fields here.</p>
            ) : (
              <div className="form-grid">
                <label>
                  <span>Value</span>
                  <input
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.value ?? "" : ""}
                    placeholder={selectedFacts.length > 1 ? "Apply a shared value" : ""}
                    onChange={(event) => applyToSelectedFacts((fact) => ({ ...fact, value: event.target.value }))}
                  />
                </label>
                <label>
                  <span>Comment</span>
                  <input
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.comment ?? "" : ""}
                    placeholder={selectedFacts.length > 1 ? "Apply a shared comment" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({ ...fact, comment: event.target.value || null }))
                    }
                  />
                </label>
                <label>
                  <span>Note Name</span>
                  <input
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.note_name ?? "" : ""}
                    placeholder={selectedFacts.length > 1 ? "Apply a shared note name" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({ ...fact, note_name: event.target.value || null }))
                    }
                  />
                </label>
                <label>
                  <span>Note Flag</span>
                  <select
                    className="surface-input"
                    value={selectedFacts.length === 1 ? String(Boolean(factFields?.note_flag)) : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        note_flag: event.target.value === "true"
                      }))
                    }
                  >
                    <option value="">Mixed</option>
                    <option value="true">true</option>
                    <option value="false">false</option>
                  </select>
                </label>
                <label>
                  <span>Note Num</span>
                  <input
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.note_num ?? "" : ""}
                    placeholder={selectedFacts.length > 1 ? "Apply a shared note num" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        note_num: event.target.value ? Number(event.target.value) : null
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Note Reference</span>
                  <input
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.note_reference ?? "" : ""}
                    placeholder={selectedFacts.length > 1 ? "Apply a shared note reference" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        note_reference: event.target.value || null
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Date</span>
                  <input
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.date ?? "" : ""}
                    placeholder={selectedFacts.length > 1 ? "Apply a shared date" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        date: event.target.value || null
                      }))
                    }
                  />
                </label>
                <label>
                  <span>Currency</span>
                  <select
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.currency ?? "" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        currency: event.target.value || null
                      }))
                    }
                  >
                    <option value="">Unset</option>
                    {["ILS", "USD", "EUR", "GBP"].map((currency) => (
                      <option key={currency}>{currency}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Scale</span>
                  <select
                    className="surface-input"
                    value={selectedFacts.length === 1 ? String(factFields?.scale ?? "") : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        scale: event.target.value ? Number(event.target.value) : null
                      }))
                    }
                  >
                    <option value="">Unset</option>
                    {[1, 1000, 1000000].map((scale) => (
                      <option key={scale} value={scale}>
                        {scale}
                      </option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Value Type</span>
                  <select
                    className="surface-input"
                    value={selectedFacts.length === 1 ? factFields?.value_type ?? "" : ""}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        value_type: event.target.value || null
                      }))
                    }
                  >
                    <option value="">Unset</option>
                    <option value="amount">amount</option>
                    <option value="%">%</option>
                  </select>
                </label>
                <label className="form-grid__wide">
                  <span>Path Levels</span>
                  <textarea
                    className="surface-input surface-input--textarea"
                    value={sharedPath}
                    placeholder={selectedFacts.length > 1 && !sharedPath ? "Paths differ across the selection." : "One path level per line"}
                    disabled={selectedFacts.length > 1 && !sharedPath}
                    onChange={(event) =>
                      applyToSelectedFacts((fact) => ({
                        ...fact,
                        path: event.target.value
                          .split("\n")
                          .map((item) => item.trim())
                          .filter(Boolean)
                      }))
                    }
                  />
                </label>
              </div>
            )}
          </section>

          <section className="surface-panel">
            <div className="panel-head">
              <div>
                <p className="eyebrow">Issues</p>
                <h3>{currentIssues?.reg_flag_count ?? 0} flags / {currentIssues?.warning_count ?? 0} warnings</h3>
              </div>
            </div>
            <div className="issues-list">
              {(currentIssues?.issues ?? []).length === 0 ? <p className="helper-copy">No current issues on this page.</p> : null}
              {(currentIssues?.issues ?? []).map((issue) => (
                <div key={`${issue.code}-${issue.fact_index}-${issue.message}`} className={`issue-card issue-card--${issue.severity}`}>
                  <strong>{issue.severity === "reg_flag" ? "Reg Flag" : "Warning"}</strong>
                  <p>{issue.message}</p>
                </div>
              ))}
              {validation?.save_warnings.length ? (
                <div className="warning-box">
                  <strong>Format warnings</strong>
                  {validation.save_warnings.map((warning) => (
                    <p key={`${warning.page}-${warning.fact_index}-${warning.issue_codes.join("-")}`}>
                      {warning.page ?? "page"} fact {warning.fact_index != null ? `#${warning.fact_index + 1}` : ""}: {warning.issue_codes.join(", ")}
                    </p>
                  ))}
                </div>
              ) : null}
            </div>
          </section>
        </aside>
      </div>

      {showImportDialog ? (
        <Modal
          title="Import annotations JSON"
          onClose={() => setShowImportDialog(false)}
          actions={
            <>
              <button className="ghost-button" onClick={() => setShowImportDialog(false)}>
                Cancel
              </button>
              <button
                className="primary-button"
                onClick={() => {
                  try {
                    const parsed = JSON.parse(importText);
                    const nextDraft = mergeImportedPayload(draft, parsed, selectedPage.image, importNormalized);
                    updateDraft(nextDraft, true);
                    setShowImportDialog(false);
                    setImportText("");
                  } catch (nextError) {
                    setError(nextError instanceof Error ? nextError.message : String(nextError));
                  }
                }}
              >
                Apply Import
              </button>
            </>
          }
        >
          <label className="modal-field">
            <span>Paste JSON in current-page or full-document shape.</span>
            <textarea
              className="surface-input surface-input--textarea surface-input--large"
              value={importText}
              onChange={(event) => setImportText(event.target.value)}
            />
          </label>
          <label className="checkbox-row">
            <input type="checkbox" checked={importNormalized} onChange={(event) => setImportNormalized(event.target.checked)} />
            <span>Interpret bbox as normalized 0..1000 when possible</span>
          </label>
        </Modal>
      ) : null}

      {showExtractionDialog ? (
        <Modal
          title="AI-assisted extraction"
          onClose={() => setShowExtractionDialog(false)}
          actions={
            <>
              <button className="ghost-button" onClick={() => setShowExtractionDialog(false)}>
                Cancel
              </button>
              <button
                className="primary-button"
                disabled={extracting}
                onClick={async () => {
                  setExtracting(true);
                  setError(null);
                  try {
                    const response = await extractPage(docId, selectedPage.image, {
                      provider: extractionConfig.provider,
                      prompt: extractionConfig.prompt || null,
                      model: extractionConfig.model || null,
                      few_shot_enabled: extractionConfig.fewShotEnabled,
                      few_shot_preset: extractionConfig.fewShotPreset || null,
                      enable_thinking: extractionConfig.enableThinking
                    });
                    const replaceExisting =
                      selectedPage.facts.length === 0 || window.confirm("Replace the current page facts with the extracted result?");
                    if (replaceExisting) {
                      applyToCurrentPage((page) => ({
                        ...page,
                        meta: response.extraction.meta,
                        facts: response.extraction.facts.map((fact, index) => ({
                          ...normalizeFact(fact as Partial<ApiFact>),
                          id: `${page.image}-extract-${Date.now()}-${index}`
                        }))
                      }));
                    }
                    setShowExtractionDialog(false);
                  } catch (nextError) {
                    setError(nextError instanceof Error ? nextError.message : String(nextError));
                  } finally {
                    setExtracting(false);
                  }
                }}
              >
                {extracting ? "Running…" : "Run Extraction"}
              </button>
            </>
          }
        >
          <div className="form-grid">
            <label>
              <span>Provider</span>
              <select
                className="surface-input"
                value={extractionConfig.provider}
                onChange={(event) =>
                  setExtractionConfig((current) => ({
                    ...current,
                    provider: event.target.value as "gemini" | "qwen"
                  }))
                }
              >
                <option value="gemini">Gemini</option>
                <option value="qwen">Qwen</option>
              </select>
            </label>
            <label>
              <span>Model</span>
              <input
                className="surface-input"
                value={extractionConfig.model}
                onChange={(event) =>
                  setExtractionConfig((current) => ({
                    ...current,
                    model: event.target.value
                  }))
                }
                placeholder={extractionConfig.provider === "gemini" ? "gemini-3-flash-preview" : "qwen-flash-gt"}
              />
            </label>
            <label className="form-grid__wide">
              <span>Prompt override</span>
              <textarea
                className="surface-input surface-input--textarea"
                value={extractionConfig.prompt}
                onChange={(event) =>
                  setExtractionConfig((current) => ({
                    ...current,
                    prompt: event.target.value
                  }))
                }
                placeholder="Leave empty to use prompts/extraction_prompt.txt"
              />
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={extractionConfig.fewShotEnabled}
                onChange={(event) =>
                  setExtractionConfig((current) => ({
                    ...current,
                    fewShotEnabled: event.target.checked
                  }))
                }
              />
              <span>Use few-shot examples</span>
            </label>
            <label>
              <span>Few-shot preset</span>
              <select
                className="surface-input"
                value={extractionConfig.fewShotPreset}
                onChange={(event) =>
                  setExtractionConfig((current) => ({
                    ...current,
                    fewShotPreset: event.target.value
                  }))
                }
              >
                <option value="classic_4">Classic 4-shot</option>
                <option value="extended_7">Extended 7-shot</option>
              </select>
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={extractionConfig.enableThinking}
                onChange={(event) =>
                  setExtractionConfig((current) => ({
                    ...current,
                    enableThinking: event.target.checked
                  }))
                }
              />
              <span>Enable model thinking when supported</span>
            </label>
          </div>
        </Modal>
      ) : null}
    </section>
  );
}

function Modal({
  title,
  children,
  actions,
  onClose
}: {
  title: string;
  children: ReactNode;
  actions: ReactNode;
  onClose: () => void;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="panel-head">
          <h3>{title}</h3>
          <button className="ghost-button" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="modal-body">{children}</div>
        <div className="modal-actions">{actions}</div>
      </div>
    </div>
  );
}

function StatCard({ label, value, tone }: { label: string; value: string; tone: "cool" | "mint" | "amber" | "slate" }) {
  return (
    <article className={`stat-card stat-card--${tone}`}>
      <p className="eyebrow">{label}</p>
      <h3>{value}</h3>
    </article>
  );
}

function toneForStatus(status: string): string {
  if (status === "Complete") {
    return "ok";
  }
  if (status === "In progress" || status === "Ready") {
    return "cool";
  }
  return "warn";
}

export default function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  );
}
