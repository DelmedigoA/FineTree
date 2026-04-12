import { useCallback, useEffect, useRef, useState, type CSSProperties, type ReactNode } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { get } from "../api/client";
import { useDatasetStore } from "../stores/datasetStore";
import type { WorkspaceDocument } from "../types/api";

function formatDateTime(timestamp: number | null | undefined): string {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleString();
}

export function DatasetVersionPage() {
  const { versionId } = useParams<{ versionId: string }>();
  const navigate = useNavigate();
  const {
    versions,
    versionLoading,
    versionError,
    datasetName,
    setDatasetName,
    loadSchemaFields,
    loadVersion,
    saveActiveVersion,
    deleteVersion,
    activeTab,
    setActiveTab,
  } = useDatasetStore();
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [saveSuccess, setSaveSuccess] = useState<string | null>(null);

  useEffect(() => {
    void loadSchemaFields();
  }, [loadSchemaFields]);

  useEffect(() => {
    if (versionId) {
      void loadVersion(versionId);
    }
  }, [versionId, loadVersion]);

  const version = versions.find((entry) => entry.version_id === versionId);

  const handleSave = useCallback(async () => {
    setSaving(true);
    setSaveError(null);
    setSaveSuccess(null);
    try {
      await saveActiveVersion();
      setSaveSuccess("Saved");
    } catch (err) {
      setSaveError(String(err));
    } finally {
      setSaving(false);
    }
  }, [saveActiveVersion]);

  async function handleDelete() {
    if (!version || !confirm(`Delete dataset "${version.name}"?`)) return;
    await deleteVersion(version.version_id);
    navigate("/datasets");
  }

  if (versionLoading && !version) {
    return <PageMessage message="Loading dataset…" />;
  }

  if (versionError && !version) {
    return <PageMessage message={versionError} danger />;
  }

  if (!version) {
    return <PageMessage message="Dataset version not found." danger />;
  }

  const tabs: { id: "splits" | "export" | "preview"; label: string }[] = [
    { id: "splits", label: "Splits" },
    { id: "export", label: "Export" },
    { id: "preview", label: "Preview" },
  ];

  return (
    <div style={{ height: "100%", overflowY: "auto", background: "var(--bg)" }}>
      <div style={{ padding: "28px 40px 20px", borderBottom: "1px solid var(--surface-border)", background: "var(--surface)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 20 }}>
          <div style={{ flex: 1, minWidth: 0 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
              <button
                onClick={() => navigate("/datasets")}
                style={{
                  padding: "5px 10px",
                  fontSize: 12,
                  fontWeight: 600,
                  borderRadius: "var(--radius-xs)",
                  border: "1px solid var(--surface-border)",
                  background: "transparent",
                  color: "var(--text-muted)",
                  cursor: "pointer",
                }}
              >
                ← Back
              </button>
              <span
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  padding: "4px 10px",
                  borderRadius: 999,
                  fontSize: 11,
                  fontWeight: 700,
                  color: version.push_status === "pushed" ? "var(--ok)" : "var(--text-muted)",
                  background: version.push_status === "pushed" ? "rgba(34,197,94,0.12)" : "var(--surface-alt)",
                  border: `1px solid ${version.push_status === "pushed" ? "rgba(34,197,94,0.3)" : "var(--surface-border)"}`,
                }}
              >
                {version.push_status === "pushed" ? "Pushed" : "Not pushed"}
              </span>
            </div>

            <input
              value={datasetName}
              onChange={(event) => setDatasetName(event.target.value)}
              style={{
                width: "100%",
                fontFamily: "var(--font-heading)",
                fontSize: 24,
                fontWeight: 700,
                letterSpacing: "-0.02em",
                background: "transparent",
                border: "none",
                color: "var(--text)",
                outline: "none",
                padding: 0,
                marginBottom: 10,
              }}
            />

            <div style={{ display: "flex", flexWrap: "wrap", gap: 14, fontSize: 12, color: "var(--text-muted)" }}>
              <span>Created {formatDateTime(version.created_at)}</span>
              <span>Updated {formatDateTime(version.updated_at)}</span>
              <span>Last pushed {formatDateTime(version.last_pushed_at)}</span>
            </div>
          </div>

          <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: 8 }}>
            <div style={{ display: "flex", gap: 8 }}>
              <button onClick={handleDelete} style={dangerBtnStyle}>
                Delete
              </button>
              <button
                onClick={() => void handleSave()}
                style={{ ...primaryBtnStyle, opacity: saving || !datasetName.trim() ? 0.6 : 1, cursor: saving || !datasetName.trim() ? "default" : "pointer" }}
                disabled={saving || !datasetName.trim()}
              >
                {saving ? "Saving…" : "Save Changes"}
              </button>
            </div>
            {saveError && <span style={{ fontSize: 12, color: "var(--danger)" }}>{saveError}</span>}
            {saveSuccess && !saveError && <span style={{ fontSize: 12, color: "var(--ok)" }}>{saveSuccess}</span>}
          </div>
        </div>

        <div style={{ display: "flex", gap: 4, marginTop: 18 }}>
          {tabs.map((tab) => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              style={{
                padding: "10px 14px",
                fontSize: 12,
                fontWeight: 600,
                borderRadius: "var(--radius-xs) var(--radius-xs) 0 0",
                background: "transparent",
                color: activeTab === tab.id ? "var(--accent)" : "var(--text-muted)",
                border: "none",
                borderBottom: activeTab === tab.id ? "2px solid var(--accent)" : "2px solid transparent",
                cursor: "pointer",
              }}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </div>

      <div style={{ padding: "24px 40px 40px" }}>
        <div style={{ background: "var(--surface)", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-lg)", padding: 20 }}>
          {activeTab === "splits" && <DatasetSplitsTab />}
          {activeTab === "export" && <DatasetExportTab />}
          {activeTab === "preview" && <DatasetPreviewTab />}
        </div>
      </div>
    </div>
  );
}

function DatasetSplitsTab() {
  const {
    splitMode,
    setSplitMode,
    trainPct,
    setTrainPct,
    testPct,
    setTestPct,
    valPct,
    setValPct,
    seed,
    setSeed,
    applyRandomSplit,
    assignments,
    setAssignment,
  } = useDatasetStore();
  const [docs, setDocs] = useState<WorkspaceDocument[]>([]);
  const [docsLoading, setDocsLoading] = useState(false);

  useEffect(() => {
    setDocsLoading(true);
    get<WorkspaceDocument[]>("/workspace/documents")
      .then(setDocs)
      .catch(() => {
        /* ignore */
      })
      .finally(() => setDocsLoading(false));
  }, []);

  // Only documents with at least one approved page are eligible for dataset creation
  const eligibleDocs = docs.filter((doc) => (doc.approved_page_count ?? 0) > 0);

  const unassigned = 100 - trainPct - testPct - valPct;
  const counts = { train: 0, test: 0, val: 0, exclude: 0, unassigned: 0 };
  const pageCounts = { train: 0, test: 0, val: 0, exclude: 0, unassigned: 0 };
  eligibleDocs.forEach((doc) => {
    const split = assignments[doc.doc_id];
    const approvedPages = doc.approved_page_count ?? 0;
    if (split) {
      counts[split] += 1;
      pageCounts[split] += approvedPages;
    } else {
      counts.unassigned += 1;
      pageCounts.unassigned += approvedPages;
    }
  });

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 18 }}>
      <SectionLabel>Split Assignment</SectionLabel>

      <div style={{ display: "flex", gap: 4 }}>
        {(["random", "manual"] as const).map((mode) => (
          <button
            key={mode}
            onClick={() => setSplitMode(mode)}
            style={{
              padding: "6px 14px",
              fontSize: 12,
              fontWeight: 600,
              borderRadius: "var(--radius-xs)",
              background: splitMode === mode ? "var(--accent)" : "transparent",
              color: splitMode === mode ? "#fff" : "var(--text-muted)",
              border: splitMode === mode ? "none" : "1px solid var(--surface-border)",
              cursor: "pointer",
            }}
          >
            {mode === "random" ? "Random" : "Manual"}
          </button>
        ))}
      </div>

      {splitMode === "random" && (
        <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
          <div style={{ display: "flex", gap: 10 }}>
            {(
              [
                { label: "Train %", value: trainPct, set: setTrainPct },
                { label: "Test %", value: testPct, set: setTestPct },
                { label: "Val %", value: valPct, set: setValPct },
              ] as const
            ).map((item) => (
              <label key={item.label} style={{ display: "flex", flexDirection: "column", gap: 4, flex: 1 }}>
                <span style={eyebrowStyle}>{item.label}</span>
                <input
                  type="number"
                  min={0}
                  max={100}
                  step={1}
                  value={item.value}
                  onChange={(event) => item.set(Number(event.target.value))}
                  style={inputStyle}
                />
              </label>
            ))}
          </div>

          {unassigned !== 0 && (
            <div
              style={{
                fontSize: 12,
                color: unassigned < 0 ? "var(--danger)" : "var(--text-muted)",
                padding: "8px 10px",
                background: unassigned < 0 ? "rgba(239,68,68,0.08)" : "var(--surface-alt)",
                borderRadius: "var(--radius-xs)",
              }}
            >
              {unassigned < 0 ? `Total exceeds 100% by ${Math.abs(unassigned)}%` : `${unassigned}% unassigned (will go to val)`}
            </div>
          )}

          <label style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 13, color: "var(--text-muted)" }}>
            <span style={{ minWidth: 36 }}>Seed</span>
            <input
              type="number"
              value={seed}
              onChange={(event) => setSeed(Number(event.target.value))}
              style={{ ...inputStyle, width: 120 }}
            />
          </label>

          <ActionBtn
            onClick={() => applyRandomSplit(eligibleDocs.map((doc) => doc.doc_id))}
            primary
            disabled={docsLoading || eligibleDocs.length === 0}
          >
            {docsLoading ? "Loading docs…" : `Apply Random Split (${eligibleDocs.length} docs)`}
          </ActionBtn>
        </div>
      )}

      {splitMode === "manual" && (
        <div style={{ maxHeight: 420, overflowY: "auto", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-sm)" }}>
          {docsLoading ? (
            <div style={tableMessageStyle}>Loading documents…</div>
          ) : eligibleDocs.length === 0 ? (
            <div style={tableMessageStyle}>No documents with approved pages found</div>
          ) : (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ background: "var(--surface-alt)", position: "sticky", top: 0 }}>
                  {["Doc ID", "Pages", "Status", "Split"].map((header) => (
                    <th key={header} style={tableHeaderStyle}>
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {eligibleDocs.map((doc, index) => {
                  const currentSplit = assignments[doc.doc_id];
                  return (
                    <tr key={doc.doc_id} style={{ background: index % 2 === 0 ? "transparent" : "var(--surface-alt)" }}>
                      <td style={tableCellStyle} title={doc.doc_id}>{doc.doc_id}</td>
                      <td style={tableCellStyle}>{doc.approved_page_count ?? 0}</td>
                      <td style={tableCellStyle}>{doc.status}</td>
                      <td style={tableCellStyle}>
                        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                          {(["train", "test", "val", "exclude"] as const).map((split) => (
                            <button
                              key={split}
                              onClick={() => setAssignment(doc.doc_id, split)}
                              style={{
                                padding: "3px 7px",
                                fontSize: 10,
                                fontWeight: 700,
                                borderRadius: 999,
                                border: "1px solid var(--surface-border)",
                                background: currentSplit === split ? splitColor(split) : "transparent",
                                color: currentSplit === split ? "#fff" : "var(--text-muted)",
                                cursor: "pointer",
                              }}
                            >
                              {split === "exclude" ? "Excl" : split.charAt(0).toUpperCase() + split.slice(1)}
                            </button>
                          ))}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      )}

      <div style={{ display: "flex", gap: 16, flexWrap: "wrap", padding: "10px 12px", background: "var(--surface-alt)", borderRadius: "var(--radius-sm)", fontSize: 12, color: "var(--text-muted)" }}>
        {(["train", "test", "val", "exclude"] as const).map((split) => (
          <span key={split}>
            <span style={{ display: "inline-block", width: 8, height: 8, borderRadius: "50%", background: splitColor(split), marginRight: 5 }} />
            <strong style={{ color: "var(--text)" }}>{split.charAt(0).toUpperCase() + split.slice(1)}:</strong> {counts[split]} docs · {pageCounts[split]} pages
          </span>
        ))}
        {counts.unassigned > 0 && <span>Unassigned: {counts.unassigned} docs · {pageCounts.unassigned} pages</span>}
      </div>
    </div>
  );
}

function DatasetExportTab() {
  const {
    schemaFields,
    imageScaling,
    setImageScaling,
    maxPixels,
    setMaxPixels,
    minPixels,
    setMinPixels,
    bboxGridNorm,
    setBboxGridNorm,
    valuesNorm,
    setValuesNorm,
    compactMode,
    setCompactMode,
    dropDate,
    setDropDate,
    approvedPagesOnly,
    setApprovedPagesOnly,
    selectedFactKeys,
    toggleFactKey,
    selectedPageMetaKeys,
    togglePageMetaKey,
    includeBbox,
    setIncludeBbox,
    hfRepo,
    setHfRepo,
    pushMode,
    setPushMode,
    isPushing,
    pushLog,
    pushResult,
    pushError,
    pushToHF,
    tokenizer,
    tokenStats,
    tokenStatsLoading,
    tokenStatsError,
    tokenStatsLog,
    computeTokenStats,
  } = useDatasetStore();
  const [instructionPreview, setInstructionPreview] = useState<string | null>(schemaFields?.instruction_preview ?? null);
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);

  const fetchPreview = useCallback(() => {
    const params = new URLSearchParams();
    selectedFactKeys.forEach((key) => params.append("fact_keys", key));
    selectedPageMetaKeys.forEach((key) => params.append("page_meta_keys", key));
    fetch(`/api/dataset/schema-fields?${params.toString()}`)
      .then((response) => response.json())
      .then((data: { instruction_preview?: string }) => {
        setInstructionPreview(data.instruction_preview ?? null);
      })
      .catch(() => {
        /* ignore */
      });
  }, [selectedFactKeys, selectedPageMetaKeys]);

  useEffect(() => {
    if (schemaFields?.instruction_preview) {
      setInstructionPreview(schemaFields.instruction_preview);
    }
  }, [schemaFields]);

  useEffect(() => {
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(fetchPreview, 400);
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, [fetchPreview]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [pushLog, tokenStatsLog]);

  const required = schemaFields?.required_prompt_canonical_keys ?? [];

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
      <SectionLabel>Normalizations</SectionLabel>

      <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
        <CheckRow checked={imageScaling} onChange={setImageScaling} label="Image scaling (qwen-vl-utils smart_resize)" />
        {imageScaling && (
          <div style={{ marginLeft: 24, display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)" }}>
              Max pixels:
              <input type="number" value={maxPixels} onChange={(event) => setMaxPixels(Number(event.target.value))} style={{ ...inputStyle, width: 120 }} />
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)" }}>
              Min pixels:
              <input
                type="number"
                value={minPixels ?? ""}
                placeholder="optional"
                onChange={(event) => setMinPixels(event.target.value === "" ? null : Number(event.target.value))}
                style={{ ...inputStyle, width: 120 }}
              />
            </label>
          </div>
        )}
        <CheckRow checked={bboxGridNorm} onChange={setBboxGridNorm} label="Bbox grid normalization" />
        <CheckRow checked={valuesNorm} onChange={setValuesNorm} label="Values normalization" />

        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Schema compaction:</span>
          {(["raw", "compact", "aggressive"] as const).map((mode) => (
            <label key={mode} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, cursor: "pointer", color: "var(--text-muted)" }}>
              <input type="radio" checked={compactMode === mode} onChange={() => setCompactMode(mode)} style={{ accentColor: "var(--accent)" }} />
              {mode.charAt(0).toUpperCase() + mode.slice(1)}
            </label>
          ))}
        </div>

        <CheckRow checked={dropDate} onChange={setDropDate} label="Drop date field" />
        <CheckRow checked={approvedPagesOnly} onChange={setApprovedPagesOnly} label="Approved pages only" />
      </div>

      <SectionLabel>Field Selection</SectionLabel>

      <div style={{ display: "flex", gap: 16 }}>
        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={eyebrowStyle}>Page Meta Fields</div>
          {schemaFields?.prompt_page_meta_keys?.map((key) => {
            const isRequired = required.includes(key);
            return (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)", cursor: isRequired ? "default" : "pointer", opacity: isRequired ? 0.7 : 1 }}>
                <input type="checkbox" checked={selectedPageMetaKeys.includes(key)} disabled={isRequired} onChange={() => !isRequired && togglePageMetaKey(key)} style={{ accentColor: "var(--accent)" }} />
                {isRequired && <span style={{ fontSize: 10 }}>🔒</span>}
                {key}
              </label>
            );
          })}
        </div>

        <div style={{ flex: 1, display: "flex", flexDirection: "column", gap: 6 }}>
          <div style={eyebrowStyle}>Fact Fields</div>
          {schemaFields?.prompt_fact_keys?.map((key) => {
            const isRequired = required.includes(key);
            return (
              <label key={key} style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, color: "var(--text-muted)", cursor: isRequired ? "default" : "pointer", opacity: isRequired ? 0.7 : 1 }}>
                <input type="checkbox" checked={selectedFactKeys.includes(key)} disabled={isRequired} onChange={() => !isRequired && toggleFactKey(key)} style={{ accentColor: "var(--accent)" }} />
                {isRequired && <span style={{ fontSize: 10 }}>🔒</span>}
                {key}
              </label>
            );
          })}
        </div>
      </div>

      <CheckRow checked={includeBbox} onChange={setIncludeBbox} label="Include bbox" />

      <SectionLabel>Instruction Preview</SectionLabel>
      <pre style={previewBlockStyle}>{instructionPreview ?? "Loading instruction template…"}</pre>

      <SectionLabel>Token Statistics</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <span style={{ fontSize: 12, color: "var(--text-muted)" }}>
            Tokenizer: <strong style={{ color: "var(--text)" }}>{tokenizer}</strong>
          </span>
          <ActionBtn onClick={() => void computeTokenStats()} primary disabled={tokenStatsLoading}>
            {tokenStatsLoading ? "Computing…" : "Compute Token Stats"}
          </ActionBtn>
        </div>

        {tokenStatsLoading && tokenStatsLog.length > 0 && (
          <div style={logBoxStyle}>
            {tokenStatsLog.map((line, index) => <div key={index}>{line}</div>)}
          </div>
        )}

        {tokenStatsError && <div style={{ fontSize: 12, color: "var(--danger)" }}>Error: {tokenStatsError}</div>}

        {tokenStats ? (
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12, border: "1px solid var(--surface-border)", borderRadius: "var(--radius-sm)", overflow: "hidden" }}>
            <thead>
              <tr style={{ background: "var(--surface-alt)" }}>
                {["Split", "Pages", "Total", "Min", "Mean", "Max", "Median"].map((header) => (
                  <th key={header} style={tableHeaderStyle}>{header}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(tokenStats.splits).map(([splitName, stats]) => (
                <tr key={splitName}>
                  <td style={tableCellStyle}>{splitName}</td>
                  <td style={tableCellStyle}>{stats.sample_count}</td>
                  <td style={tableCellStyle}>{stats.total_text_tokens}</td>
                  <td style={tableCellStyle}>{stats.per_page_full.min}</td>
                  <td style={tableCellStyle}>{Math.round(stats.per_page_full.mean)}</td>
                  <td style={tableCellStyle}>{stats.per_page_full.max}</td>
                  <td style={tableCellStyle}>{Math.round(stats.per_page_full.median)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : !tokenStatsLoading && (
          <div style={{ fontSize: 12, color: "#92400e", background: "rgba(251,191,36,0.08)", border: "1px solid rgba(251,191,36,0.25)", borderRadius: "var(--radius-sm)", padding: "8px 12px" }}>
            Token stats not computed yet.
          </div>
        )}
      </div>

      <SectionLabel>Push To HuggingFace</SectionLabel>
      <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <label style={{ fontSize: 12, color: "var(--text-muted)", minWidth: 36 }}>Repo:</label>
          <input value={hfRepo} onChange={(event) => setHfRepo(event.target.value)} placeholder="username/dataset-name" style={{ ...inputStyle, flex: 1 }} />
        </div>

        <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap" }}>
          <span style={{ fontSize: 13, color: "var(--text-muted)" }}>Push mode:</span>
          {(
            [
              { value: "single", label: "Single repo (3 splits)" },
              { value: "separate", label: "Separate repos" },
            ] as const
          ).map((option) => (
            <label key={option.value} style={{ display: "flex", alignItems: "center", gap: 4, fontSize: 12, cursor: "pointer", color: "var(--text-muted)" }}>
              <input type="radio" checked={pushMode === option.value} onChange={() => setPushMode(option.value)} style={{ accentColor: "var(--accent)" }} />
              {option.label}
            </label>
          ))}
        </div>

        <ActionBtn onClick={() => void pushToHF()} primary disabled={isPushing || !hfRepo.trim()}>
          {isPushing ? "Pushing…" : "Push to HuggingFace"}
        </ActionBtn>

        {(pushLog.length > 0 || isPushing) && (
          <div style={logBoxStyle}>
            {pushLog.map((line, index) => <div key={index}>{line}</div>)}
            <div ref={logEndRef} />
          </div>
        )}

        {pushError && (
          <div style={{ fontSize: 12, color: "var(--danger)", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.25)", borderRadius: "var(--radius-sm)", padding: "8px 12px" }}>
            Error: {pushError}
          </div>
        )}

        {pushResult && (
          <div style={{ fontSize: 12, color: "#065f46", background: "rgba(16,185,129,0.08)", border: "1px solid rgba(16,185,129,0.3)", borderRadius: "var(--radius-sm)", padding: "10px 12px", display: "flex", flexDirection: "column", gap: 4 }}>
            <div style={{ fontWeight: 700, marginBottom: 4 }}>Push successful</div>
            {Object.entries(pushResult).map(([split, url]) => (
              <div key={split}>
                <strong>{split}:</strong>{" "}
                <a href={url} target="_blank" rel="noopener noreferrer" style={{ color: "var(--accent)", textDecoration: "underline" }}>
                  {url}
                </a>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function DatasetPreviewTab() {
  const {
    previewSplit,
    setPreviewSplit,
    previewRows,
    previewLoading,
    previewError,
    loadPreview,
  } = useDatasetStore();

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
      <SectionLabel>Preview Rows</SectionLabel>

      <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
        {(["train", "test", "val"] as const).map((split) => (
          <button
            key={split}
            onClick={() => setPreviewSplit(split)}
            style={{
              padding: "5px 14px",
              fontSize: 12,
              fontWeight: 600,
              borderRadius: 999,
              background: previewSplit === split ? "var(--accent)" : "transparent",
              color: previewSplit === split ? "#fff" : "var(--text-muted)",
              border: previewSplit === split ? "none" : "1px solid var(--surface-border)",
              cursor: "pointer",
            }}
          >
            {split.charAt(0).toUpperCase() + split.slice(1)}
          </button>
        ))}
        <ActionBtn onClick={() => void loadPreview()} primary disabled={previewLoading}>
          {previewLoading ? "Loading…" : "Load Preview"}
        </ActionBtn>
      </div>

      {previewError && <div style={{ fontSize: 12, color: "var(--danger)" }}>{previewError}</div>}

      {previewRows.length === 0 && !previewLoading ? (
        <div style={{ fontSize: 13, color: "var(--text-muted)" }}>
          No preview rows loaded yet.
        </div>
      ) : (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))", gap: 16 }}>
          {previewRows.map((row) => (
            <div key={`${row.doc_id}-${row.page}`} style={{ border: "1px solid var(--surface-border)", borderRadius: "var(--radius-sm)", overflow: "hidden", background: "var(--surface-alt)" }}>
              <div style={{ aspectRatio: "4 / 3", background: "#111" }}>
                <img src={row.image_url} alt={`${row.doc_id} ${row.page}`} style={{ width: "100%", height: "100%", objectFit: "contain" }} />
              </div>
              <div style={{ padding: 12, display: "flex", flexDirection: "column", gap: 8 }}>
                <div style={{ fontSize: 12, fontWeight: 700, color: "var(--text)" }}>{row.doc_id}</div>
                <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{row.page} · {row.split}</div>
                <pre style={{ ...previewBlockStyle, maxHeight: 180 }}>{row.text_preview}</pre>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function PageMessage({ message, danger }: { message: string; danger?: boolean }) {
  return (
    <div style={{ height: "100%", display: "flex", alignItems: "center", justifyContent: "center", background: "var(--bg)", color: danger ? "var(--danger)" : "var(--text-muted)", fontSize: 14 }}>
      {message}
    </div>
  );
}

function SectionLabel({ children }: { children: ReactNode }) {
  return (
    <div style={{ fontSize: 11, fontWeight: 700, color: "var(--text-soft)", letterSpacing: "0.05em", textTransform: "uppercase" }}>
      {children}
    </div>
  );
}

function CheckRow({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: string }) {
  return (
    <label style={{ display: "flex", alignItems: "center", gap: 10, fontSize: 13, color: "var(--text-muted)" }}>
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} style={{ accentColor: "var(--accent)" }} />
      {label}
    </label>
  );
}

function ActionBtn({
  children,
  onClick,
  primary,
  disabled,
}: {
  children: ReactNode;
  onClick: () => void;
  primary?: boolean;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        padding: "8px 14px",
        fontSize: 12,
        fontWeight: 700,
        background: primary ? "var(--accent)" : "transparent",
        border: primary ? "none" : "1px solid var(--surface-border)",
        borderRadius: "var(--radius-xs)",
        color: primary ? "#fff" : "var(--text-muted)",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.6 : 1,
      }}
    >
      {children}
    </button>
  );
}

function splitColor(split: "train" | "test" | "val" | "exclude") {
  if (split === "train") return "var(--accent)";
  if (split === "test") return "#f59e0b";
  if (split === "val") return "#8b5cf6";
  return "var(--text-muted)";
}

const primaryBtnStyle: CSSProperties = {
  padding: "9px 16px",
  background: "var(--accent)",
  color: "#fff",
  border: "none",
  borderRadius: "var(--radius-xs)",
  fontSize: 12,
  fontWeight: 700,
  cursor: "pointer",
};

const dangerBtnStyle: CSSProperties = {
  padding: "9px 14px",
  background: "transparent",
  color: "var(--danger)",
  border: "1px solid var(--danger)",
  borderRadius: "var(--radius-xs)",
  fontSize: 12,
  fontWeight: 700,
  cursor: "pointer",
};

const inputStyle: CSSProperties = {
  padding: "8px 10px",
  fontSize: 12,
  background: "var(--surface-alt)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-xs)",
  color: "var(--text)",
  outline: "none",
};

const previewBlockStyle: CSSProperties = {
  fontSize: 11,
  fontFamily: "var(--font-mono, monospace)",
  color: "var(--text-muted)",
  background: "var(--surface-alt)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-sm)",
  padding: "10px 12px",
  overflowY: "auto",
  whiteSpace: "pre-wrap",
  wordBreak: "break-word",
  margin: 0,
};

const logBoxStyle: CSSProperties = {
  fontFamily: "var(--font-mono, monospace)",
  fontSize: 11,
  background: "var(--surface-alt)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-sm)",
  padding: "8px 10px",
  maxHeight: 150,
  overflowY: "auto",
  color: "var(--text-muted)",
};

const eyebrowStyle: CSSProperties = {
  fontSize: 11,
  fontWeight: 700,
  color: "var(--text-soft)",
  textTransform: "uppercase",
  letterSpacing: "0.05em",
};

const tableHeaderStyle: CSSProperties = {
  padding: "8px 10px",
  textAlign: "left",
  fontWeight: 600,
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--surface-border)",
};

const tableCellStyle: CSSProperties = {
  padding: "8px 10px",
  color: "var(--text-muted)",
  borderBottom: "1px solid var(--surface-border)",
};

const tableMessageStyle: CSSProperties = {
  padding: 16,
  fontSize: 13,
  color: "var(--text-muted)",
  textAlign: "center",
};
