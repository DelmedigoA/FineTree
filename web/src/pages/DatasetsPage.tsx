import { useEffect, type CSSProperties } from "react";
import { useNavigate } from "react-router-dom";
import { useDatasetStore, type DatasetVersion, type SplitName } from "../stores/datasetStore";

const DISPLAY_SPLITS: SplitName[] = ["train", "test", "val"];

function formatDateTime(timestamp: number | null | undefined): string {
  if (!timestamp) return "—";
  return new Date(timestamp * 1000).toLocaleString();
}

function splitSummary(version: DatasetVersion, split: SplitName): string {
  const stats = version.split_stats?.[split];
  const label = split.charAt(0).toUpperCase() + split.slice(1);
  return `${label} ${stats?.doc_count ?? 0} docs / ${stats?.page_count ?? 0} pages`;
}

export function DatasetsPage() {
  const navigate = useNavigate();
  const {
    versions,
    versionsLoading,
    versionsError,
    loadVersions,
    openCreateDialog,
  } = useDatasetStore();

  useEffect(() => {
    void loadVersions();
  }, [loadVersions]);

  return (
    <div style={{ height: "100%", overflowY: "auto", background: "var(--bg)" }}>
      <div style={{ padding: "28px 40px 20px", borderBottom: "1px solid var(--surface-border)", background: "var(--surface)" }}>
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 16 }}>
          <div>
            <h1 style={{ fontFamily: "var(--font-heading)", fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em", marginBottom: 4 }}>
              Datasets
            </h1>
            <p style={{ color: "var(--text-muted)", fontSize: 13 }}>
              Browse saved dataset versions, split history, and push status.
            </p>
          </div>
          <button
            onClick={openCreateDialog}
            style={{
              padding: "9px 16px",
              background: "var(--accent)",
              color: "#fff",
              border: "none",
              borderRadius: "var(--radius-xs)",
              fontSize: 12,
              fontWeight: 700,
              cursor: "pointer",
            }}
          >
            + Add Dataset
          </button>
        </div>
      </div>

      <div style={{ padding: "24px 40px 40px" }}>
        {versionsLoading ? (
          <div style={emptyStateStyle}>Loading dataset versions…</div>
        ) : versionsError ? (
          <div style={{ ...emptyStateStyle, color: "var(--danger)" }}>{versionsError}</div>
        ) : versions.length === 0 ? (
          <div style={emptyStateStyle}>
            <div style={{ fontFamily: "var(--font-heading)", fontSize: 18, fontWeight: 700, color: "var(--text)", marginBottom: 8 }}>
              No datasets yet
            </div>
            <div style={{ color: "var(--text-muted)", fontSize: 14, marginBottom: 16 }}>
              Create your first dataset version to start tracking split stats and push history.
            </div>
            <button
              onClick={openCreateDialog}
              style={{
                padding: "9px 16px",
                background: "var(--accent)",
                color: "#fff",
                border: "none",
                borderRadius: "var(--radius-xs)",
                fontSize: 12,
                fontWeight: 700,
                cursor: "pointer",
              }}
            >
              Add Dataset
            </button>
          </div>
        ) : (
          <div style={{ background: "var(--surface)", border: "1px solid var(--surface-border)", borderRadius: "var(--radius-lg)", overflow: "hidden" }}>
            <table style={{ width: "100%", borderCollapse: "collapse" }}>
              <thead>
                <tr style={{ background: "var(--surface-alt)" }}>
                  {["Dataset", "Created", "Split Stats", "Push Status", "Last Pushed"].map((header) => (
                    <th
                      key={header}
                      style={{
                        padding: "14px 16px",
                        textAlign: "left",
                        fontSize: 11,
                        fontWeight: 700,
                        letterSpacing: "0.05em",
                        textTransform: "uppercase",
                        color: "var(--text-soft)",
                        borderBottom: "1px solid var(--surface-border)",
                      }}
                    >
                      {header}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {versions.map((version, index) => (
                  <tr
                    key={version.version_id}
                    onClick={() => navigate(`/datasets/${version.version_id}`)}
                    style={{
                      cursor: "pointer",
                      background: index % 2 === 0 ? "transparent" : "var(--surface-alt)",
                    }}
                  >
                    <td style={cellStyle}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        <span style={{ fontSize: 14, fontWeight: 700, color: "var(--text)" }}>{version.name}</span>
                        <span style={{ fontSize: 11, color: "var(--text-soft)", fontFamily: "var(--font-mono)" }}>
                          {version.version_id}
                        </span>
                      </div>
                    </td>
                    <td style={cellStyle}>
                      <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{formatDateTime(version.created_at)}</div>
                    </td>
                    <td style={cellStyle}>
                      <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
                        {DISPLAY_SPLITS.map((split) => (
                          <span key={split} style={{ fontSize: 12, color: "var(--text-muted)" }}>
                            {splitSummary(version, split)}
                          </span>
                        ))}
                      </div>
                    </td>
                    <td style={cellStyle}>
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
                    </td>
                    <td style={cellStyle}>
                      <div style={{ fontSize: 13, color: "var(--text-muted)" }}>{formatDateTime(version.last_pushed_at)}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

const cellStyle: CSSProperties = {
  padding: "16px",
  borderBottom: "1px solid var(--surface-border)",
  verticalAlign: "top",
};

const emptyStateStyle: CSSProperties = {
  display: "flex",
  flexDirection: "column",
  alignItems: "center",
  justifyContent: "center",
  padding: "64px 32px",
  background: "var(--surface)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-lg)",
  color: "var(--text-muted)",
  textAlign: "center",
};
