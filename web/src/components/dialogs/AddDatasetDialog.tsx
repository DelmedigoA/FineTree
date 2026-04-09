import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useDatasetStore } from "../../stores/datasetStore";

export function AddDatasetDialog() {
  const { createDialogOpen, closeCreateDialog, createVersion, loadSchemaFields, schemaFields } = useDatasetStore();
  const [name, setName] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (createDialogOpen) {
      void loadSchemaFields();
    }
  }, [createDialogOpen, loadSchemaFields]);

  if (!createDialogOpen) return null;

  async function handleCreate() {
    if (!name.trim() || submitting) return;
    setSubmitting(true);
    setError(null);
    try {
      const version = await createVersion(name.trim());
      setName("");
      navigate(`/datasets/${version.version_id}`);
    } catch (err) {
      setError(String(err));
    } finally {
      setSubmitting(false);
    }
  }

  function handleClose() {
    if (submitting) return;
    setName("");
    setError(null);
    closeCreateDialog();
  }

  return (
    <div
      onClick={handleClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1000,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        background: "rgba(0,0,0,0.6)",
        backdropFilter: "blur(4px)",
      }}
    >
      <div
        onClick={(event) => event.stopPropagation()}
        style={{
          width: 420,
          background: "var(--surface-raised)",
          borderRadius: "var(--radius-lg)",
          border: "1px solid var(--surface-border)",
          boxShadow: "0 24px 48px rgba(0,0,0,0.4)",
          padding: 24,
          display: "flex",
          flexDirection: "column",
          gap: 16,
        }}
      >
        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 12 }}>
          <div>
            <div style={{ fontFamily: "var(--font-heading)", fontSize: 18, fontWeight: 700, color: "var(--text)" }}>
              Add Dataset
            </div>
            <div style={{ fontSize: 13, color: "var(--text-muted)", marginTop: 4 }}>
              Create a new dataset version, then continue editing it in the datasets workspace.
            </div>
          </div>
          <button
            onClick={handleClose}
            style={{
              width: 28,
              height: 28,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              borderRadius: "var(--radius-xs)",
              fontSize: 18,
              color: "var(--text-muted)",
              cursor: submitting ? "default" : "pointer",
              background: "transparent",
              border: "none",
              opacity: submitting ? 0.4 : 1,
            }}
          >
            ×
          </button>
        </div>

        <label style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Dataset Name
          </span>
          <input
            autoFocus
            value={name}
            onChange={(event) => setName(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter") void handleCreate();
              if (event.key === "Escape") handleClose();
            }}
            placeholder="FineTree April 2026"
            style={{
              padding: "10px 12px",
              fontSize: 14,
              background: "var(--surface-alt)",
              border: "1px solid var(--surface-border)",
              borderRadius: "var(--radius-xs)",
              color: "var(--text)",
              outline: "none",
            }}
          />
        </label>

        {error && (
          <div
            style={{
              fontSize: 12,
              color: "var(--danger)",
              background: "rgba(239,68,68,0.08)",
              border: "1px solid rgba(239,68,68,0.25)",
              borderRadius: "var(--radius-sm)",
              padding: "8px 10px",
            }}
          >
            {error}
          </div>
        )}

        <div style={{ display: "flex", justifyContent: "flex-end", gap: 8 }}>
          <button
            onClick={handleClose}
            style={{
              padding: "8px 12px",
              fontSize: 12,
              fontWeight: 600,
              background: "transparent",
              border: "1px solid var(--surface-border)",
              borderRadius: "var(--radius-xs)",
              color: "var(--text-muted)",
              cursor: submitting ? "default" : "pointer",
            }}
          >
            Cancel
          </button>
          <button
            onClick={() => void handleCreate()}
            disabled={!name.trim() || submitting || !schemaFields}
            style={{
              padding: "8px 14px",
              fontSize: 12,
              fontWeight: 700,
              background: "var(--accent)",
              border: "none",
              borderRadius: "var(--radius-xs)",
              color: "#fff",
              cursor: !name.trim() || submitting || !schemaFields ? "default" : "pointer",
              opacity: !name.trim() || submitting || !schemaFields ? 0.6 : 1,
            }}
          >
            {submitting ? "Creating…" : !schemaFields ? "Loading…" : "Create Dataset"}
          </button>
        </div>
      </div>
    </div>
  );
}
