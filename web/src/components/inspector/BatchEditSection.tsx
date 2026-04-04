/** Batch edit operations for multiple selected facts — path ops + field ops. */

import { useState } from "react";
import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import type { BoxRecord } from "../../types/schema";

interface Props {
  selectedFacts: { index: number; record: BoxRecord }[];
  pageName: string | null;
}

export function BatchEditSection({ selectedFacts, pageName }: Props) {
  const { pageStates, updatePageState } = useDocumentStore();
  const markDirty = useCanvasStore((s) => s.markDirty);

  const [pathValue, setPathValue] = useState("");
  const [fieldKey, setFieldKey] = useState("currency");
  const [fieldValue, setFieldValue] = useState("");

  const updateFacts = (
    updater: (record: BoxRecord) => BoxRecord,
  ) => {
    if (!pageName) return;
    const page = pageStates.get(pageName);
    if (!page) return;
    pushUndoSnapshot();

    const newFacts = [...page.facts];
    for (const { index } of selectedFacts) {
      const existing = newFacts[index];
      if (!existing) continue;
      newFacts[index] = updater(existing);
    }
    updatePageState(pageName, { ...page, facts: newFacts });
    markDirty("bbox");
  };

  const addAsParent = () => {
    if (!pathValue.trim()) return;
    updateFacts((r) => ({
      ...r,
      fact: {
        ...r.fact,
        path: [pathValue.trim(), ...((r.fact.path as string[]) ?? [])],
      },
    }));
  };

  const addAsChild = () => {
    if (!pathValue.trim()) return;
    updateFacts((r) => ({
      ...r,
      fact: {
        ...r.fact,
        path: [...((r.fact.path as string[]) ?? []), pathValue.trim()],
      },
    }));
  };

  const removeFirstLevel = () => {
    updateFacts((r) => ({
      ...r,
      fact: {
        ...r.fact,
        path: ((r.fact.path as string[]) ?? []).slice(1),
      },
    }));
  };

  const removeLastLevel = () => {
    updateFacts((r) => ({
      ...r,
      fact: {
        ...r.fact,
        path: ((r.fact.path as string[]) ?? []).slice(0, -1),
      },
    }));
  };

  const setField = () => {
    const value = fieldValue.trim() || null;
    updateFacts((r) => ({
      ...r,
      fact: { ...r.fact, [fieldKey]: value },
    }));
  };

  const clearField = () => {
    updateFacts((r) => ({
      ...r,
      fact: { ...r.fact, [fieldKey]: null },
    }));
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      {/* Path batch ops */}
      <div>
        <Label>Path Operations</Label>
        <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
          <input
            value={pathValue}
            onChange={(e) => setPathValue(e.target.value)}
            placeholder="Path level value"
            style={inputStyle}
          />
        </div>
        <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
          <BatchBtn onClick={addAsParent}>+ As Parent</BatchBtn>
          <BatchBtn onClick={addAsChild}>+ As Child</BatchBtn>
          <BatchBtn onClick={removeFirstLevel}>- First</BatchBtn>
          <BatchBtn onClick={removeLastLevel}>- Last</BatchBtn>
        </div>
      </div>

      {/* Field batch ops */}
      <div>
        <Label>Field Operations</Label>
        <div style={{ display: "flex", gap: 4, marginBottom: 6 }}>
          <select
            value={fieldKey}
            onChange={(e) => setFieldKey(e.target.value)}
            style={{ ...inputStyle, flex: "0 0 auto", width: 120 }}
          >
            <option value="currency">Currency</option>
            <option value="scale">Scale</option>
            <option value="date">Date</option>
            <option value="period_type">Period</option>
            <option value="period_start">Period start</option>
            <option value="period_end">Period end</option>
            <option value="value_type">Value type</option>
            <option value="value_context">Context</option>
            <option value="row_role">Row role</option>
            <option value="natural_sign">Nat. sign</option>
            <option value="path_source">Path src</option>
            <option value="note_flag">Note flag</option>
          </select>
          <input
            value={fieldValue}
            onChange={(e) => setFieldValue(e.target.value)}
            placeholder="Value"
            style={inputStyle}
          />
        </div>
        <div style={{ display: "flex", gap: 4 }}>
          <BatchBtn onClick={setField}>Set</BatchBtn>
          <BatchBtn onClick={clearField} danger>
            Clear
          </BatchBtn>
        </div>
      </div>
    </div>
  );
}

const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: "5px 8px",
  fontSize: 12,
  background: "var(--surface-alt)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-xs)",
  color: "var(--text)",
  outline: "none",
};

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div
      style={{
        fontSize: 11,
        fontWeight: 600,
        color: "var(--text-soft)",
        marginBottom: 6,
        textTransform: "uppercase",
        letterSpacing: "0.04em",
      }}
    >
      {children}
    </div>
  );
}

function BatchBtn({
  children,
  onClick,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        fontSize: 11,
        fontWeight: 600,
        color: danger ? "var(--danger)" : "var(--accent)",
        background: "transparent",
        border: `1px solid ${danger ? "var(--danger)" : "var(--surface-border)"}`,
        borderRadius: "var(--radius-xs)",
        padding: "4px 10px",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = danger
          ? "rgba(185,73,73,0.1)"
          : "var(--surface-alt)";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "transparent";
      }}
    >
      {children}
    </button>
  );
}
