/** Full fact editor — all ~20+ fields, compact layout. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import {
  FieldRow,
  FieldPair,
  FieldTriple,
  FieldCell,
  FieldInput,
  FieldSelect,
  FieldCheckbox,
} from "./FieldWidgets";
import { EquationsEditor } from "./EquationsEditor";
import { PathEditor } from "./PathEditor";
import type { BoxRecord } from "../../types/schema";

interface Props {
  selectedFacts: { index: number; record: BoxRecord }[];
  pageName: string | null;
}

export function FactEditorSection({ selectedFacts, pageName }: Props) {
  const { pageStates, updatePageState } = useDocumentStore();
  const markDirty = useCanvasStore((s) => s.markDirty);

  const isMulti = selectedFacts.length > 1;

  /** Get the shared value of a field across all selected facts. */
  const shared = (key: string): string | null => {
    const values = new Set(
      selectedFacts.map((f) => {
        const v = f.record.fact[key];
        return v == null ? "" : String(v);
      }),
    );
    if (values.size === 1) return [...values][0]!;
    return null; // multiple values
  };

  const sharedBool = (key: string): boolean | null => {
    const values = new Set(
      selectedFacts.map((f) => Boolean(f.record.fact[key])),
    );
    if (values.size === 1) return [...values][0]!;
    return null;
  };

  /** Update a field on all selected facts. */
  const updateField = (key: string, value: unknown) => {
    if (!pageName) return;
    pushUndoSnapshot();
    const page = pageStates.get(pageName);
    if (!page) return;

    const newFacts = [...page.facts];
    for (const { index } of selectedFacts) {
      const existing = newFacts[index];
      if (!existing) continue;
      newFacts[index] = {
        ...existing,
        fact: { ...existing.fact, [key]: value === "" ? null : value },
      };
    }
    updatePageState(pageName, { ...page, facts: newFacts });
    markDirty("bbox");
  };

  const v = (key: string) => shared(key) ?? "";
  const placeholder = (key: string) =>
    shared(key) === null ? "Multiple values" : "";

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {isMulti && (
        <div
          style={{
            fontSize: 11,
            color: "var(--accent)",
            fontWeight: 600,
            padding: "4px 0",
          }}
        >
          Editing {selectedFacts.length} facts
        </div>
      )}

      {/* Two-per-row fields above Note flag */}
      <FieldPair>
        <FieldCell label="Fact #">
          <div
            style={{
              fontSize: 12,
              fontFamily: "var(--font-mono)",
              color: "var(--text-soft)",
              background: "var(--surface-alt)",
              padding: "4px 8px",
              borderRadius: "var(--radius-xs)",
              minHeight: 28,
              display: "flex",
              alignItems: "center",
            }}
          >
            {v("fact_num") || "—"}
          </div>
        </FieldCell>
        <FieldCell label="Value">
          <FieldInput
            value={v("value")}
            placeholder={placeholder("value") || "Value"}
            onChange={(val) => updateField("value", val)}
          />
        </FieldCell>
      </FieldPair>
      <FieldTriple>
        <FieldCell label="Row role">
          <FieldSelect
            value={v("row_role") || "detail"}
            options={[
              { value: "detail", label: "Detail" },
              { value: "total", label: "Total" },
            ]}
            onChange={(val) => updateField("row_role", val)}
          />
        </FieldCell>
        <FieldCell label="Context">
          <FieldSelect
            value={v("value_context")}
            options={[
              { value: "", label: "\u2014" },
              { value: "textual", label: "Textual" },
              { value: "tabular", label: "Tabular" },
              { value: "mixed", label: "Mixed" },
            ]}
            onChange={(val) => updateField("value_context", val)}
          />
        </FieldCell>
        <FieldCell label="Note ref">
          <FieldInput
            value={v("note_ref")}
            placeholder={placeholder("note_ref")}
            onChange={(val) => updateField("note_ref", val)}
          />
        </FieldCell>
      </FieldTriple>
      <FieldTriple>
        <FieldCell label="Scale">
          <FieldSelect
            value={v("scale")}
            options={[
              { value: "", label: "\u2014" },
              { value: "1", label: "1" },
              { value: "1000", label: "1K" },
              { value: "1000000", label: "1M" },
            ]}
            onChange={(val) =>
              updateField("scale", val ? parseInt(val, 10) : null)
            }
          />
        </FieldCell>
        <FieldCell label="Currency">
          <FieldSelect
            value={v("currency")}
            options={[
              { value: "", label: "\u2014" },
              { value: "ILS", label: "ILS" },
              { value: "USD", label: "USD" },
              { value: "EUR", label: "EUR" },
              { value: "GBP", label: "GBP" },
            ]}
            onChange={(val) => updateField("currency", val)}
          />
        </FieldCell>
        <FieldCell label="Val. type">
          <FieldSelect
            value={v("value_type")}
            options={[
              { value: "", label: "\u2014" },
              { value: "amount", label: "Amount" },
              { value: "percent", label: "%" },
              { value: "ratio", label: "Ratio" },
              { value: "count", label: "Count" },
              { value: "points", label: "Points" },
            ]}
            onChange={(val) => updateField("value_type", val)}
          />
        </FieldCell>
      </FieldTriple>
      <FieldTriple>
        <FieldCell label="End">
          <FieldInput
            value={v("period_end")}
            placeholder={placeholder("period_end") || "YYYY-MM-DD"}
            onChange={(val) => updateField("period_end", val)}
          />
        </FieldCell>
        <FieldCell label="Start">
          <FieldInput
            value={v("period_start")}
            placeholder={placeholder("period_start") || "YYYY-MM-DD"}
            onChange={(val) => updateField("period_start", val)}
          />
        </FieldCell>
        <FieldCell label="Period">
          <FieldSelect
            value={v("period_type")}
            options={[
              { value: "", label: "\u2014" },
              { value: "instant", label: "Instant" },
              { value: "duration", label: "Duration" },
              { value: "expected", label: "Expected" },
            ]}
            onChange={(val) => updateField("period_type", val)}
          />
        </FieldCell>
      </FieldTriple>

      {/* Notes */}
      <FieldRow label="Note flag">
        <FieldCheckbox
          checked={sharedBool("note_flag") ?? false}
          label="Is note"
          onChange={(val) => updateField("note_flag", val)}
        />
      </FieldRow>
      <FieldRow label="Note #">
        <FieldInput
          value={v("note_num")}
          placeholder={placeholder("note_num")}
          onChange={(val) => updateField("note_num", val)}
        />
      </FieldRow>
      <FieldRow label="Note name">
        <FieldInput
          value={v("note_name")}
          placeholder={placeholder("note_name")}
          onChange={(val) => updateField("note_name", val)}
        />
      </FieldRow>
      <FieldRow label="Comment">
        <FieldInput
          value={v("comment_ref")}
          placeholder={placeholder("comment_ref")}
          onChange={(val) => updateField("comment_ref", val)}
        />
      </FieldRow>

      {/* Path source */}
      <FieldRow label="Path src">
        <FieldSelect
          value={v("path_source")}
          options={[
            { value: "", label: "\u2014" },
            { value: "observed", label: "Observed" },
            { value: "inferred", label: "Inferred" },
          ]}
          onChange={(val) => updateField("path_source", val)}
        />
      </FieldRow>

      {/* Path editor */}
      <SubSection title="Path">
        <PathEditor
          selectedFacts={selectedFacts}
          pageName={pageName}
        />
      </SubSection>

      {/* Equations editor (single select only) */}
      {selectedFacts.length === 1 && (
        <SubSection title="Equations">
          <EquationsEditor
            factIndex={selectedFacts[0]!.index}
            record={selectedFacts[0]!.record}
            pageName={pageName}
          />
        </SubSection>
      )}

    </div>
  );
}

function SubSection({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div
      style={{
        marginTop: 8,
        padding: "10px 0",
        borderTop: "1px solid var(--surface-border)",
      }}
    >
      <h4
        style={{
          fontSize: 11,
          fontWeight: 700,
          textTransform: "uppercase",
          letterSpacing: "0.05em",
          color: "var(--text-soft)",
          marginBottom: 8,
        }}
      >
        {title}
      </h4>
      {children}
    </div>
  );
}
