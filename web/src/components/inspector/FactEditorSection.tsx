/** Full fact editor — all ~20+ fields, compact layout. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import {
  FieldRow,
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

      {/* Core fields */}
      <FieldRow label="Value">
        <FieldInput
          value={v("value")}
          placeholder={placeholder("value") || "Fact value"}
          onChange={(val) => updateField("value", val)}
        />
      </FieldRow>
      <FieldRow label="Row role">
        <FieldSelect
          value={v("row_role") || "detail"}
          options={[
            { value: "detail", label: "Detail" },
            { value: "total", label: "Total" },
          ]}
          onChange={(val) => updateField("row_role", val)}
        />
      </FieldRow>
      <FieldRow label="Nat. sign">
        <FieldSelect
          value={v("natural_sign")}
          options={[
            { value: "", label: "\u2014 (auto)" },
            { value: "positive", label: "Positive" },
            { value: "negative", label: "Negative" },
          ]}
          onChange={(val) => updateField("natural_sign", val)}
        />
      </FieldRow>
      <FieldRow label="Fact #">
        <FieldInput
          value={v("fact_num")}
          placeholder={placeholder("fact_num")}
          onChange={(val) =>
            updateField("fact_num", val ? parseInt(val, 10) || null : null)
          }
        />
      </FieldRow>

      {/* Value context */}
      <FieldRow label="Val. type">
        <FieldSelect
          value={v("value_type")}
          options={[
            { value: "", label: "\u2014" },
            { value: "amount", label: "Amount" },
            { value: "percent", label: "Percent" },
            { value: "ratio", label: "Ratio" },
            { value: "count", label: "Count" },
            { value: "points", label: "Points" },
          ]}
          onChange={(val) => updateField("value_type", val)}
        />
      </FieldRow>
      <FieldRow label="Context">
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
      </FieldRow>

      {/* Currency & scale */}
      <FieldRow label="Currency">
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
      </FieldRow>
      <FieldRow label="Scale">
        <FieldSelect
          value={v("scale")}
          options={[
            { value: "", label: "\u2014" },
            { value: "1", label: "1" },
            { value: "1000", label: "1,000" },
            { value: "1000000", label: "1,000,000" },
          ]}
          onChange={(val) =>
            updateField("scale", val ? parseInt(val, 10) : null)
          }
        />
      </FieldRow>

      {/* Date fields */}
      <FieldRow label="Date">
        <FieldInput
          value={v("date")}
          placeholder={placeholder("date") || "YYYY-MM-DD"}
          onChange={(val) => updateField("date", val)}
        />
      </FieldRow>
      <FieldRow label="Period">
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
      </FieldRow>
      <FieldRow label="Start">
        <FieldInput
          value={v("period_start")}
          placeholder={placeholder("period_start") || "YYYY-MM-DD"}
          onChange={(val) => updateField("period_start", val)}
        />
      </FieldRow>
      <FieldRow label="End">
        <FieldInput
          value={v("period_end")}
          placeholder={placeholder("period_end") || "YYYY-MM-DD"}
          onChange={(val) => updateField("period_end", val)}
        />
      </FieldRow>
      <FieldRow label="Duration">
        <FieldSelect
          value={v("duration_type")}
          options={[
            { value: "", label: "\u2014" },
            { value: "recurrent", label: "Recurrent" },
          ]}
          onChange={(val) => updateField("duration_type", val)}
        />
      </FieldRow>
      <FieldRow label="Recurring">
        <FieldSelect
          value={v("recurring_period")}
          options={[
            { value: "", label: "\u2014" },
            { value: "daily", label: "Daily" },
            { value: "quarterly", label: "Quarterly" },
            { value: "monthly", label: "Monthly" },
            { value: "yearly", label: "Yearly" },
          ]}
          onChange={(val) => updateField("recurring_period", val)}
        />
      </FieldRow>

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
      <FieldRow label="Note ref">
        <FieldInput
          value={v("note_ref")}
          placeholder={placeholder("note_ref")}
          onChange={(val) => updateField("note_ref", val)}
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

      {/* Path editor (single select only) */}
      {selectedFacts.length === 1 && (
        <SubSection title="Path">
          <PathEditor
            factIndex={selectedFacts[0]!.index}
            record={selectedFacts[0]!.record}
            pageName={pageName}
          />
        </SubSection>
      )}

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

      {/* BBox display (read-only) */}
      {selectedFacts.length === 1 && (
        <FieldRow label="BBox">
          <div
            style={{
              fontSize: 11,
              fontFamily: "var(--font-mono)",
              color: "var(--text-soft)",
              background: "var(--surface-alt)",
              padding: "4px 8px",
              borderRadius: "var(--radius-xs)",
            }}
          >
            {`${Math.round(selectedFacts[0]!.record.bbox.x)}, ${Math.round(selectedFacts[0]!.record.bbox.y)} \u2014 ${Math.round(selectedFacts[0]!.record.bbox.w)}\u00d7${Math.round(selectedFacts[0]!.record.bbox.h)}`}
          </div>
        </FieldRow>
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
