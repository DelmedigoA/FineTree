/** Document-level metadata editor: language, direction, company, etc. */

import { useDocumentStore } from "../../stores/documentStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import { FieldRow, FieldTriple, FieldCell, FieldSelect, FieldInput } from "./FieldWidgets";

interface Props {
  meta: Record<string, unknown>;
}

export function DocumentMetaSection({ meta }: Props) {
  const updateDocumentMeta = useDocumentStore((s) => s.updateDocumentMeta);

  const update = (key: string, value: unknown) => {
    pushUndoSnapshot();
    updateDocumentMeta({ ...meta, [key]: value || null });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {/* Language | Direction | Year */}
      <FieldTriple>
        <FieldCell label="Language">
          <FieldSelect
            value={(meta.language as string) ?? ""}
            options={[
              { value: "", label: "\u2014" },
              { value: "he", label: "Hebrew" },
              { value: "en", label: "English" },
            ]}
            onChange={(v) => update("language", v)}
          />
        </FieldCell>
        <FieldCell label="Direction">
          <FieldSelect
            value={(meta.reading_direction as string) ?? ""}
            options={[
              { value: "", label: "\u2014" },
              { value: "rtl", label: "RTL" },
              { value: "ltr", label: "LTR" },
            ]}
            onChange={(v) => update("reading_direction", v)}
          />
        </FieldCell>
        <FieldCell label="Year">
          <FieldSelect
            value={meta.report_year != null ? String(meta.report_year) : ""}
            options={[
              { value: "", label: "\u2014" },
              ...Array.from({ length: 2027 - 1995 + 1 }, (_, i) => {
                const y = String(2027 - i);
                return { value: y, label: y };
              }),
            ]}
            onChange={(v) => update("report_year", v ? parseInt(v, 10) : null)}
          />
        </FieldCell>
      </FieldTriple>

      {/* Scope | Entity type | Company ID */}
      <FieldTriple>
        <FieldCell label="Scope">
          <FieldSelect
            value={(meta.report_scope as string) ?? ""}
            options={[
              { value: "", label: "\u2014" },
              { value: "separate", label: "Separate" },
              { value: "consolidated", label: "Consolidated" },
              { value: "combined", label: "Combined" },
              { value: "pro_forma", label: "Pro Forma" },
              { value: "other", label: "Other" },
            ]}
            onChange={(v) => update("report_scope", v)}
          />
        </FieldCell>
        <FieldCell label="Entity">
          <FieldSelect
            value={(meta.entity_type as string) ?? ""}
            options={[
              { value: "", label: "\u2014" },
              { value: "public_company", label: "Public" },
              { value: "private_company", label: "Private" },
              { value: "state_owned_enterprise", label: "State-owned" },
              { value: "registered_nonprofit", label: "Nonprofit" },
              { value: "nonprofit_npo", label: "NPO" },
              { value: "public_benefit_company", label: "Public benefit" },
              { value: "partnership", label: "Partnership" },
              { value: "limited_partnership", label: "LP" },
              { value: "limited_liability_company", label: "LLC" },
              { value: "other", label: "Other" },
            ]}
            onChange={(v) => update("entity_type", v)}
          />
        </FieldCell>
        <FieldCell label="Company ID">
          <FieldInput
            value={(meta.company_id as string) ?? ""}
            placeholder="ID"
            onChange={(v) => update("company_id", v)}
          />
        </FieldCell>
      </FieldTriple>

      {/* Company name — full width */}
      <FieldRow label="Company">
        <FieldInput
          value={(meta.company_name as string) ?? ""}
          placeholder="Company name"
          onChange={(v) => update("company_name", v)}
        />
      </FieldRow>
    </div>
  );
}
