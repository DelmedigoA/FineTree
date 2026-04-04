/** Page-level metadata editor: page type, statement type, title, status. */

import { useDocumentStore } from "../../stores/documentStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import { FieldRow, FieldSelect, FieldInput } from "./FieldWidgets";

interface Props {
  meta: Record<string, unknown>;
  pageName: string | null;
}

export function PageMetaSection({ meta, pageName }: Props) {
  const { pageStates, updatePageState } = useDocumentStore();

  const update = (key: string, value: unknown) => {
    if (!pageName) return;
    pushUndoSnapshot();
    const page = pageStates.get(pageName);
    if (!page) return;
    updatePageState(pageName, {
      ...page,
      meta: { ...page.meta, [key]: value || null },
    });
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      <FieldRow label="Entity">
        <FieldInput
          value={(meta.entity_name as string) ?? ""}
          placeholder="Entity name"
          onChange={(v) => update("entity_name", v)}
        />
      </FieldRow>
      <FieldRow label="Page #">
        <FieldInput
          value={(meta.page_num as string) ?? ""}
          placeholder="#"
          onChange={(v) => update("page_num", v)}
        />
      </FieldRow>
      <FieldRow label="Type">
        <FieldSelect
          value={(meta.page_type as string) ?? "other"}
          options={[
            { value: "title", label: "Title" },
            { value: "contents", label: "Contents" },
            { value: "declaration", label: "Declaration" },
            { value: "statements", label: "Statements" },
            { value: "other", label: "Other" },
          ]}
          onChange={(v) => update("page_type", v)}
        />
      </FieldRow>
      <FieldRow label="Statement">
        <FieldSelect
          value={(meta.statement_type as string) ?? ""}
          options={[
            { value: "", label: "\u2014" },
            { value: "balance_sheet", label: "Balance Sheet" },
            { value: "income_statement", label: "Income Statement" },
            { value: "cash_flow_statement", label: "Cash Flow" },
            {
              value: "statement_of_changes_in_equity",
              label: "Changes in Equity",
            },
            {
              value: "notes_to_financial_statements",
              label: "Notes",
            },
            { value: "board_of_directors_report", label: "Board Report" },
            { value: "auditors_report", label: "Auditors Report" },
            { value: "statement_of_activities", label: "Activities" },
            { value: "other_declaration", label: "Other Declaration" },
          ]}
          onChange={(v) => update("statement_type", v)}
        />
      </FieldRow>
      <FieldRow label="Title">
        <FieldInput
          value={(meta.title as string) ?? ""}
          placeholder="Page title"
          onChange={(v) => update("title", v)}
        />
      </FieldRow>
      <FieldRow label="Note">
        <FieldInput
          value={(meta.annotation_note as string) ?? ""}
          placeholder="Annotation note"
          onChange={(v) => update("annotation_note", v)}
        />
      </FieldRow>
      <FieldRow label="Status">
        <FieldSelect
          value={(meta.annotation_status as string) ?? ""}
          options={[
            { value: "", label: "\u2014" },
            { value: "approved", label: "Approved" },
            { value: "flagged", label: "Flagged" },
          ]}
          onChange={(v) => update("annotation_status", v)}
        />
      </FieldRow>
    </div>
  );
}
