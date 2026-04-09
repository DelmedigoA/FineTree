/** Equation variants editor — list, add, delete, preview. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { useSelectionStore } from "../../stores/selectionStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import {
  buildEquationPreview,
  evaluateEquationString,
  equationMatchState,
} from "../../hooks/useEquationWorkflow";
import type { BoxRecord } from "../../types/schema";

interface Props {
  factIndex: number;
  record: BoxRecord;
  pageName: string | null;
}

interface EquationVariant {
  equation: string;
  fact_equation: string | null;
}

export function EquationsEditor({ factIndex, record, pageName }: Props) {
  const { pageStates, updatePageState } = useDocumentStore();
  const markDirty = useCanvasStore((s) => s.markDirty);
  const selectedIndices = useSelectionStore((s) => s.selectedIndices);
  const equationTermIndices = useSelectionStore((s) => s.equationTermIndices);
  const equationTermOrder = useSelectionStore((s) => s.equationTermOrder);
  const equationTermOperators = useSelectionStore((s) => s.equationTermOperators);
  const toggleEquationTermOperator = useSelectionStore((s) => s.toggleEquationTermOperator);

  const equations = (record.fact.equations as EquationVariant[] | null) ?? [];
  const page = pageName ? pageStates.get(pageName) : undefined;
  const hasPendingPreview =
    selectedIndices.size === 1 &&
    selectedIndices.has(factIndex) &&
    equationTermIndices.size > 0 &&
    !!page;
  const pendingTermIndices = equationTermOrder.length > 0
    ? equationTermOrder
    : [...equationTermIndices];
  const pendingPreview = hasPendingPreview && page
    ? buildEquationPreview(pendingTermIndices, page.facts, equationTermOperators)
    : null;

  const updateEquations = (next: EquationVariant[]) => {
    if (!pageName) return;
    pushUndoSnapshot();
    const page = pageStates.get(pageName);
    if (!page) return;
    const newFacts = [...page.facts];
    const fact = newFacts[factIndex];
    if (!fact) return;
    newFacts[factIndex] = {
      ...fact,
      fact: { ...fact.fact, equations: next.length > 0 ? next : null },
    };
    updatePageState(pageName, { ...page, facts: newFacts });
    markDirty("bbox");
  };

  const removeVariant = (idx: number) => {
    updateEquations(equations.filter((_, i) => i !== idx));
  };

  const moveUp = (idx: number) => {
    if (idx <= 0) return;
    const next = [...equations];
    [next[idx - 1], next[idx]] = [next[idx]!, next[idx - 1]!];
    updateEquations(next);
  };

  if (equations.length === 0 && !pendingPreview) {
    return (
      <div style={{ fontSize: 12, color: "var(--text-soft)", padding: "4px 0" }}>
        No equations. Alt+drag to select terms, click term signs to flip +/-, then press Shift to confirm.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {pendingPreview && (
        <PendingEquationPreview
          preview={pendingPreview}
          targetValue={String(record.fact.value ?? "")}
          targetNaturalSign={String(record.fact.natural_sign ?? "") || null}
          onToggleOperator={toggleEquationTermOperator}
        />
      )}
      {equations.map((eq, idx) => (
        <EquationRow
          key={idx}
          equationText={eq.equation}
          factEquationText={eq.fact_equation ?? undefined}
          targetValue={String(record.fact.value ?? "")}
          targetNaturalSign={String(record.fact.natural_sign ?? "") || null}
          onMoveUp={idx > 0 ? () => moveUp(idx) : undefined}
          onRemove={() => removeVariant(idx)}
        />
      ))}
    </div>
  );
}

function PendingEquationPreview({
  preview,
  targetValue,
  targetNaturalSign,
  onToggleOperator,
}: {
  preview: ReturnType<typeof buildEquationPreview>;
  targetValue: string;
  targetNaturalSign?: string | null;
  onToggleOperator: (index: number) => void;
}) {
  const state = equationMatchState(preview.result, targetValue, targetNaturalSign);
  const resultColor =
    state === "ok" ? "#00e500"
    : state === "mismatch" ? "#f87171"
    : "var(--text-soft)";

  return (
    <div
      style={{
        padding: "10px",
        background: "rgba(20, 184, 166, 0.08)",
        borderRadius: "var(--radius-xs)",
        border: "1px solid rgba(20, 184, 166, 0.25)",
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ fontSize: 11, fontWeight: 700, color: "var(--accent)", letterSpacing: "0.04em", textTransform: "uppercase" }}>
        Pending Preview
      </div>

      <div style={{ display: "flex", alignItems: "baseline", gap: 6, flexWrap: "wrap" }}>
        <span style={{ fontSize: 12, fontFamily: "var(--font-mono)", color: "var(--text)" }}>
          {preview.equation ?? "No calculable equation"}
        </span>
        {preview.result !== null && (
          <span style={{ color: resultColor, fontWeight: 600, fontSize: 12, whiteSpace: "nowrap" }}>
            = {preview.result.toLocaleString()}
          </span>
        )}
      </div>

      {preview.factEquation && (
        <div style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text-soft)" }}>
          {preview.factEquation}
        </div>
      )}

      <div style={{ fontSize: 11, color: "var(--text-muted)", lineHeight: 1.5 }}>
        Click a term sign to flip its manual operator before applying with Shift.
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
        {preview.terms.map((term) => (
          <div
            key={`${term.index}-${term.factNum ?? "?"}`}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 8,
              padding: "6px 8px",
              borderRadius: 6,
              background: "rgba(255,255,255,0.45)",
              border: "1px solid var(--surface-border)",
            }}
          >
            <button
              onClick={() => onToggleOperator(term.index)}
              style={{
                width: 26,
                height: 22,
                borderRadius: 999,
                border: "1px solid var(--accent)",
                background: "var(--surface)",
                color: "var(--accent)",
                fontSize: 12,
                fontWeight: 700,
                fontFamily: "var(--font-mono)",
                cursor: "pointer",
              }}
              title={`Flip ${term.factNum ? `f${term.factNum}` : "term"} sign`}
            >
              {term.operator}
            </button>
            <span style={{ fontSize: 11, fontFamily: "var(--font-mono)", color: "var(--text)", minWidth: 28 }}>
              {term.factNum ? `f${term.factNum}` : `[${term.index + 1}]`}
            </span>
            <span style={{ fontSize: 11, color: "var(--text-muted)", flex: 1 }}>
              = {term.rawValue}
              {term.status === "normalized_dash" ? " (as 0)" : term.status === "invalid" ? " (ignored)" : ""}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function EquationRow({
  equationText,
  factEquationText,
  targetValue,
  targetNaturalSign,
  onMoveUp,
  onRemove,
}: {
  equationText: string;
  factEquationText?: string;
  targetValue: string;
  targetNaturalSign?: string | null;
  onMoveUp?: () => void;
  onRemove?: () => void;
}) {
  const result = evaluateEquationString(equationText);
  const state = equationMatchState(result, targetValue, targetNaturalSign);
  const resultColor = state === "ok" ? "#00e500" : state === "mismatch" ? "#f87171" : "var(--text-soft)";

  return (
    <div
      style={{
        padding: "8px 10px",
        background: "var(--surface-alt)",
        borderRadius: "var(--radius-xs)",
        border: "1px solid var(--surface-border)",
      }}
    >
      <div
        style={{
          fontSize: 12,
          fontFamily: "var(--font-mono)",
          color: "var(--text)",
          wordBreak: "break-all",
          marginBottom: factEquationText ? 4 : (onRemove ? 4 : 0),
          display: "flex",
          alignItems: "baseline",
          gap: 6,
        }}
      >
        <span>{equationText || "\u2026"}</span>
        {result !== null && (
          <span style={{ color: resultColor, fontWeight: 600, whiteSpace: "nowrap" }}>
            = {result.toLocaleString()}
          </span>
        )}
      </div>
      {factEquationText && (
        <div
          style={{
            fontSize: 11,
            fontFamily: "var(--font-mono)",
            color: "var(--text-soft)",
            wordBreak: "break-all",
            marginBottom: onRemove ? 6 : 0,
          }}
        >
          {factEquationText}
        </div>
      )}
      {onRemove && (
        <div style={{ display: "flex", gap: 4 }}>
          {onMoveUp && <MiniBtn onClick={onMoveUp}>Move up</MiniBtn>}
          <MiniBtn onClick={onRemove} danger>Remove</MiniBtn>
        </div>
      )}
    </div>
  );
}

function MiniBtn({
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
        fontWeight: 500,
        color: danger ? "var(--danger)" : "var(--text-muted)",
        background: "transparent",
        border: `1px solid ${danger ? "var(--danger)" : "var(--surface-border)"}`,
        borderRadius: 4,
        padding: "2px 8px",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
  );
}
