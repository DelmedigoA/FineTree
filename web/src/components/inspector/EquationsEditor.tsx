/** Equation variants editor — list, add, delete, preview. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
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

  const equations = (record.fact.equations as EquationVariant[] | null) ?? [];

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

  if (equations.length === 0) {
    return (
      <div
        style={{
          fontSize: 12,
          color: "var(--text-soft)",
          padding: "4px 0",
        }}
      >
        No equations. Use Alt+drag to assign.
      </div>
    );
  }

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {equations.map((eq, idx) => (
        <div
          key={idx}
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
              marginBottom: 4,
            }}
          >
            {eq.equation}
          </div>
          {eq.fact_equation && (
            <div
              style={{
                fontSize: 11,
                fontFamily: "var(--font-mono)",
                color: "var(--text-soft)",
                wordBreak: "break-all",
                marginBottom: 6,
              }}
            >
              {eq.fact_equation}
            </div>
          )}
          <div style={{ display: "flex", gap: 4 }}>
            {idx > 0 && (
              <MiniBtn onClick={() => moveUp(idx)}>Move up</MiniBtn>
            )}
            <MiniBtn onClick={() => removeVariant(idx)} danger>
              Remove
            </MiniBtn>
          </div>
        </div>
      ))}
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
