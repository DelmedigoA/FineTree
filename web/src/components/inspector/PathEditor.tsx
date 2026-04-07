/** Path editor — reorderable hierarchy levels. Supports multi-fact selection. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import type { BoxRecord } from "../../types/schema";

interface Props {
  selectedFacts: { index: number; record: BoxRecord }[];
  pageName: string | null;
}

/** Longest common prefix of all paths (like PyQt5 _shared_path_prefix). */
function sharedPathPrefix(paths: string[][]): string[] {
  if (paths.length === 0) return [];
  const first = paths[0]!;
  let len = first.length;
  for (const p of paths.slice(1)) {
    let k = 0;
    while (k < len && k < p.length && p[k] === first[k]) k++;
    len = k;
  }
  return first.slice(0, len);
}

/** Elements of the first path that exist in ALL paths (like PyQt5 _shared_path_elements). */
function sharedPathElements(paths: string[][]): string[] {
  if (paths.length === 0) return [];
  const first = paths[0]!;
  if (paths.length === 1) return first;
  return first.filter((el) => paths.slice(1).every((p) => p.includes(el)));
}

export function PathEditor({ selectedFacts, pageName }: Props) {
  const { pageStates, updatePageState } = useDocumentStore();
  const markDirty = useCanvasStore((s) => s.markDirty);

  const isMulti = selectedFacts.length > 1;

  // Compute the display path using shared prefix logic (matches PyQt5).
  const allPaths = selectedFacts.map(
    (f) => (f.record.fact.path as string[]) ?? [],
  );
  const path = isMulti ? sharedPathPrefix(allPaths) : (allPaths[0] ?? []);
  const sharedElements = isMulti ? sharedPathElements(allPaths) : path;

  const allSamePath = isMulti
    ? selectedFacts.every(
        (f) =>
          JSON.stringify(f.record.fact.path ?? []) ===
          JSON.stringify(allPaths[0] ?? []),
      )
    : true;

  // How many levels diverge beyond the shared prefix?
  const maxOtherLen = isMulti
    ? Math.max(...allPaths.map((p) => p.length))
    : 0;
  const hasMixedBeyondPrefix = isMulti && maxOtherLen > path.length;

  const updatePath = (next: string[]) => {
    if (!pageName) return;
    pushUndoSnapshot();
    const page = pageStates.get(pageName);
    if (!page) return;
    const newFacts = [...page.facts];
    for (const { index } of selectedFacts) {
      const fact = newFacts[index];
      if (!fact) continue;
      newFacts[index] = { ...fact, fact: { ...fact.fact, path: next } };
    }
    updatePageState(pageName, { ...page, facts: newFacts });
    markDirty("bbox");
  };

  const addLevel = () => updatePath([...path, ""]);
  const removeLevel = (idx: number) => updatePath(path.filter((_, i) => i !== idx));
  const moveUp = (idx: number) => {
    if (idx <= 0) return;
    const next = [...path];
    [next[idx - 1], next[idx]] = [next[idx]!, next[idx - 1]!];
    updatePath(next);
  };
  const moveDown = (idx: number) => {
    if (idx >= path.length - 1) return;
    const next = [...path];
    [next[idx], next[idx + 1]] = [next[idx + 1]!, next[idx]!];
    updatePath(next);
  };
  const editLevel = (idx: number, value: string) => {
    const next = [...path];
    next[idx] = value;
    updatePath(next);
  };
  const invertPath = () => updatePath([...path].reverse());

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {isMulti && (
        <div style={{ fontSize: 11, color: "var(--text-soft)", marginBottom: 2 }}>
          {allSamePath
            ? `Shared path · edits apply to all ${selectedFacts.length} facts`
            : path.length > 0
              ? `Shared prefix (${path.length} level${path.length !== 1 ? "s" : ""}) · mixed paths beyond`
              : `Mixed paths · edits apply to all ${selectedFacts.length} facts`}
        </div>
      )}
      {isMulti && hasMixedBeyondPrefix && (
        <div
          style={{
            fontSize: 11,
            color: "var(--warn)",
            fontStyle: "italic",
            marginBottom: 2,
          }}
        >
          ⚠ Levels beyond prefix differ across selection
        </div>
      )}
      {path.length === 0 ? (
        <div style={{ fontSize: 12, color: "var(--text-soft)" }}>
          No path levels.
        </div>
      ) : (
        path.map((level, idx) => (
          <div
            key={idx}
            style={{
              display: "flex",
              alignItems: "center",
              gap: 4,
            }}
          >
            <span
              style={{
                fontSize: 10,
                fontFamily: "var(--font-mono)",
                color: "var(--text-soft)",
                minWidth: 16,
                textAlign: "right",
              }}
            >
              {idx + 1}
            </span>
            <input
              value={level}
              onChange={(e) => editLevel(idx, e.target.value)}
              style={{
                flex: 1,
                padding: "4px 8px",
                fontSize: 12,
                background: "var(--surface-alt)",
                border: `1px solid ${
                  isMulti && !sharedElements.includes(level)
                    ? "var(--warn)"
                    : "var(--surface-border)"
                }`,
                borderRadius: 4,
                color: isMulti && !sharedElements.includes(level)
                  ? "var(--warn)"
                  : "var(--text)",
                outline: "none",
              }}
              title={
                isMulti && !sharedElements.includes(level)
                  ? "This level is not shared across all selected facts"
                  : undefined
              }
              onFocus={(e) =>
                (e.currentTarget.style.borderColor = "var(--accent)")
              }
              onBlur={(e) => {
                e.currentTarget.style.borderColor =
                  isMulti && !sharedElements.includes(e.currentTarget.value)
                    ? "var(--warn)"
                    : "var(--surface-border)";
              }}
            />
            <MiniBtn onClick={() => moveUp(idx)} disabled={idx === 0}>
              {"\u2191"}
            </MiniBtn>
            <MiniBtn
              onClick={() => moveDown(idx)}
              disabled={idx === path.length - 1}
            >
              {"\u2193"}
            </MiniBtn>
            <MiniBtn onClick={() => removeLevel(idx)} danger>
              {"\u00D7"}
            </MiniBtn>
          </div>
        ))
      )}
      <div style={{ display: "flex", gap: 4, marginTop: 2 }}>
        <MiniBtn onClick={addLevel}>+ Add level</MiniBtn>
        {path.length > 1 && (
          <MiniBtn onClick={invertPath}>Invert</MiniBtn>
        )}
      </div>
    </div>
  );
}

function MiniBtn({
  children,
  onClick,
  disabled,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
  danger?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      style={{
        fontSize: 11,
        fontWeight: 500,
        color: danger
          ? "var(--danger)"
          : disabled
            ? "var(--text-soft)"
            : "var(--text-muted)",
        background: "transparent",
        border: `1px solid ${danger ? "var(--danger)" : "var(--surface-border)"}`,
        borderRadius: 4,
        padding: "2px 6px",
        cursor: disabled ? "default" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: "var(--transition-fast)",
      }}
    >
      {children}
    </button>
  );
}
