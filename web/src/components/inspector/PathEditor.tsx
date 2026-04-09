/** Path editor — reorderable hierarchy levels. Supports multi-fact selection. */

import { useDocumentStore } from "../../stores/documentStore";
import { useCanvasStore } from "../../stores/canvasStore";
import { pushUndoSnapshot } from "../../hooks/useUndoRedo";
import type { BoxRecord } from "../../types/schema";

interface Props {
  selectedFacts: { index: number; record: BoxRecord }[];
  pageName: string | null;
}

interface PathRow {
  key: string;
  label: string;
  indicesByPath: number[];
  isPrefix: boolean;
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

function pathRowsEqual(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function occurrenceIndices(path: string[]): Map<string, number[]> {
  const indices = new Map<string, number[]>();
  path.forEach((label, index) => {
    const existing = indices.get(label);
    if (existing) {
      existing.push(index);
    } else {
      indices.set(label, [index]);
    }
  });
  return indices;
}

function buildSharedRows(paths: string[][]): PathRow[] {
  if (paths.length === 0) return [];
  if (paths.length === 1) {
    return paths[0]!.map((label, index) => ({
      key: `single-${index}`,
      label,
      indicesByPath: [index],
      isPrefix: true,
    }));
  }

  const firstPath = paths[0]!;
  const prefixLength = sharedPathPrefix(paths).length;
  const indicesByLabel = paths.map(occurrenceIndices);
  const seenCounts = new Map<string, number>();

  return firstPath.flatMap((label, firstIndex) => {
    const occurrence = seenCounts.get(label) ?? 0;
    seenCounts.set(label, occurrence + 1);

    const indicesByPath = indicesByLabel.map(
      (map) => map.get(label)?.[occurrence] ?? -1,
    );
    if (indicesByPath.some((index) => index < 0)) {
      return [];
    }

    return [{
      key: `${label}-${occurrence}-${firstIndex}`,
      label,
      indicesByPath,
      isPrefix:
        firstIndex < prefixLength &&
        indicesByPath.every((index) => index === firstIndex),
    }];
  });
}

function buildVariantTails(paths: string[][], rows: PathRow[]): string[] {
  const rendered = paths.map((path, pathIndex) => {
    const matched = new Set(
      rows
        .map((row) => row.indicesByPath[pathIndex] ?? -1)
        .filter((index) => index >= 0),
    );
    return path
      .filter((_, index) => !matched.has(index))
      .join(" > ");
  });

  return Array.from(new Set(rendered)).filter(Boolean);
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

  const allSamePath = isMulti
    ? selectedFacts.every(
        (f) =>
          JSON.stringify(f.record.fact.path ?? []) ===
          JSON.stringify(allPaths[0] ?? []),
      )
    : true;

  const sharedRows = isMulti && !allSamePath
    ? buildSharedRows(allPaths)
    : (allPaths[0] ?? []).map((label, index) => ({
        key: `path-${index}`,
        label,
        indicesByPath: [index],
        isPrefix: true,
      }));
  const displayRows = sharedRows;
  const variantTails = isMulti && !allSamePath
    ? buildVariantTails(allPaths, sharedRows)
    : [];
  const canAddLevel = !isMulti || allSamePath;
  const canInvert = allPaths.some((currentPath) => currentPath.length > 1);

  const updateSelectedPaths = (
    updater: (currentPath: string[], selectedPathIndex: number) => string[],
  ) => {
    if (!pageName) return;
    const page = pageStates.get(pageName);
    if (!page) return;

    const newFacts = [...page.facts];
    let changed = false;

    for (const [selectedPathIndex, { index }] of selectedFacts.entries()) {
      const fact = newFacts[index];
      if (!fact) continue;
      const currentPath = [...((fact.fact.path as string[]) ?? [])];
      const nextPath = updater(currentPath, selectedPathIndex);
      if (pathRowsEqual(currentPath, nextPath)) {
        continue;
      }
      newFacts[index] = { ...fact, fact: { ...fact.fact, path: nextPath } };
      changed = true;
    }

    if (!changed) return;

    pushUndoSnapshot();
    updatePageState(pageName, { ...page, facts: newFacts });
    markDirty("bbox");
  };

  /** Replace the entire path for all selected facts (safe when all paths are identical). */
  const updateFullPath = (next: string[]) => {
    updateSelectedPaths(() => [...next]);
  };

  const renameRow = (row: PathRow, value: string) => {
    if (!isMulti || allSamePath) {
      const next = [...path];
      const targetIndex = row.indicesByPath[0] ?? -1;
      if (targetIndex < 0 || targetIndex >= next.length) return;
      next[targetIndex] = value;
      updateFullPath(next);
      return;
    }

    updateSelectedPaths((currentPath, selectedPathIndex) => {
      const targetIndex = row.indicesByPath[selectedPathIndex] ?? -1;
      if (targetIndex < 0 || targetIndex >= currentPath.length) {
        return currentPath;
      }
      const next = [...currentPath];
      next[targetIndex] = value;
      return next;
    });
  };

  const addLevel = () => updateFullPath([...path, ""]);
  const removeRow = (row: PathRow) => {
    if (!isMulti || allSamePath) {
      const targetIndex = row.indicesByPath[0] ?? -1;
      if (targetIndex < 0) return;
      updateFullPath(path.filter((_, index) => index !== targetIndex));
      return;
    }

    updateSelectedPaths((currentPath, selectedPathIndex) => {
      const targetIndex = row.indicesByPath[selectedPathIndex] ?? -1;
      if (targetIndex < 0 || targetIndex >= currentPath.length) {
        return currentPath;
      }
      return currentPath.filter((_, index) => index !== targetIndex);
    });
  };
  const moveRow = (row: PathRow, direction: -1 | 1) => {
    if (!isMulti || allSamePath) {
      const targetIndex = row.indicesByPath[0] ?? -1;
      const swapIndex = targetIndex + direction;
      if (targetIndex < 0 || swapIndex < 0 || swapIndex >= path.length) return;
      const next = [...path];
      [next[targetIndex], next[swapIndex]] = [next[swapIndex]!, next[targetIndex]!];
      updateFullPath(next);
      return;
    }

    updateSelectedPaths((currentPath, selectedPathIndex) => {
      const targetIndex = row.indicesByPath[selectedPathIndex] ?? -1;
      const swapIndex = targetIndex + direction;
      if (targetIndex < 0 || swapIndex < 0 || swapIndex >= currentPath.length) {
        return currentPath;
      }
      const next = [...currentPath];
      [next[targetIndex], next[swapIndex]] = [next[swapIndex]!, next[targetIndex]!];
      return next;
    });
  };
  const canMoveRow = (row: PathRow, direction: -1 | 1) =>
    row.indicesByPath.every((index, pathIndex) => {
      const nextIndex = index + direction;
      return nextIndex >= 0 && nextIndex < (allPaths[pathIndex]?.length ?? 0);
    });
  const invertPath = () => {
    updateSelectedPaths((currentPath) => [...currentPath].reverse());
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {displayRows.length === 0 ? (
        <div style={{ fontSize: 12, color: "var(--text-soft)" }}>
          {isMulti && !allSamePath ? "No shared path levels." : "No path levels."}
        </div>
      ) : (
        displayRows.map((row, idx) => (
          <div
            key={row.key}
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
              value={row.label}
              onChange={(e) => renameRow(row, e.target.value)}
              style={{
                flex: 1,
                padding: "4px 8px",
                fontSize: 12,
                background: "var(--surface-alt)",
                border: "1px solid var(--surface-border)",
                borderRadius: 4,
                color: isMulti && !allSamePath ? "var(--ok)" : "var(--text)",
                outline: "none",
              }}
              onFocus={(e) =>
                (e.currentTarget.style.borderColor = "var(--accent)")
              }
              onBlur={(e) => {
                e.currentTarget.style.borderColor = "var(--surface-border)";
              }}
            />
            <MiniBtn onClick={() => moveRow(row, -1)} disabled={!canMoveRow(row, -1)}>
              {"\u2191"}
            </MiniBtn>
            <MiniBtn
              onClick={() => moveRow(row, 1)}
              disabled={!canMoveRow(row, 1)}
            >
              {"\u2193"}
            </MiniBtn>
            <MiniBtn onClick={() => removeRow(row)} danger>
              {"\u00D7"}
            </MiniBtn>
          </div>
        ))
      )}
      {/* Variant tail nodes (read-only) */}
      {variantTails.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 3, marginTop: 4 }}>
          <span style={{ fontSize: 11, color: "var(--text-soft)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Variants ({variantTails.length})
          </span>
          {variantTails.map((tail, i) => (
            <div
              key={`tail-${i}`}
              style={{
                padding: "5px 9px",
                fontSize: 13,
                background: "var(--variant-tail-bg)",
                borderLeft: "3px solid var(--variant-tail-border)",
                borderRadius: 4,
                color: "var(--variant-tail-text)",
                fontFamily: "var(--font-mono)",
              }}
            >
              {tail}
            </div>
          ))}
        </div>
      )}
      <div style={{ display: "flex", gap: 4, marginTop: 2 }}>
        <MiniBtn onClick={addLevel} disabled={!canAddLevel}>+ Add level</MiniBtn>
        {canInvert && (
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
