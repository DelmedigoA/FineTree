import { useEffect, useRef, useState, type CSSProperties, type PointerEvent as ReactPointerEvent } from "react";

import { clampBBox, createEmptyFact, normalizeBBox } from "./documentState";
import type { ApiBBox, FactDraft, PageDraft } from "./types";

type Interaction =
  | null
  | {
      type: "draw";
      origin: { x: number; y: number };
      current: { x: number; y: number };
    }
  | {
      type: "move";
      factId: string;
      origin: { x: number; y: number };
      startBBox: ApiBBox;
    }
  | {
      type: "resize";
      factId: string;
      handle: Handle;
      origin: { x: number; y: number };
      startBBox: ApiBBox;
    };

type Handle = "n" | "s" | "e" | "w" | "nw" | "ne" | "sw" | "se";

type AnnotationCanvasProps = {
  docId: string;
  page: PageDraft;
  selectedIds: string[];
  mode: "select" | "draw";
  zoom: number;
  onSelect(ids: string[]): void;
  onCommitFacts(facts: FactDraft[]): void;
};

const HANDLE_POSITIONS: Array<{ handle: Handle; style: CSSProperties }> = [
  { handle: "n", style: { top: -6, left: "50%", transform: "translateX(-50%)" } },
  { handle: "s", style: { bottom: -6, left: "50%", transform: "translateX(-50%)" } },
  { handle: "e", style: { right: -6, top: "50%", transform: "translateY(-50%)" } },
  { handle: "w", style: { left: -6, top: "50%", transform: "translateY(-50%)" } },
  { handle: "nw", style: { left: -6, top: -6 } },
  { handle: "ne", style: { right: -6, top: -6 } },
  { handle: "sw", style: { left: -6, bottom: -6 } },
  { handle: "se", style: { right: -6, bottom: -6 } }
];

export function AnnotationCanvas({
  docId,
  page,
  selectedIds,
  mode,
  zoom,
  onSelect,
  onCommitFacts
}: AnnotationCanvasProps) {
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [interaction, setInteraction] = useState<Interaction>(null);
  const [workingFacts, setWorkingFacts] = useState<FactDraft[] | null>(null);
  const factsRef = useRef(page.facts);

  useEffect(() => {
    factsRef.current = page.facts;
    setWorkingFacts(null);
    setInteraction(null);
  }, [page.image, page.facts]);

  useEffect(() => {
    if (!interaction) {
      return undefined;
    }

    const handleMove = (event: PointerEvent) => {
      const point = pointFromPointer(event, stageRef.current, zoom);
      if (!point) {
        return;
      }
      if (interaction.type === "draw") {
        setInteraction({ ...interaction, current: point });
        return;
      }
      const currentFacts = workingFacts ?? factsRef.current;
      const nextFacts = currentFacts.map((fact) => {
        if (fact.id !== interaction.factId) {
          return fact;
        }
        if (interaction.type === "move") {
          const dx = point.x - interaction.origin.x;
          const dy = point.y - interaction.origin.y;
          return {
            ...fact,
            bbox: clampBBox(
              {
                ...interaction.startBBox,
                x: interaction.startBBox.x + dx,
                y: interaction.startBBox.y + dy
              },
              page.width,
              page.height
            )
          };
        }
        return {
          ...fact,
          bbox: clampBBox(resizeBBox(interaction.startBBox, interaction.handle, point.x - interaction.origin.x, point.y - interaction.origin.y), page.width, page.height)
        };
      });
      setWorkingFacts(nextFacts);
    };

    const handleUp = () => {
      if (interaction.type === "draw") {
        const bbox = normalizeBBox({
          x: Math.min(interaction.origin.x, interaction.current.x),
          y: Math.min(interaction.origin.y, interaction.current.y),
          w: Math.abs(interaction.current.x - interaction.origin.x),
          h: Math.abs(interaction.current.y - interaction.origin.y)
        });
        if (bbox.w >= 4 && bbox.h >= 4) {
          const nextFact = createEmptyFact(bbox);
          onCommitFacts([...page.facts, nextFact]);
          onSelect([nextFact.id]);
        }
      } else if (workingFacts) {
        onCommitFacts(workingFacts);
      }
      setWorkingFacts(null);
      setInteraction(null);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", handleUp, { once: true });
    return () => {
      window.removeEventListener("pointermove", handleMove);
      window.removeEventListener("pointerup", handleUp);
    };
  }, [interaction, onCommitFacts, onSelect, page.facts, page.height, page.width, workingFacts, zoom]);

  const displayFacts = workingFacts ?? page.facts;

  return (
    <div className="canvas-shell">
      <div className="canvas-scroll">
        <div
          ref={stageRef}
          className={`canvas-stage canvas-stage--${mode}`}
          style={{ width: page.width * zoom, height: page.height * zoom }}
          onPointerDown={(event) => {
            if (mode !== "draw") {
              onSelect([]);
              return;
            }
            const point = pointFromReactEvent(event, stageRef.current, zoom);
            if (!point) {
              return;
            }
            onSelect([]);
            setInteraction({
              type: "draw",
              origin: point,
              current: point
            });
          }}
        >
          <img
            className="canvas-image"
            src={`/api/documents/${docId}/pages/${page.image}/image`}
            alt={page.image}
            draggable={false}
            style={{ width: page.width * zoom, height: page.height * zoom }}
          />
          {displayFacts.map((fact) => {
            const selected = selectedIds.includes(fact.id);
            return (
              <div
                key={fact.id}
                className={`fact-box${selected ? " fact-box--selected" : ""}`}
                style={boxStyle(fact.bbox, zoom)}
                onPointerDown={(event) => {
                  event.stopPropagation();
                  if (mode === "draw") {
                    return;
                  }
                  const nextSelection =
                    event.shiftKey
                      ? selected
                        ? selectedIds.filter((id) => id !== fact.id)
                        : [...selectedIds, fact.id]
                      : [fact.id];
                  onSelect(nextSelection);
                  const point = pointFromReactEvent(event, stageRef.current, zoom);
                  if (!point) {
                    return;
                  }
                  setInteraction({
                    type: "move",
                    factId: fact.id,
                    origin: point,
                    startBBox: fact.bbox
                  });
                }}
              >
                <span className="fact-box__label">{fact.value || "New"}</span>
                {selected &&
                  HANDLE_POSITIONS.map(({ handle, style }) => (
                    <button
                      key={handle}
                      className="fact-handle"
                      style={style}
                      onPointerDown={(event) => {
                        event.stopPropagation();
                        if (mode === "draw") {
                          return;
                        }
                        const point = pointFromReactEvent(event, stageRef.current, zoom);
                        if (!point) {
                          return;
                        }
                        onSelect([fact.id]);
                        setInteraction({
                          type: "resize",
                          factId: fact.id,
                          handle,
                          origin: point,
                          startBBox: fact.bbox
                        });
                      }}
                      aria-label={`Resize ${handle}`}
                      type="button"
                    />
                  ))}
              </div>
            );
          })}
          {interaction?.type === "draw" ? (
            <div
              className="fact-box fact-box--draft"
              style={boxStyle(
                {
                  x: Math.min(interaction.origin.x, interaction.current.x),
                  y: Math.min(interaction.origin.y, interaction.current.y),
                  w: Math.abs(interaction.current.x - interaction.origin.x),
                  h: Math.abs(interaction.current.y - interaction.origin.y)
                },
                zoom
              )}
            />
          ) : null}
        </div>
      </div>
    </div>
  );
}

function pointFromReactEvent(
  event: ReactPointerEvent<HTMLElement>,
  stage: HTMLDivElement | null,
  zoom: number
): { x: number; y: number } | null {
  return pointFromClient(event.clientX, event.clientY, stage, zoom);
}

function pointFromPointer(
  event: PointerEvent,
  stage: HTMLDivElement | null,
  zoom: number
): { x: number; y: number } | null {
  return pointFromClient(event.clientX, event.clientY, stage, zoom);
}

function pointFromClient(
  clientX: number,
  clientY: number,
  stage: HTMLDivElement | null,
  zoom: number
): { x: number; y: number } | null {
  if (!stage) {
    return null;
  }
  const rect = stage.getBoundingClientRect();
  return {
    x: (clientX - rect.left) / zoom,
    y: (clientY - rect.top) / zoom
  };
}

function boxStyle(bbox: ApiBBox, zoom: number): CSSProperties {
  return {
    left: bbox.x * zoom,
    top: bbox.y * zoom,
    width: bbox.w * zoom,
    height: bbox.h * zoom
  };
}

function resizeBBox(startBBox: ApiBBox, handle: Handle, dx: number, dy: number): ApiBBox {
  const next = { ...startBBox };
  if (handle.includes("w")) {
    next.x = startBBox.x + dx;
    next.w = startBBox.w - dx;
  }
  if (handle.includes("e")) {
    next.w = startBBox.w + dx;
  }
  if (handle.includes("n")) {
    next.y = startBBox.y + dy;
    next.h = startBBox.h - dy;
  }
  if (handle.includes("s")) {
    next.h = startBBox.h + dy;
  }
  return next;
}
