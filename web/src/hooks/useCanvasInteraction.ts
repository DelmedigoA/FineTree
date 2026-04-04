/** Core canvas interaction hook: select, move, resize, draw, pan, duplicate. */

import { useCallback, useRef, useEffect } from "react";
import { useCanvasStore } from "../stores/canvasStore";
import { useSelectionStore } from "../stores/selectionStore";
import { useDocumentStore } from "../stores/documentStore";
import { hitTest, cursorForHandle } from "../canvas/HitTester";
import { createTransform } from "../canvas/WorldTransform";
import { pushUndoSnapshot } from "./useUndoRedo";
import type { HandlePosition } from "../types/canvas";
import type { BBox } from "../types/schema";
import type { BoxRecord, PageState } from "../types/schema";

interface DragContext {
  mode:
    | "none"
    | "select-rect"
    | "move"
    | "resize"
    | "draw"
    | "pan"
    | "equation-sweep"
    | "duplicate";
  startScreenX: number;
  startScreenY: number;
  startWorldX: number;
  startWorldY: number;
  /** For resize: which handle is being dragged. */
  handle?: HandlePosition;
  /** Original bboxes at drag start (for move/resize). */
  origBboxes?: Map<number, BBox>;
  /** For pan: starting pan offset. */
  origPanX?: number;
  origPanY?: number;
  /** Has the drag passed the 4px threshold? */
  dragging: boolean;
}

const DRAG_THRESHOLD = 4;
const MIN_BBOX_SIZE = 4;

export function useCanvasInteraction(
  canvasRef: React.RefObject<HTMLCanvasElement | null>,
) {
  const dragRef = useRef<DragContext>({
    mode: "none",
    startScreenX: 0,
    startScreenY: 0,
    startWorldX: 0,
    startWorldY: 0,
    dragging: false,
  });

  const getTransform = useCallback(() => {
    const { zoom, panX, panY } = useCanvasStore.getState();
    return createTransform(zoom, panX, panY);
  }, []);

  const getCurrentPageState = useCallback((): PageState | undefined => {
    const { pageStates, pageNames, currentPageIndex } =
      useDocumentStore.getState();
    const pageName = pageNames[currentPageIndex];
    return pageName ? pageStates.get(pageName) : undefined;
  }, []);

  const getBboxes = useCallback((): BBox[] => {
    const page = getCurrentPageState();
    if (!page) return [];
    return page.facts.map((f) => f.bbox);
  }, [getCurrentPageState]);

  const updateBbox = useCallback((index: number, bbox: BBox) => {
    const { pageStates, pageNames, currentPageIndex, updatePageState } =
      useDocumentStore.getState();
    const pageName = pageNames[currentPageIndex];
    if (!pageName) return;
    const page = pageStates.get(pageName);
    if (!page) return;
    const newFacts = [...page.facts];
    const fact = newFacts[index];
    if (!fact) return;
    newFacts[index] = { ...fact, bbox };
    updatePageState(pageName, { ...page, facts: newFacts });
    useCanvasStore.getState().markDirty("bbox");
  }, []);

  const addNewBbox = useCallback((bbox: BBox) => {
    pushUndoSnapshot();
    const { pageStates, pageNames, currentPageIndex, updatePageState } =
      useDocumentStore.getState();
    const pageName = pageNames[currentPageIndex];
    if (!pageName) return;
    const page = pageStates.get(pageName);
    if (!page) return;
    const newFact: BoxRecord = {
      bbox,
      fact: {
        value: "",
        fact_num: null,
        equations: null,
        natural_sign: null,
        row_role: "detail",
        comment_ref: null,
        note_flag: false,
        note_name: null,
        note_num: null,
        note_ref: null,
        date: null,
        period_type: null,
        period_start: null,
        period_end: null,
        duration_type: null,
        recurring_period: null,
        path: [],
        path_source: null,
        currency: null,
        scale: null,
        value_type: null,
        value_context: null,
      },
    };
    const newFacts = [...page.facts, newFact];
    updatePageState(pageName, { ...page, facts: newFacts });
    // Select the newly created bbox.
    useSelectionStore.getState().select(newFacts.length - 1);
    useCanvasStore.getState().markDirty("bbox");
  }, []);

  const handleMouseDown = useCallback(
    (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const transform = getTransform();
      const [wx, wy] = transform.toWorld(sx, sy);

      const bboxes = getBboxes();
      const hit = hitTest(sx, sy, bboxes, transform);
      const selection = useSelectionStore.getState();
      const isEquationMode = selection.equationModeActive;

      const ctx: DragContext = {
        mode: "none",
        startScreenX: sx,
        startScreenY: sy,
        startWorldX: wx,
        startWorldY: wy,
        dragging: false,
      };

      // Middle/right click = pan.
      if (e.button === 1 || e.button === 2) {
        const { panX, panY } = useCanvasStore.getState();
        ctx.mode = "pan";
        ctx.origPanX = panX;
        ctx.origPanY = panY;
        dragRef.current = ctx;
        e.preventDefault();
        return;
      }

      // Equation mode: Alt+click toggles term.
      if (isEquationMode && hit.type !== "none") {
        selection.toggleEquationTerm(hit.factIndex);
        useCanvasStore.getState().markDirty("bbox");
        // Start equation sweep drag context.
        ctx.mode = "equation-sweep";
        dragRef.current = ctx;
        return;
      }

      // Ctrl+drag on empty = draw new bbox.
      if ((e.ctrlKey || e.metaKey) && hit.type === "none") {
        ctx.mode = "draw";
        dragRef.current = ctx;
        return;
      }

      // Click on handle = resize.
      if (hit.type === "handle" && selection.selectedIndices.has(hit.factIndex)) {
        pushUndoSnapshot();
        ctx.mode = "resize";
        ctx.handle = hit.handle;
        ctx.origBboxes = new Map();
        for (const idx of selection.selectedIndices) {
          const b = bboxes[idx];
          if (b) ctx.origBboxes.set(idx, { ...b });
        }
        dragRef.current = ctx;
        return;
      }

      // Cmd/Ctrl+click on body = duplicate + drag copy.
      if ((e.metaKey || e.ctrlKey) && hit.type === "body") {
        pushUndoSnapshot();
        const { pageStates, pageNames, currentPageIndex, updatePageState } =
          useDocumentStore.getState();
        const pageName = pageNames[currentPageIndex];
        if (pageName) {
          const page = pageStates.get(pageName);
          if (page) {
            const original = page.facts[hit.factIndex];
            if (original) {
              const clone: BoxRecord = {
                bbox: { ...original.bbox },
                fact: { ...original.fact },
              };
              const newFacts = [...page.facts, clone];
              updatePageState(pageName, { ...page, facts: newFacts });
              useSelectionStore.getState().select(newFacts.length - 1);
              useCanvasStore.getState().markDirty("bbox");
              // Set up move drag for the clone.
              ctx.mode = "duplicate";
              ctx.origBboxes = new Map([[newFacts.length - 1, { ...clone.bbox }]]);
            }
          }
        }
        dragRef.current = ctx;
        return;
      }

      // Click on body = select + prepare move.
      if (hit.type === "body" || hit.type === "handle") {
        pushUndoSnapshot();
        if (e.shiftKey) {
          selection.toggleSelect(hit.factIndex);
        } else if (!selection.selectedIndices.has(hit.factIndex)) {
          selection.select(hit.factIndex);
        }
        ctx.mode = "move";
        ctx.origBboxes = new Map();
        // Snapshot after selection update.
        const sel = useSelectionStore.getState().selectedIndices;
        for (const idx of sel) {
          const b = bboxes[idx];
          if (b) ctx.origBboxes.set(idx, { ...b });
        }
        dragRef.current = ctx;
        useCanvasStore.getState().markDirty("bbox");
        return;
      }

      // Click on empty space = rubber band select (or clear).
      if (hit.type === "none") {
        if (!e.shiftKey) {
          selection.clearSelection();
          useCanvasStore.getState().markDirty("bbox");
        }
        ctx.mode = "select-rect";
        dragRef.current = ctx;
        return;
      }

      dragRef.current = ctx;
    },
    [canvasRef, getTransform, getBboxes],
  );

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const rect = canvas.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const ctx = dragRef.current;

      // Hover hit-test when not dragging.
      if (ctx.mode === "none") {
        const transform = getTransform();
        const bboxes = getBboxes();
        const hit = hitTest(sx, sy, bboxes, transform);
        const selection = useSelectionStore.getState();

        if (hit.type === "handle") {
          canvas.style.cursor = cursorForHandle(hit.handle!);
        } else if (hit.type === "body") {
          canvas.style.cursor = "move";
        } else {
          canvas.style.cursor = "crosshair";
        }

        if (hit.factIndex !== selection.hoveredIndex) {
          selection.setHovered(hit.type !== "none" ? hit.factIndex : null);
          useCanvasStore.getState().markDirty("interaction");
        }
        return;
      }

      const dx = sx - ctx.startScreenX;
      const dy = sy - ctx.startScreenY;

      // Check drag threshold.
      if (!ctx.dragging) {
        if (Math.abs(dx) < DRAG_THRESHOLD && Math.abs(dy) < DRAG_THRESHOLD) {
          return;
        }
        ctx.dragging = true;
      }

      const transform = getTransform();
      const [wx, wy] = transform.toWorld(sx, sy);

      switch (ctx.mode) {
        case "pan": {
          const newPanX = (ctx.origPanX ?? 0) + dx;
          const newPanY = (ctx.origPanY ?? 0) + dy;
          useCanvasStore.getState().setPan(newPanX, newPanY);
          break;
        }

        case "move":
        case "duplicate": {
          if (!ctx.origBboxes) break;
          const dwx = wx - ctx.startWorldX;
          const dwy = wy - ctx.startWorldY;
          for (const [idx, orig] of ctx.origBboxes) {
            updateBbox(idx, {
              x: Math.max(0, orig.x + dwx),
              y: Math.max(0, orig.y + dwy),
              w: orig.w,
              h: orig.h,
            });
          }
          break;
        }

        case "resize": {
          if (!ctx.origBboxes || !ctx.handle) break;
          for (const [idx, orig] of ctx.origBboxes) {
            const newBbox = computeResize(orig, ctx.handle, wx, wy, ctx.startWorldX, ctx.startWorldY);
            updateBbox(idx, newBbox);
          }
          break;
        }

        case "draw": {
          // Show draw preview via interaction layer.
          const x = Math.min(ctx.startWorldX, wx);
          const y = Math.min(ctx.startWorldY, wy);
          const w = Math.abs(wx - ctx.startWorldX);
          const h = Math.abs(wy - ctx.startWorldY);
          useSelectionStore.getState().setSelectionRect({
            startX: x,
            startY: y,
            endX: x + w,
            endY: y + h,
          });
          useCanvasStore.getState().markDirty("interaction");
          break;
        }

        case "select-rect": {
          const x1 = Math.min(ctx.startWorldX, wx);
          const y1 = Math.min(ctx.startWorldY, wy);
          const x2 = Math.max(ctx.startWorldX, wx);
          const y2 = Math.max(ctx.startWorldY, wy);
          useSelectionStore.getState().setSelectionRect({
            startX: x1,
            startY: y1,
            endX: x2,
            endY: y2,
          });

          // Select bboxes intersecting the rect.
          const bboxes = getBboxes();
          const intersecting: number[] = [];
          for (let i = 0; i < bboxes.length; i++) {
            const b = bboxes[i]!;
            if (
              b.x < x2 &&
              b.x + b.w > x1 &&
              b.y < y2 &&
              b.y + b.h > y1
            ) {
              intersecting.push(i);
            }
          }
          if (e.shiftKey) {
            useSelectionStore.getState().addToSelection(intersecting);
          } else {
            const newSet = new Set(intersecting);
            // Only update if changed.
            useSelectionStore.setState({ selectedIndices: newSet });
          }
          useCanvasStore.getState().markDirty("bbox");
          useCanvasStore.getState().markDirty("interaction");
          break;
        }

        case "equation-sweep": {
          // Sweep-select equation terms intersecting drag rect.
          const x1 = Math.min(ctx.startWorldX, wx);
          const y1 = Math.min(ctx.startWorldY, wy);
          const x2 = Math.max(ctx.startWorldX, wx);
          const y2 = Math.max(ctx.startWorldY, wy);
          useSelectionStore.getState().setSelectionRect({
            startX: x1,
            startY: y1,
            endX: x2,
            endY: y2,
          });

          const bboxes = getBboxes();
          const selection = useSelectionStore.getState();
          const target = [...selection.selectedIndices][0];
          const swept = new Set(selection.equationTermIndices);
          for (let i = 0; i < bboxes.length; i++) {
            if (i === target) continue;
            const b = bboxes[i]!;
            if (
              b.x < x2 &&
              b.x + b.w > x1 &&
              b.y < y2 &&
              b.y + b.h > y1
            ) {
              swept.add(i);
            }
          }
          selection.setEquationTerms(swept);
          useCanvasStore.getState().markDirty("bbox");
          useCanvasStore.getState().markDirty("interaction");
          break;
        }
      }
    },
    [canvasRef, getTransform, getBboxes, updateBbox],
  );

  const handleMouseUp = useCallback(
    (e: MouseEvent) => {
      const ctx = dragRef.current;
      const canvas = canvasRef.current;
      if (!canvas) return;

      if (ctx.mode === "draw" && ctx.dragging) {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const transform = getTransform();
        const [wx, wy] = transform.toWorld(sx, sy);
        const x = Math.min(ctx.startWorldX, wx);
        const y = Math.min(ctx.startWorldY, wy);
        const w = Math.abs(wx - ctx.startWorldX);
        const h = Math.abs(wy - ctx.startWorldY);
        if (w >= MIN_BBOX_SIZE && h >= MIN_BBOX_SIZE) {
          addNewBbox({ x, y, w, h });
        }
      }

      // Clear selection rect.
      useSelectionStore.getState().setSelectionRect(null);
      useCanvasStore.getState().markDirty("interaction");

      dragRef.current = {
        mode: "none",
        startScreenX: 0,
        startScreenY: 0,
        startWorldX: 0,
        startWorldY: 0,
        dragging: false,
      };
    },
    [canvasRef, getTransform, addNewBbox],
  );

  // Attach listeners.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    canvas.addEventListener("mousedown", handleMouseDown);
    window.addEventListener("mousemove", handleMouseMove);
    window.addEventListener("mouseup", handleMouseUp);

    // Prevent context menu on canvas.
    const preventMenu = (e: MouseEvent) => e.preventDefault();
    canvas.addEventListener("contextmenu", preventMenu);

    return () => {
      canvas.removeEventListener("mousedown", handleMouseDown);
      window.removeEventListener("mousemove", handleMouseMove);
      window.removeEventListener("mouseup", handleMouseUp);
      canvas.removeEventListener("contextmenu", preventMenu);
    };
  }, [canvasRef, handleMouseDown, handleMouseMove, handleMouseUp]);
}

/** Compute a new bbox after resizing from a handle drag. */
function computeResize(
  orig: BBox,
  handle: HandlePosition,
  wx: number,
  wy: number,
  _startWx: number,
  _startWy: number,
): BBox {
  let { x, y, w, h } = orig;
  const right = x + w;
  const bottom = y + h;

  switch (handle) {
    case "nw":
      x = Math.min(wx, right - MIN_BBOX_SIZE);
      y = Math.min(wy, bottom - MIN_BBOX_SIZE);
      w = right - x;
      h = bottom - y;
      break;
    case "n":
      y = Math.min(wy, bottom - MIN_BBOX_SIZE);
      h = bottom - y;
      break;
    case "ne":
      y = Math.min(wy, bottom - MIN_BBOX_SIZE);
      w = Math.max(MIN_BBOX_SIZE, wx - x);
      h = bottom - y;
      break;
    case "w":
      x = Math.min(wx, right - MIN_BBOX_SIZE);
      w = right - x;
      break;
    case "e":
      w = Math.max(MIN_BBOX_SIZE, wx - x);
      break;
    case "sw":
      x = Math.min(wx, right - MIN_BBOX_SIZE);
      w = right - x;
      h = Math.max(MIN_BBOX_SIZE, wy - y);
      break;
    case "s":
      h = Math.max(MIN_BBOX_SIZE, wy - y);
      break;
    case "se":
      w = Math.max(MIN_BBOX_SIZE, wx - x);
      h = Math.max(MIN_BBOX_SIZE, wy - y);
      break;
  }

  return { x: Math.max(0, x), y: Math.max(0, y), w, h };
}
