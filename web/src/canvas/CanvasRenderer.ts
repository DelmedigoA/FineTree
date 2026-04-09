/** requestAnimationFrame-based canvas rendering loop with dirty flags. */

import { useCanvasStore } from "../stores/canvasStore";
import { useDocumentStore } from "../stores/documentStore";
import { useSelectionStore } from "../stores/selectionStore";
import { createTransform } from "./WorldTransform";
import { RoughBboxCache } from "./RoughBboxCache";
import { getBboxStyles } from "./bboxStyles";
import { equationMatchState, evaluateEquationString } from "../hooks/useEquationWorkflow";
import type { BBoxStyle } from "../types/canvas";
import type { BBox, BoxRecord } from "../types/schema";

const roughCache = new RoughBboxCache();

export interface CanvasRefs {
  imageCanvas: HTMLCanvasElement | null;
  bboxCanvas: HTMLCanvasElement | null;
  interactCanvas: HTMLCanvasElement | null;
  pageImage: HTMLImageElement | null;
}

let rafId = 0;

export function startRenderLoop(refs: CanvasRefs): () => void {
  let running = true;

  const tick = () => {
    if (!running) return;
    const canvasState = useCanvasStore.getState();

    if (canvasState.dirtyImage) {
      renderImageLayer(refs);
      canvasState.clearDirty("image");
    }

    if (canvasState.dirtyBbox) {
      renderBboxLayer(refs);
      canvasState.clearDirty("bbox");
    }

    if (canvasState.dirtyInteraction) {
      renderInteractionLayer(refs);
      canvasState.clearDirty("interaction");
    }

    rafId = requestAnimationFrame(tick);
  };

  rafId = requestAnimationFrame(tick);
  return () => {
    running = false;
    cancelAnimationFrame(rafId);
  };
}

function renderImageLayer(refs: CanvasRefs) {
  const canvas = refs.imageCanvas;
  const img = refs.pageImage;
  if (!canvas || !img || !img.complete || img.naturalWidth === 0) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const { zoom, panX, panY } = useCanvasStore.getState();
  ctx.drawImage(
    img,
    panX,
    panY,
    img.naturalWidth * zoom,
    img.naturalHeight * zoom,
  );
  ctx.restore();
}

/** Check if a fact's equations match its value. Returns "ok", "bad", or "none". */
function getFactEquationState(fact: BoxRecord): "ok" | "bad" | "none" {
  const equations = fact.fact.equations as Array<{ equation: string }> | null;
  if (!equations || equations.length === 0) return "none";
  const targetValue = String(fact.fact.value ?? "");
  const targetNaturalSign = String(fact.fact.natural_sign ?? "") || null;
  for (const eq of equations) {
    const result = evaluateEquationString(eq.equation);
    if (equationMatchState(result, targetValue, targetNaturalSign) === "ok") return "ok";
  }
  return "bad";
}

function renderBboxLayer(refs: CanvasRefs) {
  const canvas = refs.bboxCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const { zoom, panX, panY } = useCanvasStore.getState();
  const transform = createTransform(zoom, panX, panY);
  const { pageStates, pageNames, currentPageIndex } =
    useDocumentStore.getState();
  const pageName = pageNames[currentPageIndex];
  if (!pageName) {
    ctx.restore();
    return;
  }
  const page = pageStates.get(pageName);
  if (!page) {
    ctx.restore();
    return;
  }

  const selection = useSelectionStore.getState();
  const styles = getBboxStyles();

  for (let i = 0; i < page.facts.length; i++) {
    const fact = page.facts[i]!;
    const bbox = fact.bbox;

    // Determine style.
    let style: BBoxStyle;
    if (selection.equationTermIndices.has(i)) {
      style = styles.equationTerm;
    } else if (selection.selectedIndices.has(i)) {
      style = styles.selected;
    } else if (selection.hoveredIndex === i) {
      style = styles.hovered;
    } else {
      const eqState = getFactEquationState(fact);
      if (eqState === "ok") {
        style = styles.equationOk;
      } else if (eqState === "bad") {
        style = styles.equationBad;
      } else {
        style = styles.default;
      }
    }

    // Use rough cache.
    const key = `${pageName}:${i}`;
    const cached = roughCache.get(key, bbox, style, zoom);

    const [screenX, screenY] = transform.toScreen(bbox.x, bbox.y);
    ctx.drawImage(
      cached.canvas,
      screenX + cached.offsetX,
      screenY + cached.offsetY,
    );
  }

  ctx.restore();
}

function renderInteractionLayer(refs: CanvasRefs) {
  const canvas = refs.interactCanvas;
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  if (!ctx) return;

  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;

  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);

  const { zoom, panX, panY } = useCanvasStore.getState();
  const transform = createTransform(zoom, panX, panY);
  const selection = useSelectionStore.getState();

  // Draw selection/draw rectangle.
  if (selection.selectionRect) {
    const r = selection.selectionRect;
    const [x1, y1] = transform.toScreen(r.startX, r.startY);
    const [x2, y2] = transform.toScreen(r.endX, r.endY);
    // Read accent from CSS custom property at render time.
    const accent =
      getComputedStyle(document.documentElement)
        .getPropertyValue("--accent")
        .trim() || "#14b8a6";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    ctx.fillStyle = "rgba(20, 184, 166, 0.08)";
    ctx.fillRect(x1, y1, x2 - x1, y2 - y1);
    ctx.setLineDash([]);
  }

  // Draw resize handles for selected bboxes.
  const { pageStates, pageNames, currentPageIndex } =
    useDocumentStore.getState();
  const pageName = pageNames[currentPageIndex];
  if (pageName) {
    const page = pageStates.get(pageName);
    if (page) {
      for (const idx of selection.selectedIndices) {
        const fact = page.facts[idx];
        if (!fact) continue;
        drawResizeHandles(ctx, fact.bbox, transform);
      }
    }
  }

  ctx.restore();
}

function drawResizeHandles(
  ctx: CanvasRenderingContext2D,
  bbox: BBox,
  transform: { toScreen(x: number, y: number): [number, number] },
) {
  const [left, top] = transform.toScreen(bbox.x, bbox.y);
  const [right, bottom] = transform.toScreen(bbox.x + bbox.w, bbox.y + bbox.h);
  const cx = (left + right) / 2;
  const cy = (top + bottom) / 2;
  const size = 6;

  const handles = [
    [left, top],
    [cx, top],
    [right, top],
    [left, cy],
    [right, cy],
    [left, bottom],
    [cx, bottom],
    [right, bottom],
  ];

  const accent =
    getComputedStyle(document.documentElement)
      .getPropertyValue("--accent")
      .trim() || "#14b8a6";
  for (const [hx, hy] of handles) {
    ctx.fillStyle = "#ffffff";
    ctx.strokeStyle = accent;
    ctx.lineWidth = 1.5;
    ctx.fillRect(hx! - size / 2, hy! - size / 2, size, size);
    ctx.strokeRect(hx! - size / 2, hy! - size / 2, size, size);
  }
}

/** Clear the rough cache (call on page change). */
export function clearRoughCache() {
  roughCache.clear();
}
