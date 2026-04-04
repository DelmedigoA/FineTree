/** Cache Rough.js bbox renderings to OffscreenCanvas for performance. */

import type { RoughCanvas } from "roughjs/bin/canvas";
import rough from "roughjs";
import type { BBox } from "../types/schema";
import type { BBoxStyle } from "../types/canvas";

interface CacheEntry {
  bbox: BBox;
  style: BBoxStyle;
  zoom: number;
  canvas: OffscreenCanvas;
}

const PADDING = 4;

export class RoughBboxCache {
  private cache = new Map<string, CacheEntry>();

  /** Get or create a cached rendering. Returns the OffscreenCanvas and draw offset. */
  get(
    key: string,
    bbox: BBox,
    style: BBoxStyle,
    zoom: number,
  ): { canvas: OffscreenCanvas; offsetX: number; offsetY: number } {
    const existing = this.cache.get(key);
    if (
      existing &&
      existing.bbox.x === bbox.x &&
      existing.bbox.y === bbox.y &&
      existing.bbox.w === bbox.w &&
      existing.bbox.h === bbox.h &&
      existing.style.stroke === style.stroke &&
      existing.style.strokeWidth === style.strokeWidth &&
      existing.style.roughness === style.roughness &&
      (existing.style.bowing ?? 0) === (style.bowing ?? 0) &&
      existing.zoom === zoom &&
      areDashesEqual(existing.style.strokeLineDash, style.strokeLineDash)
    ) {
      return {
        canvas: existing.canvas,
        offsetX: -PADDING,
        offsetY: -PADDING,
      };
    }

    // Render to new OffscreenCanvas.
    const w = Math.ceil(bbox.w * zoom + PADDING * 2);
    const h = Math.ceil(bbox.h * zoom + PADDING * 2);
    const osc = new OffscreenCanvas(Math.max(w, 1), Math.max(h, 1));
    const ctx = osc.getContext("2d");
    if (ctx) {
      const rc = rough.canvas(osc as unknown as HTMLCanvasElement) as RoughCanvas;
      rc.rectangle(PADDING, PADDING, bbox.w * zoom, bbox.h * zoom, {
        stroke: style.stroke,
        strokeWidth: style.strokeWidth,
        roughness: style.roughness,
        bowing: style.bowing ?? 0,
        strokeLineDash: style.strokeLineDash,
        fill: "transparent",
        fillStyle: "solid",
        seed: hashKey(key),
      });
    }

    const entry: CacheEntry = { bbox: { ...bbox }, style: { ...style }, zoom, canvas: osc };
    this.cache.set(key, entry);

    return { canvas: osc, offsetX: -PADDING, offsetY: -PADDING };
  }

  /** Invalidate a specific entry. */
  invalidate(key: string): void {
    this.cache.delete(key);
  }

  /** Clear all cached entries. */
  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

function areDashesEqual(
  a: number[] | undefined,
  b: number[] | undefined,
): boolean {
  if (!a && !b) return true;
  if (!a || !b) return false;
  if (a.length !== b.length) return false;
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) return false;
  }
  return true;
}

/** Simple string hash to get a stable Rough.js seed per bbox key. */
function hashKey(key: string): number {
  let h = 0;
  for (let i = 0; i < key.length; i++) {
    h = ((h << 5) - h + key.charCodeAt(i)) | 0;
  }
  return Math.abs(h);
}
