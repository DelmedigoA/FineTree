/** Canvas rendering and interaction types. */

import type { BBox } from "./schema";

export interface WorldTransform {
  scale: number;
  offsetX: number;
  offsetY: number;
  toScreen(wx: number, wy: number): [sx: number, sy: number];
  toWorld(sx: number, sy: number): [wx: number, wy: number];
}

export type InteractionMode =
  | "select"
  | "draw"
  | "pan"
  | "equation";

export type HandlePosition =
  | "nw" | "n" | "ne"
  | "w"        | "e"
  | "sw" | "s" | "se";

export interface HitTestResult {
  type: "none" | "body" | "handle";
  factIndex: number;
  handle?: HandlePosition;
}

export interface SelectionRect {
  startX: number;
  startY: number;
  endX: number;
  endY: number;
}

export interface DragState {
  active: boolean;
  startX: number;
  startY: number;
  currentX: number;
  currentY: number;
}

export interface ViewportSize {
  width: number;
  height: number;
}

export interface ImageSize {
  width: number;
  height: number;
}

/** Bbox style for Rough.js rendering. */
export interface BBoxStyle {
  stroke: string;
  strokeWidth: number;
  roughness: number;
  bowing?: number;
  strokeLineDash?: number[];
  fillStyle?: string;
}

/** Cached rough rendering of a single bbox. */
export interface RoughBBoxEntry {
  bbox: BBox;
  style: BBoxStyle;
  canvas: OffscreenCanvas | null;
  dirty: boolean;
}
