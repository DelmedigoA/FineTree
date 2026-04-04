/** Hit-testing for bboxes on the canvas. */

import type { BBox } from "../types/schema";
import type { HandlePosition, HitTestResult } from "../types/canvas";
import type { WorldTransformImpl } from "./WorldTransform";

/** Hit margin in screen pixels for resize handles. */
const HANDLE_MARGIN = 8;

/** Test which part of which bbox (if any) is under screen coords (sx, sy). */
export function hitTest(
  sx: number,
  sy: number,
  bboxes: BBox[],
  transform: WorldTransformImpl,
): HitTestResult {
  // Iterate in reverse so topmost (last-drawn) bbox wins.
  for (let i = bboxes.length - 1; i >= 0; i--) {
    const bbox = bboxes[i]!;
    const [left, top] = transform.toScreen(bbox.x, bbox.y);
    const [right, bottom] = transform.toScreen(
      bbox.x + bbox.w,
      bbox.y + bbox.h,
    );

    // Check handle zones first (8px margin around edges/corners).
    const handle = testHandles(sx, sy, left, top, right, bottom);
    if (handle) {
      return { type: "handle", factIndex: i, handle };
    }

    // Check body.
    if (sx >= left && sx <= right && sy >= top && sy <= bottom) {
      return { type: "body", factIndex: i };
    }
  }

  return { type: "none", factIndex: -1 };
}

function testHandles(
  sx: number,
  sy: number,
  left: number,
  top: number,
  right: number,
  bottom: number,
): HandlePosition | undefined {
  const m = HANDLE_MARGIN;

  const nearLeft = Math.abs(sx - left) <= m;
  const nearRight = Math.abs(sx - right) <= m;
  const nearTop = Math.abs(sy - top) <= m;
  const nearBottom = Math.abs(sy - bottom) <= m;

  const withinX = sx >= left - m && sx <= right + m;
  const withinY = sy >= top - m && sy <= bottom + m;

  if (!withinX || !withinY) return undefined;

  // Corners
  if (nearLeft && nearTop) return "nw";
  if (nearRight && nearTop) return "ne";
  if (nearLeft && nearBottom) return "sw";
  if (nearRight && nearBottom) return "se";

  // Edges
  if (nearTop && sx > left + m && sx < right - m) return "n";
  if (nearBottom && sx > left + m && sx < right - m) return "s";
  if (nearLeft && sy > top + m && sy < bottom - m) return "w";
  if (nearRight && sy > top + m && sy < bottom - m) return "e";

  return undefined;
}

/** Get the cursor CSS class for a handle position. */
export function cursorForHandle(handle: HandlePosition): string {
  const map: Record<HandlePosition, string> = {
    nw: "nwse-resize",
    n: "ns-resize",
    ne: "nesw-resize",
    w: "ew-resize",
    e: "ew-resize",
    sw: "nesw-resize",
    s: "ns-resize",
    se: "nwse-resize",
  };
  return map[handle];
}
