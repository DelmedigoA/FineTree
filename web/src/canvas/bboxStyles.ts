/** Bbox rendering style constants for Rough.js. */

import type { BBoxStyle } from "../types/canvas";

/** Default unselected bbox style. */
export const BBOX_DEFAULT: BBoxStyle = {
  stroke: "rgba(75, 158, 255, 0.2)",
  strokeWidth: 1.5,
  roughness: 0.7,
  cornerRadius: 3,
};

/** Selected bbox style (dashed). */
export const BBOX_SELECTED: BBoxStyle = {
  stroke: "rgba(20, 184, 166, 0.6)",
  strokeWidth: 2,
  roughness: 0.55,
  strokeLineDash: [7, 4],
  cornerRadius: 3,
};

/** Equation OK / valid. */
export const BBOX_EQUATION_OK: BBoxStyle = {
  stroke: "rgba(0, 229, 0, 0.35)",
  strokeWidth: 1.5,
  roughness: 0.6,
  cornerRadius: 3,
};

/** Equation bad / invalid. */
export const BBOX_EQUATION_BAD: BBoxStyle = {
  stroke: "rgba(248, 113, 113, 0.35)",
  strokeWidth: 1.5,
  roughness: 0.6,
  cornerRadius: 3,
};

/** Equation term (being selected as part of an equation). */
export const BBOX_EQUATION_TERM: BBoxStyle = {
  stroke: "rgba(250, 204, 21, 0.55)",
  strokeWidth: 2,
  roughness: 0.55,
  strokeLineDash: [6, 3],
  cornerRadius: 3,
};

/** Hovered bbox style. */
export const BBOX_HOVERED: BBoxStyle = {
  stroke: "rgba(20, 184, 166, 0.6)",
  strokeWidth: 2,
  roughness: 0.55,
  cornerRadius: 3,
};
