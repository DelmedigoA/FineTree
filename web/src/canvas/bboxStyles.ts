/** Bbox rendering style constants for Rough.js. */

import type { BBoxStyle } from "../types/canvas";

/** Default unselected bbox style. */
export const BBOX_DEFAULT: BBoxStyle = {
  stroke: "#4b9eff",
  strokeWidth: 1.5,
  roughness: 0.4,
};

/** Selected bbox style (dashed). */
export const BBOX_SELECTED: BBoxStyle = {
  stroke: "#14b8a6",
  strokeWidth: 2,
  roughness: 0.3,
  strokeLineDash: [7, 4],
};

/** Equation OK / valid. */
export const BBOX_EQUATION_OK: BBoxStyle = {
  stroke: "#34d399",
  strokeWidth: 1.5,
  roughness: 0.3,
};

/** Equation bad / invalid. */
export const BBOX_EQUATION_BAD: BBoxStyle = {
  stroke: "#f87171",
  strokeWidth: 1.5,
  roughness: 0.3,
};

/** Equation term (being selected as part of an equation). */
export const BBOX_EQUATION_TERM: BBoxStyle = {
  stroke: "#facc15",
  strokeWidth: 2,
  roughness: 0.3,
  strokeLineDash: [6, 3],
};

/** Hovered bbox style. */
export const BBOX_HOVERED: BBoxStyle = {
  stroke: "#14b8a6",
  strokeWidth: 2,
  roughness: 0.3,
};
