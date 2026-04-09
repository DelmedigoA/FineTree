/** Bbox rendering style constants for Rough.js. */

import type { BBoxStyle } from "../types/canvas";
import { useSettingsStore } from "../stores/settingsStore";

/** Convert a hex color + opacity to an rgba() string. */
function toRgba(hex: string, opacity: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${opacity})`;
}

/** Returns bbox styles derived from user settings. Called per render pass. */
export function getBboxStyles(): {
  default: BBoxStyle;
  selected: BBoxStyle;
  hovered: BBoxStyle;
  equationOk: BBoxStyle;
  equationBad: BBoxStyle;
  equationTerm: BBoxStyle;
} {
  const c = useSettingsStore.getState().bboxColors;
  return {
    default: {
      stroke: toRgba(c.default.color, c.default.opacity),
      strokeWidth: 1.5,
      roughness: 0.7,
      cornerRadius: 3,
    },
    selected: {
      stroke: toRgba(c.selected.color, c.selected.opacity),
      strokeWidth: 2,
      roughness: 0.55,
      strokeLineDash: [7, 4],
      cornerRadius: 3,
    },
    hovered: {
      stroke: toRgba(c.hovered.color, c.hovered.opacity),
      strokeWidth: 2,
      roughness: 0.55,
      cornerRadius: 3,
    },
    equationOk: {
      stroke: toRgba(c.equationOk.color, c.equationOk.opacity),
      strokeWidth: 1.5,
      roughness: 0.6,
      cornerRadius: 3,
    },
    equationBad: {
      stroke: toRgba(c.equationBad.color, c.equationBad.opacity),
      strokeWidth: 1.5,
      roughness: 0.6,
      cornerRadius: 3,
    },
    equationTerm: {
      stroke: toRgba(c.equationTerm.color, c.equationTerm.opacity),
      strokeWidth: 2,
      roughness: 0.55,
      strokeLineDash: [6, 3],
      cornerRadius: 3,
    },
  };
}
