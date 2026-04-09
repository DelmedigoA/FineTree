/** User-configurable settings, persisted to localStorage. */

import { create } from "zustand";
import { useCanvasStore } from "./canvasStore";

export interface BBoxColorEntry {
  color: string;   // hex, e.g. "#4b9eff"
  opacity: number; // 0–1
}

export interface BBoxColors {
  default: BBoxColorEntry;
  selected: BBoxColorEntry;
  hovered: BBoxColorEntry;
  equationOk: BBoxColorEntry;
  equationBad: BBoxColorEntry;
  equationTerm: BBoxColorEntry;
}

const DEFAULT_BBOX_COLORS: BBoxColors = {
  default:      { color: "#4b9eff", opacity: 0.20 },
  selected:     { color: "#14b8a6", opacity: 0.60 },
  hovered:      { color: "#14b8a6", opacity: 0.60 },
  equationOk:   { color: "#00e500", opacity: 0.35 },
  equationBad:  { color: "#f87171", opacity: 0.35 },
  equationTerm: { color: "#facc15", opacity: 0.55 },
};

const STORAGE_KEY = "finetree-settings";

function loadFromStorage(): BBoxColors {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULT_BBOX_COLORS;
    const parsed = JSON.parse(raw) as Partial<{ bboxColors: BBoxColors }>;
    if (parsed.bboxColors) {
      // Merge with defaults so new keys are included if schema evolves.
      return { ...DEFAULT_BBOX_COLORS, ...parsed.bboxColors };
    }
  } catch {
    // ignore
  }
  return DEFAULT_BBOX_COLORS;
}

function saveToStorage(bboxColors: BBoxColors) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ bboxColors }));
  } catch {
    // ignore
  }
}

export interface SettingsStoreState {
  bboxColors: BBoxColors;
}

export interface SettingsStoreActions {
  setBboxColor(key: keyof BBoxColors, entry: Partial<BBoxColorEntry>): void;
  resetBboxColors(): void;
}

export const useSettingsStore = create<SettingsStoreState & SettingsStoreActions>((set) => ({
  bboxColors: loadFromStorage(),

  setBboxColor(key, entry) {
    set((s) => {
      const updated: BBoxColors = {
        ...s.bboxColors,
        [key]: { ...s.bboxColors[key], ...entry },
      };
      saveToStorage(updated);
      useCanvasStore.getState().markDirty("bbox");
      return { bboxColors: updated };
    });
  },

  resetBboxColors() {
    saveToStorage(DEFAULT_BBOX_COLORS);
    useCanvasStore.getState().markDirty("bbox");
    set({ bboxColors: { ...DEFAULT_BBOX_COLORS } });
  },
}));

export { DEFAULT_BBOX_COLORS };
