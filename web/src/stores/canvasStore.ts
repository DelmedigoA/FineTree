import { create } from "zustand";
import type { InteractionMode, ViewportSize, ImageSize } from "../types/canvas";

export interface CanvasStoreState {
  zoom: number;
  panX: number;
  panY: number;
  viewportSize: ViewportSize;
  imageSize: ImageSize;
  interactionMode: InteractionMode;
  lensEnabled: boolean;
  /** Dirty flags per canvas layer for rAF rendering. */
  dirtyImage: boolean;
  dirtyBbox: boolean;
  dirtyInteraction: boolean;
}

export interface CanvasStoreActions {
  setZoom(zoom: number): void;
  setPan(x: number, y: number): void;
  setViewportSize(size: ViewportSize): void;
  setImageSize(size: ImageSize): void;
  setInteractionMode(mode: InteractionMode): void;
  toggleLens(): void;
  markDirty(layer: "image" | "bbox" | "interaction" | "all"): void;
  clearDirty(layer: "image" | "bbox" | "interaction" | "all"): void;
  fitToView(): void;
  zoomBy(delta: number): void;
}

const MIN_ZOOM = 0.1;
const MAX_ZOOM = 10;

function clampZoom(z: number): number {
  return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, z));
}

export const useCanvasStore = create<CanvasStoreState & CanvasStoreActions>(
  (set, get) => ({
    zoom: 1,
    panX: 0,
    panY: 0,
    viewportSize: { width: 0, height: 0 },
    imageSize: { width: 0, height: 0 },
    interactionMode: "select",
    lensEnabled: false,
    dirtyImage: true,
    dirtyBbox: true,
    dirtyInteraction: true,

    setZoom(zoom) {
      set({ zoom: clampZoom(zoom), dirtyImage: true, dirtyBbox: true, dirtyInteraction: true });
    },

    setPan(x, y) {
      set({ panX: x, panY: y, dirtyImage: true, dirtyBbox: true, dirtyInteraction: true });
    },

    setViewportSize(size) {
      set({ viewportSize: size, dirtyImage: true, dirtyBbox: true, dirtyInteraction: true });
    },

    setImageSize(size) {
      set({ imageSize: size });
    },

    setInteractionMode(mode) {
      set({ interactionMode: mode });
    },

    toggleLens() {
      set((s) => ({ lensEnabled: !s.lensEnabled }));
    },

    markDirty(layer) {
      if (layer === "all") {
        set({ dirtyImage: true, dirtyBbox: true, dirtyInteraction: true });
      } else if (layer === "image") {
        set({ dirtyImage: true });
      } else if (layer === "bbox") {
        set({ dirtyBbox: true });
      } else {
        set({ dirtyInteraction: true });
      }
    },

    clearDirty(layer) {
      if (layer === "all") {
        set({ dirtyImage: false, dirtyBbox: false, dirtyInteraction: false });
      } else if (layer === "image") {
        set({ dirtyImage: false });
      } else if (layer === "bbox") {
        set({ dirtyBbox: false });
      } else {
        set({ dirtyInteraction: false });
      }
    },

    fitToView() {
      const { viewportSize, imageSize } = get();
      if (imageSize.width === 0 || imageSize.height === 0) return;
      const padding = 40;
      const scaleX = (viewportSize.width - padding * 2) / imageSize.width;
      const scaleY = (viewportSize.height - padding * 2) / imageSize.height;
      const zoom = clampZoom(Math.min(scaleX, scaleY));
      const panX =
        (viewportSize.width - imageSize.width * zoom) / 2;
      const panY =
        (viewportSize.height - imageSize.height * zoom) / 2;
      set({
        zoom,
        panX,
        panY,
        dirtyImage: true,
        dirtyBbox: true,
        dirtyInteraction: true,
      });
    },

    zoomBy(delta) {
      const { zoom, viewportSize, panX, panY } = get();
      const factor = delta > 0 ? 1.1 : 1 / 1.1;
      const newZoom = clampZoom(zoom * factor);
      const ratio = newZoom / zoom;
      const cx = viewportSize.width / 2;
      const cy = viewportSize.height / 2;
      set({
        zoom: newZoom,
        panX: cx - ratio * (cx - panX),
        panY: cy - ratio * (cy - panY),
        dirtyImage: true,
        dirtyBbox: true,
        dirtyInteraction: true,
      });
    },
  }),
);
