/** 3-layer canvas container: image, bbox, interaction. */

import { useRef, useEffect, useCallback } from "react";
import { useCanvasStore } from "../stores/canvasStore";
import { useDocumentStore } from "../stores/documentStore";
import { startRenderLoop, clearRoughCache } from "./CanvasRenderer";
import { useCanvasInteraction } from "../hooks/useCanvasInteraction";
import { useKeyboardShortcuts } from "../hooks/useKeyboardShortcuts";

function normalizeWheelDelta(delta: number, deltaMode: number): number {
  if (deltaMode === WheelEvent.DOM_DELTA_LINE) {
    return delta * 16;
  }
  if (deltaMode === WheelEvent.DOM_DELTA_PAGE) {
    return delta * window.innerHeight;
  }
  return delta;
}

export function CanvasContainer() {
  const containerRef = useRef<HTMLDivElement>(null);
  const imageCanvasRef = useRef<HTMLCanvasElement>(null);
  const bboxCanvasRef = useRef<HTMLCanvasElement>(null);
  const interactCanvasRef = useRef<HTMLCanvasElement>(null);
  const pageImageRef = useRef<HTMLImageElement | null>(null);
  const setViewportSize = useCanvasStore((s) => s.setViewportSize);
  const setImageSize = useCanvasStore((s) => s.setImageSize);
  const markDirty = useCanvasStore((s) => s.markDirty);

  const docId = useDocumentStore((s) => s.docId);
  const pageNames = useDocumentStore((s) => s.pageNames);
  const currentPageIndex = useDocumentStore((s) => s.currentPageIndex);
  const currentPageName = pageNames[currentPageIndex] ?? null;

  // Wire interaction hooks.
  useCanvasInteraction(interactCanvasRef);
  useKeyboardShortcuts();

  const handleResize = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const rect = el.getBoundingClientRect();
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    setViewportSize({ width: w, height: h });

    for (const ref of [imageCanvasRef, bboxCanvasRef, interactCanvasRef]) {
      const c = ref.current;
      if (c) {
        const dpr = window.devicePixelRatio || 1;
        c.width = w * dpr;
        c.height = h * dpr;
        c.style.width = `${w}px`;
        c.style.height = `${h}px`;
      }
    }
    markDirty("all");
  }, [setViewportSize, markDirty]);

  // ResizeObserver.
  useEffect(() => {
    handleResize();
    const observer = new ResizeObserver(handleResize);
    if (containerRef.current) observer.observe(containerRef.current);
    return () => observer.disconnect();
  }, [handleResize]);

  // Load page image when page changes.
  useEffect(() => {
    if (!docId || !currentPageName) {
      pageImageRef.current = null;
      setImageSize({ width: 0, height: 0 });
      markDirty("all");
      return;
    }

    let disposed = false;
    clearRoughCache();
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = `/api/images/${docId}/pages/${currentPageName}`;
    img.onload = () => {
      if (disposed) return;
      pageImageRef.current = img;
      setImageSize({ width: img.naturalWidth, height: img.naturalHeight });
      useCanvasStore.getState().fitToView();
    };
    img.onerror = () => {
      if (disposed) return;
      pageImageRef.current = null;
      setImageSize({ width: 0, height: 0 });
      markDirty("all");
    };

    return () => {
      disposed = true;
    };
  }, [docId, currentPageName, setImageSize, markDirty]);

  // Start render loop.
  useEffect(() => {
    const stop = startRenderLoop({
      imageCanvas: imageCanvasRef.current,
      bboxCanvas: bboxCanvasRef.current,
      interactCanvas: interactCanvasRef.current,
      get pageImage() {
        return pageImageRef.current;
      },
    });
    return stop;
  }, []);

  // Handle wheel zoom and scroll-to-pan.
  useEffect(() => {
    const el = interactCanvasRef.current;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        useCanvasStore.getState().zoomBy(-e.deltaY);
        return;
      }

      e.preventDefault();
      const deltaX = normalizeWheelDelta(e.deltaX, e.deltaMode);
      const deltaY = normalizeWheelDelta(e.deltaY, e.deltaMode);
      const horizontalDelta =
        e.shiftKey && Math.abs(deltaX) < Math.abs(deltaY) ? deltaY : deltaX;
      const verticalDelta =
        e.shiftKey && Math.abs(deltaX) < Math.abs(deltaY) ? 0 : deltaY;
      const { panX, panY, setPan } = useCanvasStore.getState();
      setPan(panX - horizontalDelta, panY - verticalDelta);
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, []);

  const canvasStyle: React.CSSProperties = {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
  };

  return (
    <div
      ref={containerRef}
      style={{
        position: "relative",
        width: "100%",
        height: "100%",
        overflow: "hidden",
        background: "var(--canvas)",
      }}
    >
      <canvas ref={imageCanvasRef} style={{ ...canvasStyle, zIndex: 0 }} />
      <canvas ref={bboxCanvasRef} style={{ ...canvasStyle, zIndex: 1 }} />
      <canvas
        ref={interactCanvasRef}
        style={{ ...canvasStyle, zIndex: 2, cursor: "crosshair" }}
      />
    </div>
  );
}
