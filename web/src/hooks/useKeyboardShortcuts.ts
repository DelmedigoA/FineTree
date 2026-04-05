/** Global keyboard shortcuts for the annotation view. */

import { useEffect } from "react";
import { useDocumentStore } from "../stores/documentStore";
import { useCanvasStore } from "../stores/canvasStore";
import { useSelectionStore } from "../stores/selectionStore";
import { pushUndoSnapshot } from "./useUndoRedo";
import { applyEquation } from "./useEquationWorkflow";
import type { BBox, PageState } from "../types/schema";

const NUDGE_SMALL = 1;
const NUDGE_LARGE = 10;
const PAN_SMALL = 20;
const PAN_LARGE = 100;
const EDGE_SMALL = 1;
const EDGE_LARGE = 10;

export function useKeyboardShortcuts() {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Ignore if typing in an input field.
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;

      const doc = useDocumentStore.getState();
      const canvas = useCanvasStore.getState();
      const selection = useSelectionStore.getState();

      const getCurrentPage = (): PageState | undefined => {
        const name = doc.pageNames[doc.currentPageIndex];
        return name ? doc.pageStates.get(name) : undefined;
      };

      // Alt+Shift: apply pending equation to selected target.
      if (e.altKey && e.shiftKey) {
        const { selectedIndices, equationTermIndices } = useSelectionStore.getState();
        if (selectedIndices.size === 1 && equationTermIndices.size > 0) {
          e.preventDefault();
          pushUndoSnapshot();
          const targetIndex = [...selectedIndices][0]!;
          const termIndices = [...equationTermIndices];
          applyEquation(targetIndex, termIndices, new Map());
          return;
        }
      }

      switch (e.key) {
        // Page navigation.
        case "a":
        case "A":
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            doc.setCurrentPageIndex(doc.currentPageIndex - 1);
          }
          if ((e.ctrlKey || e.metaKey) && e.key === "a") {
            // Ctrl+A = select all.
            e.preventDefault();
            const page = getCurrentPage();
            if (page) selection.selectAll(page.facts.length);
          }
          break;

        case "d":
        case "D":
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            doc.setCurrentPageIndex(doc.currentPageIndex + 1);
          }
          break;

        // Delete.
        case "Delete":
        case "Backspace":
          if (selection.selectedIndices.size > 0) {
            e.preventDefault();
            deleteSelectedFacts();
          }
          break;

        // Zoom.
        case "+":
        case "=":
          e.preventDefault();
          canvas.zoomBy(100);
          break;
        case "-":
          if (!e.ctrlKey && !e.metaKey) {
            e.preventDefault();
            canvas.zoomBy(-100);
          }
          break;

        // Arrow keys: nudge, pan, or resize edge.
        case "ArrowUp":
        case "ArrowDown":
        case "ArrowLeft":
        case "ArrowRight": {
          e.preventDefault();
          const step = e.shiftKey ? NUDGE_LARGE : NUDGE_SMALL;

          if (e.ctrlKey || e.metaKey) {
            // Pan.
            const panStep = e.shiftKey ? PAN_LARGE : PAN_SMALL;
            let dx = 0, dy = 0;
            if (e.key === "ArrowUp") dy = panStep;
            if (e.key === "ArrowDown") dy = -panStep;
            if (e.key === "ArrowLeft") dx = panStep;
            if (e.key === "ArrowRight") dx = -panStep;
            canvas.setPan(canvas.panX + dx, canvas.panY + dy);
          } else if (e.altKey) {
            // Resize edge.
            const edgeStep = e.shiftKey ? EDGE_LARGE : EDGE_SMALL;
            nudgeSelectedEdge(e.key, edgeStep);
          } else {
            // Nudge.
            let dx = 0, dy = 0;
            if (e.key === "ArrowUp") dy = -step;
            if (e.key === "ArrowDown") dy = step;
            if (e.key === "ArrowLeft") dx = -step;
            if (e.key === "ArrowRight") dx = step;
            nudgeSelectedBboxes(dx, dy);
          }
          break;
        }

        case "Alt":
          // equationModeActive drives canvas highlight; actual interaction uses e.altKey.
          if (selection.selectedIndices.size === 1) {
            selection.setEquationModeActive(true);
          }
          break;
      }
    };

    const keyup = (e: KeyboardEvent) => {
      if (e.key === "Alt") {
        useSelectionStore.getState().setEquationModeActive(false);
      }
    };

    window.addEventListener("keydown", handler);
    window.addEventListener("keyup", keyup);
    return () => {
      window.removeEventListener("keydown", handler);
      window.removeEventListener("keyup", keyup);
    };
  }, []);
}

function deleteSelectedFacts() {
  pushUndoSnapshot();
  const { pageStates, pageNames, currentPageIndex, updatePageState } =
    useDocumentStore.getState();
  const { selectedIndices, clearSelection } = useSelectionStore.getState();
  const pageName = pageNames[currentPageIndex];
  if (!pageName) return;
  const page = pageStates.get(pageName);
  if (!page) return;

  const newFacts = page.facts.filter((_, i) => !selectedIndices.has(i));
  updatePageState(pageName, { ...page, facts: newFacts });
  clearSelection();
  useCanvasStore.getState().markDirty("bbox");
}

function nudgeSelectedBboxes(dx: number, dy: number) {
  pushUndoSnapshot();
  const { pageStates, pageNames, currentPageIndex, updatePageState } =
    useDocumentStore.getState();
  const { selectedIndices } = useSelectionStore.getState();
  const pageName = pageNames[currentPageIndex];
  if (!pageName || selectedIndices.size === 0) return;
  const page = pageStates.get(pageName);
  if (!page) return;

  const newFacts = [...page.facts];
  for (const idx of selectedIndices) {
    const f = newFacts[idx];
    if (!f) continue;
    newFacts[idx] = {
      ...f,
      bbox: {
        x: Math.max(0, f.bbox.x + dx),
        y: Math.max(0, f.bbox.y + dy),
        w: f.bbox.w,
        h: f.bbox.h,
      },
    };
  }
  updatePageState(pageName, { ...page, facts: newFacts });
  useCanvasStore.getState().markDirty("bbox");
}

function nudgeSelectedEdge(
  key: string,
  step: number,
) {
  pushUndoSnapshot();
  const { pageStates, pageNames, currentPageIndex, updatePageState } =
    useDocumentStore.getState();
  const { selectedIndices } = useSelectionStore.getState();
  const pageName = pageNames[currentPageIndex];
  if (!pageName || selectedIndices.size === 0) return;
  const page = pageStates.get(pageName);
  if (!page) return;

  const newFacts = [...page.facts];
  for (const idx of selectedIndices) {
    const f = newFacts[idx];
    if (!f) continue;
    const b: BBox = { ...f.bbox };
    switch (key) {
      case "ArrowUp":
        b.h = Math.max(4, b.h - step);
        break;
      case "ArrowDown":
        b.h = b.h + step;
        break;
      case "ArrowLeft":
        b.w = Math.max(4, b.w - step);
        break;
      case "ArrowRight":
        b.w = b.w + step;
        break;
    }
    newFacts[idx] = { ...f, bbox: b };
  }
  updatePageState(pageName, { ...page, facts: newFacts });
  useCanvasStore.getState().markDirty("bbox");
}
