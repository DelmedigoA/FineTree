import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { post } from "../api/client";
import { useCanvasStore } from "../stores/canvasStore";
import { useDocumentStore } from "../stores/documentStore";
import { useSelectionStore } from "../stores/selectionStore";
import { useUIStore } from "../stores/uiStore";
import { pushUndoSnapshot } from "./useUndoRedo";
import { defaultFactData, type BoxRecord } from "../types/schema";
import type {
  AlignBboxesFactRequest,
  AlignBboxesResponse,
  AlignBboxesResponseItem,
} from "../types/api";

const FALLBACK_BBOX = { x: 0, y: 0, w: 50, h: 20 };

function normalizeAlignedFact(item: AlignBboxesResponseItem): BoxRecord {
  return {
    bbox: item.bbox ?? FALLBACK_BBOX,
    fact: {
      ...defaultFactData(),
      ...(item.fact ?? {}),
    },
  };
}

function formatAlignScopeMessage(count: number, selectedOnly: boolean, phase: "running" | "success"): string {
  const verb = phase === "running" ? "Aligning" : "Aligned";
  const suffix = phase === "running" ? "…" : "";
  if (selectedOnly) {
    return `${verb} ${count} selected ${count === 1 ? "bbox" : "bboxes"}${suffix}`;
  }
  return `${verb} all ${count} ${count === 1 ? "bbox" : "bboxes"}${suffix}`;
}

function coercePageIndex(value: unknown): number | null {
  if (typeof value === "number" && Number.isInteger(value)) {
    return value;
  }
  if (typeof value === "string" && value.trim() !== "") {
    const parsed = Number(value);
    if (Number.isInteger(parsed)) {
      return parsed;
    }
  }
  return null;
}

export function useAlignBBoxes() {
  const docId = useDocumentStore((s) => s.docId);
  const currentPageIndex = useDocumentStore((s) => s.currentPageIndex);
  const pageNames = useDocumentStore((s) => s.pageNames);
  const pageStates = useDocumentStore((s) => s.pageStates);
  const updatePageStateForDocument = useDocumentStore((s) => s.updatePageStateForDocument);
  const selectedIndices = useSelectionStore((s) => s.selectedIndices);
  const setAIStatus = useUIStore((s) => s.setAIStatus);

  const [isAligning, setIsAligning] = useState(false);
  const statusTimeoutRef = useRef<number | null>(null);

  const currentPageName = pageNames[currentPageIndex] ?? "";
  const currentPage = currentPageName ? pageStates.get(currentPageName) : undefined;

  useEffect(() => {
    return () => {
      if (statusTimeoutRef.current != null) {
        window.clearTimeout(statusTimeoutRef.current);
      }
    };
  }, []);

  const showAIStatus = useCallback(
    (
      status: "running" | "success" | "error",
      message: string,
      clearAfterMs?: number,
    ) => {
      setAIStatus(status, message);
      if (statusTimeoutRef.current != null) {
        window.clearTimeout(statusTimeoutRef.current);
        statusTimeoutRef.current = null;
      }
      if (clearAfterMs != null) {
        statusTimeoutRef.current = window.setTimeout(() => {
          const state = useUIStore.getState();
          if (state.aiStatus === status && state.aiMessage === message) {
            state.clearAIStatus();
          }
          statusTimeoutRef.current = null;
        }, clearAfterMs);
      }
    },
    [setAIStatus],
  );

  const targetIndices = useMemo(() => {
    if (!currentPage) return [];
    if (selectedIndices.size > 0) {
      return [...selectedIndices]
        .filter((index) => index >= 0 && index < currentPage.facts.length)
        .sort((a, b) => a - b);
    }
    return currentPage.facts.map((_, index) => index);
  }, [currentPage, selectedIndices]);

  const canAlign = Boolean(docId && currentPageName && currentPage && currentPage.facts.length > 0);

  const alignBBoxes = useCallback(async () => {
    if (isAligning || !canAlign || !docId || !currentPageName || !currentPage || targetIndices.length === 0) {
      return;
    }
    const activeDocId = docId;
    const activePageName = currentPageName;

    const selectedOnly = selectedIndices.size > 0;
    const requestFacts: AlignBboxesFactRequest[] = [];
    for (const pageIndex of targetIndices) {
      const record = currentPage.facts[pageIndex];
      if (!record) continue;
      requestFacts.push({
        ...record.fact,
        bbox: record.bbox,
        page_index: pageIndex,
      });
    }

    if (requestFacts.length === 0) return;

    const requestedCount = requestFacts.length;
    setIsAligning(true);
    showAIStatus("running", formatAlignScopeMessage(requestedCount, selectedOnly, "running"));

    try {
      const result = await post<AlignBboxesResponse>(
        "/ai/align-bboxes",
        {
          doc_id: docId,
          page_name: currentPageName,
          facts: requestFacts,
        },
      );

      if (!result.aligned_facts?.length) {
        showAIStatus("error", "Align failed: no aligned facts returned", 5000);
        return;
      }

      const nextFacts = [...currentPage.facts];
      const targetedIndexSet = new Set(targetIndices);
      let appliedCount = 0;

      for (const item of result.aligned_facts) {
        const factPayload = item.fact as Record<string, unknown> | undefined;
        let pageIndex =
          coercePageIndex(item.page_index) ??
          coercePageIndex(factPayload?.page_index);

        if (pageIndex == null) {
          const factNum = factPayload?.fact_num;
          if (typeof factNum === "number" && Number.isInteger(factNum)) {
            const matches = targetIndices.filter(
              (candidateIndex) => currentPage.facts[candidateIndex]?.fact.fact_num === factNum,
            );
            if (matches.length === 1) {
              pageIndex = matches[0] ?? null;
            }
          }
        }

        if (
          pageIndex == null ||
          pageIndex < 0 ||
          pageIndex >= nextFacts.length ||
          !targetedIndexSet.has(pageIndex)
        ) {
          continue;
        }
        nextFacts[pageIndex] = normalizeAlignedFact(item);
        appliedCount++;
      }

      if (appliedCount === 0) {
        showAIStatus("error", "Align failed: no valid page indices returned", 5000);
        return;
      }

      const { docId: currentDocId } = useDocumentStore.getState();
      if (currentDocId !== activeDocId) {
        showAIStatus("error", "Align skipped after document switch", 5000);
        return;
      }

      pushUndoSnapshot();
      const updated = updatePageStateForDocument(activeDocId, activePageName, { ...currentPage, facts: nextFacts });
      if (!updated) {
        showAIStatus("error", "Align skipped after document switch", 5000);
        return;
      }
      useCanvasStore.getState().markDirty("bbox");
      showAIStatus("success", formatAlignScopeMessage(appliedCount, selectedOnly, "success"), 3000);
    } catch (err) {
      showAIStatus("error", `Align failed: ${String(err)}`, 5000);
    } finally {
      setIsAligning(false);
    }
  }, [
    canAlign,
    currentPage,
    currentPageName,
    docId,
    isAligning,
    selectedIndices.size,
    showAIStatus,
    targetIndices,
    updatePageStateForDocument,
  ]);

  return {
    alignBBoxes,
    canAlign,
    isAligning,
    targetCount: targetIndices.length,
    selectedOnly: selectedIndices.size > 0,
  };
}
