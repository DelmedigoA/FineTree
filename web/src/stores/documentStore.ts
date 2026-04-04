import { create } from "zustand";
import type { PageState } from "../types/schema";

export interface DocumentStoreState {
  /** Current document id. Null when on dashboard. */
  docId: string | null;
  /** Raw payload as received from GET /api/annotations/{doc_id}. */
  rawPayload: Record<string, unknown> | null;
  /** Document-level metadata extracted from payload. */
  documentMeta: Record<string, unknown>;
  /** Page states keyed by page image filename (e.g. "page_001.png"). */
  pageStates: Map<string, PageState>;
  /** Ordered list of page image filenames for this document. */
  pageNames: string[];
  /** Index of the currently-viewed page. */
  currentPageIndex: number;
  /** Whether there are unsaved changes. */
  isDirty: boolean;
  /** Loading state. */
  isLoading: boolean;
}

export interface DocumentStoreActions {
  /** Load a document from the API response. */
  loadDocument(
    docId: string,
    payload: {
      raw_payload: Record<string, unknown>;
      page_states: Record<string, PageState>;
      document_meta: Record<string, unknown>;
      page_names: string[];
    },
  ): void;

  /** Navigate to a page by index. */
  setCurrentPageIndex(index: number): void;

  /** Update a single page state. Marks dirty. */
  updatePageState(pageName: string, state: PageState): void;

  /** Update document metadata. Marks dirty. */
  updateDocumentMeta(meta: Record<string, unknown>): void;

  /** Mark the document as saved (clears dirty flag). */
  markSaved(): void;

  /** Reset to empty state. */
  reset(): void;
}

export const useDocumentStore = create<
  DocumentStoreState & DocumentStoreActions
>((set) => ({
  docId: null,
  rawPayload: null,
  documentMeta: {},
  pageStates: new Map(),
  pageNames: [],
  currentPageIndex: 0,
  isDirty: false,
  isLoading: false,

  loadDocument(docId, payload) {
    const pageStates = new Map<string, PageState>();
    for (const [key, value] of Object.entries(payload.page_states)) {
      pageStates.set(key, value);
    }
    set({
      docId,
      rawPayload: payload.raw_payload,
      documentMeta: payload.document_meta,
      pageStates,
      pageNames: payload.page_names,
      currentPageIndex: 0,
      isDirty: false,
      isLoading: false,
    });
  },

  setCurrentPageIndex(index) {
    set((state) => {
      const clamped = Math.max(0, Math.min(index, state.pageNames.length - 1));
      return { currentPageIndex: clamped };
    });
  },

  updatePageState(pageName, pageState) {
    set((state) => {
      const next = new Map(state.pageStates);
      next.set(pageName, pageState);
      return { pageStates: next, isDirty: true };
    });
  },

  updateDocumentMeta(meta) {
    set({ documentMeta: meta, isDirty: true });
  },

  markSaved() {
    set({ isDirty: false });
  },

  reset() {
    set({
      docId: null,
      rawPayload: null,
      documentMeta: {},
      pageStates: new Map(),
      pageNames: [],
      currentPageIndex: 0,
      isDirty: false,
      isLoading: false,
    });
  },
}));
