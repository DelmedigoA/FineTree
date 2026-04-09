/** AI inference state store. */

import { create } from "zustand";

export type AIInitialTab =
  | "extract"
  | "fill"
  | "fix"
  | "detect"
  | "batch"
  | "infer"
  | null;

export interface AIStoreState {
  isStreaming: boolean;
  streamedText: string;
  dialogOpen: boolean;
  lastError: string | null;
  initialTab: AIInitialTab;
}

export interface AIStoreActions {
  setStreaming(streaming: boolean): void;
  setStreamedText(text: string): void;
  appendStreamedText(chunk: string): void;
  openDialog(tab?: AIInitialTab): void;
  closeDialog(): void;
  setError(error: string | null): void;
  reset(): void;
}

export const useAIStore = create<AIStoreState & AIStoreActions>((set) => ({
  isStreaming: false,
  streamedText: "",
  dialogOpen: false,
  lastError: null,
  initialTab: null,

  setStreaming(streaming) {
    set({ isStreaming: streaming });
  },

  setStreamedText(text) {
    set({ streamedText: text });
  },

  appendStreamedText(chunk) {
    set((s) => ({ streamedText: s.streamedText + chunk }));
  },

  openDialog(tab = null) {
    set({ dialogOpen: true, lastError: null, streamedText: "", initialTab: tab });
  },

  closeDialog() {
    set({ dialogOpen: false, initialTab: null });
  },

  setError(error) {
    set({ lastError: error, isStreaming: false });
  },

  reset() {
    set({
      isStreaming: false,
      streamedText: "",
      dialogOpen: false,
      lastError: null,
      initialTab: null,
    });
  },
}));
