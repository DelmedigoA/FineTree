/** AI inference state store. */

import { create } from "zustand";

export type AIProvider = "gemini" | "qwen";
export type AIInitialTab = "extract" | "fill" | "fix" | "detect" | "align" | "batch" | null;

export interface AIStoreState {
  activeProvider: AIProvider;
  isStreaming: boolean;
  streamedText: string;
  dialogOpen: boolean;
  lastError: string | null;
  initialTab: AIInitialTab;
}

export interface AIStoreActions {
  setProvider(provider: AIProvider): void;
  setStreaming(streaming: boolean): void;
  setStreamedText(text: string): void;
  appendStreamedText(chunk: string): void;
  openDialog(tab?: AIInitialTab): void;
  closeDialog(): void;
  setError(error: string | null): void;
  reset(): void;
}

export const useAIStore = create<AIStoreState & AIStoreActions>((set) => ({
  activeProvider: "gemini",
  isStreaming: false,
  streamedText: "",
  dialogOpen: false,
  lastError: null,
  initialTab: null,

  setProvider(provider) {
    set({ activeProvider: provider });
  },

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
