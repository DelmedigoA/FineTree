/** AI inference state store. */

import { create } from "zustand";

export type AIProvider = "gemini" | "qwen";

export interface AIStoreState {
  activeProvider: AIProvider;
  isStreaming: boolean;
  streamedText: string;
  dialogOpen: boolean;
  lastError: string | null;
}

export interface AIStoreActions {
  setProvider(provider: AIProvider): void;
  setStreaming(streaming: boolean): void;
  setStreamedText(text: string): void;
  appendStreamedText(chunk: string): void;
  openDialog(): void;
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

  openDialog() {
    set({ dialogOpen: true, lastError: null, streamedText: "" });
  },

  closeDialog() {
    set({ dialogOpen: false });
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
    });
  },
}));
