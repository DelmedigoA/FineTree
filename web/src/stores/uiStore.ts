/** UI state — sidebar collapse, panel visibility, save status, etc. */

import { create } from "zustand";

export type SaveStatus = "idle" | "saving" | "saved" | "error";
export type AIStatus = "idle" | "running" | "success" | "error";

export interface UIStoreState {
  sidebarCollapsed: boolean;
  saveStatus: SaveStatus;
  savedAt: number | null; // unix ms of last successful save
  aiStatus: AIStatus;
  aiMessage: string | null;
  settingsOpen: boolean;
}

export interface UIStoreActions {
  toggleSidebar(): void;
  setSidebarCollapsed(collapsed: boolean): void;
  setSaveStatus(status: SaveStatus): void;
  setAIStatus(status: AIStatus, message?: string | null): void;
  clearAIStatus(): void;
  toggleSettings(): void;
}

export const useUIStore = create<UIStoreState & UIStoreActions>((set) => ({
  sidebarCollapsed: false,
  saveStatus: "idle",
  savedAt: null,
  aiStatus: "idle",
  aiMessage: null,
  settingsOpen: false,

  toggleSidebar() {
    set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed }));
  },

  setSidebarCollapsed(collapsed) {
    set({ sidebarCollapsed: collapsed });
  },

  setSaveStatus(status) {
    set({ saveStatus: status, savedAt: status === "saved" ? Date.now() : undefined });
  },

  setAIStatus(status, message = null) {
    set({ aiStatus: status, aiMessage: message });
  },

  clearAIStatus() {
    set({ aiStatus: "idle", aiMessage: null });
  },

  toggleSettings() {
    set((s) => ({ settingsOpen: !s.settingsOpen }));
  },
}));
