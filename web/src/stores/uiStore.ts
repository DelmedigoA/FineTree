/** UI state — sidebar collapse, panel visibility, save status, etc. */

import { create } from "zustand";

export type SaveStatus = "idle" | "saving" | "saved" | "error";

export interface UIStoreState {
  sidebarCollapsed: boolean;
  saveStatus: SaveStatus;
  savedAt: number | null; // unix ms of last successful save
}

export interface UIStoreActions {
  toggleSidebar(): void;
  setSidebarCollapsed(collapsed: boolean): void;
  setSaveStatus(status: SaveStatus): void;
}

export const useUIStore = create<UIStoreState & UIStoreActions>((set) => ({
  sidebarCollapsed: false,
  saveStatus: "idle",
  savedAt: null,

  toggleSidebar() {
    set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed }));
  },

  setSidebarCollapsed(collapsed) {
    set({ sidebarCollapsed: collapsed });
  },

  setSaveStatus(status) {
    set({ saveStatus: status, savedAt: status === "saved" ? Date.now() : undefined });
  },
}));
