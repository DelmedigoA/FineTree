/** UI state — sidebar collapse, panel visibility, etc. */

import { create } from "zustand";

export interface UIStoreState {
  sidebarCollapsed: boolean;
}

export interface UIStoreActions {
  toggleSidebar(): void;
  setSidebarCollapsed(collapsed: boolean): void;
}

export const useUIStore = create<UIStoreState & UIStoreActions>((set) => ({
  sidebarCollapsed: false,

  toggleSidebar() {
    set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed }));
  },

  setSidebarCollapsed(collapsed) {
    set({ sidebarCollapsed: collapsed });
  },
}));
