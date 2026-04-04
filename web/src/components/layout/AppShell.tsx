/** Top-level app shell — collapsible dark sidebar + content area. */

import type { ReactNode } from "react";
import { useUIStore } from "../../stores/uiStore";

interface AppShellProps {
  nav: ReactNode;
  navCollapsed: ReactNode;
  children: ReactNode;
}

export function AppShell({ nav, navCollapsed, children }: AppShellProps) {
  const collapsed = useUIStore((s) => s.sidebarCollapsed);

  return (
    <div style={{ display: "flex", height: "100%", width: "100%" }}>
      <aside
        style={{
          width: collapsed ? 56 : 240,
          minWidth: collapsed ? 56 : 240,
          background: "var(--nav-bg)",
          display: "flex",
          flexDirection: "column",
          padding: collapsed ? "16px 8px" : "20px 12px 16px",
          gap: 4,
          overflowY: "auto",
          overflowX: "hidden",
          borderRight: "1px solid var(--surface-border)",
          transition: "width 0.2s ease, min-width 0.2s ease, padding 0.2s ease",
        }}
      >
        {collapsed ? navCollapsed : nav}
      </aside>
      <main
        style={{
          flex: 1,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          background: "var(--bg)",
        }}
      >
        {children}
      </main>
    </div>
  );
}
