import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { AnnotationPage } from "./pages/AnnotationPage";
import { useTheme } from "./hooks/useTheme";
import { useUIStore } from "./stores/uiStore";

export default function App() {
  const { themeName, toggleTheme } = useTheme();
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);

  // Full sidebar
  const nav = (
    <>
      {/* Brand */}
      <div style={{ padding: "0 8px", marginBottom: 24 }}>
        <div
          style={{
            fontFamily: "var(--font-heading)",
            fontSize: 20,
            fontWeight: 800,
            color: "#ffffff",
            letterSpacing: "-0.02em",
          }}
        >
          FineTree
        </div>
        <div style={{ fontSize: 11, color: "var(--nav-muted)", marginTop: 2 }}>
          Document Annotator
        </div>
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <SidebarLink to="/" icon={"\u25A6"}>
          Projects
        </SidebarLink>
      </nav>

      <div style={{ flex: 1 }} />

      <div
        style={{
          display: "flex",
          flexDirection: "column",
          gap: 2,
          borderTop: "1px solid rgba(255,255,255,0.06)",
          paddingTop: 12,
          marginTop: 12,
        }}
      >
        <SidebarBtn onClick={toggleTheme} icon={themeName === "light" ? "\u263E" : "\u2600"}>
          {themeName === "light" ? "Dark mode" : "Light mode"}
        </SidebarBtn>
        <SidebarBtn onClick={toggleSidebar} icon={"\u00AB"}>
          Collapse
        </SidebarBtn>
      </div>
    </>
  );

  // Collapsed sidebar (icons only)
  const navCollapsed = (
    <>
      <div
        style={{
          fontFamily: "var(--font-heading)",
          fontSize: 18,
          fontWeight: 800,
          color: "#ffffff",
          textAlign: "center",
          marginBottom: 16,
        }}
      >
        F
      </div>

      <nav style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        <CollapsedLink to="/" icon={"\u25A6"} />
      </nav>

      <div style={{ flex: 1 }} />

      <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        <CollapsedBtn onClick={toggleTheme} icon={themeName === "light" ? "\u263E" : "\u2600"} />
        <CollapsedBtn onClick={toggleSidebar} icon={"\u00BB"} />
      </div>
    </>
  );

  return (
    <BrowserRouter>
      <AppShell nav={nav} navCollapsed={navCollapsed}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/annotate/:docId" element={<AnnotationPage />} />
        </Routes>
      </AppShell>
    </BrowserRouter>
  );
}

// ── Sidebar components ──────────────────────────────────────────────

function SidebarLink({
  to,
  icon,
  children,
}: {
  to: string;
  icon: string;
  children: React.ReactNode;
}) {
  return (
    <NavLink
      to={to}
      end
      style={({ isActive }) => ({
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 12px",
        borderRadius: "var(--radius-sm)",
        color: isActive ? "var(--nav-selected)" : "var(--nav-text)",
        background: isActive ? "rgba(20, 184, 166, 0.12)" : "transparent",
        fontSize: 13,
        fontWeight: isActive ? 600 : 500,
        textDecoration: "none",
        transition: "var(--transition-fast)",
      })}
    >
      <span style={{ fontSize: 15, width: 20, textAlign: "center", opacity: 0.8 }}>{icon}</span>
      {children}
    </NavLink>
  );
}

function SidebarBtn({
  onClick,
  icon,
  children,
}: {
  onClick: () => void;
  icon: string;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 10,
        padding: "10px 12px",
        borderRadius: "var(--radius-sm)",
        color: "var(--nav-muted)",
        fontSize: 13,
        fontWeight: 500,
        transition: "var(--transition-fast)",
        background: "transparent",
        textAlign: "left",
        width: "100%",
        cursor: "pointer",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--nav-surface)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      <span style={{ fontSize: 15, width: 20, textAlign: "center", opacity: 0.8 }}>{icon}</span>
      {children}
    </button>
  );
}

function CollapsedLink({ to, icon }: { to: string; icon: string }) {
  return (
    <NavLink
      to={to}
      end
      style={({ isActive }) => ({
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 40,
        height: 40,
        borderRadius: "var(--radius-sm)",
        color: isActive ? "var(--nav-selected)" : "var(--nav-text)",
        background: isActive ? "rgba(20, 184, 166, 0.12)" : "transparent",
        fontSize: 16,
        textDecoration: "none",
        transition: "var(--transition-fast)",
      })}
    >
      {icon}
    </NavLink>
  );
}

function CollapsedBtn({ onClick, icon }: { onClick: () => void; icon: string }) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        width: 40,
        height: 40,
        borderRadius: "var(--radius-sm)",
        color: "var(--nav-muted)",
        fontSize: 16,
        background: "transparent",
        cursor: "pointer",
        transition: "var(--transition-fast)",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "var(--nav-surface)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      {icon}
    </button>
  );
}
