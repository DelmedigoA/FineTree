import { BrowserRouter, Routes, Route, NavLink, useNavigate } from "react-router-dom";
import { AppShell } from "./components/layout/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { AnnotationPage } from "./pages/AnnotationPage";
import { DatasetsPage } from "./pages/DatasetsPage";
import { DatasetVersionPage } from "./pages/DatasetVersionPage";
import { SettingsDialog } from "./components/dialogs/SettingsDialog";
import { AddDatasetDialog } from "./components/dialogs/AddDatasetDialog";
import { useTheme } from "./hooks/useTheme";
import { useUIStore } from "./stores/uiStore";
import { get } from "./api/client";

interface RandomApprovedPage {
  doc_id: string | null;
  page_index: number | null;
}

// ── Inner nav components (must be inside BrowserRouter for useNavigate) ──────

function SidebarNav() {
  const { themeName, toggleTheme } = useTheme();
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const toggleSettings = useUIStore((s) => s.toggleSettings);
  const navigate = useNavigate();

  async function handleRandomCheck() {
    try {
      const result = await get<RandomApprovedPage>("/workspace/random-approved-page");
      if (!result.doc_id) {
        alert("No approved pages found.");
        return;
      }
      navigate(`/annotate/${result.doc_id}?page=${result.page_index ?? 0}`);
    } catch {
      alert("Could not fetch a random approved page.");
    }
  }

  return (
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
        <SidebarLink to="/" icon={"\u25A6"} end>
          Documents
        </SidebarLink>
        <SidebarBtn onClick={handleRandomCheck} icon={"🎲"}>
          Random Check
        </SidebarBtn>
        <SidebarLink to="/datasets" icon={"🗃"}>
          Datasets
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
        <SidebarBtn onClick={toggleSettings} icon={"\u2699"}>
          Settings
        </SidebarBtn>
        <SidebarBtn onClick={toggleTheme} icon={themeName === "light" ? "\u263E" : "\u2600"}>
          {themeName === "light" ? "Dark mode" : "Light mode"}
        </SidebarBtn>
        <SidebarBtn onClick={toggleSidebar} icon={"\u00AB"}>
          Collapse
        </SidebarBtn>
      </div>
    </>
  );
}

function SidebarNavCollapsed() {
  const { themeName, toggleTheme } = useTheme();
  const toggleSidebar = useUIStore((s) => s.toggleSidebar);
  const toggleSettings = useUIStore((s) => s.toggleSettings);
  const navigate = useNavigate();

  async function handleRandomCheck() {
    try {
      const result = await get<RandomApprovedPage>("/workspace/random-approved-page");
      if (!result.doc_id) return;
      navigate(`/annotate/${result.doc_id}?page=${result.page_index ?? 0}`);
    } catch {
      /* ignore */
    }
  }

  return (
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
        <CollapsedBtn onClick={handleRandomCheck} icon={"🎲"} />
        <CollapsedLink to="/datasets" icon={"🗃"} />
      </nav>

      <div style={{ flex: 1 }} />

      <div style={{ display: "flex", flexDirection: "column", gap: 2, alignItems: "center" }}>
        <CollapsedBtn onClick={toggleSettings} icon={"\u2699"} />
        <CollapsedBtn onClick={toggleTheme} icon={themeName === "light" ? "\u263E" : "\u2600"} />
        <CollapsedBtn onClick={toggleSidebar} icon={"\u00BB"} />
      </div>
    </>
  );
}

// ── Root component ────────────────────────────────────────────────────────────

export default function App() {
  return (
    <BrowserRouter>
      <AppShell nav={<SidebarNav />} navCollapsed={<SidebarNavCollapsed />}>
        <Routes>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/datasets/:versionId" element={<DatasetVersionPage />} />
          <Route path="/annotate/:docId" element={<AnnotationPage />} />
        </Routes>
        <SettingsDialog />
        <AddDatasetDialog />
      </AppShell>
    </BrowserRouter>
  );
}

// ── Sidebar primitives ────────────────────────────────────────────────────────

function SidebarLink({
  to,
  icon,
  children,
  end,
}: {
  to: string;
  icon: string;
  children: React.ReactNode;
  end?: boolean;
}) {
  return (
    <NavLink
      to={to}
      end={end}
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
      end={to === "/"}
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
