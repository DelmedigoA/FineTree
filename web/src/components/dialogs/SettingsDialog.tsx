/** Settings dialog — bbox color customization. */

import { useUIStore } from "../../stores/uiStore";
import { useSettingsStore } from "../../stores/settingsStore";
import type { BBoxColors } from "../../stores/settingsStore";

const BBOX_LABELS: { key: keyof BBoxColors; label: string; description: string }[] = [
  { key: "default",      label: "Default",       description: "Unselected bboxes" },
  { key: "selected",     label: "Selected",       description: "Currently selected bbox(es)" },
  { key: "hovered",      label: "Hovered",        description: "Bbox under the cursor" },
  { key: "equationOk",   label: "Equation OK",    description: "Bbox with a valid equation" },
  { key: "equationBad",  label: "Equation Bad",   description: "Bbox with an invalid equation" },
  { key: "equationTerm", label: "Equation Term",  description: "Bbox selected as equation term" },
];

export function SettingsDialog() {
  const settingsOpen = useUIStore((s) => s.settingsOpen);
  const toggleSettings = useUIStore((s) => s.toggleSettings);
  const { bboxColors, setBboxColor, resetBboxColors } = useSettingsStore();

  if (!settingsOpen) return null;

  return (
    <div
      onClick={toggleSettings}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 1500,
        background: "rgba(0,0,0,0.45)",
        backdropFilter: "blur(3px)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--surface)",
          border: "1px solid var(--border)",
          borderRadius: "var(--radius-md)",
          padding: "28px 32px",
          width: 480,
          maxWidth: "calc(100vw - 48px)",
          boxShadow: "0 24px 64px rgba(0,0,0,0.35)",
        }}
      >
        {/* Header */}
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
          <div>
            <div style={{ fontSize: 16, fontWeight: 700, color: "var(--text-primary)" }}>Settings</div>
            <div style={{ fontSize: 12, color: "var(--text-muted)", marginTop: 2 }}>Bounding box colors</div>
          </div>
          <button
            onClick={toggleSettings}
            style={{
              background: "transparent",
              border: "none",
              color: "var(--text-muted)",
              fontSize: 20,
              cursor: "pointer",
              lineHeight: 1,
              padding: 4,
            }}
          >
            ×
          </button>
        </div>

        {/* Color rows */}
        <div style={{ display: "flex", flexDirection: "column", gap: 16 }}>
          {BBOX_LABELS.map(({ key, label, description }) => {
            const entry = bboxColors[key];
            const opacityPct = Math.round(entry.opacity * 100);
            return (
              <div key={key} style={{ display: "grid", gridTemplateColumns: "1fr auto 180px", alignItems: "center", gap: 12 }}>
                {/* Label */}
                <div>
                  <div style={{ fontSize: 13, fontWeight: 500, color: "var(--text-primary)" }}>{label}</div>
                  <div style={{ fontSize: 11, color: "var(--text-muted)" }}>{description}</div>
                </div>

                {/* Color picker */}
                <input
                  type="color"
                  value={entry.color}
                  onChange={(e) => setBboxColor(key, { color: e.target.value })}
                  style={{
                    width: 36,
                    height: 28,
                    border: "1px solid var(--border)",
                    borderRadius: 4,
                    padding: 2,
                    cursor: "pointer",
                    background: "var(--surface-raised)",
                  }}
                />

                {/* Opacity slider */}
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <input
                    type="range"
                    min={0}
                    max={1}
                    step={0.05}
                    value={entry.opacity}
                    onChange={(e) => setBboxColor(key, { opacity: parseFloat(e.target.value) })}
                    style={{ flex: 1, accentColor: "var(--accent)", cursor: "pointer" }}
                  />
                  <span style={{ fontSize: 11, color: "var(--text-muted)", width: 30, textAlign: "right" }}>
                    {opacityPct}%
                  </span>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer */}
        <div style={{ marginTop: 28, display: "flex", justifyContent: "flex-end" }}>
          <button
            onClick={resetBboxColors}
            style={{
              padding: "7px 16px",
              fontSize: 13,
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)",
              background: "var(--surface-raised)",
              color: "var(--text-secondary)",
              cursor: "pointer",
            }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "var(--surface-hover)")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "var(--surface-raised)")}
          >
            Reset to defaults
          </button>
        </div>
      </div>
    </div>
  );
}
