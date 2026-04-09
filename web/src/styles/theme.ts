/** Theme tokens — modern dark-first design inspired by Railway/Roboflow. */

export interface ThemeTokens {
  bg: string;
  bgAlt: string;
  surface: string;
  surfaceAlt: string;
  surfaceRaised: string;
  surfaceBorder: string;
  text: string;
  textMuted: string;
  textSoft: string;
  accent: string;
  accentSoft: string;
  accentStrong: string;
  selection: string;
  navBg: string;
  navSurface: string;
  navText: string;
  navMuted: string;
  navSelected: string;
  ok: string;
  warn: string;
  danger: string;
  variantTailText: string;
  variantTailBg: string;
  variantTailBorder: string;
  shadow: string;
  canvas: string;
}

export const lightTheme: ThemeTokens = {
  bg: "#f4f4f5",
  bgAlt: "#ebebed",
  surface: "#ffffff",
  surfaceAlt: "#fafafa",
  surfaceRaised: "#ffffff",
  surfaceBorder: "#e4e4e7",
  text: "#09090b",
  textMuted: "#71717a",
  textSoft: "#a1a1aa",
  accent: "#0d9488",
  accentSoft: "#ccfbf1",
  accentStrong: "#0f766e",
  selection: "#99f6e4",
  navBg: "#18181b",
  navSurface: "#27272a",
  navText: "#e4e4e7",
  navMuted: "#a1a1aa",
  navSelected: "#14b8a6",
  ok: "#22c55e",
  warn: "#eab308",
  danger: "#ef4444",
  variantTailText: "#1d4ed8",
  variantTailBg: "#dbeafe",
  variantTailBorder: "#3b82f6",
  shadow: "rgba(0, 0, 0, 0.04)",
  canvas: "#e4e4e7",
};

export const darkTheme: ThemeTokens = {
  bg: "#09090b",
  bgAlt: "#0c0c0f",
  surface: "#111114",
  surfaceAlt: "#18181b",
  surfaceRaised: "#1e1e22",
  surfaceBorder: "#27272a",
  text: "#fafafa",
  textMuted: "#a1a1aa",
  textSoft: "#71717a",
  accent: "#14b8a6",
  accentSoft: "#042f2e",
  accentStrong: "#2dd4bf",
  selection: "#134e4a",
  navBg: "#09090b",
  navSurface: "#18181b",
  navText: "#e4e4e7",
  navMuted: "#71717a",
  navSelected: "#14b8a6",
  ok: "#34d399",
  warn: "#facc15",
  danger: "#f87171",
  variantTailText: "#facc15",
  variantTailBg: "rgba(245,158,11,0.08)",
  variantTailBorder: "#facc15",
  shadow: "rgba(0, 0, 0, 0.3)",
  canvas: "#0c0c0f",
};

export const themes = { light: lightTheme, dark: darkTheme } as const;
export type ThemeName = keyof typeof themes;

export function applyThemeToElement(
  el: HTMLElement,
  theme: ThemeTokens,
): void {
  const s = el.style;
  s.setProperty("--bg", theme.bg);
  s.setProperty("--bg-alt", theme.bgAlt);
  s.setProperty("--surface", theme.surface);
  s.setProperty("--surface-alt", theme.surfaceAlt);
  s.setProperty("--surface-raised", theme.surfaceRaised);
  s.setProperty("--surface-border", theme.surfaceBorder);
  s.setProperty("--text", theme.text);
  s.setProperty("--text-muted", theme.textMuted);
  s.setProperty("--text-soft", theme.textSoft);
  s.setProperty("--accent", theme.accent);
  s.setProperty("--accent-soft", theme.accentSoft);
  s.setProperty("--accent-strong", theme.accentStrong);
  s.setProperty("--selection", theme.selection);
  s.setProperty("--nav-bg", theme.navBg);
  s.setProperty("--nav-surface", theme.navSurface);
  s.setProperty("--nav-text", theme.navText);
  s.setProperty("--nav-muted", theme.navMuted);
  s.setProperty("--nav-selected", theme.navSelected);
  s.setProperty("--ok", theme.ok);
  s.setProperty("--warn", theme.warn);
  s.setProperty("--danger", theme.danger);
  s.setProperty("--variant-tail-text", theme.variantTailText);
  s.setProperty("--variant-tail-bg", theme.variantTailBg);
  s.setProperty("--variant-tail-border", theme.variantTailBorder);
  s.setProperty("--shadow", theme.shadow);
  s.setProperty("--canvas", theme.canvas);
}
