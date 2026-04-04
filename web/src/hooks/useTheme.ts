import { useState, useEffect, useCallback } from "react";
import { themes, applyThemeToElement } from "../styles/theme";
import type { ThemeName } from "../styles/theme";

const STORAGE_KEY = "finetree-theme";

function loadSavedTheme(): ThemeName {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === "light" || saved === "dark") return saved;
  } catch {
    /* ignore */
  }
  return "dark";
}

export function useTheme() {
  const [themeName, setThemeNameState] = useState<ThemeName>(loadSavedTheme);

  const setThemeName = useCallback((name: ThemeName) => {
    setThemeNameState(name);
    try {
      localStorage.setItem(STORAGE_KEY, name);
    } catch {
      /* ignore */
    }
  }, []);

  const toggleTheme = useCallback(() => {
    setThemeName(themeName === "light" ? "dark" : "light");
  }, [themeName, setThemeName]);

  useEffect(() => {
    applyThemeToElement(document.documentElement, themes[themeName]);
    document.documentElement.setAttribute("data-theme", themeName);
  }, [themeName]);

  return { themeName, setThemeName, toggleTheme, tokens: themes[themeName] };
}
