/** Shared field widgets for the inspector panel. Clean, compact design. */

import type { CSSProperties } from "react";

// ── Layout ──────────────────────────────────────────────────────────

export function FieldRow({
  label,
  children,
}: {
  label: string;
  children: React.ReactNode;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <label
        style={{
          width: 72,
          minWidth: 72,
          fontSize: 12,
          fontWeight: 500,
          color: "var(--text-soft)",
          textAlign: "right",
          whiteSpace: "nowrap",
        }}
      >
        {label}
      </label>
      <div style={{ flex: 1 }}>{children}</div>
    </div>
  );
}

// ── Input ───────────────────────────────────────────────────────────

const inputStyle: CSSProperties = {
  width: "100%",
  padding: "6px 10px",
  fontSize: 13,
  fontFamily: "var(--font-body)",
  color: "var(--text)",
  background: "var(--surface-alt)",
  border: "1px solid var(--surface-border)",
  borderRadius: "var(--radius-xs)",
  outline: "none",
  transition: "var(--transition-fast)",
};

export function FieldInput({
  value,
  placeholder,
  onChange,
  type = "text",
}: {
  value: string;
  placeholder?: string;
  onChange: (value: string) => void;
  type?: string;
}) {
  return (
    <input
      type={type}
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      style={inputStyle}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
      onBlur={(e) =>
        (e.currentTarget.style.borderColor = "var(--surface-border)")
      }
    />
  );
}

// ── Select ──────────────────────────────────────────────────────────

export function FieldSelect({
  value,
  options,
  onChange,
}: {
  value: string;
  options: { value: string; label: string }[];
  onChange: (value: string) => void;
}) {
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      style={{
        ...inputStyle,
        appearance: "none",
        paddingRight: 24,
        backgroundImage:
          "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath d='M3 5l3 3 3-3' stroke='%239898ab' stroke-width='1.5' fill='none'/%3E%3C/svg%3E\")",
        backgroundRepeat: "no-repeat",
        backgroundPosition: "right 8px center",
        cursor: "pointer",
      }}
      onFocus={(e) => (e.currentTarget.style.borderColor = "var(--accent)")}
      onBlur={(e) =>
        (e.currentTarget.style.borderColor = "var(--surface-border)")
      }
    >
      {options.map((opt) => (
        <option key={opt.value} value={opt.value}>
          {opt.label}
        </option>
      ))}
    </select>
  );
}

// ── Checkbox ────────────────────────────────────────────────────────

export function FieldCheckbox({
  checked,
  label,
  onChange,
}: {
  checked: boolean;
  label: string;
  onChange: (checked: boolean) => void;
}) {
  return (
    <label
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        fontSize: 13,
        color: "var(--text)",
        cursor: "pointer",
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{
          width: 16,
          height: 16,
          accentColor: "var(--accent)",
          cursor: "pointer",
        }}
      />
      {label}
    </label>
  );
}
