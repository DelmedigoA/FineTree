from __future__ import annotations

import argparse
import os
import shutil
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse

from .config import load_benchmark_config
from .info import discover_submission_info, load_submission_info_bundle
from .inputs import validate_mapping_files
from .logging_summary import parse_logging_jsonl_bytes
from .runner import SubmissionPayload, run_submission
from .storage import (
    leaderboard_columns,
    leaderboard_details,
    load_reports,
)
from .submission import parse_submission_form_data, submission_fields_for_api


def _nav_html() -> str:
    return """
    <nav class="nav">
      <a href="/submission">Submission</a>
      <a href="/leaderboard">Leaderboard</a>
    </nav>
    """


def _page_shell(*, title: str, body: str, script: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{
      --bg: linear-gradient(135deg, #f4efe6 0%, #fbfaf7 45%, #eef4f1 100%);
      --panel: rgba(255, 255, 255, 0.88);
      --line: rgba(39, 52, 42, 0.12);
      --ink: #1f2a24;
      --muted: #5d6b63;
      --accent: #134e4a;
      --accent-soft: rgba(19, 78, 74, 0.08);
      --error: #8f1d1d;
      --shadow: 0 18px 40px rgba(19, 32, 26, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Avenir Next", "Segoe UI", sans-serif;
    }}
    .wrap {{
      max-width: 1320px;
      margin: 0 auto;
      padding: 28px 20px 36px;
    }}
    .nav {{
      display: flex;
      gap: 14px;
      margin-bottom: 22px;
    }}
    .nav a {{
      color: var(--ink);
      text-decoration: none;
      padding: 10px 14px;
      border: 1px solid var(--line);
      border-radius: 999px;
      background: rgba(255,255,255,0.6);
      backdrop-filter: blur(10px);
    }}
    .hero {{
      display: grid;
      gap: 12px;
      margin-bottom: 22px;
    }}
    .hero h1 {{
      margin: 0;
      font-size: clamp(2rem, 5vw, 3.6rem);
      line-height: 0.95;
      letter-spacing: -0.04em;
    }}
    .hero p {{
      margin: 0;
      max-width: 780px;
      color: var(--muted);
      font-size: 1rem;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, minmax(0, 1fr));
      gap: 18px;
    }}
    .card {{
      grid-column: span 12;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 24px;
      box-shadow: var(--shadow);
      padding: 20px;
      backdrop-filter: blur(14px);
    }}
    .card h2 {{
      margin: 0 0 14px;
      font-size: 1rem;
      letter-spacing: 0.01em;
    }}
    .card p {{
      color: var(--muted);
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
    }}
    .stat {{
      padding: 14px;
      border-radius: 18px;
      background: var(--accent-soft);
      border: 1px solid rgba(19, 78, 74, 0.1);
    }}
    .stat strong {{
      display: block;
      font-size: 1.05rem;
      margin-top: 4px;
    }}
    .field-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
    }}
    label {{
      display: grid;
      gap: 8px;
      font-size: 0.92rem;
      color: var(--muted);
    }}
    input, select, textarea, button {{
      font: inherit;
    }}
    input, select, textarea {{
      width: 100%;
      padding: 11px 12px;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: rgba(255,255,255,0.94);
      color: var(--ink);
    }}
    textarea {{
      min-height: 108px;
      resize: vertical;
    }}
    .actions {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-top: 18px;
      flex-wrap: wrap;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      background: var(--accent);
      color: white;
      padding: 12px 18px;
      cursor: pointer;
    }}
    button.secondary {{
      background: transparent;
      color: var(--ink);
      border: 1px solid var(--line);
    }}
    .status {{
      font-size: 0.92rem;
      color: var(--muted);
    }}
    .status.error {{
      color: var(--error);
    }}
    .status.success {{
      color: var(--accent);
    }}
    .table-wrap {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 900px;
    }}
    th, td {{
      padding: 10px 12px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      vertical-align: top;
      font-size: 0.92rem;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: rgba(250, 250, 248, 0.96);
      cursor: pointer;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 5px 9px;
      border-radius: 999px;
      font-size: 0.78rem;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.86);
    }}
    .pill.error {{
      color: var(--error);
      border-color: rgba(143,29,29,0.2);
      background: rgba(143,29,29,0.08);
    }}
    .pill.ok {{
      color: var(--accent);
      border-color: rgba(19,78,74,0.2);
      background: rgba(19,78,74,0.08);
    }}
    .toolbar {{
      display: flex;
      gap: 12px;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .mono {{
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.84rem;
    }}
    .table-actions {{
      display: flex;
      gap: 8px;
      justify-content: flex-end;
    }}
    .details-btn {{
      padding: 8px 12px;
      font-size: 0.86rem;
    }}
    .delete-btn {{
      padding: 8px 12px;
      font-size: 0.86rem;
      background: rgba(143,29,29,0.12);
      color: var(--error);
      border: 1px solid rgba(143,29,29,0.2);
    }}
    .details-panel {{
      position: fixed;
      inset: 0;
      display: none;
      align-items: stretch;
      justify-content: flex-end;
      background: rgba(16, 24, 20, 0.28);
      z-index: 20;
    }}
    .details-panel.open {{
      display: flex;
    }}
    .details-card {{
      width: min(720px, 100%);
      height: 100%;
      background: rgba(251, 250, 247, 0.98);
      border-left: 1px solid var(--line);
      box-shadow: var(--shadow);
      padding: 22px;
      overflow: auto;
    }}
    .details-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .details-head h2 {{
      margin: 0;
      font-size: 1.15rem;
    }}
    .details-sections {{
      display: grid;
      gap: 16px;
    }}
    .details-sections section {{
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px;
      background: rgba(255,255,255,0.72);
    }}
    .details-sections h3 {{
      margin: 0 0 10px;
      font-size: 0.95rem;
    }}
    .details-sections pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 0.8rem;
      line-height: 1.45;
    }}
    @media (min-width: 980px) {{
      .split-left {{ grid-column: span 4; }}
      .split-right {{ grid-column: span 8; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    {_nav_html()}
    {body}
  </div>
  <script>
    {script}
  </script>
</body>
</html>
"""


def _submission_html() -> str:
    body = """
    <section class="hero">
      <h1>Benchmark Submission</h1>
      <p>Review the YAML benchmark config, confirm mapped prediction files, and run one benchmark submission. If <span class="mono">info.json</span> is present in the input folder, the page will use it directly and skip the manual metadata form.</p>
    </section>
    <div class="grid">
      <section class="card split-left">
        <h2>Config Summary</h2>
        <div id="config-summary" class="stats"></div>
      </section>
      <section class="card split-right">
        <h2>Mapped Files</h2>
        <div id="mapping-status" class="table-wrap"></div>
      </section>
      <section class="card">
        <h2>Submission Input</h2>
        <div id="submission-source" class="stats"></div>
        <form id="submission-form">
          <div id="field-grid" class="field-grid"></div>
          <div id="logging-field">
            <label>
              Logging JSONL
              <input name="logging_jsonl" id="logging-jsonl" type="file" accept=".jsonl,application/jsonl" required>
            </label>
          </div>
          <div class="actions">
            <button id="submit-btn" type="submit">Start Benchmark Submission</button>
            <a class="pill" href="/leaderboard">Open Leaderboard</a>
          </div>
          <p id="submission-status" class="status"></p>
        </form>
      </section>
    </div>
    """
    script = """
    const summaryEl = document.getElementById("config-summary");
    const mappingEl = document.getElementById("mapping-status");
    const sourceEl = document.getElementById("submission-source");
    const fieldGridEl = document.getElementById("field-grid");
    const loggingFieldEl = document.getElementById("logging-field");
    const formEl = document.getElementById("submission-form");
    const submitBtn = document.getElementById("submit-btn");
    const statusEl = document.getElementById("submission-status");
    const state = { mode: "manual", ready: true };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function renderSummary(data) {
      const cards = [
        ["Input Dir", data.benchmark.input_dir],
        ["Output Dir", data.benchmark.output_dir],
        ["Timezone", data.benchmark.timezone],
        ["Mappings", String(data.mapping_checks.length)],
        ["Normalize Inputs", String(data.evaluation.normalize_inputs)],
      ];
      summaryEl.innerHTML = cards.map(([label, value]) => `
        <div class="stat">
          <span>${escapeHtml(label)}</span>
          <strong class="mono">${escapeHtml(value)}</strong>
        </div>
      `).join("");
    }

    function renderMappings(rows) {
      mappingEl.innerHTML = `
        <table>
          <thead>
            <tr>
              <th>Prediction</th>
              <th>Ground Truth</th>
              <th>Format</th>
              <th>GT Page</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => `
              <tr>
                <td class="mono">${escapeHtml(row.prediction_file)}</td>
                <td class="mono">${escapeHtml(row.gt_file)}</td>
                <td>${escapeHtml(row.detected_prediction_format || row.configured_prediction_format)}</td>
                <td>${escapeHtml(row.gt_page_index ?? "document")}</td>
                <td>
                  <span class="pill ${row.status === "ok" ? "ok" : "error"}">${row.status}</span>
                  ${row.errors.length ? `<div class="mono">${row.errors.map((item) => escapeHtml(item)).join("<br>")}</div>` : ""}
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
    }

    function renderFields(fields) {
      fieldGridEl.innerHTML = fields.map((field) => {
        if (field.kind === "boolean") {
          return `
            <label>
              ${escapeHtml(field.label)}
              <select name="${field.name}" required>
                <option value="">Select</option>
                <option value="true" ${field.default === "true" ? "selected" : ""}>true</option>
                <option value="false" ${field.default === "false" ? "selected" : ""}>false</option>
              </select>
            </label>
          `;
        }
        if (field.kind === "textarea") {
          return `
            <label>
              ${escapeHtml(field.label)}
              <textarea name="${field.name}" required>${escapeHtml(field.default || "")}</textarea>
            </label>
          `;
        }
        const stepAttr = field.step ? `step="${field.step}"` : "";
        const inputType = field.kind === "number" ? "number" : "text";
        return `
          <label>
            ${escapeHtml(field.label)}
            <input name="${field.name}" type="${inputType}" value="${escapeHtml(field.default || "")}" ${stepAttr} required>
          </label>
        `;
      }).join("");
    }

    function renderSubmissionSource(infoSubmission) {
      if (!infoSubmission) {
        state.mode = "manual";
        state.ready = true;
        sourceEl.innerHTML = `
          <div class="stat">
            <span>Submission Mode</span>
            <strong>manual form + logging upload</strong>
          </div>
        `;
        fieldGridEl.style.display = "";
        loggingFieldEl.style.display = "";
        document.getElementById("logging-jsonl").required = true;
        submitBtn.disabled = false;
        return;
      }

      state.mode = "info_json";
      state.ready = infoSubmission.status === "ready";
      const metadata = infoSubmission.metadata || {};
      const selected = infoSubmission.selected_checkpoint || {};
      sourceEl.innerHTML = `
        <div class="stat">
          <span>Submission Mode</span>
          <strong>info.json</strong>
        </div>
        <div class="stat">
          <span>Checkpoint</span>
          <strong class="mono">${escapeHtml(metadata.checkpoint_name || selected.checkpoint_name || "")}</strong>
        </div>
        <div class="stat">
          <span>Model</span>
          <strong class="mono">${escapeHtml(metadata.model || "")}</strong>
        </div>
        <div class="stat">
          <span>Dataset</span>
          <strong class="mono">${escapeHtml(metadata.dataset || "")}</strong>
        </div>
        <div class="stat">
          <span>Selected Best</span>
          <strong class="mono">${escapeHtml(selected.checkpoint_name || "")}</strong>
        </div>
        <div class="stat">
          <span>Logging</span>
          <strong class="mono">${escapeHtml(infoSubmission.logging_path || "")}</strong>
        </div>
        ${infoSubmission.errors && infoSubmission.errors.length ? `
          <div class="stat" style="grid-column: 1 / -1;">
            <span>Errors</span>
            <strong class="mono">${infoSubmission.errors.map((item) => escapeHtml(item)).join("<br>")}</strong>
          </div>
        ` : ""}
      `;
      fieldGridEl.innerHTML = "";
      fieldGridEl.style.display = "none";
      loggingFieldEl.style.display = "none";
      document.getElementById("logging-jsonl").required = false;
      submitBtn.disabled = !state.ready;
    }

    async function loadConfig() {
      const response = await fetch("/api/config");
      if (!response.ok) {
        throw new Error("Failed to load benchmark config.");
      }
      const data = await response.json();
      renderSummary(data);
      renderMappings(data.mapping_checks);
      renderSubmissionSource(data.info_submission);
      if (!data.info_submission) {
        renderFields(data.submission_fields);
      }
    }

    formEl.addEventListener("submit", async (event) => {
      event.preventDefault();
      statusEl.className = "status";
      statusEl.textContent = "Submitting benchmark job...";
      if (!state.ready) {
        statusEl.className = "status error";
        statusEl.textContent = "Submission files are not ready.";
        return;
      }
      const formData = new FormData(formEl);
      const response = await fetch("/api/submissions", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();
      if (!response.ok) {
        statusEl.className = "status error";
        statusEl.textContent = payload.detail || "Submission failed.";
        return;
      }
      statusEl.className = "status success";
      statusEl.innerHTML = `Saved <span class="mono">${escapeHtml(payload.submission_id)}</span> to <span class="mono">${escapeHtml(payload.submission_dir)}</span>. <a href="/leaderboard">Open leaderboard</a>.`;
    });

    loadConfig().catch((error) => {
      statusEl.className = "status error";
      statusEl.textContent = error.message;
    });
    """
    return _page_shell(title="Benchmark Submission", body=body, script=script)


def _leaderboard_html() -> str:
    body = """
    <section class="hero">
      <h1>Benchmark Leaderboard</h1>
      <p>Compare persisted submissions by evaluation metrics, training configuration, and logging-derived summary fields.</p>
    </section>
    <div class="grid">
      <section class="card">
        <div class="toolbar">
          <input id="filter-input" type="text" placeholder="Filter rows or settings">
          <button id="refresh-btn" class="secondary" type="button">Refresh</button>
          <span id="leaderboard-status" class="status"></span>
        </div>
        <div id="leaderboard-table" class="table-wrap"></div>
      </section>
    </div>
    <div id="details-panel" class="details-panel" aria-hidden="true">
      <aside class="details-card">
        <div class="details-head">
          <h2 id="details-title">Submission Details</h2>
          <div class="table-actions">
            <button id="delete-details-btn" class="delete-btn" type="button">Delete</button>
            <button id="close-details-btn" class="secondary" type="button">Close</button>
          </div>
        </div>
        <div id="details-content" class="details-sections"></div>
      </aside>
    </div>
    """
    script = """
    const filterInput = document.getElementById("filter-input");
    const refreshBtn = document.getElementById("refresh-btn");
    const statusEl = document.getElementById("leaderboard-status");
    const tableEl = document.getElementById("leaderboard-table");
    const detailsPanelEl = document.getElementById("details-panel");
    const detailsTitleEl = document.getElementById("details-title");
    const detailsContentEl = document.getElementById("details-content");
    const closeDetailsBtn = document.getElementById("close-details-btn");
    const deleteDetailsBtn = document.getElementById("delete-details-btn");
    const state = { rows: [], columns: [], sortKey: "overall_score", sortDir: "desc", activeSubmissionId: null };
    const columnLabels = {
      model: "Model Name",
      report_timestamp_israel: "Date",
      dataset: "Dataset",
      overall_score: "Overall Score",
      meta_score: "Meta Score",
      facts_score: "Facts Score",
      meta_hard_score: "Meta Hard Score",
      entity_score: "Entity Score",
      title_score: "Title Score",
      page_num_score: "Page Num Score",
      page_type_score: "Page Type Score",
      statement_type_score: "Statement Type Score",
      facts_count_score: "Fact Count Score",
      date_score: "Date Score",
      row_role_score: "Row Role Score",
      comment_ref_score: "Comment Ref Score",
      note_flag_score: "Note Flag Score",
      note_name_score: "Note Name Score",
      note_num_score: "Note Num Score",
      note_ref_score: "Note Ref Score",
      path_source_score: "Path Source Score",
      currency_score: "Currency Score",
      scale_score: "Scale Score",
      value_type_score: "Value Type Score",
      value_context_score: "Value Context Score",
    };

    function escapeHtml(value) {
      return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function formatCell(column, value) {
      if (value === null || value === undefined || value === "") {
        return "";
      }
      if (column === "report_timestamp_israel") {
        const date = new Date(String(value));
        return Number.isNaN(date.getTime()) ? String(value) : date.toLocaleString();
      }
      const numericValue = Number(value);
      if (column.includes("score") && !Number.isNaN(numericValue)) {
        return numericValue.toFixed(4);
      }
      return String(value);
    }

    function compareValues(left, right) {
      if (left === right) return 0;
      if (left === null || left === undefined || left === "") return 1;
      if (right === null || right === undefined || right === "") return -1;
      const leftNum = Number(left);
      const rightNum = Number(right);
      if (!Number.isNaN(leftNum) && !Number.isNaN(rightNum)) {
        return leftNum < rightNum ? -1 : 1;
      }
      return String(left).localeCompare(String(right), undefined, { numeric: true, sensitivity: "base" });
    }

    function filteredRows() {
      const query = filterInput.value.trim().toLowerCase();
      let rows = [...state.rows];
      if (query) {
        rows = rows.filter((row) => {
          const visibleText = Object.values(row).map((value) => String(value ?? "")).join(" ");
          const detailText = JSON.stringify(row._details || {});
          return `${visibleText} ${detailText}`.toLowerCase().includes(query);
        });
      }
      rows.sort((left, right) => {
        const result = compareValues(left[state.sortKey], right[state.sortKey]);
        return state.sortDir === "asc" ? result : -result;
      });
      return rows;
    }

    function renderDetailSections(details) {
      const sections = [
        ["Submission Metadata", details.submission_metadata],
        ["Aggregate Metrics", details.aggregate_metrics],
        ["Logging Summary", details.logging_summary],
        ["Submission Context", details.submission_context],
      ].filter(([, value]) => value && Object.keys(value).length);
      detailsContentEl.innerHTML = sections.map(([label, value]) => `
        <section>
          <h3>${escapeHtml(label)}</h3>
          <pre>${escapeHtml(JSON.stringify(value, null, 2))}</pre>
        </section>
      `).join("");
    }

    function openDetails(row) {
      const details = row._details || {};
      detailsTitleEl.textContent = row.model ? `${row.model} details` : "Submission Details";
      state.activeSubmissionId = row.submission_id;
      renderDetailSections(details);
      detailsPanelEl.classList.add("open");
      detailsPanelEl.setAttribute("aria-hidden", "false");
    }

    function closeDetails() {
      state.activeSubmissionId = null;
      detailsPanelEl.classList.remove("open");
      detailsPanelEl.setAttribute("aria-hidden", "true");
    }

    async function deleteSubmission(submissionId) {
      if (!submissionId) {
        return;
      }
      if (!window.confirm(`Delete submission ${submissionId}?`)) {
        return;
      }
      statusEl.className = "status";
      statusEl.textContent = "Deleting submission...";
      const response = await fetch(`/api/submissions/${encodeURIComponent(submissionId)}`, {
        method: "DELETE",
      });
      const payload = await response.json();
      if (!response.ok) {
        statusEl.className = "status error";
        statusEl.textContent = payload.detail || "Delete failed.";
        return;
      }
      closeDetails();
      statusEl.className = "status success";
      statusEl.textContent = `Deleted ${submissionId}.`;
      await loadLeaderboard();
    }

    function renderTable() {
      const rows = filteredRows();
      statusEl.textContent = `${rows.length} row(s)`;
      if (!state.columns.length) {
        tableEl.innerHTML = "<p class=\\"status\\">No submissions yet.</p>";
        return;
      }
      tableEl.innerHTML = `
        <table>
          <thead>
            <tr>
              ${state.columns.map((column) => `
                <th data-column="${column}">
                  ${escapeHtml(columnLabels[column] || column)}
                  ${state.sortKey === column ? (state.sortDir === "asc" ? " ▲" : " ▼") : ""}
                </th>
              `).join("")}
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            ${rows.map((row) => `
              <tr>
                ${state.columns.map((column) => `<td class="${column.includes("path") || column.includes("id") ? "mono" : ""}">${escapeHtml(formatCell(column, row[column]))}</td>`).join("")}
                <td class="table-actions">
                  <button type="button" class="secondary details-btn" data-submission-id="${escapeHtml(row.submission_id)}">View</button>
                  <button type="button" class="delete-btn row-delete-btn" data-submission-id="${escapeHtml(row.submission_id)}">Delete</button>
                </td>
              </tr>
            `).join("")}
          </tbody>
        </table>
      `;
      tableEl.querySelectorAll("th").forEach((cell) => {
        cell.addEventListener("click", () => {
          const column = cell.getAttribute("data-column");
          if (state.sortKey === column) {
            state.sortDir = state.sortDir === "asc" ? "desc" : "asc";
          } else {
            state.sortKey = column;
            state.sortDir = column.includes("score") ? "desc" : "asc";
          }
          renderTable();
        });
      });
      tableEl.querySelectorAll(".details-btn").forEach((button) => {
        button.addEventListener("click", () => {
          const submissionId = button.getAttribute("data-submission-id");
          const row = state.rows.find((item) => item.submission_id === submissionId);
          if (row) {
            openDetails(row);
          }
        });
      });
      tableEl.querySelectorAll(".row-delete-btn").forEach((button) => {
        button.addEventListener("click", () => {
          const submissionId = button.getAttribute("data-submission-id");
          deleteSubmission(submissionId).catch((error) => {
            statusEl.className = "status error";
            statusEl.textContent = error.message;
          });
        });
      });
    }

    async function loadLeaderboard() {
      statusEl.textContent = "Loading leaderboard...";
      const response = await fetch("/api/leaderboard");
      if (!response.ok) {
        throw new Error("Failed to load leaderboard.");
      }
      const data = await response.json();
      state.rows = data.rows;
      state.columns = data.columns;
      renderTable();
    }

    filterInput.addEventListener("input", renderTable);
    closeDetailsBtn.addEventListener("click", closeDetails);
    deleteDetailsBtn.addEventListener("click", () => {
      deleteSubmission(state.activeSubmissionId).catch((error) => {
        statusEl.className = "status error";
        statusEl.textContent = error.message;
      });
    });
    detailsPanelEl.addEventListener("click", (event) => {
      if (event.target === detailsPanelEl) {
        closeDetails();
      }
    });
    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape") {
        closeDetails();
      }
    });
    refreshBtn.addEventListener("click", () => loadLeaderboard().catch((error) => {
      statusEl.className = "status error";
      statusEl.textContent = error.message;
    }));
    loadLeaderboard().catch((error) => {
      statusEl.className = "status error";
      statusEl.textContent = error.message;
    });
    """
    return _page_shell(title="Benchmark Leaderboard", body=body, script=script)


def create_app(*, config_path: str | os.PathLike[str]) -> FastAPI:
    cfg_path = Path(config_path).expanduser().resolve()
    cfg = load_benchmark_config(cfg_path)
    cfg.benchmark.output_dir.mkdir(parents=True, exist_ok=True)

    app = FastAPI(title="FineTree Benchmark", version="1.0.0")
    app.state.cfg = cfg
    app.state.cfg_path = cfg_path
    app.state.submission_lock = threading.Lock()

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/submission", status_code=307)

    @app.get("/submission", response_class=HTMLResponse)
    def submission_page() -> HTMLResponse:
        return HTMLResponse(_submission_html())

    @app.get("/leaderboard", response_class=HTMLResponse)
    def leaderboard_page() -> HTMLResponse:
        return HTMLResponse(_leaderboard_html())

    @app.get("/api/config")
    def api_config() -> dict[str, Any]:
        info_submission = discover_submission_info(cfg.benchmark.input_dir)
        return {
            "config_path": str(app.state.cfg_path),
            "benchmark": cfg.benchmark.model_dump(mode="json"),
            "methods": cfg.methods.model_dump(mode="json"),
            "weighting": cfg.weighting.model_dump(mode="json"),
            "evaluation": cfg.evaluation.model_dump(mode="json"),
            "submission_fields": submission_fields_for_api(cfg.model_metadata),
            "info_submission": info_submission,
            "mapping_checks": validate_mapping_files(cfg),
        }

    @app.get("/api/leaderboard")
    def api_leaderboard() -> dict[str, Any]:
        reports = load_reports(cfg.benchmark.output_dir)
        rows = [
            {
                **report.leaderboard_row,
                "_details": leaderboard_details(report),
            }
            for report in reports
        ]
        return {"rows": rows, "columns": leaderboard_columns(rows)}

    @app.post("/api/submissions")
    async def api_submissions(request: Request) -> dict[str, Any]:
        submission_lock = app.state.submission_lock
        if not submission_lock.acquire(blocking=False):
            raise HTTPException(status_code=409, detail="A benchmark submission is already running.")
        try:
            info_submission = discover_submission_info(cfg.benchmark.input_dir)
            try:
                if info_submission is not None:
                    bundle = load_submission_info_bundle(cfg.benchmark.input_dir)
                    submission = SubmissionPayload(
                        submission_metadata=bundle.info.model_metadata,
                        logging_filename=bundle.logging_path.name,
                        logging_text=bundle.logging_text,
                        logging_rows=bundle.logging_rows,
                        source="info_json",
                        source_submission_dir=bundle.submission_dir,
                        submission_context=bundle.context,
                    )
                    report, submission_dir = run_submission(
                        cfg=cfg,
                        cfg_path=app.state.cfg_path,
                        submission=submission,
                        input_dir_override=bundle.submission_dir,
                    )
                else:
                    form = await request.form()
                    upload = form.get("logging_jsonl")
                    if upload is None or not hasattr(upload, "filename") or not hasattr(upload, "read"):
                        raise HTTPException(status_code=422, detail="logging.jsonl upload is required.")
                    if Path(upload.filename or "").name != "logging.jsonl":
                        raise HTTPException(status_code=422, detail="Uploaded file must be named logging.jsonl.")
                    submission_metadata = parse_submission_form_data(form)
                    logging_bytes = await upload.read()
                    logging_text, logging_rows = parse_logging_jsonl_bytes(logging_bytes)
                    submission = SubmissionPayload(
                        submission_metadata=submission_metadata,
                        logging_filename="logging.jsonl",
                        logging_text=logging_text,
                        logging_rows=logging_rows,
                    )
                    report, submission_dir = run_submission(
                        cfg=cfg,
                        cfg_path=app.state.cfg_path,
                        submission=submission,
                    )
            except (FileNotFoundError, ValueError) as exc:
                raise HTTPException(status_code=422, detail=str(exc)) from exc
            return {
                "status": "ok",
                "submission_id": report.submission_id,
                "submission_dir": str(submission_dir),
                "aggregate_metrics": report.aggregate_metrics,
            }
        finally:
            submission_lock.release()

    @app.delete("/api/submissions/{submission_id}")
    def api_delete_submission(submission_id: str) -> dict[str, Any]:
        submissions_dir = cfg.benchmark.output_dir / "submissions"
        target_dir = (submissions_dir / submission_id).resolve()
        if target_dir.parent != submissions_dir.resolve():
            raise HTTPException(status_code=400, detail="Invalid submission id.")
        if not target_dir.is_dir():
            raise HTTPException(status_code=404, detail="Submission not found.")
        shutil.rmtree(target_dir)
        return {"status": "ok", "submission_id": submission_id}

    return app


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the FineTree benchmark submission and leaderboard UI.")
    parser.add_argument("--config", required=True, help="Benchmark YAML config path.")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", "8123")),
        help="Bind port (default: 8123 or $PORT)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        import uvicorn
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Benchmark UI requires uvicorn. Install with `pip install uvicorn`.") from exc
    app = create_app(config_path=args.config)
    uvicorn.run(app, host=str(args.host), port=int(args.port), log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
