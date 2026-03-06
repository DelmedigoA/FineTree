import type { CSSProperties, ReactNode } from "react";

type IconName =
  | "cursor"
  | "hand"
  | "zoom"
  | "tag"
  | "box"
  | "polygon"
  | "review"
  | "left"
  | "right"
  | "fullscreen"
  | "done"
  | "chevron"
  | "lock"
  | "eye"
  | "trash";

type ToolItem = {
  label: string;
  icon: IconName;
  active?: boolean;
  badge?: string;
};

type PageItem = {
  name: string;
  type: string;
  state: "done" | "active" | "flagged";
};

type RegionItem = {
  id: string;
  name: string;
  path: string;
  value: string;
  tone: "mint" | "amber" | "sky" | "rose" | "lime";
  active?: boolean;
  locked?: boolean;
  hidden?: boolean;
};

type StageRegion = {
  label: string;
  tone: RegionItem["tone"];
  style: CSSProperties;
  active?: boolean;
};

const tools: ToolItem[] = [
  { label: "Select", icon: "cursor", active: true },
  { label: "Pan", icon: "hand" },
  { label: "Zoom", icon: "zoom" },
  { label: "Tag", icon: "tag" },
  { label: "Box", icon: "box", badge: "B" },
  { label: "Path", icon: "polygon" },
  { label: "Review", icon: "review", badge: "3" }
];

const pages: PageItem[] = [
  { name: "page_0001.png", type: "title", state: "done" },
  { name: "page_0002.png", type: "contents", state: "done" },
  { name: "page_0003.png", type: "balance_sheet", state: "done" },
  { name: "page_0004.png", type: "income_statement", state: "done" },
  { name: "page_0005.png", type: "cash_flow", state: "done" },
  { name: "page_0006.png", type: "notes", state: "flagged" },
  { name: "page_0007.png", type: "notes", state: "active" },
  { name: "page_0008.png", type: "notes", state: "done" },
  { name: "page_0009.png", type: "audit", state: "done" }
];

const regions: RegionItem[] = [
  {
    id: "01",
    name: "Cash and cash equivalents",
    path: "Current Assets / Cash",
    value: "12,430",
    tone: "mint"
  },
  {
    id: "02",
    name: "Trade receivables",
    path: "Current Assets / Receivables",
    value: "1,980",
    tone: "sky"
  },
  {
    id: "03",
    name: "Operating fees",
    path: "Operating Activities / Net Cash",
    value: "3,400",
    tone: "amber",
    active: true
  },
  {
    id: "04",
    name: "Tax provision",
    path: "Disclosures / Tax",
    value: "640",
    tone: "rose",
    locked: true
  },
  {
    id: "05",
    name: "Audit fees",
    path: "Disclosures / Audit Fees",
    value: "1,250",
    tone: "lime",
    hidden: true
  }
];

const stageRegions: StageRegion[] = [
  {
    label: "Cash 2008",
    tone: "mint",
    style: { left: "18%", top: "23%", width: "26%", height: "11%" }
  },
  {
    label: "Cash 2007",
    tone: "sky",
    style: { left: "53%", top: "23%", width: "16%", height: "10%" }
  },
  {
    label: "Operating fees",
    tone: "amber",
    style: { left: "16%", top: "42%", width: "32%", height: "12%" },
    active: true
  },
  {
    label: "Tax provision",
    tone: "rose",
    style: { left: "55%", top: "43%", width: "15%", height: "10%" }
  },
  {
    label: "Total 2008",
    tone: "lime",
    style: { left: "19%", top: "63%", width: "23%", height: "11%" }
  },
  {
    label: "Total 2007",
    tone: "sky",
    style: { left: "53%", top: "63%", width: "22%", height: "11%" }
  }
];

const activity = [
  "Auto-detected 5 candidate facts on page_0007.png",
  "One selected fact needs a note reference",
  "Page type confidence dropped below 0.80"
];

const summary = [
  ["Company", "Maccabi Sports Association"],
  ["Report year", "2008"],
  ["Direction", "RTL"],
  ["Annotators", "2 active"],
  ["Pages done", "7 / 9"],
  ["Open flags", "3"]
] as const;

const taskNotes = [
  "Verify values in the obligations note before publishing.",
  "Keep labels mapped to the Finetree fact taxonomy.",
  "Do not edit OCR text in this visual pass."
];

function App() {
  const selectedRegion = regions.find((region) => region.active) ?? regions[0];

  return (
    <div className="ft-app-shell">
      <header className="ft-header">
        <div className="ft-header__brand">
          <div>
            <p className="ft-kicker">Finetree annotator</p>
            <h1>page_0007.png</h1>
            <p className="ft-header__subtitle">
              Maccabi Sports Association / 2008 annual report / notes / static shell
            </p>
          </div>
        </div>

        <div className="ft-header__actions">
          <HeaderAction icon="left" label="Prev" />
          <HeaderAction icon="right" label="Next" />
          <HeaderAction icon="fullscreen" label="Fullscreen" />
          <HeaderAction icon="done" label="Done" emphasis />
        </div>
      </header>

      <div className="ft-workspace">
        <aside className="ft-toolrail">
          <div className="ft-toolrail__section">
            {tools.map((tool) => (
              <button
                key={tool.label}
                type="button"
                className={`ft-toolrail__button${tool.active ? " ft-toolrail__button--active" : ""}`}
              >
                <span className="ft-toolrail__icon">
                  <Icon name={tool.icon} />
                </span>
                <span className="ft-toolrail__label">{tool.label}</span>
                {tool.badge ? <span className="ft-toolrail__badge">{tool.badge}</span> : null}
              </button>
            ))}
          </div>

          <div className="ft-toolrail__footer">
            <span>Zoom</span>
            <strong>125%</strong>
          </div>
        </aside>

        <main className="ft-stage-column">
          <section className="ft-stage-card">
            <div className="ft-stage-card__topline">
              <div>
                <strong>page_0007.png</strong>
                <span>Notes / liabilities / zoom 125%</span>
              </div>
              <div className="ft-stage-card__controls">
                <button type="button">Fit</button>
                <button type="button">100%</button>
                <button type="button">Guides</button>
              </div>
            </div>

            <div className="ft-stage-matte">
              <div className="ft-stage-page">
                <div className="ft-stage-page__header">
                  <span>Statement of notes and obligations</span>
                  <strong>2008 annual report</strong>
                </div>

                <div className="ft-stage-page__ledger">
                  <div className="ft-ledger-row" />
                  <div className="ft-ledger-row" />
                  <div className="ft-ledger-row" />
                  <div className="ft-ledger-row" />
                  <div className="ft-ledger-row" />
                  <div className="ft-ledger-row" />
                </div>

                {stageRegions.map((region) => (
                  <div
                    key={region.label}
                    className={`ft-region ft-region--${region.tone}${region.active ? " ft-region--active" : ""}`}
                    style={region.style}
                  >
                    <span>{region.label}</span>
                    {region.active ? (
                      <>
                        <i className="ft-region__handle ft-region__handle--nw" />
                        <i className="ft-region__handle ft-region__handle--ne" />
                        <i className="ft-region__handle ft-region__handle--sw" />
                        <i className="ft-region__handle ft-region__handle--se" />
                      </>
                    ) : null}
                  </div>
                ))}

                <div className="ft-stage-page__footer">
                  <span>Currency: ILS</span>
                  <span>Scale: 1</span>
                  <span>Direction: RTL</span>
                </div>
              </div>

              <div className="ft-floating-editor">
                <div className="ft-floating-editor__tag">box</div>
                <div className="ft-floating-editor__head">
                  <strong>{selectedRegion.name}</strong>
                  <button type="button" className="ft-icon-button" aria-label="Delete region">
                    <Icon name="trash" />
                  </button>
                </div>
                <div className="ft-floating-editor__field">
                  <span>Fact path</span>
                  <div>{selectedRegion.path}</div>
                </div>
                <div className="ft-floating-editor__field">
                  <span>Value</span>
                  <div>{selectedRegion.value}</div>
                </div>
                <div className="ft-floating-editor__footer">
                  <button type="button">Attach note</button>
                  <button type="button" className="ft-floating-editor__confirm">
                    Apply
                  </button>
                </div>
              </div>
            </div>

            <div className="ft-stage-card__footer">
              <div className="ft-mini-metrics">
                <div>
                  <span>Document</span>
                  <strong>Maccabi Sports Association</strong>
                </div>
                <div>
                  <span>Queue</span>
                  <strong>12 pending docs</strong>
                </div>
                <div>
                  <span>Reviewer</span>
                  <strong>finance-team-a</strong>
                </div>
              </div>

              <div className="ft-mini-pages">
                {pages.slice(4).map((page) => (
                  <button
                    key={page.name}
                    type="button"
                    className={`ft-mini-page ft-mini-page--${page.state}`}
                  >
                    <span className="ft-mini-page__thumb" />
                    <small>{page.name.replace("page_", "").replace(".png", "")}</small>
                  </button>
                ))}
              </div>
            </div>
          </section>
        </main>

        <aside className="ft-sidebar">
          <SidebarBox title="Task Description">
            <ul className="ft-note-list">
              {taskNotes.map((note) => (
                <li key={note}>{note}</li>
              ))}
            </ul>
          </SidebarBox>

          <SidebarBox title={`Pages (${pages.length})`}>
            <div className="ft-page-list">
              {pages.map((page) => (
                <button
                  key={page.name}
                  type="button"
                  className={`ft-page-row ft-page-row--${page.state}`}
                >
                  <span className="ft-page-row__thumb" />
                  <span className="ft-page-row__meta">
                    <strong>{page.name}</strong>
                    <span>{page.type}</span>
                  </span>
                </button>
              ))}
            </div>
          </SidebarBox>

          <SidebarBox title="Regions">
            <div className="ft-region-table">
              <div className="ft-region-table__head">
                <span>#</span>
                <span>Fact</span>
                <span>Value</span>
                <span>State</span>
              </div>
              {regions.map((region, index) => (
                <button
                  key={region.id}
                  type="button"
                  className={`ft-region-row${region.active ? " ft-region-row--active" : ""}`}
                >
                  <span className="ft-region-row__index">{index + 1}</span>
                  <span className="ft-region-row__fact">
                    <i className={`ft-tone-dot ft-tone-dot--${region.tone}`} />
                    <span>
                      <strong>{region.name}</strong>
                      <small>{region.path}</small>
                    </span>
                  </span>
                  <code>{region.value}</code>
                  <span className="ft-region-row__state">
                    <span className="ft-icon-swatch" aria-hidden="true">
                      <Icon name={region.locked ? "lock" : "eye"} />
                    </span>
                    <span className="ft-icon-swatch ft-icon-swatch--muted" aria-hidden="true">
                      <Icon name={region.hidden ? "eye" : "trash"} />
                    </span>
                  </span>
                </button>
              ))}
            </div>
          </SidebarBox>

          <SidebarBox title="Validation">
            <div className="ft-activity-list">
              {activity.map((item) => (
                <div key={item} className="ft-activity-item">
                  {item}
                </div>
              ))}
            </div>
          </SidebarBox>

          <SidebarBox title="Session Summary">
            <div className="ft-summary-grid">
              {summary.map(([label, value]) => (
                <div key={label}>
                  <span>{label}</span>
                  <strong>{value}</strong>
                </div>
              ))}
            </div>
          </SidebarBox>
        </aside>
      </div>
    </div>
  );
}

function HeaderAction({ icon, label, emphasis }: { icon: IconName; label: string; emphasis?: boolean }) {
  return (
    <button type="button" className={`ft-header-action${emphasis ? " ft-header-action--emphasis" : ""}`}>
      <Icon name={icon} />
      <span>{label}</span>
    </button>
  );
}

function SidebarBox({
  title,
  subtitle,
  badge,
  children
}: {
  title: string;
  subtitle?: string;
  badge?: ReactNode;
  children: ReactNode;
}) {
  return (
    <section className="ft-sidebar-box">
      <div className="ft-sidebar-box__head">
        <div>
          {subtitle ? <p className="ft-kicker">{subtitle}</p> : null}
          <h3>{title}</h3>
        </div>
        <div className="ft-sidebar-box__actions">
          {badge}
          <button type="button" className="ft-icon-button" aria-label={`Toggle ${title}`}>
            <Icon name="chevron" />
          </button>
        </div>
      </div>
      <div className="ft-sidebar-box__body">{children}</div>
    </section>
  );
}

function Icon({ name }: { name: IconName }) {
  switch (name) {
    case "cursor":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 3l11 10h-5l2 8-3 1-2-8-3 3V3z" />
        </svg>
      );
    case "hand":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M8 11V5a1 1 0 112 0v5h1V4a1 1 0 112 0v6h1V5a1 1 0 112 0v7h1V8a1 1 0 112 0v6c0 3.3-2.7 6-6 6h-2.3c-1.4 0-2.7-.6-3.6-1.6L6 14.4A1.4 1.4 0 017 12h1z" />
        </svg>
      );
    case "zoom":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M10.5 4a6.5 6.5 0 014.9 10.8l4.4 4.4-1.4 1.4-4.4-4.4A6.5 6.5 0 1110.5 4zm0 2a4.5 4.5 0 100 9 4.5 4.5 0 000-9z" />
        </svg>
      );
    case "tag":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 5.5A1.5 1.5 0 015.5 4H11l9 9-7 7-9-9V5.5zm3 1.5a1.5 1.5 0 100 3 1.5 1.5 0 000-3z" />
        </svg>
      );
    case "box":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M5 5h14v14H5zM3 3v4h2V5h2V3H3zm14 0v2h2v2h2V3h-4zM3 17v4h4v-2H5v-2H3zm16 0v2h-2v2h4v-4h-2z" />
        </svg>
      );
    case "polygon":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M6 5l11 3 2 8-8 4-7-6 2-9zm1.5 1.8L6.1 13l5 4.3 6.1-3.1-1.4-5.5-8.3-1.9z" />
        </svg>
      );
    case "review":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 5h16v10H7l-3 3V5zm3 3v2h10V8H7zm0 4v1h7v-1H7z" />
        </svg>
      );
    case "left":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M14.7 6.3L9 12l5.7 5.7-1.4 1.4L6.2 12l7.1-7.1 1.4 1.4z" />
        </svg>
      );
    case "right":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M9.3 17.7L15 12 9.3 6.3l1.4-1.4 7.1 7.1-7.1 7.1-1.4-1.4z" />
        </svg>
      );
    case "fullscreen":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M4 9V4h5v2H6v3H4zm10-5h6v6h-2V6h-4V4zM4 15h2v3h3v2H4v-5zm14 3v-3h2v5h-5v-2h3z" />
        </svg>
      );
    case "done":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M9.6 16.2L5.4 12l-1.4 1.4 5.6 5.6L20 8.6l-1.4-1.4-9 9z" />
        </svg>
      );
    case "chevron":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M7.4 9.4L12 14l4.6-4.6 1.4 1.4-6 6-6-6 1.4-1.4z" />
        </svg>
      );
    case "lock":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M7 10V8a5 5 0 1110 0v2h1a1 1 0 011 1v8a1 1 0 01-1 1H6a1 1 0 01-1-1v-8a1 1 0 011-1h1zm2 0h6V8a3 3 0 10-6 0v2z" />
        </svg>
      );
    case "eye":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 5c5.2 0 9.2 4.4 10.5 7-1.3 2.6-5.3 7-10.5 7S2.8 14.6 1.5 12C2.8 9.4 6.8 5 12 5zm0 2c-3.7 0-6.8 3-6.8 5s3.1 5 6.8 5 6.8-3 6.8-5-3.1-5-6.8-5zm0 2.2a2.8 2.8 0 110 5.6 2.8 2.8 0 010-5.6z" />
        </svg>
      );
    case "trash":
      return (
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M9 3h6l1 2h4v2H4V5h4l1-2zm1 6h2v8h-2V9zm4 0h2v8h-2V9zM7 9h2v8H7V9zm-1 11V8h12v12H6z" />
        </svg>
      );
    default:
      return null;
  }
}

export default App;
