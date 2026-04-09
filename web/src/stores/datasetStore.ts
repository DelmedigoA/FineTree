/** Dataset state — dataset versions list/detail, draft editing, preview, token stats, and push flow. */

import { create } from "zustand";
import { get as apiGet, post, put, del, postSSE, parseSSEStream } from "../api/client";

export type SplitName = "train" | "test" | "val" | "exclude";
export type SplitMode = "random" | "manual";
export type PushMode = "single" | "separate";
export type CompactMode = "raw" | "compact" | "aggressive";
export type PushStatus = "never" | "pushed";

export interface ExportConfig {
  hf_repo: string;
  push_mode: PushMode;
  image_scaling: boolean;
  max_pixels: number;
  min_pixels: number | null;
  bbox_grid_norm: boolean;
  values_norm: boolean;
  compact_mode: CompactMode;
  drop_date: boolean;
  selected_fact_keys: string[];
  selected_page_meta_keys: string[];
  include_bbox: boolean;
  approved_pages_only: boolean;
}

export interface SplitStatsEntry {
  doc_count: number;
  page_count: number;
}

export interface DatasetVersion {
  version_id: string;
  name: string;
  created_at: number;
  updated_at: number | null;
  split_assignments: Record<string, SplitName>;
  export_config: ExportConfig;
  split_stats: Partial<Record<SplitName, SplitStatsEntry>>;
  push_status: PushStatus;
  last_pushed_at: number | null;
  pushed_repos: Record<string, string>;
}

export interface PreviewRow {
  doc_id: string;
  page: string;
  split: string;
  instruction: string;
  text_preview: string;
  image_url: string;
}

export interface TokenStatsResult {
  tokenizer: string;
  splits: Record<
    string,
    {
      sample_count: number;
      total_text_tokens: number;
      per_page_text: { min: number; max: number; mean: number; median: number; count: number };
      per_page_image: { min: number; max: number; mean: number; median: number; count: number };
      per_page_full: { min: number; max: number; mean: number; median: number; count: number };
    }
  >;
}

interface DatasetDraftState {
  datasetName: string;
  splitMode: SplitMode;
  trainPct: number;
  testPct: number;
  valPct: number;
  seed: number;
  assignments: Record<string, SplitName>;
  hfRepo: string;
  pushMode: PushMode;
  imageScaling: boolean;
  maxPixels: number;
  minPixels: number | null;
  bboxGridNorm: boolean;
  valuesNorm: boolean;
  compactMode: CompactMode;
  dropDate: boolean;
  selectedFactKeys: string[];
  selectedPageMetaKeys: string[];
  includeBbox: boolean;
  approvedPagesOnly: boolean;
}

export interface DatasetStoreState extends DatasetDraftState {
  createDialogOpen: boolean;
  activeTab: "splits" | "export" | "preview";

  versions: DatasetVersion[];
  versionsLoading: boolean;
  versionsError: string | null;

  activeVersionId: string | null;
  versionLoading: boolean;
  versionError: string | null;

  schemaFields: {
    prompt_fact_keys: string[];
    prompt_page_meta_keys: string[];
    required_prompt_canonical_keys: string[];
    instruction_preview?: string;
  } | null;

  previewSplit: "train" | "test" | "val";
  previewRows: PreviewRow[];
  previewLoading: boolean;
  previewError: string | null;

  tokenizer: string;
  tokenStats: TokenStatsResult | null;
  tokenStatsLoading: boolean;
  tokenStatsError: string | null;
  tokenStatsLog: string[];

  isPushing: boolean;
  pushLog: string[];
  pushResult: Record<string, string> | null;
  pushError: string | null;
}

export interface DatasetStoreActions {
  openCreateDialog(): void;
  closeCreateDialog(): void;
  setActiveTab(tab: DatasetStoreState["activeTab"]): void;

  loadVersions(): Promise<void>;
  loadVersion(id: string): Promise<void>;
  createVersion(name: string): Promise<DatasetVersion>;
  saveActiveVersion(): Promise<DatasetVersion | null>;
  deleteVersion(id: string): Promise<void>;
  setActiveVersion(id: string | null): void;
  setDatasetName(value: string): void;
  resetDraft(): void;

  loadSchemaFields(): Promise<void>;

  setSplitMode(mode: SplitMode): void;
  setTrainPct(v: number): void;
  setTestPct(v: number): void;
  setValPct(v: number): void;
  setSeed(v: number): void;
  applyRandomSplit(docs: string[]): void;
  setAssignment(docId: string, split: SplitName): void;

  setHfRepo(v: string): void;
  setPushMode(v: PushMode): void;
  setImageScaling(v: boolean): void;
  setMaxPixels(v: number): void;
  setMinPixels(v: number | null): void;
  setBboxGridNorm(v: boolean): void;
  setValuesNorm(v: boolean): void;
  setCompactMode(v: CompactMode): void;
  setDropDate(v: boolean): void;
  toggleFactKey(key: string): void;
  togglePageMetaKey(key: string): void;
  setIncludeBbox(v: boolean): void;
  setApprovedPagesOnly(v: boolean): void;

  setPreviewSplit(split: "train" | "test" | "val"): void;
  loadPreview(): Promise<void>;

  computeTokenStats(): Promise<void>;
  pushToHF(): Promise<void>;
}

const defaultDraftState = (): DatasetDraftState => ({
  datasetName: "",
  splitMode: "random",
  trainPct: 70,
  testPct: 15,
  valPct: 15,
  seed: 42,
  assignments: {},
  hfRepo: "",
  pushMode: "single",
  imageScaling: true,
  maxPixels: 1400000,
  minPixels: null,
  bboxGridNorm: true,
  valuesNorm: true,
  compactMode: "raw",
  dropDate: false,
  selectedFactKeys: [],
  selectedPageMetaKeys: [],
  includeBbox: true,
  approvedPagesOnly: true,
});

function buildExportConfig(state: DatasetDraftState): ExportConfig {
  return {
    hf_repo: state.hfRepo,
    push_mode: state.pushMode,
    image_scaling: state.imageScaling,
    max_pixels: state.maxPixels,
    min_pixels: state.minPixels,
    bbox_grid_norm: state.bboxGridNorm,
    values_norm: state.valuesNorm,
    compact_mode: state.compactMode,
    drop_date: state.dropDate,
    selected_fact_keys: state.selectedFactKeys,
    selected_page_meta_keys: state.selectedPageMetaKeys,
    include_bbox: state.includeBbox,
    approved_pages_only: state.approvedPagesOnly,
  };
}

function applyVersionToDraft(version: DatasetVersion): Partial<DatasetStoreState> {
  const c = version.export_config;
  return {
    activeVersionId: version.version_id,
    datasetName: version.name,
    splitMode: "manual",
    assignments: version.split_assignments,
    hfRepo: c.hf_repo,
    pushMode: c.push_mode,
    imageScaling: c.image_scaling,
    maxPixels: c.max_pixels,
    minPixels: c.min_pixels,
    bboxGridNorm: c.bbox_grid_norm,
    valuesNorm: c.values_norm,
    compactMode: c.compact_mode,
    dropDate: c.drop_date,
    selectedFactKeys: c.selected_fact_keys,
    selectedPageMetaKeys: c.selected_page_meta_keys,
    includeBbox: c.include_bbox,
    approvedPagesOnly: c.approved_pages_only,
    pushResult: Object.keys(version.pushed_repos ?? {}).length > 0 ? version.pushed_repos : null,
  };
}

function upsertVersion(versions: DatasetVersion[], incoming: DatasetVersion): DatasetVersion[] {
  const withoutCurrent = versions.filter((version) => version.version_id !== incoming.version_id);
  const next = [...withoutCurrent, incoming];
  next.sort((a, b) => (b.created_at ?? 0) - (a.created_at ?? 0));
  return next;
}

function mulberry32(seed: number) {
  return function () {
    let t = (seed += 0x6d2b79f5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function seededShuffle<T>(arr: T[], seed: number): T[] {
  const result = [...arr];
  const rng = mulberry32(seed);
  for (let i = result.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [result[i], result[j]] = [result[j]!, result[i]!];
  }
  return result;
}

export const useDatasetStore = create<DatasetStoreState & DatasetStoreActions>((set, get) => ({
  ...defaultDraftState(),

  createDialogOpen: false,
  activeTab: "splits",

  versions: [],
  versionsLoading: false,
  versionsError: null,

  activeVersionId: null,
  versionLoading: false,
  versionError: null,

  schemaFields: null,

  previewSplit: "train",
  previewRows: [],
  previewLoading: false,
  previewError: null,

  tokenizer: "Qwen/Qwen3.5-27B",
  tokenStats: null,
  tokenStatsLoading: false,
  tokenStatsError: null,
  tokenStatsLog: [],

  isPushing: false,
  pushLog: [],
  pushResult: null,
  pushError: null,

  openCreateDialog() {
    const schemaFields = get().schemaFields;
    const draft = defaultDraftState();
    if (schemaFields) {
      draft.selectedFactKeys = schemaFields.prompt_fact_keys;
      draft.selectedPageMetaKeys = schemaFields.prompt_page_meta_keys;
    }
    set({
      ...draft,
      activeVersionId: null,
      activeTab: "splits",
      createDialogOpen: true,
      versionError: null,
      previewRows: [],
      previewError: null,
      tokenStats: null,
      tokenStatsError: null,
      tokenStatsLog: [],
      pushLog: [],
      pushResult: null,
      pushError: null,
    });
  },

  closeCreateDialog() {
    set({ createDialogOpen: false });
  },

  setActiveTab(tab) {
    set({ activeTab: tab });
  },

  async loadVersions() {
    set({ versionsLoading: true, versionsError: null });
    try {
      const versions = await apiGet<DatasetVersion[]>("/dataset/versions");
      set({ versions, versionsLoading: false });
    } catch (err) {
      set({ versionsError: String(err), versionsLoading: false });
    }
  },

  async loadVersion(id) {
    set({
      versionLoading: true,
      versionError: null,
      previewRows: [],
      previewError: null,
      tokenStats: null,
      tokenStatsError: null,
      tokenStatsLog: [],
      pushLog: [],
      pushError: null,
    });
    try {
      const version = await apiGet<DatasetVersion>(`/dataset/versions/${id}`);
      set((state) => ({
        ...applyVersionToDraft(version),
        versions: upsertVersion(state.versions, version),
        versionLoading: false,
      }));
    } catch (err) {
      set({ versionError: String(err), versionLoading: false });
    }
  },

  async createVersion(name) {
    const state = get();
    const saved = await post<DatasetVersion>("/dataset/versions", {
      name,
      split_assignments: state.assignments,
      export_config: buildExportConfig(state),
    });
    set((prev) => ({
      ...applyVersionToDraft(saved),
      versions: upsertVersion(prev.versions, saved),
      createDialogOpen: false,
      pushLog: [],
      pushError: null,
      previewRows: [],
      previewError: null,
      tokenStats: null,
      tokenStatsError: null,
      tokenStatsLog: [],
    }));
    return saved;
  },

  async saveActiveVersion() {
    const state = get();
    if (!state.activeVersionId) return null;
    const saved = await put<DatasetVersion>(`/dataset/versions/${state.activeVersionId}`, {
      name: state.datasetName,
      split_assignments: state.assignments,
      export_config: buildExportConfig(state),
    });
    set((prev) => ({
      ...applyVersionToDraft(saved),
      versions: upsertVersion(prev.versions, saved),
    }));
    return saved;
  },

  async deleteVersion(id) {
    await del(`/dataset/versions/${id}`);
    set((state) => ({
      versions: state.versions.filter((version) => version.version_id !== id),
      activeVersionId: state.activeVersionId === id ? null : state.activeVersionId,
    }));
  },

  setActiveVersion(id) {
    set({ activeVersionId: id });
  },

  setDatasetName(value) {
    set({ datasetName: value });
  },

  resetDraft() {
    const schemaFields = get().schemaFields;
    const draft = defaultDraftState();
    if (schemaFields) {
      draft.selectedFactKeys = schemaFields.prompt_fact_keys;
      draft.selectedPageMetaKeys = schemaFields.prompt_page_meta_keys;
    }
    set(draft);
  },

  async loadSchemaFields() {
    if (get().schemaFields) return;
    try {
      const fields = await apiGet<{
        prompt_fact_keys: string[];
        prompt_page_meta_keys: string[];
        required_prompt_canonical_keys: string[];
        instruction_preview?: string;
      }>("/dataset/schema-fields");
      const { selectedFactKeys, selectedPageMetaKeys } = get();
      set({
        schemaFields: fields,
        selectedFactKeys: selectedFactKeys.length === 0 ? fields.prompt_fact_keys : selectedFactKeys,
        selectedPageMetaKeys: selectedPageMetaKeys.length === 0 ? fields.prompt_page_meta_keys : selectedPageMetaKeys,
      });
    } catch {
      /* non-fatal */
    }
  },

  setSplitMode(mode) {
    set({ splitMode: mode });
  },

  setTrainPct(v) {
    set({ trainPct: v });
  },

  setTestPct(v) {
    set({ testPct: v });
  },

  setValPct(v) {
    set({ valPct: v });
  },

  setSeed(v) {
    set({ seed: v });
  },

  applyRandomSplit(docs) {
    const { trainPct, testPct, seed } = get();
    const shuffled = seededShuffle(docs, seed);
    const n = shuffled.length;
    const trainCount = Math.round((trainPct / 100) * n);
    const testCount = Math.round((testPct / 100) * n);
    const assignments: Record<string, SplitName> = {};
    shuffled.forEach((docId, index) => {
      if (index < trainCount) assignments[docId] = "train";
      else if (index < trainCount + testCount) assignments[docId] = "test";
      else assignments[docId] = "val";
    });
    set({ assignments });
  },

  setAssignment(docId, split) {
    set((state) => ({ assignments: { ...state.assignments, [docId]: split } }));
  },

  setHfRepo(v) {
    set({ hfRepo: v });
  },

  setPushMode(v) {
    set({ pushMode: v });
  },

  setImageScaling(v) {
    set({ imageScaling: v });
  },

  setMaxPixels(v) {
    set({ maxPixels: v });
  },

  setMinPixels(v) {
    set({ minPixels: v });
  },

  setBboxGridNorm(v) {
    set({ bboxGridNorm: v });
  },

  setValuesNorm(v) {
    set({ valuesNorm: v });
  },

  setCompactMode(v) {
    set({ compactMode: v });
  },

  setDropDate(v) {
    set({ dropDate: v });
  },

  toggleFactKey(key) {
    set((state) => ({
      selectedFactKeys: state.selectedFactKeys.includes(key)
        ? state.selectedFactKeys.filter((current) => current !== key)
        : [...state.selectedFactKeys, key],
    }));
  },

  togglePageMetaKey(key) {
    set((state) => ({
      selectedPageMetaKeys: state.selectedPageMetaKeys.includes(key)
        ? state.selectedPageMetaKeys.filter((current) => current !== key)
        : [...state.selectedPageMetaKeys, key],
    }));
  },

  setIncludeBbox(v) {
    set({ includeBbox: v });
  },

  setApprovedPagesOnly(v) {
    set({ approvedPagesOnly: v });
  },

  setPreviewSplit(split) {
    set({ previewSplit: split, previewRows: [], previewError: null });
  },

  async loadPreview() {
    const state = get();
    set({ previewLoading: true, previewError: null, previewRows: [] });
    try {
      const response = await post<{ rows: PreviewRow[] }>("/dataset/preview", {
        split: state.previewSplit,
        assignments: state.assignments,
        export_config: buildExportConfig(state),
      });
      set({ previewRows: response.rows ?? [], previewLoading: false });
    } catch (err) {
      set({ previewError: String(err), previewLoading: false });
    }
  },

  async computeTokenStats() {
    const state = get();
    set({ tokenStatsLoading: true, tokenStatsError: null, tokenStatsLog: [], tokenStats: null });
    try {
      const response = await postSSE("/dataset/token-stats", {
        tokenizer: state.tokenizer,
        assignments: state.assignments,
        export_config: buildExportConfig(state),
      });
      for await (const { data } of parseSSEStream(response)) {
        try {
          const payload = JSON.parse(data) as Record<string, unknown>;
          if (payload.type === "log" && typeof payload.message === "string") {
            set((prev) => ({ tokenStatsLog: [...prev.tokenStatsLog, payload.message as string] }));
          } else if (payload.type === "done") {
            set({ tokenStats: payload.result as TokenStatsResult, tokenStatsLoading: false });
          } else if (payload.type === "error") {
            set({ tokenStatsError: String(payload.message ?? "Unknown error"), tokenStatsLoading: false });
          }
        } catch {
          /* ignore malformed event */
        }
      }
      set((prev) => ({
        tokenStatsLoading: prev.tokenStats === null && !prev.tokenStatsError ? false : prev.tokenStatsLoading,
      }));
    } catch (err) {
      set({ tokenStatsError: String(err), tokenStatsLoading: false });
    }
  },

  async pushToHF() {
    const state = get();
    if (!state.activeVersionId) {
      set({ pushError: "Save the dataset version before pushing." });
      return;
    }

    set({ isPushing: true, pushLog: [], pushResult: null, pushError: null });
    try {
      const saved = await get().saveActiveVersion();
      const versionId = saved?.version_id ?? state.activeVersionId;
      const response = await postSSE(`/dataset/versions/${versionId}/push`, {});
      for await (const { data } of parseSSEStream(response)) {
        try {
          const payload = JSON.parse(data) as Record<string, unknown>;
          if (payload.type === "log" && typeof payload.message === "string") {
            set((prev) => ({ pushLog: [...prev.pushLog, payload.message as string] }));
          } else if (payload.type === "done") {
            set({ pushResult: payload.repos as Record<string, string>, isPushing: false });
          } else if (payload.type === "error") {
            set({ pushError: String(payload.message ?? "Push failed"), isPushing: false });
          }
        } catch {
          /* ignore malformed event */
        }
      }
      await Promise.all([get().loadVersion(versionId), get().loadVersions()]);
      set((prev) => ({
        isPushing: prev.pushResult === null && !prev.pushError ? false : prev.isPushing,
      }));
    } catch (err) {
      set({ pushError: String(err), isPushing: false });
    }
  },
}));
