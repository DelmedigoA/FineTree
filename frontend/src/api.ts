import type {
  ApiAppConfig,
  ApiDocumentDetail,
  ApiDocumentSaveRequest,
  ApiDocumentSaveResponse,
  ApiDocumentValidateResponse,
  ApiExtractionResponse,
  ApiWorkspaceDocumentSummary
} from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: {
      "Content-Type": "application/json",
      ...(init?.headers ?? {})
    },
    ...init
  });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      message = String(payload.detail ?? payload.message ?? message);
    } catch {
      // ignore
    }
    throw new Error(message);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

export function getAppConfig(): Promise<ApiAppConfig> {
  return request<ApiAppConfig>("/api/app-config");
}

export function getDocuments(): Promise<ApiWorkspaceDocumentSummary[]> {
  return request<ApiWorkspaceDocumentSummary[]>("/api/workspace/documents");
}

export function getDocument(docId: string): Promise<ApiDocumentDetail> {
  return request<ApiDocumentDetail>(`/api/documents/${docId}`);
}

export function saveDocument(docId: string, body: ApiDocumentSaveRequest): Promise<ApiDocumentSaveResponse> {
  return request<ApiDocumentSaveResponse>(`/api/documents/${docId}`, {
    method: "PUT",
    body: JSON.stringify(body)
  });
}

export function validateDocument(docId: string, body: ApiDocumentSaveRequest): Promise<ApiDocumentValidateResponse> {
  return request<ApiDocumentValidateResponse>(`/api/documents/${docId}/validate`, {
    method: "POST",
    body: JSON.stringify(body)
  });
}

export function prepareDocument(docId: string): Promise<ApiDocumentDetail> {
  return request<ApiDocumentDetail>(`/api/workspace/documents/${docId}/prepare`, {
    method: "POST"
  });
}

export function toggleChecked(docId: string, value: boolean): Promise<ApiWorkspaceDocumentSummary> {
  return request<ApiWorkspaceDocumentSummary>(`/api/workspace/documents/${docId}/checked`, {
    method: "POST",
    body: JSON.stringify({ value })
  });
}

export function toggleReviewed(docId: string, value: boolean): Promise<ApiWorkspaceDocumentSummary> {
  return request<ApiWorkspaceDocumentSummary>(`/api/workspace/documents/${docId}/reviewed`, {
    method: "POST",
    body: JSON.stringify({ value })
  });
}

export function uploadPdf(filename: string, contentB64: string, dpi = 200): Promise<ApiDocumentDetail> {
  return request<ApiDocumentDetail>("/api/workspace/import-pdf", {
    method: "POST",
    body: JSON.stringify({ filename, content_b64: contentB64, dpi })
  });
}

export function extractPage(
  docId: string,
  pageImage: string,
  body: {
    provider: "gemini" | "qwen";
    prompt: string | null;
    model: string | null;
    few_shot_enabled: boolean;
    few_shot_preset: string | null;
    enable_thinking: boolean | null;
  }
): Promise<ApiExtractionResponse> {
  return request<ApiExtractionResponse>(`/api/documents/${docId}/pages/${pageImage}/extract`, {
    method: "POST",
    body: JSON.stringify(body)
  });
}
