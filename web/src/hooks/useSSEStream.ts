/** SSE streaming hook for AI inference endpoints. */

import { useState, useCallback, useRef } from "react";
import { postSSE, parseSSEStream } from "../api/client";

export interface SSEStreamState {
  isStreaming: boolean;
  chunks: string[];
  fullText: string;
  error: string | null;
  cancelled: boolean;
}

export function useSSEStream() {
  const [state, setState] = useState<SSEStreamState>({
    isStreaming: false,
    chunks: [],
    fullText: "",
    error: null,
    cancelled: false,
  });
  const abortRef = useRef<AbortController | null>(null);

  const start = useCallback(
    async (
      path: string,
      body: unknown,
      onEvent?: (event: { event: string | null; data: unknown }) => void,
    ) => {
      // Cancel any existing stream.
      abortRef.current?.abort();
      const abort = new AbortController();
      abortRef.current = abort;

      setState({
        isStreaming: true,
        chunks: [],
        fullText: "",
        error: null,
        cancelled: false,
      });

      try {
        const response = await postSSE(path, body, abort.signal);
        let accumulated = "";
        const allChunks: string[] = [];

        for await (const { event, data } of parseSSEStream(response)) {
          if (abort.signal.aborted) break;

          let parsed: unknown;
          try {
            parsed = JSON.parse(data);
          } catch {
            parsed = data;
          }

          // Notify caller.
          onEvent?.({ event, data: parsed });

          // Handle standard event types.
          const obj = parsed as Record<string, unknown>;
          if (obj.type === "chunk" && typeof obj.text === "string") {
            accumulated += obj.text;
            allChunks.push(obj.text);
            setState((s) => ({
              ...s,
              chunks: [...allChunks],
              fullText: accumulated,
            }));
          } else if (obj.type === "error") {
            setState((s) => ({
              ...s,
              error: String(obj.message ?? "Unknown error"),
              isStreaming: false,
            }));
            return;
          } else if (obj.type === "cancelled") {
            setState((s) => ({ ...s, cancelled: true, isStreaming: false }));
            return;
          } else if (obj.type === "done") {
            setState((s) => ({ ...s, isStreaming: false }));
            return;
          }
        }

        setState((s) => ({ ...s, isStreaming: false }));
      } catch (err) {
        if ((err as Error).name === "AbortError") {
          setState((s) => ({ ...s, cancelled: true, isStreaming: false }));
        } else {
          setState((s) => ({
            ...s,
            error: String(err),
            isStreaming: false,
          }));
        }
      }
    },
    [],
  );

  const cancel = useCallback(() => {
    abortRef.current?.abort();
    setState((s) => ({ ...s, cancelled: true, isStreaming: false }));
  }, []);

  return { ...state, start, cancel };
}
