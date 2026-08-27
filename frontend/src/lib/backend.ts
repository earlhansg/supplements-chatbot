/**
 * Server-side client for the FastAPI backend.
 *
 * The Python app registers no CORS middleware, so the browser cannot call
 * `http://127.0.0.1:8000` directly. Everything goes through this module from a
 * Route Handler instead — same-origin from the browser's point of view, and the
 * backend URL never reaches the client bundle.
 */

import { serverConfig } from "@/lib/server-config";
import type { ChatResponse } from "@/lib/types";

export class BackendError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "BackendError";
    this.status = status;
  }
}

/** FastAPI validation errors arrive as `{detail: [{msg, loc, ...}]}` or `{detail: "..."}`. */
async function describeFailure(response: Response): Promise<string> {
  const fallback = `Backend responded ${response.status} ${response.statusText}`.trim();
  try {
    const body = await response.json();
    const detail = (body as { detail?: unknown }).detail;

    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
      const messages = detail
        .map((item) => (item as { msg?: unknown }).msg)
        .filter((msg): msg is string => typeof msg === "string");
      if (messages.length > 0) return messages.join("; ");
    }
    return fallback;
  } catch {
    return fallback;
  }
}

/** POST `/chat` on the FastAPI app (`app/main.py:39`). */
export async function askBackend(question: string): Promise<ChatResponse> {
  let response: Response;

  try {
    response = await fetch(`${serverConfig.backendUrl}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
      cache: "no-store",
      signal: AbortSignal.timeout(serverConfig.chatTimeoutMs),
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "TimeoutError") {
      throw new BackendError(
        `The backend did not respond within ${Math.round(serverConfig.chatTimeoutMs / 1000)}s. ` +
          "A cache miss has to call the local LLM, which can be slow on a cold start.",
        504,
      );
    }
    throw new BackendError(
      `Cannot reach the chatbot backend at ${serverConfig.backendUrl}. ` +
        "Is `uvicorn app.main:app` running?",
      502,
    );
  }

  if (!response.ok) {
    throw new BackendError(await describeFailure(response), response.status);
  }

  return (await response.json()) as ChatResponse;
}

/** GET `/health` on the FastAPI app (`app/main.py:34`). */
export async function pingBackend(): Promise<boolean> {
  try {
    const response = await fetch(`${serverConfig.backendUrl}/health`, {
      cache: "no-store",
      signal: AbortSignal.timeout(3_000),
    });
    return response.ok;
  } catch {
    return false;
  }
}
