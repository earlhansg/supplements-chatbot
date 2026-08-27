/**
 * POST /api/chat — thin proxy to the FastAPI `/chat` endpoint.
 *
 * Exists because the backend ships no CORS middleware. The request/response
 * bodies pass through unchanged, so the client sees the real backend contract
 * (`is_cached`, `cache_similarity`, `sources`) rather than a reshaped one.
 */

import { askBackend, BackendError } from "@/lib/backend";
import type { ApiError } from "@/lib/types";

// A cache miss runs KB retrieval plus a local LLM call; give it room.
export const maxDuration = 300;

export async function POST(request: Request): Promise<Response> {
  let payload: unknown;

  try {
    payload = await request.json();
  } catch {
    return errorResponse("Request body must be valid JSON.", 400);
  }

  const question =
    typeof (payload as { question?: unknown })?.question === "string"
      ? (payload as { question: string }).question.trim()
      : "";

  // Mirrors the backend's own `min_length=1` constraint so an empty submission
  // fails here instead of coming back as a 422.
  if (!question) {
    return errorResponse("`question` is required and must be a non-empty string.", 400);
  }

  try {
    const answer = await askBackend(question);
    return Response.json(answer, { headers: { "Cache-Control": "no-store" } });
  } catch (error) {
    if (error instanceof BackendError) {
      // 4xx from the backend is the caller's fault; anything else is an upstream outage.
      const status = error.status >= 400 && error.status < 500 ? error.status : 502;
      return errorResponse(error.message, status);
    }
    console.error("[api/chat]", error);
    return errorResponse("Unexpected error while contacting the backend.", 500);
  }
}

function errorResponse(message: string, status: number): Response {
  const body: ApiError = { error: message };
  return Response.json(body, { status, headers: { "Cache-Control": "no-store" } });
}
