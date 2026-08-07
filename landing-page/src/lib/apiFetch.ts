import {
  authHeaders,
  clearToken,
  clearRefreshToken,
  getRefreshToken,
  setRefreshToken,
  setToken,
} from "../hooks/useAuth";
import { toast } from "./toast";

// replaces raw fetch() for all /api/* calls
let onUnauthenticated: (() => void) | null = null;

export function registerUnauthenticatedHandler(fn: () => void) {
  onUnauthenticated = fn;
}

export async function apiFetch(
  input: RequestInfo,
  init?: RequestInit,
): Promise<Response> {
  const resp = await fetch(input, {
    ...init,
    headers: { ...authHeaders(), ...(init?.headers ?? {}) },
  });

  if (resp.status !== 401) return resp;
  const rt = getRefreshToken();
  if (!rt) {
    clearToken();
    clearRefreshToken();
    toast.info("Your session has expired. Please log in again.");
    onUnauthenticated?.();
    return resp;
  }

  // attempt a silent refresh
  const refresh = await fetch("/api/auth/refresh", {
    method: "POST",
    headers: { "X-Refresh-Token": rt },
  });
  if (!refresh.ok) {
    clearToken();
    clearRefreshToken();
    toast.info("Your session has expired. Please log in again.");
    onUnauthenticated?.();
    return resp;
  }
  const { access_token, refresh_token } = await refresh.json();
  setToken(access_token);
  setRefreshToken(refresh_token);
  return fetch(input, {
    ...init,
    headers: {
      Authorization: `Bearer ${access_token}`,
      ...(init?.headers ?? {}),
    },
  });
}

// convenience wrapper for callers that just want "success or throw" —
// use for uploads/deletes instead of hand-rolling !r.ok checks per call site
export async function apiFetchOrThrow(
  input: RequestInfo,
  init?: RequestInit,
): Promise<Response> {
  const r = await apiFetch(input, init);
  if (!r.ok) {
    const d = await r.json().catch(() => ({}));
    throw new Error(
      (d as { detail?: string }).detail || `request failed (${r.status})`,
    );
  }
  return r;
}

// kicks off a job (POST returning {job_id}) then connects to its SSE
// stream — the returned Response is exactly what ProcessingPage.tsx's
// streamRequest contract expects, so no change is needed there.
export async function apiFetchJobStream(
  kickoffUrl: string,
  kickoffInit: RequestInit,
  signal: AbortSignal,
  onJobId?: (jobId: string) => void,
): Promise<Response> {
  const kickoff = await apiFetch(kickoffUrl, { ...kickoffInit, signal });
  if (!kickoff.ok) return kickoff;
  const { job_id } = await kickoff.json();
  onJobId?.(job_id);
  return apiFetch(`/api/jobs/${job_id}/stream`, { signal });
}
