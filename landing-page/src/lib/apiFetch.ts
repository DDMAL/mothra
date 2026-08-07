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

// Refresh tokens rotate on every use (see auth_api.py) -- if two 401s land
// at once, each holding the SAME (still-valid-for-now) refresh token, the
// first request to actually hit /api/auth/refresh revokes that token; a
// second, independent refresh call sent moments later with the
// now-already-revoked token fails and clears the new token pair the first
// call just wrote, logging out an otherwise-valid session. This module-level
// in-flight promise serializes rotation: whichever 401 handler gets here
// first performs the real network call, and any concurrent handler just
// awaits and reuses ITS result instead of sending its own.
let refreshInFlight: Promise<{
  access_token: string;
  refresh_token: string;
} | null> | null = null;

function refreshTokens(
  rt: string,
): Promise<{ access_token: string; refresh_token: string } | null> {
  if (!refreshInFlight) {
    refreshInFlight = fetch("/api/auth/refresh", {
      method: "POST",
      headers: { "X-Refresh-Token": rt },
    })
      .then(async (refresh) => {
        if (!refresh.ok) return null;
        const data = await refresh.json();
        setToken(data.access_token);
        setRefreshToken(data.refresh_token);
        return data;
      })
      .catch(() => null)
      .finally(() => {
        refreshInFlight = null;
      });
  }
  return refreshInFlight;
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

  // attempt a silent refresh -- shared with any other concurrent 401
  // handler currently in flight, see refreshTokens's docstring above
  const refreshed = await refreshTokens(rt);
  if (!refreshed) {
    clearToken();
    clearRefreshToken();
    toast.info("Your session has expired. Please log in again.");
    onUnauthenticated?.();
    return resp;
  }
  return fetch(input, {
    ...init,
    headers: {
      Authorization: `Bearer ${refreshed.access_token}`,
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
