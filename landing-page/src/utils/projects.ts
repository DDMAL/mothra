import type { Dispatch, SetStateAction } from "react";
import type { Project } from "../types";
import { apiFetch } from "../lib/apiFetch";

export function normalizeProjects(raw: Project[]): Project[] {
  return raw.map((p) => ({
    ...p,
    images: p.images.map((img) => ({
      ...img,
      src: img.src ?? `/api/images/${img.id}`,
    })),
  }));
}

/**
 * GETs the user's whole project list into local state.
 *
 * Leaves state untouched on any non-OK response: an error body isn't a
 * project array, and feeding it to normalizeProjects throws. "unauthenticated"
 * means a 401 apiFetch couldn't fix by refreshing, so it has already logged
 * the user out and toasted -- callers only need to report "failed". Network
 * failures reject, as with any fetch.
 */
export async function loadProjects(
  setProjects: Dispatch<SetStateAction<Project[]>>,
): Promise<"ok" | "unauthenticated" | "failed"> {
  const r = await apiFetch("/api/projects");
  if (r.status === 401) return "unauthenticated";
  if (!r.ok) return "failed";
  setProjects(normalizeProjects(await r.json()));
  return "ok";
}

/**
 * Re-GETs one project and merges the server's copy into local state.
 *
 * Several fields (icXmlFiles, stafflines, text alignments) are written
 * server-side by a Celery job with no corresponding client-side mutation --
 * unlike meiFiles, which the encode flow adds to local state itself as each
 * item's "result" event arrives (see useEncodingFlow.ts). Those fields only
 * ever reach the browser through a full refetch like this one.
 *
 * useActiveJobWatcher (App.tsx) already did this, but only for a job whose
 * completion it detects itself via polling -- a job actively being watched
 * by a mounted ProcessingPage instead settles through its own SSE "done"
 * event, which calls markJobSettled() directly and removes the job from the
 * watcher's store before its next poll tick, so the watcher's refetch never
 * fires for it. ProcessingPage's completion handlers call this directly so
 * every encode job gets the same refresh regardless of which path it
 * finishes through.
 */
export async function refreshProject(
  projectId: number,
  setProjects: Dispatch<SetStateAction<Project[]>>,
): Promise<Project | null> {
  const r = await apiFetch(`/api/projects/${projectId}`);
  if (!r.ok) return null;
  const fresh: Project = await r.json();
  const [normalized] = normalizeProjects([fresh]);
  setProjects((prev) =>
    prev.map((p) => (p.id === normalized.id ? normalized : p)),
  );
  return normalized;
}
