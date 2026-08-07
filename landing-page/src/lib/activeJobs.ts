export interface ActiveJob {
  jobId: string;
  projectId: number | null;
  kind: string;
}

let activeJobs: ActiveJob[] = [];
const listeners = new Set<() => void>();

function emitChange() {
  for (const l of listeners) l();
}

export function registerActiveJobs(
  jobId: string,
  projectId: number | null,
  kind: string,
) {
  activeJobs = [...activeJobs, { jobId, projectId, kind }];
  emitChange();
}

/** Removes jobId from the active set and reports whether it actually did
 * so (false when some other caller already settled it first). Two
 * concurrent watcher poll cycles can both see the same job as terminal
 * (see useActiveJobWatcher.ts) -- the boolean return lets only the poll
 * that "wins" the removal fire onJobDone, so a slow poll cycle can never
 * cause a duplicate completion notification. */
export function markJobSettled(jobId: string): boolean {
  const next = activeJobs.filter((j) => j.jobId !== jobId);
  if (next.length === activeJobs.length) return false;
  activeJobs = next;
  emitChange();
  return true;
}

export function subscribeActiveJobs(listener: () => void): () => void {
  listeners.add(listener);
  return () => listeners.delete(listener);
}

export function getActiveJobsSnapshot(): ActiveJob[] {
  return activeJobs;
}

export function getActiveJobForProject(
  projectId: number | null,
): ActiveJob | null {
  if (projectId == null) return null;
  return activeJobs.find((j) => j.projectId === projectId) ?? null;
}
