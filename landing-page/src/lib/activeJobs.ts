export interface ActiveJob { jobId: string; projectId: number | null; kind: string }

let activeJobs: ActiveJob[] = [];
const listeners = new Set<() => void>();

function emitChange() {
    for (const l of listeners) l();
}

export function registerActiveJobs(jobId: string, projectId: number | null, kind: string) {
    activeJobs = [...activeJobs, { jobId, projectId, kind }];
    emitChange();
}

export function markJobSettled(jobId: string) {
    const next = activeJobs.filter((j) => j.jobId !== jobId);
    if (next.length === activeJobs.length) return;
    activeJobs = next;
    emitChange();
}

export function subscribeActiveJobs(listener: () => void): () => void {
    listeners.add(listener);
    return () => listeners.delete(listener);
}

export function getActiveJobsSnapshot(): ActiveJob[] {
    return activeJobs;
}

export function getActiveJobForProject(projectId: number | null): ActiveJob | null {
    if (projectId == null) return null;
    return activeJobs.find((j) => j.projectId === projectId) ?? null;
}
