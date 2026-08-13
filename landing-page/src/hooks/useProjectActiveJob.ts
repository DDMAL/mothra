import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../lib/apiFetch";

export interface ProjectActiveJob {
    jobId: string;
    kind: string;
    status: string;
    createdAt: string | null;
}

/** Polls GET /api/projects/{id}/active-job -- the server-side source of
 * truth for job_store.py's get_active_job_for_project -- so the project page
 * can show progress/cancel controls even after a reload or from a different
 * tab than the one that kicked the job off. activeJobs.ts's in-memory
 * registry only covers the same-tab-session case; this covers the rest. */
export function useProjectActiveJob(projectId: number | null) {
    const [job, setJob] = useState<ProjectActiveJob | null>(null);

    const refetch = useCallback(async () => {
        if (projectId == null) {
            setJob(null);
            return;
        }
        try {
            const r = await apiFetch(`/api/projects/${projectId}/active-job`);
            if (!r.ok) return;
            // The backend (projects_api.py's get_project_active_job) returns
            // snake_case ({job_id, kind, status}), matching every other API
            // response in this codebase -- map it explicitly rather than
            // casting, or every consumer of the server-only path (a job
            // discovered after a reload/from a different tab, with nothing
            // in the in-memory activeJobs.ts registry to fall back on) reads
            // `jobId` as undefined.
            const data: {
                job_id: string;
                kind: string;
                status: string;
                created_at: string | null;
            } | null = await r.json();
            setJob(
                data
                    ? {
                          jobId: data.job_id,
                          kind: data.kind,
                          status: data.status,
                          createdAt: data.created_at,
                      }
                    : null,
            );
        } catch {
            // network hiccup -- keep the last known value rather than flashing to null
        }
    }, [projectId]);

    // One refetch on mount (or when projectId changes) -- this alone is what
    // catches a post-reload or a different tab's already-in-progress job.
    useEffect(() => {
        refetch();
    }, [refetch]);

    // Keep polling on the same 5s cadence as useActiveJobWatcher's
    // terminal-status poll, but ONLY while a job is actually known to be
    // active -- polling unconditionally for as long as ProjectDetail stays
    // mounted (even with nothing ever having run) spammed the backend with
    // GET .../active-job requests forever for no reason. Known gap: a job
    // started elsewhere while THIS hook's `job` is null won't be noticed
    // until it remounts -- whether "elsewhere" is a different tab, or this
    // same tab kicking one off without a remount in between. The in-memory
    // activeJobs.ts registry covers the same-tab case for the "is Continue
    // disabled" check itself; what's missed here is just this hook's own
    // `status` label staying stale until the next mount/poll tick.
    useEffect(() => {
        if (!job) return;
        const interval = setInterval(refetch, 5000);
        return () => clearInterval(interval);
    }, [job, refetch]);

    return { activeJob: job, refetchActiveJob: refetch };
}