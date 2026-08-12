import { useCallback, useEffect, useState } from "react";
import { apiFetch } from "../lib/apiFetch";

export interface ProjectActiveJob {
    jobId: string;
    kind: string;
    status: string;
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
            const data: ProjectActiveJob | null = await r.json();
            setJob(data);
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
    // GET .../active-job requests forever for no reason. The tradeoff: a
    // job started in a DIFFERENT tab while this one sits idle here with no
    // job of its own won't be noticed until this component remounts -- an
    // edge case, versus indefinite chatty polling being the common case.
    useEffect(() => {
        if (!job) return;
        const interval = setInterval(refetch, 5000);
        return () => clearInterval(interval);
    }, [job, refetch]);

    return { activeJob: job, refetchActiveJob: refetch };
}