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

    useEffect(() => {
        refetch();
        // Same 5s cadence as useActiveJobWatcher's terminal-status poll -- cheap,
        // and this is the only way a different tab's kickoff or a post-reload
        // in-progress job is ever discovered on this page.
        const interval = setInterval(refetch, 5000);
        return () => clearInterval(interval);
    }, [refetch]);

    return { activeJob: job, refetchActiveJob: refetch };
}