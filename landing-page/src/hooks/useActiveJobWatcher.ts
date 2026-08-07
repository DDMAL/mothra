import { useEffect, useSyncExternalStore } from "react";
import {
  getActiveJobsSnapshot,
  subscribeActiveJobs,
  markJobSettled,
} from "../lib/activeJobs";
import { type ActiveJob } from "../lib/activeJobs";
import { apiFetch } from "../lib/apiFetch";

export function useActiveJobWatcher(
  onJobDone: (job: ActiveJob, status: string) => void,
) {
  const jobs = useSyncExternalStore(subscribeActiveJobs, getActiveJobsSnapshot);
  useEffect(() => {
    if (jobs.length === 0) return;
    const interval = setInterval(async () => {
      for (const job of jobs) {
        const r = await apiFetch(`/api/jobs/${job.jobId}`); // new endpoint, see Feature 3
        if (!r.ok) continue;
        const data = await r.json();
        if (["succeeded", "failed", "cancelled"].includes(data.status)) {
          // Only the poll cycle that actually removes this job (see
          // markJobSettled's docstring) reports it done -- otherwise a
          // slow-to-resolve fetch overlapping the next 5s tick can settle
          // the same job twice and fire onJobDone twice for it.
          if (markJobSettled(job.jobId)) {
            onJobDone(job, data.status);
          }
        }
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [jobs, onJobDone]);
}
