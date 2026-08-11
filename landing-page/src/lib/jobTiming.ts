// Historical per-job-kind duration samples, used to give ProcessingPage.tsx
// a real (not fabricated) time estimate. The backend's 3 stage checkpoints
// (checking/validating/processing) aren't proportional to actual wall-clock
// time — validation is near-instant, "processing" is the expensive part —
// so extrapolating "total time" from elapsed-time-so-far/percent-so-far
// produces nonsense (issue #186). Instead, remember how long past jobs of
// the same kind actually took (or, for batch jobs, how long one item took)
// and use that as the estimate, refined every time a job completes.

const STORAGE_KEY = "mothra_job_durations";
const MAX_SAMPLES = 5;

type DurationSamples = Record<string, number[]>;

function isValidSample(value: unknown): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

function load(): DurationSamples {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    const parsed: unknown = raw ? JSON.parse(raw) : {};
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
      return {};
    }
    // localStorage is user-editable and can outlive schema changes, so
    // don't trust its shape — drop any entry that isn't an array of finite
    // positive numbers rather than letting a malformed value throw later.
    const sanitized: DurationSamples = {};
    for (const [kind, values] of Object.entries(parsed)) {
      if (!Array.isArray(values)) continue;
      const samples = values.filter(isValidSample).slice(-MAX_SAMPLES);
      if (samples.length > 0) sanitized[kind] = samples;
    }
    return sanitized;
  } catch {
    return {};
  }
}

function save(samples: DurationSamples) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(samples));
  } catch {
    // localStorage unavailable/full — the estimate is a nice-to-have, not
    // worth surfacing an error for.
  }
}

/** Average duration (ms) of the last few completed jobs/items of this kind,
 * or null if none have completed yet in this browser. */
export function getAverageDurationMs(kind: string): number | null {
  const samples = load()[kind];
  if (!samples || samples.length === 0) return null;
  return samples.reduce((a, b) => a + b, 0) / samples.length;
}

/** Records a completed job/item's actual duration, keeping only the most
 * recent MAX_SAMPLES so the estimate tracks recent behavior (e.g. after a
 * model or hardware change) rather than averaging in stale history forever. */
export function recordDurationMs(kind: string, durationMs: number) {
  if (!isValidSample(durationMs)) return;
  const all = load();
  const samples = [...(all[kind] ?? []), durationMs].slice(-MAX_SAMPLES);
  save({ ...all, [kind]: samples });
}
