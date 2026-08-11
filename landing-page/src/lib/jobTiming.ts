const STORAGE_KEY = "mothra_job_durations";
const MAX_SAMPLES = 5;

type DurationSamples = Record<string, number[]>;

function load(): DurationSamples {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        return raw ? JSON.parse(raw) : {};
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
export function getAverageDurationMs(kind: string): number | null  {
    const samples = load()[kind];
    if (!samples || samples.length === 0) return null;
    return samples.reduce((a, b) => a + b, 0) / samples.length;
}

/** Records a completed job/item's actual duration, keeping only the most
 * recent MAX_SAMPLES so the estimate tracks recent behavior (e.g. after a
 * model or hardware change) rather than averaging in stale history forever. */
export function recordDurationMs(kind: string, durationMs: number) {
    if (!Number.isFinite(durationMs) || durationMs <= 0) return;
    const all = load();
    const samples = [...(all[kind] ?? []), durationMs].slice(-MAX_SAMPLES);
    save({ ...all, [kind]: samples});
}