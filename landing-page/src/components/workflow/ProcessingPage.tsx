import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../../lib/apiFetch";
import { registerActiveJobs, markJobSettled } from "../../lib/activeJobs";
import { getAverageDurationMs, recordDurationMs } from "../../lib/jobTiming";

interface Stage {
  text: boolean;
  check: boolean;
}

interface ProcessingPageProps {
  onBack: () => void;
  onComplete: () => void;
  singleLabel?: string;
  intervalMs?: number;
  completionDelayMs?: number;
  logs?: string[];
  streamRequest?: (
    signal: AbortSignal,
    onJobId?: (id: string) => void,
  ) => Promise<Response>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onResult?: (data: any) => void;
  onLogsReady?: (logs: string[]) => void;
  onBatchDone?: (summary: { succeeded: unknown[]; failed: unknown[] }) => void;
  projectId?: number | null;
  jobKind?: string;
  initialLogsOpen?: boolean;
  startedAtMs?: number;
}

const STAGE_LABELS = ["checking", "validating", "processing"];
// mothra#236: the item-progress banner used to hardcode "encoding" — fine
// while encode_batch was the only job kind emitting item_start, but
// predict/text_batch now do too.
const ITEM_ACTION_LABELS: Record<string, string> = {
  encode_batch: "encoding",
  predict: "processing",
  text_batch: "finding text in",
};
const STAGE_IDX: Record<string, number> = {
  checking: 0,
  validating: 1,
  processing: 2,
};
const STAGE_PROGRESS: Record<string, number> = {
  checking: 3,
  validating: 6,
  processing: 100,
};

function formatRemaining(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s <= 0) return "almost done...";
  if (s >= 60) return `~${Math.floor(s / 60)}m ${s % 60}s remaining`;
  return `~${s}s remaining`;
}

function formatElapsed(ms: number): string {
  const s = Math.round(ms / 1000);
  if (s < 2) return "estimating...";
  if (s >= 60) return `${Math.floor(s / 60)}m ${s % 60}s elapsed`;
  return `${s}s elapsed`;
}

export default function ProcessingPage({
  onBack,
  onComplete,
  singleLabel,
  intervalMs = 100,
  completionDelayMs = 400,
  logs = [],
  streamRequest,
  onResult,
  onLogsReady,
  onBatchDone,
  projectId,
  jobKind,
  initialLogsOpen = false,
  startedAtMs,
}: ProcessingPageProps) {
  const [done, setDone] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stages, setStages] = useState<Stage[]>([
    { text: false, check: false },
    { text: false, check: false },
    { text: false, check: false },
  ]);
  const [logsOpen, setLogsOpen] = useState(initialLogsOpen);
  const [cancelPrompt, setCancelPrompt] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [cancelError, setCancelError] = useState<string | null>(null);

  const completedRef = useRef(false);
  const streamAbortRef = useRef<AbortController | null>(null);
  const jobIdRef = useRef<string | null>(null);
  const [revealedLogs, setRevealedLogs] = useState<string[]>([]);
  const [timeDisplay, setTimeDisplay] = useState<string>("estimating...");
  const logEndRef = useRef<HTMLDivElement>(null);

  const startTimeRef = useRef(Date.now());
  const progressRef = useRef(0);
  // Single-item jobs: estimated total job duration (ms), seeded from this
  // browser's history of past jobs of the same kind. Null until at least
  // one historical sample exists for this kind.
  const estimatedTotalMsRef = useRef<number | null>(null);
  // Batch jobs: estimated per-item duration (ms) — seeded from history the
  // same way, then overwritten with this run's own live average as soon as
  // one item in this run finishes (more relevant than history, since it
  // reflects the exact files being processed right now).
  const avgItemMsRef = useRef<number | null>(null);
  const itemDurationsRef = useRef<number[]>([]);
  const currentItemStartRef = useRef<number | null>(null);
  const itemProgressRef = useRef<{
    index: number;
    total: number;
    name?: string;
  } | null>(null);

  const confirmedProgressRef = useRef(0);
  const stageCeilingRef = useRef<number | null>(null);
  const stagePhaseStartRef = useRef<number | null>(null);

  const [streamError, setStreamError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [retryingJob, setRetryingJob] = useState(false);

  const [itemProgress, setItemProgress] = useState<{
    index: number;
    total: number;
    name?: string;
  } | null>(null);

  useEffect(() => {
    if (!logs || logs.length === 0) return;
    const totalMs = 100 * (intervalMs ?? 100);
    const timers = logs.map((line, i) => {
      const delay = Math.round((totalMs / (logs.length + 1)) * (i + 1));
      return setTimeout(() => {
        setRevealedLogs((prev) => [...prev, line]);
      }, delay);
    });
    return () => timers.forEach(clearTimeout);
  }, [logs]);

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [revealedLogs]);

  useEffect(() => {
    progressRef.current = progress;
  }, [progress]);

  useEffect(() => {
    itemProgressRef.current = itemProgress;
  }, [itemProgress]);

  useEffect(() => {
    const timer = setInterval(() => {
      const p = progressRef.current;
      if (p >= 100) {
        setTimeDisplay("");
        clearInterval(timer);
        return;
      }
      const ip = itemProgressRef.current;
      const now = Date.now();
      if (ip && ip.total > 1) {
        // Batch job: don't extrapolate from the stage checkpoints at all —
        // real per-item timing (this run's completed items, falling back
        // to history until one exists) is the reliable signal here.
        if (avgItemMsRef.current == null) {
          setTimeDisplay("estimating...");
          return;
        }
        const itemsRemaining = ip.total - ip.index - 1;
        const elapsedOnCurrentItem = currentItemStartRef.current
          ? now - currentItemStartRef.current
          : 0;
        const remainingMs =
          avgItemMsRef.current * itemsRemaining +
          Math.max(0, avgItemMsRef.current - elapsedOnCurrentItem);
        setTimeDisplay(formatRemaining(remainingMs));
        return;
      }
      // Single-item job: no in-run signal to average over. Countdown from
      // this browser's historical average for the same job kind if one
      // exists; otherwise show real elapsed time rather than a fabricated
      // number.
      const elapsedMs = now - startTimeRef.current;
      if (estimatedTotalMsRef.current == null) {
        setTimeDisplay(formatElapsed(elapsedMs));
        return;
      }
      setTimeDisplay(formatRemaining(estimatedTotalMsRef.current - elapsedMs));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    const timer = setInterval(() => {
      if (progressRef.current >= 100) return;
      const ceiling = stageCeilingRef.current;
      const floor = confirmedProgressRef.current;
      const phaseStart = stagePhaseStartRef.current;
      if (ceiling == null || ceiling <= floor || !phaseStart) return;
      const ip = itemProgressRef.current;

      const estMs = ip && ip.total > 1 ? avgItemMsRef.current : estimatedTotalMsRef.current;
      if (!estMs) return;
      const frac = Math.min((Date.now() - phaseStart) / estMs, 0.95);
      const interpolated = floor + frac * (ceiling - floor);
      if (interpolated > progressRef.current) setProgress(interpolated);
    }, 150);
    return () => clearInterval(timer);
  }, []);

  // extracted so a server-tracked job retry (handleRetryJob below) can feed a
  // freshly-opened job stream through the exact same parsing/progress logic
  // without re-running the kickoff effect below
  const consumeStream = useCallback(
    async (resp: Response) => {
      const collectedLogs: string[] = [];
      if (!resp.ok || !resp.body) {
        let msg = !resp.body
          ? "no response body"
          : `server error (HTTP ${resp.status})`;
        if (resp.body) {
          try {
            const data = await resp.json();
            if (data?.detail) msg = data.detail;
          } catch {
            // not JSON — keep the generic message
          }
        }
        setStreamError(msg);
        setRevealedLogs((prev) => [...prev, `error: ${msg}`]);
        return;
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const ev = JSON.parse(line.slice(6));
          if (ev.type === "item_start") {
            // Assign the ref synchronously, not just via setItemProgress —
            // if a single stream chunk contains item_start followed by
            // stage_done, the effect that syncs itemProgressRef from
            // itemProgress state hasn't run yet, so stage_done below would
            // otherwise read the *previous* item's index/total.
            const nextItemProgress = {
              index: ev.item,
              total: ev.total,
              name: ev.name,
            };
            itemProgressRef.current = nextItemProgress;
            setItemProgress(nextItemProgress);
            currentItemStartRef.current = Date.now();
            if (ev.total > 1 && avgItemMsRef.current == null) {
              avgItemMsRef.current = getAverageDurationMs(
                `${jobKind ?? "unknown"}:item`,
              );
            }
            // mothra#236: stage checkmarks are no longer reset here — each
            // stage's own re-arrival (the "stage" handler below) now clears
            // its own checkmark when it legitimately restarts (encode_batch's
            // per-item pipeline). predict/text_batch never re-emit
            // "stage"/checking|validating per item, so those stay lit across
            // the whole batch instead of flickering blank.
            //
            // mothra#233/#236 follow-up: predict/text_batch also never
            // re-emit "stage" for "processing" per item (only once, before
            // the loop), so item_start is this job kind's only per-item
            // signal — set the animation ceiling/phase-start here too. This
            // is a harmless redundant overwrite for encode_batch, which
            // follows up with its own real "stage" event a moment later.
            const stagePct = STAGE_PROGRESS["processing"] ?? 0;
            stageCeilingRef.current = Math.round(
              ((ev.item + stagePct / 100) / ev.total) * 100,
            );
            stagePhaseStartRef.current = Date.now();
          }
          if (ev.type === "stage") {
            const idx = STAGE_IDX[ev.name];
            if (idx !== undefined) {
              setStages((prev) =>
                prev.map((s, i) => (i === idx ? { ...s, text: true, check: false } : s)));
              const ip = itemProgressRef.current;
              const stagePct = STAGE_PROGRESS[ev.name] ?? 0;
              stageCeilingRef.current = ip ? Math.round(((ip.index + stagePct / 100) / ip.total) * 100) : stagePct;
              stagePhaseStartRef.current = Date.now();
            }
          }
          if (ev.type === "stage_done") {
            const idx = STAGE_IDX[ev.name];
            if (idx !== undefined) {
              setStages((prev) =>
                prev.map((s, i) => (i === idx ? { ...s, check: true } : s)),
              );
              const stagePct = STAGE_PROGRESS[ev.name] ?? 0;
              const ip = itemProgressRef.current;
              const exact = ip
                ? Math.round(((ip.index + stagePct / 100) / ip.total) * 100)
                : stagePct;
              // Keep the animation's floor in exact sync with every real,
              // event-driven progress update — the interpolation tick only
              // ever reads this, never writes it, so it always has an exact
              // value to animate from.
              confirmedProgressRef.current = exact;
              setProgress(exact);
              if (
                ev.name === "processing" &&
                ip &&
                ip.total > 1 &&
                currentItemStartRef.current
              ) {
                const itemDurationMs = Date.now() - currentItemStartRef.current;
                itemDurationsRef.current = [
                  ...itemDurationsRef.current,
                  itemDurationMs,
                ];
                avgItemMsRef.current =
                  itemDurationsRef.current.reduce((a, b) => a + b, 0) /
                  itemDurationsRef.current.length;
                recordDurationMs(
                  `${jobKind ?? "unknown"}:item`,
                  itemDurationMs,
                );
              }
            }
          }
          if (ev.type === "log") {
            const ip = itemProgressRef.current;
            const prefix =
              typeof ev.item === "number"
                ? `[${ev.item + 1}/${ip?.total ?? "?"}] `
                : "";
            const line = `${prefix}${ev.message}`;
            collectedLogs.push(line);
            setRevealedLogs((prev) => [...prev, line]);
          }
          if (ev.type === "result" && onResult) {
            // collectedLogs is complete for this item by the time its "result"
            // fires -- _encode_one (tasks_encode.py) always emits its "log"
            // events before "result", and nothing logs after "result" for that
            // same item. For a batch job collectedLogs accumulates across ALL
            // items processed so far (they run sequentially, each tagged with
            // its own `[item+1/total]` prefix -- see the "log" branch above),
            // so filter down to just this item's lines rather than attaching
            // every earlier item's logs to each one's own add_mei call too.
            const logs =
              typeof ev.item === "number"
                ? collectedLogs.filter((l) => l.startsWith(`[${ev.item + 1}/`))
                : [...collectedLogs];
            onResult({ ...ev, logs });
          }
          if (ev.type === "error") {
            setStreamError(ev.message);
            setRevealedLogs((prev) => [...prev, `error: ${ev.message}`]);
          }
          if (ev.type === "done" && !completedRef.current) {
            completedRef.current = true;
            if (singleLabel) setDone(true);
            if (jobIdRef.current) markJobSettled(jobIdRef.current);
            const ip = itemProgressRef.current;
            if (!ip || ip.total <= 1) {
              recordDurationMs(
                jobKind ?? "unknown",
                Date.now() - startTimeRef.current,
              );
            }
            onLogsReady?.(collectedLogs);
            if (ev.succeeded || ev.failed) {
              onBatchDone?.({
                succeeded: ev.succeeded ?? [],
                failed: ev.failed ?? [],
              });
            }
            setTimeout(onComplete, completionDelayMs);
          }
        }
      }
    },
    [onResult, onLogsReady, onBatchDone, onComplete, completionDelayMs],
  );

  const consumeStreamRef = useRef(consumeStream);
  useEffect(() => {
    consumeStreamRef.current = consumeStream;
  }, [consumeStream]);

  useEffect(() => {
    if (!streamRequest) return;
    setStreamError(null);
    setProgress(0);
    setDone(false);
    setStages([
      { text: false, check: false },
      { text: false, check: false },
      { text: false, check: false },
    ]);
    setRevealedLogs([]);
    setItemProgress(null);
    completedRef.current = false;
    startTimeRef.current =
      startedAtMs != null && Number.isFinite(startedAtMs) ? startedAtMs : Date.now();
    estimatedTotalMsRef.current = getAverageDurationMs(jobKind ?? "unknown");
    avgItemMsRef.current = null;
    itemDurationsRef.current = [];
    currentItemStartRef.current = null;
    confirmedProgressRef.current = 0;
    stageCeilingRef.current = null;
    stagePhaseStartRef.current = null;

    const abort = new AbortController();
    streamAbortRef.current = abort;

    async function run() {
      try {
        const resp = await streamRequest!(abort.signal, (id) => {
          jobIdRef.current = id;
          registerActiveJobs(id, projectId ?? null, jobKind ?? "unknown");
        });
        await consumeStreamRef.current(resp);
      } catch (e) {
        if ((e as Error).name !== "AbortError") {
          const msg = (e as Error).message;
          setStreamError(msg);
          setRevealedLogs((prev) => [...prev, `error: ${msg}`]);
        }
      }
    }
    run();
    return () => abort.abort();
    // intentionally excludes `streamRequest`/`consumeStream` (see consumeStreamRef
    // above) — this must only re-run on an explicit retry, not on every re-render
    // that happens to hand ProcessingPage new inline callback props (e.g. from
    // useActiveJobWatcher's store updates), or it kicks off a duplicate backend job.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [retryKey]);

  // server-tracked retry: re-runs the failed job's stored params via the
  // backend (jobs_api.py's POST /jobs/{id}/retry) instead of restarting the
  // whole kickoff client-side — distinct from the "restart" button below,
  // which re-collects params from scratch and re-invokes streamRequest.
  const handleRetryJob = useCallback(async () => {
    if (!jobIdRef.current) return;
    setRetryingJob(true);
    setStreamError(null);
    try {
      const r = await apiFetch(`/api/jobs/${jobIdRef.current}/retry`, {
        method: "POST",
      });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(
          (d as { detail?: string }).detail || `retry failed (${r.status})`,
        );
      }
      const { job_id: newId } = await r.json();
      jobIdRef.current = newId;
      registerActiveJobs(newId, projectId ?? null, jobKind ?? "unknown");
      completedRef.current = false;
      startTimeRef.current = Date.now();
      estimatedTotalMsRef.current = getAverageDurationMs(jobKind ?? "unknown");
      avgItemMsRef.current = null;
      itemDurationsRef.current = [];
      currentItemStartRef.current = null;
      confirmedProgressRef.current = 0;
      stageCeilingRef.current = null;
      stagePhaseStartRef.current = null;
      setProgress(0);
      setStages([
        { text: false, check: false },
        { text: false, check: false },
        { text: false, check: false },
      ]);
      const abort = new AbortController();
      streamAbortRef.current = abort;
      const stream = await apiFetch(`/api/jobs/${newId}/stream`, {
        signal: abort.signal,
      });
      await consumeStream(stream);
    } catch (e) {
      setStreamError((e as Error).message);
    } finally {
      setRetryingJob(false);
    }
  }, [consumeStream, projectId, jobKind]);

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48">
      <div className="w-full max-w-2xl">
        {itemProgress && (
          <div className="text-white/70 text-sm font-mono mb-2">
            {ITEM_ACTION_LABELS[jobKind ?? ""] ?? "processing"}{" "}
            {itemProgress.index + 1} of {itemProgress.total}
            {itemProgress.name ? ` - ${itemProgress.name}` : ""}
          </div>
        )}
        {singleLabel ? (
          <div className="text-4xl font-bold italic text-white leading-snug">
            {singleLabel}...{" "}
            <span
              className={`transition-opacity duration-500 ${done ? "opacity-100" : "opacity-0"}`}
            >
              [√]
            </span>
          </div>
        ) : (
          STAGE_LABELS.map((label, i) => (
            <div
              key={label}
              className={`text-4xl font-bold italic text-white transition-opacity duration-500 leading-snug
                            ${stages[i].text ? "opacity-100" : "opacity-0"}`}
            >
              {label}...{" "}
              <span
                className={`transition-opacity duration-500 ${stages[i].check ? "opacity-100" : "opacity-0"}`}
              >
                [√]
              </span>
            </div>
          ))
        )}

        {/* progress bar */}
        <div className="mt-8 w-full bg-white/30 rounded-full h-6 overflow-hidden">
          <div
            className="h-full bg-[#1E6B70] rounded-full transition-all duration-100"
            style={{ width: `${progress}%` }}
          />
        </div>

        {/* cancel + time estimate */}
        <div className="mt-4 flex items-center">
          {!cancelPrompt ? (
            <button
              onClick={() => setCancelPrompt(true)}
              className="text-white/50 text-sm hover:text-white cursor-pointer"
            >
              cancel
            </button>
          ) : (
            <div className="flex items-center gap-3 text-sm text-white">
              <span> are you sure? </span>
              <button
                onClick={async () => {
                  setCancelling(true);
                  setCancelError(null);
                  streamAbortRef.current?.abort();
                  const id = jobIdRef.current;
                  if (id) {
                    // Awaited and validated -- an unawaited fire-and-forget
                    // POST here meant a 404/403/409 or a network failure was
                    // silently ignored, and the job got marked settled and
                    // navigated away from regardless, leaving it running
                    // server-side with no way left on this page to retry the
                    // cancel. On failure, stay on this page (stream is
                    // already aborted either way -- the user confirmed they
                    // want to stop watching) and surface the error instead.
                    try {
                      const r = await apiFetch(`/api/jobs/${id}/cancel`, {
                        method: "POST",
                      });
                      if (!r.ok) {
                        const d = await r.json().catch(() => ({}));
                        setCancelError(
                          (d as { detail?: string }).detail ??
                            `cancel failed (${r.status})`,
                        );
                        setCancelling(false);
                        return;
                      }
                    } catch {
                      setCancelError("cancel failed -- check your connection");
                      setCancelling(false);
                      return;
                    }
                    markJobSettled(id);
                  }
                  // No job id yet (kickoff hasn't gotten one from the server
                  // -- nothing exists server-side to cancel) is the only
                  // path that reaches here without a confirmed cancel; it's
                  // still correct to just leave.
                  onBack();
                }}
                disabled={cancelling}
                className="px-3 py-1 bg-white text-[#4AADAA] rounded-lg font-semibold hover:opacity-90 cursor-pointer disabled:opacity-50"
              >
                {cancelling ? "cancelling..." : "yes"}
              </button>
              <button
                onClick={() => setCancelPrompt(false)}
                className="px-3 py-1 border border-white/40 text-white rounded-lg hover:opacity-90 cursor-pointer"
              >
                no
              </button>
            </div>
          )}
          {!cancelPrompt && timeDisplay && (
            <span className="flex-1 text-center text-white/60 text-sm font-mono">
              {timeDisplay}
            </span>
          )}
        </div>
        {cancelError && (
          <p className="mt-2 text-red-200 text-sm font-mono">{cancelError}</p>
        )}
        {streamError && !done && (
          <div className="mt-4 flex flex-col items-start gap-2">
            <p className="text-red-200 text-sm font-mono">{streamError}</p>
            <div className="flex gap-2">
              {jobIdRef.current && (
                <button
                  onClick={handleRetryJob}
                  disabled={retryingJob}
                  className="px-4 py-2 bg-white text-[#4AADAA] rounded-xl font-semibold text-sm hover:opacity-90 cursor-pointer disabled:opacity-50"
                >
                  {retryingJob ? "retrying..." : "retry job"}
                </button>
              )}
              <button
                onClick={() => setRetryKey((k) => k + 1)}
                className="px-4 py-2 border border-white text-white rounded-xl font-semibold text-sm hover:opacity-90 cursor-pointer"
              >
                restart
              </button>
            </div>
          </div>
        )}

        <div className="mt-4">
          <button
            onClick={() => setLogsOpen((o) => !o)}
            className="text-white/60 text-sm hover:text-white cursor-pointer select-none"
          >
            {logsOpen ? "v" : ">"} view logs
          </button>
          {logsOpen && (
            <div className="mt-2  bg-[#1D3335] rounded-xl h-32 w-full overflow-y-auto p-3">
              {revealedLogs.length > 0 ? (
                <>
                  {revealedLogs.map((line, i) => (
                    <div
                      key={i}
                      className={`text-xs font-mono leading-5 ${
                        line.startsWith("error: ")
                          ? "text-red-300"
                          : "text-white/70"
                      }`}
                    >
                      {line}
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </>
              ) : (
                <div className="text-white/30 text-xs font-mono">
                  waiting for logs...
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
