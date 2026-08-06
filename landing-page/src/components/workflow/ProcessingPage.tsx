import { useCallback, useEffect, useRef, useState } from "react";
import { apiFetch } from "../../lib/apiFetch";
import { registerActiveJobs, markJobSettled } from "../../lib/activeJobs";

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
  streamRequest?: (signal: AbortSignal, onJobId?: (id: string) => void) => Promise<Response>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  onResult?: (data: any) => void;
  onLogsReady?: (logs: string[]) => void
  onBatchDone?: (summary: { succeeded: unknown[]; failed: unknown[]; }) => void;
  projectId?: number | null;
  jobKind?: string;
}

const STAGE_LABELS = ["checking", "validating", "processing"];
const STAGE_IDX: Record<string, number> = { checking: 0, validating: 1, processing: 2 };
const STAGE_PROGRESS: Record<string, number> = { checking: 33, validating: 66, processing: 100 };

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
}: ProcessingPageProps) {
  const [done, setDone] = useState(false);
  const [progress, setProgress] = useState(0);
  const [stages, setStages] = useState<Stage[]>([
    { text: false, check: false },
    { text: false, check: false },
    { text: false, check: false },
  ]);
  const [logsOpen, setLogsOpen] = useState(false);
  const [cancelPrompt, setCancelPrompt] = useState(false);
  const pausedRef = useRef(false);
  const completedRef = useRef(false);
  const streamAbortRef = useRef<AbortController | null>(null);
  const jobIdRef = useRef<string | null>(null);
  const [revealedLogs, setRevealedLogs] = useState<string[]>([]);
  const [timeDisplay, setTimeDisplay] = useState<string>("estimating...")
  const logEndRef = useRef<HTMLDivElement>(null);
  const startTimeRef = useRef(Date.now());
  const progressRef = useRef(0);
  const itemProgressRef = useRef<{ index: number; total: number; name?: string } | null>(null);

  const [streamError, setStreamError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  const [retryingJob, setRetryingJob] = useState(false);

  const [itemProgress, setItemProgress] = useState<{ index: number; total: number; name?: string } | null>(null);

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
    pausedRef.current = cancelPrompt;
  }, [cancelPrompt]);

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
      const elapsedMs = Date.now() - startTimeRef.current;
      if (p <= 0) {
        setTimeDisplay("estimating...");
        return;
      }
      const totalMs = elapsedMs / (p / 100);
      const remainingMs = Math.max(0, totalMs - elapsedMs);
      const s = Math.round(remainingMs / 1000);
      if (s <= 0) {
        setTimeDisplay("almost done...");
        return;
      }
      if (s >= 60) {
        setTimeDisplay(`~${Math.floor(s / 60)}m ${s % 60}s remaining`);
      } else {
        setTimeDisplay(`~${s}s remaining`);
      }
    }, 1000);
    return () => clearInterval(timer);
  })

  useEffect(() => {
    if (streamRequest) return;
    const reveal = (stageIdx: number, key: keyof Stage, ms: number) =>
      setTimeout(
        () =>
          setStages((prev) =>
            prev.map((s, i) => (i === stageIdx ? { ...s, [key]: true } : s)),
          ),
        ms,
      );

    const timers = singleLabel
      ? []
      : [
          reveal(0, "text", 2000),
          reveal(0, "check", 3000),
          reveal(1, "text", 5000),
          reveal(1, "check", 6000),
          reveal(2, "text", 7000),
          reveal(2, "check", 8000),
        ];

    // fills to 100 over 10 s; pauses while cancel prompt is open
    const interval = setInterval(() => {
      if (!pausedRef.current) {
        setProgress((p) => {
          const next = Math.min(100, p + 1);
          if (next === 100 && !completedRef.current) {
            completedRef.current = true;
            if (singleLabel) setDone(true);
            setTimeout(onComplete, completionDelayMs);
          }
          return next;
        });
      }
    }, intervalMs);
    return () => {
      timers.forEach(clearTimeout);
      clearInterval(interval);
    };
  }, []);

  // extracted so a server-tracked job retry (handleRetryJob below) can feed a
  // freshly-opened job stream through the exact same parsing/progress logic
  // without re-running the kickoff effect below
  const consumeStream = useCallback(async (resp: Response) => {
    const collectedLogs: string[] = [];
    if (!resp.ok || !resp.body) {
      const msg = !resp.body ? "no response body" : `server error (HTTP ${resp.status})`;
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
          setItemProgress({ index: ev.item, total: ev.total, name: ev.name });
          setStages([{ text: false, check: false }, { text: false, check: false }, { text: false, check: false }]);
        }
        if (ev.type === "stage") {
          const idx = STAGE_IDX[ev.name];
          if (idx !== undefined)
            setStages((prev) => prev.map((s, i) => (i === idx ? { ...s, text: true } : s)));
        }
        if (ev.type === "stage_done") {
          const idx = STAGE_IDX[ev.name];
          if (idx !== undefined) {
            setStages((prev) => prev.map((s, i) => (i === idx ? { ...s, check: true } : s)));
            const stagePct = STAGE_PROGRESS[ev.name] ?? 0;
            const ip = itemProgressRef.current;
            setProgress(ip ? Math.round(((ip.index + stagePct / 100) / ip.total) * 100) : stagePct);
          }
        }
        if (ev.type === "log") {
          const ip = itemProgressRef.current;
          const prefix = typeof ev.item === "number" ? `[${ev.item + 1}/${ip?.total ?? "?"}] ` : "";
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
          const logs = typeof ev.item === "number"
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
          if (jobIdRef.current) markJobSettled(jobIdRef.current);
          onLogsReady?.(collectedLogs);
          if (ev.succeeded || ev.failed) {
            onBatchDone?.({ succeeded: ev.succeeded ?? [], failed: ev.failed ?? [] });
          }
          setTimeout(onComplete, completionDelayMs);
        }
      }
    }
  }, [onResult, onLogsReady, onBatchDone, onComplete, completionDelayMs]);

  const consumeStreamRef = useRef(consumeStream);
  useEffect(() => {
    consumeStreamRef.current = consumeStream;
  }, [consumeStream]);

  useEffect(() => {
    if (!streamRequest) return;
    setStreamError(null);
    setProgress(0);
    setStages([{ text: false, check: false}, { text: false, check: false }, { text: false, check: false }]);
    setRevealedLogs([]);
    setItemProgress(null);
    completedRef.current = false;
    startTimeRef.current = Date.now();

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
      const r = await apiFetch(`/api/jobs/${jobIdRef.current}/retry`, { method: "POST" });
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error((d as { detail?: string }).detail || `retry failed (${r.status})`);
      }
      const { job_id: newId } = await r.json();
      jobIdRef.current = newId;
      completedRef.current = false;
      setProgress(0);
      setStages([{ text: false, check: false }, { text: false, check: false }, { text: false, check: false }]);
      const abort = new AbortController();
      streamAbortRef.current = abort;
      const stream = await apiFetch(`/api/jobs/${newId}/stream`, { signal: abort.signal });
      await consumeStream(stream);
    } catch (e) {
      setStreamError((e as Error).message);
    } finally {
      setRetryingJob(false);
    }
  }, [consumeStream]);

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48">
      <div className="w-full max-w-2xl">
        {itemProgress && (
          <div className="text-white/70 text-sm font-mono mb-2">
            encoding {itemProgress.index + 1} of {itemProgress.total}
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
                  streamAbortRef.current?.abort(); 
                  if (jobIdRef.current) {
                    apiFetch(`/api/jobs/${jobIdRef.current}/cancel`, { method: "POST" }).catch(() => {});
                  }
                  onBack(); }}
                className="px-3 py-1 bg-white text-[#4AADAA] rounded-lg font-semibold hover:opacity-90 cursor-pointer"
              >
                yes
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
                onClick={() => setRetryKey(k => k + 1)}
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
                      className="text-white/70 text-xs font-mono leading-5"
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
