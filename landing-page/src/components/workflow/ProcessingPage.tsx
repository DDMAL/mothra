import { useEffect, useRef, useState } from "react";

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
}

const STAGE_LABELS = ["checking", "validating", "processing"];

export default function ProcessingPage({
  onBack,
  onComplete,
  singleLabel,
  intervalMs = 100,
  completionDelayMs = 400,
  logs = []
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
  const [revealedLogs, setRevealedLogs] = useState<string[]>([]);
  const logEndRef = useRef<HTMLDivElement>(null);

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
    const reveal = (stageIdx: number, key: keyof Stage, ms: number) =>
      setTimeout(
        () =>
          setStages((prev) =>
            prev.map((s, i) => (i === stageIdx ? { ...s, [key]: true } : s)),
          ),
        ms,
      );

    const timers = singleLabel ? [] : [
      // hardcoded rnw for testing
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

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48">
      <div className="w-full max-w-2xl">

        {singleLabel ? (
          <div className="text-4xl font-bold italic text-white leading-snug">
            {singleLabel}...{" "}
            <span className={`transition-opacity duration-500 ${done ? "opacity-100" : "opacity-0"}`}>[√]</span>
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

        {/* cancel */}
        <div className="mt-4">
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
                onClick={onBack}
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
        </div>

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
                    <div key={i} className="text-white/70 text-xs font-mono leading-5">
                      {line}
                    </div>
                  ))}
                  <div ref={logEndRef} />
                </>
              ) : (
                <div className="text-white/30 text-xs font-mono">waiting for logs...</div>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
