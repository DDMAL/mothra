import { useState, useEffect, useRef } from "react";
import type { Project, MeiFile } from "../../types";
import { apiFetch } from "../../lib/apiFetch";

interface BatchSession {
  session_id: string;
  manifest_id: string;
}

interface NeonBatchEditorProps {
  project: Project;
  meiFiles: MeiFile[];
  onFinish: () => void;
  onBack: () => void;
  onFileCorrected?: (id: string) => void;
}

// Shared sizing/typography for every button below; each call site adds its
// own bg-*/border-*/cursor-*/opacity-* classes explicitly (never bundled in
// here) so two conflicting utilities for the same property never land in
// one className string -- Tailwind resolves same-property conflicts by
// stylesheet order, not by where they appear in the string, so duplicating
// e.g. two different `bg-*` classes here would be a real (if subtle) bug.
const BTN_BASE =
  "px-3.5 py-1.5 rounded-md text-white text-[13px] whitespace-nowrap";

export default function NeonBatchEditor({
  project,
  meiFiles,
  onFinish,
  onBack,
  onFileCorrected,
}: NeonBatchEditorProps) {
  const [sessions, setSessions] = useState<Map<string, BatchSession>>(
    new Map(),
  );
  const [currentIndex, setCurrentIndex] = useState(() => {
    const firstUncorrected = meiFiles.findIndex((f) => !f.corrected);
    return firstUncorrected === -1 ? 0 : firstUncorrected;
  });
  const [corrected, setCorrected] = useState<Set<string>>(
    () => new Set(meiFiles.filter((f) => f.corrected).map((f) => f.id)),
  );
  const [loading, setLoading] = useState(true);
  const iframeRef = useRef<HTMLIFrameElement>(null);

  useEffect(() => {
    async function initSessions() {
      const results = await Promise.all(
        meiFiles.map(async (file) => {
          const r = await apiFetch(
            `/api/projects/${project.id}/mei/${file.id}/edit-session`,
            { method: "POST" },
          );
          if (!r.ok) return [file.id, null] as const;
          const data: BatchSession = await r.json();
          return [file.id, data] as const;
        }),
      );
      const map = new Map<string, BatchSession>();
      for (const [id, session] of results) {
        if (session) map.set(id, session);
      }
      setSessions(map);
      setLoading(false);
    }
    initSessions();
  }, [project.id, meiFiles]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key === "Enter") {
        e.preventDefault();
        handleDoneAndNext();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentIndex, corrected]);

  const currentFile = meiFiles[currentIndex];
  const currentSession = currentFile ? sessions.get(currentFile.id) : null;
  const allCorrected =
    meiFiles.length > 0 && meiFiles.every((f) => corrected.has(f.id));

  function nearestUncorrected(from: number, dir: 1 | -1): number {
    let i = from + dir;
    while (i >= 0 && i < meiFiles.length) {
      if (!corrected.has(meiFiles[i].id)) return i;
      i += dir;
    }
    return -1;
  }

  function triggerNeonSave() {
    const iframeBody = iframeRef.current?.contentDocument?.body;
    if (iframeBody) {
      iframeBody.dispatchEvent(
        new KeyboardEvent("keydown", { key: "s", bubbles: true }),
      );
    }
  }

  async function markCurrentDone() {
    if (!currentFile || corrected.has(currentFile.id)) return;
    triggerNeonSave();
    // brief wait for the async PUT inside Neon's updateDatabase() to complete
    await new Promise((r) => setTimeout(r, 800));
    await apiFetch(`/api/projects/${project.id}/mei/${currentFile.id}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ corrected: true }),
    });
    setCorrected((prev) => new Set([...prev, currentFile.id]));
    onFileCorrected?.(currentFile.id);
  }

  async function handleDoneAndNext() {
    await markCurrentDone();
    const next = nearestUncorrected(currentIndex, 1);
    if (next !== -1) setCurrentIndex(next);
  }

  const prevDisabled = nearestUncorrected(currentIndex, -1) === -1;
  const nextDisabled = nearestUncorrected(currentIndex, 1) === -1;
  const currentDone = corrected.has(currentFile?.id ?? "");

  return (
    <div className="flex flex-col h-screen bg-[#0f0f1a]">
      <div
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "ArrowRight") {
            const n = nearestUncorrected(currentIndex, 1);
            if (n !== -1) setCurrentIndex(n);
          }
          if (e.key === "ArrowLeft") {
            const n = nearestUncorrected(currentIndex, -1);
            if (n !== -1) setCurrentIndex(n);
          }
        }}
        className="flex items-center gap-2 px-4 py-2 bg-[#1a1a2e] border-b-2 border-[#4AADAA] shrink-0"
      >
        <span className="text-[#4AADAA55] text-[11px]">
          ← → navigate · Ctrl+Enter mark done
        </span>
        <button
          onClick={onBack}
          className={`${BTN_BASE} border-none bg-[#2d2d4e] cursor-pointer`}
        >
          ← Back
        </button>
        <button
          onClick={() => {
            const prev = nearestUncorrected(currentIndex, -1);
            if (prev !== -1) setCurrentIndex(prev);
          }}
          disabled={prevDisabled}
          className={`${BTN_BASE} border-none bg-[#2d2d4e] cursor-pointer ${prevDisabled ? "opacity-40" : "opacity-100"}`}
        >
          ← Prev
        </button>

        <div className="flex-1 flex gap-1.5 overflow-x-auto py-0.5">
          {meiFiles.map((f, i) => {
            const done = corrected.has(f.id);
            const isCurrent = i === currentIndex;
            const bg = isCurrent
              ? "bg-[#4AADAA]"
              : done
                ? "bg-[#1e4d4b]"
                : "bg-[#2d2d4e]";
            const border = isCurrent
              ? "border-none"
              : "border border-[#4AADAA44]";
            return (
              <button
                key={f.id}
                onClick={() => {
                  if (!done) setCurrentIndex(i);
                }}
                disabled={done}
                className={`${BTN_BASE} shrink-0 ${bg} ${border} ${done ? "opacity-40 cursor-not-allowed" : "opacity-100 cursor-pointer"}`}
              >
                {done ? "✓ " : ""}
                {f.name}
              </button>
            );
          })}
        </div>

        <button
          onClick={() => {
            const next = nearestUncorrected(currentIndex, 1);
            if (next !== -1) setCurrentIndex(next);
          }}
          disabled={nextDisabled}
          className={`${BTN_BASE} border-none bg-[#2d2d4e] cursor-pointer ${nextDisabled ? "opacity-40" : "opacity-100"}`}
        >
          Next →
        </button>

        <button
          onClick={handleDoneAndNext}
          className={`${BTN_BASE} border-none cursor-pointer ${currentDone ? "bg-[#1e4d4b]" : "bg-[#4AADAA]"}`}
        >
          {currentDone ? "✓ Done" : "Mark Done"}
          {currentIndex < meiFiles.length - 1 ? " & Next" : ""}
        </button>

        <button
          onClick={allCorrected ? onFinish : undefined}
          disabled={!allCorrected}
          className={`${BTN_BASE} border-none ${
            allCorrected
              ? "bg-[#22c55e] cursor-pointer opacity-100"
              : "bg-[#2d2d4e] cursor-not-allowed opacity-40"
          }`}
        >
          Finish All
        </button>
      </div>

      {loading ? (
        <div className="flex-1 flex items-center justify-center text-[#4AADAA]">
          Preparing editor...
        </div>
      ) : currentSession ? (
        <iframe
          ref={iframeRef}
          key={currentSession.session_id}
          src={`/neon/editor.html?manifest=${currentSession.session_id}`}
          className="flex-1 border-none w-full"
          title={`Neon editor - ${currentFile?.name ?? ""}`}
        />
      ) : (
        <div className="flex-1 flex items-center justify-center text-[#ef4444]">
          Failed to load editor for this file.
        </div>
      )}
    </div>
  );
}
