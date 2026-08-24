import {
  useState,
  useEffect,
  useRef,
  forwardRef,
  useImperativeHandle,
} from "react";
import type { Project, MeiFile } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import UnsavedChangesModal from "./UnsavedChangesModal";

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

// Exposed to App.tsx so the browser back/forward popstate handler (which
// lives outside this component, next to the rest of the view-history
// machinery -- see App.tsx) can run the same unsaved-work gate as the five
// in-app navigation actions below (issue #266) instead of jumping the view
// away immediately.
export interface NeonEditorHandle {
  isUnsaved: () => boolean;
  attemptNavigation: (action: () => void) => void;
}

// Shared sizing/typography for every button below; each call site adds its
// own bg-*/border-*/cursor-*/opacity-* classes explicitly (never bundled in
// here) so two conflicting utilities for the same property never land in
// one className string -- Tailwind resolves same-property conflicts by
// stylesheet order, not by where they appear in the string, so duplicating
// e.g. two different `bg-*` classes here would be a real (if subtle) bug.
const BTN_BASE =
  "px-3.5 py-1.5 rounded-md text-white text-[13px] whitespace-nowrap";

// Neon's own Square/Hufnagel font toggle (its Display panel's "notation"
// dropdown -- see neon/src/DisplayPanel/DisplayControls.ts's
// setNotationTypeControls()) only ever reflects a per-BROWSER LocalSettings
// value from whichever option was last clicked -- it never looks at the
// document it just loaded, so switching between a square- and a
// hufnagel-encoded file here would otherwise keep whatever font the last
// file happened to leave selected. Mirrors triggerNeonSave() below: drive
// Neon's own UI via a synthetic DOM interaction inside the iframe's
// contentDocument rather than patching the neon submodule itself.
//
// notationtype is read straight off the MEI's own <staffDef> (written by
// encode_to_mei.py's build_mei -- see mothra#210) rather than threaded
// through as a separate prop, since the MEI itself is the single source of
// truth for which notation a given file actually uses.
//
// Scoped to <staffDef> specifically (via getElementsByTagNameNS, since MEI
// is namespaced and plain querySelector tag matching doesn't cross that) --
// a blind substring/regex match over the whole document could in principle
// hit a `notationtype="neume.hufnagel"` string anywhere else in the file,
// not just the element Neon/Verovio actually read it from.
function readNotationType(xmlContent: string): "square" | "hufnagel" | null {
  let doc: Document;
  try {
    doc = new DOMParser().parseFromString(xmlContent, "application/xml");
  } catch {
    return null;
  }
  if (doc.getElementsByTagName("parsererror").length > 0) return null;
  const staffDefs = doc.getElementsByTagNameNS("*", "staffDef");
  for (let i = 0; i < staffDefs.length; i++) {
    const value = staffDefs[i].getAttribute("notationtype");
    if (value === "neume.square") return "square";
    if (value === "neume.hufnagel") return "hufnagel";
  }
  return null;
}

function applyNotationTypeFont(
  iframe: HTMLIFrameElement,
  xmlContent: string | undefined,
  timerRef: { current: number | null },
) {
  // A previous file's poll (still waiting for Neon's Display panel to
  // appear) must not fire after this file has taken over the same iframe,
  // or after the iframe itself has been unmounted (session switch remounts
  // it under a new `key`) -- either way the stale closure would still hold
  // a reference to the old iframe/contentDocument. One timer at a time.
  if (timerRef.current !== null) {
    window.clearTimeout(timerRef.current);
    timerRef.current = null;
  }
  if (!xmlContent) return;
  // No notationtype, or a value neither Neon control corresponds to --
  // leave whatever font Neon already has selected rather than guessing.
  const notationType = readNotationType(xmlContent);
  if (notationType === null) return;
  const targetId =
    notationType === "hufnagel"
      ? "notation-type-hufnagel"
      : "notation-type-square";
  // Neon's Display panel doesn't exist yet the instant the iframe's `load`
  // event fires -- its own manifest fetch + NeonView/SingleView init still
  // need to run inside the iframe first. Poll briefly rather than assume a
  // fixed delay (matches the tolerance markCurrentDone() already gives
  // Neon's own async save, just for the opposite direction: waiting for
  // something to appear instead of finish).
  let attempts = 0;
  const tryClick = () => {
    const el = iframe.contentDocument?.getElementById(targetId);
    if (el) {
      el.click();
      timerRef.current = null;
      return;
    }
    if (attempts++ < 40) {
      timerRef.current = window.setTimeout(tryClick, 250);
    } else {
      timerRef.current = null;
    }
  };
  tryClick();
}

const NeonBatchEditor = forwardRef<NeonEditorHandle, NeonBatchEditorProps>(
  function NeonBatchEditor(
    { project, meiFiles, onFinish, onBack, onFileCorrected },
    ref,
  ) {
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
    const [showUnsavedModal, setShowUnsavedModal] = useState(false);
    // Holds whichever navigation was attempted while the current file had
    // unsaved edits, so the confirm modal's "leave anyway" can run the exact
    // action the user originally clicked (Back, Prev, Next, or a specific
    // filmstrip file) rather than a hardcoded one.
    const pendingActionRef = useRef<(() => void) | null>(null);

    const iframeRef = useRef<HTMLIFrameElement>(null);
    // Shared across every applyNotationTypeFont call for this component
    // instance so a new call (new file, new session) always cancels whatever
    // poll the previous one left running -- see that function's comment.
    const notationTimerRef = useRef<number | null>(null);
    useEffect(() => {
      return () => {
        // Deliberately reads notationTimerRef.current at cleanup/unmount
        // time, not a value captured at mount -- this ref holds a plain
        // timer id we set ourselves (not a DOM node), so the
        // exhaustive-deps rule's usual concern (a ref that may have gone
        // stale/null by unmount) doesn't apply here.
        if (notationTimerRef.current !== null) {
          // eslint-disable-next-line react-hooks/exhaustive-deps
          window.clearTimeout(notationTimerRef.current);
        }
      };
    }, []);

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

    function isNeonUnsaved(iframe: HTMLIFrameElement | null): boolean {
      const indicator = iframe?.contentDocument?.getElementById("file-saved");
      return indicator?.getAttribute("alt") === "You have unsaved work";
    }

    // Routes every "switch away from the current file" action (Back, Prev,
    // Next, filmstrip, arrow keys) through the same unsaved-work check
    // (issue #266) instead of running it immediately.
    function attemptNavigation(action: () => void) {
      if (isNeonUnsaved(iframeRef.current)) {
        pendingActionRef.current = action;
        setShowUnsavedModal(true);
      } else {
        action();
      }
    }

    // Lets App.tsx's browser back/forward popstate handler reuse this exact
    // gate (see the NeonEditorHandle comment above) instead of duplicating
    // the unsaved-work check outside the component.
    useImperativeHandle(ref, () => ({
      isUnsaved: () => isNeonUnsaved(iframeRef.current),
      attemptNavigation,
    }));

    // Neon's own "s" keydown handler (EditControls.ts) calls
    // neonView.save().then(() => Notification.queueNotification('Saved',
    // 'success')) -- that Promise is the real completion signal for the async
    // PUT inside updateDatabase(), but nothing forwards it out of the iframe's
    // private keydown listener. Rather than guessing a fixed delay (the former
    // 800ms here) or patching the neon submodule to expose it, poll for the
    // "Saved" toast Notification.ts actually renders into
    // #notification-content -- same bounded-polling idiom as
    // applyNotationTypeFont above, just watching for something to *appear*
    // instead of *finish* being the opposite direction of the same tolerance.
    // Tracks which notification ids already existed before the save so a
    // stale, not-yet-auto-cleared toast from an earlier save can't be
    // mistaken for this one's completion.
    // Returns whether the success toast was actually observed -- a timeout is
    // not confirmation, so callers must not treat it as success (see
    // markCurrentDone below, mothra CodeRabbit review on #256).
    function waitForNeonSaveComplete(
      iframe: HTMLIFrameElement,
      timeoutMs = 5000,
    ): Promise<boolean> {
      return new Promise((resolve) => {
        const container = iframe.contentDocument?.getElementById(
          "notification-content",
        );
        const seenIds = new Set(
          container ? Array.from(container.children).map((el) => el.id) : [],
        );
        const start = Date.now();
        const poll = () => {
          const newSuccessToast = container
            ? Array.from(container.children).find(
                (el) =>
                  el.classList.contains("neon-notification-success") &&
                  !seenIds.has(el.id),
              )
            : undefined;
          if (newSuccessToast) {
            resolve(true);
            return;
          }
          if (Date.now() - start > timeoutMs) {
            resolve(false);
            return;
          }
          setTimeout(poll, 100);
        };
        poll();
      });
    }

    // Returns whether the file was actually marked corrected -- false when the
    // Neon save was never confirmed (iframe unavailable, save failed, or the
    // toast timed out), so the caller knows not to advance past unsaved edits.
    async function markCurrentDone(): Promise<boolean> {
      if (!currentFile || corrected.has(currentFile.id)) return false;
      triggerNeonSave();
      if (
        !iframeRef.current ||
        !(await waitForNeonSaveComplete(iframeRef.current))
      ) {
        return false;
      }
      await apiFetch(`/api/projects/${project.id}/mei/${currentFile.id}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ corrected: true }),
      });
      setCorrected((prev) => new Set([...prev, currentFile.id]));
      onFileCorrected?.(currentFile.id);
      return true;
    }

    async function handleDoneAndNext() {
      if (!(await markCurrentDone())) return;
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
              if (n !== -1) attemptNavigation(() => setCurrentIndex(n));
            }
            if (e.key === "ArrowLeft") {
              const n = nearestUncorrected(currentIndex, -1);
              if (n !== -1) attemptNavigation(() => setCurrentIndex(n));
            }
          }}
          className="flex items-center gap-2 px-4 py-2 bg-[#1a1a2e] border-b-2 border-[#4AADAA] shrink-0"
        >
          <span className="text-[#4AADAA55] text-[11px]">
            ← → navigate · Ctrl+Enter mark done
          </span>
          <button
            onClick={() => attemptNavigation(onBack)}
            className={`${BTN_BASE} border-none bg-[#2d2d4e] cursor-pointer`}
          >
            ← Back
          </button>
          <button
            onClick={() => {
              const prev = nearestUncorrected(currentIndex, -1);
              if (prev !== -1) attemptNavigation(() => setCurrentIndex(prev));
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
                    if (!done) attemptNavigation(() => setCurrentIndex(i));
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
              if (next !== -1) attemptNavigation(() => setCurrentIndex(next));
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
            onClick={
              // issue #266 (CodeRabbit): allCorrected only reflects each
              // file having been marked done at some point -- editing the
              // current file again afterward doesn't clear its "done" mark,
              // so this still needs the same unsaved-work gate as every
              // other exit from the editor.
              allCorrected ? () => attemptNavigation(onFinish) : undefined
            }
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
            onLoad={(e) => {
              e.currentTarget.contentWindow?.focus(); // focus new iframe immediately so shortcuts work
              applyNotationTypeFont(
                e.currentTarget,
                currentFile?.xmlContent,
                notationTimerRef,
              )
            }}
          />
        ) : (
          <div className="flex-1 flex items-center justify-center text-[#ef4444]">
            Failed to load editor for this file.
          </div>
        )}
        {showUnsavedModal && (
          <UnsavedChangesModal
            onConfirm={() => {
              setShowUnsavedModal(false);
              const action = pendingActionRef.current;
              pendingActionRef.current = null;
              action?.();
            }}
            onCancel={() => {
              setShowUnsavedModal(false);
              pendingActionRef.current = null;
            }}
          />
        )}
      </div>
    );
  },
);

export default NeonBatchEditor;
