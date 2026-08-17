import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ProjectImage } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { buildEncodePair } from "../../utils/icQueue";
import type { EncodePair } from "../../utils/icQueue";
import { AuthImage } from "../shared/AuthImage";

interface InteractiveClassifierProps {
  // Only the pages step 1 still has work for - see pendingIcImages().
  images: ProjectImage[];
  // Page to open on instead of the first one - set when the user picked a
  // saved session in "manage IC sessions". Nothing session-specific needs to
  // be threaded through beyond this: /ic/start resumes whatever session is
  // saved for the selected page, and IC keeps at most one per page.
  initialImageName?: string | null;
  // How many pages the project has selected in total, pending or not. Lets the
  // empty state tell "nothing selected yet" apart from "all already encoded".
  usedImageCount: number;
  projectId: number | null;
  onBack: () => void;
  onEncodeBatch: (pairs: EncodePair[]) => void;
  clefShape: "C" | "F";
  onClefShapeChange: (s: "C" | "F") => void;
  clefLine: number;
  onClefLineChange: (n: number) => void;
  // Training set picked on the project page ("Classifier settings").
  // Pre-selected in
  // each page's IC create-session screen via the ic:prefill-training reply
  // below; empty when the user made no choice there, in which case the
  // classifier opens with nothing pre-selected (its own default).
  trainingPresets: string[];
  trainingFiles: File[];
}

export default function InteractiveClassifier({
  images,
  initialImageName = null,
  usedImageCount,
  projectId,
  onBack,
  onEncodeBatch,
  clefShape,
  onClefShapeChange,
  clefLine,
  onClefLineChange,
  trainingPresets,
  trainingFiles,
}: InteractiveClassifierProps) {
  // Resolved once, at mount: this view is remounted on every entry (AppRouter
  // switches on `view`), and after that the filmstrip owns the selection - a
  // resume shouldn't keep yanking the user back to its page.
  const [currentIdx, setCurrentIdx] = useState(() => {
    const i = initialImageName
      ? images.findIndex((im) => im.name === initialImageName)
      : -1;
    return i === -1 ? 0 : i;
  });
  const [icUrl, setIcUrl] = useState<string | null>(null);
  const [icOrigin, setIcOrigin] = useState<string | null>(null);
  // True when /ic/start staged a fabricated placeholder bbox grid instead of
  // real YOLO detections (no predict run has happened for this page yet --
  // the VITE_SKIP_PREDICT dev bypass, or a page reached here some other way
  // with no annotation row). Not reported on a *resumed* session (IC's own
  // staging service doesn't track this after the fact) -- mothra#220 DL-1.
  const [synthetic, setSynthetic] = useState(false);
  // Set only once the user finishes IC's create-session screen (the iframe
  // posts it back); until then there's nothing to encode.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "starting" | "ready" | "error">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  // Set while "encode batch" finalises the queued sessions (see
  // handleEncodeBatch) - queueing itself is local state now, so nothing else
  // is async enough to need a busy flag.
  const [finalizing, setFinalizing] = useState(false);
  // Queued pages hold a *reference* to their still-editable IC session, not
  // its GameraXML: finalising is deferred to "encode batch" (see
  // handleEncodeBatch) so a queued-but-not-encoded page keeps its corrections.
  const [queue, setQueue] = useState<
    { image: ProjectImage; sessionId: string }[]
  >([]);
  // Set when IC's in-iframe "auto-export" fires — there's no "queue page"
  // click on that path, so we run the queue logic ourselves once state settles.
  const [autoQueueRequested, setAutoQueueRequested] = useState(false);

  const queuedNames = useMemo(
    () => new Set(queue.map((q) => q.image.name)),
    [queue],
  );

  const img = images[currentIdx];
  // Guards against a slow /ic/start response landing after the user has
  // already switched pages — only the latest request may set state.
  const startSeq = useRef(0);

  // Training selection from the project page, readable from the message
  // handler below without re-subscribing the listener when it changes.
  const trainingRef = useRef({
    presets: trainingPresets,
    files: trainingFiles,
  });
  useEffect(() => {
    trainingRef.current = { presets: trainingPresets, files: trainingFiles };
  }, [trainingPresets, trainingFiles]);

  // Stage a fresh page + bboxes in IC whenever the selected page changes.
  useEffect(() => {
    if (!img || projectId == null) return;
    const seq = ++startSeq.current;
    setStatus("starting");
    setError(null);
    setIcUrl(null);
    setIcOrigin(null);
    setSessionId(null);
    setSynthetic(false);

    apiFetch(`/api/projects/${projectId}/ic/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageName: img.name, imageId: img.id }),
    })
      .then(async (r) => {
        if (!r.ok)
          throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
        return r.json();
      })
      .then((data) => {
        if (seq !== startSeq.current) return; // superseded by a newer page
        setIcUrl(data.ic_url);
        setSynthetic(Boolean(data.synthetic));
        try {
          setIcOrigin(new URL(data.ic_url).origin);
        } catch {
          setIcOrigin(null);
        }
        setStatus("ready");
      })
      .catch((err) => {
        if (seq !== startSeq.current) return;
        setError(String(err.message ?? err));
        setStatus("error");
      });
  }, [img?.name, projectId]);

  // The embedded IC posts its new session id once the user starts the
  // session on the create-session screen. Accept it only from IC's origin.
  useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (icOrigin && e.origin !== icOrigin) return;
      const data = e.data;
      // The embedded create-session screen announces readiness; reply with the
      // project-level training set (if any) so the user doesn't have to
      // re-pick it on every page. The iframe seeds its own inputs from this
      // and won't overwrite a selection already made there.
      if (data?.type === "ic:ready") {
        const { presets, files } = trainingRef.current;
        if (presets.length > 0 || files.length > 0) {
          (e.source as Window | null)?.postMessage(
            { type: "ic:prefill-training", presets, files },
            e.origin || "*",
          );
        }
        return;
      }
      if (
        data?.type === "ic:session-created" &&
        typeof data.sessionId === "string"
      ) {
        setSessionId(data.sessionId);
      }
      // "Trust the classifier" path: IC ran classify + skipped the interactive
      // step. Adopt the session and flag it for auto-queuing (handled by the
      // effect below, not here, to avoid a not-yet-flushed sessionId and a
      // stale handleQueuePage closure).
      if (
        data?.type === "ic:auto-export" &&
        typeof data.sessionId === "string"
      ) {
        setSessionId(data.sessionId);
        setAutoQueueRequested(true);
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [icOrigin]);

  // Mark this page's session for encoding and move on. Deliberately does *not*
  // call /complete: that transitions the IC session to EXPORT, which is
  // terminal and read-only, so /ic/start can no longer resume it (see IC's
  // store.lookup()). Doing it here meant a page queued but never encoded - the
  // user goes back to the project, or just never presses "encode batch" - lost
  // every correction made in it, since re-entering the page started a fresh
  // session. Finalising happens in handleEncodeBatch instead.
  const handleQueuePage = useCallback(() => {
    if (!sessionId || !img) return;
    setError(null);
    setQueue((prev) =>
      prev.some((q) => q.image.name === img.name)
        ? // Same page re-queued (it stays editable, so this can happen): keep
          //   the newest session id rather than adding a second entry.
          prev.map((q) =>
            q.image.name === img.name ? { image: img, sessionId } : q,
          )
        : [...prev, { image: img, sessionId }],
    );
    const nextIdx = images.findIndex(
      (im, idx) => idx > currentIdx && !queuedNames.has(im.name),
    );
    if (nextIdx !== -1) setCurrentIdx(nextIdx);
  }, [sessionId, img, images, currentIdx, queuedNames]);

  // Run the queue path for an auto-exported page once the session id and
  // current page have settled. Gated on the flag (reset immediately) so it
  // fires exactly once per auto-export, and skips a page already queued.
  useEffect(() => {
    if (autoQueueRequested && sessionId && img && !queuedNames.has(img.name)) {
      setAutoQueueRequested(false);
      handleQueuePage();
    }
  }, [autoQueueRequested, sessionId, img, queuedNames, handleQueuePage]);

  // Finalise every queued session (CLASSIFYING → EXPORT) and hand the
  // resulting GameraXML + image pairs to the batch encoder. This is the *only*
  // place a session is ended: up to here each one stays editable and resumable,
  // so abandoning the step costs nothing. Because the XML is snapshotted here
  // rather than at queue time, corrections made to a page *after* queueing it
  // are picked up too.
  const handleEncodeBatch = useCallback(async () => {
    if (queue.length === 0 || finalizing) return;
    setFinalizing(true);
    setError(null);
    try {
      const pairs: EncodePair[] = [];
      for (const { image, sessionId: sid } of queue) {
        const r = await apiFetch(`/api/ic/${sid}/complete`, { method: "POST" });
        if (!r.ok) {
          const detail = await r.text().catch(() => `HTTP ${r.status}`);
          throw new Error(`${image.name}: ${detail}`);
        }
        const data = await r.json();
        pairs.push(await buildEncodePair(image, data.xml_base64));
      }
      onEncodeBatch(pairs);
    } catch (err) {
      // Left queued on failure: the sessions that did finalise are re-exportable
      // (IC allows re-export from EXPORT), so pressing the button again retries
      // the whole queue rather than dropping the pages that already worked.
      setError(String((err as Error).message ?? err));
    } finally {
      setFinalizing(false);
    }
  }, [queue, finalizing, onEncodeBatch]);

  // show 5 thumbnails at a time, centered on currentIdx
  const VISIBLE = 5;
  const half = Math.floor(VISIBLE / 2);
  const start = Math.max(
    0,
    Math.min(currentIdx - half, images.length - VISIBLE),
  );
  const visibleImages = images.slice(start, start + VISIBLE);

  return (
    <div className="animate-fade-in flex-1 min-h-0 bg-[#4AADAA] flex flex-col pb-3">
      <div className="flex items-center gap-6 px-8 py-3">
        <button
          onClick={onBack}
          title="back to project"
          className="text-white text-2xl hover:opacity-70 transition-opacity cursor-pointer shrink-0"
        >
          ←
        </button>
        <h1 className="text-4xl font-bold italic text-white">
          interactive classifier
        </h1>
        {images.length > 1 && (
          <span className="text-white/80 text-sm font-mono">
            page {currentIdx + 1}/{images.length}
            {img ? ` — ${img.name}` : ""}
          </span>
        )}
        <div className="flex items-center gap-2 text-white/80 text-sm">
          <span className="text-white/50 text-xs">clef</span>
          <select
            value={clefShape}
            onChange={(e) => onClefShapeChange(e.target.value as "C" | "F")}
            className="bg-transparent border border-white/30 rounded px-1 text-sm cursor-pointer text-white"
          >
            <option value="C">C</option>
            <option value="F">F</option>
          </select>
          <input
            type="number"
            min={1}
            max={5}
            value={clefLine}
            onChange={(e) => onClefLineChange(Number(e.target.value))}
            className="w-10 bg-transparent border border-white/30 rounded px-1 text-sm text-center text-white"
          />
        </div>

        <div className="flex-1" />
        {status === "ready" && !sessionId && (
          <span className="text-white/80 text-sm">
            queue the page from the classifier, or start a session to correct it
            first
          </span>
        )}
        {/* Only the error panel below covers status==="error" (it replaces the
            iframe); a failure from "encode batch" leaves the classifier up, so
            it needs to be said here or the click looks like a no-op. */}
        {status !== "error" && error && (
          <span
            className="text-red-100 text-xs font-mono max-w-sm truncate"
            title={error}
          >
            {error}
          </span>
        )}
        {queue.length > 0 && (
          <span className="text-white/80 text-sm">
            {queue.length} page{queue.length > 1 ? "s" : ""} queued
          </span>
        )}
        {/* Staging-time queueing now happens inside the IC iframe ("queue
            page" on the classifier's staging screen). This button is only for
            the interactive path — queuing a page after a live session. */}
        {sessionId && (
          <button
            onClick={handleQueuePage}
            disabled={finalizing || queuedNames.has(img?.name ?? "")}
            className="px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {queuedNames.has(img?.name ?? "") ? "queued" : "queue page"}
          </button>
        )}
        <button
          onClick={handleEncodeBatch}
          disabled={queue.length === 0 || finalizing}
          className="px-6 py-2 bg-[#1D3335] text-white border border-white/30 rounded-xl hover:opacity-90 cursor-pointer font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {finalizing ? "finalising..." : `encode batch (${queue.length})`}
        </button>
      </div>

      {/* canvas */}
      <div className="relative flex-1 min-h-[750px] bg-[#1D3335] mx-6 rounded-2xl flex flex-col overflow-hidden">
        {/* mothra#220 DL-1: the fabricated placeholder bbox grid (no real
            predict run for this page -- generate_bboxes()'s fallback) must
            never look like real detections. Overlaid, not just a normal
            list item, so it can't be scrolled past or missed. */}
        {synthetic && icUrl && (
          <div className="absolute top-0 left-0 right-0 z-10 bg-yellow-400 text-[#1D3335] text-sm font-semibold text-center py-1.5">
            SYNTHETIC — no prediction ran for this page; the boxes below are a
            placeholder grid, not real detections
          </div>
        )}
        {/* IC editor area */}
        <div className="flex-1 min-h-0 flex items-stretch justify-stretch overflow-hidden">
          {images.length === 0 ? (
            // Reached by navigating to step 1 once every selected page is
            // already encoded (each one is past step 1, so pendingIcImages()
            // filters them all out) - the classifier has nothing to stage, so
            // say which case this is instead of showing an empty canvas.
            <div className="flex-1 flex flex-col items-center justify-center gap-3 text-center px-8">
              <p className="text-white/70 text-sm">
                {usedImageCount === 0
                  ? "no pages are selected for this project yet"
                  : `every selected page (${usedImageCount}) has already been classified and encoded`}
              </p>
              <p className="text-white/40 text-xs max-w-md">
                {usedImageCount === 0
                  ? "select images on the project page and run detection first."
                  : "there's nothing left to classify here — carry on with correction (step 3), or select more pages on the project page."}
              </p>
              <button
                onClick={onBack}
                className="mt-1 px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold"
              >
                back to project
              </button>
            </div>
          ) : status === "error" ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center px-8">
              <p className="text-red-300 text-sm">
                couldn't start the interactive classifier
              </p>
              <p className="text-white/50 text-xs font-mono max-w-lg break-words">
                {error}
              </p>
              <p className="text-white/40 text-xs">
                is the IC service running on its port? (see CLAUDE.md)
              </p>
            </div>
          ) : icUrl ? (
            <iframe
              key={icUrl}
              src={icUrl}
              title={`Interactive Classifier — ${img?.name ?? ""}`}
              className="flex-1 w-full border-0"
            />
          ) : (
            <div className="flex-1 flex items-center justify-center text-white/50 text-sm">
              starting classifier…
            </div>
          )}
        </div>

        {/* filmstrip for page selection */}
        {images.length > 1 && (
          <div className="flex items-center justify-center px-6 pb-2 pt-2 gap-4">
            <div className="flex items-center justify-center gap-3">
              <button
                onClick={() => setCurrentIdx((i) => i - 1)}
                disabled={currentIdx === 0}
                className="text-white text-xl hover:opacity-70 disabled:opacity-20 cursor-pointer"
              >
                &lt;
              </button>
              {visibleImages.map((thumb, i) => {
                const globalIdx = start + i;
                const active = globalIdx === currentIdx;
                const queued = queuedNames.has(thumb.name);
                return (
                  <button
                    key={thumb.id}
                    onClick={() => setCurrentIdx(globalIdx)}
                    className={`relative w-12 aspect-square rounded-lg overflow-hidden flex-shrink-0 cursor-pointer transition-all
                      ${active ? "ring-2 ring-white ring-offset-2 ring-offset-[#1D3335]" : "opacity-50 hover:opacity-80"}`}
                  >
                    <AuthImage
                      src={`/api/images/${thumb.id}`}
                      alt={thumb.name}
                      className="w-full h-full object-cover"
                    />
                    {queued && (
                      <span className="absolute top-0.5 right-0.5 w-4 h-4 flex items-center justify-center rounded-full bg-green-500 text-white text-[10px] leading-none">
                        ✓
                      </span>
                    )}
                  </button>
                );
              })}
              <button
                onClick={() => setCurrentIdx((i) => i + 1)}
                disabled={currentIdx === images.length - 1}
                className="text-white text-xl hover:opacity-70 disabled:opacity-20 cursor-pointer"
              >
                &gt;
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
