import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { ProjectImage } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { AuthImage } from "../shared/AuthImage";

interface InteractiveClassifierProps {
  images: ProjectImage[];
  projectId: number | null;
  onEncodeBatch: (pairs: { xmlFile: File; imageFile: File }[]) => void;
  clefShape: "C" | "F";
  onClefShapeChange: (s: "C" | "F") => void;
  clefLine: number;
  onClefLineChange: (n: number) => void;
}

const stemOf = (name: string) => name.replace(/\.[^.]+$/, "");

export default function InteractiveClassifier({
  images,
  projectId,
  onEncodeBatch,
  clefShape,
  onClefShapeChange,
  clefLine,
  onClefLineChange
}: InteractiveClassifierProps) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [icUrl, setIcUrl] = useState<string | null>(null);
  const [icOrigin, setIcOrigin] = useState<string | null>(null);
  // Set only once the user finishes IC's create-session screen (the iframe
  // posts it back); until then there's nothing to encode.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "starting" | "ready" | "error">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  const [encoding, setEncoding] = useState(false);
  const [queue, setQueue] = useState<{ xmlFile: File; imageFile: File }[]>([]);
  // Set when IC's in-iframe "auto-export" fires — there's no "queue page"
  // click on that path, so we run the queue logic ourselves once state settles.
  const [autoQueueRequested, setAutoQueueRequested] = useState(false);

  // Shared training set (built-in presets + uploaded GameraXML) applied to
  // every page by "queue all available" — picked once at the parent level so
  // the user doesn't re-select it in each page's IC iframe.
  const [availablePresets, setAvailablePresets] = useState<string[]>([]);
  const [trainingPresets, setTrainingPresets] = useState<string[]>([]);
  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);
  const [showTrainingPanel, setShowTrainingPanel] = useState(false);
  // Progress of a "queue all available" run: null when idle.
  const [queueAll, setQueueAll] = useState<{ done: number; total: number } | null>(
    null,
  );
  // Bumped to force the IC iframe to remount (and re-run its ic:ready
  // handshake) when the batch training set changes while a create-session page
  // is open — see the reload effect below.
  const [reloadNonce, setReloadNonce] = useState(0);

  const queuedNames = useMemo(
    () => new Set(queue.map((p) => p.imageFile.name)),
    [queue],
  );
  const totalTrainingSets = trainingPresets.length + trainingFiles.length;
  const unqueuedCount = images.filter((im) => !queuedNames.has(im.name)).length;

  const img = images[currentIdx];
  // Guards against a slow /ic/start response landing after the user has
  // already switched pages — only the latest request may set state.
  const startSeq = useRef(0);

  // Latest batch training selection, readable from the message handler below
  // without re-subscribing the listener whenever the selection changes.
  const trainingRef = useRef({ presets: trainingPresets, files: trainingFiles });
  useEffect(() => {
    trainingRef.current = { presets: trainingPresets, files: trainingFiles };
  }, [trainingPresets, trainingFiles]);

  // Reload the iframe once if a create-session page is currently open (staged,
  // no session started yet) so a just-changed batch training set shows there
  // immediately — the remount re-runs the ic:ready handshake and re-pulls the
  // selection. Called from the training-set handlers rather than an effect so
  // it only fires on a real user change (not page navigation, which already
  // remounts the iframe via icUrl) and stays out of the resumed/live-session
  // case (sessionId set), where reloading would blow away in-progress work.
  const reloadOpenCreateSession = useCallback(() => {
    if (icUrl && sessionId == null) setReloadNonce((n) => n + 1);
  }, [icUrl, sessionId]);

  // Stage a fresh page + bboxes in IC whenever the selected page changes.
  useEffect(() => {
    if (!img || projectId == null) return;
    const seq = ++startSeq.current;
    setStatus("starting");
    setError(null);
    setIcUrl(null);
    setIcOrigin(null);
    setSessionId(null);

    apiFetch(`/api/projects/${projectId}/ic/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageName: img.name }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
        return r.json();
      })
      .then((data) => {
        if (seq !== startSeq.current) return; // superseded by a newer page
        setIcUrl(data.ic_url);
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
      // batch-level training set (if any) so the user doesn't have to re-pick
      // it on every page. The iframe seeds its own inputs from this and won't
      // overwrite a selection already made there.
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
      if (data?.type === "ic:session-created" && typeof data.sessionId === "string") {
        setSessionId(data.sessionId);
      }
      // "Trust the classifier" path: IC ran classify + skipped the interactive
      // step. Adopt the session and flag it for auto-queuing (handled by the
      // effect below, not here, to avoid a not-yet-flushed sessionId and a
      // stale handleQueuePage closure).
      if (data?.type === "ic:auto-export" && typeof data.sessionId === "string") {
        setSessionId(data.sessionId);
        setAutoQueueRequested(true);
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [icOrigin]);

  // Load the built-in training-set presets once so the parent-level picker can
  // offer the same list IC's create-session screen shows.
  useEffect(() => {
    apiFetch("/api/ic/training-presets")
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => setAvailablePresets(Array.isArray(list) ? list : []))
      .catch(() => setAvailablePresets([]));
  }, []);

  // Presets are mutually exclusive here for the same reason they are in the
  // IC iframe's own picker: checking one unchecks the rest. Kept as an array
  // because the batch prefill and the API both take a list, and an uploaded
  // set can still be combined with a preset — only preset-vs-preset is
  // exclusive.
  const togglePreset = (name: string, checked: boolean) => {
    setTrainingPresets(checked ? [name] : []);
    reloadOpenCreateSession();
  };

  // Turn IC's GameraXML (base64) + a project image into the {xmlFile, imageFile}
  // pair the encode-batch flow consumes. Shared by the interactive "queue page"
  // and the batch "queue all available" paths.
  const buildPair = useCallback(
    async (image: ProjectImage, xmlBase64: string) => {
      const xmlBytes = Uint8Array.from(atob(xmlBase64), (c) => c.charCodeAt(0));
      const xmlFile = new File([xmlBytes], `${stemOf(image.name)}.xml`, {
        type: "application/xml",
      });
      const imgResp = await apiFetch(`/api/images/${image.id}`);
      if (!imgResp.ok) throw new Error(`image fetch failed (${imgResp.status})`);
      const blob = await imgResp.blob();
      const imageFile = new File([blob], image.name, {
        type: blob.type || "image/png",
      });
      return { xmlFile, imageFile };
    },
    [],
  );

  const handleQueuePage = useCallback(async () => {
    if (!sessionId || !img) return;
    setEncoding(true);
    setError(null);
    try {
      // 1. Finalise the IC session → GameraXML.
      const r = await apiFetch(`/api/ic/${sessionId}/complete`, {
        method: "POST",
      });
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      const data = await r.json();

      // 2. Build the pair and hand it to the encode flow; advance to the next
      //    un-queued page.
      const pair = await buildPair(img, data.xml_base64);
      setQueue((prev) => [...prev, pair]);
      const nextIdx = images.findIndex((im, idx) => idx > currentIdx && !queuedNames.has(im.name));
      if (nextIdx !== -1) setCurrentIdx(nextIdx);
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setEncoding(false);
    }
  }, [sessionId, img, buildPair, images, currentIdx, queuedNames]);

  // "Queue all available": classify every not-yet-queued page with the shared
  // training set (server-side, no per-page iframe) and add each to the encode
  // queue. Runs sequentially so the IC service isn't hammered and progress is
  // legible. Requires a non-empty training set (classify needs a training pool).
  const handleQueueAll = useCallback(async () => {
    if (projectId == null || totalTrainingSets === 0) return;
    const pending = images.filter((im) => !queuedNames.has(im.name));
    if (pending.length === 0) return;
    setError(null);
    setQueueAll({ done: 0, total: pending.length });
    try {
      for (let i = 0; i < pending.length; i++) {
        const image = pending[i];
        const form = new FormData();
        form.append("imageName", image.name);
        if (trainingPresets.length > 0)
          form.append("training_presets", JSON.stringify(trainingPresets));
        trainingFiles.forEach((f) => form.append("training_files", f));
        const r = await apiFetch(`/api/projects/${projectId}/ic/auto-queue`, {
          method: "POST",
          body: form,
        });
        if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
        const data = await r.json();
        const pair = await buildPair(image, data.xml_base64);
        setQueue((prev) =>
          prev.some((p) => p.imageFile.name === pair.imageFile.name)
            ? prev
            : [...prev, pair],
        );
        setQueueAll({ done: i + 1, total: pending.length });
      }
    } catch (err) {
      setError(String((err as Error).message ?? err));
    } finally {
      setQueueAll(null);
    }
  }, [projectId, totalTrainingSets, images, queuedNames, trainingPresets, trainingFiles, buildPair]);

  // Run the queue path for an auto-exported page once the session id and
  // current page have settled. Gated on the flag (reset immediately) so it
  // fires exactly once per auto-export, and skips a page already queued.
  useEffect(() => {
    if (autoQueueRequested && sessionId && img && !queuedNames.has(img.name)) {
      setAutoQueueRequested(false);
      handleQueuePage();
    }
  }, [autoQueueRequested, sessionId, img, queuedNames, handleQueuePage]);

  const handleEncodeBatch = useCallback(() => {
    if (queue.length === 0) return;
    onEncodeBatch(queue);
  }, [queue, onEncodeBatch]);

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
            onChange={e => onClefShapeChange(e.target.value as "C" | "F")}
            className="bg-transparent border border-white/30 rounded px-1 text-sm cursor-pointer text-white"
          >
            <option value="C">C</option>
            <option value="F">F</option>
          </select>
          <input
            type="number" min={1} max={5} value={clefLine}
            onChange={e => onClefLineChange(Number(e.target.value))}
            className="w-10 bg-transparent border border-white/30 rounded px-1 text-sm text-center text-white"
          />
        </div>

        {/* Shared training set picker — presets + uploaded GameraXML applied
            to every page by "queue all available". */}
        <div className="relative">
          <button
            onClick={() => setShowTrainingPanel((v) => !v)}
            className="flex items-center gap-2 px-3 py-1.5 rounded-lg border border-white/30 text-white text-sm hover:bg-white/10 cursor-pointer"
          >
            training set
            {totalTrainingSets > 0 && (
              <span className="bg-white text-[#1D3335] rounded-full px-1.5 text-xs font-semibold">
                {totalTrainingSets}
              </span>
            )}
          </button>
          {showTrainingPanel && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setShowTrainingPanel(false)}
              />
              <div className="absolute z-50 top-full mt-2 left-0 w-80 bg-white rounded-xl shadow-2xl p-4 text-[#1D3335] text-sm">
                <h3 className="font-semibold">training set for all pages</h3>
                <p className="text-xs text-[#1D3335]/60 mb-3">
                  Applied to every page when you "queue all available".
                </p>

                <div className="mb-3">
                  <span className="mb-1 block text-xs font-medium text-[#1D3335]/70">
                    Presets
                  </span>
                  {availablePresets.length === 0 ? (
                    <span className="text-xs text-[#1D3335]/50">
                      No presets available.
                    </span>
                  ) : (
                    <div className="space-y-1 max-h-40 overflow-y-auto">
                      {availablePresets.map((name) => (
                        <label
                          key={name}
                          className="flex items-center gap-2 cursor-pointer"
                        >
                          <input
                            type="checkbox"
                            checked={trainingPresets.includes(name)}
                            onChange={(e) => togglePreset(name, e.target.checked)}
                          />
                          <span className="truncate">{name}</span>
                        </label>
                      ))}
                    </div>
                  )}
                </div>

                <label className="block">
                  <span className="mb-1 block text-xs font-medium text-[#1D3335]/70">
                    Upload GameraXML (.xml)
                  </span>
                  <input
                    type="file"
                    accept=".xml"
                    multiple
                    onChange={(e) => {
                      setTrainingFiles(Array.from(e.target.files ?? []));
                      reloadOpenCreateSession();
                    }}
                    className="block w-full text-xs text-[#1D3335]/70 file:mr-2 file:cursor-pointer file:rounded-lg file:border-0 file:bg-[#4AADAA] file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:file:opacity-90"
                  />
                </label>
                {trainingFiles.length > 0 && (
                  <span className="mt-1 block text-xs text-[#1D3335]/60">
                    {trainingFiles.length} file
                    {trainingFiles.length === 1 ? "" : "s"} selected
                  </span>
                )}

                <p className="mt-3 text-xs text-[#1D3335]/60">
                  {totalTrainingSets > 0
                    ? `${totalTrainingSets} training set${totalTrainingSets === 1 ? "" : "s"} will classify each page.`
                    : "Pick presets or upload sets to enable queue all."}
                </p>
              </div>
            </>
          )}
        </div>

        <div className="flex-1" />
        {status === "ready" && !sessionId && (
          <span className="text-white/80 text-sm">
            queue the page from the classifier, or start a session to correct it first
          </span>
        )}
        { queue.length > 0 && (
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
            disabled={encoding || queuedNames.has(img?.name ?? "")}
            className="px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {encoding ? "queuing..." : queuedNames.has(img?.name ?? "") ? "queued" : "queue page"}
          </button>
        )}
        <button
          onClick={handleEncodeBatch}
          disabled={queue.length === 0}
          className="px-6 py-2 bg-[#1D3335] text-white border border-white/30 rounded-xl hover:opacity-90 cursor-pointer font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
        >
          encode batch ({queue.length})
        </button>
      </div>

      {/* canvas */}
      <div className="flex-1 min-h-[750px] bg-[#1D3335] mx-6 rounded-2xl flex flex-col overflow-hidden">
        {/* IC editor area */}
        <div className="flex-1 min-h-0 flex items-stretch justify-stretch overflow-hidden">
          {images.length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-white/40 text-sm italic">
              no images selected
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
              key={`${icUrl}:${reloadNonce}`}
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
          <div className="flex items-center px-6 pb-2 pt-2 gap-4">
            {/* left spacer keeps the thumbnails centered opposite the button */}
            <div className="flex-1" />
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
            {/* queue every not-yet-queued page with the shared training set */}
            <div className="flex-1 flex justify-end">
              <button
                onClick={handleQueueAll}
                disabled={
                  queueAll !== null || totalTrainingSets === 0 || unqueuedCount === 0
                }
                title={
                  totalTrainingSets === 0
                    ? "Pick a training set first (top bar)"
                    : unqueuedCount === 0
                      ? "Every page is already queued"
                      : "Classify and queue every remaining page with the training set"
                }
                className="px-4 py-1.5 bg-white text-[#1D3335] rounded-lg text-sm font-semibold hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap"
              >
                {queueAll
                  ? `queuing ${queueAll.done}/${queueAll.total}…`
                  : `queue all available (${unqueuedCount})`}
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
