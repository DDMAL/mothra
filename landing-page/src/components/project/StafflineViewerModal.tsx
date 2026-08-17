import { useState, useRef, useCallback, useEffect, useMemo } from "react";
import { type StafflineSet, type JsomrLineRecord } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { AuthImage } from "../shared/AuthImage";
import { computeRhythmGaps } from "../../lib/rhythmGaps";
import RhythmChart from "./RhythmChart";

type ViewState =
  | { status: "loading" }
  | { status: "error"; message: string }
  | {
      status: "ready";
      imageUrl: string;
      records: JsomrLineRecord[];
      prettyJson: string;
    };

// Matches AnnotationViewerModal's class-color palette, extended here to
// color-code by stave_id instead of YOLO class -- keeps the "index into a
// fixed palette" visual language consistent across viewer modals.
const PALETTE = [
  "#4AADAA",
  "#FFA500",
  "#E87BF7",
  "#F76B6B",
  "#6BF7A5",
  "#F7E16B",
];
const UNASSIGNED_COLOR = "#888888";
// Blue rather than red -- red overlay lines were getting lost against the
// red rubrics/staff-lines common in these manuscripts (see the pages this
// viewer is actually used on).
const ANOMALY_COLOR = "#2563EB";

interface Props {
  detection: StafflineSet;
  projectId: number;
  onClose: () => void;
  label?: string;
  // Called once a previewed interpolation is accepted and persisted as a
  // new staffline_detections row -- lets the caller (StafflinesTab) merge
  // it into project.stafflines and swap this modal's `detection` prop to
  // show it, without a full project refetch.
  onAccepted?: (newSet: StafflineSet) => void;
}

export default function StafflineViewerModal({
  detection,
  projectId,
  onClose,
  label,
  onAccepted,
}: Props) {
  const [viewState, setViewState] = useState<ViewState>({ status: "loading" });
  const [notesOpen, setNotesOpen] = useState(false);
  const [tab, setTab] = useState<"overlay" | "rhythm" | "raw" | "classifier">("overlay");
  const [copied, setCopied] = useState(false);
  const [copyFailed, setCopyFailed] = useState(false);
  // Non-null once interpolate-preview has returned -- presence alone means
  // "we're reviewing a preview", not yet persisted. interpolate_missing is
  // "not yet validated across the corpus" (staff-finding/dox/STATUS.md),
  // hence review-before-persist rather than a one-click apply.
  const [interpolatePreview, setInterpolatePreview] = useState<
    JsomrLineRecord[] | null
  >(null);
  const [previewLoading, setPreviewLoading] = useState(false);
  const [confirmLoading, setConfirmLoading] = useState(false);
  const [interpolateError, setInterpolateError] = useState<string | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);


  const handleCopyText = (text: string) => {
    navigator.clipboard
      .writeText(text)
      .then(() => setCopied(true))
      .catch(() => setCopyFailed(true))
      .finally(() => {
        setTimeout(() => {
          setCopied(false);
          setCopyFailed(false);
        }, 1500);
      });
  };

  const handleDownload = (
    text: string,
    extension: string,
    mimeType = "text/plain;charset=utf-8",
  ) => {
    const blob = new Blob([text], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${(label ?? detection.imageName).replace(/\.[^.]+$/, "")}.${extension}`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const rhythmSummary = useMemo(
    () =>
      viewState.status === "ready"
        ? computeRhythmGaps(viewState.records)
        : null,
    [viewState],
  );

  const visibleTabs = useMemo(() => {
    const tabs: Array<"overlay" | "rhythm" | "raw" | "classifier"> = ["overlay"];
    if (rhythmSummary) tabs.push("rhythm");
    if (detection.hasClassifierImage) tabs.push("classifier");
    tabs.push("raw");
    return tabs;
  }, [rhythmSummary, detection.hasClassifierImage]);

  // While previewing, the overlay draws the previewed (unpersisted) records
  // instead of the real detection's -- everything downstream of this single
  // switch (anomaly flags, the canvas overlay, the notes list) just works
  // off whichever set is active. Memoized so drawOverlay's own useCallback
  // deps don't churn on every render.
  const activeRecords: JsomrLineRecord[] = useMemo(
    () =>
      interpolatePreview ??
      (viewState.status === "ready" ? viewState.records : []),
    [interpolatePreview, viewState],
  );

  const anomalousStaveIds = useMemo(
    () =>
      new Set(
        activeRecords
          .filter((r) => r.rhythm_status && r.stave_id !== null)
          .map((r) => r.stave_id as number),
      ),
    [activeRecords],
  );

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !img.naturalWidth) return;
    const dw = img.clientWidth;
    const dh = img.clientHeight;
    canvas.width = dw;
    canvas.height = dh;
    const scaleX = dw / img.naturalWidth;
    const scaleY = dh / img.naturalHeight;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, dw, dh);

    activeRecords.forEach((r) => {
      const isAnomalous =
        r.stave_id !== null && anomalousStaveIds.has(r.stave_id);
      const color = isAnomalous
        ? ANOMALY_COLOR
        : r.stave_id !== null
          ? PALETTE[r.stave_id % PALETTE.length]
          : UNASSIGNED_COLOR;
      const dash: number[] = r.source === "interpolated" ? [4, 3] : [];

      if (r.bounding_box) {
        const { ulx, uly, lrx, lry } = r.bounding_box;
        ctx.setLineDash([]);
        ctx.strokeStyle = color + "40";
        ctx.lineWidth = 1;
        ctx.strokeRect(
          ulx * scaleX,
          uly * scaleY,
          (lrx - ulx) * scaleX,
          (lry - uly) * scaleY,
        );
      }

      const { x_start, y_values } = r.centerline_page;
      ctx.setLineDash(dash);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      y_values.forEach((y, i) => {
        const px = (x_start + i) * scaleX;
        const py = y * scaleY;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });
    ctx.setLineDash([]);
  }, [activeRecords, anomalousStaveIds]);

  // The image itself doesn't reload when only interpolatePreview changes
  // (same detection, same imageUrl), so re-draw explicitly rather than
  // relying solely on the <img>'s onLoad, which won't fire again.
  useEffect(() => {
    drawOverlay();
  }, [drawOverlay]);

  useEffect(() => {
    let disposed = false;
    let imageUrl: string | undefined;
    setViewState({ status: "loading" });
    // A fresh detection (e.g. onAccepted swapping in the just-confirmed
    // one) means any prior preview is stale -- clear it rather than
    // showing last detection's preview over this one's image.
    setInterpolatePreview(null);
    setInterpolateError(null);

    if (!detection.imageSrc) {
      setViewState({
        status: "error",
        message: "No image source for this detection.",
      });
      return () => {
        disposed = true;
      };
    }
    Promise.all([
      apiFetch(`/api/projects/${projectId}/stafflines/${detection.id}`).then(
        (r) => (r.ok ? r.json() : Promise.reject("staffline fetch failed")),
      ),
      apiFetch(detection.imageSrc).then((r) =>
        r.ok ? r.blob() : Promise.reject("image fetch failed"),
      ),
    ])
      .then(([data, blob]) => {
        imageUrl = URL.createObjectURL(blob);
        if (disposed) {
          URL.revokeObjectURL(imageUrl);
          return;
        }
        // jsomrJson is a native array (JSONB column) -- no JSON.parse needed.
        const records =
          (data as { jsomrJson: JsomrLineRecord[] }).jsomrJson ?? [];
        setViewState({
          status: "ready",
          imageUrl,
          records,
          // Pretty-printed purely for the "raw" tab's readability, same
          // as TextAlignmentViewerModal's prettyJson.
          prettyJson: JSON.stringify(records, null, 2),
        });
      })
      .catch(() => {
        if (!disposed) {
          setViewState({
            status: "error",
            message: "Failed to load staffline detection view.",
          });
        }
      });
    return () => {
      disposed = true;
      if (imageUrl) URL.revokeObjectURL(imageUrl);
    };
  }, [detection.id, detection.imageSrc, projectId]);

  const anomalyNotes = Array.from(
    new Map(
      activeRecords
        .filter((r) => r.rhythm_status && r.stave_id !== null)
        .map((r) => [r.stave_id, r.rhythm_status as string]),
    ),
  );

  const handlePreviewInterpolation = async () => {
    setInterpolateError(null);
    setPreviewLoading(true);
    try {
      const r = await apiFetch(
        `/api/projects/${projectId}/stafflines/${detection.id}/interpolate-preview`,
        { method: "POST" },
      );
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(
          (d as { detail?: string }).detail || "interpolation preview failed",
        );
      }
      const data = await r.json();
      setInterpolatePreview(
        (data as { jsomrJson: JsomrLineRecord[] }).jsomrJson ?? [],
      );
    } catch (e) {
      setInterpolateError((e as Error).message);
    } finally {
      setPreviewLoading(false);
    }
  };

  const handleDiscardInterpolation = () => {
    setInterpolatePreview(null);
    setInterpolateError(null);
  };

  const handleAcceptInterpolation = async () => {
    setInterpolateError(null);
    setConfirmLoading(true);
    try {
      const r = await apiFetch(
        `/api/projects/${projectId}/stafflines/${detection.id}/interpolate-confirm`,
        { method: "POST" },
      );
      if (!r.ok) {
        const d = await r.json().catch(() => ({}));
        throw new Error(
          (d as { detail?: string }).detail || "interpolation failed",
        );
      }
      const newSet = (await r.json()) as StafflineSet;
      setInterpolatePreview(null);
      onAccepted?.(newSet);
    } catch (e) {
      setInterpolateError((e as Error).message);
    } finally {
      setConfirmLoading(false);
    }
  };

  return (
    <>
      {/* Matches AnnotationViewerModal's overlay/panel shell verbatim (not Modal.tsx --
                Modal.tsx only supports a vertically-centered, fixed-size dialog, not this
                viewport-stretched layout both image viewers need). */}
      <div
        className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60"
        onClick={onClose}
      />
      <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
        <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
          <p className="font-mono text-sm text-[#1D3335] font-semibold truncate flex-1">
            {label ?? detection.imageName}
          </p>
          <span className="text-xs text-[#1D3335]/60">
            {detection.staveCount ?? 0} stave
            {detection.staveCount !== 1 ? "s" : ""}
          </span>
          {anomalousStaveIds.size > 0 && (
            <span className="text-xs font-semibold text-[#2563EB]">
              {anomalousStaveIds.size} flagged for review
            </span>
          )}
          {interpolatePreview ? (
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono text-[#2563EB] italic">
                reviewing preview
              </span>
              <button
                onClick={handleDiscardInterpolation}
                disabled={confirmLoading}
                className="px-3 py-1 rounded-full text-xs font-mono border border-[#1D3335]/30 text-[#1D3335]/70 hover:text-[#1D3335] cursor-pointer disabled:opacity-50"
              >
                discard
              </button>
              <button
                onClick={handleAcceptInterpolation}
                disabled={confirmLoading}
                className="px-3 py-1 rounded-full text-xs font-mono bg-[#2563EB] text-white hover:opacity-90 cursor-pointer disabled:opacity-50"
              >
                {confirmLoading ? "accepting..." : "accept"}
              </button>
            </div>
          ) : (
            <>
              {anomalousStaveIds.size > 0 && viewState.status === "ready" && (
                <button
                  onClick={handlePreviewInterpolation}
                  disabled={previewLoading}
                  className="px-3 py-1 rounded-full text-xs font-mono border border-[#2563EB] text-[#2563EB] hover:bg-[#2563EB]/10 cursor-pointer disabled:opacity-50"
                >
                  {previewLoading
                    ? "computing preview..."
                    : "interpolate missing lines"}
                </button>
              )}
              {viewState.status === "ready" && (
                <div className="flex bg-[#1D3335]/10 rounded-full p-0.5 text-xs font-mono">
                  {visibleTabs.map((t) => (
                    <button
                      key={t}
                      onClick={() => setTab(t)}
                      className={`px-3 py-1 rounded-full cursor-pointer transition-colors ${
                        tab === t
                          ? "bg-[#1D3335] text-white"
                          : "text-[#1D3335]/60 hover:text-[#1D3335]"
                      }`}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              )}
            </>
          )}
          <button
            onClick={onClose}
            className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
          >
            ✕
          </button>
        </div>
        {interpolateError && (
          <div
            role="alert"
            className="px-6 py-2 bg-[#FF3B30]/10 text-[#FF3B30] text-xs font-mono border-b border-[#1D3335]/10"
          >
            {interpolateError}
          </div>
        )}
        <div className="flex-1 min-h-0 overflow-auto">
          {viewState.status === "loading" ? (
            <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
              loading…
            </div>
          ) : viewState.status === "error" ? (
            <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
              {viewState.message}
            </div>
          ) : !interpolatePreview && tab === "rhythm" && rhythmSummary ? (
            <div className="p-4">
              <RhythmChart summary={rhythmSummary} />
            </div>
          ) : !interpolatePreview && tab === "classifier" ? (
            <div className="p-4 flex flex-col items-center">
              <AuthImage
                src={`/api/projects/${projectId}/stafflines/${detection.id}/classifier-image`}
                alt={`${detection.imageName} — paco-classifier stafflines layer`}
                className="block max-w-full rounded-xl"
              />
              <p className="mt-2 text-[#1D3335]/50 text-[11px] font-mono">
                paco-classifier's stafflines-only layer — the image the stave
                model actually detected boxes against, not the raw page.
              </p>
            </div>
          ) : !interpolatePreview && tab === "raw" ? (
            <div className="p-4">
              <div className="flex items-center justify-end gap-3 mb-2">
                <button
                  onClick={() =>
                    handleDownload(
                      viewState.prettyJson,
                      "json",
                      "application/json",
                    )
                  }
                  className="text-xs font-mono text-[#1D3335]/60 hover:text-[#1D3335] cursor-pointer"
                >
                  download
                </button>
                <button
                  onClick={() => handleCopyText(viewState.prettyJson)}
                  className="text-xs font-mono text-[#1D3335]/60 hover:text-[#1D3335] cursor-pointer"
                >
                  {copyFailed ? "copy failed" : copied ? "copied" : "copy"}
                </button>
              </div>
              <pre className="bg-[#1D3335] text-white/80 text-xs font-mono rounded-xl p-4 overflow-auto h-[min(65vh,650px)] whitespace-pre">
                {viewState.prettyJson}
              </pre>
            </div>
          ) : (
            <div className="p-4 flex flex-col items-center">
              <div className="relative inline-block">
                <img
                  ref={imgRef}
                  src={viewState.imageUrl}
                  alt={detection.imageName}
                  className="block max-w-full"
                  onLoad={drawOverlay}
                />
                <canvas
                  ref={canvasRef}
                  className="absolute inset-0 pointer-events-none"
                />
              </div>
              <p className="mt-2 text-[#1D3335]/50 text-[11px] font-mono">
                {interpolatePreview
                  ? "previewing unpersisted interpolated lines -- accept or discard above"
                  : "dashed = interpolated · gray = unassigned · blue = flagged for review"}
              </p>
              {anomalyNotes.length > 0 && (
                <div className="mt-4 w-full">
                  <button
                    onClick={() => setNotesOpen((o) => !o)}
                    className="text-[#1D3335]/60 text-sm hover:text-[#1D3335] cursor-pointer select-none"
                  >
                    {notesOpen ? "v" : ">"} view flagged staves (
                    {anomalyNotes.length})
                  </button>
                  {notesOpen && (
                    <div className="mt-2 bg-[#1D3335] rounded-xl h-32 w-full overflow-y-auto p-3">
                      {anomalyNotes.map(([staveId, status]) => (
                        <div
                          key={staveId}
                          className="text-white/70 text-xs font-mono leading-5"
                        >
                          stave {staveId}: {status}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </div>
      </div>
    </>
  );
}
