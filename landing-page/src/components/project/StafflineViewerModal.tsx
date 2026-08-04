import { useState, useRef, useCallback, useEffect } from "react";
import { type StafflineSet, type JsomrLineRecord } from "../../types";
import { apiFetch } from "../../lib/apiFetch";

type ViewState =
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; imageUrl: string; records: JsomrLineRecord[] };

// Matches AnnotationViewerModal's class-color palette, extended here to
// color-code by stave_id instead of YOLO class -- keeps the "index into a
// fixed palette" visual language consistent across viewer modals.
const PALETTE = ["#4AADAA", "#FFA500", "#E87BF7", "#F76B6B", "#6BF7A5", "#F7E16B"];
const UNASSIGNED_COLOR = "#888888";
const ANOMALY_COLOR = "#FF3B30";

interface Props {
    detection: StafflineSet;
    projectId: number;
    onClose: () => void;
    label?: string;
}

export default function StafflineViewerModal({ detection, projectId, onClose, label }: Props) {
    const [viewState, setViewState] = useState<ViewState>({ status: "loading" });
    const [notesOpen, setNotesOpen] = useState(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);

    const anomalousStaveIds =
        viewState.status === "ready"
            ? new Set(
                  viewState.records
                      .filter((r) => r.rhythm_status && r.stave_id !== null)
                      .map((r) => r.stave_id as number),
              )
            : new Set<number>();

    const drawOverlay = useCallback(() => {
        if (viewState.status !== "ready") return;
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

        viewState.records.forEach((r) => {
            const isAnomalous = r.stave_id !== null && anomalousStaveIds.has(r.stave_id);
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
    }, [viewState, anomalousStaveIds]);

    useEffect(() => {
        let disposed = false;
        let imageUrl: string | undefined;
        setViewState({ status: "loading" });

        if (!detection.imageSrc) {
            setViewState({ status: "error", message: "No image source for this detection." });
            return () => {
                disposed = true;
            };
        }
        Promise.all([
            apiFetch(`/api/projects/${projectId}/stafflines/${detection.id}`)
                .then((r) => (r.ok ? r.json() : Promise.reject("staffline fetch failed"))),
            apiFetch(detection.imageSrc)
                .then((r) => (r.ok ? r.blob() : Promise.reject("image fetch failed"))),
        ])
            .then(([data, blob]) => {
                imageUrl = URL.createObjectURL(blob);
                if (disposed) {
                    URL.revokeObjectURL(imageUrl);
                    return;
                }
                setViewState({
                    status: "ready",
                    imageUrl,
                    // jsomrJson is a native array (JSONB column) -- no JSON.parse needed.
                    records: (data as { jsomrJson: JsomrLineRecord[] }).jsomrJson ?? [],
                });
            })
            .catch(() => {
                if (!disposed) {
                    setViewState({ status: "error", message: "Failed to load staffline detection view." });
                }
            });
        return () => {
            disposed = true;
            if (imageUrl) URL.revokeObjectURL(imageUrl);
        };
    }, [detection.id, detection.imageSrc, projectId]);

    const anomalyNotes =
        viewState.status === "ready"
            ? Array.from(
                  new Map(
                      viewState.records
                          .filter((r) => r.rhythm_status && r.stave_id !== null)
                          .map((r) => [r.stave_id, r.rhythm_status as string]),
                  ),
              )
            : [];

    return (
        <>
            {/* Matches AnnotationViewerModal's overlay/panel shell verbatim (not Modal.tsx --
                Modal.tsx only supports a vertically-centered, fixed-size dialog, not this
                viewport-stretched layout both image viewers need). */}
            <div className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60" onClick={onClose} />
            <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
                <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
                    <p className="font-mono text-sm text-[#1D3335] font-semibold truncate flex-1">
                        {label ?? detection.imageName}
                    </p>
                    <span className="text-xs text-[#1D3335]/60">
                        {detection.staveCount ?? 0} stave{detection.staveCount !== 1 ? "s" : ""}
                    </span>
                    {anomalousStaveIds.size > 0 && (
                        <span className="text-xs font-semibold text-[#FF3B30]">
                            {anomalousStaveIds.size} flagged for review
                        </span>
                    )}
                    <button
                        onClick={onClose}
                        className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
                    >
                        ✕
                    </button>
                </div>
                <div className="flex-1 min-h-0 overflow-auto">
                    {viewState.status === "loading" ? (
                        <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                            loading…
                        </div>
                    ) : viewState.status === "error" ? (
                        <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                            {viewState.message}
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
                                dashed = interpolated · gray = unassigned · red = flagged for review
                            </p>
                            {anomalyNotes.length > 0 && (
                                <div className="mt-4 w-full">
                                    <button
                                        onClick={() => setNotesOpen((o) => !o)}
                                        className="text-[#1D3335]/60 text-sm hover:text-[#1D3335] cursor-pointer select-none"
                                    >
                                        {notesOpen ? "v" : ">"} view flagged staves ({anomalyNotes.length})
                                    </button>
                                    {notesOpen && (
                                        <div className="mt-2 bg-[#1D3335] rounded-xl h-32 w-full overflow-y-auto p-3">
                                            {anomalyNotes.map(([staveId, status]) => (
                                                <div key={staveId} className="text-white/70 text-xs font-mono leading-5">
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
