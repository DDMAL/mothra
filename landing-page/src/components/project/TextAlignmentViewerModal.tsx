import { useState, useRef, useCallback, useEffect } from "react";
import { type View, type TextAlignment } from "../../types";
import { apiFetch } from "../../lib/apiFetch";

interface SylBox {
    syl: string;
    ul: [number, number];
    lr: [number, number];
}

type ViewState =
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; imageUrl: string; boxes: SylBox[]; logText: string };

const BOX_COLOR = "#4AADAA";

interface Props {
    alignment: TextAlignment;
    projectId: number;
    onClose: () => void;
    label?: string;
}

export default function TextAlignmentViewerModal({ alignment, projectId, onClose, label }: Props) {
    const [viewState, setViewState] = useState<ViewState>({ status: "loading"});
    const [logsOpen, setLogsOpen] = useState(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);

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
        viewState.boxes.forEach((b) => {
            const x = b.ul[0] * scaleX;
            const y = b.ul[1] * scaleY;
            const w = (b.lr[0] - b.ul[0]) * scaleX;
            const h = (b.lr[1] - b.ul[1]) * scaleY;
            ctx.fillStyle = BOX_COLOR + "20";
            ctx.strokeStyle = BOX_COLOR;
            ctx.lineWidth = 1.5;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = BOX_COLOR;
            ctx.font = "11px monospace";
            ctx.fillText(b.syl, x, Math.max(10, y - 2));
        });
    }, [viewState]);

    useEffect(() => {
        if (!alignment.imageSrc) {
            setViewState({ status: "error", message: "No image source for this alignment." });
            return;
        }
        Promise.all([
            apiFetch(`/api/projects/${projectId}/text-alignments/${alignment.id}`)
                .then((r) => (r.ok ? r.json() : Promise.reject("alignment fetch failed"))),
            apiFetch(alignment.imageSrc)
                .then((r) => (r.ok ? r.blob() : Promise.reject("image fetch failed"))),
        ])
            .then(([data, blob]) => {
                const imageUrl = URL.createObjectURL(blob);
                const parsed = JSON.parse((data as { alignmentJson: string }).alignmentJson);
                setViewState({
                    status: "ready",
                    imageUrl,
                    boxes: parsed.syl_boxes ?? [],
                    logText: (data as { logText?: string }).logText ?? "",
                });
            })
            .catch(() =>
                setViewState({ status: "error", message: "Failed to load text alignment view." }),
            );
    }, []);

    return (
        <>
            <div className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60" onClick={onClose} />
            <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
                <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
                    <p className="font-mono text-sm text-[#1D3335] font-semibold truncate flex-1">
                        {label ?? alignment.imageName}
                    </p>
                    <span className="text-xs text-[#1D3335]/60">
                        {alignment.syllableCount} syllable{alignment.syllableCount !== 1 ? "s" : ""}
                    </span>
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
                                    alt={alignment.imageName}
                                    className="block max-w-full"
                                    onLoad={drawOverlay}
                                />
                                <canvas
                                    ref={canvasRef}
                                    className="absolute inset-0 pointer-events-none"
                                />
                            </div>
                            <div className="mt-4 w-full">
                                <button
                                    onClick={() => setLogsOpen((o) => !o)}
                                    className="text-[#1D3335]/60 text-sm hover:text-[#1D3335] cursor-pointer select-none"
                                >
                                    {logsOpen ? "v" : ">"} view logs
                                </button>
                                {logsOpen && (
                                    <div className="mt-2 bg-[#1D3335] rounded-xl h-32 w-full overflow-y-auto p-3">
                                        {viewState.logText ? (
                                            viewState.logText.split("\n").map((line, i) => (
                                                <div key={i} className="text-white/70 text-xs font-mono leading-5">
                                                    {line}
                                                </div>
                                            ))
                                        ) : (
                                            <div className="text-white/30 text-xs font-mono">
                                                no logs recorded for this run
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}