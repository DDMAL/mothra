import { useState, useRef, useCallback, useEffect } from "react";
import { type TextAlignment } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import TruncatedName from "../shared/TruncatedName";
import { useZoomPan } from "../../hooks/useZoomPan";

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

interface PlainTextToken {
    index: number;
    syl: string;
}

function reconstructPlainText(boxes: SylBox[], lineSpacing: number): PlainTextToken[][] {
    if (boxes.length === 0) return [];
    const threshold = (lineSpacing > 0 ? lineSpacing : 40) * 0.6;
    const indexed = boxes.map((box, index) => ({ box, index }));
    const sorted = [...indexed].sort((a, b) => a.box.ul[1] - b.box.ul[1]);
    const grouped: (typeof indexed)[] = [];
    for (const item of sorted) {
        const line = grouped[grouped.length - 1];
        if (line && Math.abs(item.box.ul[1] - line[0].box.ul[1]) < threshold) {
            line.push(item);
        } else {
            grouped.push([item]);
        }
    }
    return grouped.map((line) =>
        [...line]
            .sort((a, b) => a.box.ul[0] - b.box.ul[0])
            .map(({ box, index }) => ({ index, syl: box.syl })),
    );
}

function plainTextLinesToString(lines: PlainTextToken[][]): string {
    return lines.map((line) => line.map((t) => t.syl).join(" ")).join("\n");
}

function boxScreenRect(b: SylBox, scaleX: number, scaleY: number) {
    return {
        x: b.ul[0] * scaleX,
        y: b.ul[1] * scaleY,
        w: (b.lr[0] - b.ul[0]) * scaleX,
        h: (b.lr[1] - b.ul[1]) * scaleY,
    };
}

export default function TextAlignmentViewerModal({ alignment, projectId, onClose, label }: Props) {
    const [viewState, setViewState] = useState<ViewState>({ status: "loading"});
    const [logsOpen, setLogsOpen] = useState(false);
    const [textPanelOpen, setTextPanelOpen] = useState(false);
    const [copied, setCopied] = useState(false);
    const [hoveredBoxIndex, setHoveredBoxIndex] = useState<number | null>(null);
    const [imgReady, setImgReady] = useState(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const zoom = useZoomPan();

    const handleCopyText = (text: string) => {
        navigator.clipboard.writeText(text).then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1500);
        });
    };

    const handleDownloadText = (text: string) => {
        const blob = new Blob([text], { type: "text/plain;charset=utf-8" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${(label ?? alignment.imageName).replace(/\.[^.]+$/, "")}.txt`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };
    
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
        viewState.boxes.forEach((b, i) => {
            const { x, y, w, h } = boxScreenRect(b, scaleX, scaleY);
            const hovered = i === hoveredBoxIndex;
            ctx.fillStyle = BOX_COLOR + (hovered ? "45" : "20");
            ctx.strokeStyle = hovered ? "#1D3335" : BOX_COLOR;
            ctx.lineWidth = hovered ? 2.5 : 1.5;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = hovered ? "#1D3335" : BOX_COLOR;
            ctx.font = hovered ? "bold 11px monospace" : "11px monospace";
            ctx.fillText(b.syl, x, Math.max(10, y - 2));
        });
    }, [viewState, hoveredBoxIndex]);

    useEffect(() => {
        drawOverlay();
    }, [drawOverlay]);

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
                    <TruncatedName
                        name={label ?? alignment.imageName}
                        className="font-mono text-sm text-[#1D3335] font-semibold flex-1 min-w-0"
                    />
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
                        <div className="p-4">
                            <div className="flex gap-3 items-start">
                                <div className="flex-1 min-w-0 flex flex-col items-center">
                                    <div
                                        ref={zoom.containerRef}
                                        {...zoom.panHandlers}
                                        onDoubleClick={zoom.reset}
                                        className={`relative w-full h-[min(65vh,650px)] overflow-hidden rounded-xl bg-black/5 flex items-center justify-center ${
                                            zoom.isPannable ? (zoom.isDragging ? "cursor-grabbing" : "cursor-grab") : ""
                                        }`}
                                    >
                                        <div className="relative" style={zoom.transformStyle}>
                                            <img
                                                ref={imgRef}
                                                src={viewState.imageUrl}
                                                alt={alignment.imageName}
                                                className="block max-h-[min(65vh,650px)] max-w-full select-none"
                                                draggable={false}
                                                onLoad={() => {
                                                    setImgReady(true);
                                                    drawOverlay();
                                                }}
                                            />
                                            <canvas
                                                ref={canvasRef}
                                                className="absolute inset-0 pointer-events-none"
                                            />
                                            {imgReady && viewState.status === "ready" && imgRef.current && (() => {
                                                const img = imgRef.current;
                                                const scaleX = img.clientWidth / img.naturalWidth;
                                                const scaleY = img.clientHeight / img.naturalHeight;
                                                return viewState.boxes.map((b, i) => {
                                                    const { x, y, w, h } = boxScreenRect(b, scaleX, scaleY);
                                                    return (
                                                        <div
                                                            key={i}
                                                            onMouseEnter={() => setHoveredBoxIndex(i)}
                                                            onMouseLeave={() =>
                                                                // Only clear if this box is still the one hovered — if the
                                                                // cursor already moved into an overlapping box, that box's
                                                                // onMouseEnter may have already fired first, and this
                                                                // shouldn't stomp on it.
                                                                setHoveredBoxIndex((cur) => (cur === i ? null : cur))
                                                            }
                                                            style={{ position: "absolute", left: x, top: y, width: w, height: h }}
                                                        />
                                                    );
                                                });
                                            })()}
                                        </div>
                                        <div
                                            className="absolute bottom-3 right-3 z-10 flex items-center gap-1 bg-white/85 rounded-lg shadow px-1 py-1"
                                            onDoubleClick={(e) => e.stopPropagation()}
                                        >
                                            <button
                                                onClick={zoom.zoomOut}
                                                disabled={!zoom.canZoomOut}
                                                className="w-7 h-7 flex items-center justify-center text-[#1D3335] text-base leading-none rounded hover:bg-[#C8E6E3] disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                                            >
                                                −
                                            </button>
                                            <button
                                                onClick={zoom.reset}
                                                className="px-2 h-7 flex items-center justify-center text-[#1D3335] text-xs font-mono rounded hover:bg-[#C8E6E3] cursor-pointer"
                                            >
                                                {Math.round(zoom.scale * 100)}%
                                            </button>
                                            <button
                                                onClick={zoom.zoomIn}
                                                disabled={!zoom.canZoomIn}
                                                className="w-7 h-7 flex items-center justify-center text-[#1D3335] text-base leading-none rounded hover:bg-[#C8E6E3] disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                                            >
                                                +
                                            </button>
                                        </div>
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

                                <div className="shrink-0 flex h-[min(65vh,650px)]">
                                    <button
                                        onClick={() => setTextPanelOpen((o) => !o)}
                                        title={textPanelOpen ? "Hide plain text" : "Show plain text"}
                                        className="shrink-0 w-6 flex items-center justify-center bg-[#1D3335]/10 hover:bg-[#1D3335]/20 rounded-l-lg text-[#1D3335] text-xs cursor-pointer select-none"
                                    >
                                        {textPanelOpen ? "›" : "‹"}
                                    </button>
                                    {textPanelOpen && (() => {
                                        const lines =
                                            viewState.status === "ready"
                                                ? reconstructPlainText(viewState.boxes, alignment.medianLineSpacing)
                                                : [];
                                        const plainText = plainTextLinesToString(lines);
                                        return (
                                            <div className="w-72 bg-white/60 rounded-r-xl p-3 overflow-y-auto">
                                                <div className="flex items-center justify-end gap-3 mb-2">
                                                    <button
                                                        onClick={() => handleDownloadText(plainText)}
                                                        disabled={!plainText}
                                                        className="text-xs font-mono text-[#1D3335]/60 hover:text-[#1D3335] disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                                                    >
                                                        download
                                                    </button>
                                                    <button
                                                        onClick={() => handleCopyText(plainText)}
                                                        disabled={!plainText}
                                                        className="text-xs font-mono text-[#1D3335]/60 hover:text-[#1D3335] disabled:opacity-30 disabled:cursor-not-allowed cursor-pointer"
                                                    >
                                                        {copied ? "copied" : "copy"}
                                                    </button>
                                                </div>
                                                {lines.length > 0 ? (
                                                    <div className="text-[#1D3335] text-sm whitespace-pre-wrap font-serif">
                                                        {lines.map((line, li) => (
                                                            <div key={li}>
                                                                {line.map((tok, ti) => (
                                                                    <span key={tok.index}>
                                                                        <span
                                                                            className={
                                                                                tok.index === hoveredBoxIndex
                                                                                    ? "bg-[#4AADAA]/50 rounded px-0.5"
                                                                                    : ""
                                                                            }
                                                                        >
                                                                            {tok.syl}
                                                                        </span>
                                                                        {ti < line.length - 1 ? " " : ""}
                                                                    </span>
                                                                ))}
                                                            </div>
                                                        ))}
                                                    </div>
                                                ) : (
                                                    <p className="text-[#1D3335]/40 text-sm font-mono">
                                                        no syllables detected
                                                    </p>
                                                )}
                                            </div>
                                        );
                                    })()}
                                </div>
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}