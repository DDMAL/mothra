import { useState, useRef, useCallback, useEffect } from "react";
import type { AnnotationSet } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import TruncatedName from "../shared/TruncatedName";
import { useZoomPan, MAX_SCALE } from "../../hooks/useZoomPan";

interface BBox {
    cls: number;
    cx: number;
    cy: number;
    bw: number;
    bh: number
}

type ViewState =
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; imageUrl: string; boxes: BBox[]; rawYolo: string };

const PALETTE = ["#4AADAA", "#FFA500", "#E87BF7", "#F76B6B", "#6BF7A5", "#F7E16B"];

function parseYolo(txt: string): BBox[] {
    return txt
        .trim()
        .split("\n")
        .filter(Boolean)
        .map((line) => {
            const [cls, cx, cy, bw, bh] = line.trim().split(/\s+/).map(Number);
            return { cls, cx, cy, bw, bh };
        });
}

interface Props {
    set: AnnotationSet;
    projectId: number;
    onClose: () => void;
}


export default function AnnotationViewerModal({ set, projectId, onClose }: Props) {
    const [viewState, setViewState] = useState<ViewState>({ status: "loading" });
    const [activeTab, setActiveTab] = useState<"image" | "raw">("image");
    const [copied, setCopied] = useState(false);
    const [copyFailed, setCopyFailed] = useState(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);
    const zoom = useZoomPan();

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

    const handleDownload = (text: string, extension: string, mimeType = "text/plain;charset=utf-8") => {
        const blob = new Blob([text], { type: mimeType });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `${set.imageName.replace(/\.[^.]+$/, "")}.${extension}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };

    const drawOverlay = useCallback(() => {
        if (viewState.status !== "ready") return;
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img) return;
        const dw = img.clientWidth;
        const dh = img.clientHeight;
        // Backing buffer sized for MAX_SCALE (not just the 100%-zoom display
        // size) so the ancestor's CSS `transform: scale()` zoom stays crisp
        // all the way up instead of blurrily upscaling a buffer that was
        // only ever rendered at 100% 
        canvas.width = dw * MAX_SCALE;
        canvas.height = dh * MAX_SCALE;
        // <canvas> is a replaced element: with no explicit CSS size it
        // displays at its backing-buffer resolution (canvas.width/height),
        // not shrunk to fit its container — pin the *displayed* size to
        // dw/dh explicitly so the bigger MAX_SCALE backing buffer doesn't
        // render at its full (huge) intrinsic size.
        canvas.style.width = `${dw}px`;
        canvas.style.height = `${dh}px`;
        const ctx = canvas.getContext("2d")!;
        ctx.scale(MAX_SCALE, MAX_SCALE);
        ctx.clearRect(0, 0, dw, dh);
        viewState.boxes.forEach((b) => {
            const color = PALETTE[b.cls % PALETTE.length];
            const x = (b.cx - b.bw / 2) * dw;
            const y = (b.cy - b.bh / 2) * dh;
            const w = b.bw * dw;
            const h = b.bh * dh;
            ctx.fillStyle = color + "20";
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
        });
    }, [viewState]);

    useEffect(() => {
        if (viewState.status === "ready" && imgRef.current?.complete) {
            drawOverlay();
        }
    }, [viewState.status, drawOverlay]);

    useEffect(() => {
        return () => {
            if (viewState.status === "ready") URL.revokeObjectURL(viewState.imageUrl);
        };
    }, [viewState]);

    useEffect(() => {
        if (!set.imageSrc) {
            setViewState({ status: "error", message: "No image source for this annotation." });
            return;
        }
        Promise.all([
            apiFetch(`/api/projects/${projectId}/annotations/${set.id}`)
                .then((r) => (r.ok ? r.json() : Promise.reject("annotation fetch failed"))),
            apiFetch(set.imageSrc)
                .then((r) => (r.ok ? r.blob(): Promise.reject("image fetch failed"))),
        ])
            .then(([ann, blob]) => {
                const imageUrl = URL.createObjectURL(blob);
                const rawYolo = (ann as { yoloTxt: string }).yoloTxt ?? "";
                const boxes = parseYolo(rawYolo);
                setViewState({ status: "ready", imageUrl, boxes, rawYolo });
            })
            .catch(() => 
                setViewState({ status: "error", message: "Failed to load annotation view." }),
        );
    }, []);

    return (
        <>
            <div className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60" onClick={onClose} />
            <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
                {/* header */}
                <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
                    <TruncatedName
                        name={set.imageName}
                        className="font-mono text-sm text-[#1D3335] font-semibold flex-1 min-w-0"
                    />
                    <div className="flex items-center gap-1 bg-[#1D3335]/10 rounded-full p-0.5 shrink-0">
                        {(["image", "raw"] as const).map((t) => (
                            <button
                                key={t}
                                onClick={() => setActiveTab(t)}
                                className={`px-3 py-1 rounded-full text-xs font-mono transition-colors cursor-pointer ${
                                    activeTab === t
                                        ? "bg-[#1D3335] text-white"
                                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                                }`}
                            >
                                {t === "image" ? "image overlay" : "raw"}
                            </button>
                        ))}
                    </div>
                    {set.detectionCount !== undefined && (
                        <span className="text-xs text-[#1D3335]/60">
                            {set.detectionCount} detection{set.detectionCount !== 1 ? "s" : ""}
                        </span>
                    )}
                    <button
                        onClick={onClose}
                        className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
                    >
                        ✕
                    </button>
                </div>

                {/* content */}
                <div className="flex-1 min-h-0 overflow-auto">
                    {viewState.status === "loading" ? (
                        <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                            loading…
                        </div>
                    ) : viewState.status === "error" ? (
                        <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                            {viewState.message}
                        </div>
                    ) : activeTab === "raw" ? (
                        <div className="p-4">
                            <div className="flex items-center justify-end gap-3 mb-2">
                                <button
                                    onClick={() => handleDownload(viewState.rawYolo, "txt")}
                                    className="text-xs font-mono text-[#1D3335]/60 hover:text-[#1D3335] cursor-pointer"
                                >
                                    download
                                </button>
                                <button
                                    onClick={() => handleCopyText(viewState.rawYolo)}
                                    className="text-xs font-mono text-[#1D3335]/60 hover:text-[#1D3335] cursor-pointer"
                                >
                                    {copyFailed ? "copy failed" : copied ? "copied" : "copy"}
                                </button>
                            </div>
                            <pre className="bg-[#1D3335] text-white/80 text-xs font-mono rounded-xl p-4 overflow-auto h-[min(65vh,650px)] whitespace-pre">
                                {viewState.rawYolo}
                            </pre>
                        </div>
                    ) : (
                        <div className="p-4">
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
                                        alt={set.imageName}
                                        className="block max-h-[min(65vh,650px)] max-w-full select-none"
                                        draggable={false}
                                        onLoad={drawOverlay}
                                    />
                                    <canvas
                                        ref={canvasRef}
                                        className="absolute inset-0 pointer-events-none"
                                    />
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
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}