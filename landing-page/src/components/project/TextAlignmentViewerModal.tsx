import { useState, useRef, useCallback, useEffect } from "react";
import { type TextAlignment } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import TruncatedName from "../shared/TruncatedName";
import { useZoomPan, MAX_SCALE } from "../../hooks/useZoomPan";

interface SylBox {
    syl: string;
    ul: [number, number];
    lr: [number, number];
}

type ViewState =
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; imageUrl: string; boxes: SylBox[]; logText: string; prettyJson: string };

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

/**
 * Group syllable boxes into reading-order lines for the plaintext side panel.
 * Boxes are clustered by vertical position (within `lineSpacing * 0.6` of the
 * first box in each cluster) since `syl_boxes` from the alignment JSON have
 * no line grouping of their own, then each line is sorted left-to-right.
 */
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

/** Flatten reconstructed lines into the plain-text form used for copy/download. */
function plainTextLinesToString(lines: PlainTextToken[][]): string {
    return lines.map((line) => line.map((t) => t.syl).join(" ")).join("\n");
}

/** Convert a syllable box's image-space `ul`/`lr` corners to on-screen pixels. */
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
    const [activeTab, setActiveTab] = useState<"image" | "json">("image");
    const [logsOpen, setLogsOpen] = useState(false);
    const [textPanelOpen, setTextPanelOpen] = useState(false);
    const [copied, setCopied] = useState(false);
    const [copyFailed, setCopyFailed] = useState(false);
    const [hoveredBoxIndex, setHoveredBoxIndex] = useState<number | null>(null);
    const [imgSize, setImgSize] = useState<{ dw: number; dh: number; nw: number; nh: number } | null>(null);
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
        a.download = `${(label ?? alignment.imageName).replace(/\.[^.]+$/, "")}.${extension}`;
        document.body.appendChild(a);
        a.click();
        a.remove();
        URL.revokeObjectURL(url);
    };
    
    const measure = useCallback(() => {
        const img = imgRef.current;
        if (!img || !img.naturalWidth) return;
        setImgSize({
            dw: img.clientWidth,
            dh: img.clientHeight,
            nw: img.naturalWidth,
            nh: img.naturalHeight,
        });
    }, []);

    useEffect(() => {
        const img = imgRef.current;
        if (!img || viewState.status !== "ready") return;
        const ro = new ResizeObserver(measure);
        ro.observe(img);
        return () => ro.disconnect();
    }, [measure, viewState.status]);

    const drawOverlay = useCallback(() => {
        if (viewState.status !== "ready" || !imgSize) return;
        const canvas = canvasRef.current;
        if (!canvas) return;
        const { dw, dh, nw, nh } = imgSize;
        // Backing buffer sized for MAX_SCALE (not just the 100%-zoom display
        // size) so the ancestor's CSS `transform: scale()` zoom stays crisp
        // all the way up instead of blurrily upscaling a buffer that was
        // only ever rendered at 100% — see issue #139's zoomed-in blur report.
        canvas.width = dw * MAX_SCALE;
        canvas.height = dh * MAX_SCALE;
        // <canvas> is a replaced element: with no explicit CSS size it
        // displays at its backing-buffer resolution (canvas.width/height),
        // not shrunk to fit its container — the `w-full h-full` Tailwind
        // classes on the element pin the *displayed* size to its container
        // (which matches dw/dh) so the bigger MAX_SCALE backing buffer
        // doesn't render at its full (huge) intrinsic size.
        const scaleX = dw / nw;
        const scaleY = dh / nh;
        const ctx = canvas.getContext("2d")!;
        ctx.scale(MAX_SCALE, MAX_SCALE);
        ctx.clearRect(0, 0, dw, dh);
        // Divided by the *current* zoom (not MAX_SCALE) so the label stays a
        // roughly constant, legible on-screen size instead of growing right
        // along with the (now-crisp) boxes and overlapping its neighbors —
        // box outlines are still meant to visibly grow when zoomed in, only
        // the text needs to stay put.
        const fontPx = 11 / zoom.scale;
        const labelYFloor = 10 / zoom.scale;
        viewState.boxes.forEach((b, i) => {
            const { x, y, w, h } = boxScreenRect(b, scaleX, scaleY);
            const hovered = i === hoveredBoxIndex;
            ctx.fillStyle = BOX_COLOR + (hovered ? "45" : "20");
            ctx.strokeStyle = hovered ? "#1D3335" : BOX_COLOR;
            ctx.lineWidth = hovered ? 1.25 : 0.75;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
            ctx.fillStyle = hovered ? "#1D3335" : BOX_COLOR;
            ctx.font = `${hovered ? "bold " : ""}${fontPx}px monospace`;
            ctx.fillText(b.syl, x, Math.max(labelYFloor, y - 2));
        });
    }, [viewState, hoveredBoxIndex, imgSize, zoom.scale]);

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
                    // alignmentJson comes back compact (no whitespace) from the
                    // backend's json.dumps(alignment) — re-stringify with
                    // indentation purely for the "json" tab's readability.
                    prettyJson: JSON.stringify(parsed, null, 2),
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
                    <div className="flex items-center gap-1 bg-[#1D3335]/10 rounded-full p-0.5 shrink-0">
                        {(["image", "json"] as const).map((t) => (
                            <button
                                key={t}
                                onClick={() => setActiveTab(t)}
                                className={`px-3 py-1 rounded-full text-xs font-mono transition-colors cursor-pointer ${
                                    activeTab === t
                                        ? "bg-[#1D3335] text-white"
                                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                                }`}
                            >
                                {t === "image" ? "image overlay" : "json"}
                            </button>
                        ))}
                    </div>
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
                    ) : activeTab === "json" ? (
                        <div className="p-4">
                            <div className="flex items-center justify-end gap-3 mb-2">
                                <button
                                    onClick={() => handleDownload(viewState.prettyJson, "json", "application/json")}
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
                                                onLoad={measure}
                                            />
                                            <canvas
                                                ref={canvasRef}
                                                className="absolute inset-0 w-full h-full pointer-events-none"
                                            />
                                            {imgSize && viewState.status === "ready" && (() => {
                                                const scaleX = imgSize.dw / imgSize.nw;
                                                const scaleY = imgSize.dh / imgSize.nh;
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
                                                        onClick={() => handleDownload(plainText, "txt")}
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
                                                        {copyFailed ? "copy failed" : copied ? "copied" : "copy"}
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