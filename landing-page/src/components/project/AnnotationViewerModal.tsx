import { useState, useRef, useCallback, useEffect } from "react";
import type { AnnotationSet } from "../../types";
import { authHeaders } from "../../hooks/useAuth";

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
    | { status: "ready"; imageUrl: string; boxes: BBox[] };

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
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);

    const drawOverlay = useCallback(() => {
        if (viewState.status !== "ready") return;
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img) return;
        const dw = img.clientWidth;
        const dh = img.clientHeight;
        canvas.width = dw;
        canvas.height = dh;
        const ctx = canvas.getContext("2d")!;
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
            fetch(`/api/projects/${projectId}/annotations/${set.id}`, {
                headers: authHeaders(),
            }).then((r) => (r.ok ? r.json() : Promise.reject("annotation fetch failed"))),
            fetch(set.imageSrc, {
                headers: authHeaders(),
            }).then((r) => (r.ok ? r.blob(): Promise.reject("image fetch failed"))),
        ])
            .then(([ann, blob]) => {
                const imageUrl = URL.createObjectURL(blob);
                const boxes = parseYolo((ann as { yoloTxt: string }).yoloTxt ?? "");
                setViewState({ status: "ready", imageUrl, boxes });
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
                    <p className="font-mono text-sm text-[#1D3335] font-semibold truncate flex-1">
                        {set.imageName}
                    </p>
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
                    ) : (
                        <div className="p-4 flex justify-center">
                            <div className="relative inline-block">
                                <img
                                    ref={imgRef}
                                    src={viewState.imageUrl}
                                    alt={set.imageName}
                                    className="block max-w-full"
                                    onLoad={drawOverlay}
                                />
                                <canvas
                                    ref={canvasRef}
                                    className="absolute inset-0 pointer-events-none"
                                />
                            </div>
                        </div>
                    )}
                </div>
            </div>
        </>
    );
}