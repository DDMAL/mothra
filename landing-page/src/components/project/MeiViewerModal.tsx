import { useState, useRef, useCallback, useEffect } from "react";
import type { Project, MeiFile } from "../../types";
import { authHeaders } from "../../hooks/useAuth";
import { downloadBlob } from "../../utils/download";

type Tab = "text" | "score";

interface Zone {
    id: string;
    zoneType: "staff" | "other";
    ulx: number;
    uly: number;
    lrx: number;
    lry: number;
}

type ScoreState = 
    | { status: "idle" }
    | { status: "loading" }
    | { status: "error"; message: string }
    | { status: "ready"; imageUrl: string; zones: Zone[]; surfaceW: number; surfaceH: number };

function parseMeiZones(
    xml: string,
): { zones: Zone[]; surfaceW: number; surfaceH: number } | null {
    try {
        const doc = new DOMParser().parseFromString(xml, "application/xml");
        const surface = doc.querySelector("surface");
        if (!surface) return null;
        const surfaceW = parseInt(surface.getAttribute("lrx") ?? "0");
        const surfaceH = parseInt(surface.getAttribute("lry") ?? "0");
        if (!surfaceW || !surfaceH) return null;
        const zones: Zone[] = [];
        surface.querySelectorAll("zone").forEach((el) => {
            zones.push({
                id: el.getAttribute("xml:id") ?? "",
                zoneType: el.getAttribute("type") === "staff" ? "staff" : "other",
                ulx: parseInt(el.getAttribute("ulx") ?? "0"),
                uly: parseInt(el.getAttribute("uly") ?? "0"),
                lrx: parseInt(el.getAttribute("lrx") ?? "0"),
                lry: parseInt(el.getAttribute("lry") ?? "0"),
            });
        });
        return { zones, surfaceW, surfaceH };
    } catch {
        return null;
    }
}

interface Props {
    file: MeiFile;
    project: Project;
    onClose: () => void;
}

export default function MeiViewerModal({ file, project, onClose }: Props) {
    const [tab, setTab] = useState<Tab>("text");
    const [scoreState, setScoreState] = useState<ScoreState>({ status: "idle" });
    const fetched = useRef(false);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const imgRef = useRef<HTMLImageElement>(null);

    const handleExport = () => {
        downloadBlob(
            new Blob([file.xmlContent ?? ""], { type: "application/xml" }),
            file.name,
        );
    }

    const drawOverlay = useCallback(() => {
        if (scoreState.status !== "ready") return;
        const canvas = canvasRef.current;
        const img = imgRef.current;
        if (!canvas || !img) return;
        const { zones, surfaceW, surfaceH } = scoreState;
        const dw = img.clientWidth;
        const dh = img.clientHeight;
        canvas.width = dw;
        canvas.height = dh;
        const ctx = canvas.getContext("2d")!;
        ctx.clearRect(0, 0, dw, dh);
        const sx = dw / surfaceW;
        const sy = dh / surfaceH;
        zones.forEach((zone) => {
            const isStaff = zone.zoneType === "staff";
            ctx.fillStyle = isStaff ? "rgba(74,173,170,0.12)" : "rgba(255,165,0,0.12)";
            ctx.strokeStyle = isStaff ? "#4AADAA" : "#FFA500";
            ctx.lineWidth = isStaff ? 2 : 1;
            const x = zone.ulx * sx;
            const y = zone.uly * sy;
            const w = (zone.lrx - zone.ulx) * sx;
            const h = (zone.lry - zone.uly) * sy;
            ctx.fillRect(x, y, w, h);
            ctx.strokeRect(x, y, w, h);
        });
    }, [scoreState]);

    useEffect(() => {
        if (scoreState.status === "ready" && imgRef.current?.complete) {
            drawOverlay();
        }
    }, [scoreState.status, drawOverlay]);

    useEffect(() => {
        return () => {
            if (scoreState.status === "ready") URL.revokeObjectURL(scoreState.imageUrl);
        };
    }, [scoreState]);

    const openScoreTab = () => {
        setTab("score");
        if (fetched.current) return;
        fetched.current = true;

        if (!file.imageName) {
            setScoreState({ status: "error", message: "No image is associated with this MEI file." });
            return;
        }
        if (!file.xmlContent) {
            setScoreState({ status: "error", message: "MEI file has no content."});
            return;
        }
        const parsed = parseMeiZones(file.xmlContent);
        if (!parsed) {
            setScoreState({ status: "error", message: "Could not parse zone coordinates from MEI." });
            return;
        }
        const imgRecord = project.images.find((i) => i.name === file.imageName);
        if (!imgRecord) {
            setScoreState({ status: "error", message: "Associated image not found in project." });
            return;
        }
        setScoreState({ status: "loading" });
        fetch(`/api/images/${imgRecord.id}`, {
            headers: authHeaders()
        })
            .then((r) => (r.ok ? r.blob() : Promise.reject()))
            .then((blob) => {
                const url = URL.createObjectURL(blob);
                setScoreState({ status: "ready", imageUrl: url, ...parsed });
            })
            .catch(() => setScoreState({ status: "error", message: "Failed to load score view." }));
    };

    return (
        <>
            <div className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60" onClick={onClose} />
            <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
                {/* header */}
                <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
                <p className="font-mono text-sm text-[#1D3335] font-semibold truncate flex-1">
                    {file.name}
                </p>
                <div className="flex gap-1 bg-white/40 rounded-xl p-1">
                    <button
                    onClick={() => setTab("text")}
                    className={`px-4 py-1 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                        tab === "text"
                        ? "bg-white text-[#4AADAA]"
                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                    }`}
                    >
                    Text
                    </button>
                    <button
                    onClick={openScoreTab}
                    className={`px-4 py-1 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                        tab === "score"
                        ? "bg-white text-[#4AADAA]"
                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                    }`}
                    >
                    Score
                    </button>
                </div>
                <button
                    onClick={handleExport}
                    className="px-4 py-1.5 bg-white text-[#1D3335] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
                >
                    export
                </button>
                <button
                    onClick={onClose}
                    className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
                >
                    ✕
                </button>
                </div>

                {/* content */}
                <div className="flex-1 min-h-0 overflow-auto">
                {tab === "text" ? (
                    <pre className="text-[#1D3335]/80 text-xs font-mono h-full whitespace-pre-wrap p-6">
                    {file.xmlContent ?? "(no content)"}
                    </pre>
                ) : scoreState.status === "loading" ? (
                    <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                    loading…
                    </div>
                ) : scoreState.status === "error" ? (
                    <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                    {scoreState.message}
                    </div>
                ) : scoreState.status === "ready" ? (
                    <div className="p-4 flex justify-center">
                    <div className="relative inline-block">
                        <img
                        ref={imgRef}
                        src={scoreState.imageUrl}
                        alt={file.imageName ?? "manuscript"}
                        className="block max-w-full"
                        onLoad={drawOverlay}
                        />
                        <canvas
                        ref={canvasRef}
                        className="absolute inset-0 pointer-events-none"
                        />
                    </div>
                    </div>
                ) : null}
                </div>
            </div>
            </>
    );
}