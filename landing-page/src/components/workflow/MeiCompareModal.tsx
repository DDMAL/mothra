import { useState, useMemo } from "react";
import type { MeiFile, ProjectImage } from "../../types";
import { diffZones } from "../../utils/meiZoneDiff";
import MeiImageDiffView from "./MeiImageDiffView";

interface Props {
    originalFiles: MeiFile[];
    correctedFiles: MeiFile[];
    onClose: () => void;
    projectImages?: ProjectImage[];
}

export default function MeiCompareModal({ originalFiles, correctedFiles, onClose, projectImages }: Props) {
    const [selectedIndex, setSelectedIndex] = useState(0);
    const [viewMode, setViewMode] = useState<"xml" | "image-overlay">("xml");

    const pairs = correctedFiles.map((cf) => ({
        name: cf.name,
        original: originalFiles.find((f) => f.id === cf.id),
        corrected: cf,
    }));

    const activePair = pairs[selectedIndex] ?? null;

    const diff = useMemo(() => {
        if (viewMode !== "image-overlay") return null;
        const orig = activePair?.original?.xmlContent;
        const corr = activePair?.corrected?.xmlContent;
        if (!orig || !corr) return null;
        return diffZones(orig, corr);
    }, [viewMode, activePair]);

    const imageId = useMemo(() => {
        const name = activePair?.corrected?.imageName;
        if (!name || !projectImages?.length) return null;
        return projectImages.find((img) => img.name === name)?.id ?? null;
    }, [projectImages, activePair]);

    return (
        <>
        {/* overlay */}
        <div className="fixed inset-0 z-50 bg-black/60" onClick={onClose} />

        {/* panel */}
        <div className="fixed z-50 top-6 bottom-6 left-1/2 -translate-x-1/2 w-[calc(100vw-3rem)] max-w-7xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">

            {/* header */}
            <div className="flex items-center gap-3 px-6 py-3 border-b border-[#1D3335]/20 shrink-0 flex-wrap">
            <span className="font-semibold text-[#1D3335] text-sm mr-auto">
                compare before &amp; after
            </span>

            {/* file tabs (only shown when multiple files) */}
            {pairs.length > 1 && (
                <div className="flex gap-1 bg-white/40 rounded-xl p-1">
                {pairs.map((p, i) => (
                    <button
                    key={p.corrected.id}
                    onClick={() => { setSelectedIndex(i); setViewMode("xml"); }}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer truncate max-w-[140px] ${
                        i === selectedIndex
                        ? "bg-white text-[#4AADAA]"
                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                    }`}
                    >
                    {p.name}
                    </button>
                ))}
                </div>
            )}

            {/* view mode toggle (only when images available) */}
            {projectImages && (
                <div className="flex gap-1 rounded-xl bg-white/30 p-1">
                {(["xml", "image-overlay"] as const).map((mode) => (
                    <button
                    key={mode}
                    onClick={() => setViewMode(mode)}
                    className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors cursor-pointer ${
                        mode === viewMode
                        ? "bg-white text-[#4AADAA]"
                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                    }`}
                    >
                    {mode === "xml" ? "XML" : "Image Overlay"}
                    </button>
                ))}
                </div>
            )}

            <button
                onClick={onClose}
                className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
            >
                ✕
            </button>
            </div>

            {/* body */}
            {activePair ? (
            viewMode === "image-overlay" ? (
                diff ? (
                <MeiImageDiffView
                    imageId={imageId}
                    diff={diff}
                    imageName={activePair.corrected.imageName ?? activePair.name}
                />
                ) : (
                <div className="flex-1 flex items-center justify-center text-[#1D3335]/50 text-sm italic">
                    no XML content available for overlay
                </div>
                )
            ) : (
            <div className="flex flex-1 min-h-0 divide-x divide-[#1D3335]/20">
                {/* left: before */}
                <div className="flex-1 min-w-0 flex flex-col">
                <p className="text-xs font-semibold text-[#1D3335]/50 px-4 pt-3 pb-1 shrink-0">
                    before correction
                </p>
                <pre className="flex-1 overflow-auto text-[#1D3335]/80 text-xs font-mono whitespace-pre-wrap p-4 pt-0">
                    {activePair.original?.xmlContent ?? "(no original snapshot)"}
                </pre>
                </div>

                {/* right: after */}
                <div className="flex-1 min-w-0 flex flex-col">
                <p className="text-xs font-semibold text-[#4AADAA] px-4 pt-3 pb-1 shrink-0">
                    after correction
                </p>
                <pre className="flex-1 overflow-auto text-[#1D3335]/80 text-xs font-mono whitespace-pre-wrap p-4 pt-0">
                    {activePair.corrected.xmlContent ?? "(no corrected content)"}
                </pre>
                </div>
            </div>
            )
            ) : (
            <div className="flex-1 flex items-center justify-center text-[#1D3335]/50 text-sm">
                No corrected files to compare.
            </div>
            )}
        </div>
        </>
    );
}
