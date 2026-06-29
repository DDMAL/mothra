import { useState } from "react";
import type { MeiFile } from "../../types";

interface Props {
    originalFiles: MeiFile[];
    correctedFiles: MeiFile[];
    onClose: () => void;
}

export default function MeiCompareModal({ originalFiles, correctedFiles, onClose }: Props) {
    const [selectedIndex, setSelectedIndex] = useState(0);

    // pair by id
    const pairs = correctedFiles.map((cf) => ({
        name: cf.name,
        original: originalFiles.find((f) => f.id === cf.id),
        corrected: cf,
    }));

    // fallback: no pairs matched by id, show originals only
    const activePair = pairs[selectedIndex] ?? null;

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
                    onClick={() => setSelectedIndex(i)}
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

            <button
                onClick={onClose}
                className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
            >
                ✕
            </button>
            </div>

            {/* two-pane body */}
            {activePair ? (
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
            ) : (
            <div className="flex-1 flex items-center justify-center text-[#1D3335]/50 text-sm">
                No corrected files to compare.
            </div>
            )}
        </div>
        </>
    );
}