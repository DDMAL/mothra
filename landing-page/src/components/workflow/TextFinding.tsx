import { useState } from "react";
import type { ProjectImage, TextAlignment } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { AuthImage } from "../shared/AuthImage";
import ProcessingPage from "./ProcessingPage";

interface TextFindingProps {
    images: ProjectImage[];
    projectId: number | null;
    onResult: (alignment: TextAlignment) => void;
    onBack: () => void;
}

export default function TextFinding({ images, projectId, onResult, onBack }: TextFindingProps) {
    const [currentIdx, setCurrentIdx] = useState(0);
    const [running, setRunning] = useState(false);
    const img = images[currentIdx];

    if (running && img && projectId != null) {
        return (
            <ProcessingPage
                onBack={() => setRunning(false)}
                onComplete={() => setRunning(false)}
                streamRequest={(signal) => apiFetch(`/api/projects/${projectId}/text-finding/run?image_name=${encodeURIComponent(img.name)}`,
                { method: "POST", signal },)}
                onResult={(ev) => {
                    onResult({
                        id: ev.alignment_id,
                        imageName: img.name,
                        medianLineSpacing: ev.text_alignment?.median_line_spacing ?? 0,
                        syllableCount: ev.text_alignment?.syl_boxes?.length ?? 0,
                    });
                }}
            />
        );
    }

    const VISIBLE = 5;
    const half = Math.floor(VISIBLE / 2);
    const start = Math.max(0, Math.min(currentIdx - half, images.length - VISIBLE));
    const visibleImages = images.slice(start, start + VISIBLE);

    return (
        <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col pb-6">
            <div className="flex items-center gap-6 px-8 py-5">
                <h1 className="text-4xl font-bold italic text-white">text finding</h1>
                {images.length > 1 && (
                <span className="text-white/80 text-sm font-mono">
                    page {currentIdx + 1}/{images.length}
                    {img ? ` — ${img.name}` : ""}
                </span>
                )}
                <div className="flex-1" />
                <button
                onClick={onBack}
                className="px-6 py-2 border-2 border-white text-white rounded-xl hover:opacity-90 cursor-pointer font-semibold"
                >
                back to project
                </button>
                <button
                onClick={() => setRunning(true)}
                disabled={!img || projectId == null}
                className="px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
                >
                run text-finding
                </button>
            </div>

            <div className="flex-1 bg-[#1D3335] mx-6 rounded-2xl flex flex-col overflow-hidden">
                <div className="flex-1 flex items-center justify-center overflow-hidden">
                {images.length === 0 ? (
                    <div className="text-white/40 text-sm italic">no images selected</div>
                ) : img ? (
                    <AuthImage src={`/api/images/${img.id}`} alt={img.name} className="max-h-full max-w-full object-contain" />
                ) : null}
                </div>

                {images.length > 1 && (
                <div className="flex items-center px-6 pb-6 pt-4 gap-4">
                    <div className="flex-1 flex items-center justify-center gap-3">
                    <button onClick={() => setCurrentIdx((i) => i - 1)} disabled={currentIdx === 0}
                        className="text-white text-xl hover:opacity-70 disabled:opacity-20 cursor-pointer">
                        &lt;
                    </button>
                    {visibleImages.map((thumb, i) => {
                        const globalIdx = start + i;
                        const active = globalIdx === currentIdx;
                        return (
                        <button key={thumb.id} onClick={() => setCurrentIdx(globalIdx)}
                            className={`relative w-16 aspect-square rounded-lg overflow-hidden flex-shrink-0 cursor-pointer transition-all
                            ${active ? "ring-2 ring-white ring-offset-2 ring-offset-[#1D3335]" : "opacity-50 hover:opacity-80"}`}>
                            <AuthImage src={`/api/images/${thumb.id}`} alt={thumb.name} className="w-full h-full object-cover" />
                        </button>
                        );
                    })}
                    <button onClick={() => setCurrentIdx((i) => i + 1)} disabled={currentIdx === images.length - 1}
                        className="text-white text-xl hover:opacity-70 disabled:opacity-20 cursor-pointer">
                        &gt;
                    </button>
                    </div>
                </div>
                )}
            </div>
        </div>
    )
}