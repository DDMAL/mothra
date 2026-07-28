import { useState, useEffect } from "react";
import type { CantusSource, Project, ProjectImage } from "../../types";
import { apiFetchOrThrow } from "../../lib/apiFetch";
import type { useTextFindingSettings } from "../../hooks/useTextFindingSettings";
import { findFolioConflict } from "../../utils/folio";
import { downloadBlob } from "../../utils/download";

interface CantusSourcePanelProps {
    textFindingSettings: ReturnType<typeof useTextFindingSettings>;
    project: Project,
    onUpdateSourceId: (sourceId: string) => void,
    onSourceLoaded?: (s: CantusSource | null) => void
    imageSubTab: "grid" | "batch";
    batchStartFolio: string;
    batchEndFolio: string;
    onBatchStartFolioChange: (folio: string) => void;
    onBatchEndFolioChange: (folio: string) => void;
    batchFolioSequence: string[];
}

export default function CantusSourcePanel({
    textFindingSettings, project, onUpdateSourceId, onSourceLoaded,
    imageSubTab, batchStartFolio, batchEndFolio, onBatchStartFolioChange, onBatchEndFolioChange,
    batchFolioSequence, 
}: CantusSourcePanelProps) {
    const { ocrOnlyMode, sourceId, folio, patch } = textFindingSettings;
    const [sourceIdInput, setSourceIdInput] = useState(sourceId);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loadedSource, setLoadedSource] = useState<CantusSource | null>(null);
    const [conflict, setConflict] = useState<ProjectImage | null>(null);
    const [exportError, setExportError] = useState<string | null>(null);

    const loadSource = async (id: string) => {
        setLoading(true);
        setError(null);
        try {
            const data: CantusSource = await apiFetchOrThrow(`/api/cantus/source/${id}`).then((r) => r.json());
            setLoadedSource(data);
            onSourceLoaded?.(data);
            patch({ sourceId: data.sourceId, folio: ""});
            onUpdateSourceId(data.sourceId);
        } catch (e) {
            setLoadedSource(null);
            onSourceLoaded?.(null);
            patch({ sourceId: "", folio: "" });
            setError((e as Error).message);
        } finally {
            setLoading(false);
        }
    };

    const handleLoad = async() => {
        const trimmed = sourceIdInput.trim();
        if (trimmed) loadSource(trimmed);
    };

    useEffect(() => {
        if (project.cantusSourceId && !sourceId) {
            setSourceIdInput(project.cantusSourceId);
            loadSource(project.cantusSourceId);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [project.cantusSourceId]);

    return (
        <div className="mb-4 bg-white/10 rounded-xl p-4 flex flex-col gap-3 text-sm text-white max-w-xl">
        <label className="flex items-center gap-2 text-white/80 text-xs">
            <input
            type="checkbox"
            checked={ocrOnlyMode}
            onChange={(e) => patch({ ocrOnlyMode: e.target.checked })}
            className="accent-[#1D3335]"
            />
            OCR only (skip CantusDB alignment)
        </label>

        {!ocrOnlyMode && (
            <div className="flex flex-col gap-3 pl-1">
            <div className="flex items-center gap-2">
                <input
                type="text"
                value={sourceIdInput}
                onChange={(e) => setSourceIdInput(e.target.value)}
                placeholder="CantusDB source ID"
                className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-40"
                />
                <button
                onClick={handleLoad}
                disabled={loading || !sourceIdInput.trim()}
                className="px-3 py-1 border border-white/40 text-white text-xs rounded-lg hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                >
                {loading ? "loading..." : "load"}
                </button>
            </div>
            <a
                href="https://cantusdatabase.org/sources/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/50 text-xs hover:text-white underline w-fit"
            >
                don't know your source ID?
            </a>
            {error && <p className="text-red-200 text-xs">{error}</p>}
            {loadedSource && (
                <div className="flex flex-col gap-2">
                    <div className="flex items-center gap-2">
                         <p className="text-white/80 text-xs">{loadedSource.name}</p>
                         <button
                            onClick={async () => {
                                setExportError(null);
                                try {
                                    const r = await apiFetchOrThrow(
                                        `/api/projects/${project.id}/sources/${loadedSource.sourceId}/export`,
                                    );
                                    downloadBlob(await r.blob(), `source-${loadedSource.sourceId}-export.zip`);
                                } catch (e) {
                                    setExportError((e as Error).message);
                                }
                            }}
                            className="text-white/50 hover:text-white text-[10px] underline cursor-pointer"
                        >
                            download zip
                        </button>
                    </div>
                    {exportError && <p className="text-red-200 text-xs">{exportError}</p>}
                {imageSubTab === "grid" ? (
                    <>
                        <select
                            value={folio}
                            onChange={(e) => {
                                patch({ folio: e.target.value });
                                setConflict(findFolioConflict(project.images, e.target.value) ?? null);
                            }}
                            className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-40"
                        >
                            <option value="">select folio...</option>
                            {loadedSource.folios.map((f) => (
                            <option key={f} value={f}>{f}</option>
                            ))}
                        </select>
                        {conflict && (
                            <p className="text-yellow-200 text-xs">
                                ⚠ folio "{folio}" is already used by {conflict.name}
                            </p>
                        )}
                        {folio ? (
                            <p className="text-white/50 text-xs">
                            the next image you upload in the images tab will be tagged as folio "{folio}"
                            </p>
                        ) : (
                            <p className="text-white/50 text-xs">
                            select a folio above before uploading
                            </p>
                        )}
                    </>
                ) : (
                    <div className="flex flex-col gap-3">
                        <div className="flex gap-4">
                            <label className="flex flex-col gap-1">
                                <span className="text-xs text-white/70">start folio</span>
                                <select
                                    value={batchStartFolio}
                                    onChange={(e) => onBatchStartFolioChange(e.target.value)}
                                    className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-32"
                                >
                                    <option value="">select...</option>
                                    {loadedSource.folios.map((f) => <option key={f} value={f}>{f}</option>)}
                                </select>
                            </label>
                            <label className="flex flex-col gap-1">
                                <span className="text-xs text-white/70">end folio</span>
                                <select
                                    value={batchEndFolio}
                                    onChange={(e) => onBatchEndFolioChange(e.target.value)}
                                    className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-32"
                                >
                                    <option value="">select...</option>
                                    {loadedSource.folios.map((f) => <option key={f} value={f}>{f}</option>)}
                                </select>
                            </label>
                        </div>
                        {batchFolioSequence.length > 0 ? (
                            <p className="text-xs text-white/70">{batchFolioSequence.length} folio(s) in this range</p>
                        ) : (
                            <p className="text-xs text-white/50">
                                select a range, then use "+ new image" above to upload — files are tagged with folios in order
                            </p>
                        )}
                    </div>
                )}
                </div>
            )}
            </div>
        )}
        </div>
    );
}
