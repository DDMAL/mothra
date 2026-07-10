import { useState, useEffect } from "react";
import type { CantusSource, Project } from "../../types";
import { apiFetchOrThrow } from "../../lib/apiFetch";
import type { useTextFindingSettings } from "../../hooks/useTextFindingSettings";

interface CantusSourcePanelProps {
    textFindingSettings: ReturnType<typeof useTextFindingSettings>;
    project: Project,
    onUpdateSourceId: (sourceId: string) => void,
    onSourceLoaded?: (s: CantusSource | null) => void
}

export default function CantusSourcePanel({ textFindingSettings }: CantusSourcePanelProps) {
    const { ocrOnlyMode, sourceId, folio, patch } = textFindingSettings;
    const [sourceIdInput, setSourceIdInput] = useState(sourceId);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loadedSource, setLoadedSource] = useState<CantusSource | null>(null);

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
                <p className="text-white/80 text-xs">{loadedSource.name}</p>
                <select
                    value={folio}
                    onChange={(e) => patch({ folio: e.target.value })}
                    className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-40"
                >
                    <option value="">select folio...</option>
                    {loadedSource.folios.map((f) => (
                    <option key={f} value={f}>{f}</option>
                    ))}
                </select>
                {folio && (
                    <p className="text-white/50 text-xs">
                    the next image you upload in the images tab will be tagged as folio "{folio}"
                    </p>
                )}
                </div>
            )}
            </div>
        )}
        </div>
    );
}