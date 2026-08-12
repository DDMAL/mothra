import { useState, useEffect, useRef } from "react";
import type { CantusSource, Project, ProjectImage } from "../../types";
import { apiFetchOrThrow } from "../../lib/apiFetch";
import type { useTextFindingSettings } from "../../hooks/useTextFindingSettings";
import type { useIcSettings } from "../../hooks/useIcSettings";
import { findFolioConflict, compareFolios } from "../../utils/folio";
import { downloadBlob } from "../../utils/download";
import FolioSelect from "../shared/FolioSelect";
import IcSettingsSection from "./IcSettingsSection";

// Persists across CantusSourcePanel unmount/remount (e.g. navigating to the
// processing/IC view and back) so a slow or failed background revalidation
// never has to hide a source that was already loaded once this session.
const sourceCache = new Map<string, CantusSource>();

interface CantusSourcePanelProps {
  textFindingSettings: ReturnType<typeof useTextFindingSettings>;
  project: Project;
  onUpdateSourceId: (sourceId: string) => void;
  onSourceLoaded?: (s: CantusSource | null) => void;
  imageSubTab: "grid" | "batch";
  batchStartFolio: string;
  batchEndFolio: string;
  onBatchStartFolioChange: (folio: string) => void;
  onBatchEndFolioChange: (folio: string) => void;
  batchFolioSequence: string[];
  locked: boolean;
  icSettings: ReturnType<typeof useIcSettings>;
}

export default function CantusSourcePanel({
  textFindingSettings,
  project,
  onUpdateSourceId,
  onSourceLoaded,
  imageSubTab,
  batchStartFolio,
  batchEndFolio,
  onBatchStartFolioChange,
  onBatchEndFolioChange,
  batchFolioSequence,
  locked,
  icSettings,
}: CantusSourcePanelProps) {
  const { ocrOnlyMode, sourceId, folio, patch } = textFindingSettings;
  const [sourceIdInput, setSourceIdInput] = useState(sourceId);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loadedSource, setLoadedSource] = useState<CantusSource | null>(null);
  const [conflict, setConflict] = useState<ProjectImage | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  // Guards against a stale in-flight load committing its (now-outdated)
  // result after a newer load has already started — e.g. rapidly loading
  // source A then source B before A's response arrives.
  const loadRequestRef = useRef(0);

  const loadSource = async (
    id: string,
    opts: { preserveFolio?: boolean } = {},
  ) => {
    const requestId = ++loadRequestRef.current;
    const cached = sourceCache.get(id);
    if (cached) {
      setLoadedSource(cached);
      onSourceLoaded?.(cached);
      patch(
        opts.preserveFolio
          ? {
              sourceId: cached.sourceId,
            }
          : { sourceId: cached.sourceId, folio: "" },
      );
      onUpdateSourceId(cached.sourceId);
      setError(null);
      setLoading(false);
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const raw: CantusSource = await apiFetchOrThrow(
        `/api/cantus/source/${id}`,
      ).then((r) => r.json());
      const data: CantusSource = {
        ...raw,
        folios: [...raw.folios].sort(compareFolios),
      };
      sourceCache.set(id, data);
      if (requestId !== loadRequestRef.current) return;
      setLoadedSource(data);
      onSourceLoaded?.(data);
      patch(
        opts.preserveFolio
          ? { sourceId: data.sourceId }
          : { sourceId: data.sourceId, folio: "" },
      );
      onUpdateSourceId(data.sourceId);
    } catch (e) {
      if (requestId !== loadRequestRef.current) return;
      setLoadedSource(null);
      onSourceLoaded?.(null);
      patch({ sourceId: "", folio: "" });
      setError((e as Error).message);
    } finally {
      if (requestId === loadRequestRef.current) setLoading(false);
    }
  };

  const handleLoad = async () => {
    const trimmed = sourceIdInput.trim();
    if (trimmed) loadSource(trimmed);
  };

  // textFindingSettings (incl. `folio`) is one application-level state
  // instance shared across all projects (created once in AppRouter), so
  // "preserve the folio" is only correct when this effect is re-running for
  // the *same* project (e.g. a remount after navigating to processing/IC and
  // back) — otherwise a leftover folio from a different project could leak
  // into a new project's upload, even when two projects share a source id.
  const prevProjectIdRef = useRef<number | null>(null);

  useEffect(() => {
    // project switched — clear all per-project Cantus/OCR state, then
    // load the newly-selected project's own saved source (if any). Keyed on
    // project.id (not just project.cantusSourceId) so this fires on every
    // switch, including between two projects that happen to share a source
    // id or both have none.
    const isSameProject = prevProjectIdRef.current === project.id;
    prevProjectIdRef.current = project.id;

    setLoadedSource(null);
    setError(null);
    setConflict(null);
    setExportError(null);
    onSourceLoaded?.(null);
    patch({ ocrOnlyMode: false });

    if (project.cantusSourceId) {
      setSourceIdInput(project.cantusSourceId);
      loadSource(project.cantusSourceId, { preserveFolio: isSameProject });
    } else {
      // invalidate any still-in-flight load from a previous project so
      // its (now-stale) response can't land here and set a source that
      // this project never had
      loadRequestRef.current++;
      setSourceIdInput("");
      patch({ sourceId: "", folio: "" });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [project.id, project.cantusSourceId]);

  return (
    // Two settings columns side by side: the Cantus source (left) and the IC
    // step (right). Both are pre-run choices, and side-by-side keeps the panel
    // from pushing the tab bar below the fold.
    <div className="mb-4 bg-white/10 rounded-xl p-4 flex gap-8 text-sm text-white max-w-3xl">
      <div className="flex flex-col gap-3 flex-[2] min-w-0">
        <h3 className="text-base text-white font-semibold">
          CantusDB settings
        </h3>
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
                disabled={locked}
                className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-40"
              />
              <button
                onClick={handleLoad}
                disabled={
                  loading ||
                  !sourceIdInput.trim() ||
                  locked ||
                  sourceIdInput.trim() === loadedSource?.sourceId
                }
                className="px-3 py-1 border border-white/40 text-white text-xs rounded-lg hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
              >
                {loading ? "loading..." : "load"}
              </button>
            </div>
            {locked ? (
              <p className="text-white/50 text-xs w-fit">
                source is locked - this project has already moved past the
                upload step
              </p>
            ) : (
              <a
                href="https://cantusdatabase.org/sources/"
                target="_blank"
                rel="noopener noreferrer"
                className="text-white/50 text-xs hover:text-white underline w-fit"
              >
                don't know your source ID?
              </a>
            )}
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
                        downloadBlob(
                          await r.blob(),
                          `source-${loadedSource.sourceId}-export.zip`,
                        );
                      } catch (e) {
                        setExportError((e as Error).message);
                      }
                    }}
                    className="text-white/50 hover:text-white text-[10px] underline cursor-pointer"
                  >
                    download zip
                  </button>
                </div>
                {exportError && (
                  <p className="text-red-200 text-xs">{exportError}</p>
                )}
                {imageSubTab === "grid" ? (
                  <>
                    <FolioSelect
                      value={folio}
                      options={loadedSource.folios}
                      onChange={(v) => {
                        patch({ folio: v });
                        setConflict(
                          findFolioConflict(project.images, v) ?? null,
                        );
                      }}
                      className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-40"
                    />
                    {conflict && (
                      <p className="text-yellow-200 text-xs">
                        ⚠ folio "{folio}" is already used by {conflict.name}
                      </p>
                    )}
                    {folio ? (
                      <p className="text-white/50 text-xs">
                        the next image you upload in the images tab will be
                        tagged as folio "{folio}"
                      </p>
                    ) : (
                      <p className="text-white/50 text-xs">
                        select a folio above to tag your next upload —
                        already-uploaded images keep the folio they were tagged
                        with
                      </p>
                    )}
                  </>
                ) : (
                  <div className="flex flex-col gap-3">
                    <div className="flex gap-4">
                      <label className="flex flex-col gap-1">
                        <span className="text-xs text-white/70">
                          start folio
                        </span>
                        <FolioSelect
                          value={batchStartFolio}
                          options={loadedSource.folios}
                          onChange={onBatchStartFolioChange}
                          className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-32"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        <span className="text-xs text-white/70">end folio</span>
                        <FolioSelect
                          value={batchEndFolio}
                          options={loadedSource.folios}
                          onChange={onBatchEndFolioChange}
                          className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none w-32"
                        />
                      </label>
                    </div>
                    {batchFolioSequence.length > 0 ? (
                      <p className="text-xs text-white/70">
                        {batchFolioSequence.length} folio(s) in this range
                      </p>
                    ) : (
                      <p className="text-xs text-white/50">
                        select a range, then use "+ new image" above to upload —
                        files are tagged with folios in order
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Step-1 settings share this box - both are choices made before the
          pipeline runs. Outside the OCR-only branch above: the IC step
          happens either way. */}
      <div className="flex-[3] min-w-0 border-l border-white/20 pl-8">
        <IcSettingsSection icSettings={icSettings} />
      </div>
    </div>
  );
}
