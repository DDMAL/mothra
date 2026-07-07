import { useRef, useState } from "react";
import type { Project, ProjectModel, ModelKind } from "../../types";
import { useAssetSection, ITEMS_PER_PAGE } from "../../hooks/useAssetSection";
import type { useInferenceSettings } from "../../hooks/useInferenceSettings";
import type { useTextFindingSettings } from "../../hooks/useTextFindingSettings";
import { apiFetch } from "../../lib/apiFetch";
import Modal from "../shared/Modal";
import ContextMenu from "../shared/ContextMenu";
import AssetGrid from "../shared/AssetGrid";
import RenameModal from "./RenameModal";
import FileDropZone from "../shared/FileDropZone";


const KIND_EXTENSIONS: Record<ModelKind, RegExp> = {
  yolo: /\.pt$/i,
  segmentation: /\.(mlmodel|safetensors)$/i,
  recognition: /\.(mlmodel|safetensors)$/i,
};
const KIND_ACCEPT: Record<ModelKind, string> = {
  yolo: ".pt",
  segmentation: ".mlmodel,.safetensors",
  recognition: ".mlmodel,.safetensors",
};
const KIND_DROPZONE_LABEL: Record<ModelKind, string> = {
  yolo: "drag & drop .pt files here",
  segmentation: "drag & drop .mlmodel / .safetensors files here",
  recognition: "drag & drop .mlmodel / .safetensors files here",
};
const KIND_OPTION_LABEL: Record<ModelKind, string> = {
  yolo: "YOLO detection model (.pt)",
  segmentation: "text segmentation model (.mlmodel / .safetensors)",
  recognition: "OCR / recognition model (.mlmodel / .safetensors)",
};
const KIND_BADGE: Record<ModelKind, string> = {
  yolo: "PT",
  segmentation: "SEG",
  recognition: "OCR",
};


interface ModelTabProps {
  project: Project;
  section: ReturnType<typeof useAssetSection<ProjectModel>>;
  usedNames: { images: string[]; models: string[]; annotations: string[] };
  onUpdateProject: (p: Project) => void;
  onUsedNamesChange: (names: { images: string[]; models: string[]; annotations: string[] }) => void;
  onUploadModel: (file: File, kind: ModelKind) => Promise<{ id: string; name: string, kind: ModelKind }>;
  setValidationError: (e: string | null) => void;
  inferenceSettings: ReturnType<typeof useInferenceSettings>;
  textFindingSettings: ReturnType<typeof useTextFindingSettings>;
}

export default function ModelTab({
  project,
  section,
  usedNames,
  onUpdateProject,
  onUsedNamesChange,
  onUploadModel,
  setValidationError,
  inferenceSettings,
  textFindingSettings,
}: ModelTabProps) {
  const modelFileInputRef = useRef<HTMLInputElement>(null);

  const [settingsOpen, setSettingsOpen] = useState(false);
  const [textSettingsOpen, setTextSettingsOpen] = useState(false);
  const [textAdvancedOpen, setTextAdvancedOpen] = useState(false);

  const [uploadKind, setUploadKind] = useState<ModelKind>("yolo");

  const segmentationModels = project.models.filter((m) => m.kind === "segmentation");
  const recognitionModels = project.models.filter((m) => m.kind === "recognition");

  // model actions
  const deleteModel = async (id: string) => {
    section.setMenu(null);
    const r = await apiFetch(`/api/projects/${project.id}/models/${id}`, {
      method: "DELETE",
    });
    if (!r.ok) return;
    onUpdateProject({
      ...project,
      models: project.models.filter((m) => m.id !== id),
    });
  };

  const renameModel = () => {
    const current = project.models.find(
      (m) => m.id === section.renameModal?.id,
    );
    onUpdateProject({
      ...project,
      models: project.models.map((m) =>
        m.id === section.renameModal?.id
          ? { ...m, name: section.renameName.trim() || current!.name }
          : m,
      ),
    });
    section.setRenameModal(null);
  };

  const handleModelFiles = async (files: FileList | File[]) => {
    const valid = Array.from(files).filter((f) => KIND_EXTENSIONS[uploadKind].test(f.name));
    if (valid.length === 0) return;
    const entries = await Promise.all(
      valid.map(async (f) => {
        const result = await onUploadModel(f, uploadKind);
        return { id: result.id, name: result.name || f.name, kind: result.kind ?? uploadKind };
      }),
    );
    onUpdateProject({ ...project, models: [...project.models, ...entries] });
    section.setUploadModal(false);
    section.setDragging(false);
    setUploadKind("yolo");
  };

  const totalModelPages = Math.ceil(project.models.length / ITEMS_PER_PAGE);
  const pagedModels = project.models.slice(
    section.page * ITEMS_PER_PAGE,
    (section.page + 1) * ITEMS_PER_PAGE,
  );

  return (
    <>
      <div className="mt-6" onClick={() => section.clearSelection()}>
        {project.models.length === 0 ? (
          <p className="text-white/70 text-sm">no models yet</p>
        ) : (
          <AssetGrid
            pagedItems={pagedModels}
            pageOffset={section.page * ITEMS_PER_PAGE}
            section={section}
            usedNames={usedNames.models}
            totalPages={totalModelPages}
            renderThumbnail={(item) => (
              <svg width="56" height="64" viewBox="0 0 56 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M4 0H36L56 20V60C56 62.2 54.2 64 52 64H4C1.8 64 0 62.2 0 60V4C0 1.8 1.8 0 4 0Z" fill="white" fillOpacity="0.25" />
                <path d="M36 0L56 20H40C37.8 20 36 18.2 36 16V0Z" fill="white" fillOpacity="0.45" />
                <text x="28" y="46" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold" fontFamily="monospace">
                  {KIND_BADGE[item.kind]}
                </text>
              </svg>
            )}
          />
        )}

        {usedNames.models.length > 0 && (
          <div className="mt-4">
            <button
              onClick={() => setSettingsOpen(o => !o)}
              className="text-white/60 text-xs hover:text-white cursor-pointer select-none flex items-center gap-1"
            >
              {settingsOpen ? "▾" : "▸"} inference settings
            </button>
            {settingsOpen && (
              <div className="mt-2 bg-white/10 rounded-xl p-4 flex flex-col gap-4 text-sm text-white">
                <label className="flex flex-col gap-1">
                  <span className="text-white/70 text-xs">
                    confidence threshold: {inferenceSettings.threshold.toFixed(2)}
                  </span>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={inferenceSettings.threshold}
                    onChange={(e) => inferenceSettings.patch({ threshold: Number(e.target.value) })}
                    className="accent-[#1D3335]"
                  />
                </label>
                <div className="flex flex-col gap-1">
                  <span className="text-white/70 text-xs">device</span>
                  <div className="flex gap-3">
                    {(["cpu", "cuda", "mps"] as const).map(d => (
                      <label key={d} className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="radio" name="inference-device" value={d}
                          checked={inferenceSettings.device === d}
                          onChange={() => inferenceSettings.patch({ device: d })}
                          className="accent-[#1D3335]"
                        />
                        {d}
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {usedNames.models.length > 0 && (
          <div className="mt-2">
            <button
              onClick={() => setTextSettingsOpen((o) => !o)}
              className="text-white/60 text-xs hover:text-white cursor-pointer select-none flex items-center gap-1"
            >
              {textSettingsOpen ? "▾" : "▸"} text-finding settings
            </button>
            {textSettingsOpen && (
              <div className="mt-2 bg-white/10 rounded-xl p-4 flex flex-col gap-4 text-sm text-white">
                <div className="flex flex-col gap-1">
                  <span className="text-white/70 text-xs">column count</span>
                  <div className="flex gap-3">
                    {(["auto", "1", "2"] as const).map((c) => (
                      <label key={c} className="flex items-center gap-1 cursor-pointer">
                        <input
                          type="radio"
                          name="text-column-count"
                          value={c}
                          checked={textFindingSettings.columnCount === c}
                          onChange={() => textFindingSettings.patch({ columnCount: c })}
                          className="accent-[#1D3335]"
                        />
                        {c === "auto" ? "auto-detect" : c}
                      </label>
                    ))}
                  </div>
                </div>

                <label className="flex flex-col gap-1">
                  <span className="text-white/70 text-xs">custom segmentation model</span>
                  <select
                    value={textFindingSettings.segmentationModelId}
                    onChange={(e) => textFindingSettings.patch({ segmentationModelId: e.target.value })}
                    className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none"
                  >
                    <option value="">default: Kraken's built-in BLLA model</option>
                    {segmentationModels.map((m) => (
                      <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                  </select>
                  {segmentationModels.length === 0 && (
                    <span className="text-white/40 text-xs italic">
                      no segmentation models uploaded yet — upload one above (type: text segmentation model)
                    </span>
                  )}
                </label>

                <label className="flex flex-col gap-1">
                  <span className="text-white/70 text-xs">custom OCR model</span>
                  <select
                    value={textFindingSettings.recognitionModelId}
                    onChange={(e) => textFindingSettings.patch({ recognitionModelId: e.target.value })}
                    className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm text-white outline-none"
                  >
                    <option value="">default: auto-detected Tridis model (or stub if not installed)</option>
                    {recognitionModels.map((m) => (
                      <option key={m.id} value={m.id}>{m.name}</option>
                    ))}
                  </select>
                  {recognitionModels.length === 0 && (
                    <span className="text-white/40 text-xs italic">
                      no OCR models uploaded yet — upload one above (type: OCR / recognition model)
                    </span>
                  )}
                </label>

                <div>
                  <button
                    onClick={() => setTextAdvancedOpen((o) => !o)}
                    className="text-white/60 text-xs hover:text-white cursor-pointer select-none flex items-center gap-1"
                  >
                    {textAdvancedOpen ? "▾" : "▸"} advanced
                  </button>
                  {textAdvancedOpen && (
                    <div className="mt-2 flex flex-col gap-4 pl-3 border-l border-white/20">
                      <div className="flex flex-col gap-1">
                        <span className="text-white/70 text-xs">device</span>
                        <div className="flex gap-3">
                          {(["cpu", "cuda"] as const).map((d) => (
                            <label key={d} className="flex items-center gap-1 cursor-pointer">
                              <input
                                type="radio"
                                name="text-device"
                                value={d}
                                checked={textFindingSettings.device === d}
                                onChange={() => textFindingSettings.patch({ device: d })}
                                className="accent-[#1D3335]"
                              />
                              {d}
                            </label>
                          ))}
                        </div>
                      </div>
                      <label className="flex flex-col gap-1">
                        <span className="text-white/70 text-xs">
                          column-split sensitivity: {textFindingSettings.columnBimodalThreshold.toFixed(2)}
                          {textFindingSettings.columnCount === "1" && " (ignored — column count forced to 1)"}
                        </span>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={textFindingSettings.columnBimodalThreshold}
                          disabled={textFindingSettings.columnCount === "1"}
                          onChange={(e) =>
                            textFindingSettings.patch({ columnBimodalThreshold: Number(e.target.value) })
                          }
                          className="accent-[#1D3335] disabled:opacity-40"
                        />
                      </label>
                    </div>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
      </div>

      {section.menu && (
        <ContextMenu
          x={section.menu.x}
          y={section.menu.y}
          onClose={() => section.setMenu(null)}
          items={[
            ...(project.models.find((m) => m.id === section.menu!.id)?.kind === "yolo"
              ? [{
                  label: "Use Model",
                  onClick: () => {
                    const model = project.models.find((m) => m.id === section.menu!.id);
                    if (model && !usedNames.models.includes(model.name))
                      onUsedNamesChange({ ...usedNames, models: [...usedNames.models, model.name] });
                    section.setMenu(null);
                    setValidationError(null);
                  },
                }]
              : []),
            { label: "Delete Model", onClick: () => deleteModel(section.menu!.id) },
            {
              label: "Rename Model",
              onClick: () => {
                const m = project.models.find((m) => m.id === section.menu!.id)!;
                section.setRenameModal({ id: section.menu!.id });
                section.setRenameName(m.name);
                section.setMenu(null);
              },
            },
          ]}
        />
      )}

      {section.renameModal && (
        <RenameModal
          label="model"
          value={section.renameName}
          onChange={section.setRenameName}
          onSubmit={renameModel}
          onClose={() => section.setRenameModal(null)}
        />
      )}

      {section.uploadModal && (
        <Modal
          onClose={() => {
            section.setUploadModal(false);
            section.setDragging(false);
            setUploadKind("yolo");
          }}
        >
          <h2 className="text-xl text-[#1D3335] text-center">upload model</h2>
            <label className="flex flex-col gap-1">
              <span className="text-xs text-[#1D3335]/70">model type</span>
              <select
                value={uploadKind}
                onChange={(e) => setUploadKind(e.target.value as ModelKind)}
                className="border border-[#1D3335]/30 rounded px-2 py-1 text-sm text-[#1D3335] bg-white/60"
              >
                {(["yolo", "segmentation", "recognition"] as const).map((k) => (
                  <option key={k} value={k}>{KIND_OPTION_LABEL[k]}</option>
                ))}
              </select>
            </label>
            <FileDropZone
              dragging={section.dragging}
              onDragOver={(e) => { e.preventDefault(); section.setDragging(true); }}
              onDragEnter={(e) => { e.preventDefault(); section.setDragging(true); }}
              onDragLeave={() => section.setDragging(false)}
              onDrop={(e) => { e.preventDefault(); handleModelFiles(e.dataTransfer.files); }}
              onClick={() => modelFileInputRef.current?.click()}
              label={KIND_DROPZONE_LABEL[uploadKind]}
            >
              <button
                onClick={(e) => { e.stopPropagation(); modelFileInputRef.current?.click(); }}
                className="text-sm text-[#1D3335] underline hover:opacity-70 cursor-pointer"
              >
                select files
              </button>
            </FileDropZone>
            <input
              ref={modelFileInputRef}
              type="file"
              accept={KIND_ACCEPT[uploadKind]}
              multiple
              className="hidden"
              onChange={(e) => { if (e.target.files) handleModelFiles(e.target.files); }}
            />
        </Modal>
      )}
    </>
  );
}
