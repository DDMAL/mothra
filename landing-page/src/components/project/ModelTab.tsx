import { useRef, useState } from "react";
import type { Project, ProjectModel } from "../../types";
import { useAssetSection, ITEMS_PER_PAGE } from "../../hooks/useAssetSection";
import { apiFetch } from "../../lib/apiFetch";
import Modal from "../shared/Modal";
import ContextMenu from "../shared/ContextMenu";
import AssetGrid from "../shared/AssetGrid";
import RenameModal from "./RenameModal";
import FileDropZone from "../shared/FileDropZone";

interface ModelTabProps {
  project: Project;
  section: ReturnType<typeof useAssetSection<ProjectModel>>;
  usedNames: { images: string[]; models: string[]; annotations: string[] };
  onUpdateProject: (p: Project) => void;
  onUsedNamesChange: (names: { images: string[]; models: string[]; annotations: string[] }) => void;
  onUploadModel: (file: File) => Promise<{ id: string; name: string }>;
  setValidationError: (e: string | null) => void;
  inferenceThreshold: number;
  onInferenceThresholdChange: (v: number) => void;
  inferenceDevice: "cpu" | "cuda" | "mps";
  onInferenceDeviceChange: (v: "cpu" | "cuda" | "mps") => void;
}

export default function ModelTab({
  project,
  section,
  usedNames,
  onUpdateProject,
  onUsedNamesChange,
  onUploadModel,
  setValidationError,
  inferenceThreshold,
  onInferenceThresholdChange,
  inferenceDevice,
  onInferenceDeviceChange,
}: ModelTabProps) {
  const modelFileInputRef = useRef<HTMLInputElement>(null);

  const [settingsOpen, setSettingsOpen] = useState(false);

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
    const valid = Array.from(files).filter((f) => /\.pt$/i.test(f.name));
    if (valid.length === 0) return;
    const entries = await Promise.all(
      valid.map(async (f) => {
        const result = await onUploadModel(f);
        return { id: result.id, name: result.name || f.name };
      }),
    );
    onUpdateProject({ ...project, models: [...project.models, ...entries] });
    section.setUploadModal(false);
    section.setDragging(false);
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
            renderThumbnail={() => (
              <svg
                width="56"
                height="64"
                viewBox="0 0 56 64"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
              >
                <path
                  d="M4 0H36L56 20V60C56 62.2 54.2 64 52 64H4C1.8 64 0 62.2 0 60V4C0 1.8 1.8 0 4 0Z"
                  fill="white"
                  fillOpacity="0.25"
                />
                <path
                  d="M36 0L56 20H40C37.8 20 36 18.2 36 16V0Z"
                  fill="white"
                  fillOpacity="0.45"
                />
                <text
                  x="28"
                  y="46"
                  textAnchor="middle"
                  fill="white"
                  fontSize="16"
                  fontWeight="bold"
                  fontFamily="monospace"
                >
                  PT
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
                    confidence threshold: {inferenceThreshold.toFixed(2)}
                  </span>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={inferenceThreshold}
                    onChange={(e) => onInferenceThresholdChange(Number(e.target.value))}
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
                          checked={inferenceDevice === d}
                          onChange={() => onInferenceDeviceChange(d)}
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
      </div>

      {section.menu && (
        <ContextMenu
          x={section.menu.x}
          y={section.menu.y}
          onClose={() => section.setMenu(null)}
          items={[
            {
              label: "Use Model",
              onClick: () => {
                const model = project.models.find(
                  (m) => m.id === section.menu!.id,
                );
                if (model && !usedNames.models.includes(model.name))
                  onUsedNamesChange({
                    ...usedNames,
                    models: [...usedNames.models, model.name],
                  });
                section.setMenu(null);
                setValidationError(null);
              },
            },
            {
              label: "Delete Model",
              onClick: () => deleteModel(section.menu!.id),
            },
            {
              label: "Rename Model",
              onClick: () => {
                const m = project.models.find(
                  (m) => m.id === section.menu!.id,
                )!;
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
          }}
        >
          <h2 className="text-xl text-[#1D3335] text-center">upload model</h2>
          <FileDropZone
            dragging={section.dragging}
            onDragOver={(e) => {
              e.preventDefault();
              section.setDragging(true);
            }}
            onDragEnter={(e) => {
              e.preventDefault();
              section.setDragging(true);
            }}
            onDragLeave={() => section.setDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              handleModelFiles(e.dataTransfer.files);
            }}
            onClick={() => modelFileInputRef.current?.click()}
            label="drag & drop .pt files here"
          >
            <button
              onClick={(e) => {
                e.stopPropagation();
                modelFileInputRef.current?.click();
              }}
              className="text-sm text-[#1D3335] underline hover:opacity-70 cursor-pointer"
            >
              select files
            </button>
          </FileDropZone>
          <input
            ref={modelFileInputRef}
            type="file"
            accept=".pt"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleModelFiles(e.target.files);
            }}
          />
        </Modal>
      )}
    </>
  );
}
