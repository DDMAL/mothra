import { useRef } from "react";
import type { Project, ProjectModel } from "../../types";
import { useAssetSection, ITEMS_PER_PAGE } from "../../hooks/useAssetSection";
import Modal from "../shared/Modal";
import ContextMenu from "../shared/ContextMenu";
import AssetGrid from "../shared/AssetGrid";
import RenameModal from "./RenameModal";
import FileDropZone from "../shared/FileDropZone";

interface ModelTabProps {
    project: Project;
    section: ReturnType<typeof useAssetSection<ProjectModel>>;
    usedNames: { images: string[]; models: string[]; };
    onUpdateProject: (p: Project) => void;
    onUsedNamesChange: (names: { images: string[]; models: string[] }) => void;
    onUploadModel: (name: string) => Promise<{ id: string; name: string }>;
    setValidationError: (e: string | null) => void;
}

export default function ModelTab({
    project, section, usedNames, onUpdateProject, onUsedNamesChange, onUploadModel, setValidationError,
}: ModelTabProps) {
    const modelFileInputRef = useRef<HTMLInputElement>(null);

    // model actions
      const deleteModel = (id: string) => {
        onUpdateProject({
          ...project,
          models: project.models.filter((m) => m.id !== id),
        });
        section.setMenu(null);
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
        const valid = Array.from(files).filter((f) => /\.(h5|hdf5)$/i.test(f.name));
        if (valid.length === 0) return;
        const entries = await Promise.all(valid.map(async (f) => {
          const result = await onUploadModel(f.name);
          return { id: result.id, name: result.name || f.name };
        }));
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
                        <svg width="56" height="64" viewBox="0 0 56 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                            <path d="M4 0H36L56 20V60C56 62.2 54.2 64 52 64H4C1.8 64 0 62.2 0 60V4C0 1.8 1.8 0 4 0Z" fill="white" fillOpacity="0.25" />
                            <path d="M36 0L56 20H40C37.8 20 36 18.2 36 16V0Z" fill="white" fillOpacity="0.45" />
                            <text x="28" y="46" textAnchor="middle" fill="white" fontSize="16" fontWeight="bold" fontFamily="monospace">H5</text>
                        </svg>
                        )}
                    />
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
                        const model = project.models.find(m => m.id === section.menu!.id);
                        if (model && !usedNames.models.includes(model.name))
                        onUsedNamesChange({ ...usedNames, models: [...usedNames.models, model.name] });
                        section.setMenu(null);
                        setValidationError(null);
                    },
                    },
                    { label: "Delete Model", onClick: () => deleteModel(section.menu!.id) },
                    {
                    label: "Rename Model",
                    onClick: () => {
                        const m = project.models.find(m => m.id === section.menu!.id)!;
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
                <Modal onClose={() => { section.setUploadModal(false); section.setDragging(false); }}>
                <h2 className="text-xl text-[#1D3335] text-center">upload model</h2>
                <FileDropZone
                    dragging={section.dragging}
                    onDragOver={(e) => { e.preventDefault(); section.setDragging(true); }}
                    onDragEnter={(e) => { e.preventDefault(); section.setDragging(true); }}
                    onDragLeave={() => section.setDragging(false)}
                    onDrop={(e) => { e.preventDefault(); handleModelFiles(e.dataTransfer.files); }}
                    onClick={() => modelFileInputRef.current?.click()}
                    label="drag & drop .h5 or .hdf5 files here"
                >
                    <button
                    onClick={(e) => { e.stopPropagation(); modelFileInputRef.current?.click(); }}
                    className="text-sm text-[#1D3335] underline hover:opacity-70 cursor-pointer"
                    >select files</button>
                </FileDropZone>
                <input ref={modelFileInputRef} type="file" accept=".h5,.hdf5" multiple className="hidden"
                    onChange={(e) => { if (e.target.files) handleModelFiles(e.target.files); }} />
                </Modal>
            )}
        </>
    );
}