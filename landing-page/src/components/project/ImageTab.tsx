import { useRef, useState } from "react";
import type { Project, ProjectImage } from "../../types";
import * as pdfjsLib from "pdfjs-dist";
import { useAssetSection, ITEMS_PER_PAGE } from "../../hooks/useAssetSection";
import { AuthImage } from "../shared/AuthImage";
import Modal from "../shared/Modal";
import ContextMenu from "../shared/ContextMenu";
import AssetGrid from "../shared/AssetGrid";
import RenameModal from "./RenameModal";
import QuickLookModal from "../shared/QuickLookModal";
import FileDropZone from "../shared/FileDropZone";

pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

interface ImageTabProps {
    project: Project;
    section: ReturnType<typeof useAssetSection<ProjectImage>>;
    usedNames: { images: string[]; models: string[] };
    onUpdateProject: (p: Project) => void;
    onUsedNamesChange: (names: { images: string[]; models: string[] }) => void;
    onUploadImage: (file: File) => Promise<{ id: string; name: string}>;
    onDeleteImage: (imageId: string) => Promise<void>;
    setValidationError: (e: string | null) => void;
}

export default function ImageTab({
    project, section, usedNames, onUpdateProject, onUsedNamesChange, onUploadImage, onDeleteImage, setValidationError,
} : ImageTabProps) {
    const [quickLookId, setQuickLookId] = useState<string | null>(null);
    const [converting, setConverting] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const folderInputRef = useRef<HTMLInputElement>(null);

    const deleteImage = async (id: string) => {
    await onDeleteImage(id);
    onUpdateProject({
      ...project,
      images: project.images.filter((img) => img.id !== id),
    });
    section.setMenu(null);
  };

  const renameImage = () => {
    const current = project.images.find(
      (img) => img.id === section.renameModal?.id,
    );
    onUpdateProject({
      ...project,
      images: project.images.map((img) =>
        img.id === section.renameModal?.id
          ? { ...img, name: section.renameName.trim() || current!.name }
          : img,
      ),
    });
    section.setRenameModal(null);
  };

  const pdfToImages = async (
    file: File,
  ): Promise<{ name: string; src: string }[]> => {
    const baseName = file.name.replace(/\.pdf$/i, "");
    const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() })
      .promise;
    const results: { name: string; src: string }[] = [];
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 300 / 72 });
      const canvas = document.createElement("canvas");
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      await page.render({
        canvasContext: canvas.getContext("2d")!,
        canvas,
        viewport,
      }).promise;
      const blob = await new Promise<Blob>((res) =>
        canvas.toBlob((b) => res(b!), "image/png"),
      );
      results.push({
        name: `${baseName} (page${i}).png`,
        src: URL.createObjectURL(blob),
      });
    }
    return results;
  };

  const handleFiles = async (files: FileList | File[]) => {
    const all = Array.from(files);
    const imageFiles = all.filter((f) => f.type.startsWith("image/"));
    const pdfFiles = all.filter((f) => f.type === "application/pdf");
    if (imageFiles.length === 0 && pdfFiles.length === 0) return;
    setConverting(true);

    const imageEntries = await Promise.all(imageFiles.map(async (f) => {
      const result = await onUploadImage(f);
      return { id: result.id, name: result.name, src: `/api/images/${result.id}` };
    }));
    
    const pdfImages = (await Promise.all(pdfFiles.map(pdfToImages))).flat();
    const pdfEntries = await Promise.all(pdfImages.map(async ({ name, src: blobUrl }) => {
      const blob = await fetch(blobUrl).then((r) => r.blob());
      URL.revokeObjectURL(blobUrl);
      const file = new File([blob], name, { type: "image/png "});
      const result = await onUploadImage(file);
      return { id: result.id, name: result.name, src: `/api/images/${result.id}` };
    }));

    onUpdateProject({ ...project, images: [...project.images, ...imageEntries, ...pdfEntries] });
    setConverting(false);
    section.setUploadModal(false);
    section.setDragging(false);
  };

  const totalImagePages = Math.ceil(project.images.length / ITEMS_PER_PAGE);
    const pagedImages = project.images.slice(
      section.page * ITEMS_PER_PAGE,
      (section.page + 1) * ITEMS_PER_PAGE,
    );

    return (
        <>
            <div className="mt-6" onClick={() => section.clearSelection()}>
                {project.images.length === 0 ? (
                    <p className="text-white/70 text-sm">no images yet</p>
                ) : (
                    <AssetGrid
                        pagedItems={pagedImages}
                        pageOffset={section.page * ITEMS_PER_PAGE}
                        section={section}
                        usedNames={usedNames.images}
                        totalPages={totalImagePages}
                        renderThumbnail={img => 
                            img.src ? <AuthImage src={img.src} alt={img.name} className="w-full h-full object-cover" /> : null
                        }
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
                         label: "Quick Look",
                         onClick: () => { setQuickLookId(section.menu!.id); section.setMenu(null); },
                       },
                       {
                         label: "Use Image",
                         onClick: () => {
                           const img = project.images.find(i => i.id === section.menu!.id);
                           if (img && !usedNames.images.includes(img.name)) {
                             onUsedNamesChange({ ...usedNames, images: [...usedNames.images, img.name] });
                           }
                           section.setMenu(null);
                           setValidationError(null);
                         },
                       },
                       {
                         label: "Delete Image",
                         onClick: () => deleteImage(section.menu!.id),
                       },
                       {
                         label: "Rename Image",
                         onClick: () => {
                           const img = project.images.find(i => i.id === section.menu!.id)!;
                           section.setRenameModal({ id: section.menu!.id });
                           section.setRenameName(img.name);
                           section.setMenu(null);
                         },
                       },
                     ]}
                   />
            )}

            {section.renameModal && (
                    <RenameModal
                      label="image"
                      value={section.renameName}
                      onChange={section.setRenameName}
                      onSubmit={renameImage}
                      onClose={() => section.setRenameModal(null)}
                    />
            )}


            {quickLookId && (() => {
                    const img = project.images.find((i) => i.id === quickLookId)!;
                    const isUsed = usedNames.images.includes(img.name);
                    return (
                      <QuickLookModal onClose={() => setQuickLookId(null)}>
                        <div className="flex items-center justify-center bg-[#C8E6E3]/20 rounded-xl overflow-hidden max-h-[60vh]">
                          {img.src ? (
                            <AuthImage src={img.src} alt={img.name} className="object-contain max-h-[60vh] w-full" />
                          ) : (
                            <span className="text-white/40 text-sm py-16">{img.name}</span>
                          )}
                        </div>
                        <div className="flex gap-3 justify-center">
                          {!isUsed && (
                            <button
                              onClick={() => {
                                onUsedNamesChange({ ...usedNames, images: [...usedNames.images, img.name] });
                                setValidationError(null);
                                setQuickLookId(null);
                              }}
                              className="px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm">
                                Use Image
                              </button>
                          )}
                          <button
                            onClick={() => { deleteImage(quickLookId); setQuickLookId(null); }}
                            className="px-5 py-2 border-2 border-white/40 text-white rounded-xl hover:opacity-90 cursor-pointer text-sm"
                          >
                            Delete Image
                          </button>
                        </div>
                      </QuickLookModal>
                    );
             })()}


             {section.uploadModal && (
                <Modal onClose={() => { if (!converting) { section.setUploadModal(false); section.setDragging(false); } }}>
                <h2 className="text-xl text-[#1D3335] text-center">upload image</h2>
                {converting ? (
                    <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-[#1D3335]/30 bg-white/40 py-12">
                    <p className="text-sm text-[#1D3335] text-center">converting PDF pages...</p>
                    </div>
                ) : (
                    <FileDropZone
                    dragging={section.dragging}
                    onDragOver={(e) => { e.preventDefault(); section.setDragging(true); }}
                    onDragEnter={(e) => { e.preventDefault(); section.setDragging(true); }}
                    onDragLeave={() => section.setDragging(false)}
                    onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); }}
                    onClick={() => fileInputRef.current?.click()}
                    label="drag & drop images, folders, or PDFs here"
                    >
                    <div className="flex gap-4 text-sm text-[#1D3335]">
                        <button onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }} className="underline hover:opacity-70 cursor-pointer">select files</button>
                        <span className="text-[#1D3335]/40">or</span>
                        <button onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click(); }} className="underline hover:opacity-70 cursor-pointer">select folder</button>
                    </div>
                    </FileDropZone>
                )}
                <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*,application/pdf"
                    multiple
                    className="hidden"
                    onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}
                />
                <input
                    ref={folderInputRef}
                    type="file"
                    // @ts-expect-error
                    webkitdirectory=""
                    className="hidden"
                    onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}
                />
                </Modal>
            )}
        </>
    )
  
}


