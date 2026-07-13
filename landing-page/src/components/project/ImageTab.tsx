import { useEffect, useRef, useState, useMemo } from "react";
import { compareFolios } from "../../utils/folio";
import type { Project, ProjectImage } from "../../types";
import { getImageProgress } from "../../utils/imageStep";
import * as pdfjsLib from "pdfjs-dist";
import { useAssetSection, ITEMS_PER_PAGE } from "../../hooks/useAssetSection";
import { AuthImage } from "../shared/AuthImage";
import { apiFetch, apiFetchOrThrow } from "../../lib/apiFetch";
import Modal from "../shared/Modal";
import ContextMenu from "../shared/ContextMenu";
import AssetGrid from "../shared/AssetGrid";
import RenameModal from "./RenameModal";
import QuickLookModal from "../shared/QuickLookModal";
import FileDropZone from "../shared/FileDropZone";
import EditFolioModal from "./EditFolioModel";
import BatchTab from "./BatchTab";


pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

interface ImageTabProps {
  project: Project;
  section: ReturnType<typeof useAssetSection<ProjectImage>>;
  usedNames: { images: string[]; models: string[]; annotations: string[] };
  onUpdateProject: (p: Project) => void;
  onUsedNamesChange: (names: { images: string[]; models: string[]; annotations: string[] }) => void;
  onUploadImage: (
    file: File,
    folio?: string,
    sourceId?: string,
    sourceName?: string,
  ) => Promise<{ id: string; name: string; folio?: string; sourceId?: string; sourceName?: string }>;
  onDeleteImage: (imageId: string) => Promise<void>;
  setValidationError: (e: string | null) => void;
  activeFolio?: string;
  onFolioConsumed?: () => void;
  cantusFolios?: string[];
  cantusSourceId?: string;
  cantusSourceName?: string;
  onRunBatch: (imageIds: string[], folios: string[]) => void;
  imageSubTab: "grid" | "batch";
  onImageSubTabChange: (tab: "grid" | "batch") => void;
  batchImages: { id: string; name: string }[];
  batchFolioSequence: string[];
  onBatchImageUploaded: (img: { id: string; name: string }) => void;
}

export default function ImageTab({
  project,
  section,
  usedNames,
  onUpdateProject,
  onUsedNamesChange,
  onUploadImage,
  onDeleteImage,
  setValidationError,
  activeFolio,
  onFolioConsumed,
  cantusFolios = [],
  cantusSourceId,
  cantusSourceName,
  onRunBatch,
  imageSubTab,
  onImageSubTabChange,
  batchImages,
  batchFolioSequence,
  onBatchImageUploaded,
}: ImageTabProps) {
  const [quickLookId, setQuickLookId] = useState<string | null>(null);
  const [quickLookTab, setQuickLookTab] = useState<"preview" | "info">(
    "preview",
  );
  const [quickLookMeta, setQuickLookMeta] = useState<{
    mimeType: string;
    sizeBytes: number;
    createdAt: string | null;
  } | null>(null);
  const [quickLookDims, setQuickLookDims] = useState<{
    w: number;
    h: number;
  } | null>(null);
  const [converting, setConverting] = useState(false);
  const [pdfProgress, setPdfProgress] = useState<{
    done: number;
    total: number;
  } | null>(null);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  const [editFolioModal, setEditFolioModal] = useState<{ id: string } | null>(null);
  const [editFolioValue, setEditFolioValue] = useState("");

  const sortedImages = useMemo(
    () =>
      [...project.images].sort((a, b) => {
        const bySource = (a.sourceName || "￿").localeCompare(b.sourceName || "￿");
        return bySource !== 0 ? bySource : compareFolios(a.folio, b.folio);
      }),
    [project.images],
  );

  useEffect(() => {
    if (quickLookTab !== "info" || !quickLookId) return;
    setQuickLookMeta(null);
    apiFetch(`/api/images/${quickLookId}/meta`)
      .then((r) => r.ok ? r.json() : Promise.reject(new Error(`HTTP ${r.status}`)))
      .then(setQuickLookMeta)
      .catch(() => setQuickLookMeta({ mimeType: "unknown", sizeBytes: 0, createdAt: null }));
  }, [quickLookTab, quickLookId]);

  const formatBytes = (b: number) =>
    b < 1024
      ? `${b} B`
      : b < 1024 ** 2
        ? `${(b / 1024).toFixed(1)} KB`
        : `${(b / 1024 ** 2).toFixed(2)} MB`;

  const deleteImage = async (id: string) => {
    try {
      await onDeleteImage(id);
      onUpdateProject({
        ...project,
        images: project.images.filter((img) => img.id !== id),
      });
    } catch (err) {
      // delete failed - leave state unchanged / image remains visible
      console.error("Failed to delete image:", err);
    }
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
    onPageDone: () => void,
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
      onPageDone();
    }
    return results;
  };

  const folioAt = (seq: number): string | undefined => 
    imageSubTab === "batch" ? batchFolioSequence[batchImages.length + seq] : activeFolio;

  const handleFiles = async (files: FileList | File[]) => {
    setUploadError(null);
    if (imageSubTab === "batch" && batchFolioSequence.length === 0) {
      setUploadError("select a start/end folio range above before uploading");
      return;
    }
    try {
      const all = Array.from(files);
      const imageFiles = all.filter((f) => f.type.startsWith("image/"));
      const pdfFiles = all.filter((f) => f.type === "application/pdf");
      if (imageFiles.length === 0 && pdfFiles.length === 0) return;
      setConverting(true);

      let seq = 0;
      const imageEntries = await Promise.all(
        imageFiles.map(async (f) => {
          const result = await onUploadImage(f, folioAt(seq++), cantusSourceId, cantusSourceName);
          if (imageSubTab === "batch") {
            onBatchImageUploaded({ id: result.id, name: result.name });
          }
          return {
            id: result.id,
            name: result.name,
            src: `/api/images/${result.id}`,
            folio: result.folio,
            sourceId: result.sourceId,
            sourceName: result.sourceName,
          };
        }),
      );

      const pdfDocs = await Promise.all(
        pdfFiles.map(
          async (f) =>
            pdfjsLib.getDocument({ data: await f.arrayBuffer() }).promise,
        ),
      );
      const total = pdfDocs.reduce((s, d) => s + d.numPages, 0);
      setPdfProgress({ done: 0, total });

      let done = 0;
      const pdfImages: { name: string; src: string }[] = [];
      for (let i = 0; i < pdfFiles.length; i++) {
        const pages = await pdfToImages(pdfFiles[i], () => {
          done++;
          setPdfProgress({ done, total });
        });
        pdfImages.push(...pages);
      }
      const pdfEntries = await Promise.all(
        pdfImages.map(async ({ name, src: blobUrl }) => {
          const blob = await fetch(blobUrl).then((r) => r.blob());
          URL.revokeObjectURL(blobUrl);
          const file = new File([blob], name, { type: "image/png " });
          const result = await onUploadImage(file, folioAt(seq++), cantusSourceId, cantusSourceName);
          if (imageSubTab === "batch") {
            onBatchImageUploaded({ id: result.id, name: result.name });
          }
          return {
            id: result.id,
            name: result.name,
            src: `/api/images/${result.id}`,
            folio: result.folio,
            sourceId: result.sourceId,
            sourceName: result.sourceName,
          };
        }),
      );

      onUpdateProject({
        ...project,
        images: [...project.images, ...imageEntries, ...pdfEntries],
      });
      if (imageSubTab === "grid" && activeFolio && (imageEntries.length > 0 || pdfEntries.length > 0)) {
        onFolioConsumed?.();
      }
      setConverting(false);
      section.setUploadModal(false);
      section.setDragging(false);
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "upload failed");
    } finally {
      setConverting(false);
      setPdfProgress(null);
    }
  };

  const submitEditFolio = async() => {
    if (!editFolioModal) return;
    const imageId = editFolioModal.id;
    const newFolio = editFolioValue || undefined;
    await apiFetchOrThrow(`/api/projects/${project.id}/images/${imageId}`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ folio: newFolio ?? null }),
    });
    onUpdateProject({
      ...project,
      images: project.images.map((img) => 
        img.id === imageId ? { ...img, folio: newFolio } : img,
      ),
    });
    setEditFolioModal(null);
  }

  const totalImagePages = Math.ceil(sortedImages.length / ITEMS_PER_PAGE);
  const pagedImages = sortedImages.slice(
    section.page * ITEMS_PER_PAGE,
    (section.page + 1) * ITEMS_PER_PAGE,
  );
  return (
    <>
      <div className="mt-6" onClick={() => section.clearSelection()}>
        <div className="flex gap-2 mb-4" onClick={(e) => e.stopPropagation()}>
          {(["grid", "batch"] as const).map((t) => (
            <button
              key={t}
              onClick={() => onImageSubTabChange(t)}
              className={`px-3 py-1 text-xs rounded-full cursor-pointer transition-colors ${
                imageSubTab === t ? "bg-white/20 text-white" : "text-white/50 hover:text-white/70"
              }`}
            >
              {t === "grid" ? "images" : "batch run"}
            </button>
          ))}
        </div>

        {imageSubTab === "batch" ? (
          <BatchTab
            batchImages={batchImages}
            folioSequence={batchFolioSequence}
            onRunBatch={onRunBatch}
          />
        ) : project.images.length === 0 ? (
          <p className="text-white/70 text-sm">no images yet</p>
        ) : (
          <AssetGrid
            pagedItems={pagedImages}
            pageOffset={section.page * ITEMS_PER_PAGE}
            section={section}
            usedNames={usedNames.images}
            groupBy={(img) => img.sourceName || "no source"}
            totalPages={totalImagePages}
            renderThumbnail={(img) =>
              img.src ? (
                <>
                  <AuthImage
                    src={img.src}
                    alt={img.name}
                    className="w-full h-full object-cover"
                  />
                  {img.folio && (
                    <span className="absolute top-1 left-1 bg-[#1D3335]/80 text-white text-[10px] font-mono px-1.5 py-0.5 rounded">
                      {img.folio}
                    </span>
                  )}
                </>
              ) : null
            }
            getItemBadge={(name) =>
              getImageProgress(name, project.annotations ?? [], project.meiFiles ?? [])?.badge ?? null
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
              onClick: () => {
                setQuickLookId(section.menu!.id);
                setQuickLookTab("preview");
                setQuickLookDims(null);
                section.setMenu(null);
              },
            },
            {
              label: "Use Image",
              onClick: () => {
                const img = project.images.find(
                  (i) => i.id === section.menu!.id,
                );
                if (img && !usedNames.images.includes(img.name)) {
                  onUsedNamesChange({
                    ...usedNames,
                    images: [...usedNames.images, img.name],
                  });
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
                const img = project.images.find(
                  (i) => i.id === section.menu!.id,
                )!;
                section.setRenameModal({ id: section.menu!.id });
                section.setRenameName(img.name);
                section.setMenu(null);
              },
            },
            {
              label: "Edit Folio",
              onClick: () => {
                const img = project.images.find((i) => i.id === section.menu!.id)!;
                setEditFolioModal({ id: section.menu!.id });
                setEditFolioValue(img.folio ?? "");
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

      {editFolioModal && (
        <EditFolioModal
          image={project.images.find((i) => i.id === editFolioModal.id)!}
          images={project.images}
          folioOptions={cantusFolios}
          value={editFolioValue}
          onChange={setEditFolioValue}
          onSubmit={submitEditFolio}
          onClose={() => setEditFolioModal(null)}
        />
      )}
      {quickLookId &&
        (() => {
          const img = project.images.find((i) => i.id === quickLookId)!;
          const isUsed = usedNames.images.includes(img.name);
          return (
            <QuickLookModal onClose={() => setQuickLookId(null)}>
              <div className="flex gap-2">
                {(["preview", "info"] as const).map((tab) => (
                  <button
                    key={tab}
                    onClick={() => setQuickLookTab(tab)}
                    className={`px-4 py-1 rounded-lg text-sm font-semibold transition-colors cursor-pointer
                                ${quickLookTab === tab ? "bg-white text-[#1D3335]" : "text-white/60 hover:text-white/90"}`}
                  >
                    {tab}
                  </button>
                ))}
              </div>
              {quickLookTab === "preview" ? (
                <div className="flex items-center justify-center bg-[#C8E6E3]/20 rounded-xl overflow-hidden max-h-[60vh]">
                  {img.src ? (
                    <AuthImage
                      src={img.src}
                      alt={img.name}
                      className="object-contain max-h-[60vh] w-full"
                      onLoad={(e) => {
                        const el = e.currentTarget;
                        setQuickLookDims({
                          w: el.naturalWidth,
                          h: el.naturalHeight,
                        });
                      }}
                    />
                  ) : (
                    <span className="text-white/40 text-sm py-16">
                      {img.name}
                    </span>
                  )}
                </div>
              ) : quickLookMeta ? (
                <div className="flex flex-col gap-2 text-sm text-white/70 font-mono">
                  <div className="flex justify-between gap-4">
                    <span>name</span>
                    <span className="text-white truncate">{img.name}</span>
                  </div>
                  {img.folio && (
                    <div className="flex justify-between gap-4">
                      <span>folio</span>
                      <span className="text-white">{img.folio}</span>
                    </div>
                  )}
                  <div className="flex justify-between gap-4">
                    <span>type</span>
                    <span className="text-white">{quickLookMeta.mimeType}</span>
                  </div>
                  <div className="flex justify-between gap-4">
                    <span>size</span>
                    <span className="text-white">
                      {formatBytes(quickLookMeta.sizeBytes)}
                    </span>
                  </div>
                  {quickLookDims && (
                    <div className="flex justify-between gap-4">
                      <span>dimensions</span>
                      <span className="text-white">
                        {quickLookDims.w} × {quickLookDims.h} px
                      </span>
                    </div>
                  )}
                  {quickLookMeta.createdAt && (
                    <div className="flex justify-between gap-4">
                      <span>uploaded</span>
                      <span className="text-white">
                        {new Date(quickLookMeta.createdAt).toLocaleDateString()}
                      </span>
                    </div>
                  )}
                </div>
              ) : (
                <p className="text-white/40 text-sm text-center py-8">
                  loading...
                </p>
              )}
              <div className="flex gap-3 justify-center">
                {!isUsed && (
                  <button
                    onClick={() => {
                      onUsedNamesChange({
                        ...usedNames,
                        images: [...usedNames.images, img.name],
                      });
                      setValidationError(null);
                      setQuickLookId(null);
                    }}
                    className="px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
                  >
                    Use Image
                  </button>
                )}
                <button
                  onClick={() => {
                    deleteImage(quickLookId);
                    setQuickLookId(null);
                  }}
                  className="px-5 py-2 border-2 border-white/40 text-white rounded-xl hover:opacity-90 cursor-pointer text-sm"
                >
                  Delete Image
                </button>
              </div>
            </QuickLookModal>
          );
        })()}

      {section.uploadModal && (
        <Modal
          onClose={() => {
            if (!converting) {
              section.setUploadModal(false);
              section.setDragging(false);
            }
          }}
        >
          <h2 className="text-xl text-[#1D3335] text-center">upload image</h2>
          {converting ? (
            <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-[#1D3335]/30 bg-white/40 py-12">
              {pdfProgress ? (
                <>
                  <p className="text-sm text-[#1D3335] text-center">
                    converting PDF pages... {pdfProgress.done} /{" "}
                    {pdfProgress.total}
                  </p>
                  <div className="w-full bg-white/30 rounded-full h-3 overflow-hidden">
                    <div
                      className="h-full bg-[#1E6B70] rounded-full transition-all duration-100"
                      style={{
                        width: `${(pdfProgress.done / pdfProgress.total) * 100}%`,
                      }}
                    />
                  </div>
                </>
              ) : (
                <p className="text-sm text-[#1D3335] text-center">
                  uploading...
                </p>
              )}
            </div>
          ) : (
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
                handleFiles(e.dataTransfer.files);
              }}
              onClick={() => fileInputRef.current?.click()}
              label={
                imageSubTab === "batch"
                  ? "drag & drop images or PDFs here, in manuscript order"
                  : "drag & drop images, folders, or PDFs here"
              }
            >
              <div className="flex gap-4 text-sm text-[#1D3335]">
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    fileInputRef.current?.click();
                  }}
                  className="underline hover:opacity-70 cursor-pointer"
                >
                  select files
                </button>
                <span className="text-[#1D3335]/40">or</span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    folderInputRef.current?.click();
                  }}
                  className="underline hover:opacity-70 cursor-pointer"
                >
                  select folder
                </button>
              </div>
            </FileDropZone>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,application/pdf"
            multiple
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
            }}
          />
          <input
            ref={folderInputRef}
            type="file"
            // @ts-expect-error
            webkitdirectory=""
            className="hidden"
            onChange={(e) => {
              if (e.target.files) handleFiles(e.target.files);
            }}
          />
          {uploadError && (
            <p className="text-red-600 text-sm text-center">{uploadError}</p>
          )}
        </Modal>
      )}
    </>
  );
}
