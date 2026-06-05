import { useEffect, useRef, useState } from "react";
import type { Project, ProjectModel } from "../App";
import * as pdfjsLib from "pdfjs-dist";
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

const ITEMS_PER_PAGE = 10;

interface ProjectDetailProps {
  project: Project;
  onBack: () => void;
  onUpdateProject: (updated: Project) => void;
}

export default function ProjectDetail({
  project,
  onBack,
  onUpdateProject,
}: ProjectDetailProps) {
  const [activeTab, setActiveTab] = useState<"images" | "models">("images");

  // image state
  const [imageMenu, setImageMenu] = useState<{
    id: string;
    x: number;
    y: number;
  } | null>(null);
  const [renameModal, setRenameModal] = useState<{ id: string } | null>(null);
  const [renameName, setRenameName] = useState("");
  const [uploadModal, setUploadModal] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [lastSelectedIdx, setLastSelectedIdx] = useState<number | null>(null);
  const [usedImageNames, setUsedImageNames] = useState<string[]>([]);
  const [usedModelNames, setUsedModelNames] = useState<string[]>([]);
  const [quickLookId, setQuickLookId] = useState<string | null>(null);
  const [converting, setConverting] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);

  // model state
  const [modelMenu, setModelMenu] = useState<{
    id: string;
    x: number;
    y: number;
  } | null>(null);
  const [modelRenameModal, setModelRenameModal] = useState<{
    id: string;
  } | null>(null);
  const [modelRenameName, setModelRenameName] = useState("");
  const [modelUploadModal, setModelUploadModal] = useState(false);
  const [modelDragging, setModelDragging] = useState(false);
  const [selectedModelIds, setSelectedModelIds] = useState<Set<string>>(
    new Set(),
  );
  const [lastSelectedModelIdx, setLastSelectedModelIdx] = useState<
    number | null
  >(null);
  const modelFileInputRef = useRef<HTMLInputElement>(null);

  // pagination state
  const [imagePage, setImagePage] = useState(0);
  const [modelPage, setModelPage] = useState(0);

  const switchTab = (tab: "images" | "models") => {
    setActiveTab(tab);
    setSelectedIds(new Set());
    setSelectedModelIds(new Set());
    setLastSelectedIdx(null);
    setLastSelectedModelIdx(null);
    setImagePage(0);
    setModelPage(0);
  };

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setImageMenu(null);
        setRenameModal(null);
        setSelectedIds(new Set());
        setLastSelectedIdx(null);
        setModelMenu(null);
        setModelRenameModal(null);
        setSelectedModelIds(new Set());
        setLastSelectedModelIdx(null);
        setQuickLookId(null);
      }
      if (
        e.key === "Delete" &&
        activeTab === "images" &&
        selectedIds.size > 0
      ) {
        onUpdateProject({
          ...project,
          images: project.images.filter((img) => !selectedIds.has(img.id)),
        });
        setSelectedIds(new Set());
        setLastSelectedIdx(null);
      }
      if (
        e.key === "Delete" &&
        activeTab === "models" &&
        selectedModelIds.size > 0
      ) {
        onUpdateProject({
          ...project,
          models: project.models.filter((m) => !selectedModelIds.has(m.id)),
        });
        setSelectedModelIds(new Set());
        setLastSelectedModelIdx(null);
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [selectedIds, selectedModelIds, activeTab, project, onUpdateProject]);

  useEffect(() => {
    const max = Math.max(
      0,
      Math.ceil(project.images.length / ITEMS_PER_PAGE) - 1,
    );
    setImagePage((p) => Math.min(p, max));
  }, [project.images.length]);

  useEffect(() => {
    const max = Math.max(
      0,
      Math.ceil(project.models.length / ITEMS_PER_PAGE) - 1,
    );
    setModelPage((p) => Math.min(p, max));
  }, [project.models.length]);

  // image actions
  const deleteImage = (id: string) => {
    onUpdateProject({
      ...project,
      images: project.images.filter((img) => img.id !== id),
    });
    setImageMenu(null);
  };

  const renameImage = () => {
    const current = project.images.find((img) => img.id === renameModal?.id);
    onUpdateProject({
      ...project,
      images: project.images.map((img) =>
        img.id === renameModal?.id
          ? { ...img, name: renameName.trim() || current!.name }
          : img,
      ),
    });
    setRenameModal(null);
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

    const imageEntries = imageFiles.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name,
      src: URL.createObjectURL(f),
    }));

    const pdfEntries = (await Promise.all(pdfFiles.map(pdfToImages)))
      .flat()
      .map(({ name, src }) => ({ id: crypto.randomUUID(), name, src }));

    onUpdateProject({
      ...project,
      images: [...project.images, ...imageEntries, ...pdfEntries],
    });
    setConverting(false);
    setUploadModal(false);
    setDragging(false);
  };

  const handleImageClick = (e: React.MouseEvent, id: string, idx: number) => {
    e.stopPropagation();
    if (e.shiftKey) {
      e.preventDefault();
      setSelectedIds((prev) => {
        const next = new Set(prev);
        if (lastSelectedIdx !== null && lastSelectedIdx !== idx) {
          const lo = Math.min(lastSelectedIdx, idx);
          const hi = Math.max(lastSelectedIdx, idx);
          project.images.slice(lo, hi + 1).forEach((img) => next.add(img.id));
        } else {
          next.has(id) ? next.delete(id) : next.add(id);
        }
        return next;
      });
      setLastSelectedIdx(idx);
    } else {
      if (selectedIds.has(id)) {
        setSelectedIds((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      } else {
        setSelectedIds(new Set([id]));
        setLastSelectedIdx(idx);
      }
    }
  };

  // model actions
  const deleteModel = (id: string) => {
    onUpdateProject({
      ...project,
      models: project.models.filter((m) => m.id !== id),
    });
    setModelMenu(null);
  };

  const renameModel = () => {
    const current = project.models.find((m) => m.id === modelRenameModal?.id);
    onUpdateProject({
      ...project,
      models: project.models.map((m) =>
        m.id === modelRenameModal?.id
          ? { ...m, name: modelRenameName.trim() || current!.name }
          : m,
      ),
    });
    setModelRenameModal(null);
  };

  const handleModelFiles = (files: FileList | File[]) => {
    const valid = Array.from(files).filter((f) => /\.(h5|hdf5)$/i.test(f.name));
    if (valid.length === 0) return;
    const entries: ProjectModel[] = valid.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name,
    }));
    onUpdateProject({ ...project, models: [...project.models, ...entries] });
    setModelUploadModal(false);
    setModelDragging(false);
  };

  const handleModelClick = (e: React.MouseEvent, id: string, idx: number) => {
    e.stopPropagation();
    if (e.shiftKey) {
      e.preventDefault();
      setSelectedModelIds((prev) => {
        const next = new Set(prev);
        if (lastSelectedModelIdx !== null && lastSelectedModelIdx !== idx) {
          const lo = Math.min(lastSelectedModelIdx, idx);
          const hi = Math.max(lastSelectedModelIdx, idx);
          project.models.slice(lo, hi + 1).forEach((m) => next.add(m.id));
        } else {
          next.has(id) ? next.delete(id) : next.add(id);
        }
        return next;
      });
      setLastSelectedModelIdx(idx);
    } else {
      if (selectedModelIds.has(id)) {
        setSelectedModelIds((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      } else {
        setSelectedModelIds(new Set([id]));
        setLastSelectedModelIdx(idx);
      }
    }
  };

  const totalImagePages = Math.ceil(project.images.length / ITEMS_PER_PAGE);
  const pagedImages = project.images.slice(
    imagePage * ITEMS_PER_PAGE,
    (imagePage + 1) * ITEMS_PER_PAGE,
  );

  const totalModelPages = Math.ceil(project.models.length / ITEMS_PER_PAGE);
  const pagedModels = project.models.slice(
    modelPage * ITEMS_PER_PAGE,
    (modelPage + 1) * ITEMS_PER_PAGE,
  );

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 pt-10 pb-48 relative">
      <div
        className={`absolute inset-0 z-30 bg-black/30 transition-opacity pointer-events-none
                            ${uploadModal || !!renameModal || modelUploadModal || !!modelRenameModal ? "opacity-100" : "opacity-0"}`}
      />

      {/* main layout */}
      <div className="flex gap-8 max-w-6xl mx-auto">
        <div className="flex-1 min-w-0">
          {/* header */}
          <div className="flex items-center gap-4 mb-8">
            <button
              onClick={onBack}
              className="text-white text-2xl hover:opacity-70 transition-opacity cursor-pointer"
            >
              ←
            </button>
            <h1 className="text-4xl font-bold italic text-white">
              {project.name}
            </h1>

            {activeTab === "images" ? (
              <button
                onClick={() => setUploadModal(true)}
                className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer"
              >
                + new image
              </button>
            ) : (
              <button
                onClick={() => setModelUploadModal(true)}
                className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer"
              >
                + upload model
              </button>
            )}

            {activeTab === "images" && selectedIds.size > 0 && (
              <>
                <button
                  onClick={() => {
                    const names = project.images
                      .filter((img) => selectedIds.has(img.id))
                      .map((img) => img.name);
                    setUsedImageNames((prev) => [
                      ...prev,
                      ...names.filter((n) => !prev.includes(n)),
                    ]);
                    setSelectedIds(new Set());
                    setLastSelectedIdx(null);
                    setValidationError(null);
                  }}
                  className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  use {selectedIds.size} image{selectedIds.size > 1 ? "s" : ""}
                </button>
                <button
                  onClick={() => {
                    onUpdateProject({
                      ...project,
                      images: project.images.filter(
                        (img) => !selectedIds.has(img.id),
                      ),
                    });
                    setSelectedIds(new Set());
                    setLastSelectedIdx(null);
                  }}
                  className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  delete {selectedIds.size} image
                  {selectedIds.size > 1 ? "s" : ""}
                </button>
              </>
            )}

            {activeTab === "models" && selectedModelIds.size > 0 && (
              <>
                <button
                  onClick={() => {
                    const names = project.models
                      .filter((m) => selectedModelIds.has(m.id))
                      .map((m) => m.name);
                    setUsedModelNames((prev) => [
                      ...prev,
                      ...names.filter((n) => !prev.includes(n)),
                    ]);
                    setSelectedModelIds(new Set());
                    setLastSelectedModelIdx(null);
                    setValidationError(null);
                  }}
                  className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  use {selectedModelIds.size} model
                  {selectedModelIds.size > 1 ? "s" : ""}
                </button>
                <button
                  onClick={() => {
                    onUpdateProject({
                      ...project,
                      models: project.models.filter(
                        (m) => !selectedModelIds.has(m.id),
                      ),
                    });
                    setSelectedModelIds(new Set());
                    setLastSelectedModelIdx(null);
                  }}
                  className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  delete {selectedModelIds.size} model
                  {selectedModelIds.size > 1 ? "s" : ""}
                </button>
              </>
            )}
          </div>

          {/* tab bar + content */}
          <div>
            <div className="flex items-end">
              {(["images", "models"] as const).map((tab, i) => (
                <button
                  key={tab}
                  onClick={() => switchTab(tab)}
                  className={`relative px-8 pt-3 pb-2 text-2xl font-bold italic rounded-t-xl cursor-pointer transition-colors
                ${
                  activeTab === tab
                    ? "text-white border border-white/50 border-b-0 bg-[#4AADAA] z-10"
                    : "text-white/50 hover:text-white/70 border border-transparent"
                }
                ${i > 0 ? "-ml-px" : ""}`}
                >
                  {tab}
                </button>
              ))}
              <div className="flex-1 border-b border-white/50" />
            </div>

            {/* images tab */}
            {activeTab === "images" && (
              <div
                className="mt-6"
                onClick={() => {
                  setSelectedIds(new Set());
                  setLastSelectedIdx(null);
                }}
              >
                {project.images.length === 0 ? (
                  <p className="text-white/70 text-sm">no images yet</p>
                ) : (
                  <>
                    <div
                      className="grid grid-cols-5 gap-4"
                      onMouseDown={(e) => {
                        if (e.shiftKey) e.preventDefault();
                      }}
                    >
                      {pagedImages.map((img, pageIdx) => {
                        const idx = imagePage * ITEMS_PER_PAGE + pageIdx;
                        return (
                          <div key={img.id} className="flex flex-col gap-2">
                            <div
                              className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer
                                                transition-shadow
                                                ${selectedIds.has(img.id) ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                                                ${usedImageNames.includes(img.name) ? "opacity-40 cursor-default" : ""}`}
                              onClick={(e) => {
                                if (!usedImageNames.includes(img.name))
                                  handleImageClick(e, img.id, idx);
                              }}
                            >
                              {img.src && (
                                <img
                                  src={img.src}
                                  alt={img.name}
                                  className="w-full h-full object-cover"
                                />
                              )}
                            </div>
                            <div className="flex items-center justify-between gap-1">
                              <span
                                className={`text-sm text-white truncate ${usedImageNames.includes(img.name) ? "opacity-40" : ""}`}
                              >
                                {img.name}
                              </span>
                              {!usedImageNames.includes(img.name) && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setImageMenu({
                                      id: img.id,
                                      x: e.clientX,
                                      y: e.clientY,
                                    });
                                  }}
                                  className="text-white text-lg leading-none hover:opacity-70 cursor-pointer flex-shrink-0"
                                >
                                  ⋮
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    {totalImagePages > 1 && (
                      <div className="flex items-center justify-center gap-4 mt-6 text-white text-sm">
                        <button
                          onClick={() => {
                            setImagePage((p) => p - 1);
                            setLastSelectedIdx(null);
                          }}
                          disabled={imagePage === 0}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer"
                        >
                          ←
                        </button>
                        <span>
                          page {imagePage + 1} of {totalImagePages}
                        </span>
                        <button
                          onClick={() => {
                            setImagePage((p) => p + 1);
                            setLastSelectedIdx(null);
                          }}
                          disabled={imagePage === totalImagePages - 1}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer"
                        >
                          →
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}

            {/* models tab */}
            {activeTab === "models" && (
              <div
                className="mt-6"
                onClick={() => {
                  setSelectedModelIds(new Set());
                  setLastSelectedModelIdx(null);
                }}
              >
                {project.models.length === 0 ? (
                  <p className="text-white/70 text-sm">no models yet</p>
                ) : (
                  <>
                    <div
                      className="grid grid-cols-5 gap-4"
                      onMouseDown={(e) => {
                        if (e.shiftKey) e.preventDefault();
                      }}
                    >
                      {pagedModels.map((model, pageIdx) => {
                        const idx = modelPage * ITEMS_PER_PAGE + pageIdx;
                        return (
                          <div key={model.id} className="flex flex-col gap-2">
                            <div
                              className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer
                            transition-shadow flex items-center justify-center
                            ${selectedModelIds.has(model.id) ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                            ${usedModelNames.includes(model.name) ? "opacity-40 cursor-default" : ""}
                            `}
                              onClick={(e) => {
                                if (!usedModelNames.includes(model.name))
                                  handleModelClick(e, model.id, idx);
                              }}
                            >
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
                                  H5
                                </text>
                              </svg>
                            </div>
                            <div className="flex items-center justify-between gap-1">
                              <span
                                className={`text-sm text-white truncate ${usedModelNames.includes(model.name) ? "opacity-40" : ""}`}
                              >
                                {model.name}
                              </span>
                              {!usedModelNames.includes(model.name) && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    setModelMenu({
                                      id: model.id,
                                      x: e.clientX,
                                      y: e.clientY,
                                    });
                                  }}
                                  className="text-white text-lg leading-none hover:opacity-70 cursor-pointer flex-shrink-0"
                                >
                                  ⋮
                                </button>
                              )}
                            </div>
                          </div>
                        );
                      })}
                    </div>
                    {totalModelPages > 1 && (
                      <div className="flex items-center justify-center gap-4 mt-6 text-white text-sm">
                        <button
                          onClick={() => {
                            setModelPage((p) => p - 1);
                            setLastSelectedModelIdx(null);
                          }}
                          disabled={modelPage === 0}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer"
                        >
                          ←
                        </button>
                        <span>
                          page {modelPage + 1} of {totalModelPages}
                        </span>
                        <button
                          onClick={() => {
                            setModelPage((p) => p + 1);
                            setLastSelectedModelIdx(null);
                          }}
                          disabled={modelPage === totalModelPages - 1}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer"
                        >
                          →
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </div>
        </div>
        {/* end left column */}

        {/* right sidebar */}
        <div className="flex flex-col gap-3 w-52 flex-shrink-0 pt-2">
          <button
            onClick={() => {
              if (usedModelNames.length === 0) {
                setValidationError("must select at least one model!");
              } else if (usedImageNames.length === 0) {
                setValidationError("must select at least one image!");
              } else {
                setValidationError(null);
              }
            }}
            className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1"
          >
            continue &rarr;
          </button>
          <div className="bg-[#C8E6E3]/40 rounded-2xl p-4 flex flex-col gap-2 text-white text-sm">
            <span className="text-white/80">selected:</span>
            {usedModelNames.map((name) => (
              <div key={name} className="flex items-center justify-between">
                <span className="truncate flex-1 mr-2">{name}</span>
                <button
                  onClick={() =>
                    setUsedModelNames((prev) => prev.filter((n) => n !== name))
                  }
                  className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
                >
                  ×
                </button>
              </div>
            ))}
            <hr className="border-white/40 my-1" />
            {usedImageNames.map((name) => (
              <div key={name} className="flex items-center justify-between">
                <span className="truncate flex-1 mr-2">{name}</span>
                <button
                  onClick={() =>
                    setUsedImageNames((prev) => prev.filter((n) => n !== name))
                  }
                  className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
          {validationError && (
            <p className="text-white text-xs text-center">{validationError}</p>
          )}
        </div>
      </div>
      {/* end flex layout */}

      {/* image context menu */}
      {imageMenu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setImageMenu(null)}
          />
          <div
            className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
            style={{ top: imageMenu.y + 8, left: imageMenu.x - 80 }}
          >
            <button
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
              onClick={() => {
                setQuickLookId(imageMenu!.id);
                setImageMenu(null);
              }}>
              Quick Look
            </button>
            <button
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
              onClick={() => {
                const img = project.images.find((i) => i.id === imageMenu!.id);
                if (img && !usedImageNames.includes(img.name)) {
                  setUsedImageNames((prev) => [...prev, img.name]);
                }
                setImageMenu(null);
                setValidationError(null);
              }}
            >
              Use Image
            </button>
            <button
              onClick={() => deleteImage(imageMenu.id)}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Delete Image
            </button>
            <button
              onClick={() => {
                const img = project.images.find((i) => i.id === imageMenu.id)!;
                setRenameModal({ id: imageMenu.id });
                setRenameName(img.name);
                setImageMenu(null);
              }}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Rename Image
            </button>
          </div>
        </>
      )}

      {/* model context menu */}
      {modelMenu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setModelMenu(null)}
          />
          <div
            className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
            style={{ top: modelMenu.y + 8, left: modelMenu.x - 80 }}
          >
            <button
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
              onClick={() => {
                const model = project.models.find(
                  (m) => m.id === modelMenu!.id,
                );
                if (model && !usedModelNames.includes(model.name)) {
                  setUsedModelNames((prev) => [...prev, model.name]);
                }
                setModelMenu(null);
                setValidationError(null);
              }}
            >
              Use Model
            </button>
            <button
              onClick={() => deleteModel(modelMenu.id)}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Delete Model
            </button>
            <button
              onClick={() => {
                const m = project.models.find((m) => m.id === modelMenu.id)!;
                setModelRenameModal({ id: modelMenu.id });
                setModelRenameName(m.name);
                setModelMenu(null);
              }}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Rename Model
            </button>
          </div>
        </>
      )}

      {/* image rename modal */}
      {renameModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setRenameModal(null)}
          />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl">
            <button
              onClick={() => setRenameModal(null)}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              ✕
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">rename image</h2>
            <input
              autoFocus
              value={renameName}
              onChange={(e) => setRenameName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") renameImage();
              }}
              className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm"
            />
            <button
              onClick={renameImage}
              className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
            >
              rename image
            </button>
          </div>
        </>
      )}

      {quickLookId && (() => {
        const img = project.images.find((i) => i.id === quickLookId)!;
        const isUsed = usedImageNames.includes(img.name);
        return (
          <>
            <div
              className="fixed inset-0 z-40 bg-black/60"
              onClick={() => setQuickLookId(null)}
            />
            <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
              <div className="relative bg-[#1D3335] rounded-2xl shadow-2xl p-6 flex flex-col gap-4 max-w-2xl w-full mx-4 pointer-events-auto animate-fade-in">
                {/* × close button */}
                <button
                  onClick={() => setQuickLookId(null)}
                  className="absolute top-3 right-4 text-white/60 hover:text-white text-2xl leading-none cursor-pointer"
                >
                  ×
                </button>
                {/* image */}
                <div className="flex items-center justify-center bg-[#C8E6E3]/20 rounded-xl overflow-hidden max-h-[60vh]">
                  {img.src
                    ? <img src={img.src} alt={img.name} className="object-contain max-h-[60vh] w-full" />
                    : <span className="text-white/40 text-sm py-16">{img.name}</span>
                  }
                </div>
                {/* action buttons */}
                <div className="flex gap-3 justify-center">
                  {!isUsed && (
                    <button
                      onClick={() => {
                        setUsedImageNames((prev) => [...prev, img.name]);
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
              </div>
            </div>
          </>
        );
      })()}
      
      {/* model rename modal */}
      {modelRenameModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => setModelRenameModal(null)}
          />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl">
            <button
              onClick={() => setModelRenameModal(null)}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              ✕
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">rename model</h2>
            <input
              autoFocus
              value={modelRenameName}
              onChange={(e) => setModelRenameName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") renameModel();
              }}
              className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm"
            />
            <button
              onClick={renameModel}
              className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
            >
              rename model
            </button>
          </div>
        </>
      )}

      {/* image upload modal */}
      {uploadModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => {
              setUploadModal(false);
              setDragging(false);
            }}
          />

          <div
            className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                                w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-6 relative shadow-2xl"
          >
            <button
              onClick={() => {
                if (!converting) {
                  setUploadModal(false);
                  setDragging(false);
                }
              }}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              x
            </button>

            <h2 className="text-xl text-[#1D3335] text-center">upload image</h2>

            {converting ? (
              <div
                className="flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
                                        border-[#1D3335]/30 bg-white/40 py-12"
              >
                <p className="text-sm text-[#1D3335] text-center">
                  converting PDF pages...
                </p>
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragEnter={(e) => {
                  e.preventDefault();
                  setDragging(true);
                }}
                onDragLeave={() => setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  handleFiles(e.dataTransfer.files);
                }}
                className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
                            py-12 cursor-pointer transition-colors
                            ${
                              dragging
                                ? "border-[#1E6B70] bg-[#1E6B70]/10"
                                : "border-[#1D3335]/30 bg-white/40 hover:bg-white/60"
                            }`}
              >
                <span className="text-3xl">↑</span>
                <p className="text-sm text-[#1D3335] text-center">
                  drag & drop images, folders, or PDFs here
                </p>
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
              </div>
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
          </div>
        </>
      )}

      {/* model upload modal */}
      {modelUploadModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => {
              setModelUploadModal(false);
              setModelDragging(false);
            }}
          />
          <div
            className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
              w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-6 relative shadow-2xl"
          >
            <button
              onClick={() => {
                setModelUploadModal(false);
                setModelDragging(false);
              }}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              x
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">upload model</h2>
            <div
              onClick={() => modelFileInputRef.current?.click()}
              onDragOver={(e) => {
                e.preventDefault();
                setModelDragging(true);
              }}
              onDragEnter={(e) => {
                e.preventDefault();
                setModelDragging(true);
              }}
              onDragLeave={() => setModelDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                handleModelFiles(e.dataTransfer.files);
              }}
              className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
                py-12 cursor-pointer transition-colors
                ${modelDragging ? "border-[#1E6B70] bg-[#1E6B70]/10" : "border-[#1D3335]/30 bg-white/40 hover:bg-white/60"}`}
            >
              <span className="text-3xl">↑</span>
              <p className="text-sm text-[#1D3335] text-center">
                drag & drop .h5 or .hdf5 files here
              </p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  modelFileInputRef.current?.click();
                }}
                className="text-sm text-[#1D3335] underline hover:opacity-70 cursor-pointer"
              >
                select files
              </button>
            </div>
            <input
              ref={modelFileInputRef}
              type="file"
              accept=".h5,.hdf5"
              multiple
              className="hidden"
              onChange={(e) => {
                if (e.target.files) handleModelFiles(e.target.files);
              }}
            />
          </div>
        </>
      )}
    </div>
  );
}
