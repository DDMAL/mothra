import { useEffect, useRef, useState } from "react";
import type { Project, ProjectModel, MeiFile } from "../App";
import { useAssetSection, ITEMS_PER_PAGE } from "../hooks/useAssetSection";
import RenameModal from "./RenameModal";
import * as pdfjsLib from "pdfjs-dist";
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  "pdfjs-dist/build/pdf.worker.min.mjs",
  import.meta.url,
).href;

const STEPS = [
  "interactive classifier",
  "encoding",
  "neon.js",
  "send to cantus ultimus",
];

interface ProjectDetailProps {
  project: Project;
  onBack: () => void;
  onContinue: () => void;
  onUpdateProject: (updated: Project) => void;
  usedNames: { images: string[]; models: string[] };
  onUsedNamesChange: (names: { images: string[]; models: string[] }) => void;
  stepsUnlocked: number;
  onStepClick: (step: number) => void;
}

export default function ProjectDetail({
  project,
  onBack,
  onContinue,
  onUpdateProject,
  usedNames,
  onUsedNamesChange,
  stepsUnlocked,
  onStepClick,
}: ProjectDetailProps) {
  const [activeTab, setActiveTab] = useState<
    "images" | "models" | "annotations" | "mei produced"
  >("images");
  const [quickLookId, setQuickLookId] = useState<string | null>(null);
  const [converting, setConverting] = useState(false);
  const [validationError, setValidationError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const folderInputRef = useRef<HTMLInputElement>(null);
  const modelFileInputRef = useRef<HTMLInputElement>(null);

  const imgSection = useAssetSection(project.images);
  const mdlSection = useAssetSection(project.models);
  const meiSection = useAssetSection(project.meiFiles);
  
  const [meiLookId, setMeiLookId] = useState<string | null>(null);

  const switchTab = (tab: "images" | "models" | "annotations" | "mei produced") => {
    setActiveTab(tab);
    imgSection.clearSelection();
    mdlSection.clearSelection();
    meiSection.clearSelection();
    imgSection.setPage(0);
    mdlSection.setPage(0);
  };

  const tabs = [
    "images",
    "models",
    ...(stepsUnlocked >= 1 ? ["annotations"] : []),
    ...(stepsUnlocked >= 3 ? ["mei produced"] : []),
  ] as const;

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        imgSection.setMenu(null);
        imgSection.setRenameModal(null);
        imgSection.clearSelection();
        mdlSection.setMenu(null);
        mdlSection.setRenameModal(null);
        mdlSection.clearSelection();
        meiSection.setMenu(null);
        meiSection.clearSelection();
        setMeiLookId(null);
        setQuickLookId(null);
      }
      if (
        e.key === "Delete" &&
        activeTab === "images" &&
        imgSection.selectedIds.size > 0
      ) {
        onUpdateProject({
          ...project,
          images: project.images.filter(
            (img) => !imgSection.selectedIds.has(img.id),
          ),
        });
        imgSection.clearSelection();
      }
      if (
        e.key === "Delete" &&
        activeTab === "models" &&
        mdlSection.selectedIds.size > 0
      ) {
        onUpdateProject({
          ...project,
          models: project.models.filter(
            (m) => !mdlSection.selectedIds.has(m.id),
          ),
        });
        mdlSection.clearSelection();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    imgSection.selectedIds,
    mdlSection.selectedIds,
    activeTab,
    project,
    onUpdateProject,
  ]);

  // image actions
  const deleteImage = (id: string) => {
    onUpdateProject({
      ...project,
      images: project.images.filter((img) => img.id !== id),
    });
    imgSection.setMenu(null);
  };

  const renameImage = () => {
    const current = project.images.find(
      (img) => img.id === imgSection.renameModal?.id,
    );
    onUpdateProject({
      ...project,
      images: project.images.map((img) =>
        img.id === imgSection.renameModal?.id
          ? { ...img, name: imgSection.renameName.trim() || current!.name }
          : img,
      ),
    });
    imgSection.setRenameModal(null);
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
    imgSection.setUploadModal(false);
    imgSection.setDragging(false);
  };

  // model actions
  const deleteModel = (id: string) => {
    onUpdateProject({
      ...project,
      models: project.models.filter((m) => m.id !== id),
    });
    mdlSection.setMenu(null);
  };

  const renameModel = () => {
    const current = project.models.find(
      (m) => m.id === mdlSection.renameModal?.id,
    );
    onUpdateProject({
      ...project,
      models: project.models.map((m) =>
        m.id === mdlSection.renameModal?.id
          ? { ...m, name: mdlSection.renameName.trim() || current!.name }
          : m,
      ),
    });
    mdlSection.setRenameModal(null);
  };

  const handleModelFiles = (files: FileList | File[]) => {
    const valid = Array.from(files).filter((f) => /\.(h5|hdf5)$/i.test(f.name));
    if (valid.length === 0) return;
    const entries: ProjectModel[] = valid.map((f) => ({
      id: crypto.randomUUID(),
      name: f.name,
    }));
    onUpdateProject({ ...project, models: [...project.models, ...entries] });
    mdlSection.setUploadModal(false);
    mdlSection.setDragging(false);
  };

  const handleExportMei = (file: MeiFile) => {
    const blob = new Blob([file.xmlContent ?? ""], { type: "application/xml" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = file.name;
    a.click();
    URL.revokeObjectURL(url);
  };

  const totalImagePages = Math.ceil(project.images.length / ITEMS_PER_PAGE);
  const pagedImages = project.images.slice(
    imgSection.page * ITEMS_PER_PAGE,
    (imgSection.page + 1) * ITEMS_PER_PAGE,
  );

  const totalModelPages = Math.ceil(project.models.length / ITEMS_PER_PAGE);
  const pagedModels = project.models.slice(
    mdlSection.page * ITEMS_PER_PAGE,
    (mdlSection.page + 1) * ITEMS_PER_PAGE,
  );

  const totalMeiPages = Math.ceil(project.meiFiles.length / ITEMS_PER_PAGE);
  const pagedMei = project.meiFiles.slice(
    meiSection.page * ITEMS_PER_PAGE,
    (meiSection.page + 1) * ITEMS_PER_PAGE,
  );

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 pt-10 pb-48 relative">
      <div
        className={`absolute inset-0 z-30 bg-black/30 transition-opacity pointer-events-none
          ${imgSection.uploadModal || !!imgSection.renameModal || mdlSection.uploadModal || !!mdlSection.renameModal ? "opacity-100" : "opacity-0"}`}
      />

      {/* main layout */}
      <div className="flex gap-8 max-w-6xl mx-auto">
        {/* progress sidebar */}
        <div className="w-48 shrink-0 bg-[#C8E6E3]/30 rounded-2xl p-5 flex flex-col gap-2 self-start mt-[4.5rem]">
          <span className="text-white/60 text-sm font-medium mb-1">
            progress:
          </span>
          {STEPS.map((label, i) => {
            const stepNum = i + 1;
            const unlocked = stepsUnlocked >= stepNum;
            return (
              <button
                key={stepNum}
                disabled={!unlocked}
                onClick={() => onStepClick(stepNum)}
                className={`text-left text-sm px-3 py-2 rounded-xl transition-opacity ${
                  unlocked
                    ? "text-white hover:bg-white/10 cursor-pointer"
                    : "text-white/30 cursor-not-allowed"
                }`}
              >
                {stepNum}) {label}
              </button>
            );
          })}
        </div>

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
                onClick={() => imgSection.setUploadModal(true)}
                className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer"
              >
                + new image
              </button>
            ) : activeTab === "models" ? (
              <button
                onClick={() => mdlSection.setUploadModal(true)}
                className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer"
              >
                + upload model
              </button>
            ) : null}

            {activeTab === "images" && imgSection.selectedIds.size > 0 && (
              <>
                <button
                  onClick={() => {
                    const names = project.images
                      .filter((img) => imgSection.selectedIds.has(img.id))
                      .map((img) => img.name);
                    onUsedNamesChange({
                      ...usedNames,
                      images: [
                        ...usedNames.images,
                        ...names.filter((n) => !usedNames.images.includes(n)),
                      ],
                    });
                    imgSection.clearSelection();
                    setValidationError(null);
                  }}
                  className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  use {imgSection.selectedIds.size} image
                  {imgSection.selectedIds.size > 1 ? "s" : ""}
                </button>
                <button
                  onClick={() => {
                    onUpdateProject({
                      ...project,
                      images: project.images.filter(
                        (img) => !imgSection.selectedIds.has(img.id),
                      ),
                    });
                    imgSection.clearSelection();
                  }}
                  className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  delete {imgSection.selectedIds.size} image
                  {imgSection.selectedIds.size > 1 ? "s" : ""}
                </button>
              </>
            )}

            {activeTab === "models" && mdlSection.selectedIds.size > 0 && (
              <>
                <button
                  onClick={() => {
                    const names = project.models
                      .filter((m) => mdlSection.selectedIds.has(m.id))
                      .map((m) => m.name);
                    onUsedNamesChange({
                      ...usedNames,
                      models: [
                        ...usedNames.models,
                        ...names.filter((n) => !usedNames.models.includes(n)),
                      ],
                    });
                    mdlSection.clearSelection();
                    setValidationError(null);
                  }}
                  className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  use {mdlSection.selectedIds.size} model
                  {mdlSection.selectedIds.size > 1 ? "s" : ""}
                </button>
                <button
                  onClick={() => {
                    onUpdateProject({
                      ...project,
                      models: project.models.filter(
                        (m) => !mdlSection.selectedIds.has(m.id),
                      ),
                    });
                    mdlSection.clearSelection();
                  }}
                  className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  delete {mdlSection.selectedIds.size} model
                  {mdlSection.selectedIds.size > 1 ? "s" : ""}
                </button>
              </>
            )}
          </div>

          {/* tab bar + content */}
          <div>
            <div className="flex items-end">
              {tabs.map((tab, i) => (
                <button
                  key={tab}
                  onClick={() =>
                    switchTab(tab as "images" | "models" | "annotations")
                  }
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
              <div className="mt-6" onClick={() => imgSection.clearSelection()}>
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
                        const idx = imgSection.page * ITEMS_PER_PAGE + pageIdx;
                        return (
                          <div key={img.id} className="flex flex-col gap-2">
                            <div
                              className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer transition-shadow
                                ${imgSection.selectedIds.has(img.id) ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                                ${usedNames.images.includes(img.name) ? "opacity-40 cursor-default" : ""}`}
                              onClick={(e) => {
                                if (!usedNames.images.includes(img.name))
                                  imgSection.handleClick(e, img.id, idx);
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
                                className={`text-sm text-white truncate ${usedNames.images.includes(img.name) ? "opacity-40" : ""}`}
                              >
                                {img.name}
                              </span>
                              {!usedNames.images.includes(img.name) && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    imgSection.setMenu({
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
                          onClick={() => imgSection.setPage((p) => p - 1)}
                          disabled={imgSection.page === 0}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer"
                        >
                          ←
                        </button>
                        <span>
                          page {imgSection.page + 1} of {totalImagePages}
                        </span>
                        <button
                          onClick={() => imgSection.setPage((p) => p + 1)}
                          disabled={imgSection.page === totalImagePages - 1}
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
              <div className="mt-6" onClick={() => mdlSection.clearSelection()}>
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
                        const idx = mdlSection.page * ITEMS_PER_PAGE + pageIdx;
                        return (
                          <div key={model.id} className="flex flex-col gap-2">
                            <div
                              className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer
                                transition-shadow flex items-center justify-center
                                ${mdlSection.selectedIds.has(model.id) ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                                ${usedNames.models.includes(model.name) ? "opacity-40 cursor-default" : ""}`}
                              onClick={(e) => {
                                if (!usedNames.models.includes(model.name))
                                  mdlSection.handleClick(e, model.id, idx);
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
                                className={`text-sm text-white truncate ${usedNames.models.includes(model.name) ? "opacity-40" : ""}`}
                              >
                                {model.name}
                              </span>
                              {!usedNames.models.includes(model.name) && (
                                <button
                                  onClick={(e) => {
                                    e.stopPropagation();
                                    mdlSection.setMenu({
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
                          onClick={() => mdlSection.setPage((p) => p - 1)}
                          disabled={mdlSection.page === 0}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer"
                        >
                          ←
                        </button>
                        <span>
                          page {mdlSection.page + 1} of {totalModelPages}
                        </span>
                        <button
                          onClick={() => mdlSection.setPage((p) => p + 1)}
                          disabled={mdlSection.page === totalModelPages - 1}
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

            {activeTab === "annotations" && (
              <div className="mt-6">
                {project.annotations.length === 0 ? (
                  <p className="text-white/70 text-sm">no annotations yet</p>
                ) : (
                  <div className="grid grid-cols-5 gap-4">
                    {project.annotations.map((set) => (
                      <div key={set.id} className="flex flex-col gap-2">
                        <div className="relative aspect-square">
                          {/* bottom: txt */}
                          <div className="absolute inset-0 translate-x-2 translate-y-2 bg-[#C8E6E3]/25 rounded-xl flex items-end justify-start p-2">
                            <span className="text-[10px] text-white/50 font-mono">.txt</span>
                          </div>
                          {/* middle: json */}
                          <div className="absolute inset-0 translate-x-1 translate-y-1 bg-[#C8E6E3]/35 rounded-xl flex items-end justify-start p-2">
                            <span className="text-[10px] text-white/60 font-mono">.json</span>
                          </div>
                          {/* top: image */}
                          <div className="absolute inset-0 bg-[#C8E6E3]/50 rounded-xl overflow-hidden flex items-end justify-start p-2">
                            {set.imageSrc && (
                              <img
                                src={set.imageSrc}
                                alt={set.imageName}
                                className="absolute inset-0 w-full h-full object-cover opacity-60"
                              />
                            )}
                            <span className="relative text-[10px] text-white/80 font-mono z-10">.png</span>
                          </div>
                        </div>
                        <span className="text-sm text-white truncate">
                          {set.imageName.replace(/\.[^.]+$/, "")}
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}


            {activeTab === "mei produced" && (
              <div className="mt-6" onClick={() => meiSection.clearSelection()}>
                {project.meiFiles.length === 0 ? (
                  <p className="text-white/70 text-sm"> no mei files yet </p>
                ) : (
                  <>
                    <div
                      className="grid grid-cols-5 gap-4"
                      onMouseDown={(e) => { if (e.shiftKey) e.preventDefault(); }}
                    >
                      {pagedMei.map((file, pageIdx) => {
                        const idx = meiSection.page * ITEMS_PER_PAGE + pageIdx;
                        const selected = meiSection.selectedIds.has(file.id);
                        return (
                        <div key={file.id} className="flex flex-col gap-2">
                          <div
                            className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden flex items-center justify-center cursor-pointer transition-shadow
                              ${selected ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}`}
                            onClick={(e) => meiSection.handleClick(e, file.id, idx)}
                          >
                            <svg width="56" height="64" viewBox="0 0 56 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                              <path d="M4 0H36L56 20V60C56 62.2 54.2 64 52 64H4C1.8 64 0 62.2 0 60V4C0 1.8 1.8 0 4 0Z" fill="white" fillOpacity="0.25" />
                              <path d="M36 0L56 20H40C37.8 20 36 18.2 36 16V0Z" fill="white" fillOpacity="0.45" />
                              <text x="28" y="46" textAnchor="middle" fill="white" fontSize="14" fontWeight="bold" fontFamily="monospace">MEI</text>
                            </svg>
                          </div>
                          <div className="flex items-center justify-between gap-1">
                            <span className="text-sm text-white truncate">{file.name}</span>
                            <button
                              onClick={(e) => {
                                e.stopPropagation();
                                meiSection.setMenu({ id: file.id, x: e.clientX, y: e.clientY });
                              }}
                              className="text-white text-lg leading-none hover:opacity-70 cursor-pointer flex-shrink-0">
                                ⋮
                            </button>
                          </div>
                        </div>
                        );
                      })}
                    </div>
                    {totalMeiPages > 1 && (
                      <div className="flex items-center justify-center gap-4 mt-6 text-white text-sm">
                        <button
                          onClick={() => meiSection.setPage((p) => p - 1 )}
                          disabled={meiSection.page === 0}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer">
                            ←
                        </button>
                        <span>page {meiSection.page + 1} of {totalMeiPages}</span>
                        <button
                          onClick={() => meiSection.setPage((p) => p + 1)}
                          disabled = {meiSection.page === totalMeiPages - 1}
                          className="hover:opacity-70 disabled:opacity-30 cursor-pointer">
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
          {meiSection.selectedIds.size > 0 ? (
            <button
              onClick={() => {
                imgSection.clearSelection();
                mdlSection.clearSelection();
                meiSection.clearSelection();
              }}
              className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1"
            >
              send to cantus ultimus &rarr;
            </button>
          ) : (
            <button
              onClick={() => {
                if (stepsUnlocked === 0) {
                  if (usedNames.models.length === 0) {
                    setValidationError("must select at least one model!");
                    return;
                  }
                  if (usedNames.images.length === 0) {
                    setValidationError("must select at least one image!");
                    return;
                  }
                  setValidationError(null);
                }
                onContinue();
              }}
              className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1"
            >
              {stepsUnlocked === 0 ? "begin" : "continue"} &rarr;
            </button>
          )}
          <div className="bg-[#C8E6E3]/40 rounded-2xl p-4 flex flex-col gap-2 text-white text-sm">
            <span className="text-white/80">selected:</span>
            {usedNames.models.map((name) => (
              <div key={name} className="flex items-center justify-between">
                <span className="truncate flex-1 mr-2">{name}</span>
                <button
                  onClick={() =>
                    onUsedNamesChange({
                      ...usedNames,
                      models: usedNames.models.filter((n) => n !== name),
                    })
                  }
                  className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
                >
                  ×
                </button>
              </div>
            ))}
            <hr className="border-white/40 my-1" />
            {usedNames.images.map((name) => (
              <div key={name} className="flex items-center justify-between">
                <span className="truncate flex-1 mr-2">{name}</span>
                <button
                  onClick={() =>
                    onUsedNamesChange({
                      ...usedNames,
                      images: usedNames.images.filter((n) => n !== name),
                    })
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
      {imgSection.menu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => imgSection.setMenu(null)}
          />
          <div
            className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
            style={{ top: imgSection.menu.y + 8, left: imgSection.menu.x - 80 }}
          >
            <button
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
              onClick={() => {
                setQuickLookId(imgSection.menu!.id);
                imgSection.setMenu(null);
              }}
            >
              Quick Look
            </button>
            <button
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
              onClick={() => {
                const img = project.images.find(
                  (i) => i.id === imgSection.menu!.id,
                );
                if (img && !usedNames.images.includes(img.name)) {
                  onUsedNamesChange({
                    ...usedNames,
                    images: [...usedNames.images, img.name],
                  });
                }
                imgSection.setMenu(null);
                setValidationError(null);
              }}
            >
              Use Image
            </button>
            <button
              onClick={() => deleteImage(imgSection.menu!.id)}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Delete Image
            </button>
            <button
              onClick={() => {
                const img = project.images.find(
                  (i) => i.id === imgSection.menu!.id,
                )!;
                imgSection.setRenameModal({ id: imgSection.menu!.id });
                imgSection.setRenameName(img.name);
                imgSection.setMenu(null);
              }}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Rename Image
            </button>
          </div>
        </>
      )}

      {/* model context menu */}
      {mdlSection.menu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => mdlSection.setMenu(null)}
          />
          <div
            className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
            style={{ top: mdlSection.menu.y + 8, left: mdlSection.menu.x - 80 }}
          >
            <button
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
              onClick={() => {
                const model = project.models.find(
                  (m) => m.id === mdlSection.menu!.id,
                );
                if (model && !usedNames.models.includes(model.name)) {
                  onUsedNamesChange({
                    ...usedNames,
                    models: [...usedNames.models, model.name],
                  });
                }
                mdlSection.setMenu(null);
                setValidationError(null);
              }}
            >
              Use Model
            </button>
            <button
              onClick={() => deleteModel(mdlSection.menu!.id)}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Delete Model
            </button>
            <button
              onClick={() => {
                const m = project.models.find(
                  (m) => m.id === mdlSection.menu!.id,
                )!;
                mdlSection.setRenameModal({ id: mdlSection.menu!.id });
                mdlSection.setRenameName(m.name);
                mdlSection.setMenu(null);
              }}
              className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
            >
              Rename Model
            </button>
          </div>
        </>
      )}

      {meiSection.menu && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => meiSection.setMenu(null)}/>
          <div
            className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
            style={{ top: meiSection.menu.y + 8, left: meiSection.menu.x - 80 }}>
              <button
                className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
                onClick={() => {
                  setMeiLookId(meiSection.menu!.id);
                  meiSection.setMenu(null);
                }}>
                  View
              </button>
              <button
                className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
                onClick={() => {
                  const file = project.meiFiles.find((f) => f.id === meiSection.menu!.id)!;
                  handleExportMei(file);
                  meiSection.setMenu(null);
                }}>
                  Export
              </button>
          </div>
        </>
      )}

      {/* rename modals */}
      {imgSection.renameModal && (
        <RenameModal
          label="image"
          value={imgSection.renameName}
          onChange={imgSection.setRenameName}
          onSubmit={renameImage}
          onClose={() => imgSection.setRenameModal(null)}
        />
      )}
      {mdlSection.renameModal && (
        <RenameModal
          label="model"
          value={mdlSection.renameName}
          onChange={mdlSection.setRenameName}
          onSubmit={renameModel}
          onClose={() => mdlSection.setRenameModal(null)}
        />
      )}

      {/* quick look modal */}
      {quickLookId &&
        (() => {
          const img = project.images.find((i) => i.id === quickLookId)!;
          const isUsed = usedNames.images.includes(img.name);
          return (
            <>
              <div
                className="fixed inset-0 z-40 bg-black/60"
                onClick={() => setQuickLookId(null)}
              />
              <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
                <div className="relative bg-[#1D3335] rounded-2xl shadow-2xl p-6 flex flex-col gap-4 max-w-2xl w-full mx-4 pointer-events-auto animate-fade-in">
                  <button
                    onClick={() => setQuickLookId(null)}
                    className="absolute top-3 right-4 text-white/60 hover:text-white text-2xl leading-none cursor-pointer"
                  >
                    ×
                  </button>
                  <div className="flex items-center justify-center bg-[#C8E6E3]/20 rounded-xl overflow-hidden max-h-[60vh]">
                    {img.src ? (
                      <img
                        src={img.src}
                        alt={img.name}
                        className="object-contain max-h-[60vh] w-full"
                      />
                    ) : (
                      <span className="text-white/40 text-sm py-16">
                        {img.name}
                      </span>
                    )}
                  </div>
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
                </div>
              </div>
            </>
          );
        })()}


      {meiLookId && (() => {
        const file = project.meiFiles.find((f) => f.id === meiLookId)!;
        return (
          <>
            <div
              className="fixed inset-0 z-40 bg-black/60"
              onClick={() => setMeiLookId(null)} />
            <div className="fixed inset-0 z-50 flex items-center justify-center pointer-events-none">
              <div className="relative bg-[#1D3335] rounded-2xl shadow-2xl p-6 flex flex-col gap-4 max-w-2xl w-full mx-4 pointer-events-auto animate-fade-in">
                <button
                  onClick={() => setMeiLookId(null)}
                  className="absolute top-3 right-4 text-white/60 hover:text-white text-2xl leading-none cursor-pointer">
                    ×
                </button>
                <p className="text-white font-mono text-sm">{file.name}</p>
                <pre className="text-white/80 text-xs font-mono overflow-auto max-h-[60vh] whitespace-pre-wrap bg-black/20 rounded-xl p-4">
                  {file.xmlContent ?? "(no content)"}
                </pre>
                <div className="flex justify-center">
                  <button
                    onClick={() => handleExportMei(file)}
                    className="px-5 py-2 bg-white text-[#1D3335] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm">
                      export
                    </button>
                </div>
              </div>
            </div>
          </>
        );
      })()}
      {/* image upload modal */}
      {imgSection.uploadModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => {
              imgSection.setUploadModal(false);
              imgSection.setDragging(false);
            }}
          />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-6 relative shadow-2xl">
            <button
              onClick={() => {
                if (!converting) {
                  imgSection.setUploadModal(false);
                  imgSection.setDragging(false);
                }
              }}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              x
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">upload image</h2>
            {converting ? (
              <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed border-[#1D3335]/30 bg-white/40 py-12">
                <p className="text-sm text-[#1D3335] text-center">
                  converting PDF pages...
                </p>
              </div>
            ) : (
              <div
                onClick={() => fileInputRef.current?.click()}
                onDragOver={(e) => {
                  e.preventDefault();
                  imgSection.setDragging(true);
                }}
                onDragEnter={(e) => {
                  e.preventDefault();
                  imgSection.setDragging(true);
                }}
                onDragLeave={() => imgSection.setDragging(false)}
                onDrop={(e) => {
                  e.preventDefault();
                  handleFiles(e.dataTransfer.files);
                }}
                className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed py-12 cursor-pointer transition-colors
                  ${imgSection.dragging ? "border-[#1E6B70] bg-[#1E6B70]/10" : "border-[#1D3335]/30 bg-white/40 hover:bg-white/60"}`}
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
      {mdlSection.uploadModal && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => {
              mdlSection.setUploadModal(false);
              mdlSection.setDragging(false);
            }}
          />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-6 relative shadow-2xl">
            <button
              onClick={() => {
                mdlSection.setUploadModal(false);
                mdlSection.setDragging(false);
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
                mdlSection.setDragging(true);
              }}
              onDragEnter={(e) => {
                e.preventDefault();
                mdlSection.setDragging(true);
              }}
              onDragLeave={() => mdlSection.setDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                handleModelFiles(e.dataTransfer.files);
              }}
              className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed py-12 cursor-pointer transition-colors
                ${mdlSection.dragging ? "border-[#1E6B70] bg-[#1E6B70]/10" : "border-[#1D3335]/30 bg-white/40 hover:bg-white/60"}`}
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
