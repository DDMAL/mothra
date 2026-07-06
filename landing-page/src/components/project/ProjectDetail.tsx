import { useEffect, useState } from "react";
import type { Project } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { getImageProgress, minNextStep } from "../../utils/imageStep";
import { useAssetSection } from "../../hooks/useAssetSection";
import RenameModal from "./RenameModal";
import DeleteProjectModal from "./DeleteProjectModal";
import ActivityLog from "./ActivityLog";
import ImageTab from "./ImageTab";
import ModelTab from "./ModelTab";
import MeiTab from "./MeiTab";
import AnnotationsTab from "./AnnotationsTab";
import { downloadBlob } from "../../utils/download";

const STEPS = [
  "annotate",
  "interactive classifier",
  "encoding",
  "neon",
  "send to cantus ultimus",
];

interface ProjectDetailProps {
  project: Project;
  onBack: () => void;
  onContinue: () => void;
  onUpdateProject: (updated: Project) => void;
  usedNames: { images: string[]; models: string[]; annotations: string[] };
  onUsedNamesChange: (names: { images: string[]; models: string[]; annotations: string[] }) => void;
  stepsUnlocked: number;
  onStepClick: (step: number) => void;
  onSendToCantus: () => void;
  onRenameProject: (newName: string) => void;
  onUploadImage: (file: File) => Promise<{ id: string; name: string }>;
  onUploadModel: (file: File) => Promise<{ id: string; name: string }>;
  onDeleteImage: (imageId: string) => Promise<void>;
  onDeleteModel: (modelId: string) => Promise<void>;
  onDeleteAnnotation: (annotationId: string) => Promise<void>;
  onDownloadAnnotation: (annotationId: string, format: "txt" | "json") => Promise<void>;
  onDeleteMei: (meiId: string) => Promise<void>;
  onDeleteProject: () => void;
  inferenceThreshold: number;
  onInferenceThresholdChange: (v: number) => void;
  inferenceDevice: "cpu" | "cuda" | "mps";
  onInferenceDeviceChange: (v: "cpu" | "cuda" | "mps") => void;
  onGoToTextFinding?: () => void;
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
  onSendToCantus,
  onRenameProject,
  onUploadImage,
  onUploadModel,
  onDeleteImage,
  onDeleteModel,
  onDeleteAnnotation,
  onDownloadAnnotation,
  onDeleteMei,
  onDeleteProject,
  inferenceThreshold,
  onInferenceThresholdChange,
  inferenceDevice,
  onInferenceDeviceChange,
  onGoToTextFinding,
}: ProjectDetailProps) {
  const [activeTab, setActiveTab] = useState<
    "images" | "models" | "annotations" | "mei files"
  >("images");
  const [validationError, setValidationError] = useState<string | null>(null);
  const [projectMenu, setProjectMenu] = useState(false);
  const [projectRenameModal, setProjectRenameModal] = useState(false);
  const [projectRenameName, setProjectRenameName] = useState("");
  const [showDeleteModal, setShowDeleteModal] = useState(false);

  const imgSection = useAssetSection(project.images);
  const mdlSection = useAssetSection(project.models);
  const meiSection = useAssetSection(project.meiFiles);
  const annSection = useAssetSection(project.annotations ?? []);

  const switchTab = (
    tab: "images" | "models" | "annotations" | "mei files",
  ) => {
    setActiveTab(tab);
    imgSection.clearSelection();
    mdlSection.clearSelection();
    meiSection.clearSelection();
    annSection.clearSelection();
    imgSection.setPage(0);
    mdlSection.setPage(0);
    annSection.setPage(0);
  };

  const tabs = [
    "images",
    "models",
    ...(stepsUnlocked >= 1 ? ["annotations"] : []),
    ...(stepsUnlocked >= 3 ? ["mei files"] : []),
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
        annSection.clearSelection();
        annSection.setMenu(null);
      }
      if (e.key === "Delete" || e.key === "Backspace") {
        if (
          document.activeElement?.tagName === "INPUT" ||
          document.activeElement?.tagName === "TEXTAREA"
        )
          return;
        if (activeTab === "images" && imgSection.selectedIds.size > 0) {
          const ids = [...imgSection.selectedIds];
          imgSection.clearSelection();
          Promise.all(ids.map((id) => onDeleteImage(id))).then(() => {
            onUpdateProject({
              ...project,
              images: project.images.filter(
                (img) => !imgSection.selectedIds.has(img.id),
              ),
            });
          });
        }
        if (activeTab === "models" && mdlSection.selectedIds.size > 0) {
          const ids = [...mdlSection.selectedIds];
          const deleted = new Set(ids);
          mdlSection.clearSelection();
          Promise.all(ids.map((id) => onDeleteModel(id))).then(() => {
            onUpdateProject({
              ...project,
              models: project.models.filter((m) => !deleted.has(m.id)),
            });
          });
        }
        if (activeTab === "annotations" && annSection.selectedIds.size > 0) {
          const ids = [...annSection.selectedIds];
          const deleted = new Set(ids);
          annSection.clearSelection();
          Promise.all(ids.map((id) => onDeleteAnnotation(id))).then(() => {
            onUpdateProject({
              ...project,
              annotations: project.annotations.filter((a) => !deleted.has(a.id)),
            });
          });
        }
        if (activeTab === "mei files" && meiSection.selectedIds.size > 0) {
          const ids = [...meiSection.selectedIds];
          const deleted = new Set(ids);
          meiSection.clearSelection();
          Promise.all(ids.map((id) => onDeleteMei(id))).then(() => {
            onUpdateProject({
              ...project,
              meiFiles: project.meiFiles.filter((f) => !deleted.has(f.id)),
            });
          });
        }
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [
    activeTab,
    imgSection,
    mdlSection,
    meiSection,
    annSection,
    project,
    onDeleteImage,
    onUpdateProject,
  ]);

  const selectionButtons = (
    noun: "image" | "model" | "annotation",
    count: number,
    onUse: () => void,
    onDelete: () => void,
  ) => (
    <>
      <button
        onClick={onUse}
        className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
      >
        use {count} {noun}
        {count > 1 ? "s" : ""}
      </button>
      <button
        onClick={onDelete}
        className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
      >
        delete {count} {noun}
        {count > 1 ? "s" : ""}
      </button>
    </>
  );

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 pt-10 pb-48 relative">
      <div
        className={`absolute inset-0 z-30 bg-black/30 transition-opacity pointer-events-none
          ${imgSection.uploadModal || !!imgSection.renameModal || mdlSection.uploadModal || !!mdlSection.renameModal ? "opacity-100" : "opacity-0"}`}
      />

      <div className="flex gap-8 max-w-6xl mx-auto">
        {/* progress sidebar */}
        <div className="flex flex-col gap-3 shrink-0 mt-[4.5rem]">
          <div className="w-48 bg-[#C8E6E3]/30 rounded-2xl p-5 flex flex-col gap-2 self-start">
            <span className="text-white/60 text-sm font-medium mb-1">
              progress:
            </span>
            {STEPS.map((label, i) => {
              const stepNum = i;
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
                  {stepNum} {label}
                </button>
              );
            })}
            <button
              onClick={async () => {
                const res = await apiFetch(`/api/projects/${project.id}/export`);
                downloadBlob(await res.blob(), `${project.name}.zip`);
              }}
              className="mt-3 text-xs text-white/60 hover:text-white text-left px-3 py-2 rounded-xl hover:bg-white/10 cursor-pointer transition-colors"
            >
              export all files ↓
            </button>
            {onGoToTextFinding && stepsUnlocked >= 1 && (
              <div className="w-48 bg-[#C8E6E3]/30 rounded-2xl px-5 py-3">
                <button
                  onClick={onGoToTextFinding}
                  className="w-full text-xs text-white/60 hover:text-white text-left cursor-pointer transition-colors"
                >
                  text finding →
                </button>
              </div>
            )}
          </div>
          <ActivityLog projectId={project.id} />
          <div className="w-48 bg-[#C8E6E3]/30 rounded-2xl px-5 py-3">
            <button
              onClick={async () => {
                const res = await apiFetch(
                  `/api/projects/${project.id}/logs/download`,
                );
                downloadBlob(await res.blob(), `${project.name}_logs.zip`);
              }}
              className="w-full text-xs text-white/60 hover:text-white text-left cursor-pointer transition-colors"
            >
              download all logs ↓
            </button>
          </div>
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
            <div className="relative">
              <button
                onClick={() => setProjectMenu((v) => !v)}
                className="text-white text-2xl hover:opacity-70 cursor-pointer leading-none"
              >
                ⋮
              </button>
              {projectMenu && (
                <>
                  <div
                    className="fixed inset-0 z-40"
                    onClick={() => setProjectMenu(false)}
                  />
                  <div className="absolute z-50 top-full left-0 mt-1 bg-white rounded-2xl shadow-lg p-3 flex flex-col gap-1 min-w-[160px]">
                    <button
                      onClick={() => {
                        setProjectRenameName(project.name);
                        setProjectRenameModal(true);
                        setProjectMenu(false);
                      }}
                      className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
                    >
                      rename
                    </button>
                    <button
                      onClick={() => {
                        setShowDeleteModal(true);
                        setProjectMenu(false);
                      }}
                      className="text-sm text-red-500 text-left px-2 py-1.5 hover:opacity-70 cursor-pointer"
                    >
                      delete project
                    </button>
                  </div>
                </>
              )}
            </div>

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

            {activeTab === "images" &&
              imgSection.selectedIds.size > 0 &&
              selectionButtons(
                "image",
                imgSection.selectedIds.size,
                () => {
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
                },
                async () => {
                  const ids = [...imgSection.selectedIds];
                  await Promise.all(ids.map((id) => onDeleteImage(id)));
                  onUpdateProject({
                    ...project,
                    images: project.images.filter(
                      (img) => !imgSection.selectedIds.has(img.id),
                    ),
                  });
                  imgSection.clearSelection();
                },
              )}

            {activeTab === "models" &&
              mdlSection.selectedIds.size > 0 &&
              selectionButtons(
                "model",
                mdlSection.selectedIds.size,
                () => {
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
                },
                () => {
                  onUpdateProject({
                    ...project,
                    models: project.models.filter(
                      (m) => !mdlSection.selectedIds.has(m.id),
                    ),
                  });
                  mdlSection.clearSelection();
                },
              )}
              {activeTab === "annotations" && annSection.selectedIds.size > 0 && (
                <>
                  {selectionButtons(
                    "annotation",
                    annSection.selectedIds.size,
                    () => {
                      const names = project.annotations
                        .filter((a) => annSection.selectedIds.has(a.id))
                        .map((a) => a.imageName);
                      onUsedNamesChange({
                        ...usedNames,
                        annotations: [
                          ...usedNames.annotations,
                          ...names.filter((n) => !usedNames.annotations.includes(n)),
                        ],
                      });
                      annSection.clearSelection();
                      setValidationError(null);
                    },
                    async () => {
                      const ids = [...annSection.selectedIds];
                      const deleted = new Set(ids);
                      annSection.clearSelection();
                      await Promise.all(ids.map((id) => onDeleteAnnotation(id)));
                      onUpdateProject({
                        ...project,
                        annotations: project.annotations.filter((a) => !deleted.has(a.id)),
                      });
                    },
                  )}
                  <button
                    onClick={() =>
                      project.annotations
                        .filter((a) => annSection.selectedIds.has(a.id))
                        .forEach((a) => onDownloadAnnotation(a.id, "txt"))
                    }
                    className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                  >
                    download {annSection.selectedIds.size > 1 ? `${annSection.selectedIds.size} ` : ""}annotation{annSection.selectedIds.size > 1 ? "s" : ""} (.txt)
                  </button>
                  <button
                    onClick={() =>
                      project.annotations
                        .filter((a) => annSection.selectedIds.has(a.id))
                        .forEach((a) => onDownloadAnnotation(a.id, "json"))
                    }
                    className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                  >
                    download {annSection.selectedIds.size > 1 ? `${annSection.selectedIds.size} ` : ""}annotation{annSection.selectedIds.size > 1 ? "s" : ""} (.json)
                  </button>
                </>
              )}
              {activeTab === "mei files" && meiSection.selectedIds.size > 0 && (
                <button
                  onClick={async () => {
                    const ids = [...meiSection.selectedIds];
                    const deleted = new Set(ids);
                    meiSection.clearSelection();
                    await Promise.all(ids.map((id) => onDeleteMei(id)));
                    onUpdateProject({
                      ...project,
                      meiFiles: project.meiFiles.filter((f) => !deleted.has(f.id)),
                    });
                  }}
                  className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20"
                >
                  delete {meiSection.selectedIds.size} mei file{meiSection.selectedIds.size > 1 ? "s" : ""}
                </button>
              )}
          </div>

          {/* tab bar + content */}
          <div>
            <div className="flex items-end">
              {tabs.map((tab, i) => (
                <button
                  key={tab}
                  onClick={() =>
                    switchTab(
                      tab as "images" | "models" | "annotations" | "mei files",
                    )
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

            {activeTab === "images" && (
              <ImageTab
                project={project}
                section={imgSection}
                usedNames={usedNames}
                onUpdateProject={onUpdateProject}
                onUsedNamesChange={onUsedNamesChange}
                onUploadImage={onUploadImage}
                onDeleteImage={onDeleteImage}
                setValidationError={setValidationError}
              />
            )}
            {activeTab === "models" && (
              <ModelTab
                project={project}
                section={mdlSection}
                usedNames={usedNames}
                onUpdateProject={onUpdateProject}
                onUsedNamesChange={onUsedNamesChange}
                onUploadModel={onUploadModel}
                setValidationError={setValidationError}
                inferenceThreshold={inferenceThreshold}
                onInferenceThresholdChange={onInferenceThresholdChange}
                inferenceDevice={inferenceDevice}
                onInferenceDeviceChange={onInferenceDeviceChange}
              />
            )}
            {activeTab === "annotations" && (
              <AnnotationsTab 
                annotations={project.annotations} 
                projectId={project.id}
                section={annSection}
                usedNames={usedNames}
                onUsedNamesChange={onUsedNamesChange}
              />
            )}
            {activeTab === "mei files" && (
              <MeiTab
                project={project}
                section={meiSection}
                onUpdateProject={onUpdateProject}
                onDeleteMei={onDeleteMei}
              />
            )}
          </div>
        </div>

        {/* right sidebar */}
        <div className="flex flex-col gap-3 w-52 flex-shrink-0 pt-2">
          {meiSection.selectedIds.size > 0 ? (
            <button
              onClick={() => {
                imgSection.clearSelection();
                mdlSection.clearSelection();
                meiSection.clearSelection();
                onSendToCantus();
              }}
              className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1"
            >
              send to cantus ultimus &rarr;
            </button>
          ) : (() => {
            const annotations = project.annotations ?? [];
            const meiFiles = project.meiFiles ?? [];
            const nextStep = minNextStep(usedNames.images, annotations, meiFiles);
            const continueLabel =
              usedNames.images.length === 0 || nextStep === 0 ? "begin" :
              nextStep === 1 ? "continue: ic" :
              nextStep === 3 ? "continue: neon" :
              "continue: send";
            return (
              <button
                onClick={() => {
                  if (nextStep === 0) {
                    if (usedNames.models.length === 0) {
                      setValidationError("must select at least one model!");
                      return;
                    }
                    if (usedNames.images.length === 0) {
                      setValidationError("must select at least one image!");
                      return;
                    }
                    setValidationError(null);
                  } else if (nextStep === 1 && stepsUnlocked <= 1) {
                    if (usedNames.annotations.length === 0) {
                      setValidationError("must select at least one annotation!");
                      return;
                    }
                    if (usedNames.annotations.length !== usedNames.images.length) {
                      setValidationError("number of annotations must match number of images!");
                      return;
                    }
                    setValidationError(null);
                  }
                  onContinue();
                }}
                className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1"
              >
                {continueLabel} &rarr;
              </button>
            );
          })()}
          <div className="bg-[#C8E6E3]/40 rounded-2xl p-4 flex flex-col gap-2 text-white text-sm">
            <span className="text-white/80">selected:</span>
            {usedNames.models.map((name) => (
              <div key={name} className="flex items-center justify-between">
                <span className="truncate flex-1 mr-2">{name}</span>
                {stepsUnlocked === 0 && (
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
                )}
              </div>
            ))}
            {stepsUnlocked >= 1 && (
              <>
                <hr className="border-white/40 my-1" />
                {usedNames.annotations.map((name) => (
                  <div key={name} className="flex items-center justify-between">
                    <span className="truncate flex-1 mr-2">{name}</span>
                    {stepsUnlocked < 2 && (
                      <button
                        onClick={() => onUsedNamesChange({ ...usedNames, annotations: usedNames.annotations.filter((n) => n !== name) })}
                        className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
                      >×</button>
                    )}
                  </div>
                ))}
              </>
            )}
            <hr className="border-white/40 my-1" />
            {usedNames.images.map((name) => {
              const hasProgress = getImageProgress(name, project.annotations ?? [], project.meiFiles ?? []) !== null;
              return (
                <div key={name} className="flex items-center justify-between">
                  <span className="truncate flex-1 mr-2">{name}</span>
                  {!hasProgress && (
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
                  )}
                </div>
              );
            })}
          </div>
          {validationError && (
            <p className="text-white text-xs text-center">{validationError}</p>
          )}
        </div>
      </div>

      {projectRenameModal && (
        <RenameModal
          label="project"
          value={projectRenameName}
          onChange={setProjectRenameName}
          onSubmit={() => {
            onRenameProject(projectRenameName.trim() || project.name);
            setProjectRenameModal(false);
          }}
          onClose={() => setProjectRenameModal(false)}
        />
      )}
      {showDeleteModal && (
        <DeleteProjectModal
          project={project}
          onConfirm={() => {
            onDeleteProject();
            onBack();
          }}
          onCancel={() => setShowDeleteModal(false)}
        />
      )}
    </div>
  );
}
