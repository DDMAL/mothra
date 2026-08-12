import { useEffect, useMemo, useState, useSyncExternalStore } from "react";
import type { Project, ModelKind, CantusSource } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { getImageProgress, minNextStep } from "../../utils/imageStep";
import { useAssetSection } from "../../hooks/useAssetSection";
import type { useInferenceSettings } from "../../hooks/useInferenceSettings";
import type { useTextFindingSettings } from "../../hooks/useTextFindingSettings";
import type { useIcSettings } from "../../hooks/useIcSettings";
import RenameModal from "./RenameModal";
import DeleteProjectModal from "./DeleteProjectModal";
import IcSessionsModal from "./IcSessionsModal";
import type { IcResumeRequest } from "./IcSessionsModal";
import ActivityLog from "./ActivityLog";
import ImageTab from "./ImageTab";
import ModelTab from "./ModelTab";
import MeiTab from "./MeiTab";
import TextAlignmentsTab from "./TextAlignmentsTab";
import AnnotationsTab from "./AnnotationsTab";
import StafflinesTab from "./StafflinesTab";
import { downloadBlob } from "../../utils/download";
import CantusSourcePanel from "./CantusSourcePanel";
import TruncatedName from "../shared/TruncatedName";
import {
  subscribeActiveJobs,
  getActiveJobsSnapshot,
} from "../../lib/activeJobs";

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
  onUsedNamesChange: (names: {
    images: string[];
    models: string[];
    annotations: string[];
  }) => void;
  stepsUnlocked: number;
  onStepClick: (step: number) => void;
  /** Open a saved IC session picked in the "manage IC sessions" modal on the
   * IC step page (step 1), rather than inside the modal's iframe. */
  onResumeIcSession: (req: IcResumeRequest) => void;
  onSendToCantus: () => void;
  sendingBundle?: boolean;
  sendBundleError?: string | null;
  onRenameProject: (newName: string) => void;
  onUploadImage: (
    file: File,
    folio?: string,
    sourceId?: string,
    sourceName?: string,
    originalFile?: File,
  ) => Promise<{
    id: string;
    name: string;
    folio?: string;
    sourceId?: string;
    sourceName?: string;
  }>;
  onUploadModel: (
    file: File,
    kind: ModelKind,
  ) => Promise<{ id: string; name: string; kind: ModelKind }>;
  onDeleteImage: (imageId: string) => Promise<void>;
  onDeleteModel: (modelId: string) => Promise<void>;
  onDeleteAnnotation: (annotationId: string) => Promise<void>;
  onDownloadAnnotation: (
    annotationId: string,
    format: "txt" | "json",
  ) => Promise<void>;
  onDeleteMei: (meiId: string) => Promise<void>;
  onDeleteProject: () => void;
  onUpdateCantusSourceId: (sourceId: string) => void;
  inferenceSettings: ReturnType<typeof useInferenceSettings>;
  textFindingSettings: ReturnType<typeof useTextFindingSettings>;
  icSettings: ReturnType<typeof useIcSettings>;
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
  onResumeIcSession,
  onSendToCantus,
  sendingBundle = false,
  sendBundleError = null,
  onRenameProject,
  onUploadImage,
  onUploadModel,
  onDeleteImage,
  onDeleteModel,
  onDeleteAnnotation,
  onDownloadAnnotation,
  onUpdateCantusSourceId,
  onDeleteMei,
  onDeleteProject,
  inferenceSettings,
  textFindingSettings,
  icSettings,
}: ProjectDetailProps) {
  const [activeTab, setActiveTab] = useState<"images" | "models" | "generated">(
    "images",
  );
  const [generatedSubTab, setGeneratedSubTab] = useState<
    "annotations" | "text" | "stafflines" | "mei files"
  >("annotations");
  const [validationError, setValidationError] = useState<string | null>(null);
  // Client-side mirror of the backend's cross-kind "one active job per
  // project" guard (job_store.py's get_active_job_for_project) — disables
  // Continue before the user can even attempt a kickoff that the backend
  // would reject with a 409, instead of them hitting an error. The backend
  // check remains the actual source of truth (this registry is in-memory,
  // reset on reload, not shared across tabs).
  const activeJobsSnapshot = useSyncExternalStore(
    subscribeActiveJobs,
    getActiveJobsSnapshot,
  );
  const activeJobForProject =
    activeJobsSnapshot.find((j) => j.projectId === project.id) ?? null;
  const [projectMenu, setProjectMenu] = useState(false);
  const [projectRenameModal, setProjectRenameModal] = useState(false);
  const [projectRenameName, setProjectRenameName] = useState("");
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [icSessionsModal, setIcSessionsModal] = useState(false);
  // Saved IC sessions for this project, or null while unknown (still loading,
  // or IC didn't answer). Sessions live in IC's store, so this is the only way
  // to know they exist - see the gate on the "manage IC sessions" button.
  const [icSessionCount, setIcSessionCount] = useState<number | null>(null);
  const [loadedCantusSource, setLoadedCantusSource] =
    useState<CantusSource | null>(null);
  const [imageSubTab, setImageSubTab] = useState<"grid" | "batch">("grid");
  const [batchStartFolio, setBatchStartFolio] = useState("");
  const [batchEndFolio, setBatchEndFolio] = useState("");
  const [batchImages, setBatchImages] = useState<
    { id: string; name: string }[]
  >([]);

  useEffect(() => {
    setBatchStartFolio("");
    setBatchEndFolio("");
    setBatchImages([]);
  }, [project.id]);

  // Re-read on modal close too: sessions can be deleted in there, and the
  // count is in the button's own label.
  useEffect(() => {
    let cancelled = false;
    apiFetch(`/api/projects/${project.id}/ic/session-count`)
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => {
        if (!cancelled)
          setIcSessionCount(typeof d?.count === "number" ? d.count : null);
      })
      .catch(() => {
        if (!cancelled) setIcSessionCount(null);
      });
    return () => {
      cancelled = true;
    };
  }, [project.id, icSessionsModal]);

  const batchFolioSequence = useMemo(() => {
    const folios = loadedCantusSource?.folios ?? [];
    if (!batchStartFolio || !batchEndFolio) return [];
    const startIdx = folios.indexOf(batchStartFolio);
    const endIdx = folios.indexOf(batchEndFolio);
    // A manually-typed (off-canonical) start/end boundary - via FolioSelect's
    // "custom folio..." entry - won't be found by indexOf. Manual entry
    // doesn't need the adjacency gate auto-detection uses (the human already
    // vouched for it), so trust it outright rather than collapsing the whole
    // range to empty the way a genuine not-found value used to.
    if (startIdx === -1 && endIdx === -1)
      return [batchStartFolio, batchEndFolio];
    if (startIdx === -1)
      return endIdx === -1
        ? []
        : [batchStartFolio, ...folios.slice(0, endIdx + 1)];
    if (endIdx === -1) return [...folios.slice(startIdx), batchEndFolio];
    if (startIdx > endIdx) return [];
    return folios.slice(startIdx, endIdx + 1);
  }, [loadedCantusSource, batchStartFolio, batchEndFolio]);
  const nextStep = useMemo(
    () =>
      minNextStep(
        usedNames.images,
        project.annotations ?? [],
        project.meiFiles ?? [],
        stepsUnlocked,
      ),
    [usedNames.images, project.annotations, project.meiFiles, stepsUnlocked],
  );
  const sourceLocked = !(usedNames.images.length === 0 || nextStep === 0);
  // Auto IC classifies server-side, which has no training pool without a
  // training set - so Continue is greyed out until one is picked (or the mode
  // is switched to manual). Gated on the steps that lead into IC, and checked
  // at step 0 too: that step flows straight into step 1 in auto mode, so
  // waiting until the IC step would waste a whole detection run.
  const autoIcNeedsTraining =
    nextStep <= 1 && icSettings.mode === "auto" && !icSettings.hasTrainingSet;

  const imgSection = useAssetSection(project.images);
  const mdlSection = useAssetSection(project.models);
  const meiSection = useAssetSection(project.meiFiles);
  const annSection = useAssetSection(project.annotations ?? []);

  const switchTab = (tab: "images" | "models" | "generated") => {
    setActiveTab(tab);
    imgSection.clearSelection();
    mdlSection.clearSelection();
    meiSection.clearSelection();
    annSection.clearSelection();
    imgSection.setPage(0);
    mdlSection.setPage(0);
    annSection.setPage(0);
  };

  const switchGeneratedSubTab = (
    tab: "annotations" | "text" | "stafflines" | "mei files",
  ) => {
    setGeneratedSubTab(tab);
    meiSection.clearSelection();
    annSection.clearSelection();
    meiSection.setPage(0);
    annSection.setPage(0);
  };

  const TAB_LABELS: Record<string, string> = {
    images: "Images",
    models: "Models",
    generated: "Generated files",
  };

  const GENERATED_SUBTAB_LABELS: Record<string, string> = {
    annotations: "Detected layers",
    text: "Detected text",
    stafflines: "Stafflines",
    "mei files": "MEI files",
  };

  const tabs = ["images", "models", "generated"] as const;

  const generatedSubTabs = [
    "annotations",
    "text",
    "stafflines",
    "mei files",
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
        if (
          activeTab === "generated" &&
          generatedSubTab === "annotations" &&
          annSection.selectedIds.size > 0
        ) {
          const ids = [...annSection.selectedIds];
          const deleted = new Set(ids);
          annSection.clearSelection();
          Promise.all(ids.map((id) => onDeleteAnnotation(id))).then(() => {
            onUpdateProject({
              ...project,
              annotations: project.annotations.filter(
                (a) => !deleted.has(a.id),
              ),
            });
          });
        }
        if (
          activeTab === "generated" &&
          generatedSubTab === "mei files" &&
          meiSection.selectedIds.size > 0
        ) {
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
    generatedSubTab,
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
        className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20 shrink-0 whitespace-nowrap"
      >
        use {count} {noun}
        {count > 1 ? "s" : ""}
      </button>
      <button
        onClick={onDelete}
        className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20 shrink-0 whitespace-nowrap"
      >
        delete {count} {noun}
        {count > 1 ? "s" : ""}
      </button>
    </>
  );

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 pt-10 pb-48 relative overflow-x-auto">
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
                const res = await apiFetch(
                  `/api/projects/${project.id}/export`,
                );
                downloadBlob(await res.blob(), `${project.name}.zip`);
              }}
              className="mt-3 text-xs text-white/60 hover:text-white text-left px-3 py-2 rounded-xl hover:bg-white/10 cursor-pointer transition-colors"
            >
              export all files ↓
            </button>
            {/* Hidden only when IC positively reports no sessions AND the IC
                step was never reached. `stepsUnlocked >= 1` alone used to gate
                this, which hid the button in exactly the cases where a saved
                session most needs clearing - sessions can exist at step 0 (see
                ic_api.py's session-count endpoint). A null count means IC
                didn't answer, so show it and let the modal report the error
                rather than silently hiding the only way in. */}
            {(icSessionCount === null ||
              icSessionCount > 0 ||
              stepsUnlocked >= 1) && (
              <button
                onClick={() => setIcSessionsModal(true)}
                className="text-xs text-white/60 hover:text-white text-left px-3 py-2 rounded-xl hover:bg-white/10 cursor-pointer transition-colors"
              >
                manage IC sessions
                {icSessionCount ? ` (${icSessionCount})` : ""}
              </button>
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

        <div className="flex-1">
          {/* header */}
          <div className="flex items-center gap-4 mb-3">
            <button
              onClick={onBack}
              className="text-white text-2xl hover:opacity-70 transition-opacity cursor-pointer shrink-0"
            >
              ←
            </button>
            <h1 className="text-4xl font-bold italic text-white min-w-0 shrink">
              <TruncatedName name={project.name} />
            </h1>
            <div className="relative shrink-0">
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
          </div>

          {/* action buttons — on their own row so a long project name never crowds them out */}
          <div className="flex items-center gap-3 flex-wrap mb-8">
            {activeTab === "images" ? (
              <button
                onClick={() => imgSection.setUploadModal(true)}
                className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer shrink-0"
              >
                + new image
              </button>
            ) : activeTab === "models" ? (
              <button
                onClick={() => mdlSection.setUploadModal(true)}
                className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer shrink-0"
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
            {activeTab === "generated" &&
              generatedSubTab === "annotations" &&
              annSection.selectedIds.size > 0 && (
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
                          ...names.filter(
                            (n) => !usedNames.annotations.includes(n),
                          ),
                        ],
                      });
                      annSection.clearSelection();
                      setValidationError(null);
                    },
                    async () => {
                      const ids = [...annSection.selectedIds];
                      const deleted = new Set(ids);
                      annSection.clearSelection();
                      await Promise.all(
                        ids.map((id) => onDeleteAnnotation(id)),
                      );
                      onUpdateProject({
                        ...project,
                        annotations: project.annotations.filter(
                          (a) => !deleted.has(a.id),
                        ),
                      });
                    },
                  )}
                  <button
                    onClick={() =>
                      project.annotations
                        .filter((a) => annSection.selectedIds.has(a.id))
                        .forEach((a) => onDownloadAnnotation(a.id, "txt"))
                    }
                    className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20 shrink-0 whitespace-nowrap"
                  >
                    download{" "}
                    {annSection.selectedIds.size > 1
                      ? `${annSection.selectedIds.size} `
                      : ""}
                    annotation{annSection.selectedIds.size > 1 ? "s" : ""}{" "}
                    (.txt)
                  </button>
                  <button
                    onClick={() =>
                      project.annotations
                        .filter((a) => annSection.selectedIds.has(a.id))
                        .forEach((a) => onDownloadAnnotation(a.id, "json"))
                    }
                    className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20 shrink-0 whitespace-nowrap"
                  >
                    download{" "}
                    {annSection.selectedIds.size > 1
                      ? `${annSection.selectedIds.size} `
                      : ""}
                    annotation{annSection.selectedIds.size > 1 ? "s" : ""}{" "}
                    (.json)
                  </button>
                </>
              )}
            {activeTab === "generated" &&
              generatedSubTab === "mei files" &&
              meiSection.selectedIds.size > 0 && (
                <>
                  <button
                    onClick={() =>
                      project.meiFiles
                        .filter((f) => meiSection.selectedIds.has(f.id))
                        .forEach((f) =>
                          downloadBlob(
                            new Blob([f.xmlContent ?? ""], {
                              type: "application/xml",
                            }),
                            f.name,
                          ),
                        )
                    }
                    className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20 shrink-0 whitespace-nowrap"
                  >
                    download {meiSection.selectedIds.size} mei file
                    {meiSection.selectedIds.size > 1 ? "s" : ""}
                  </button>
                  <button
                    onClick={async () => {
                      const ids = [...meiSection.selectedIds];
                      const deleted = new Set(ids);
                      meiSection.clearSelection();
                      await Promise.all(ids.map((id) => onDeleteMei(id)));
                      onUpdateProject({
                        ...project,
                        meiFiles: project.meiFiles.filter(
                          (f) => !deleted.has(f.id),
                        ),
                      });
                    }}
                    className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20 shrink-0 whitespace-nowrap"
                  >
                    delete {meiSection.selectedIds.size} mei file
                    {meiSection.selectedIds.size > 1 ? "s" : ""}
                  </button>
                </>
              )}
          </div>

          {/* tab bar + content */}
          <div>
            <CantusSourcePanel
              textFindingSettings={textFindingSettings}
              project={project}
              onUpdateSourceId={onUpdateCantusSourceId}
              onSourceLoaded={setLoadedCantusSource}
              imageSubTab={imageSubTab}
              batchStartFolio={batchStartFolio}
              batchEndFolio={batchEndFolio}
              onBatchStartFolioChange={setBatchStartFolio}
              onBatchEndFolioChange={setBatchEndFolio}
              batchFolioSequence={batchFolioSequence}
              locked={sourceLocked}
              icSettings={icSettings}
            />
            <div className="flex items-end">
              {tabs.map((tab, i) => (
                <button
                  key={tab}
                  onClick={() =>
                    switchTab(tab as "images" | "models" | "generated")
                  }
                  className={`relative px-8 pt-3 pb-2 text-2xl font-bold italic rounded-t-xl cursor-pointer transition-colors
                    ${
                      activeTab === tab
                        ? "text-white border border-white/50 border-b-0 bg-[#4AADAA] z-10"
                        : "text-white/50 hover:text-white/70 border border-transparent"
                    }
                    ${i > 0 ? "-ml-px" : ""}`}
                >
                  {TAB_LABELS[tab] ?? tab}
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
                activeFolio={
                  !textFindingSettings.ocrOnlyMode
                    ? textFindingSettings.folio || undefined
                    : undefined
                }
                onFolioConsumed={() => textFindingSettings.patch({ folio: "" })}
                cantusFolios={loadedCantusSource?.folios ?? []}
                cantusSourceId={loadedCantusSource?.sourceId}
                cantusSourceName={loadedCantusSource?.name}
                ocrOnlyMode={textFindingSettings.ocrOnlyMode}
                imageSubTab={imageSubTab}
                onImageSubTabChange={setImageSubTab}
                batchImages={batchImages}
                batchFolioSequence={batchFolioSequence}
                onBatchImageUploaded={(img) =>
                  setBatchImages((prev) => [...prev, img])
                }
                onBatchUsed={() => {
                  setBatchImages([]);
                  setBatchStartFolio("");
                  setBatchEndFolio("");
                }}
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
                inferenceSettings={inferenceSettings}
                textFindingSettings={textFindingSettings}
              />
            )}
            {activeTab === "generated" && (
              <>
                <div className="flex gap-2 mt-6 mb-4">
                  {generatedSubTabs.map((sub) => (
                    <button
                      key={sub}
                      onClick={() => switchGeneratedSubTab(sub)}
                      className={`px-4 py-1.5 rounded-lg text-sm font-semibold transition-colors cursor-pointer
                        ${generatedSubTab === sub ? "bg-white text-[#4AADAA]" : "text-white/60 hover:text-white/90"}`}
                    >
                      {GENERATED_SUBTAB_LABELS[sub]}
                    </button>
                  ))}
                </div>
                {generatedSubTab === "annotations" && (
                  <AnnotationsTab
                    annotations={project.annotations}
                    images={project.images}
                    projectId={project.id}
                    section={annSection}
                    usedNames={usedNames}
                    onUsedNamesChange={onUsedNamesChange}
                  />
                )}
                {generatedSubTab === "text" && (
                  <TextAlignmentsTab
                    textAlignments={project.textAlignments}
                    images={project.images}
                    projectId={project.id}
                    debugDataByImage={textFindingSettings.debugDataByImage}
                  />
                )}
                {generatedSubTab === "stafflines" && (
                  <StafflinesTab
                    stafflines={project.stafflines}
                    images={project.images}
                    projectId={project.id}
                    onAddStaffline={(newSet) =>
                      onUpdateProject({
                        ...project,
                        stafflines: [...project.stafflines, newSet],
                      })
                    }
                  />
                )}
                {generatedSubTab === "mei files" && (
                  <MeiTab
                    project={project}
                    section={meiSection}
                    onUpdateProject={onUpdateProject}
                    onDeleteMei={onDeleteMei}
                  />
                )}
              </>
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
              disabled={sendingBundle}
              className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-default"
            >
              {sendingBundle ? (
                "preparing bundle..."
              ) : (
                <>send to cantus ultimus &rarr;</>
              )}
            </button>
          ) : null}
          {sendBundleError && (
            <p className="text-red-200 text-xs text-center">
              {sendBundleError}
            </p>
          )}
          {meiSection.selectedIds.size === 0 &&
            (() => {
              const continueLabel =
                usedNames.images.length === 0 || nextStep === 0
                  ? "begin"
                  : nextStep === 1
                    ? "continue: ic"
                    : nextStep === 3
                      ? "continue: neon"
                      : "continue: send";
              return (
                <>
                  <button
                    onClick={() => {
                      if (activeJobForProject) return; // defensive; button is disabled below anyway
                      if (nextStep === 0) {
                        const hasUsableModel =
                          inferenceSettings.modelPreset === "medieval" ||
                          (inferenceSettings.modelPreset === "custom" &&
                            (inferenceSettings.customModelId ||
                              usedNames.models.length > 0));
                        if (!hasUsableModel) {
                          setValidationError(
                            inferenceSettings.modelPreset === "custom"
                              ? "must select a custom YOLO model!"
                              : "must select at least one model!",
                          );
                          return;
                        }
                        if (usedNames.images.length === 0) {
                          setValidationError("must select at least one image!");
                          return;
                        }
                        setValidationError(null);
                      } else if (nextStep === 1 && stepsUnlocked <= 1) {
                        if (usedNames.annotations.length === 0) {
                          setValidationError(
                            "must select at least one annotation!",
                          );
                          return;
                        }
                        if (
                          usedNames.annotations.length !==
                          usedNames.images.length
                        ) {
                          setValidationError(
                            "number of annotations must match number of images!",
                          );
                          return;
                        }
                        setValidationError(null);
                      }
                      if (autoIcNeedsTraining) return; // defensive; button is disabled below
                      onContinue();
                    }}
                    disabled={!!activeJobForProject || autoIcNeedsTraining}
                    className="w-full px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl border-2 border-white hover:opacity-90 cursor-pointer flex items-center justify-center gap-1 disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    {continueLabel} &rarr;
                  </button>
                  {activeJobForProject && (
                    <p className="text-white/70 text-xs mt-1 text-center">
                      a {activeJobForProject.kind} job is already running for
                      this project — please wait for it to finish
                    </p>
                  )}
                  {autoIcNeedsTraining && (
                    <p className="text-white/70 text-xs mt-1 text-center">
                      the classifier is set to auto — pick training data under
                      "Classifier settings", or switch it to manual
                    </p>
                  )}
                </>
              );
            })()}
          <div className="bg-[#C8E6E3]/40 rounded-2xl p-4 flex flex-col gap-2 text-white text-sm">
            <span className="text-white/80">selected:</span>
            {usedNames.models.map((name) => (
              <div key={name} className="flex items-center justify-between">
                <TruncatedName name={name} className="flex-1 min-w-0 mr-2" />
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
                    <TruncatedName
                      name={name}
                      className="flex-1 min-w-0 mr-2"
                    />
                    {stepsUnlocked < 2 && (
                      <button
                        onClick={() =>
                          onUsedNamesChange({
                            ...usedNames,
                            annotations: usedNames.annotations.filter(
                              (n) => n !== name,
                            ),
                          })
                        }
                        className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
                      >
                        ×
                      </button>
                    )}
                  </div>
                ))}
              </>
            )}
            <hr className="border-white/40 my-1" />
            {usedNames.images.map((name) => {
              const hasProgress =
                getImageProgress(
                  name,
                  project.annotations ?? [],
                  project.meiFiles ?? [],
                  stepsUnlocked,
                ) !== null;
              return (
                <div key={name} className="flex items-center justify-between">
                  <TruncatedName name={name} className="flex-1 min-w-0 mr-2" />
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
      {icSessionsModal && (
        <IcSessionsModal
          projectId={project.id}
          onClose={() => setIcSessionsModal(false)}
          onResumeSession={(req) => {
            setIcSessionsModal(false);
            onResumeIcSession(req);
          }}
        />
      )}
    </div>
  );
}
