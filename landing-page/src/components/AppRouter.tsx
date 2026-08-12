import { useState, useEffect, useRef } from "react";
import type { Dispatch, SetStateAction } from "react";
import type {
  View,
  Project,
  AnnotationSet,
  MeiFile,
  ModelKind,
} from "../types";
import type { CurrentUser } from "../hooks/useAuth";
import { apiFetch, apiFetchOrThrow, apiFetchJobStream } from "../lib/apiFetch";
import { minNextStep, pendingIcImages } from "../utils/imageStep";
import { downloadBlob } from "../utils/download";
import type { useProjectMutations } from "../hooks/useProjectMutations";
import { useInferenceSettings } from "../hooks/useInferenceSettings";
import { useTextFindingSettings } from "../hooks/useTextFindingSettings";
import Hero from "./landing/Hero";
import Documentation from "./documentation/Documentation";
import AuthPage from "./auth/AuthPage";
import MyAccount from "./account/MyAccount";
import MyProjects from "./project/MyProjects";
import ProjectDetail from "./project/ProjectDetail";
import type { IcResumeRequest } from "./project/IcSessionsModal";
import ProcessingPage from "./workflow/ProcessingPage";
import CompletionPage from "./workflow/CompletionPage";
import InteractiveClassifier from "./workflow/InteractiveClassifier";
import IcSessionUnavailable from "./workflow/IcSessionUnavailable";
import IcCompletionTestPage from "./workflow/ICCompletionTestPage";
import NeonCompletionPage from "./workflow/NeonCompletionPage";
import NeonBatchEditor from "./workflow/NeonBatchEditor";

const STEP_TIMING = { intervalMs: 60, completionDelayMs: 4000 } as const;

// Dev escape hatch for machines that can't run ultralytics: when set, the
// predict/processing step is bypassed and "Continue" on a fresh project jumps
// straight to the Interactive Classifier (which falls back to placeholder
// bboxes server-side when no YOLO annotations exist). Pair with MOTHRA_SKIP_YOLO
// on the backend so model uploads skip checkpoint inspection too.
// Parsed the same way as the backend's MOTHRA_SKIP_YOLO rather than with a bare
// Boolean(), which treats *any* non-empty string as true — including the
// "VITE_SKIP_PREDICT=0" someone writes to turn the bypass back off.
const SKIP_PREDICT = ["1", "true", "yes"].includes(
  String(import.meta.env.VITE_SKIP_PREDICT ?? "")
    .trim()
    .toLowerCase(),
);

// A project's used images are eligible for the batch (cross-folio-aware)
// text-finding pipeline only when every one of them was tagged with a folio
// against the project's single Cantus source at upload time.
function computeBatchRun(
  project: Project,
): { imageIds: string[]; folios: string[] } | null {
  if (!project.cantusSourceId) return null;
  const used = project.images.filter((img) =>
    project.usedImageNames.includes(img.name),
  );
  if (used.length === 0 || !used.every((img) => img.folio)) return null;
  return {
    imageIds: used.map((img) => img.id),
    folios: used.map((img) => img.folio!),
  };
}

function yoloTxtToJson(yoloTxt: string, imageName: string): string {
  const annotations = yoloTxt
    .split("\n")
    .filter(Boolean)
    .map((line) => {
      const [cls, x, y, w, h] = line.trim().split(" ").map(Number);
      return { class: cls, x_center: x, y_center: y, width: w, height: h };
    });
  return JSON.stringify({ imageName, annotations }, null, 2);
}

interface AppRouterProps {
  view: View;
  setView: (v: View) => void;
  currentUser: CurrentUser | null;
  setCurrentUser: (u: CurrentUser) => void;
  projects: Project[];
  setProjects: Dispatch<SetStateAction<Project[]>>;
  selectedProject: Project | null;
  selectedProjectId: number | null;
  setSelectedProjectId: (id: number | null) => void;
  pendingXmlFile: File | null;
  setPendingXmlFile: (f: File | null) => void;
  pendingImageFile: File | null;
  setPendingImageFile: (f: File | null) => void;
  meiContent: { bytes: string; stem: string } | null;
  handleDownloadManifest: () => void;
  handleDownloadMei: () => void;
  handleLoginSuccess: (user: CurrentUser, token: string) => void;
  handleLogout: () => void;
  mutations: ReturnType<typeof useProjectMutations>;
  handleEncodeResult: (ev: {
    session_id: string;
    mei_base64: string;
    manifest: Record<string, unknown> | null;
    stave_source?: string | null;
    logs?: string[];
  }) => void;
  pendingBatchPairs: { xmlFile: File; imageFile: File }[];
  setPendingBatchPairs: (pairs: { xmlFile: File; imageFile: File }[]) => void;
  handleEncodeBatchResult: (ev: {
    item: number;
    session_id: string;
    mei_base64: string;
    manifest: Record<string, unknown> | null;
    image_name?: string;
    stem?: string;
    stave_source?: string | null;
    logs?: string[];
  }) => void;
}

export default function AppRouter({
  view,
  setView,
  currentUser,
  setCurrentUser,
  projects,
  setProjects,
  selectedProject,
  selectedProjectId,
  setSelectedProjectId,
  pendingXmlFile,
  setPendingXmlFile,
  pendingImageFile,
  setPendingImageFile,
  meiContent,
  handleDownloadManifest,
  handleDownloadMei,
  handleLoginSuccess,
  handleLogout,
  mutations,
  handleEncodeResult,
  pendingBatchPairs,
  setPendingBatchPairs,
  handleEncodeBatchResult,
}: AppRouterProps) {
  const {
    createProject,
    renameProject,
    deleteProject,
    restoreProject,
    permanentlyDeleteProject,
    duplicateProject,
    updateProjectSteps,
    updateUsedImageNames,
    updateUsedModelNames,
    updateUsedAnnotationNames,
    updateCantusSourceId,
    togglePin,
  } = mutations;
  const [encodingLogs, setEncodingLogs] = useState<string[]>([]);
  const [annotationLogs, setAnnotationLogs] = useState<string[]>([]);
  const [originalMeiFiles, setOriginalMeiFiles] = useState<MeiFile[]>([]);

  // thread inference settings + text-finding settings (mothra-text optional inputs)
  const inferenceSettings = useInferenceSettings();
  const textFindingSettings = useTextFindingSettings();

  // batch text-alignment run (run_chain.py) state
  const [batchRunIds, setBatchRunIds] = useState<{
    imageIds: string[];
    folios: string[];
  } | null>(null);
  // Set when the user clicks "view progress" on the project page for a job
  // this tab didn't kick off itself (ProjectDetail.tsx's onViewActiveJob) --
  // tells the "processing" case below to reattach ProcessingPage to an
  // existing job_id's stream instead of running its normal kickoff.
  const [resumeJob, setResumeJob] = useState<{
    jobId: string;
    kind: string;
  } | null>(null);
  const [batchResult, setBatchResult] = useState<{
    batchId: string;
    fileCount: number;
  } | null>(null);
  const [batchSummary, setBatchSummary] = useState<{
    succeeded: unknown[];
    failed: unknown[];
  } | null>(null);

  // thread clef settings
  const [clefShape, setClefShape] = useState<"C" | "F">("C");
  const [clefLine, setClefLine] = useState(3);

  // A saved IC session picked in the project page's "manage IC sessions"
  // modal, to be opened on the IC step page. Cleared by goToIc() below, so
  // every other route into that view starts on the first page to classify.
  const [resumeIcSession, setResumeIcSession] =
    useState<IcResumeRequest | null>(null);

  const [sendingBundle, setSendingBundle] = useState(false);
  const [sendBundleError, setSendBundleError] = useState<string | null>(null);
  // Bumped whenever the active project changes (or a new send-to-Cantus
  // request starts), so an in-flight request whose project has since been
  // navigated away from can recognize itself as stale and skip mutating
  // state for whatever project is on screen by the time it resolves.
  const cantusRequestIdRef = useRef(0);

  useEffect(() => {
    cantusRequestIdRef.current += 1;
    setSendBundleError(null);
    setSendingBundle(false);
  }, [selectedProjectId]);

  const handleSendToCantus = async () => {
    if (!selectedProject?.cantusSourceId) {
      setSendBundleError("link a Cantus source to this project first");
      return;
    }
    const requestId = ++cantusRequestIdRef.current;
    const isStale = () => cantusRequestIdRef.current !== requestId;
    setSendingBundle(true);
    setSendBundleError(null);
    try {
      const r = await apiFetchOrThrow(
        `/api/projects/${selectedProject.id}/sources/${selectedProject.cantusSourceId}/cantus-bundle`,
      );
      const blob = await r.blob();
      if (isStale()) return;
      const cd = r.headers.get("Content-Disposition") ?? "";
      const match = cd.match(/filename="?([^"]+)"?/);
      downloadBlob(
        blob,
        match
          ? match[1]
          : `cantus-bundle-${selectedProject.cantusSourceId}.zip`,
      );
      setView("send-completion");
    } catch (e) {
      if (isStale()) return;
      setSendBundleError(
        e instanceof Error ? e.message : "failed to prepare bundle",
      );
    } finally {
      if (!isStale()) setSendingBundle(false);
    }
  };

  useEffect(() => {
    const PROJECT_VIEWS: View[] = [
      "project",
      "processing",
      "completion",
      "ic",
      "encoding-processing",
      "encoding-completion",
      "neon-editor",
      "neon-completion",
    ];
    if (PROJECT_VIEWS.includes(view) && !selectedProject) setView("projects");
  }, [view, selectedProject]);

  // Ordinary way into the IC step: start on the first page to classify. Any
  // saved-session resume left over from a previous visit is dropped here, so
  // only the "manage IC sessions" path (which sets it) ever opens one - the
  // request has to outlive the navigation itself, since the "ic" case below
  // keeps reading it to hold the session's page in the filmstrip.
  const goToIc = () => {
    setResumeIcSession(null);
    setView("ic");
  };

  switch (view) {
    case "landing":
      return (
        <main>
          <Hero
            onLogin={() => setView("login")}
            onViewWalkthrough={() => setView("docs")}
          />
        </main>
      );
    case "docs":
      return <Documentation onHome={() => setView("landing")} />;
    case "account":
      return currentUser ? (
        <MyAccount
          currentUser={currentUser}
          onUserUpdate={(u) => setCurrentUser(u)}
          onLogout={handleLogout}
        />
      ) : null;
    case "projects":
      return (
        <MyProjects
          projects={projects}
          onSelectProject={(id) => {
            setSelectedProjectId(id);
            setView("project");
            const now = new Date().toISOString();
            apiFetch(`/api/projects/${id}`, {
              method: "PUT",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ lastOpenedAt: now }),
            });
            setProjects((prev) =>
              prev.map((p) => (p.id === id ? { ...p, lastOpenedAt: now } : p)),
            );
          }}
          onCreateProject={createProject}
          onRenameProject={renameProject}
          onDeleteProject={deleteProject}
          onRestoreProject={restoreProject}
          onPermanentlyDeleteProject={permanentlyDeleteProject}
          onTogglePin={togglePin}
          onDuplicateProject={duplicateProject}
        />
      );
    case "project":
      return selectedProject ? (
        <ProjectDetail
          project={selectedProject}
          onBack={() => setView("projects")}
          onContinue={() => {
            const step = minNextStep(
              selectedProject.usedImageNames,
              selectedProject.annotations ?? [],
              selectedProject.meiFiles ?? [],
              selectedProject.stepsUnlocked,
            );
            if (step >= 4) handleSendToCantus();
            else if (step >= 3) setView("neon-editor");
            else if (step >= 1 || SKIP_PREDICT) goToIc();
            else {
              setBatchRunIds(computeBatchRun(selectedProject));
              setBatchResult(null);
              setView("processing");
            }
          }}
          onUpdateProject={(updated) =>
            setProjects((prev) =>
              prev.map((p) => (p.id === updated.id ? updated : p)),
            )
          }
          onStepClick={(step) => {
            if (step === 0) {
              if (SKIP_PREDICT) {
                goToIc();
                return;
              }
              setBatchRunIds(computeBatchRun(selectedProject));
              setBatchResult(null);
              setView("processing");
            } else if (step === 1) goToIc();
            else if (step === 2) setView("ic-completion");
            else if (step === 3) setView("neon-editor");
          }}
          onResumeIcSession={(req) => {
            setResumeIcSession(req);
            setView("ic");
          }}
          onSendToCantus={handleSendToCantus}
          onViewActiveJob={(jobId, kind) => {
            setResumeJob({ jobId, kind });
            setView("processing");
          }}
          sendingBundle={sendingBundle}
          sendBundleError={sendBundleError}
          onRenameProject={(newName) =>
            renameProject(selectedProject.id, newName)
          }
          onUpdateCantusSourceId={(sourceId) =>
            updateCantusSourceId(selectedProject.id, sourceId)
          }
          usedNames={{
            images: selectedProject.usedImageNames,
            models: selectedProject.usedModelNames ?? [],
            annotations: selectedProject.usedAnnotationNames ?? [],
          }}
          onUsedNamesChange={(names) => {
            updateUsedImageNames(selectedProject.id, names.images);
            updateUsedModelNames(selectedProject.id, names.models);
            updateUsedAnnotationNames(selectedProject.id, names.annotations);
          }}
          stepsUnlocked={selectedProject.stepsUnlocked}
          onUploadImage={async (
            file,
            folio,
            sourceId,
            sourceName,
            originalFile,
          ) => {
            const form = new FormData();
            form.append("file", file);
            if (folio) form.append("folio", folio);
            if (sourceId) form.append("source_id", sourceId);
            if (sourceName) form.append("source_name", sourceName);
            if (originalFile) form.append("original_file", originalFile);
            const r = await apiFetchOrThrow(
              `/api/projects/${selectedProject.id}/images`,
              {
                method: "POST",
                body: form,
              },
            );
            return r.json();
          }}
          onUploadModel={async (file: File, kind: ModelKind) => {
            const form = new FormData();
            form.append("file", file);
            form.append("kind", kind);
            const r = await apiFetchOrThrow(
              `/api/projects/${selectedProject.id}/models`,
              {
                method: "POST",
                body: form,
              },
            );
            return r.json();
          }}
          onDeleteModel={async (modelId) => {
            await apiFetchOrThrow(
              `/api/projects/${selectedProject.id}/models/${modelId}`,
              { method: "DELETE" },
            );
          }}
          onDeleteAnnotation={async (annotationId) => {
            await apiFetchOrThrow(
              `/api/projects/${selectedProject.id}/annotations/${annotationId}`,
              { method: "DELETE" },
            );
          }}
          onDownloadAnnotation={async (annotationId, format) => {
            const r = await apiFetch(
              `/api/projects/${selectedProject.id}/annotations/${annotationId}`,
            );
            const data = await r.json();
            const stem = (data.imageName as string).replace(/\.[^.]+$/, "");
            if (format === "json") {
              downloadBlob(
                new Blob([yoloTxtToJson(data.yoloTxt, data.imageName)], {
                  type: "application/json",
                }),
                `${stem}_annotations.json`,
              );
            } else {
              downloadBlob(
                new Blob([data.yoloTxt], { type: "text/plain" }),
                `${stem}_annotations.txt`,
              );
            }
          }}
          onDeleteMei={async (meiId) => {
            await apiFetchOrThrow(
              `/api/projects/${selectedProject.id}/mei/${meiId}`,
              { method: "DELETE" },
            );
          }}
          onDeleteImage={async (imageId) => {
            await apiFetchOrThrow(
              `/api/projects/${selectedProject.id}/images/${imageId}`,
              {
                method: "DELETE",
              },
            );
          }}
          onDeleteProject={() => {
            deleteProject(selectedProject.id);
            setView("projects");
          }}
          inferenceSettings={inferenceSettings}
          textFindingSettings={textFindingSettings}
        />
      ) : null;
    case "processing":
      return selectedProject ? (
        <ProcessingPage
          onBack={() => {
            setResumeJob(null);
            setView("project");
          }}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 1),
              );
            }
            setResumeJob(null);
            setView("completion");
          }}
          projectId={selectedProject.id}
          jobKind={resumeJob?.kind ?? (batchRunIds ? "text_batch" : "predict")}
          streamRequest={
            resumeJob
              ? (signal, onJobId) => {
                  onJobId?.(resumeJob.jobId);
                  return apiFetch(`/api/jobs/${resumeJob.jobId}/stream`, {
                    signal,
                  });
              }
            : (signal, onJobId) => {
                const usedModelId =
                  selectedProject.models.find((m) =>
                    (selectedProject.usedModelNames ?? []).includes(m.name),
                  )?.id ?? "";
                const resolvedCustomModelId =
                  inferenceSettings.customModelId || usedModelId;
                if (batchRunIds) {
                  return apiFetchJobStream(
                    `/api/projects/${selectedProject.id}/text-batch/run`,
                    {
                      method: "POST",
                      headers: { "Content-Type": "application/json" },
                      body: JSON.stringify({
                        image_ids: batchRunIds.imageIds,
                        folios: batchRunIds.folios,
                        source_id: Number(textFindingSettings.sourceId),
                        segmentation_model:
                          textFindingSettings.segmentationModelId || null,
                        recognition_model:
                          textFindingSettings.recognitionModelId || null,
                        device: textFindingSettings.device,
                        column_count:
                          textFindingSettings.columnCount === "auto"
                            ? null
                            : Number(textFindingSettings.columnCount),
                        column_bimodal_threshold:
                          textFindingSettings.columnBimodalThreshold,
                        masking_enabled: textFindingSettings.maskingEnabled,
                        mask_padding: textFindingSettings.maskPadding,
                        music_overlap_filter_enabled:
                          textFindingSettings.musicOverlapFilterEnabled,
                        debug_mode: textFindingSettings.debugMode,
                        mask_model_id: textFindingSettings.maskModelId || null,
                        model_preset: inferenceSettings.modelPreset,
                        model_id:
                          inferenceSettings.modelPreset === "custom"
                            ? resolvedCustomModelId
                            : null,
                        yolo_confidence_threshold: inferenceSettings.threshold,
                        yolo_device: inferenceSettings.device,
                        text_music_confidence_threshold:
                          inferenceSettings.useSharedDetectorSettings
                            ? null
                            : inferenceSettings.textMusicSettings.threshold,
                        text_music_device:
                          inferenceSettings.useSharedDetectorSettings
                            ? null
                            : inferenceSettings.textMusicSettings.device,
                        stave_confidence_threshold:
                          inferenceSettings.useSharedDetectorSettings
                            ? null
                            : inferenceSettings.staveSettings.threshold,
                        stave_device: inferenceSettings.useSharedDetectorSettings
                          ? null
                          : inferenceSettings.staveSettings.device,
                      }),
                    },
                    signal,
                    onJobId,
                  );
                }
                const usedImageIds = selectedProject.images
                  .filter((i) => selectedProject.usedImageNames.includes(i.name))
                  .map((i) => i.id);
                return apiFetchJobStream(
                  `/api/projects/${selectedProject.id}/predict`,
                  {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                      model_preset: inferenceSettings.modelPreset,
                      model_id:
                        inferenceSettings.modelPreset === "custom"
                          ? resolvedCustomModelId
                          : null,
                      image_ids: usedImageIds,
                      confidence_threshold: inferenceSettings.threshold,
                      device: inferenceSettings.device,
                      text_music_confidence_threshold:
                        inferenceSettings.useSharedDetectorSettings
                          ? null
                          : inferenceSettings.textMusicSettings.threshold,
                      text_music_device: inferenceSettings.useSharedDetectorSettings
                        ? null
                        : inferenceSettings.textMusicSettings.device,
                      stave_confidence_threshold:
                        inferenceSettings.useSharedDetectorSettings
                          ? null
                          : inferenceSettings.staveSettings.threshold,
                      stave_device: inferenceSettings.useSharedDetectorSettings
                        ? null
                        : inferenceSettings.staveSettings.device,
                      text_column_count:
                        textFindingSettings.columnCount === "auto"
                          ? null
                          : Number(textFindingSettings.columnCount),
                      text_segmentation_model_id:
                        textFindingSettings.segmentationModelId || null,
                      text_recognition_model_id:
                        textFindingSettings.recognitionModelId || null,
                      text_device: textFindingSettings.device,
                      text_column_bimodal_threshold:
                        textFindingSettings.columnBimodalThreshold,
                      text_masking_enabled: textFindingSettings.maskingEnabled,
                      text_mask_padding: textFindingSettings.maskPadding,
                      text_music_overlap_filter_enabled:
                        textFindingSettings.musicOverlapFilterEnabled,
                      text_debug_mode: textFindingSettings.debugMode,
                      text_mask_model_id: textFindingSettings.maskModelId || null,
                      text_source_id:
                        !textFindingSettings.ocrOnlyMode &&
                        textFindingSettings.sourceId
                          ? Number(textFindingSettings.sourceId)
                          : null,
                    }),
                  },
                  signal,
                  onJobId,
                );
              }}
          onResult={(ev) => {
            if (batchRunIds) {
              const { text_debug_data: batchDebugData } = ev as {
                text_debug_data?: Record<string, unknown>;
              };
              if (batchDebugData) {
                textFindingSettings.setDebugDataByImage((prev) => ({
                  ...prev,
                  ...batchDebugData,
                }));
              }
              setBatchResult(ev as { batchId: string; fileCount: number });
              apiFetch(`/api/projects/${selectedProject.id}`)
                .then((r) => r.json())
                .then((fresh: Project) => {
                  setProjects((prev) =>
                    prev.map((p) =>
                      p.id === selectedProject.id
                        ? {
                            ...p,
                            annotations: fresh.annotations,
                            textAlignments: fresh.textAlignments,
                            stafflines: fresh.stafflines,
                          }
                        : p,
                    ),
                  );
                })
                .catch(() => {});
              return;
            }
            const { annotations, text_debug_data } = ev as {
              annotations: AnnotationSet[];
              text_debug_data?: Record<string, unknown>;
            };
            if (text_debug_data) {
              textFindingSettings.setDebugDataByImage((prev) => ({
                ...prev,
                ...text_debug_data,
              }));
            }
            setProjects((prev) =>
              prev.map((p) =>
                p.id === selectedProject.id
                  ? {
                      ...p,
                      annotations: [
                        ...p.annotations.filter(
                          (a) =>
                            !annotations.some(
                              (na) => na.imageName === a.imageName,
                            ),
                        ),
                        ...annotations,
                      ],
                    }
                  : p,
              ),
            );
            apiFetch(`/api/projects/${selectedProject.id}`)
              .then((r) => r.json())
              .then((fresh: Project) => {
                setProjects((prev) =>
                  prev.map((p) =>
                    p.id === selectedProject.id
                      ? {
                          ...p,
                          annotations: fresh.annotations,
                          textAlignments: fresh.textAlignments,
                          stafflines: fresh.stafflines,
                        }
                      : p,
                  ),
                );
              })
              .catch(() => {});
          }}
          onLogsReady={setAnnotationLogs}
        />
      ) : null;
    case "completion":
      return (
        <CompletionPage
          onContinue={() => goToIc()}
          onBackToProject={() => setView("project")}
          logsFileName="annotatorlogs.txt"
          logContent={annotationLogs.join("\n")}
          onDownloadAnnotations={
            selectedProject?.annotations?.length
              ? async () => {
                  for (const ann of selectedProject.annotations) {
                    const r = await apiFetch(
                      `/api/projects/${selectedProject.id}/annotations/${ann.id}`,
                    );
                    const data = await r.json();
                    const stem = (data.imageName as string).replace(
                      /\.[^.]+$/,
                      "",
                    );
                    downloadBlob(
                      new Blob([data.yoloTxt], { type: "text/plain" }),
                      `${stem}_annotations.txt`,
                    );
                  }
                }
              : undefined
          }
          onDownloadAnnotationsJson={
            selectedProject?.annotations?.length
              ? async () => {
                  for (const ann of selectedProject.annotations) {
                    const r = await apiFetch(
                      `/api/projects/${selectedProject.id}/annotations/${ann.id}`,
                    );
                    const data = await r.json();
                    const stem = (data.imageName as string).replace(
                      /\.[^.]+$/,
                      "",
                    );
                    downloadBlob(
                      new Blob([yoloTxtToJson(data.yoloTxt, data.imageName)], {
                        type: "application/json",
                      }),
                      `${stem}_annotations.json`,
                    );
                  }
                }
              : undefined
          }
          onDownloadZip={
            batchResult && selectedProject
              ? async () => {
                  const r = await apiFetch(
                    `/api/projects/${selectedProject.id}/text-batch/${batchResult.batchId}/download`,
                  );
                  downloadBlob(
                    await r.blob(),
                    `batch-${batchResult.batchId}.zip`,
                  );
                }
              : undefined
          }
        />
      );
    case "ic": {
      if (!selectedProject) return null;
      const pending = pendingIcImages(
        selectedProject.images,
        selectedProject.usedImageNames,
        selectedProject.annotations ?? [],
        selectedProject.meiFiles ?? [],
        selectedProject.stepsUnlocked,
      );
      // The page a resume request (from "manage IC sessions") points at. IC
      // records mothra's image id when the session is staged, so that's the
      // authoritative match; the file-name stem it also stores is a fallback
      // only for sessions saved without an id. Deliberately not a fallback for
      // an id that fails to resolve - that means the page was deleted, and a
      // stem match could land on a different image that reuses the filename
      // (the session wouldn't even be the one /ic/start finds for it).
      const resumeImage = resumeIcSession
        ? ((resumeIcSession.imageId == null
            ? selectedProject.images.find(
                (img) =>
                  img.name.replace(/\.[^.]+$/, "") ===
                  resumeIcSession.sourceName,
              )
            : selectedProject.images.find(
                (img) => img.id === resumeIcSession.imageId,
              )) ?? null)
        : null;
      // An unresolvable resume must not fall through to the classifier: it
      // would mount on the first pending page, and queueing there would pair
      // this session's GameraXML with that other page's image.
      if (resumeIcSession && !resumeImage)
        return (
          <IcSessionUnavailable
            sourceName={resumeIcSession.sourceName}
            sessionId={resumeIcSession.sessionId}
            onBack={() => setView("project")}
            onOpenClassifier={goToIc}
          />
        );
      // A saved session's page is normally still pending, but it doesn't have
      // to be (its page may have been encoded by some other route). Add it
      // back rather than silently ignoring the click - and so the queued XML
      // is paired with the right image, which needs the page to be selectable.
      const images =
        resumeImage && !pending.some((img) => img.id === resumeImage.id)
          ? [...pending, resumeImage]
          : pending;
      return (
        <InteractiveClassifier
          images={images}
          initialImageName={resumeImage?.name ?? null}
          usedImageCount={selectedProject.usedImageNames.length}
          onBack={() => setView("project")}
          projectId={selectedProjectId}
          clefShape={clefShape}
          onClefShapeChange={setClefShape}
          clefLine={clefLine}
          onClefLineChange={setClefLine}
          onEncodeBatch={(pairs) => {
            setPendingBatchPairs(pairs);
            setBatchSummary(null);
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 2),
              );
            }
            setView("encoding-processing");
          }}
        />
      );
    }
    case "ic-completion":
      return (
        <IcCompletionTestPage
          onContinue={() => setView("encoding-processing")}
          onBackToProject={() => setView("project")}
          xmlFile={pendingXmlFile}
          onXmlFileChange={setPendingXmlFile}
          imageFile={pendingImageFile}
          onImageFileChange={setPendingImageFile}
        />
      );
    case "encoding-processing": {
      if (pendingBatchPairs.length > 0) {
        return (
          <ProcessingPage
            {...STEP_TIMING}
            logs={encodingLogs}
            onBack={() => goToIc()}
            onComplete={() => {
              if (selectedProjectId && selectedProject) {
                updateProjectSteps(
                  selectedProjectId,
                  Math.max(selectedProject.stepsUnlocked, 3),
                );
              }
              setPendingBatchPairs([]);
              setView("encoding-completion");
            }}
            projectId={selectedProjectId}
            jobKind="encode_batch"
            streamRequest={(signal, onJobId) => {
              const form = new FormData();
              pendingBatchPairs.forEach((pair) =>
                form.append("xml_files", pair.xmlFile),
              );
              pendingBatchPairs.forEach((pair) =>
                form.append("image_files", pair.imageFile),
              );
              pendingBatchPairs.forEach((pair) =>
                form.append("image_names", pair.imageFile.name),
              );
              form.append("clef_shape", clefShape);
              form.append("clef_line", String(clefLine));
              if (selectedProjectId)
                form.append("project_id", String(selectedProjectId));
              return apiFetchJobStream(
                "/api/encode-batch",
                { method: "POST", body: form },
                signal,
                onJobId,
              );
            }}
            onResult={handleEncodeBatchResult}
            onLogsReady={setEncodingLogs}
            onBatchDone={setBatchSummary}
          />
        );
      }
      return pendingXmlFile ? (
        <ProcessingPage
          {...STEP_TIMING}
          singleLabel={
            pendingImageFile ? `encoding ${pendingImageFile.name}` : "encoding"
          }
          logs={encodingLogs}
          onBack={() => goToIc()}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 3),
              );
            }
            setView("encoding-completion");
          }}
          projectId={selectedProjectId}
          jobKind="encode_upload"
          streamRequest={(signal, onJobId) => {
            const form = new FormData();
            form.append("xml_file", pendingXmlFile!);
            if (pendingImageFile) {
              form.append("image_file", pendingImageFile);
              form.append("image_name", pendingImageFile.name);
              form.append("clef_shape", clefShape);
              form.append("clef_line", String(clefLine));
              if (selectedProjectId)
                form.append("project_id", String(selectedProjectId));
            }
            return apiFetchJobStream(
              "/api/encode-upload",
              { method: "POST", body: form },
              signal,
              onJobId,
            );
          }}
          onResult={handleEncodeResult}
          onLogsReady={setEncodingLogs}
        />
      ) : null;
    }
    case "encoding-completion": {
      const remainingIcImages = selectedProject
        ? pendingIcImages(
            selectedProject.images,
            selectedProject.usedImageNames,
            selectedProject.annotations ?? [],
            selectedProject.meiFiles ?? [],
            selectedProject.stepsUnlocked,
          )
        : [];

      const batchDescription = batchSummary
        ? `batch encoding complete: ${batchSummary.succeeded.length} succeeded${
            batchSummary.failed.length
              ? `, ${batchSummary.failed.length} failed`
              : ""
          }.`
        : null;

      return (
        <CompletionPage
          description={
            batchDescription ??
            (remainingIcImages.length > 0
              ? `encoding complete! ${remainingIcImages.length} page${remainingIcImages.length > 1 ? "s" : ""} still need${remainingIcImages.length === 1 ? "s" : ""} classifying.`
              : "encoding successfully completed! you can now view mei files on the project page, and send them to cantus ultimus.")
          }
          continueLabel="correction"
          onContinue={() => setView("neon-editor")}
          onBackToProject={() => setView("project")}
          logsFileName="encoding-logs.txt"
          logContent={encodingLogs.join("\n")}
          onDownloadMei={meiContent ? handleDownloadMei : undefined}
          onDownloadManifest={meiContent ? handleDownloadManifest : undefined}
          onClassifyMore={
            remainingIcImages.length > 0 ? () => goToIc() : undefined
          }
          classifyMoreCount={
            remainingIcImages.length > 0 ? remainingIcImages.length : undefined
          }
        />
      );
    }
    case "neon-editor":
      return selectedProject && selectedProject.meiFiles.length > 0 ? (
        <NeonBatchEditor
          project={selectedProject}
          meiFiles={selectedProject.meiFiles}
          onFileCorrected={(id) =>
            setProjects((prev) =>
              prev.map((p) =>
                p.id === selectedProjectId
                  ? {
                      ...p,
                      meiFiles: p.meiFiles.map((f) =>
                        f.id === id ? { ...f, corrected: true } : f,
                      ),
                    }
                  : p,
              ),
            )
          }
          onFinish={() => {
            setOriginalMeiFiles([...(selectedProject?.meiFiles ?? [])]);
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 4),
              );
            }
            setView("neon-completion");
          }}
          onBack={() => setView("encoding-completion")}
        />
      ) : null;
    case "neon-completion":
      return selectedProject ? (
        <NeonCompletionPage
          project={selectedProject}
          originalMeiFiles={originalMeiFiles}
          onSendToCantus={handleSendToCantus}
          sendingBundle={sendingBundle}
          sendBundleError={sendBundleError}
          onBackToProject={() => setView("project")}
        />
      ) : null;
    case "send-completion":
      return (
        <CompletionPage
          description="MEI bundle downloaded! hand this zip off to a Cantus Ultimus maintainer — it includes a README.txt with the exact steps to commit it to production_mei_files and index it."
          continueHref="https://cantus.simssa.ca/"
          continueLabel="view cantus ultimus"
          onBackToProject={() => setView("project")}
        />
      );
    default:
      return (
        <AuthPage
          mode={view as "login" | "register"}
          onSwitchMode={(m) => setView(m)}
          onSuccess={handleLoginSuccess}
        />
      );
  }
}
