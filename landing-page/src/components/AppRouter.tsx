import { useState, useEffect, useRef } from "react";
import type { Dispatch, SetStateAction, RefObject } from "react";
import type {
  View,
  Project,
  AnnotationSet,
  MeiFile,
  ModelKind,
  ProjectImage,
  ProjectInitialTab,
} from "../types";
import type { CurrentUser } from "../hooks/useAuth";
import { apiFetch, apiFetchOrThrow, apiFetchJobStream } from "../lib/apiFetch";
import { minNextStep, pendingIcImages } from "../utils/imageStep";
import { downloadBlob } from "../utils/download";
import { latestMeiPerImage } from "../utils/mei";
import { yoloTxtToJson } from "../utils/yolo";
import type { useProjectMutations } from "../hooks/useProjectMutations";
import { useInferenceSettings } from "../hooks/useInferenceSettings";
import { useTextFindingSettings } from "../hooks/useTextFindingSettings";
import { useIcSettings } from "../hooks/useIcSettings";
import { useTutorialFlow } from "../hooks/useTutorialFlow";
import TutorialOverlay from "../tutorial/TutorialOverlay";
import { TUTORIAL_IMAGE_NAMES } from "../tutorial/tutorialSteps";
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
import IcAutoQueue from "./workflow/IcAutoQueue";
import IcSessionUnavailable from "./workflow/IcSessionUnavailable";
import IcCompletionTestPage from "./workflow/ICCompletionTestPage";
import NeonCompletionPage from "./workflow/NeonCompletionPage";
import NeonBatchEditor from "./workflow/NeonBatchEditor";
import type { NeonEditorHandle } from "./workflow/NeonBatchEditor";

// completionDelayMs used to be 4000 -- a purely cosmetic pause after the real
// work already finished, per mothra#220 DL-10. Trimmed to a brief settle
// animation instead (<1s, matching ProcessingPage's own progress-bar
// transition duration) so "done" reads as done.
const STEP_TIMING = { intervalMs: 60, completionDelayMs: 500 } as const;

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
    project.usedImageIds.includes(img.id),
  );
  if (used.length === 0 || !used.every((img) => img.folio)) return null;
  return {
    imageIds: used.map((img) => img.id),
    folios: used.map((img) => img.folio!),
  };
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
  pendingBatchPairs: { xmlFile: File; imageFile: File; imageId: string }[];
  setPendingBatchPairs: (
    pairs: { xmlFile: File; imageFile: File; imageId: string }[],
  ) => void;
  resumeJob: { jobId: string; kind: string; startedAt?: string | null} | null;
  setResumeJob: (
    job: { jobId: string; kind: string; startedAt?: string | null } | null,
  ) => void;
  pendingProjectTab: ProjectInitialTab | null;
  setPendingProjectTab: (tab: ProjectInitialTab | null) => void;
  // Lets App.tsx's browser back/forward popstate handler reuse
  // NeonBatchEditor's own unsaved-work confirmation gate (issue #266) --
  // see NeonEditorHandle's doc comment.
  neonEditorRef: RefObject<NeonEditorHandle | null>;
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
  resumeJob,
  setResumeJob,
  pendingProjectTab,
  setPendingProjectTab,
  neonEditorRef,
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
    updateUsedImageIds,
    updateUsedModelNames,
    updateCantusSourceId,
    togglePin,
  } = mutations;
  const [encodingLogs, setEncodingLogs] = useState<string[]>([]);
  const [annotationLogs, setAnnotationLogs] = useState<string[]>([]);
  const [originalMeiFiles, setOriginalMeiFiles] = useState<MeiFile[]>([]);

  // thread inference settings + text-finding settings (mothra-text optional inputs)
  const inferenceSettings = useInferenceSettings();
  const textFindingSettings = useTextFindingSettings();
  // IC step settings (auto/manual + shared training set), picked on the
  // project page — see IcSettingsSection.
  const icSettings = useIcSettings(selectedProjectId);
  // Guided-tour state for the auto-provisioned tutorial project (see
  // tutorial_store.py) -- see hooks/useTutorialFlow.ts.
  const tutorialFlow = useTutorialFlow(selectedProject, view, icSettings);

  // batch text-alignment run (run_chain.py) state
  const [batchRunIds, setBatchRunIds] = useState<{
    imageIds: string[];
    folios: string[];
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

  // Saved IC sessions to reopen on the IC step page -- one when it came from
  // the project page's "manage IC sessions" modal, any number when it came
  // from the IC step page's own picker (IcSessionPicker). Cleared by goToIc()
  // below, so every other route into that view starts on the first page to
  // classify.
  const [resumeIcSessions, setResumeIcSessions] = useState<IcResumeRequest[]>(
    [],
  );
  // mothra#294: set when the user clicks an already-progressed selected
  // image on the project page, so the "ic"/"neon-editor" cases land on that
  // exact page instead of the first pending/uncorrected one. Cleared by the
  // ordinary goToIc()/goToNeon() entry points so a stale click target can
  // never leak into a later plain Continue/step click.
  const [icFocusImageId, setIcFocusImageId] = useState<string | null>(null);
  const [neonFocusFileId, setNeonFocusFileId] = useState<string | null>(null);

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
      "ic-auto",
      "encoding-processing",
      "encoding-completion",
      "neon-editor",
      "neon-completion",
    ];
    if (PROJECT_VIEWS.includes(view) && !selectedProject) setView("projects");
  }, [view, selectedProject]);

  // Ordinary way into the IC step. Any saved-session resume left over from a
  // previous visit is dropped here, so only the "manage IC sessions" path
  // (which sets it) ever opens one - the request has to outlive the navigation
  // itself, since the "ic" case below keeps reading it to hold the session's
  // page in the filmstrip.
  //
  // "auto" mode never shows the classifier: it classifies and queues every
  // pending page with the project page's training set, then goes straight to
  // encoding (see IcAutoQueue). "manual" opens the classifier on the first
  // page to classify. A saved-session resume always goes to the classifier -
  // the user picked that session explicitly.
  const goToIc = () => {
    setResumeIcSessions([]);
    setIcFocusImageId(null);
    setView(icSettings.mode === "auto" ? "ic-auto" : "ic");
  };

  // mothra#294: entry point for clicking a specific "ic"-stage image (in the
  // Images tab grid or the "selected:" panel) -- lands the classifier on
  // that exact page instead of the first pending one. Auto mode has no
  // per-image picking UI, so icFocusImageId is simply unread there; this
  // still routes into auto mode's queue-everything-pending behavior, same as
  // goToIc() already does for the ordinary Continue path.
  const focusIc = (imageId: string) => {
    setResumeIcSessions([]);
    setIcFocusImageId(imageId);
    setView(icSettings.mode === "auto" ? "ic-auto" : "ic");
  };

  // mothra#294: ordinary entry into the Neon editor -- always lands on the
  // first uncorrected page (NeonBatchEditor's own default), never a stale
  // focus target left over from a previous click.
  const goToNeon = () => {
    setNeonFocusFileId(null);
    setView("neon-editor");
  };

  // Entry point for clicking a specific "neon"/"done"-stage image -- opens
  // the editor directly on that exact page.
  const focusNeon = (fileId: string) => {
    setNeonFocusFileId(fileId);
    setView("neon-editor");
  };

  // Tutorial hand-off: reaching these steps in the guided tour deep-links
  // into IC/Neon for that phase's dedicated fixture page (see
  // tutorial/tutorialSteps.ts's TUTORIAL_IMAGE_NAMES) via the same
  // focusIc/focusNeon entry points a real user clicking a specific image
  // already uses -- no separate routing path for the tutorial.
  useEffect(() => {
    if (!selectedProject) return;
    if (tutorialFlow.step?.id === "ic-handoff") {
      const img = selectedProject.images.find(
        (i) => i.name === TUTORIAL_IMAGE_NAMES.ic,
      );
      if (img) focusIc(img.id);
    }
    if (tutorialFlow.step?.id === "neon-handoff") {
      const mei = selectedProject.meiFiles.find(
        (f) => f.imageName === TUTORIAL_IMAGE_NAMES.neon,
      );
      if (mei) focusNeon(mei.id);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tutorialFlow.step?.id]);

  // Hand a finished IC queue (either mode) to the batch encoder.
  const startEncodeBatch = (
    pairs: { xmlFile: File; imageFile: File; imageId: string }[],
  ) => {
    setPendingBatchPairs(pairs);
    setBatchSummary(null);
    if (selectedProjectId && selectedProject) {
      updateProjectSteps(
        selectedProjectId,
        Math.max(selectedProject.stepsUnlocked, 2),
      );
    }
    setView("encoding-processing");
  };

  // Wrapped in an IIFE (rather than the switch returning directly from the
  // component) so TutorialOverlay can be mounted as a sibling of whatever
  // view is showing, below, instead of needing a render branch added to
  // every single case.
  const routedContent = (() => {
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
              selectedProject.images.filter((img) =>
                selectedProject.usedImageIds.includes(img.id),
              ),
              selectedProject.annotations ?? [],
              selectedProject.meiFiles ?? [],
              selectedProject.stepsUnlocked,
            );
            if (step >= 4) handleSendToCantus();
            else if (step >= 3) goToNeon();
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
            else if (step === 3) goToNeon();
          }}
          onResumeIcSession={(req) => {
            // IC's own manage page hands back one session at a time; the IC
            // step page's picker (IcSessionPicker) is what supplies several.
            setResumeIcSessions([req]);
            setView("ic");
          }}
          onSendToCantus={handleSendToCantus}
          onViewActiveJob={(jobId, kind, startedAt) => {
            setResumeJob({ jobId, kind, startedAt });
            setView("processing");
          }}
          initialTab={pendingProjectTab}
          onInitialTabConsumed={() => setPendingProjectTab(null)}
          sendingBundle={sendingBundle}
          sendBundleError={sendBundleError}
          onRenameProject={(newName) =>
            renameProject(selectedProject.id, newName)
          }
          onUpdateCantusSourceId={(sourceId) =>
            updateCantusSourceId(selectedProject.id, sourceId)
          }
          // mothra#241 follow-up (CodeRabbit): `images` now holds
          // project_images.id values, not names -- see
          // Project.usedImageIds's comment in types.ts. `models` is
          // unaffected (no duplicate-name concern there). mothra#294:
          // annotations are no longer a separately-"used" concept -- an
          // image's pipeline stage is derived (see utils/imageStep.ts), not
          // selected. `usedAnnotationNames`/`used_annotation_names` still
          // exist server-side, just never written from here anymore.
          usedNames={{
            images: selectedProject.usedImageIds,
            models: selectedProject.usedModelNames ?? [],
          }}
          onUsedNamesChange={(names) => {
            updateUsedImageIds(selectedProject.id, names.images);
            updateUsedModelNames(selectedProject.id, names.models);
          }}
          onFocusIcImage={focusIc}
          onFocusNeonFile={focusNeon}
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
          icSettings={icSettings}
          showTutorialStart={tutorialFlow.canStart}
          onStartTutorial={tutorialFlow.start}
        />
      ) : null;
    case "processing":
      return selectedProject ? (
        <ProcessingPage
          // Force a remount when a NEW resumed job is set while this exact
          // case is already rendering (onViewActiveJob/App.tsx's toast
          // handler can both fire while view is already "processing") --
          // without this, ProcessingPage's kickoff effect (keyed on
          // retryKey, not streamRequest) keeps streaming whatever job it
          // mounted with, so "view progress"/toast "view" on a second job
          // would silently keep showing the first one's logs.
          key={resumeJob?.jobId ?? "new"}
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
            const wasBatchRun = resumeJob
              ? resumeJob.kind === "text_batch"
              : !!batchRunIds;
            setResumeJob(null);
            // In auto IC mode, "continue to IC" on the completion page is a
            // no-op confirmation -- IcAutoQueue self-runs the moment it
            // mounts, so the click accomplishes nothing but an extra step.
            // Skip straight there for a plain detection run. Batch
            // text-finding runs still stop here regardless of mode: that
            // completion page is also the only place to download the batch
            // zip, which auto-IC has no equivalent for.
            if (!wasBatchRun && icSettings.mode === "auto") {
              goToIc();
            } else {
              setView("completion");
            }
          }}
          projectId={selectedProject.id}
          jobKind={resumeJob?.kind ?? (batchRunIds ? "text_batch" : "predict")}
          initialLogsOpen={!!resumeJob}
          startedAtMs={
            resumeJob?.startedAt ? new Date(resumeJob.startedAt).getTime() : undefined
          }
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
                // Filtered through selectedProject.images (not used
                // directly) so a stale usedImageIds entry referencing a
                // since-deleted image can't leak into the request.
                const usedImageIds = selectedProject.images
                  .filter((i) => selectedProject.usedImageIds.includes(i.id))
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
            // Effective kind for THIS mount: a resumed job's own kind wins
            // over whatever this tab last kicked off. batchRunIds reflects
            // this tab's last kickoff, not the job actually being watched --
            // resuming a text_batch job in a fresh tab (batchRunIds null)
            // would otherwise take the predict branch and read `ev.annotations`
            // off a batch result that doesn't carry it, and resuming a
            // predict job after this tab last ran a batch (batchRunIds still
            // set) would take the batch branch and write a bogus batchResult.
            const isBatchResult = resumeJob
              ? resumeJob.kind === "text_batch"
              : !!batchRunIds;
            if (isBatchResult) {
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
        selectedProject.usedImageIds,
        selectedProject.annotations ?? [],
        selectedProject.meiFiles ?? [],
        selectedProject.stepsUnlocked,
      );
      // The pages the resume requests point at. IC records mothra's image id
      // when the session is staged, so that's the authoritative match; the
      // file-name stem it also stores is a fallback only for sessions saved
      // without an id. Deliberately not a fallback for an id that fails to
      // resolve - that means the page was deleted, and a stem match could
      // land on a different image that reuses the filename (the session
      // wouldn't even be the one /ic/start finds for it).
      const resumeImages = resumeIcSessions
        .map((req) =>
          req.imageId == null
            ? selectedProject.images.find(
                (img) => img.name.replace(/\.[^.]+$/, "") === req.sourceName,
              )
            : selectedProject.images.find((img) => img.id === req.imageId),
        )
        .filter((img): img is ProjectImage => img != null);
      // An unresolvable resume must not fall through to the classifier: it
      // would mount on the first pending page, and queueing there would pair
      // that session's GameraXML with the wrong page's image. Only reported
      // when *nothing* resolved -- IcSessionPicker already refuses to select
      // an unresolvable session, so this is the "manage IC sessions" path
      // (which doesn't) and it only ever asks for one.
      if (resumeIcSessions.length > 0 && resumeImages.length === 0)
        return (
          <IcSessionUnavailable
            sourceName={resumeIcSessions[0].sourceName}
            sessionId={resumeIcSessions[0].sessionId}
            onBack={() => setView("project")}
            onOpenClassifier={goToIc}
          />
        );
      // A saved session's page is normally still pending, but it doesn't have
      // to be (its page may have been encoded by some other route). Add those
      // back rather than silently ignoring the click - and so each queued XML
      // is paired with the right image, which needs the page to be
      // selectable. Order: the pending pages first, then the reopened ones in
      // the order they were picked.
      const images = [
        ...pending,
        ...resumeImages.filter(
          (img, i) =>
            !pending.some((p) => p.id === img.id) &&
            resumeImages.findIndex((other) => other.id === img.id) === i,
        ),
      ];
      return (
        <InteractiveClassifier
          // Remount on a resume (or a mothra#294 focus click): `initialImageId`
          // is only read by a lazy useState initializer, so picking sessions
          // from *inside* this view (its own "saved sessions" button) would
          // otherwise change the prop with nothing reading it -- same `view`,
          // same element, no remount, and the click would look like a no-op.
          // Not strictly load-bearing for focusIc() itself (that always
          // navigates in from the project view, so `view` switching already
          // remounts this element fresh) -- included anyway for consistency.
          key={
            resumeIcSessions.map((r) => r.sessionId).join(",") ||
            icFocusImageId ||
            "fresh"
          }
          images={images}
          initialImageId={resumeImages[0]?.id ?? icFocusImageId ?? null}
          usedImageCount={selectedProject.usedImageIds.length}
          onBack={() => setView("project")}
          projectId={selectedProjectId}
          clefShape={clefShape}
          onClefShapeChange={setClefShape}
          clefLine={clefLine}
          onClefLineChange={setClefLine}
          trainingPresets={icSettings.trainingPresets}
          trainingFiles={icSettings.trainingFiles}
          onEncodeBatch={startEncodeBatch}
          allImages={selectedProject.images}
          onResumeIcSessions={setResumeIcSessions}
        />
      );
    }
    case "ic-auto": {
      if (!selectedProject) return null;
      return (
        <IcAutoQueue
          images={pendingIcImages(
            selectedProject.images,
            selectedProject.usedImageIds,
            selectedProject.annotations ?? [],
            selectedProject.meiFiles ?? [],
            selectedProject.stepsUnlocked,
          )}
          usedImageCount={selectedProject.usedImageIds.length}
          projectId={selectedProjectId}
          trainingPresets={icSettings.trainingPresets}
          trainingFiles={icSettings.trainingFiles}
          onDone={startEncodeBatch}
          onBack={() => setView("project")}
          onOpenManualClassifier={() => {
            setResumeIcSessions([]);
            setView("ic");
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
            // issue #272: in auto IC mode, goToIc() would land on "ic-auto",
            // a page with no UI of its own that immediately re-triggers
            // auto-classification the moment it mounts -- not a page worth
            // going "back" to. Manual mode's "ic" is a real, resumable
            // classifier session, so it keeps the original behavior.
            onBack={() =>
              icSettings.mode === "auto" ? setView("project") : goToIc()
            }
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
              pendingBatchPairs.forEach((pair) =>
                form.append("image_ids", pair.imageId),
              );
              form.append("clef_shape", clefShape);
              form.append("clef_line", String(clefLine));
              form.append("notation_type", icSettings.notationType);
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
              form.append("notation_type", icSettings.notationType);
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
            selectedProject.usedImageIds,
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
          onContinue={() => goToNeon()}
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
    case "neon-editor": {
      if (!selectedProject) return null;
      // One file per page, newest revision only. Re-encoding a page (which
      // reopening its saved IC session and pressing "encode batch" again
      // does) appends a second mei_files row rather than replacing the
      // first, and the editor would otherwise list the same page twice --
      // stale revision first. See utils/mei.ts.
      const neonMeiFiles = latestMeiPerImage(selectedProject.meiFiles);
      return neonMeiFiles.length > 0 ? (
        <NeonBatchEditor
          ref={neonEditorRef}
          project={selectedProject}
          meiFiles={neonMeiFiles}
          initialFileId={neonFocusFileId}
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
            // The same deduped list the editor just worked through, so the
            // completion page's before/after compare pairs one original per
            // page instead of a superseded revision.
            setOriginalMeiFiles([...neonMeiFiles]);
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 4),
              );
            }
            setNeonFocusFileId(null);
            setView("neon-completion");
          }}
          // issue #272: once you're editing in Neon, none of the
          // processing/IC/encoding stages that led here are worth
          // revisiting -- back always returns straight to the project page.
          onBack={() => {
            setNeonFocusFileId(null);
            setView("project");
          }}
        />
      ) : null;
    }
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
  })();

  return (
    <>
      {routedContent}
      {tutorialFlow.active && tutorialFlow.step && (
        <TutorialOverlay
          step={tutorialFlow.step}
          onNext={tutorialFlow.advance}
          onSkip={tutorialFlow.dismiss}
        />
      )}
    </>
  );
}
