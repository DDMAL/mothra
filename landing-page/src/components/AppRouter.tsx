import { useState, useEffect } from "react";
import type { Dispatch, SetStateAction } from "react";
import type { View, Project, AnnotationSet, MeiFile, ModelKind } from "../types";
import type { CurrentUser } from "../hooks/useAuth";
import { apiFetch } from "../lib/apiFetch";
import { getImageProgress, minNextStep } from "../utils/imageStep";
import { downloadBlob } from "../utils/download";
import type { useProjectMutations } from "../hooks/useProjectMutations";
import Hero from "./landing/Hero";
import Features from "./landing/Features";
import About from "./landing/About";
import Documentation from "./documentation/Documentation";
import AuthPage from "./auth/AuthPage";
import MyAccount from "./account/MyAccount";
import MyProjects from "./project/MyProjects";
import ProjectDetail from "./project/ProjectDetail";
import ProcessingPage from "./workflow/ProcessingPage";
import CompletionPage from "./workflow/CompletionPage";
import InteractiveClassifier from "./workflow/InteractiveClassifier";
import IcCompletionTestPage from "./workflow/ICCompletionTestPage";
import NeonCompletionPage from "./workflow/NeonCompletionPage";
import NeonBatchEditor from "./workflow/NeonBatchEditor";

const STEP_TIMING = { intervalMs: 60, completionDelayMs: 4000 } as const;

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
  handleEncodeResult: (ev: { session_id: string; mei_base64: string; manifest: Record<string, unknown> | null }) => void;
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
    togglePin,
  } = mutations;
  const [encodingLogs, setEncodingLogs] = useState<string[]>([]);
  const [originalMeiFiles, setOriginalMeiFiles] = useState<MeiFile[]>([]);

  // thread inference settings
  const [inferenceThreshold, setInferenceThreshold] = useState(0.5);
  const [inferenceDevice, setInferenceDevice] = useState<"cpu" | "cuda" | "mps">("cpu");

  // thread text-finding settings (mothra-text optional inputs)
  const [textColumnCount, setTextColumnCount] = useState<"auto" | "1" | "2">("auto");
  const [textSegmentationModelId, setTextSegmentationModelId] = useState("");
  const [textRecognitionModelId, setTextRecognitionModelId] = useState("");
  const [textDevice, setTextDevice] = useState<"cpu" | "cuda">("cpu");
  const [textColumnBimodalThreshold, setTextColumnBimodalThreshold] = useState(0.5);

  // thread clef settings
  const [clefShape, setClefShape] = useState<"C" | "F">("C");
  const [clefLine, setClefLine] = useState(3);

  useEffect(() => {
    const PROJECT_VIEWS: View[] = [
    "project", "processing", "completion", "ic", "encoding-processing", "encoding-completion", "neon-editor", "neon-completion", "sending",
    ];
    if (PROJECT_VIEWS.includes(view) && !selectedProject) setView("projects");
  }, [view, selectedProject]);
  
  switch (view) {
    case "landing":
      return (
        <main>
          <Hero
            onGetStarted={() => setView("register")}
            onViewWalkthrough={() => setView("docs")}
          />
          <Features />
        </main>
      );
    case "about":
      return <About />;
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
            );
            if (step >= 4) setView("sending");
            else if (step >= 3) setView("neon-editor");
            else if (step >= 1) setView("ic");
            else setView("processing");
          }}
          onUpdateProject={(updated) =>
            setProjects((prev) =>
              prev.map((p) => (p.id === updated.id ? updated : p)),
            )
          }
          onStepClick={(step) => {
            if (step === 0) setView("processing");
            else if (step === 1) setView("ic");
            else if (step === 2) setView("ic-completion");
            else if (step === 3) setView("neon-editor");
          }}
          onSendToCantus={() => setView("sending")}
          onRenameProject={(newName) =>
            renameProject(selectedProject.id, newName)
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
          onUploadImage={async (file) => {
            const form = new FormData();
            form.append("file", file);
            const r = await apiFetch(
              `/api/projects/${selectedProject.id}/images`,
              {
                method: "POST",
                body: form,
              },
            );
            if (!r.ok) {
              const d = await r.json().catch(() => ({}));
              throw new Error(
                (d as { detail?: string }).detail || "upload failed",
              );
            }
            return r.json();
          }}
          onUploadModel={async (file: File, kind: ModelKind) => {
            const form = new FormData();
            form.append("file", file);
            form.append("kind", kind);
            const r = await apiFetch(
              `/api/projects/${selectedProject.id}/models`,
              {
                method: "POST",
                body: form,
              });
            return r.json();
          }}
          onDeleteModel={async (modelId) => {
            await apiFetch(
              `/api/projects/${selectedProject.id}/models/${modelId}`,
              { method: "DELETE" },
            );
          }}
          onDeleteAnnotation={async (annotationId) => {
            await apiFetch(
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
                new Blob([yoloTxtToJson(data.yoloTxt, data.imageName)], { type: "application/json" }),
                `${stem}.json`,
              );
            } else {
              downloadBlob(new Blob([data.yoloTxt], { type: "text/plain" }), `${stem}.txt`);
            }
          }}
          onDeleteMei={async (meiId) => {
            await apiFetch(
              `/api/projects/${selectedProject.id}/mei/${meiId}`,
              { method: "DELETE" },
            );
          }}
          onDeleteImage={async (imageId) => {
            const r = await apiFetch(
              `/api/projects/${selectedProject.id}/images/${imageId}`,
              {
                method: "DELETE",
              },
            );
            if (!r.ok) {
              const d = await r.json().catch(() => ({}));
              throw new Error(
                (d as { detail?: string }).detail || "delete failed",
              );
            }
          }}
          onDeleteProject={() => {
            deleteProject(selectedProject.id);
            setView("projects");
          }}
          inferenceThreshold={inferenceThreshold}
          onInferenceThresholdChange={setInferenceThreshold}
          inferenceDevice={inferenceDevice}
          onInferenceDeviceChange={setInferenceDevice}
          textColumnCount={textColumnCount}
          onTextColumnCountChange={setTextColumnCount}
          textSegmentationModelId={textSegmentationModelId}
          onTextSegmentationModelIdChange={setTextSegmentationModelId}
          textRecognitionModelId={textRecognitionModelId}
          onTextRecognitionModelIdChange={setTextRecognitionModelId}
          textDevice={textDevice}
          onTextDeviceChange={setTextDevice}
          textColumnBimodalThreshold={textColumnBimodalThreshold}
          onTextColumnBimodalThresholdChange={setTextColumnBimodalThreshold}
        />
      ) : null;
    case "processing":
      return selectedProject ? (
        <ProcessingPage
          onBack={() => setView("project")}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 1),
              );
            }
            setView("completion");
          }}
          streamRequest={(signal) => {
            const usedModelId =
              selectedProject.models.find((m) =>
                (selectedProject.usedModelNames ?? []).includes(m.name),
              )?.id ?? "";
            const usedImageIds = selectedProject.images
              .filter((i) => selectedProject.usedImageNames.includes(i.name))
              .map((i) => i.id);
            return apiFetch(`/api/projects/${selectedProject.id}/predict`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                model_id: usedModelId,
                image_ids: usedImageIds,
                confidence_threshold: inferenceThreshold,
                device: inferenceDevice,
                text_column_count: textColumnCount === "auto" ? null : Number(textColumnCount),
                text_segmentation_model_id: textSegmentationModelId || null,
                text_recognition_model_id: textRecognitionModelId || null,
                text_device: textDevice,
                text_column_bimodal_threshold: textColumnBimodalThreshold,
              }),
              signal,
            });
          }}
          onResult={(ev: { annotations: AnnotationSet[] }) => {
            setProjects((prev) =>
              prev.map((p) =>
                p.id === selectedProject.id ? { ...p, annotations: ev.annotations } : p,
              ),
            );
          }}
        />
      ) : null;
    case "completion":
      return (
        <CompletionPage
          onContinue={() => setView("ic")}
          onBackToProject={() => setView("project")}
          logsFileName="annotatorlogs.txt"
          onDownloadAnnotations={
            selectedProject?.annotations?.length
              ? async () => {
                  for (const ann of selectedProject.annotations) {
                    const r = await apiFetch(
                      `/api/projects/${selectedProject.id}/annotations/${ann.id}`,
                    );
                    const data = await r.json();
                    const stem = (data.imageName as string).replace(/\.[^.]+$/, "");
                    downloadBlob(new Blob([data.yoloTxt], { type: "text/plain" }), `${stem}.txt`);
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
                    const stem = (data.imageName as string).replace(/\.[^.]+$/, "");
                    downloadBlob(
                      new Blob([yoloTxtToJson(data.yoloTxt, data.imageName)], { type: "application/json" }),
                      `${stem}.json`,
                    );
                  }
                }
              : undefined
          }
        />
      );
    case "ic":
      return selectedProject ? (
        <InteractiveClassifier
          images={selectedProject.images
            .filter((img) => {
              if (!selectedProject.usedImageNames.includes(img.name)) return false;
              const p = getImageProgress(img.name, selectedProject.annotations ?? [], selectedProject.meiFiles ?? []);
              return p === null || p.nextStep <= 1;
            })
            .sort((a, b) => {
              const pa = getImageProgress(a.name, selectedProject.annotations ?? [], selectedProject.meiFiles ?? []);
              const pb = getImageProgress(b.name, selectedProject.annotations ?? [], selectedProject.meiFiles ?? []);
              return (pa?.nextStep ?? 0) - (pb?.nextStep ?? 0);
            })
          }
          projectId={selectedProjectId}
          setPendingXmlFile={setPendingXmlFile}
          setPendingImageFile={setPendingImageFile}
          clefShape={clefShape}
          onClefShapeChange={setClefShape}
          clefLine={clefLine}
          onClefLineChange={setClefLine}
          onEncode={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(
                selectedProjectId,
                Math.max(selectedProject.stepsUnlocked, 2),
              );
            }
            setView("encoding-processing");
          }}
        />
      ) : null;
    case "ic-completion":
      return (
        <IcCompletionTestPage
          onContinue={() => setView("encoding-processing")}
          onBackToProject={() => setView("project")}
          logsFileName="iclogs.txt"
          xmlFile={pendingXmlFile}
          onXmlFileChange={setPendingXmlFile}
          imageFile={pendingImageFile}
          onImageFileChange={setPendingImageFile}
        />
      );
    case "encoding-processing":
      return pendingXmlFile ? (
        <ProcessingPage
          {...STEP_TIMING}
          singleLabel={pendingImageFile ? `encoding ${pendingImageFile.name}` : "encoding"}
          logs={encodingLogs}
          onBack={() => setView("ic")}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 3));
            }
            setView("encoding-completion");
          }}
          streamRequest={(signal) => {
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
            return apiFetch("/api/encode-upload", { method: "POST", body: form, signal });
          }}
          onResult={handleEncodeResult}
          onLogsReady={setEncodingLogs}
        />
  ) : null;
    case "encoding-completion": {
      const remainingIcImages = selectedProject?.images.filter((img) => {
        if (!selectedProject.usedImageNames.includes(img.name)) return false;
        const p = getImageProgress(
          img.name,
          selectedProject.annotations ?? [],
          selectedProject.meiFiles ?? [],
        );
        return p === null || p.nextStep <= 1;
      }) ?? [];

      return (
        <CompletionPage
          description={
            remainingIcImages.length > 0 ? `encoding complete! ${remainingIcImages.length} page${remainingIcImages.length > 1 ? "s" : ""} still need${remainingIcImages.length === 1 ? "s" : ""} classifying.`
            : "encoding successfully completed! you can now view mei files on the project page, and send them to cantus ultimus."}
          continueLabel="correction"
          onContinue={() => setView("neon-editor")}
          onBackToProject={() => setView("project")}
          logsFileName="encoding-logs.txt"
          logContent={encodingLogs.join("\n")}
          onDownloadMei={meiContent ? handleDownloadMei : undefined}
          onDownloadManifest={meiContent ? handleDownloadManifest : undefined}
          onClassifyMore={remainingIcImages.length > 0 ? () => setView("ic") : undefined}
          classifyMoreCount={remainingIcImages.length > 0 ? remainingIcImages.length : undefined}
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
                  ? { ...p, meiFiles: p.meiFiles.map((f) => f.id === id ? { ...f, corrected: true } : f) }
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
          onSendToCantus={() => setView("sending")}
          onBackToProject={() => setView("project")}
        />
      ) : null;
    case "sending":
      return (
        <ProcessingPage
          {...STEP_TIMING}
          singleLabel="sending..."
          onBack={() => setView("project")}
          onComplete={() => setView("send-completion")}
        />
      );
    case "send-completion":
      return (
        <CompletionPage
          description="voila, sent to cantus ultimus!"
          logsFileName="sendlogs.txt"
          continueHref="https://cantus.simssa.ca/"
          continueLabel="view on cantus ultimus"
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
