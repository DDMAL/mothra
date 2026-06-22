import type { Dispatch, SetStateAction } from "react";
import type { View, Project } from "../types";
import type { CurrentUser } from "../hooks/useAuth";
import { authHeaders } from "../hooks/useAuth";
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
import NeonBatchEditor from "./workflow/NeonBatchEditor";

const STEP_TIMING = { intervalMs: 60, completionDelayMs: 4000 } as const;

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
  encodingLogs: string[];
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
}

export default function AppRouter({
  view, setView,
  currentUser, setCurrentUser,
  projects, setProjects,
  selectedProject, selectedProjectId, setSelectedProjectId,
  encodingLogs,
  pendingXmlFile, setPendingXmlFile,
  pendingImageFile, setPendingImageFile,
  meiContent, handleDownloadManifest, handleDownloadMei,
  handleLoginSuccess, handleLogout,
  mutations,
}: AppRouterProps) {
  const {
    createProject, renameProject, deleteProject, restoreProject,
    permanentlyDeleteProject, updateProjectSteps, updateUsedImageNames,
    updateUsedModelNames, togglePin,
  } = mutations;

  switch (view) {
    case "landing":
      return (
        <main>
          <Hero onGetStarted={() => setView("register")} onViewWalkthrough={() => setView("docs")} />
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
            fetch(`/api/projects/${id}`, {
              method: "PUT",
              headers: { ...authHeaders(), "Content-Type": "application/json" },
              body: JSON.stringify({ lastOpenedAt: now }),
            });
            setProjects(prev => prev.map(p => p.id === id ? { ...p, lastOpenedAt: now } : p));
          }}
          onCreateProject={createProject}
          onRenameProject={renameProject}
          onDeleteProject={deleteProject}
          onRestoreProject={restoreProject}
          onPermanentlyDeleteProject={permanentlyDeleteProject}
          onTogglePin={togglePin}
        />
      );
    case "project":
      return selectedProject ? (
        <ProjectDetail
          project={selectedProject}
          onBack={() => setView("projects")}
          onContinue={() => {
            if (selectedProject.stepsUnlocked >= 3) setView("neon-editor");
            else if (selectedProject.stepsUnlocked >= 2) setView("ic-completion");
            else if (selectedProject.stepsUnlocked >= 1) setView("ic");
            else setView("processing");
          }}
          onUpdateProject={(updated) =>
            setProjects((prev) => prev.map((p) => (p.id === updated.id ? updated : p)))
          }
          onStepClick={(step) => {
            if (step === 1) setView("ic");
            else if (step === 2) setView("ic-completion");
            else if (step === 3) setView("neon-editor");
          }}
          onSendToCantus={() => setView("sending")}
          onRenameProject={(newName) => renameProject(selectedProject.id, newName)}
          usedNames={{ images: selectedProject.usedImageNames, models: selectedProject.usedModelNames ?? [] }}
          onUsedNamesChange={(names) => {
            updateUsedImageNames(selectedProject.id, names.images);
            updateUsedModelNames(selectedProject.id, names.models);
          }}
          stepsUnlocked={selectedProject.stepsUnlocked}
          onUploadImage={async (file) => {
            const form = new FormData();
            form.append("file", file);
            const r = await fetch(`/api/projects/${selectedProject.id}/images`, {
              method: "POST",
              headers: authHeaders(),
              body: form,
            });
            if (!r.ok) {
              const d = await r.json().catch(() => ({}));
              throw new Error ((d as { detail?: string }).detail || "upload failed");
            }
            return r.json();
          }}
          onUploadModel={async (name) => {
            const r = await fetch(`/api/projects/${selectedProject.id}/models`, {
              method: "POST",
              headers: { ...authHeaders(), "Content-Type": "application/json" },
              body: JSON.stringify({ name }),
            });
            return r.json();
          }}
          onDeleteImage={async (imageId) => {
            const r = await fetch(`/api/projects/${selectedProject.id}/images/${imageId}`, {
              method: "DELETE",
              headers: authHeaders(),
            });
            if (!r.ok) {
              const d = await r.json().catch(() => ({}));
              throw new Error((d as { detail?: string }).detail || "delete failed");
            }
          }}
          onDeleteProject={() => {
            deleteProject(selectedProject.id);
            setView("projects");
          }}
        />
      ) : null;
    case "processing":
      return (
        <ProcessingPage
          onBack={() => setView("project")}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 1));
            }
            setView("completion");
          }}
        />
      );
    case "completion":
      return (
        <CompletionPage
          onContinue={() => setView("ic")}
          onBackToProject={() => setView("project")}
          logsFileName="annotatorlogs.txt"
        />
      );
    case "ic":
      return selectedProject ? (
        <InteractiveClassifier
          images={selectedProject.images.filter((img) =>
            selectedProject.usedImageNames.includes(img.name)
          )}
          onProcessAll={() => setView("ic-processing")}
        />
      ) : null;
    case "ic-processing":
      return (
        <ProcessingPage
          {...STEP_TIMING}
          singleLabel="classifying all pages"
          onBack={() => setView("ic")}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 2));
            }
            setView("ic-completion");
          }}
        />
      );
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
      return (
        <ProcessingPage
          {...STEP_TIMING}
          singleLabel="processing"
          logs={encodingLogs}
          onBack={() => setView("ic-completion")}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 3));
            }
            setView("encoding-completion");
          }}
        />
      );
    case "encoding-completion":
      return (
        <CompletionPage
          description="encoding successfully completed! you can now view mei files on the project page, and send them to cantus ultimus."
          continueLabel="correction"
          onContinue={() => setView("neon-editor")}
          onBackToProject={() => setView("project")}
          onDownloadMei={meiContent ? handleDownloadMei : undefined}
          onDownloadManifest={meiContent ? handleDownloadManifest : undefined}
        />
      );
    case "neon-editor":
      return selectedProject && selectedProject.meiFiles.length > 0 ? (
        <NeonBatchEditor
          project={selectedProject}
          meiFiles={selectedProject.meiFiles}
          onFinish={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 4));
            }
            setView("project");
          }}
          onBack={() => setView("encoding-completion")}
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
