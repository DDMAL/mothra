import { useEffect, useState } from "react";
import Navbar from "./components/layout/Navbar";
import Hero from "./components/landing/Hero";
import Features from "./components/landing/Features";
import Footer from "./components/layout/Footer";
import AuthPage from "./components/auth/AuthPage";
import MyProjects from "./components/project/MyProjects";
import ProjectDetail from "./components/project/ProjectDetail";
import About from "./components/landing/About";
import ProcessingPage from "./components/workflow/ProcessingPage";
import CompletionPage from "./components/workflow/CompletionPage";
import InteractiveClassifier from "./components/workflow/InteractiveClassifier";
import Documentation from "./components/documentation/Documentation";
import IcCompletionTestPage from "./components/workflow/ICCompletionTestPage";
import MyAccount from "./components/account/MyAccount";
import type { Project } from "./types";

import type { CurrentUser } from "./hooks/useAuth";
import { getToken, setToken, clearToken, authHeaders } from "./hooks/useAuth";
import { useProjectMutations } from "./hooks/useProjectMutations";
import { useEncodingFlow } from "./hooks/useEncodingFlow";

export type View =
  | "landing"
  | "about"
  | "login"
  | "register"
  | "account"
  | "docs"
  | "projects"
  | "project"
  | "processing"
  | "completion"
  | "ic"
  | "ic-processing"
  | "ic-completion"
  | "encoding-processing"
  | "encoding-completion"
  | "sending"
  | "send-completion"
  | "neon-test";


export default function App() {
  const [view, setView] = useState<View>("landing");
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  const normalizeProjects = (raw: Project[]) =>
    raw.map(p => ({ ...p, images: p.images.map(img => ({ ...img, src: img.src ?? `/api/images/${img.id}` })) }));

  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);

  const selectedProject = projects.find((p) => p.id === selectedProjectId) ?? null;

  const {
    createProject, renameProject, deleteProject, restoreProject, permanentlyDeleteProject, 
    updateProjectSteps, updateUsedImageNames, updateUsedModelNames, togglePin,
  } = useProjectMutations(setProjects);

  const {
    encodingLogs,
    pendingXmlFile, setPendingXmlFile,
    pendingImageFile, setPendingImageFile,
    meiContent,
    handleDownloadManifest,
    handleDownloadMei,
  } = useEncodingFlow(view, selectedProjectId, setProjects);

  // auth

  const handleLoginSuccess = (user: CurrentUser, token: string) => {
    setToken(token);
    setCurrentUser(user);
    fetch("/api/projects", { headers: authHeaders() })
      .then(r => r.json())
      .then(data => setProjects(normalizeProjects(data)));
    setView("projects");
  };

  const handleLogout = () => {
    clearToken();
    setCurrentUser(null);
    setProjects([]);
    setSelectedProjectId(null);
    setView("landing");
  };

  

    // session restore

    useEffect(() => {
      const token = getToken();
      if (!token) return;
      fetch("/api/me", { headers: { Authorization: `Bearer ${token}` } })
        .then((r) => (r.ok? r.json() : Promise.reject()))
        .then((user) => {
          setCurrentUser(user);
          return fetch("/api/projects", { headers: { Authorization: `Bearer ${token}` } });
        })
        .then((r) => r.json())
        .then(data => setProjects(normalizeProjects(data)))
        .catch(() => clearToken());
    }, []);
  
  
  useEffect(() => {
    if (view !== "landing" && view !== "about") {
      document
        .querySelectorAll(".fade-target")
        .forEach((el) => el.classList.add("visible"));
      return;
    }

    let timer: ReturnType<typeof setTimeout> | undefined;
    if (view === "landing") {
      const heroTargets = document.querySelectorAll(".hero-fade");
      timer = setTimeout(() => {
        heroTargets.forEach((el) => el.classList.add("visible"));
      }, 100);
    }

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            entry.target.classList.add("visible");
            observer.unobserve(entry.target);
          }
        });
      },
      { threshold: 0.1 },
    );


    document
      .querySelectorAll(".scroll-fade")
      .forEach((el) => observer.observe(el));

    return () => {
      if (timer) clearTimeout(timer);
      observer.disconnect();
    };
  }, [view]);

  function renderContent() {
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
              setProjects(prev => prev.map(p => p.id === id ? { ...p, lastOpenedAt: now }: p));
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
              if (selectedProject.stepsUnlocked >= 3) window.open("https://ddmal.ca/Neon", "_blank");
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
              else if (step === 3) window.open("https://ddmal.ca/Neon/");
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
              return r.json();
            }}
            onUploadModel={async (name) => {
              const r = await fetch(`/api/projects/${selectedProject.id}/models`, {
                method: "POST",
                headers: { ...authHeaders(), "Content-Type": "application/json"},
                body: JSON.stringify({ name }),
              });
              return r.json();
            }}
            onDeleteImage={async (imageId) => {
              await fetch(`/api/projects/${selectedProject.id}/images/${imageId}`, {
                method: "DELETE",
                headers: authHeaders(),
              });
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
            onBack={() => setView("ic")}
            onComplete={() => {
              if (selectedProjectId && selectedProject) {
                updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 2));
              }
              setView("ic-completion");
            }}
            singleLabel="classifying all pages"
            intervalMs={60}
            completionDelayMs={4000}
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
            onBack={() => setView("ic-completion")}
            onComplete={() => {
              if (selectedProjectId && selectedProject) {
                updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 3));
              }
              setView("encoding-completion");
            }}
            singleLabel="processing"
            intervalMs={60}
            completionDelayMs={4000}
            logs={encodingLogs}
          />
        );
      case "encoding-completion":
        return (
          <CompletionPage
            description="encoding successfully completed! you can now view mei files on the project page, and send them to cantus ultimus."
            continueLabel="correction"
            continueHref="https://ddmal.ca/Neon/"
            logsFileName="encodinglogs.txt"
            onBackToProject={() => setView("project")}
            onDownloadMei={meiContent ? handleDownloadMei : undefined}
            onDownloadManifest={meiContent ? handleDownloadManifest : undefined}
          />
        );
      case "sending":
        return (
          <ProcessingPage
            onBack={() => setView("project")}
            onComplete={() => setView("send-completion")}
            singleLabel="sending..."
            intervalMs={60}
            completionDelayMs={4000}
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

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar
        currentUser={currentUser}
        onLogout={handleLogout}
        onLogin={() => setView("login")}
        onGetStarted={() => setView("register")}
        onMyProjects={() => setView("projects")}
        onAbout={() => setView("about")}
        onDocs={() => setView("docs")}
        onHome={() => setView("landing")}
        onAccount={() => setView("account")}
      />
      {renderContent()}
      <Footer />
    </div>
  );
}
