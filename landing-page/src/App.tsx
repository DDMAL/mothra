import { useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Features from "./components/Features";
import Footer from "./components/Footer";
import AuthPage from "./components/AuthPage";
import MyProjects from "./components/MyProjects";
import ProjectDetail from "./components/ProjectDetail";
import About from "./components/About";
import ProcessingPage from "./components/workflow/ProcessingPage";
import CompletionPage from "./components/workflow/CompletionPage";
import InteractiveClassifier from "./components/workflow/InteractiveClassifier";
import Documentation from "./components/documentation/Documentation";
import IcCompletionTestPage from "./components/workflow/ICCompletionTestPage";
import MyAccount from "./components/MyAccount";

import type { CurrentUser } from "./hooks/useAuth";
import { getToken, setToken, clearToken, authHeaders } from "./hooks/useAuth";

type View =
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

export interface ProjectImage {
  id: string;
  name: string;
  src?: string;
}
export interface Project {
  id: number;
  name: string;
  user: string;
  images: ProjectImage[];
  models: ProjectModel[];
  annotations: AnnotationSet[];
  meiFiles: MeiFile[];
  stepsUnlocked: number;
  usedImageNames: string[];
  usedModelNames: string[];
  deletedAt?: number;
  lastOpenedAt?: string;
  isPinned?: boolean;
}
export interface ProjectModel {
  id: string;
  name: string;
}

export interface AnnotationSet {
  id: string;
  imageName: string;
  imageSrc?: string;
  jsonName: string;
  txtName: string;
}

export interface MeiFile {
  id: string;
  name: string;
  xmlContent?: string;
  corrected?: boolean;
}

export default function App() {
  const [view, setView] = useState<View>("landing");
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);

  const normalizeProjects = (raw: Project[]) =>
    raw.map(p => ({ ...p, images: p.images.map(img => ({ ...img, src: img.src ?? `/api/images/${img.id}` })) }));
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(null);
  const [encodingLogs, setEncodingLogs] = useState<string[]>([]);
  const [pendingXmlFile, setPendingXmlFile] = useState<File | null>(null);
  const [pendingImageFile, setPendingImageFile] = useState<File | null>(null);
  const [neonManifest, setNeonManifest] = useState<Record<string, unknown> | null>(null);
  const [meiContent, settleMeiContent] = useState<{ bytes: string; stem: string } | null>(null);

  const selectedProject = projects.find((p) => p.id === selectedProjectId) ?? null;

  const togglePin = (id: number) => {
    const project = projects.find(p => p.id === id);
    if (!project) return;
    const isPinned = !project.isPinned;
    fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ isPinned }),
    });
    setProjects(prev => prev.map(p => p.id === id ? { ...p, isPinned } : p));
  };

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

  // project mutations

  const createProject = async (name: string) => { 
    const r = await fetch("/api/projects", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json"},
      body: JSON.stringify({ name }),
    });
    const project = await r.json();
    setProjects(prev => [...prev, project]);
  }

  const renameProject = async (id: number, newName: string) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName }),
    });
    setProjects((prev) => prev.map((p) => p.id === id ? { ...p, name: newName }: p));
  };

  const deleteProject = async (id: number) => {
    const deletedAt = new Date().toISOString();
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ deletedAt }),
    });
    setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, deletedAt: Date.now() } : p)));
  };

  const restoreProject = async (id: number) => {
    await fetch(`/api/projects/${id}/restore`, {
      method: "POST", headers: authHeaders(),
    });
    setProjects(prev => prev.map(p => p.id === id ? { ...p, deletedAt: undefined} : p));
  };

  const permanentlyDeleteProject = async (id: number) => {
    await fetch(`/api/projects/${id}`, { method: "DELETE", headers: authHeaders() })
    setProjects(prev => prev.filter(p => p.id !== id));
  }

  const updateProjectSteps = async (id: number, steps: number) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json"},
      body: JSON.stringify({ stepsUnlocked: steps }),
    });
    setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, stepsUnlocked: steps } : p)));
  };

  const updateUsedImageNames = async (id: number, names: string[]) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json"},
      body: JSON.stringify({ usedImageNames: names }),
    });
    setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, usedImageNames: names} : p)));
  };

  const updateUsedModelNames = async (id: number, names: string[]) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json"},
      body: JSON.stringify({ usedModelNames: names }),
    });
    setProjects((prev) => prev.map((p) => (p.id === id ? { ...p, usedModelNames: names} : p)));
  };


  // download helpers

  const handleDownloadManifest = () => {
    if (!neonManifest || !meiContent) return;
    const blob = new Blob([JSON.stringify(neonManifest, null, 2)], { type: "application/ld+json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `${meiContent.stem}_manifest.jsonld`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadMei = () => {
      if (!meiContent?.bytes) return;
      const bytes = Uint8Array.from(atob(meiContent.bytes), (c) => c.charCodeAt(0));
      const blob = new Blob([bytes], { type: "application/xml" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${meiContent.stem}.mei`;
      a.click();
      URL.revokeObjectURL(url);
    }

  

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

  // encoding effect

  useEffect(() => {
    if (view !== "encoding-processing") return;
    settleMeiContent(null);
    setNeonManifest(null);

    if (pendingXmlFile) {
      const form = new FormData();
      form.append("xml_file", pendingXmlFile);
      if (pendingImageFile) {
        form.append("image_file", pendingImageFile);
      }
      fetch("/api/encode-upload", { method: "POST", body: form })
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then(async (data) => {
          setNeonManifest(data.manifest ?? null);
          setEncodingLogs(data.logs ?? []);
          settleMeiContent({ bytes: data.mei_base64, stem: pendingXmlFile.name.replace(".xml", "")});

          const xmlBytes = Uint8Array.from(atob(data.mei_base64), (c) => c.charCodeAt(0));
          const xmlText = new TextDecoder().decode(xmlBytes);
          const stem = pendingXmlFile.name.replace(".xml", "");
          const newMeiFile: MeiFile = {
            id: crypto.randomUUID(),
            name: `${stem}.mei`,
            xmlContent: xmlText,
            corrected: false,
          };
          if (selectedProjectId) {
            const r = await fetch(`/api/projects/${selectedProjectId}/mei`, {
              method: "POST",
              headers: { ...authHeaders(), "Content-Type": "application/json"},
              body: JSON.stringify({ name: newMeiFile.name, xmlContent: xmlText }),
            });
            const saved = await r.json();
            newMeiFile.id = saved.id;
          }

          setProjects((prev) => 
            prev.map((p) => 
              p.id === selectedProjectId
                ? { ...p, meiFiles: [...p.meiFiles, newMeiFile]}
              : p,
            ),
          );
        })
        .catch((err) => console.error("Encoding failed:", err));
    } else {
      // mock fallback
      fetch("/api/encode", { method: "POST"})
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then((data) => {
          setEncodingLogs(data.logs ?? []);
        })
        .catch((err) => console.error("encoding failed:", err));
    }
  }, [view]);
  
  
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
      {view === "landing" ? (
        <main>
          <Hero onGetStarted={() => setView("register")} onViewWalkthrough={() => setView("docs")} />
          <Features />
        </main>
      ) : view === "about" ? (
        <About />
      ) : view === "projects" ? (
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
      ) : view === "project" && selectedProject ? (
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
            setProjects((prev) =>
              prev.map((p) => (p.id === updated.id ? updated : p)),
            )
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
      ) : view === "processing" ? (
        <ProcessingPage 
          onBack={() => setView("project")}
          onComplete={() => {
            if (selectedProjectId && selectedProject) {
              updateProjectSteps(selectedProjectId, Math.max(selectedProject.stepsUnlocked, 1));
            }
            setView("completion");
          }} />
      ) : view === "completion" ? (
        <CompletionPage
          onContinue={() => setView("ic")}
          onBackToProject={() => setView("project")}
          logsFileName="annotatorlogs.txt" />
      ) : view === "ic" && selectedProject ? (
          <InteractiveClassifier
            images={selectedProject.images.filter((img) => 
              selectedProject.usedImageNames.includes(img.name)
            )}
            onProcessAll={() => setView("ic-processing")}
          />
      ) : view === "ic-processing" ? (
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
            completionDelayMs={4000}/>
      ) : view === "ic-completion" ? (
          <IcCompletionTestPage
            onContinue={() => setView("encoding-processing")}
            onBackToProject={() => setView("project")}
            logsFileName="iclogs.txt"
            xmlFile={pendingXmlFile}
            onXmlFileChange={setPendingXmlFile}
            imageFile={pendingImageFile}
            onImageFileChange={setPendingImageFile} />
      ) : view === "encoding-processing" ? (
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
            logs={encodingLogs} />
      ) : view === "encoding-completion" ? (
        <CompletionPage
          description="encoding successfully completed! you can now view mei files on the project page, and send them to cantus ultimus."
          continueLabel="correction"
          continueHref="https://ddmal.ca/Neon/"
          logsFileName="encodinglogs.txt"
          onBackToProject={() => setView("project")}
          onDownloadMei={meiContent ? handleDownloadMei : undefined}
          onDownloadManifest={meiContent ? handleDownloadManifest : undefined} />
      ) : view === "sending" ? (
        <ProcessingPage 
          onBack={() => setView("project")}
          onComplete={() => setView("send-completion")}
          singleLabel="sending..."
          intervalMs={60}
          completionDelayMs={4000}
          />
      ) : view === "send-completion" ? (
        <CompletionPage
          description="voila, sent to cantus ultimus!"
          logsFileName="sendlogs.txt"
          continueHref="https://cantus.simssa.ca/"
          continueLabel="view on cantus ultimus"
          onBackToProject={() => setView("project")}
        />
      ) : view === "docs" ? (
        <Documentation onHome={() => setView("landing")} />
      ) : view === "account" && currentUser ? (
        <MyAccount 
          currentUser={currentUser} 
          onUserUpdate={(u) => setCurrentUser(u)}
          onLogout={handleLogout}
         />
      ) : (
        <AuthPage
          mode={view as "login" | "register"}
          onSwitchMode={(m) => setView(m)}
          onSuccess={handleLoginSuccess}
        />
      )}
      <Footer />
    </div>
  );
}
