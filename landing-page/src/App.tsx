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

type View =
  | "landing"
  | "about"
  | "login"
  | "register"
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
  name: string;
  user: string;
  images: ProjectImage[];
  models: ProjectModel[];
  annotations: AnnotationSet[];
  meiFiles: MeiFile[];
  deletedAt?: number;
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
  const [projects, setProjects] = useState<Project[]>([
    {
      name: "project alpha",
      user: "username",
      images: [
        { id: "1", name: "image 1" },
        { id: "2", name: "image 2" },
        { id: "3", name: "image 3" },
      ],
      models: [],
      annotations: [],
      meiFiles: []
    },
    {
      name: "project beta",
      user: "username",
      images: Array.from({ length: 7 }, (_, i) => ({
        id: String(i + 1),
        name: `image ${i + 1}`,
      })),
      models: [],
      annotations: [],
      meiFiles: []
    },
  ]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);
  const [usedNames, setUsedNames] = useState<{
    images: string[];
    models: string[];
  }>({ images: [], models: [] });
  const [stepsUnlocked, setStepsUnlocked] = useState(0);
  const [encodingLogs, setEncodingLogs] = useState<string[]>([]);
  const [pendingXmlFile, setPendingXmlFile] = useState<File | null>(null);
  const [pendingImageFile, setPendingImageFile] = useState<File | null>(null);
  const [neonManifest, setNeonManifest] = useState<File | null>(null);
  const [meiContent, settleMeiContent] = useState<{ bytes: string; stem: string } | null>(null);

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

  useEffect(() => {
    if (view !== "encoding-processing") return;
    settleMeiContent(null); 

    if (pendingXmlFile) {
      const buildForm = async () => {
        const form = new FormData();
        form.append("xml_file", pendingXmlFile);
        if (pendingImageFile) {
          const img = new Image();
          const url = URL.createObjectURL(pendingImageFile);
          await new Promise<void>((res) => { img.onload = () => res(); img.src = url; });
          form.append("image_width", String(img.naturalWidth));
          form.append("image_height", String(img.naturalHeight));
          URL.revokeObjectURL(url);
        }
        return form;
      };
      buildForm().then((form) =>
      fetch("/api/encode-upload", { method: "POST", body: form }))
        .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
        .then((data) => {
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
          setProjects((prev) => 
            prev.map((p) => 
              p.name === selectedProject
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
          setNeonManifest(data.manifest ?? null);
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
        onLogin={() => setView("login")}
        onGetStarted={() => setView("register")}
        onMyProjects={() => setView("projects")}
        onAbout={() => setView("about")}
        onDocs={() => setView("docs")}
        loggedIn={view === "projects" || view === "project"}
        onHome={() => setView("landing")}
        onLogout={() => setView("landing")}
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
          setProjects={setProjects}
          onSelectProject={(name) => {
            setSelectedProject(name);
            setView("project");
          }}
        />
      ) : view === "project" && selectedProject ? (
        <ProjectDetail
          project={projects.find((p) => p.name === selectedProject)!}
          onBack={() => setView("projects")}
          onContinue={() => {
            if (stepsUnlocked >= 3) window.open("https://ddmal.ca/Neon/", "_blank");
            else if (stepsUnlocked >= 2) setView("ic-completion");
            else if (stepsUnlocked >= 1) setView("ic");
            else setView("processing");
          }}
          onUpdateProject={(updated) =>
            setProjects((prev) =>
              prev.map((p) => (p.name === updated.name ? updated : p)),
            )
          }
          onStepClick={(step) => {
            if (step === 1) setView("ic");
            else if (step === 2) setView("ic-completion");
            else if (step === 3) window.open("https://ddmal.ca/Neon/");
          }}
          onSendToCantus={() => setView("sending")}
          onRenameProject={(newName) => {
            setProjects((prev) =>
              prev.map((p) => (p.name === selectedProject ? { ...p, name: newName } : p)),
            );
            setSelectedProject(newName);
          }}
          usedNames={usedNames}
          onUsedNamesChange={setUsedNames}
          stepsUnlocked={stepsUnlocked}
        />
      ) : view === "processing" ? (
        <ProcessingPage 
          onBack={() => setView("project")}
          onComplete={() => {
            setStepsUnlocked((s) => Math.max(s, 1));
            setView("completion");
          }} />
      ) : view === "completion" ? (
        <CompletionPage
          onContinue={() => setView("ic")}
          onBackToProject={() => setView("project")}
          logsFileName="annotatorlogs.txt" />
      ) : view === "ic" && selectedProject ? (
          <InteractiveClassifier
            images={
              projects
                .find((p) => p.name === selectedProject)!
                .images.filter((img) => usedNames.images.includes(img.name))
            }
            onProcessAll={() => setView("ic-processing")}
          />
      ) : view === "ic-processing" ? (
          <ProcessingPage
            onBack={() => setView("ic")}
            onComplete={() => {
              setStepsUnlocked((s) => Math.max(s, 2));
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
              setStepsUnlocked((s) => Math.max(s, 3));
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
          onDownloadMei={meiContent ? handleDownloadMei : undefined} />
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
      ) : (
        <AuthPage
          mode={view as "login" | "register"}
          onSwitchMode={(m) => setView(m)}
        />
      )}
      <Footer />
    </div>
  );
}
