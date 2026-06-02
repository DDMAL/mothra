import { useEffect, useState } from "react";
import Navbar from "./components/Navbar";
import Hero from "./components/Hero";
import Features from "./components/Features";
import Footer from "./components/Footer";
import AuthPage from "./components/AuthPage";
import MyProjects from "./components/MyProjects";
import MyModels from "./components/MyModels";
import ProjectDetail from "./components/ProjectDetail";
import About from "./components/About";

type View =
  | "landing"
  | "about"
  | "login"
  | "register"
  | "projects"
  | "project"
  | "models";

export interface ProjectImage {
  id: string;
  name: string;
  src?: string;
}
export interface Project {
  name: string;
  user: string;
  images: ProjectImage[];
}
export interface ProjectModel {
  id: string;
  name: string;
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
    },
    {
      name: "project beta",
      user: "username",
      images: Array.from({ length: 7 }, (_, i) => ({
        id: String(i + 1),
        name: `image ${i + 1}`,
      })),
    },
  ]);
  const [models, setModels] = useState<ProjectModel[]>([]);
  const [selectedProject, setSelectedProject] = useState<string | null>(null);

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
        onMyModels={() => setView("models")}
        onAbout={() => setView("about")}
        loggedIn={
          view === "projects" || view === "project" || view === "models"
        }
        onHome={() => setView("landing")}
        onLogout={() => setView("landing")}
      />
      {view === "landing" ? (
        <main>
          <Hero onGetStarted={() => setView("register")} />
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
          onUpdateProject={(updated) =>
            setProjects((prev) =>
              prev.map((p) => (p.name === updated.name ? updated : p)),
            )
          }
        />
      ) : view === "models" ? (
        <MyModels models={models} onUpdateModels={setModels} />
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
