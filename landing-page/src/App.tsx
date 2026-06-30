import { useEffect, useState } from "react";
import Navbar from "./components/layout/Navbar";
import Footer from "./components/layout/Footer";
import AppRouter from "./components/AppRouter";
import type { View, Project } from "./types";
import type { CurrentUser } from "./hooks/useAuth";
import { getToken, setToken, clearToken } from "./hooks/useAuth";
import { normalizeProjects } from "./utils/projects";
import { useProjectMutations } from "./hooks/useProjectMutations";
import { useEncodingFlow } from "./hooks/useEncodingFlow";
import { useScrollFade } from "./hooks/useScrollFade";
import { apiFetch, registerUnauthenticatedHandler } from "./lib/apiFetch";

export default function App() {
  const [view, setView] = useState<View>("landing");
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(
    null,
  );
  const [mutationError, setMutationError] = useState<string | null>(null);

  const selectedProject =
    projects.find((p) => p.id === selectedProjectId) ?? null;

  const mutations = useProjectMutations(setProjects, setMutationError);

  const {
    pendingXmlFile,
    setPendingXmlFile,
    pendingImageFile,
    setPendingImageFile,
    meiContent,
    handleDownloadManifest,
    handleDownloadMei,
    handleEncodeResult,
  } = useEncodingFlow(selectedProjectId, setProjects);

  useScrollFade(view);

  const handleLoginSuccess = (user: CurrentUser, token: string) => {
    setToken(token);
    setCurrentUser(user);
    apiFetch("/api/projects")
      .then((r) => r.json())
      .then((data) => setProjects(normalizeProjects(data)));
    setView("projects");
  };

  const handleLogout = () => {
    clearToken();
    setCurrentUser(null);
    setProjects([]);
    setSelectedProjectId(null);
    setView("landing");
  };

  useEffect(() => {
    registerUnauthenticatedHandler(handleLogout);
  }, []);
  
  useEffect(() => {
    const token = getToken();
    if (!token) return;
    apiFetch("/api/me")
      .then((r) => (r.ok ? r.json() : Promise.reject()))
      .then((user) => {
        setCurrentUser(user);
        return apiFetch("/api/projects");
      })
      .then((r) => r.json())
      .then((data) => setProjects(normalizeProjects(data)))
      .catch(() => clearToken());
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      {mutationError && (
        <div className="fixed top-14 inset-x-0 z-50 flex justify-center pointer-events-none">
          <div className="pointer-events-auto flex items-center gap-3 bg-[#1D3335] text-white text-sm px-5 py-2.5 rounded-xl shadow-lg mt-2">
            <span>{mutationError}</span>
            <button
              onClick={() => setMutationError(null)}
              className="text-white/60 hover:text-white cursor-pointer text-base leading-none"
            >
              ✕
            </button>
          </div>
        </div>
      )}
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
      <AppRouter
        view={view}
        setView={setView}
        currentUser={currentUser}
        setCurrentUser={setCurrentUser}
        projects={projects}
        setProjects={setProjects}
        selectedProject={selectedProject}
        selectedProjectId={selectedProjectId}
        setSelectedProjectId={setSelectedProjectId}
        pendingXmlFile={pendingXmlFile}
        setPendingXmlFile={setPendingXmlFile}
        pendingImageFile={pendingImageFile}
        setPendingImageFile={setPendingImageFile}
        meiContent={meiContent}
        handleDownloadManifest={handleDownloadManifest}
        handleDownloadMei={handleDownloadMei}
        handleLoginSuccess={handleLoginSuccess}
        handleLogout={handleLogout}
        mutations={mutations}
        handleEncodeResult={handleEncodeResult}
      />
      <Footer />
    </div>
  );
}
