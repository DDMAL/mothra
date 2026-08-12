import { useEffect, useState, useRef } from "react";
import Navbar from "./components/layout/Navbar";
import Footer from "./components/layout/Footer";
import AppRouter from "./components/AppRouter";
import ToastContainer from "./components/shared/ToastContainer";
import type { View, Project, ProjectInitialTab } from "./types";
import type { CurrentUser } from "./hooks/useAuth";
import { getToken, setToken, clearToken } from "./hooks/useAuth";
import { normalizeProjects } from "./utils/projects";
import { useProjectMutations } from "./hooks/useProjectMutations";
import { useEncodingFlow } from "./hooks/useEncodingFlow";
import { useScrollFade } from "./hooks/useScrollFade";
import { apiFetch, registerUnauthenticatedHandler } from "./lib/apiFetch";
import { useActiveJobWatcher } from "./hooks/useActiveJobWatcher";
import { toast } from "./lib/toast";

// Where a job-done toast's "view" button should actually land (issue #196):
// - succeeded: the tab holding what the job produced.
// - failed/cancelled, for a kind resumeJob knows how to reattach to
//   (see AppRouter.tsx's case "processing"): reopen ProcessingPage on the
//   job's own stream, which replays its full history including the error.
// encode_upload/encode_batch failures have no resume path yet (same
// deliberate limit as issue #195's ProjectDetail "view progress" -- see
// AppRouter.tsx) and fall through to just landing on the project page.
const RESUMABLE_JOB_KINDS = new Set(["predict", "text_batch"]);
const SUCCESS_TAB_BY_JOB_KIND: Record<string, ProjectInitialTab> = {
  predict: { tab: "generated", subTab: "annotations" },
  text_batch: { tab: "generated", subTab: "text" },
  encode_upload: { tab: "generated", subTab: "mei files" },
  encode_batch: { tab: "generated", subTab: "mei files" },
};

// The stepsUnlocked floor a succeeded job of this kind guarantees -- mirrors
// what AppRouter.tsx's ProcessingPage onComplete handlers already bump to
// when the user is actively watching (Math.max(current, N)). Needed because
// getImageProgress (utils/imageStep.ts) gates its "ic" step behind
// `stepsUnlocked >= 1`, so if nobody's ProcessingPage was mounted to run
// that onComplete -- e.g. the user stayed on the project page the whole
// time a job ran instead of watching it finish -- stepsUnlocked (and the
// project's own annotations/meiFiles) never advance, and the "begin" button
// stays stuck forever even though the job actually succeeded server-side.
const STEPS_UNLOCKED_BY_JOB_KIND: Record<string, number> = {
  predict: 1,
  text_batch: 1,
  encode_upload: 3,
  encode_batch: 3,
};

export default function App() {
  const [view, setView] = useState<View>("landing");
  const [currentUser, setCurrentUser] = useState<CurrentUser | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [selectedProjectId, setSelectedProjectId] = useState<number | null>(
    null,
  );
  const [resumeJob, setResumeJob] = useState<{
    jobId: string;
    kind: string;
  } | null>(null);
  const [pendingProjectTab, setPendingProjectTab] =
    useState<ProjectInitialTab | null>(null);

  const selectedProject =
    projects.find((p) => p.id === selectedProjectId) ?? null;
  const mutations = useProjectMutations(setProjects);
  const { updateProjectSteps } = mutations;

  // Browser back/forward should step through Mothra's own view history
  // instead of leaving the app on the first Back press (issue #133). There's
  // no router/URL sync in this app, so `view`/`selectedProjectId` are pushed
  // into `history.state` by hand here and restored on `popstate` - every
  // existing `setView`/`setSelectedProjectId` call site is unaffected since
  // they just call these same state setters.
  const isPoppingRef = useRef(false);
  const hasMountedHistoryRef = useRef(false);

  useEffect(() => {
    window.history.replaceState({ view, selectedProjectId }, "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!hasMountedHistoryRef.current) {
      hasMountedHistoryRef.current = true;
      return;
    }
    if (isPoppingRef.current) {
      isPoppingRef.current = false;
      return;
    }
    window.history.pushState({ view, selectedProjectId }, "");
  }, [view, selectedProjectId]);

  useEffect(() => {
    const onPopState = (e: PopStateEvent) => {
      const state = e.state as {
        view?: View;
        selectedProjectId?: number | null;
      } | null;
      isPoppingRef.current = true;
      setView(state?.view ?? "landing");
      setSelectedProjectId(state?.selectedProjectId ?? null);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const {
    pendingXmlFile,
    setPendingXmlFile,
    pendingImageFile,
    setPendingImageFile,
    meiContent,
    handleDownloadManifest,
    handleDownloadMei,
    handleEncodeResult,
    pendingBatchPairs,
    setPendingBatchPairs,
    handleEncodeBatchResult,
  } = useEncodingFlow(selectedProjectId, setProjects);

  useScrollFade(view);

  useActiveJobWatcher((job, status) => {
    if (status === "succeeded" && job.projectId != null) {
      const projectId = job.projectId;
      // Refresh this project's data + bump stepsUnlocked regardless of
      // whether anyone's ProcessingPage is actually mounted watching this
      // job right now -- this is the only path that does so for a job that
      // finishes while the user just sat on the project page instead of
      // watching it (AppRouter.tsx's onResult/onComplete only run for a
      // mounted ProcessingPage, and there's real overlap when one *is*
      // mounted, but both end up idempotent).
      //
      // The stepsUnlocked bump is chained AFTER the refetch resolves and
      // floored against the server's own just-fetched value (normalized.
      // stepsUnlocked), not fired concurrently against a possibly-stale
      // `projects` closure -- firing both at once let an in-flight GET that
      // started before the PUT resolve AFTER it and overwrite the
      // just-bumped local stepsUnlocked back down to its pre-bump value.
      apiFetch(`/api/projects/${projectId}`)
        .then((r) => (r.ok ? r.json() : null))
        .then((fresh: Project | null) => {
          if (!fresh) return;
          const [normalized] = normalizeProjects([fresh]);
          setProjects((prev) =>
            prev.map((p) => (p.id === normalized.id ? normalized : p)),
          );
          const minSteps = STEPS_UNLOCKED_BY_JOB_KIND[job.kind];
          if (minSteps != null) {
            updateProjectSteps(
              projectId,
              Math.max(normalized.stepsUnlocked, minSteps),
            );
          }
        })
        .catch(() => {});
    }
    toast[
      status === "succeeded"
        ? "success"
        : status === "failed"
          ? "error"
          : "info"
    ](`${job.kind} job ${status}`, {
      duration: 0,
      action: {
        label: "view",
        onClick: () => {
          if (job.projectId) setSelectedProjectId(job.projectId);
          if (status === "succeeded") {
            setPendingProjectTab(SUCCESS_TAB_BY_JOB_KIND[job.kind] ?? null);
            setView("project");
          } else if (
            (status === "failed" || status === "cancelled") &&
            RESUMABLE_JOB_KINDS.has(job.kind)
          ) {
            setResumeJob({ jobId: job.jobId, kind: job.kind });
            setView("processing");
          } else {
            setView("project");
          }
        },
      },
    });
  });

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
        setView((v) => (v === "landing" ? "projects" : v));
        return apiFetch("/api/projects");
      })
      .then((r) => r.json())
      .then((data) => setProjects(normalizeProjects(data)))
      .catch(() => clearToken());
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <ToastContainer />
      <Navbar
        currentUser={currentUser}
        onLogout={handleLogout}
        onLogin={() => setView("login")}
        onGetStarted={() => setView("register")}
        onMyProjects={() => setView("projects")}
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
        pendingBatchPairs={pendingBatchPairs}
        setPendingBatchPairs={setPendingBatchPairs}
        resumeJob={resumeJob}
        setResumeJob={setResumeJob}
        pendingProjectTab={pendingProjectTab}
        setPendingProjectTab={setPendingProjectTab}
        handleEncodeBatchResult={handleEncodeBatchResult}
      />
      <Footer />
    </div>
  );
}
