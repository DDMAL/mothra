import { useEffect, useState, useRef, useCallback } from "react";
import Navbar from "./components/layout/Navbar";
import Footer from "./components/layout/Footer";
import AlphaBanner from "./components/layout/AlphaBanner";
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
import { toast, clearToasts } from "./lib/toast";
import type { NeonEditorHandle } from "./components/workflow/NeonBatchEditor";

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

// Views a user can meaningfully return to via the browser Back button.
// Leaving one of these pushes a fresh history entry for wherever the user
// is headed; leaving any other view (i.e. anywhere inside a project's
// processing/IC/encoding/Neon pipeline) instead replaces the current
// entry -- see the view-history effect below. That collapses the whole
// pipeline into a single history slot, so Back from any depth inside it
// (mid-predict, past IC, past encoding, editing in Neon, ...) always lands
// directly back on the anchor view instead of walking back through
// now-irrelevant processing/completion screens one hop at a time
// (issue #272).
const HISTORY_ANCHOR_VIEWS = new Set<View>([
  "landing",
  "login",
  "register",
  "account",
  "docs",
  "projects",
  "project",
]);

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
    startedAt?: string | null;
  } | null>(null);
  const [pendingProjectTab, setPendingProjectTab] =
    useState<ProjectInitialTab | null>(null);
  // Lets the popstate handler below reuse NeonBatchEditor's own
  // unsaved-work confirmation gate for the browser back/forward buttons
  // (issue #266), the same gate its in-app Back/Prev/Next/filmstrip/arrow-key
  // navigation already goes through.
  const neonEditorRef = useRef<NeonEditorHandle>(null);

  const selectedProject =
    projects.find((p) => p.id === selectedProjectId) ?? null;
  const mutations = useProjectMutations(setProjects);
  const { updateProjectSteps } = mutations;

  // Issue #266/#272: gates any action that would navigate away from the
  // Neon editor -- not just its own in-app Back/Prev/Next/filmstrip/
  // arrow-key buttons (which already go through NeonBatchEditor's own
  // attemptNavigation) -- behind the exact same unsaved-work confirmation.
  // Covers everything else that's still clickable while the editor is on
  // screen: the Navbar's nav buttons and logout, and a job-status toast's
  // "view" action. The popstate handler below reuses this too, on top of
  // its own extra history bookkeeping (the browser has already moved its
  // history cursor by the time that event fires, which none of these
  // other triggers ever do).
  const guardNeonExit = useCallback(
    (action: () => void) => {
      if (view === "neon-editor" && neonEditorRef.current?.isUnsaved()) {
        neonEditorRef.current.attemptNavigation(action);
      } else {
        action();
      }
    },
    [view],
  );

  // Browser back/forward should step through Mothra's own view history
  // instead of leaving the app on the first Back press (issue #133). There's
  // no router/URL sync in this app, so `view`/`selectedProjectId` are pushed
  // into `history.state` by hand here and restored on `popstate` - every
  // existing `setView`/`setSelectedProjectId` call site is unaffected since
  // they just call these same state setters.
  const isPoppingRef = useRef(false);
  const hasMountedHistoryRef = useRef(false);
  // The view this effect last saw, i.e. the one being left -- read BEFORE
  // it's overwritten below, so the push-vs-replace decision reflects where
  // the user is navigating FROM, not where they just landed.
  const prevViewRef = useRef<View>(view);

  useEffect(() => {
    window.history.replaceState({ view, selectedProjectId }, "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!hasMountedHistoryRef.current) {
      hasMountedHistoryRef.current = true;
      prevViewRef.current = view;
      return;
    }
    if (isPoppingRef.current) {
      isPoppingRef.current = false;
      prevViewRef.current = view;
      return;
    }
    // Issue #272: only push a new entry when leaving an anchor view: that's
    // what creates the single history slot for "somewhere inside a
    // project's pipeline". Every further hop within the pipeline replaces
    // that same slot instead of stacking a new one on top of it.
    if (HISTORY_ANCHOR_VIEWS.has(prevViewRef.current)) {
      window.history.pushState({ view, selectedProjectId }, "");
    } else {
      window.history.replaceState({ view, selectedProjectId }, "");
    }
    prevViewRef.current = view;
  }, [view, selectedProjectId]);

  useEffect(() => {
    const onPopState = (e: PopStateEvent) => {
      const state = e.state as {
        view?: View;
        selectedProjectId?: number | null;
      } | null;
      const targetView = state?.view ?? "landing";
      const targetProjectId = state?.selectedProjectId ?? null;

      // Issue #266/#272: the browser Back/Forward buttons drive this same
      // view history (see the comment above), so a Back press away from the
      // Neon editor needs the same unsaved-work confirmation guardNeonExit
      // gives every other exit from it.
      if (view === "neon-editor" && neonEditorRef.current?.isUnsaved()) {
        // The browser has already moved its history cursor by the time this
        // event fires -- push the current state straight back on top to
        // undo that move while the confirm modal decides what happens,
        // keeping history in sync with the editor still being on screen.
        window.history.pushState({ view, selectedProjectId }, "");
        guardNeonExit(() => {
          // Deliberately NOT flagged as a pop: if confirmed, this should
          // behave like any other forward navigation (a plain setView,
          // handled by the push-vs-replace effect above like any other),
          // the same as NeonBatchEditor's in-app Back button already does
          // -- not like a true history back() to the entry we just pushed
          // over. "neon-editor" isn't a HISTORY_ANCHOR_VIEWS entry, so this
          // replaces that entry with the target view rather than pushing a
          // second one on top of it.
          setView(targetView);
          setSelectedProjectId(targetProjectId);
        });
        return;
      }

      isPoppingRef.current = true;
      setView(targetView);
      setSelectedProjectId(targetProjectId);
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, [view, selectedProjectId, guardNeonExit]);

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

  // Issue #265: drop any lingering persistent (duration: 0) toast when the
  // user navigates to a different view -- see clearToasts's doc comment for
  // why this is scoped to persistent toasts only, not every toast.
  useEffect(() => {
    clearToasts();
  }, [view]);

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
        onClick: () =>
          guardNeonExit(() => {
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
          }),
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

  const doLogout = () => {
    clearToken();
    setCurrentUser(null);
    setProjects([]);
    setSelectedProjectId(null);
    setView("landing");
  };
  // The user-initiated "logout" button goes through the same unsaved-work
  // confirmation as any other exit from the Neon editor. The server-driven
  // forced logout below (a dead/expired session even a refresh couldn't
  // fix, see apiFetch's registerUnauthenticatedHandler) deliberately calls
  // doLogout directly instead -- the session is already gone server-side by
  // that point, so there's nothing a "stay and keep editing" cancel could
  // actually preserve, only a client UI stuck believing it's still logged in.
  const handleLogout = () => guardNeonExit(doLogout);

  useEffect(() => {
    registerUnauthenticatedHandler(doLogout);
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
      {/* mothra#290: shown on every view, not just "landing" -- the whole
          app is alpha, not just its marketing page. */}
      <AlphaBanner />
      <Navbar
        currentUser={currentUser}
        onLogout={handleLogout}
        onLogin={() => guardNeonExit(() => setView("login"))}
        onGetStarted={() => guardNeonExit(() => setView("register"))}
        onMyProjects={() => guardNeonExit(() => setView("projects"))}
        onDocs={() => guardNeonExit(() => setView("docs"))}
        onHome={() => guardNeonExit(() => setView("landing"))}
        onAccount={() => guardNeonExit(() => setView("account"))}
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
        neonEditorRef={neonEditorRef}
        handleEncodeBatchResult={handleEncodeBatchResult}
      />
      <Footer />
    </div>
  );
}
