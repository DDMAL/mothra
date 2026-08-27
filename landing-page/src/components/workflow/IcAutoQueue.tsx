import { useEffect, useRef, useState } from "react";
import type { ProjectImage } from "../../types";
import { autoQueueImage } from "../../utils/icQueue";
import type { EncodePair } from "../../utils/icQueue";
import IcSessionPicker from "../project/IcSessionPicker";
import type { IcResumeRequest } from "../project/IcSessionsModal";

interface IcAutoQueueProps {
  /** Only the pages step 1 still has work for — see pendingIcImages(). */
  images: ProjectImage[];
  /** Total pages selected on the project, pending or not — lets the empty
   *  state tell "nothing selected yet" apart from "all already encoded". */
  usedImageCount: number;
  projectId: number | null;
  /** *Every* image in the project — what IcSessionPicker resolves the listed
   *  sessions against, since a session worth reopening usually belongs to a
   *  page `images` (pending only) has already filtered out. */
  allImages: ProjectImage[];
  trainingPresets: string[];
  trainingFiles: File[];
  onDone: (pairs: EncodePair[]) => void;
  onBack: () => void;
  /** Reopen the sessions picked here. Handled by the host (AppRouter), which
   *  routes them into the manual classifier: reopening is an explicit choice
   *  about specific pages, so it lands in the classifier in either IC mode. */
  onResumeIcSessions: (requests: IcResumeRequest[]) => void;
  /** Escape hatch to the interactive classifier — offered when the automatic
   *  pass can't run (no training set) or fails partway. */
  onOpenManualClassifier: () => void;
}

/**
 * The "auto" IC mode: classify and queue every pending page with the training
 * set picked on the project page (Classifier settings), then hand the queue
 * to encoding. This is the old "queue all available" button from the IC
 * page's filmstrip, run without ever showing the classifier interface.
 *
 * Pages are classified sequentially so the IC service isn't hammered and
 * progress stays legible.
 */
export default function IcAutoQueue({
  images,
  usedImageCount,
  projectId,
  allImages,
  trainingPresets,
  trainingFiles,
  onDone,
  onBack,
  onOpenManualClassifier,
  onResumeIcSessions,
}: IcAutoQueueProps) {
  const [done, setDone] = useState(0);
  const [sessionsModal, setSessionsModal] = useState(false);
  const [current, setCurrent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Bumped by "try again" to re-run the effect below.
  const [attempt, setAttempt] = useState(0);
  const hasTrainingSet = trainingPresets.length + trainingFiles.length > 0;
  const canRun = projectId != null && images.length > 0 && hasTrainingSet;

  // Guards the run against StrictMode's double-invoked effect (dev): the
  // second invocation is skipped rather than firing a duplicate pass. Note
  // this deliberately isn't a cleanup-driven cancel flag — StrictMode's
  // simulated unmount would trip that and leave the run cancelled but never
  // restarted. Real abandonment is handled by `abortedRef` on the back button.
  const startedRef = useRef(-1);
  const abortedRef = useRef(false);

  useEffect(() => {
    if (!canRun || startedRef.current === attempt) return;
    startedRef.current = attempt;
    (async () => {
      const pairs: EncodePair[] = [];
      try {
        for (let i = 0; i < images.length; i++) {
          if (abortedRef.current) return;
          setCurrent(images[i].name);
          pairs.push(
            await autoQueueImage(
              projectId,
              images[i],
              trainingPresets,
              trainingFiles,
            ),
          );
          setDone(i + 1);
        }
        if (!abortedRef.current) onDone(pairs);
      } catch (err) {
        if (!abortedRef.current)
          setError(String((err as Error).message ?? err));
      } finally {
        setCurrent(null);
      }
    })();
    // Re-runs only on an explicit retry; the deps this closure reads are all
    // fixed for the lifetime of this view (it's mounted per IC-step entry).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [attempt, canRun]);

  const handleBack = () => {
    abortedRef.current = true;
    onBack();
  };

  const pct = images.length > 0 ? Math.round((done / images.length) * 100) : 0;

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48">
      {sessionsModal && projectId != null && (
        <IcSessionPicker
          projectId={projectId}
          images={allImages}
          onClose={() => setSessionsModal(false)}
          onOpen={(requests) => {
            setSessionsModal(false);
            // Same abandonment as "back to project": we're navigating away,
            // so an in-flight automatic pass must not report itself done
            // (and its partial queue is dropped) once we're gone.
            abortedRef.current = true;
            onResumeIcSessions(requests);
          }}
        />
      )}
      <div className="flex flex-col items-center gap-6 w-full max-w-xl">
        <h1 className="text-4xl font-bold italic text-white text-center">
          interactive classifier
        </h1>

        {images.length === 0 ? (
          <>
            <p className="text-[#1D3335] text-center">
              {usedImageCount === 0
                ? "no pages are selected for this project yet"
                : `every selected page (${usedImageCount}) has already been classified and encoded`}
            </p>
            <p className="text-white/60 text-xs text-center max-w-md">
              {usedImageCount === 0
                ? "select images on the project page and run detection first."
                : "there's nothing left to classify — carry on with correction (step 3), or select more pages on the project page."}
            </p>
          </>
        ) : !hasTrainingSet ? (
          <>
            <p className="text-[#1D3335] text-center">
              automatic classification needs a training set
            </p>
            <p className="text-white/70 text-sm text-center max-w-md">
              pick a preset or upload GameraXML under "Classifier settings" on
              the project page, or classify these {images.length} page
              {images.length === 1 ? "" : "s"} yourself.
            </p>
          </>
        ) : error ? (
          <>
            <p className="text-[#1D3335] text-center">
              couldn't classify every page automatically
            </p>
            <p className="text-white/60 text-xs font-mono max-w-lg break-words text-center">
              {error}
            </p>
            <p className="text-white/70 text-sm text-center">
              {done} of {images.length} page{images.length === 1 ? "" : "s"}{" "}
              finished before this — retrying starts the pass over.
            </p>
          </>
        ) : (
          <>
            <p className="text-[#1D3335] text-center">
              classifying and queuing every page with your training set
            </p>
            <div className="w-full h-2 bg-[#1D3335]/30 rounded-full overflow-hidden">
              <div
                className="h-full bg-white transition-all duration-300"
                style={{ width: `${pct}%` }}
              />
            </div>
            <p className="text-white/80 text-sm font-mono">
              {done} / {images.length}
              {current ? ` — ${current}` : ""}
            </p>
            <p className="text-white/50 text-xs text-center">
              this runs the classifier for you — encoding starts automatically
              when it's done.
            </p>
          </>
        )}

        <div className="flex items-center gap-4 flex-wrap justify-center">
          {error && (
            <button
              onClick={() => {
                setError(null);
                setDone(0);
                setAttempt((n) => n + 1);
              }}
              className="px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold"
            >
              try again
            </button>
          )}
          {images.length > 0 && (error || !hasTrainingSet) && (
            <button
              onClick={() => {
                abortedRef.current = true;
                onOpenManualClassifier();
              }}
              className="px-6 py-2 bg-[#1D3335] text-white border border-white/30 rounded-xl hover:opacity-90 cursor-pointer font-semibold"
            >
              classify manually
            </button>
          )}
          {/* Reopening a saved session isn't a manual-mode feature: the
              pages worth reopening are usually already encoded (so no
              automatic pass will ever visit them again), and this is the only
              entry point the IC step has in auto mode. Same modal and same
              destination as the manual classifier's own "saved sessions"
              button. */}
          {projectId != null && (
            <button
              onClick={() => setSessionsModal(true)}
              className="px-6 py-2 bg-[#1D3335] text-white border border-white/30 rounded-xl hover:opacity-90 cursor-pointer font-semibold"
            >
              {images.length === 0
                ? "reopen a saved session"
                : "saved sessions"}
            </button>
          )}
          <button
            onClick={handleBack}
            className="px-6 py-2 border-2 border-white text-white rounded-xl hover:bg-white/10 cursor-pointer font-semibold"
          >
            back to project
          </button>
        </div>
      </div>
    </div>
  );
}
