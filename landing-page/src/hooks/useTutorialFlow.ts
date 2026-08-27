import { useEffect, useState } from "react";
import type { Project, View } from "../types";
import type { useIcSettings } from "./useIcSettings";
import { TUTORIAL_STEPS, type TutorialStep } from "../tutorial/tutorialSteps";

const dismissedKey = (projectId: number) => `mothra-tutorial-dismissed-${projectId}`;

// Views the overlay is allowed to render on. Deliberately excludes "ic" /
// "ic-auto" / "neon-editor" -- those are embedded services, not mothra's
// own DOM, so a coachmark there would either float over nothing or clash
// with content mothra doesn't control.
const OVERLAY_VIEWS: View[] = ["project", "completion"];

// IC's built-in training presets are GameraXML filenames, not stable ids
// (e.g. "Square notation training data 05.08.26.xml", see ic/api's
// presets_dir()) -- the date suffix can change on a retrain, so match on
// "square" appearing in the name rather than an exact filename.
const SQUARE_PRESET = /square/i;

export function useTutorialFlow(
    project: Project | null,
    view: View,
    icSettings: ReturnType<typeof useIcSettings>,
) {
    const [stepIndex, setStepIndex] = useState(0);
    const [dismissed, setDismissed] = useState(false);
    const [awaitingProcess, setAwaitingProcess] = useState(false);

    useEffect(() => {
        setStepIndex(0);
        setAwaitingProcess(false);
        if (project) {
            setDismissed(localStorage.getItem(dismissedKey(project.id)) === "1");
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [project?.id]);

    const step: TutorialStep | null =
        project?.isTutorial && !dismissed ? TUTORIAL_STEPS[stepIndex] : null;
    const active = !!step && OVERLAY_VIEWS.includes(view);

    function advance() {
        setStepIndex((i) => Math.min(i + 1, TUTORIAL_STEPS.length - 1));
    }

    function dismiss() {
        setDismissed(true);
        if (project) localStorage.setItem(dismissedKey(project.id), "1");
    }

    // Lets a "restart tutorial" control (see canStart below) bring the tour
    // back from the top -- same effect as a first-ever visit, just
    // triggered explicitly instead of by the dismissed-flag default.
    function start() {
        setStepIndex(0);
        setAwaitingProcess(false);
        setDismissed(false);
        if (project) localStorage.removeItem(dismissedKey(project.id));
    }

    // Gates "ic-settings-prompt" -- waits until the user has actually
    // switched mode to "manual" and checked the "square" training preset
    // (not just visited the settings panel) before advancing to
    // "process-prompt". trainingPresets starts empty (no default to be
    // trivially satisfied by, unlike notationType), so this genuinely
    // requires the user to have clicked the checkbox themselves.
    useEffect(() => {
        if (
            step?.id === "ic-settings-prompt" &&
            icSettings.mode === "manual" &&
            icSettings.trainingPresets.some((name) => SQUARE_PRESET.test(name))
        ) {
            advance();
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [step?.id, icSettings.mode, icSettings.trainingPresets]);

    // step?.id is a real dependency here, not just view: stepIndex reaches
    // "process-prompt" while still ON view "project" (advancing past
    // "ic-settings-prompt" doesn't change view at all), so an effect keyed
    // on [view] alone would never re-run at that moment and awaitingProcess
    // would never flip true -- then when view later changed to
    // "completion", the guard below would exit immediately and advance()
    // would never fire, leaving the step stuck on "process-prompt" forever.
    useEffect(() => {
        if (step?.id === "process-prompt" && view === "project") {
            setAwaitingProcess(true);
            return;
        }
        if (!awaitingProcess) return;

        if (view === "completion") {
            setAwaitingProcess(false);
            advance();
        } else if (view === "ic" || view === "ic-auto") {
            // Defensive fallback only: "ic-settings-prompt" (above) already
            // forces mode to "manual" before process-prompt ever runs, so
            // predict's own onComplete (AppRouter.tsx) should always land on
            // "completion", not jump straight to "ic-auto". Kept in case the
            // user flips mode back to "auto" after the prompt.
            setAwaitingProcess(false);
      setStepIndex((i) => Math.min(i + 2, TUTORIAL_STEPS.length - 1));
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [view, step?.id]);

    // Only the tutorial project ever shows a start/restart control, and only
    // once nothing is currently on screen for it -- avoids a redundant
    // button while the overlay itself is already active.
    const canStart = !!project?.isTutorial && !active;

    return { step, active, advance, dismiss, start, canStart };
}
