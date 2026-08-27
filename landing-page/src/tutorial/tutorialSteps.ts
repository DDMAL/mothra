export type TutorialStepType = "info" | "action";

export interface TutorialStep {
    id: string;
    phase: "intro" | "settings" | "process" | "ic-handoff" | "neon-handoff" | "done";
    type: TutorialStepType;
    /** data-tutorial-target value to highlight, or null for an untargeted
     *  (bottom-center) callout. */
    target: string | null;
    title: string;
    body: string;
}

export const TUTORIAL_IMAGE_NAMES = {
    process: "Aarau_MsMurF2_6v.jpg",
    ic: "Antiphonal_1v_hfngl.jpg",
    neon: "CDN-Hsmu_M2149.L4_097r_demo.png",
} as const;

export const TUTORIAL_STEPS: TutorialStep[] = [
  {
    id: "welcome",
    phase: "intro",
    type: "info",
    target: "project-header",
    title: "Welcome to your tutorial project",
    body: "This is a real project, pre-loaded with three sample pages, so you can try the pipeline before using your own manuscript. Let's take a quick look around.",
  },
  {
    id: "progress-sidebar",
    phase: "intro",
    type: "info",
    target: "progress-sidebar",
    title: "Your progress",
    body: "Every project moves through these steps: annotate, classify, encode, correct, and export. In this project every step is unlocked, so you're free to jump around and explore.",
  },
  {
    id: "tabs",
    phase: "intro",
    type: "info",
    target: "tab-bar",
    title: "Images, models, and generated files",
    body: "Your uploaded pages live under Images. Once you produce them, detected layers, classifier XML, and MEI files show up under Generated files.",
  },
  // Placed before "process-prompt" deliberately, not just after the intro:
  // icSettings.mode also decides what happens right after predict finishes
  // (AppRouter.tsx's ProcessingPage onComplete goes straight to "ic-auto"
  // in auto mode, skipping "completion" entirely). Getting mode to "manual"
  // here first means that branch is a defensive fallback by the time
  // process-prompt runs, not the expected path -- see useTutorialFlow.ts.
  {
    id: "ic-settings-prompt",
    phase: "settings",
    type: "action",
    target: "ic-settings",
    title: "Set up the classifier",
    body: 'Switch mode to "manual" so you classify pages yourself in the Interactive Classifier, and set notation to "square" to match these sample pages. We\'ll wait here until both are set.',
  },
  {
    id: "process-prompt",
    phase: "process",
    type: "action",
    target: "continue-button",
    title: "Try it: process a page",
    body: `Click "begin" to run automatic detection on the sample pages. Watch what happens to ${TUTORIAL_IMAGE_NAMES.process} -- we'll look at its results together next.`,
  },
  {
    id: "process-results",
    phase: "process",
    type: "info",
    target: null,
    title: "Here's what detection found",
    body: `Open Generated files -> Detected layers and view ${TUTORIAL_IMAGE_NAMES.process} to see the boxes it found -- the raw material the next step, classification, works from. Click next when you're ready to try classifying.`,
  },
  // -- Placeholders. IC and Neon are separate embedded services (iframes) --
  // there is no mothra-controlled DOM to coachmark once the user is inside
  // them, so these steps' only real job today is to trigger the handoff
  // (see AppRouter.tsx's effect below); their title/body are not guaranteed
  // to be seen on screen yet. See "Known simplifications".
  {
    id: "ic-handoff",
    phase: "ic-handoff",
    type: "info",
    target: null,
    title: "Now let's classify",
    body: `TODO: real intro copy. Hands off into the Interactive Classifier for ${TUTORIAL_IMAGE_NAMES.ic}.`,
  },
  {
    id: "neon-handoff",
    phase: "neon-handoff",
    type: "info",
    target: null,
    title: "Now let's correct in Neon",
    body: `TODO: real intro copy. Hands off into the Neon editor for ${TUTORIAL_IMAGE_NAMES.neon}.`,
  },
  {
    id: "done",
    phase: "done",
    type: "info",
    target: null,
    title: "That's the tour",
    body: "TODO: wrap-up copy + a way back to the user's real projects.",
  },
];