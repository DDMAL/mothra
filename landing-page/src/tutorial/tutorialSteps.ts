export type TutorialStepType = "info" | "action";

export type IcTourTarget = 
    | {kind: "selector"; value: string }
    | {kind: "glyphClass"; pattern: string};

export interface TutorialStep {
    id: string;
    phase: "intro" | "settings" | "process" | "ic-handoff" | "ic-tour" | "neon-handoff" | "done";
    type: TutorialStepType;
    /** data-tutorial-target value to highlight, or null for an untargeted
     *  (bottom-center) callout. */
    target: string | null;
    title: string;
    body: string;
    /** Only for phase "ic-tour" -- what to highlight inside the IC iframe. */
    icTarget?: IcTourTarget;
    /** Only for phase "ic-tour": the ic:tour-event name (see the IC-side
     *  useTourBridge.ts) that advances past this step. Every ic-tour step
     *  also accepts the generic "next" (plain click-through) and "skip"
     *  (ends the tour) regardless of this field. */
    icAdvanceOn?: string;
    /** Default true. type/action steps are ALWAYS unblocked already (they
     *  need a real click to advance) -- this is for "info" steps whose own
     *  copy instructs interacting with the underlying page (a glyph, a
     *  zoom gesture, a tab) even though nothing gates advancing past them.
     *  Set false on those; leave true (the default) for purely narrative
     *  steps where a scrim is harmless. */
    blocksInteraction?: boolean;
}

// Antiphonal does double duty -- it's both the "process a page" example and
// the IC-classifying example (the doc this tour is built from, "IC
// tutorial.docx", is written specifically against this page's Hufnagel
// notation). A third fixture image (Aarau_MsMurF2_6v.jpg) was originally
// meant to carry the "process a page" role separately, keeping it out of
// the IC step entirely -- dropped in favor of this simpler two-image
// tutorial project (see seed_demo_project.py).
export const TUTORIAL_IMAGE_NAMES = {
    process: "Antiphonal_1v_hfngl.jpg",
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
    body: "This is a real project, pre-loaded with two sample pages, so you can try the pipeline before using your own manuscript. Let's take a quick look around.",
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
    body: 'Switch mode to "manual" so you classify pages yourself in the Interactive Classifier. Then, under training data → presets, check the "hufnagel" preset -- these sample pages are written in Hufnagelschrift. We\'ll wait here until both are done.',
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
    blocksInteraction: false, // instructs switching tabs to view the results
    title: "Here's what detection found",
    body: `Open Generated files -> Detected layers and view ${TUTORIAL_IMAGE_NAMES.process} to see the boxes it found -- the raw material the next step, classification, works from. Click next when you're ready to try classifying.`,
  },
  // ic-handoff's own popup isn't guaranteed to be seen -- reaching it
  // immediately fires focusIc() (AppRouter.tsx's effect), which changes
  // view away from OVERLAY_VIEWS and hides mothra's own overlay the same
  // render, then that same effect calls advance() so stepIndex doesn't
  // linger here once its one job (the handoff) is done. See "Known
  // simplifications". The real, seen coachmarking for the classifier lives
  // in the "ic-tour" steps below, rendered INSIDE the IC iframe itself
  // (see ic/frontend's IcTourOverlay.tsx) -- mothra never renders anything
  // of its own while any of those are current.
  {
    id: "ic-handoff",
    phase: "ic-handoff",
    type: "info",
    target: null,
    title: "Now let's classify",
    body: `Hands off into the Interactive Classifier for ${TUTORIAL_IMAGE_NAMES.ic}.`,
  },
  {
    id: "ic-select-preset",
    phase: "ic-tour",
    type: "action",
    target: null,
    icTarget: { kind: "selector", value: "presets" },
    icAdvanceOn: "preset-selected:hufnagel",
    title: "Choose a training set",
    body: 'This demo folio is written in Hufnagelschrift. In the Training data -> Presets menu, check "Hufnagel training data" -- these are the glyphs the classifier will compare each new glyph against.',
  },
  {
    id: "ic-start-session",
    phase: "ic-tour",
    type: "action",
    target: null,
    icTarget: { kind: "selector", value: "start-session" },
    icAdvanceOn: "session-created",
    title: "Start the session",
    body: "Once your training data is selected, click \"Start session\" to open the page for classifying.",
  },
  {
    id: "ic-zoom-hint",
    phase: "ic-tour",
    type: "info",
    target: null,
    icTarget: { kind: "selector", value: "page-image" },
    blocksInteraction: false, // the whole point is to actually try zooming
    title: "Zoom in on the page",
    body: "While hovering over the manuscript image, hold ctrl/cmd and scroll to zoom in or out. The grey boxes show every glyph the computer has identified.",
  },
  {
    id: "ic-neumes-panel",
    phase: "ic-tour",
    type: "info",
    target: null,
    icTarget: { kind: "selector", value: "neumes-panel" },
    title: "See how glyphs were classified",
    body: "The thumbnails in the Neumes panel show how the computer classified each identified glyph.",
  },
  {
    id: "ic-select-correct-glyph",
    phase: "ic-tour",
    type: "info",
    target: null,
    icTarget: { kind: "glyphClass", pattern: "^neume\\." },
    blocksInteraction: false, // instructs clicking the glyph itself
    title: "A correct example",
    body: 'This glyph is correctly identified! Click on it to select it, then click "Apply & reclassify" (or press Enter) to confirm the classification.',
  },
  {
    id: "ic-apply-1",
    phase: "ic-tour",
    type: "action",
    target: null,
    icTarget: { kind: "selector", value: "apply-button" },
    icAdvanceOn: "reclassify-applied",
    title: "Confirm it",
    body: 'Click "Apply & reclassify" (or press Enter) to confirm the classification.',
  },
  {
    id: "ic-select-incorrect-glyph",
    phase: "ic-tour",
    type: "info",
    target: null,
    icTarget: { kind: "glyphClass", pattern: "podatus3" },
    blocksInteraction: false, // instructs clicking the glyph itself
    title: "An incorrect example",
    body: 'This podatus is incorrectly identified -- the "3" at the end of its classification says the interval between the two notes is a third, when it\'s really a second. Click on it to select it.',
  },
  {
    id: "ic-correct-class-name",
    phase: "ic-tour",
    type: "info",
    target: null,
    icTarget: { kind: "selector", value: "class-name-input" },
    blocksInteraction: false, // instructs typing in the field / clicking a class
    title: "Correct the classification",
    body: 'In the Class name field, replace "neume.podatus3" with "neume.podatus2" -- or click the correct class in the left-hand Classes menu instead. Your classification must match one of the available classes, or the glyph won\'t get encoded properly.',
  },
  {
    id: "ic-apply-2",
    phase: "ic-tour",
    type: "action",
    target: null,
    icTarget: { kind: "selector", value: "apply-button" },
    icAdvanceOn: "reclassify-applied",
    title: "Confirm the correction",
    body: 'Click "Apply & reclassify" (or press Enter) to confirm your correction.',
  },
  {
    id: "ic-training-data",
    phase: "ic-tour",
    type: "action",
    target: null,
    icTarget: { kind: "selector", value: "training-data-handle" },
    icAdvanceOn: "training-panel-opened",
    title: "The training data",
    body: "Click here to see every glyph currently in the training pool -- what each new glyph gets compared against. You can delete glyphs you think aren't helping, but that can't be undone.",
  },
  {
    id: "ic-wrap-up",
    phase: "ic-tour",
    type: "info",
    target: null,
    title: "Nice work classifying!",
    body: "You've corrected a glyph and seen how the training data works. Next, let's see how to correct a page in Neon.",
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