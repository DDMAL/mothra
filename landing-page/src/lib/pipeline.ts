// The six user-facing pipeline steps, in order. The project page's progress
// list renders these, and anything else that names a step should read its
// label from here rather than spelling it out again.
//
// The displayed numbers (1-6) are NOT `stepsUnlocked` values. That counter
// predates "set up" being shown as a step and counts from detection:
// 0 detect, 1 classify, 2 encode, 3 correct, 4 submit -- which is what
// `unlockIndex` records. It is also the space utils/imageStep.ts's
// `nextStep` is in. Setting up is never locked, so it has no unlock index.
export const PIPELINE_STEPS = [
  { id: "setup", label: "set up", unlockIndex: null },
  { id: "detect", label: "detect", unlockIndex: 0 },
  { id: "classify", label: "classify", unlockIndex: 1 },
  { id: "encode", label: "encode", unlockIndex: 2 },
  { id: "correct", label: "correct", unlockIndex: 3 },
  { id: "submit", label: "submit", unlockIndex: 4 },
] as const;

export type PipelineStepId = (typeof PIPELINE_STEPS)[number]["id"];

// The steps that leave the project page. "setup" and "encode" open a tab on
// the page itself instead: encoding has no screen of its own (it runs as a
// hand-off between classify and correct, so `nextStep` is never 2), and its
// results are the project's MEI files.
export type RoutedStepId = Exclude<PipelineStepId, "setup" | "encode">;
