import type { AnnotationSet, MeiFile, ProjectImage } from "../types";

export interface ImageProgress {
  nextStep: number; // 1=ic, 3=neon, 4=send
  badge: string; // label shown on the greyed-out card
}

export function getImageProgress(
  imageName: string,
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): ImageProgress | null {
  if (meiFiles.some((f) => f.imageName === imageName && f.corrected))
    return { nextStep: 4, badge: "done" };
  if (meiFiles.some((f) => f.imageName === imageName))
    return { nextStep: 3, badge: "neon" };
  // Gated on stepsUnlocked, not just row presence: a batch text-finding job
  // commits per-image annotation rows during its YOLO "checking" stage
  // before the later text-service stage that can still fail the job overall
  // (tasks_text_batch.py) - stepsUnlocked only advances on a job's confirmed
  // "done" event (AppRouter.tsx's onComplete), so a failed batch run leaves
  // annotation rows behind without unlocking step 1. Without this gate those
  // leftover rows made the UI believe step 1 had completed.
  if (stepsUnlocked >= 1 && annotations.some((a) => a.imageName === imageName))
    return { nextStep: 1, badge: "ic" };
  return null;
}

// The used images the interactive classifier still has work for: never
// predicted (progress null) or predicted but not yet encoded (nextStep 1).
// An image that already has an MEI file is past step 1 and drops out - which
// is why this can legitimately come back empty once every page is encoded
// (InteractiveClassifier renders an explanatory empty state for that).
export function pendingIcImages(
  images: ProjectImage[],
  usedImageNames: string[],
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): ProjectImage[] {
  const progressOf = (name: string) =>
    getImageProgress(name, annotations, meiFiles, stepsUnlocked);
  return images
    .filter((img) => {
      if (!usedImageNames.includes(img.name)) return false;
      const p = progressOf(img.name);
      return p === null || p.nextStep <= 1;
    })
    .sort(
      (a, b) =>
        (progressOf(a.name)?.nextStep ?? 0) -
        (progressOf(b.name)?.nextStep ?? 0),
    );
}

export function minNextStep(
  imageNames: string[],
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): number {
  if (imageNames.length === 0) return 0;
  return imageNames.reduce((min, name) => {
    const p = getImageProgress(name, annotations, meiFiles, stepsUnlocked);
    return Math.min(min, p?.nextStep ?? 0);
  }, Infinity);
}
