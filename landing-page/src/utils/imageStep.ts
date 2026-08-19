import type { AnnotationSet, MeiFile, ProjectImage } from "../types";

export interface ImageProgress {
  nextStep: number; // 1=ic, 3=neon, 4=send
  badge: string; // label shown on the greyed-out card
}

// mothra#241: prefer matching by id — it survives same-named duplicate
// uploads (a re-uploaded image now always gets its own project_images.id,
// see images_api.py) instead of matching every image that happens to share
// this name. Only fall back to the name when the child row predates the id
// being recorded (refId is null/undefined) — the same id-first/name-fallback
// rule AppRouter.tsx's IC-session resume already uses.
function matchesImage(
  image: Pick<ProjectImage, "id" | "name">,
  refId: string | null | undefined,
  refName: string | null | undefined,
): boolean {
  if (refId) return image.id === refId;
  if (refName) return image.name === refName;
  return false;
}

export function getImageProgress(
  image: Pick<ProjectImage, "id" | "name">,
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): ImageProgress | null {
  if (meiFiles.some((f) => matchesImage(image, f.imageId, f.imageName) && f.corrected))
    return { nextStep: 4, badge: "done" };
  if (meiFiles.some((f) => matchesImage(image, f.imageId, f.imageName)))
    return { nextStep: 3, badge: "neon" };
  // Gated on stepsUnlocked, not just row presence: a batch text-finding job
  // commits per-image annotation rows during its YOLO "checking" stage
  // before the later text-service stage that can still fail the job overall
  // (tasks_text_batch.py) - stepsUnlocked only advances on a job's confirmed
  // "done" event (AppRouter.tsx's onComplete), so a failed batch run leaves
  // annotation rows behind without unlocking step 1. Without this gate those
  // leftover rows made the UI believe step 1 had completed.
  if (stepsUnlocked >= 1 && annotations.some((a) => matchesImage(image, a.imageId, a.imageName)))
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
  // mothra#241 follow-up (CodeRabbit): id-keyed, not name-keyed -- lets two
  // duplicate-named "used" images be queued/skipped independently instead
  // of a name match pulling in every same-named row as one unit.
  usedImageIds: string[],
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): ProjectImage[] {
  const progressOf = (img: ProjectImage) =>
    getImageProgress(img, annotations, meiFiles, stepsUnlocked);
  return images
    .filter((img) => {
      if (!usedImageIds.includes(img.id)) return false;
      const p = progressOf(img);
      return p === null || p.nextStep <= 1;
    })
    .sort(
      (a, b) =>
        (progressOf(a)?.nextStep ?? 0) - (progressOf(b)?.nextStep ?? 0),
    );
}

export function minNextStep(
  images: Pick<ProjectImage, "id" | "name">[],
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): number {
  if (images.length === 0) return 0;
  return images.reduce((min, img) => {
    const p = getImageProgress(img, annotations, meiFiles, stepsUnlocked);
    return Math.min(min, p?.nextStep ?? 0);
  }, Infinity);
}
