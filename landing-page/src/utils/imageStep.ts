import type { AnnotationSet, MeiFile } from "../types";

export interface ImageProgress {
  nextStep: number; // 1=ic, 3=neon, 4=send
  badge: string;    // label shown on the greyed-out card
}

export function getImageProgress(
  imageName: string,
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
): ImageProgress | null {
  if (meiFiles.some((f) => f.imageName === imageName && f.corrected))
    return { nextStep: 4, badge: "done" };
  if (meiFiles.some((f) => f.imageName === imageName))
    return { nextStep: 3, badge: "neon" };
  if (annotations.some((a) => a.imageName === imageName))
    return { nextStep: 1, badge: "ic" };
  return null;
}

export function minNextStep(
  imageNames: string[],
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
): number {
  if (imageNames.length === 0) return 0;
  return imageNames.reduce((min, name) => {
    const p = getImageProgress(name, annotations, meiFiles);
    return Math.min(min, p?.nextStep ?? 0);
  }, Infinity);
}
