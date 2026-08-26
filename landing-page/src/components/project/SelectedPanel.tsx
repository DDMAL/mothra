import type { AnnotationSet, MeiFile, Project, ProjectImage } from "../../types";
import { getImageProgress } from "../../utils/imageStep";
import TruncatedName from "../shared/TruncatedName";

// mothra#294: an image's pipeline stage is never separately "selected" --
// it's derived from getImageProgress (the same function that already draws
// each thumbnail's badge), and used here to bucket the selected images into
// the three sections the project page shows: "begin" (progress === null),
// "ic" (has an annotation, no MEI yet), "neon" (has an MEI, corrected or
// not -- both are worked on inside the same Neon editor).
type Bucket = "begin" | "ic" | "neon";

function bucketOf(
  image: Pick<ProjectImage, "id" | "name">,
  annotations: AnnotationSet[],
  meiFiles: MeiFile[],
  stepsUnlocked: number,
): Bucket {
  const p = getImageProgress(image, annotations, meiFiles, stepsUnlocked);
  if (!p) return "begin";
  return p.nextStep === 1 ? "ic" : "neon";
}

const SECTION_LABELS: Record<Bucket, string> = {
  begin: "begin",
  ic: "interactive classifier",
  neon: "neon",
};

interface SelectedPanelProps {
  project: Project;
  usedNames: { images: string[]; models: string[] };
  stepsUnlocked: number;
  onUsedNamesChange: (names: { images: string[]; models: string[] }) => void;
  // Returns true if it navigated (IC or Neon) -- a Begin-stage image has
  // nothing to jump into, so this is always false there.
  onNavigateImage: (image: Pick<ProjectImage, "id" | "name">) => boolean;
}

export default function SelectedPanel({
  project,
  usedNames,
  stepsUnlocked,
  onUsedNamesChange,
  onNavigateImage,
}: SelectedPanelProps) {
  const removeImage = (imageId: string) =>
    onUsedNamesChange({
      ...usedNames,
      images: usedNames.images.filter((id) => id !== imageId),
    });

  const buckets: Record<Bucket, string[]> = { begin: [], ic: [], neon: [] };
  for (const imageId of usedNames.images) {
    const img = project.images.find((i) => i.id === imageId);
    buckets[
      bucketOf(
        img ?? { id: imageId, name: imageId },
        project.annotations ?? [],
        project.meiFiles ?? [],
        stepsUnlocked,
      )
    ].push(imageId);
  }

  const renderImageRow = (imageId: string, bucket: Bucket) => {
    const img = project.images.find((i) => i.id === imageId);
    const displayName = img?.name ?? imageId;
    const clickable = bucket !== "begin";
    return (
      <div key={imageId} className="flex items-center justify-between">
        {clickable ? (
          <button
            onClick={() => onNavigateImage(img ?? { id: imageId, name: displayName })}
            className="flex-1 min-w-0 mr-2 text-left hover:underline cursor-pointer"
          >
            <TruncatedName name={displayName} className="flex-1 min-w-0" />
          </button>
        ) : (
          <TruncatedName name={displayName} className="flex-1 min-w-0 mr-2" />
        )}
        {/* mothra#247: always removable, regardless of bucket -- this only
            excludes the page from future predict/IC/batch runs
            (usedImageIds), it never deletes its existing annotations/MEI
            files. Re-"use"-ing it later (via ImageTab.tsx) picks up right
            where it left off, landing back in the correct section. */}
        <button
          onClick={() => removeImage(imageId)}
          title={
            clickable
              ? "Remove from selection (its existing annotations/MEI are kept, just excluded from future runs)"
              : "Remove from selection"
          }
          className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
        >
          ×
        </button>
      </div>
    );
  };

  return (
    <div className="bg-[#C8E6E3]/40 rounded-2xl p-4 flex flex-col gap-2 text-white text-sm">
      <span className="text-white/80">selected:</span>
      {usedNames.models.map((name) => (
        <div key={name} className="flex items-center justify-between">
          <TruncatedName name={name} className="flex-1 min-w-0 mr-2" />
          {stepsUnlocked === 0 && (
            <button
              onClick={() =>
                onUsedNamesChange({
                  ...usedNames,
                  models: usedNames.models.filter((n) => n !== name),
                })
              }
              className="text-white/60 hover:text-white flex-shrink-0 leading-none cursor-pointer"
            >
              ×
            </button>
          )}
        </div>
      ))}
      {(["begin", "ic", "neon"] as const).map((bucket) =>
        buckets[bucket].length > 0 ? (
          <div key={bucket} className="flex flex-col gap-2">
            <hr className="border-white/40 my-1" />
            <span className="text-white/60 text-xs uppercase tracking-wide">
              {SECTION_LABELS[bucket]}
            </span>
            {buckets[bucket].map((imageId) => renderImageRow(imageId, bucket))}
          </div>
        ) : null,
      )}
    </div>
  );
}
