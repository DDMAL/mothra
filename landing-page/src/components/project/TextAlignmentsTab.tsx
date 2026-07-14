import { Fragment, useState } from "react";
import type { ProjectImage, TextAlignment } from "../../types";
import { AuthImage } from "../shared/AuthImage";
import TextAlignmentViewerModal from "./TextAlignmentViewerModal";
import { sortBySourceThenFolio, sourceGroupLabel } from "../../utils/folio";

interface TextAlignmentsTabProps {
  textAlignments: TextAlignment[];
  images: ProjectImage[];
  projectId: number;
}

export default function TextAlignmentsTab({
  textAlignments,
  images,
  projectId,
}: TextAlignmentsTabProps) {
  const [viewSet, setViewSet] = useState<TextAlignment | null>(null);
  if (textAlignments.length === 0) {
    return (
      <p className="mt-6 text-white/70 text-sm">no text-finding results yet</p>
    );
  }

  // textAlignments arrives oldest-first; label reruns of the same image
  // "name", "name (1)", "name (2)", ... in run order so users can tell
  // multiple text-finding passes on one image apart.
  const seenCounts = new Map<string, number>();
  const labelById = new Map<string, string>();
  for (const set of textAlignments) {
    const baseName = set.imageName.replace(/\.[^.]+$/, "");
    const n = seenCounts.get(baseName) ?? 0;
    seenCounts.set(baseName, n + 1);
    labelById.set(set.id, n === 0 ? baseName : `${baseName} (${n})`);
  }

  const sortedAlignments = sortBySourceThenFolio(
    textAlignments,
    images,
    (a) => a.imageName,
  );

  return (
    <>
      {viewSet && (
        <TextAlignmentViewerModal
          alignment={viewSet}
          projectId={projectId}
          onClose={() => setViewSet(null)}
          label={labelById.get(viewSet.id)}
        />
      )}
      <div className="mt-6 grid grid-cols-5 gap-4">
        {sortedAlignments.map((set, idx) => {
          const group = sourceGroupLabel(images, set.imageName);
          const prevGroup =
            idx > 0
              ? sourceGroupLabel(images, sortedAlignments[idx - 1].imageName)
              : undefined;
          const showHeader = group !== prevGroup;
          return (
            <Fragment key={set.id}>
              {showHeader && (
                <div className="col-span-5 text-white/70 text-xs font-mono uppercase tracking-wide mt-4 first:mt-0 pb-1 border-b border-white/20">
                  {group}
                </div>
              )}
              <div
                className="flex flex-col gap-2 cursor-pointer"
                onClick={() => setViewSet(set)}
              >
                <div className="relative aspect-square">
                  <div className="absolute inset-0 bg-[#C8E6E3]/50 rounded-xl overflow-hidden flex items-end justify-start p-2">
                    {set.imageSrc && (
                      <AuthImage
                        src={set.imageSrc}
                        alt={set.imageName}
                        className="absolute inset-0 w-full h-full object-cover opacity-60"
                      />
                    )}
                    <span className="relative text-[10px] text-white/80 font-mono z-10">
                      .json
                    </span>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setViewSet(set);
                      }}
                      className="absolute top-1.5 right-1.5 z-20 px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer"
                    >
                      view
                    </button>
                  </div>
                </div>
                <span className="text-sm text-white truncate">
                  {labelById.get(set.id)}
                </span>
                <span className="text-xs text-white/50">
                  {set.syllableCount} syllable
                  {set.syllableCount !== 1 ? "s" : ""}
                </span>
              </div>
            </Fragment>
          );
        })}
      </div>
    </>
  );
}
