import { Fragment, useState } from "react";
import type { AnnotationSet, ProjectImage } from "../../types";
import { AuthImage } from "../shared/AuthImage";
import AnnotationViewerModal from "./AnnotationViewerModal";
import TruncatedName from "../shared/TruncatedName";
import type { useAssetSection } from "../../hooks/useAssetSection";
import { sortBySourceThenFolio, sourceGroupLabel } from "../../utils/folio";

// mothra#241 follow-up (CodeRabbit): despite the name, `images` holds
// project_images.id values, not names, since duplicate-named uploads need
// to be selected/removed independently -- see Project.usedImageIds's
// comment in types.ts. `models`/`annotations` are unaffected (still names).
type UsedNames = { images: string[]; models: string[]; annotations: string[] };

interface AnnotationsTabProps {
  annotations: AnnotationSet[];
  images: ProjectImage[];
  projectId: number;
  section: ReturnType<typeof useAssetSection<AnnotationSet>>;
  usedNames: { images: string[]; models: string[]; annotations: string[] };
  onUsedNamesChange: (names: UsedNames) => void;
}

export default function AnnotationsTab({
  annotations,
  images,
  projectId,
  section,
  usedNames,
  onUsedNamesChange,
}: AnnotationsTabProps) {
  const [viewSet, setViewSet] = useState<AnnotationSet | null>(null);
  if (annotations.length === 0) {
    return <p className="mt-6 text-white/70 text-sm">no detected layers yet</p>;
  }

  const sortedAnnotations = sortBySourceThenFolio(
    annotations,
    images,
    (a) => a.imageName,
  );

  return (
    <>
      {viewSet && (
        <AnnotationViewerModal
          set={viewSet}
          projectId={projectId}
          onClose={() => setViewSet(null)}
        />
      )}
      <div
        className="mt-6 grid grid-cols-5 gap-4"
        onMouseDown={(e) => {
          if (e.shiftKey) e.preventDefault();
        }}
        onClick={() => section.clearSelection()}
      >
        {sortedAnnotations.map((set, idx) => {
          const isUsed = usedNames.annotations.includes(set.imageName);
          const isSelected = section.selectedIds.has(set.id);
          const group = sourceGroupLabel(images, set.imageName);
          const prevGroup =
            idx > 0
              ? sourceGroupLabel(images, sortedAnnotations[idx - 1].imageName)
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
                className={`flex flex-col gap-2 ${!isUsed ? "cursor-pointer" : ""}`}
                onClick={(e) => {
                  e.stopPropagation();
                  if (!isUsed) section.handleClick(e, set.id, idx);
                }}
              >
                <div className="relative aspect-square">
                  <div className="absolute inset-0 translate-x-2 translate-y-2 bg-[#C8E6E3]/25 rounded-xl flex items-end justify-start p-2">
                    <span className="text-[10px] text-white/50 font-mono">
                      .txt
                    </span>
                  </div>
                  <div
                    className={`absolute inset-0 bg-[#C8E6E3]/50 rounded-xl overflow-hidden flex items-end justify-start p-2
                    ${isSelected ? "ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]" : ""}
                    ${isUsed ? "opacity-50" : ""}`}
                  >
                    {set.imageSrc && (
                      <AuthImage
                        src={set.imageSrc}
                        alt={set.imageName}
                        className="absolute inset-0 w-full h-full object-cover opacity-60"
                      />
                    )}
                    <span className="relative text-[10px] text-white/80 font-mono z-10">
                      .png
                    </span>
                    {isUsed ? (
                      <span className="absolute top-1.5 left-1.5 z-20 px-1.5 py-0.5 bg-black/40 text-white/60 text-[9px] font-mono rounded">
                        ✓
                      </span>
                    ) : (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          onUsedNamesChange({
                            ...usedNames,
                            annotations: [
                              ...usedNames.annotations,
                              set.imageName,
                            ],
                          });
                        }}
                        className="absolute top-1.5 left-1.5 z-20 px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer"
                      >
                        use
                      </button>
                    )}
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
                <TruncatedName
                  name={set.imageName.replace(/\.[^.]+$/, "")}
                  suffix="_annotations"
                  className="text-sm text-white"
                />
                {set.detectionCount !== undefined && (
                  <span className="text-xs text-white/50">
                    {set.detectionCount} detection
                    {set.detectionCount !== 1 ? "s" : ""}
                  </span>
                )}
                {set.modelLabel && (
                  <span
                    className="text-xs text-white/40 italic truncate"
                    title={set.modelLabel}
                  >
                    {set.modelLabel}
                  </span>
                )}
              </div>
            </Fragment>
          );
        })}
      </div>
    </>
  );
}
