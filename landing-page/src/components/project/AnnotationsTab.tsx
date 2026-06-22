import type { AnnotationSet } from "../../types";
import { AuthImage } from "../shared/AuthImage";

interface AnnotationsTabProps {
  annotations: AnnotationSet[];
}

export default function AnnotationsTab({ annotations }: AnnotationsTabProps) {
  if (annotations.length === 0) {
    return <p className="mt-6 text-white/70 text-sm">no annotations yet</p>;
  }

  return (
    <div className="mt-6 grid grid-cols-5 gap-4">
      {annotations.map((set) => (
        <div key={set.id} className="flex flex-col gap-2">
          <div className="relative aspect-square">
            <div className="absolute inset-0 translate-x-2 translate-y-2 bg-[#C8E6E3]/25 rounded-xl flex items-end justify-start p-2">
              <span className="text-[10px] text-white/50 font-mono">.txt</span>
            </div>
            <div className="absolute inset-0 bg-[#C8E6E3]/50 rounded-xl overflow-hidden flex items-end justify-start p-2">
              {set.imageSrc && (
                <AuthImage
                  src={set.imageSrc}
                  alt={set.imageName}
                  className="absolute inset-0 w-full h-full object-cover opacity-60"
                />
              )}
              <span className="relative text-[10px] text-white/80 font-mono z-10">.png</span>
            </div>
          </div>
          <span className="text-sm text-white truncate">
            {set.imageName.replace(/\.[^.]+$/, "")}
          </span>
          {set.detectionCount !== undefined && (
            <span className="text-xs text-white/50">
              {set.detectionCount} detection{set.detectionCount !== 1 ? "s" : ""}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
