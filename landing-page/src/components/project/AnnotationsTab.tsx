import { useState } from "react";
import type { Project, AnnotationSet } from "../../types";
import { AuthImage } from "../shared/AuthImage";

interface AnnotationsTabProps {
  annotations: AnnotationSet[];
  project: Project;
  onRunDetection: (modelId: string, imageIds: string[]) => Promise<void>;
}

export default function AnnotationsTab({ 
  annotations,
  project,
  onRunDetection,
}: AnnotationsTabProps) {
  const [selectedModelId, setSelectedModelId] = useState<string>(
    project.models[0]?.id ?? "",
  );
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const usedImages = project.images.filter((img) => project.usedImageNames.includes(img.name), );

  const handleRun = async () => {
    if (!selectedModelId || usedImages.length === 0) return;
    setRunning(true);
    setError(null);
    try {
      await onRunDetection(
        selectedModelId,
        usedImages.map((i) => i.id),
      );
    } catch {
      setError("Inference failed. Check that the model file is valid.");
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="mt-6 flex flex-col gap-6">
      {/* run detection panel */}
      <div className="flex items-center gap-4 flex-wrap">
        <select
          value={selectedModelId}
          onChange={(e) => setSelectedModelId(e.target.value)}
          disabled={project.models.length === 0 || running}
          className="px-3 py-1.5 rounded-xl bg-white/20 text-white text-sm border border-white/20 cursor-pointer disabled:opacity-40"
        >
          {project.models.length === 0 ? (
            <option value="">no models uploaded</option>
          ) : (
            project.models.map((m) => (
              <option key={m.id} value={m.id}>
                {m.name}
              </option>
            ))
          )}
        </select>
        <span className="text-white/50 text-sm">
          {usedImages.length} image{usedImages.length !== 1 ? "s" : ""} selected
        </span>
        <button
          onClick={handleRun}
          disabled={!selectedModelId || usedImages.length === 0 || running}
          className="px-5 py-1.5 bg-[#4AADAA] text-white text-sm font-semibold rounded-xl hover:opacity-90 disabled:opacity-40 cursor-pointer disabled:cursor-not-allowed transition-opacity"
        >
          {running ? "running…" : "run detection"}
        </button>
        {error && <span className="text-red-300 text-sm">{error}</span>}
      </div>

      {/* results grid */}
      {annotations.length === 0 ? (
        <p className="text-white/70 text-sm">no annotations yet</p>
      ) : (
        <div className="grid grid-cols-5 gap-4">
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
      )}
    </div>
  );
}
