import type { Project } from "../types";

interface DeleteProjectModalProps {
  project: Project;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function DeleteProjectModal({ project, onConfirm, onCancel }: DeleteProjectModalProps) {
  return (
    <>
      <div className="fixed inset-0 z-40 bg-black/30" onClick={onCancel} />
      <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-5 relative shadow-2xl">
        <button
          onClick={onCancel}
          className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
        >
          ✕
        </button>
        <h2 className="text-xl text-[#1D3335] font-semibold text-center">delete project?</h2>
        <p className="text-sm text-[#1D3335] text-center leading-relaxed">
          deleting <span className="font-semibold">"{project.name}"</span> will result in{" "}
          {project.images.length} image{project.images.length !== 1 ? "s" : ""},{" "}
          {project.models.length} model{project.models.length !== 1 ? "s" : ""},{" "}
          {project.annotations.length} annotation{project.annotations.length !== 1 ? "s" : ""}, and{" "}
          {project.meiFiles.length} mei file{project.meiFiles.length !== 1 ? "s" : ""} being deleted.
          deleted projects can be found and restored in the "trash" tab on the "my projects" page for up to 30 days.
        </p>
        <div className="flex gap-3 justify-center">
          <button
            onClick={onConfirm}
            className="px-6 py-2.5 bg-[#1E6B70] text-white font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
          >
            delete project
          </button>
          <button
            onClick={onCancel}
            className="px-6 py-2.5 border-2 border-[#1D3335]/30 text-[#1D3335] font-semibold rounded-xl hover:opacity-70 cursor-pointer text-sm"
          >
            cancel
          </button>
        </div>
      </div>
    </>
  );
}
