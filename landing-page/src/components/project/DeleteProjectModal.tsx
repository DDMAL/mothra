import type { Project } from "../types";
import Modal from "./Modal";

interface DeleteProjectModalProps {
  project: Project;
  onConfirm: () => void;
  onCancel: () => void;
}

export default function DeleteProjectModal({ project, onConfirm, onCancel }: DeleteProjectModalProps) {
  return (
    <Modal onClose={onCancel} backdrop="dim">
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
    </Modal> 
  );
}
