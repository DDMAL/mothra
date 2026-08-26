import type { Dispatch, SetStateAction } from "react";
import { apiFetch } from "../lib/apiFetch";
import { toast } from "../lib/toast";
import type { Project } from "../types";
import { normalizeProjects } from "../utils/projects";

type SetProjects = Dispatch<SetStateAction<Project[]>>;

export function useProjectMutations(setProjects: SetProjects) {
  const createProject = async (name: string, imageFile?: File) => {
    try {
      const r = await apiFetch("/api/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error("failed to create project");
      const project: Project = await r.json();
      // Add the project to local state right away — a slow or stalled
      // image upload shouldn't leave the (already server-side-created)
      // project missing from the list, where the user could be tempted to
      // submit the create flow again.
      setProjects((prev) => [...prev, project]);
      if (imageFile) {
        // A failed auto-upload shouldn't undo project creation — the project
        // still exists and is usable, the user just needs to upload the
        // image manually from the Images tab instead.
        try {
          const form = new FormData();
          form.append("file", imageFile);
          const ir = await apiFetch(`/api/projects/${project.id}/images`, {
            method: "POST",
            body: form,
          });
          if (!ir.ok) throw new Error("image upload failed");
          const uploaded = await ir.json();
          const uploadedImage: Project["images"][number] = {
            id: uploaded.id,
            name: uploaded.name,
            src: `/api/images/${uploaded.id}`,
            folio: uploaded.folio,
            sourceId: uploaded.sourceId,
            sourceName: uploaded.sourceName,
          };
          setProjects((prev) =>
            prev.map((p) =>
              p.id === project.id
                ? { ...p, images: [...p.images, uploadedImage] }
                : p,
            ),
          );
        } catch {
          toast.error(
            `"${name}" was created, but the image failed to upload — try uploading it from the Images tab instead`,
          );
        }
      }
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const renameProject = async (id: number, newName: string) => {
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName }),
      });
      if (!r.ok) throw new Error("failed to rename project");
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, name: newName } : p)),
      );
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const deleteProject = async (id: number) => {
    const deletedAt = new Date().toISOString();
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ deletedAt }),
      });
      if (!r.ok) throw new Error("failed to delete project");
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, deletedAt } : p)),
      );
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const restoreProject = async (id: number) => {
    try {
      const r = await apiFetch(`/api/projects/${id}/restore`, {
        method: "POST",
      });
      if (!r.ok) throw new Error("failed to restore project");
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, deletedAt: undefined } : p)),
      );
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const permanentlyDeleteProject = async (id: number) => {
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error("failed to permanently delete project");
      setProjects((prev) => prev.filter((p) => p.id !== id));
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const duplicateProject = async (id: number) => {
    const r = await apiFetch(`/api/projects/${id}/duplicate`, {
      method: "POST",
    });
    if (!r.ok) throw new Error("duplicate failed");
    const newProject: Project = normalizeProjects([await r.json()])[0];
    setProjects((prev) => [newProject, ...prev]);
    return newProject;
  };

  const updateProjectSteps = async (id: number, steps: number) => {
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ stepsUnlocked: steps }),
      });
      if (!r.ok) return;
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, stepsUnlocked: steps } : p)),
      );
    } catch {
      // internal bookkeeping — silent
    }
  };

  // mothra#241 follow-up (CodeRabbit): ids, not names -- see
  // Project.usedImageIds's comment in types.ts.
  const updateUsedImageIds = async (id: number, ids: string[]) => {
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ usedImageIds: ids }),
      });
      if (!r.ok) return;
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, usedImageIds: ids } : p)),
      );
    } catch {
      // internal bookkeeping — silent
    }
  };

  const updateUsedModelNames = async (id: number, names: string[]) => {
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ usedModelNames: names }),
      });
      if (!r.ok) return;
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, usedModelNames: names } : p)),
      );
    } catch {
      // internal bookkeeping — silent
    }
  };

  const updateCantusSourceId = async (id: number, sourceId: string) => {
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ cantusSourceId: sourceId }),
      });
      if (!r.ok) return;
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, cantusSourceId: sourceId } : p)),
      );
    } catch {
      // internal bookkeeping - silent
    }
  };

  const togglePin = async (id: number) => {
    let newIsPinned: boolean | undefined;
    setProjects((prev) => {
      const project = prev.find((p) => p.id === id);
      if (!project) return prev;
      newIsPinned = !project.isPinned;
      return prev.map((p) =>
        p.id === id ? { ...p, isPinned: newIsPinned! } : p,
      );
    });
    try {
      const r = await apiFetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ isPinned: newIsPinned }),
      });
      if (!r.ok) throw new Error("failed to update pin");
    } catch (e) {
      // revert optimistic update
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, isPinned: !newIsPinned } : p)),
      );
      toast.error((e as Error).message);
    }
  };

  return {
    createProject,
    renameProject,
    deleteProject,
    restoreProject,
    permanentlyDeleteProject,
    duplicateProject,
    updateProjectSteps,
    updateUsedImageIds,
    updateUsedModelNames,
    updateCantusSourceId,
    togglePin,
  };
}
