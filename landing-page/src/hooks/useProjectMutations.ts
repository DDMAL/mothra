import type { Dispatch, SetStateAction } from "react";
import { authHeaders } from "./useAuth";
import type { Project } from "../types";

type SetProjects = Dispatch<SetStateAction<Project[]>>;

export function useProjectMutations(
  setProjects: SetProjects,
  onError?: (message: string) => void,
) {
  const createProject = async (name: string) => {
    try {
      const r = await fetch("/api/projects", {
        method: "POST",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
      });
      if (!r.ok) throw new Error("failed to create project");
      const project = await r.json();
      setProjects((prev) => [...prev, project]);
    } catch (e) {
      onError?.((e as Error).message);
    }
  };

  const renameProject = async (id: number, newName: string) => {
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ name: newName }),
      });
      if (!r.ok) throw new Error("failed to rename project");
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, name: newName } : p)),
      );
    } catch (e) {
      onError?.((e as Error).message);
    }
  };

  const deleteProject = async (id: number) => {
    const deletedAt = new Date().toISOString();
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ deletedAt }),
      });
      if (!r.ok) throw new Error("failed to delete project");
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, deletedAt: Date.now() } : p)),
      );
    } catch (e) {
      onError?.((e as Error).message);
    }
  };

  const restoreProject = async (id: number) => {
    try {
      const r = await fetch(`/api/projects/${id}/restore`, {
        method: "POST",
        headers: authHeaders(),
      });
      if (!r.ok) throw new Error("failed to restore project");
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, deletedAt: undefined } : p)),
      );
    } catch (e) {
      onError?.((e as Error).message);
    }
  };

  const permanentlyDeleteProject = async (id: number) => {
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "DELETE",
        headers: authHeaders(),
      });
      if (!r.ok) throw new Error("failed to permanently delete project");
      setProjects((prev) => prev.filter((p) => p.id !== id));
    } catch (e) {
      onError?.((e as Error).message);
    }
  };

  const updateProjectSteps = async (id: number, steps: number) => {
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
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

  const updateUsedImageNames = async (id: number, names: string[]) => {
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ usedImageNames: names }),
      });
      if (!r.ok) return;
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, usedImageNames: names } : p)),
      );
    } catch {
      // internal bookkeeping — silent
    }
  };

  const updateUsedModelNames = async (id: number, names: string[]) => {
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
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

  const updateUsedAnnotationNames = async (id: number, names: string[]) => {
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ usedAnnotationNames: names }),
      });
      if (!r.ok) return;
      setProjects((prev) =>
        prev.map((p) =>
          p.id === id ? { ...p, usedAnnotationNames: names } : p,
        ),
      );
    } catch {
      // internal bookkeeping — silent
    }
  };

  const togglePin = async (id: number) => {
    let newIsPinned: boolean | undefined;
    setProjects((prev) => {
      const project = prev.find((p) => p.id === id);
      if (!project) return prev;
      newIsPinned = !project.isPinned;
      return prev.map((p) => (p.id === id ? { ...p, isPinned: newIsPinned! } : p));
    });
    try {
      const r = await fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ isPinned: newIsPinned }),
      });
      if (!r.ok) throw new Error("failed to update pin");
    } catch (e) {
      // revert optimistic update
      setProjects((prev) =>
        prev.map((p) => (p.id === id ? { ...p, isPinned: !newIsPinned } : p)),
      );
      onError?.((e as Error).message);
    }
  };

  return {
    createProject,
    renameProject,
    deleteProject,
    restoreProject,
    permanentlyDeleteProject,
    updateProjectSteps,
    updateUsedImageNames,
    updateUsedModelNames,
    updateUsedAnnotationNames,
    togglePin,
  };
}