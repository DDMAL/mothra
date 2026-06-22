import type { Dispatch, SetStateAction } from "react";
import { authHeaders } from "./useAuth";
import type { Project } from "../types";

type SetProjects = Dispatch<SetStateAction<Project[]>>;

export function useProjectMutations(setProjects: SetProjects) {
  const createProject = async (name: string) => {
    const r = await fetch("/api/projects", {
      method: "POST",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    const project = await r.json();
    setProjects((prev) => [...prev, project]);
  };

  const renameProject = async (id: number, newName: string) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ name: newName }),
    });
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, name: newName } : p)),
    );
  };

  const deleteProject = async (id: number) => {
    const deletedAt = new Date().toISOString();
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ deletedAt }),
    });
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, deletedAt: Date.now() } : p)),
    );
  };

  const restoreProject = async (id: number) => {
    await fetch(`/api/projects/${id}/restore`, {
      method: "POST",
      headers: authHeaders(),
    });
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, deletedAt: undefined } : p)),
    );
  };

  const permanentlyDeleteProject = async (id: number) => {
    await fetch(`/api/projects/${id}`, {
      method: "DELETE",
      headers: authHeaders(),
    });
    setProjects((prev) => prev.filter((p) => p.id !== id));
  };

  const updateProjectSteps = async (id: number, steps: number) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ stepsUnlocked: steps }),
    });
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, stepsUnlocked: steps } : p)),
    );
  };

  const updateUsedImageNames = async (id: number, names: string[]) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ usedImageNames: names }),
    });
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, usedImageNames: names } : p)),
    );
  };

  const updateUsedModelNames = async (id: number, names: string[]) => {
    await fetch(`/api/projects/${id}`, {
      method: "PUT",
      headers: { ...authHeaders(), "Content-Type": "application/json" },
      body: JSON.stringify({ usedModelNames: names }),
    });
    setProjects((prev) =>
      prev.map((p) => (p.id === id ? { ...p, usedModelNames: names } : p)),
    );
  };

  const togglePin = (id: number) => {
    setProjects((prev) => {
      const project = prev.find((p) => p.id === id);
      if (!project) return prev;
      const isPinned = !project.isPinned;
      fetch(`/api/projects/${id}`, {
        method: "PUT",
        headers: { ...authHeaders(), "Content-Type": "application/json" },
        body: JSON.stringify({ isPinned }),
      });
      return prev.map((p) => (p.id === id ? { ...p, isPinned } : p));
    });
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
    togglePin,
  };
}
