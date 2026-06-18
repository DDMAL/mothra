import type { Project } from "../types";

export function normalizeProjects(raw: Project[]): Project[] {
  return raw.map(p => ({
    ...p,
    images: p.images.map(img => ({ ...img, src: img.src ?? `/api/images/${img.id}` })),
  }));
}
