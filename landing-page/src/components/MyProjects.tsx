import { useState } from "react";
import type { Project } from "../App";
import DeleteProjectModal from "./DeleteProjectModal";

interface MyProjectsProps {
  projects: Project[];
  onSelectProject: (id: number) => void;
  onCreateProject: (name: string) => void;
  onRenameProject: (id: number, newName: string) => void;
  onDeleteProject: (id: number) => void;
  onRestoreProject: (id: number) => void;
}

export default function MyProjects({
  projects,
  onSelectProject,
  onCreateProject,
  onRenameProject,
  onDeleteProject,
  onRestoreProject,
}: MyProjectsProps) {
  const [tab, setTab] = useState<"active" | "trash">("active");
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  const [deleteConfirmProject, setDeleteConfirmProject] = useState<number | null>(null);
  const [renamingRow, setRenamingRow] = useState<number | null>(null);
  const [renameName, setRenameName] = useState("");

  const activeProjects = projects.filter((p) => !p.deletedAt);
  const trashedProjects = projects.filter((p) => !!p.deletedAt);

  const projectToDelete = deleteConfirmProject !== null
    ? projects.find((p) => p.id === deleteConfirmProject) ?? null
    : null;

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 py-10 relative">
      <div
        className={`absolute inset-0 z-30 bg-black/30 transition-opacity pointer-events-none
                      ${showCreate || deleteConfirmProject !== null ? "opacity-100" : "opacity-0"}`}
      />
      <div className="max-w-4xl mx-auto flex items-center gap-6 mb-6">
        <h1 className="text-4xl font-bold italic text-white">My Projects</h1>
        {tab === "active" && (
          <button
            onClick={() => setShowCreate(true)}
            className="px-5 py-2 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer"
          >
            + new project
          </button>
        )}
      </div>

      {/* tab bar */}
      <div className="max-w-4xl mx-auto flex items-end mb-0">
        {(["active", "trash"] as const).map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`relative px-8 pt-3 pb-2 text-xl font-bold italic rounded-t-xl cursor-pointer transition-colors
              ${tab === t
                ? "text-white border border-white/50 border-b-0 bg-[#4AADAA] z-10"
                : "text-white/50 hover:text-white/70 border border-transparent"}
              ${i > 0 ? "-ml-px" : ""}`}
          >
            {t === "active" ? "my projects" : "trash"}
          </button>
        ))}
        <div className="flex-1 border-b border-white/50" />
      </div>

      <div className="max-w-4xl mx-auto bg-[#C8E6E3] rounded-b-2xl rounded-tr-2xl overflow-hidden">
        {tab === "active" ? (
          <>
            <div className="grid grid-cols-[2fr_2fr_1fr_5rem] px-6 py-3 text-[#1D3335] text-sm font-medium border-b border-[#1D3335]/10">
              <span>project name</span>
              <span>creator</span>
              <span>number of images</span>
              <span />
            </div>
            {activeProjects.map((p) => (
              <div
                key={p.id}
                onMouseEnter={() => setHoveredRow(p.id)}
                onMouseLeave={() => setHoveredRow(null)}
                className="grid grid-cols-[2fr_2fr_1fr_5rem] px-6 py-4 border-b border-[#1D3335]/10 last:border-0 items-center text-[#1D3335] text-sm transition-colors hover:bg-[#b0cdc9]"
              >
                {renamingRow === p.id ? (
                  <input
                    autoFocus
                    value={renameName}
                    onChange={(e) => setRenameName(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        onRenameProject(p.id, renameName.trim() || p.name);
                        setRenamingRow(null);
                      } else if (e.key === "Escape") {
                        setRenamingRow(null);
                      }
                    }}
                    onBlur={() => {
                      onRenameProject(p.id, renameName.trim() || p.name);
                      setRenamingRow(null);
                    }}
                    className="bg-white rounded-lg px-3 py-1 text-[#1D3335] outline-none text-sm w-2/3"
                  />
                ) : (
                  <span
                    onClick={() => onSelectProject(p.id)}
                    className="cursor-pointer hover:underline"
                  >
                    {p.name}
                  </span>
                )}
                <span>{p.user}</span>
                <span>{p.images.length}</span>
                <div
                  className={`flex gap-3 justify-end transition-opacity ${hoveredRow === p.id ? "opacity-100" : "opacity-0 pointer-events-none"}`}
                >
                  <button
                    onClick={() => {
                      setRenamingRow(p.id);
                      setRenameName(p.name);
                    }}
                    className="cursor-pointer text-base"
                  >
                    ✏️
                  </button>
                  <button
                    onClick={() => setDeleteConfirmProject(p.id)}
                    className="cursor-pointer text-base"
                  >
                    🗑
                  </button>
                </div>
              </div>
            ))}
            {activeProjects.length === 0 && (
              <p className="px-6 py-6 text-sm text-[#1D3335]/60">no projects yet</p>
            )}
          </>
        ) : (
          <>
            <div className="grid grid-cols-[2fr_2fr_1fr_6rem] px-6 py-3 text-[#1D3335] text-sm font-medium border-b border-[#1D3335]/10">
              <span>project name</span>
              <span>creator</span>
              <span>days remaining</span>
              <span />
            </div>
            {trashedProjects.map((p) => {
              const daysLeft = Math.max(
                0,
                30 - Math.floor((Date.now() - (p.deletedAt ?? 0)) / 86400000),
              );
              return (
                <div
                  key={p.id}
                  className="grid grid-cols-[2fr_2fr_1fr_6rem] px-6 py-4 border-b border-[#1D3335]/10 last:border-0 items-center text-[#1D3335] text-sm"
                >
                  <span className="opacity-60">{p.name}</span>
                  <span className="opacity-60">{p.user}</span>
                  <span className="opacity-60">{daysLeft}d</span>
                  <div className="flex justify-end">
                    <button
                      onClick={() => onRestoreProject(p.id)}
                      className="text-xs text-[#1E6B70] font-semibold hover:opacity-70 cursor-pointer"
                    >
                      restore
                    </button>
                  </div>
                </div>
              );
            })}
            {trashedProjects.length === 0 && (
              <p className="px-6 py-6 text-sm text-[#1D3335]/60">trash is empty</p>
            )}
          </>
        )}
      </div>

      {showCreate && (
        <>
          <div
            className="fixed inset-0 z-40"
            onClick={() => {
              setShowCreate(false);
              setNewName("");
            }}
          />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl">
            <button
              onClick={() => {
                setShowCreate(false);
                setNewName("");
              }}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              ✕
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">
              create new project
            </h2>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="project name"
              className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
            />
            <button
              onClick={() => {
                if (!newName.trim()) return;
                onCreateProject(newName.trim());
                setNewName("");
                setShowCreate(false);
              }}
              className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
            >
              create project
            </button>
          </div>
        </>
      )}

      {projectToDelete && (
        <DeleteProjectModal
          project={projectToDelete}
          onConfirm={() => {
            onDeleteProject(projectToDelete.id);
            setDeleteConfirmProject(null);
          }}
          onCancel={() => setDeleteConfirmProject(null)}
        />
      )}
    </div>
  );
}
