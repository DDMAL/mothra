import { useState } from "react";
import { AuthImage } from "../shared/AuthImage";
import type { Project, MeiFile } from "../../types";
import DeleteProjectModal from "./DeleteProjectModal";
import { formatLastOpened } from "../../utils/time";
import Modal from "../shared/Modal";

const LIST_COLS = "grid-cols-[2fr_1fr_1fr_5rem]";
const TRASH_COLS = "grid-cols-[2fr_2fr_1fr_8rem]";

interface MyProjectsProps {
  projects: Project[];
  onSelectProject: (id: number) => void;
  onCreateProject: (name: string) => void;
  onRenameProject: (id: number, newName: string) => void;
  onDeleteProject: (id: number) => void;
  onRestoreProject: (id: number) => void;
  onPermanentlyDeleteProject: (id: number) => void;
  onTogglePin: (id: number) => void;
}

function MeiProgress({ meiFiles }: { meiFiles: MeiFile[] }) {
  if (meiFiles.length === 0) return null;
  const corrected = meiFiles.filter((f) => f.corrected).length;
  return (
    <div className="flex items-center gap-2 mt-0.5">
      <div className="h-1.5 w-20 bg-[#1D3335]/10 rounded-full overflow-hidden flex-shrink-0">
        <div
          className="h-full bg-[#1E6B70] rounded-full transition-all"
          style={{ width: `${(corrected / meiFiles.length) * 100}%` }}
        />
      </div>
      <span className="text-xs text-[#1D3335]/50">
        {corrected}/{meiFiles.length} corrected
      </span>
    </div>
  );
}

function sortProjects(
  a: Project,
  b: Project,
  sortBy: "lastOpened" | "dateCreated" | "nameAZ",
): number {
  if (a.isPinned && !b.isPinned) return -1;
  if (!a.isPinned && b.isPinned) return 1;
  if (sortBy === "nameAZ") return a.name.localeCompare(b.name);
  if (sortBy === "dateCreated") return b.id - a.id;
  if (!a.lastOpenedAt && !b.lastOpenedAt) return 0;
  if (!a.lastOpenedAt) return 1;
  if (!b.lastOpenedAt) return -1;
  return (
    new Date(b.lastOpenedAt).getTime() - new Date(a.lastOpenedAt).getTime()
  );
}

function PinButton({
  isPinned,
  onToggle,
}: {
  isPinned: boolean;
  onToggle: () => void;
}) {
  return (
    <button
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      className="text-sm leading-none cursor-pointer shrink-0"
      title={isPinned ? "unpin" : "pin to top"}
    >
      {isPinned ? "★" : "☆"}
    </button>
  );
}

export default function MyProjects({
  projects,
  onSelectProject,
  onCreateProject,
  onRenameProject,
  onDeleteProject,
  onRestoreProject,
  onPermanentlyDeleteProject,
  onTogglePin,
}: MyProjectsProps) {
  const [tab, setTab] = useState<"active" | "trash">("active");
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [hoveredRow, setHoveredRow] = useState<number | null>(null);
  const [deleteConfirmProject, setDeleteConfirmProject] = useState<
    number | null
  >(null);
  const [renamingRow, setRenamingRow] = useState<number | null>(null);
  const [renameName, setRenameName] = useState("");
  const [viewMode, setViewMode] = useState<"list" | "gallery">("list");
  const [showDeleteAllConfirm, setShowDeleteAllConfirm] = useState(false);

  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"lastOpened" | "dateCreated" | "nameAZ">(
    "lastOpened",
  );

  const activeProjects = projects
    .filter((p) => !p.deletedAt)
    .filter(
      (p) => !search || p.name.toLowerCase().includes(search.toLowerCase()),
    )
    .sort((a, b) => sortProjects(a, b, sortBy));

  const trashedProjects = projects.filter((p) => !!p.deletedAt);

  const projectToDelete =
    deleteConfirmProject !== null
      ? (projects.find((p) => p.id === deleteConfirmProject) ?? null)
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
        <div className="ml-auto flex items-center gap-1 bg-[#C8E6E3]/30 rounded-lg p-1">
          <button
            onClick={() => setViewMode("list")}
            title="list view"
            className={`px-2 py-1 rounded text-sm transition-colors cursor-pointer ${viewMode === "list" ? "bg-white text-[#1D3335]" : "text-white/70 hover:text-white"}`}
          >
            ☰
          </button>
          <button
            onClick={() => setViewMode("gallery")}
            title="gallery view"
            className={`px-2 py-1 rounded text-sm transition-colors cursor-pointer ${viewMode === "gallery" ? "bg-white text-[#1D3335]" : "text-white/70 hover:text-white"}`}
          >
            ⊞
          </button>
        </div>
      </div>

      {/* tab bar */}
      <div className="max-w-4xl mx-auto flex items-end mb-0">
        {(["active", "trash"] as const).map((t, i) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`relative px-8 pt-3 pb-2 text-xl font-bold italic rounded-t-xl cursor-pointer transition-colors
              ${
                tab === t
                  ? "text-white border border-white/50 border-b-0 bg-[#4AADAA] z-10"
                  : "text-white/50 hover:text-white/70 border border-transparent"
              }
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
            <div className="px-6 pt-4 pb-3 flex items-center gap-3 border-b border-[#1D3335]/10">
              <input
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                placeholder="search projects..."
                className="flex-1 bg-white/70 rounded-xl px-4 py-1.5 text-sm text-[#1D3335] outline-none placeholder:text-[#1D3335]/40"
              />
              <select
                value={sortBy}
                onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
                className="bg-white/70 rounded-xl px-3 py-1.5 text-sm text-[#1D3335] outline-none cursor-pointer"
              >
                <option value="lastOpened">last opened</option>
                <option value="dateCreated">date created</option>
                <option value="nameAZ">name a–z</option>
              </select>
            </div>
            {viewMode === "list" ? (
              <>
                <div
                  className={`grid ${LIST_COLS} px-6 py-3 text-[#1D3335] text-sm font-medium border-b border-[#1D3335]/10`}
                >
                  <span>project name</span>
                  <span>creator</span>
                  <span>last opened</span>
                  <span />
                </div>
                {activeProjects.map((p) => (
                  <div
                    key={p.id}
                    onMouseEnter={() => setHoveredRow(p.id)}
                    onMouseLeave={() => setHoveredRow(null)}
                    className={`grid ${LIST_COLS} px-6 py-4 border-b border-[#1D3335]/10 last:border-0 items-center text-[#1D3335] text-sm transition-colors hover:bg-[#b0cdc9]`}
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
                      <div className="flex flex-col">
                        <div className="flex items-center gap-1.5">
                          <PinButton
                            isPinned={!!p.isPinned}
                            onToggle={() => onTogglePin(p.id)}
                          />
                          <span
                            onClick={() => onSelectProject(p.id)}
                            className="cursor-pointer hover:underline"
                          >
                            {p.name}
                          </span>
                        </div>
                        <MeiProgress meiFiles={p.meiFiles} />
                      </div>
                    )}
                    <span>{p.user}</span>
                    <span className="text-[#1D3335]/60">
                      {formatLastOpened(p.lastOpenedAt)}
                    </span>
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
                  <p className="px-6 py-6 text-sm text-[#1D3335]/60">
                    no projects yet
                  </p>
                )}
              </>
            ) : (
              <>
                {activeProjects.length === 0 ? (
                  <p className="px-6 py-6 text-sm text-[#1D3335]/60">
                    no projects yet
                  </p>
                ) : (
                  <div className="grid grid-cols-2 sm:grid-cols-3 gap-4 p-4">
                    {activeProjects.map((p) => (
                      <div
                        key={p.id}
                        onClick={() => onSelectProject(p.id)}
                        onMouseEnter={() => setHoveredRow(p.id)}
                        onMouseLeave={() => setHoveredRow(null)}
                        className="bg-white rounded-2xl overflow-hidden cursor-pointer hover:opacity-90 transition-opacity relative"
                      >
                        <div className="aspect-[4/3] bg-[#1D3335]/10">
                          {p.images.length > 0 ? (
                            <AuthImage
                              src={`/api/images/${p.images[0].id}`}
                              className="w-full h-full"
                            />
                          ) : (
                            <div className="w-full h-full flex items-center justify-center text-[#1D3335]/30 text-xs">
                              no images
                            </div>
                          )}
                        </div>
                        <div className="p-3 flex flex-col gap-1">
                          <div className="flex items-center gap-1.5">
                            <PinButton
                              isPinned={!!p.isPinned}
                              onToggle={() => onTogglePin(p.id)}
                            />
                            <span className="font-semibold text-[#1D3335] text-sm truncate">
                              {p.name}
                            </span>
                          </div>
                          <span className="text-[#1D3335]/50 text-xs">
                            {formatLastOpened(p.lastOpenedAt)}
                          </span>
                          <MeiProgress meiFiles={p.meiFiles} />
                        </div>
                        {hoveredRow === p.id && (
                          <div
                            className="absolute top-2 right-2 flex gap-1"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <button
                              onClick={() => {
                                setRenamingRow(p.id);
                                setRenameName(p.name);
                                setViewMode("list");
                              }}
                              className="bg-white/80 rounded-lg px-2 py-1 text-xs cursor-pointer hover:opacity-70"
                            >
                              ✏️
                            </button>
                            <button
                              onClick={() => setDeleteConfirmProject(p.id)}
                              className="bg-white/80 rounded-lg px-2 py-1 text-xs cursor-pointer hover:opacity-70"
                            >
                              🗑
                            </button>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </>
            )}
          </>
        ) : (
          <>
            {trashedProjects.length > 0 && (
              <div className="flex gap-3 px-6 pt-4 pb-2">
                <button
                  onClick={() =>
                    trashedProjects.forEach((p) => onRestoreProject(p.id))
                  }
                  className="text-xs text-[#1E6B70] font-semibold hover:opacity-70 cursor-pointer"
                >
                  restore all
                </button>
                <button
                  onClick={() => setShowDeleteAllConfirm(true)}
                  className="text-xs text-red-600 font-semibold hover:opacity-70 cursor-pointer"
                >
                  delete all
                </button>
              </div>
            )}
            <div
              className={`grid ${TRASH_COLS} px-6 py-3 text-[#1D3335] text-sm font-medium border-b border-[#1D3335]/10`}
            >
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
                  className={`grid ${TRASH_COLS} px-6 py-4 border-b border-[#1D3335]/10 last:border-0 items-center text-[#1D3335] text-sm`}
                >
                  <span className="opacity-60">{p.name}</span>
                  <span className="opacity-60">{p.user}</span>
                  <span className="opacity-60">{daysLeft}d</span>
                  <div className="flex justify-end gap-3">
                    <button
                      onClick={() => onRestoreProject(p.id)}
                      className="text-xs text-[#1E6B70] font-semibold hover:opacity-70 cursor-pointer"
                    >
                      restore
                    </button>
                    <button
                      onClick={() => onPermanentlyDeleteProject(p.id)}
                      className="text-xs text-red-500 font-semibold hover:opacity-70 cursor-pointer"
                    >
                      delete
                    </button>
                  </div>
                </div>
              );
            })}
            {trashedProjects.length === 0 && (
              <p className="px-6 py-6 text-sm text-[#1D3335]/60">
                trash is empty
              </p>
            )}
          </>
        )}
      </div>

      {showCreate && (
        <Modal
          onClose={() => {
            setShowCreate(false);
            setNewName("");
          }}
          size="lg"
          backdrop="none"
        >
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
        </Modal>
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

      {showDeleteAllConfirm && (
        <Modal
          size="sm"
          backdrop="dim"
          onClose={() => setShowDeleteAllConfirm(false)}
        >
          <h2 className="text-xl text-[#1D3335] text-center">
            delete all trashed projects?
          </h2>
          <p className="text-sm text-[#1D3335]/70 text-center">
            this will permanently delete {trashedProjects.length} project
            {trashedProjects.length !== 1 ? "s" : ""} and cannot be undone.
          </p>
          <div className="flex gap-3 justify-center">
            <button
              onClick={() => {
                trashedProjects.forEach((p) =>
                  onPermanentlyDeleteProject(p.id),
                );
                setShowDeleteAllConfirm(false);
              }}
              className="px-6 py-2.5 bg-red-600 text-white font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
            >
              yes, delete all
            </button>
            <button
              onClick={() => setShowDeleteAllConfirm(false)}
              className="px-6 py-2.5 border-2 border-[#1D3335]/30 text-[#1D3335] font-semibold rounded-xl hover:opacity-70 cursor-pointer text-sm"
            >
              cancel
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}
