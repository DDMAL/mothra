import { useState } from 'react';
import type { Dispatch, SetStateAction } from 'react';
import type { Project } from '../App';

interface MyProjectsProps {
  projects: Project[];
  setProjects: Dispatch<SetStateAction<Project[]>>;
  onSelectProject: (name: string) => void;
}


export default function MyProjects({ projects, setProjects, onSelectProject }: MyProjectsProps) {
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState('');
  const [hoveredRow, setHoveredRow] = useState<string | null>(null);
  const [deletePopup, setDeletePopup] = useState<{ name: string; x: number; y: number } | null>(null);
  const [renamingRow, setRenamingRow] = useState<string | null>(null);
  const [renameName, setRenameName] = useState('');

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 py-10">
      <div className="max-w-4xl mx-auto flex items-center gap-6 mb-8">
        <h1 className="text-4xl font-bold italic text-white">My Projects</h1>
        <button
          onClick={() => setShowCreate(true)}
          className="px-5 py-2 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer"
        >
          + new project
        </button>
      </div>

      <div className="max-w-4xl mx-auto bg-[#C8E6E3] rounded-2xl overflow-hidden">
        <div className="grid grid-cols-[2fr_2fr_1fr_5rem] px-6 py-3 text-[#1D3335] text-sm font-medium border-b border-[#1D3335]/10">
          <span>project name</span>
          <span>creator</span>
          <span>number of images</span>
          <span />
        </div>
        {projects.map((p) => (
          <div
            key={p.name}
            onMouseEnter={() => setHoveredRow(p.name)}
            onMouseLeave={() => setHoveredRow(null)}
            className="grid grid-cols-[2fr_2fr_1fr_5rem] px-6 py-4 border-b border-[#1D3335]/10 last:border-0 items-center text-[#1D3335] text-sm transition-colors hover:bg-[#b0cdc9]"
          >
            {renamingRow === p.name ? (
              <input
                autoFocus
                value={renameName}
                onChange={(e) => setRenameName(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    setProjects((prev) =>
                      prev.map((proj) =>
                        proj.name === renamingRow
                          ? {...proj, name: renameName.trim() || renamingRow }
                          : proj
                  )
                );
                setRenamingRow(null);
              } else if (e.key === 'Escape') {
                setRenamingRow(null);
              }
            }}
            onBlur={() => {
              setProjects((prev) =>
                prev.map((proj) =>
                  proj.name === renamingRow
                    ? {...proj, name: renameName.trim() || renamingRow }
                      : proj
                )
            );
            setRenamingRow(null);
          }}
          className="bg-white rounded-lg px-3 py-1 text-[#1D3335] outline-none text-sm w-2/3" />
        ) : (
          <span
            onClick={() => onSelectProject(p.name)}
            className="cursor-pointer hover:underline"
          >{p.name}</span>
        )}
            <span>{p.user}</span>
            <span>{p.images.length}</span>
            <div className={`flex gap-3 justify-end transition-opacity ${hoveredRow === p.name ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}>
              <button
                onClick={() => { setRenamingRow(p.name); setRenameName(p.name); }}
                className="cursor-pointer text-base"
              >
                ✏️
              </button>
              <button
                onClick={(e) => setDeletePopup({ name: p.name, x: e.clientX, y: e.clientY })}
                className="cursor-pointer text-base"
              >
                🗑
              </button>
            </div>
            
          </div>
        ))}
      </div>

      {showCreate && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => { setShowCreate(false); setNewName(''); }} />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl">
            <button
              onClick={() => { setShowCreate(false); setNewName(''); }}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              ✕
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">create new project</h2>
            <input
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="project name"
              className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm placeholder:text-[#1D3335]/60"
            />
            <button
              onClick={() => {
                if (!newName.trim()) return;
                setProjects((prev) => [...prev, { name: newName.trim(), user: 'username', images: [] }]);
                setNewName('');
                setShowCreate(false);
              }}
              className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
            >
              create project
            </button>
          </div>
        </>
      )}

      {deletePopup && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setDeletePopup(null)} />
          <div
            className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-2 min-w-[200px]"
            style={{ top: deletePopup.y + 8, left: deletePopup.x - 100 }}
          >
            <p className="text-sm text-[#1D3335] text-center font-medium mb-1">delete this project?</p>
            <button
              onClick={() => {
                setProjects((prev) => prev.filter((p) => p.name !== deletePopup.name));
                setDeletePopup(null);
              }}
              className="bg-[#1E6B70] text-white rounded-xl px-4 py-2 text-sm hover:opacity-90 cursor-pointer"
            >
              yes, delete
            </button>
            <button
              onClick={() => setDeletePopup(null)}
              className="text-sm text-[#1D3335] hover:opacity-70 cursor-pointer text-center"
            >
              no, keep this project
            </button>
          </div>
        </>
      )}
    </div>
  );
}