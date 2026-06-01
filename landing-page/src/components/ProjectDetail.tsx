import { useEffect, useState } from 'react';
import type { Project } from '../App'

interface ProjectDetailProps {
    project: Project;
    onBack: () => void;
    onUpdateProject: (updated: Project) => void;
}

export default function ProjectDetail({ project, onBack, onUpdateProject }: ProjectDetailProps) {
    const [imageMenu, setImageMenu] = useState<{id: string; x: number; y: number } | null>(null);
    const [renameModal, setRenameModal] = useState<{id: string} | null>(null);
    const [renameName, setRenameName] = useState('');

    useEffect (() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') { setImageMenu(null); setRenameModal(null);}

        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, []);

    const deleteImage = (id: string) => {
        onUpdateProject({ ...project, images: project.images.filter((img) => img.id !== id) });
        setImageMenu(null);
    }

    const renameImage = () => {
        const current = project.images.find((img) => img.id === renameModal?.id);
        onUpdateProject({
            ...project,
            images: project.images.map((img) =>
                img.id === renameModal?.id ? { ...img, name: renameName.trim() || current!.name } : img),
        });
        setRenameModal(null);
    };

    const addImage = () => {
        const n = project.images.length + 1;
        onUpdateProject({
            ...project,
            images: [...project.images, { id: crypto.randomUUID(), name: `image ${n}`}],
        });
    };

    return (
        <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 py-10">
            <div className="max-w-5xl mx-auto flex items-center gap-4 mb-8">
                <button
                    onClick={onBack}
                    className="text-white text-2xl hover:opacity-70 transition-opacity cursor-pointer">
                    ←
                </button>
                <h1 className="text-4xl font-bold italic text-white">{project.name}</h1>
                <button
                    onClick={addImage}
                    className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer">
                    + new image
                </button>
            </div>

            <div className="max-w-5xl mx-auto">
                <h2 className="text-3xl font-bold italic text-white mb-6">images</h2>
                {project.images.length === 0 ? (
                    <p className="text-white/70 text-sm">no images yet</p>
                ) : (
                    <div className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-4">
                        {project.images.map((img) => (
                            <div key={img.id} className="flex flex-col gap-2">
                                <div className="aspect-square bg-[#C8E6E3]/40 rounded-xl" />
                                <div className="flex items-center justify-between gap-1">
                                    <span className="text-sm text-white truncate">{img.name}</span>
                                    <button
                                        onClick={(e) => setImageMenu({ id: img.id, x: e.clientX, y: e.clientY })}
                                        className="text-white text-lg leading-none hover:opacity-70 cursor-pointer flex-shrink-0">
                                        ⋮
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {imageMenu && (
                <>
                    <div className="fixed inset-0 z-40" onClick={() => setImageMenu(null)} />
                    <div 
                        className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
                        style={{ top: imageMenu.y + 8, left: imageMenu.x - 80 }}
                    >
                        <button className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer">
                            Use Image
                        </button>
                        <button 
                            onClick={() => deleteImage(imageMenu.id)}
                            className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer">
                            Delete Image
                        </button>
                        <button
                            onClick={() => {
                                const img = project.images.find((i) => i.id === imageMenu.id)!;
                                setRenameModal({ id: imageMenu.id });
                                setRenameName(img.name);
                                setImageMenu(null);
                            }}
                            className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer">
                            Rename Image
                        </button>
                    </div>
                </>
            )}

            {renameModal && (
                <>
                    <div className="fixed inset-0 z-40" onClick={() => setRenameModal(null)} />
          <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-4 relative shadow-2xl">
            <button
              onClick={() => setRenameModal(null)}
              className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer"
            >
              ✕
            </button>
            <h2 className="text-xl text-[#1D3335] text-center">rename image</h2>
            <input
              autoFocus
              value={renameName}
              onChange={(e) => setRenameName(e.target.value)}
              onKeyDown={(e) => { if (e.key === 'Enter') renameImage(); }}
              className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm"
            />
            <button
              onClick={renameImage}
              className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
            >
              rename image
            </button>
          </div>
                </>
        )}
        </div>
    );
        
}