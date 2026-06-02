import { useEffect, useRef, useState } from 'react';
import type { ProjectModel } from '../App';

interface MyModelsProps {
    models: ProjectModel[];
    onUpdateModels: (models: ProjectModel[]) => void;
}

export default function MyModels({ models, onUpdateModels }: MyModelsProps) {
    const [modelMenu, setModelMenu] = useState<{id: string; x: number; y: number} | null>(null);
    const [renameModal, setRenameModal] = useState<{id: string} | null>(null);
    const [renameName, setRenameName] = useState('');
    const [uploadModal, setUploadModal] = useState(false);
    const [dragging, setDragging] = useState(false);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [lastSelectedIdx, setLastSelectedIdx] = useState<number | null>(null);
    const fileInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                setModelMenu(null);
                setRenameModal(null);
                setSelectedIds(new Set());
                setLastSelectedIdx(null);
            }
            if (e.key === 'Delete' && selectedIds.size > 0) {
                onUpdateModels(models.filter(m => !selectedIds.has(m.id)));
                setSelectedIds(new Set());
                setLastSelectedIdx(null);
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [selectedIds, models, onUpdateModels]);

    const deleteModel = (id: string) => {
        onUpdateModels(models.filter(m => m.id !== id));
        setModelMenu(null);
    };

    const renameModel = () => {
        const current = models.find(m => m.id === renameModal?.id);
        onUpdateModels(models.map(m =>
            m.id === renameModal?.id ? { ...m, name: renameName.trim() || current!.name } : m
        ));
        setRenameModal(null);
    };

    const handleFiles = (files: FileList | File[]) => {
        const valid = Array.from(files).filter(f => /\.(h5|hdf5)$/i.test(f.name));
        if (valid.length === 0) return;
        const entries = valid.map(f => ({ id: crypto.randomUUID(), name: f.name }));
        onUpdateModels([...models, ...entries]);
        setUploadModal(false);
        setDragging(false);
    };

    const handleModelClick = (e: React.MouseEvent, id: string, idx: number) => {
        if (e.shiftKey) {
            e.preventDefault();
            setSelectedIds(prev => {
                const next = new Set(prev);
                if (lastSelectedIdx !== null && lastSelectedIdx !== idx) {
                    const lo = Math.min(lastSelectedIdx, idx);
                    const hi = Math.max(lastSelectedIdx, idx);
                    models.slice(lo, hi + 1).forEach(m => next.add(m.id));
                } else {
                    next.has(id) ? next.delete(id) : next.add(id);
                }
                return next;
            });
            setLastSelectedIdx(idx);
        } else {
            if (selectedIds.has(id)) {
                setSelectedIds(prev => { const next = new Set(prev); next.delete(id); return next; });
            } else {
                setSelectedIds(new Set([id]));
                setLastSelectedIdx(idx);
            }
        }
    };

    return (
        <div className="animate-fade-in flex-1 bg-[#4AADAA] px-6 pt-10 pb-48 relative">
            <div className={`absolute inset-0 z-30 bg-black/30 transition-opacity pointer-events-none
                            ${(uploadModal || !!renameModal) ? 'opacity-100' : 'opacity-0'}`} />

            <div className="max-w-5xl mx-auto flex items-center gap-4 mb-8">
                <h1 className="text-4xl font-bold italic text-white">My Models</h1>
                <button
                    onClick={() => setUploadModal(true)}
                    className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer">
                    + upload model
                </button>
                {selectedIds.size > 0 && (
                    <>
                        <button className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20">
                            use {selectedIds.size} model{selectedIds.size > 1 ? 's' : ''}
                        </button>
                        <button
                            onClick={() => {
                                onUpdateModels(models.filter(m => !selectedIds.has(m.id)));
                                setSelectedIds(new Set());
                                setLastSelectedIdx(null);
                            }}
                            className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20">
                            delete {selectedIds.size} model{selectedIds.size > 1 ? 's' : ''}
                        </button>
                    </>
                )}
            </div>

            <div className="max-w-5xl mx-auto">
                {models.length === 0 ? (
                    <p className="text-white/70 text-sm">no models yet</p>
                ) : (
                    <div
                        className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-4"
                        onMouseDown={(e) => { if (e.shiftKey) e.preventDefault(); }}>
                        {models.map((model, idx) => (
                            <div key={model.id} className="flex flex-col gap-2">
                                <div
                                    className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer
                                                transition-shadow flex items-center justify-center
                                                ${selectedIds.has(model.id) ? 'ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]' : ''}`}
                                    onClick={(e) => handleModelClick(e, model.id, idx)}>
                                    <svg width="56" height="64" viewBox="0 0 56 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                                        <path d="M4 0H36L56 20V60C56 62.2 54.2 64 52 64H4C1.8 64 0 62.2 0 60V4C0 1.8 1.8 0 4 0Z"
                                              fill="white" fillOpacity="0.25"/>
                                        <path d="M36 0L56 20H40C37.8 20 36 18.2 36 16V0Z"
                                              fill="white" fillOpacity="0.45"/>
                                        <text x="28" y="46" textAnchor="middle" fill="white"
                                              fontSize="16" fontWeight="bold" fontFamily="monospace">H5</text>
                                    </svg>
                                </div>
                                <div className="flex items-center justify-between gap-1">
                                    <span className="text-sm text-white truncate">{model.name}</span>
                                    <button
                                        onClick={(e) => setModelMenu({ id: model.id, x: e.clientX, y: e.clientY })}
                                        className="text-white text-lg leading-none hover:opacity-70 cursor-pointer flex-shrink-0">
                                        ⋮
                                    </button>
                                </div>
                            </div>
                        ))}
                    </div>
                )}
            </div>

            {modelMenu && (
                <>
                    <div className="fixed inset-0 z-40" onClick={() => setModelMenu(null)} />
                    <div
                        className="fixed z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[160px]"
                        style={{ top: modelMenu.y + 8, left: modelMenu.x - 80 }}>
                        <button className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer">
                            Use Model
                        </button>
                        <button
                            onClick={() => deleteModel(modelMenu.id)}
                            className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer">
                            Delete Model
                        </button>
                        <button
                            onClick={() => {
                                const m = models.find(m => m.id === modelMenu.id)!;
                                setRenameModal({ id: modelMenu.id });
                                setRenameName(m.name);
                                setModelMenu(null);
                            }}
                            className="text-sm text-[#1D3335] text-left px-2 py-1.5 hover:opacity-70 cursor-pointer">
                            Rename Model
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
                            className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer">
                            ✕
                        </button>
                        <h2 className="text-xl text-[#1D3335] text-center">rename model</h2>
                        <input
                            autoFocus
                            value={renameName}
                            onChange={(e) => setRenameName(e.target.value)}
                            onKeyDown={(e) => { if (e.key === 'Enter') renameModel(); }}
                            className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm"
                        />
                        <button
                            onClick={renameModel}
                            className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer">
                            rename model
                        </button>
                    </div>
                </>
            )}

            {uploadModal && (
                <div
                    className="fixed inset-0 z-[60] flex items-center justify-center p-6"
                    onClick={() => { setUploadModal(false); setDragging(false); }}>
                    <div
                        className="w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-6 relative shadow-2xl"
                        onClick={(e) => e.stopPropagation()}>
                        <button
                            onClick={() => { setUploadModal(false); setDragging(false); }}
                            className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer">
                            x
                        </button>
                        <h2 className="text-xl text-[#1D3335] text-center">upload model</h2>
                        <div
                            onClick={() => fileInputRef.current?.click()}
                            onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
                            onDragEnter={(e) => { e.preventDefault(); setDragging(true); }}
                            onDragLeave={() => setDragging(false)}
                            onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); }}
                            className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
                                py-12 cursor-pointer transition-colors
                                ${dragging ? 'border-[#1E6B70] bg-[#1E6B70]/10' : 'border-[#1D3335]/30 bg-white/40 hover:bg-white/60'}`}>
                            <span className="text-3xl">↑</span>
                            <p className="text-sm text-[#1D3335] text-center">
                                drag & drop .h5 or .hdf5 files here
                            </p>
                            <button
                                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                                className="text-sm text-[#1D3335] underline hover:opacity-70 cursor-pointer">
                                select files
                            </button>
                        </div>
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".h5,.hdf5"
                            multiple
                            className="hidden"
                            onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}
                        />
                    </div>
                </div>
            )}
        </div>
    );
}