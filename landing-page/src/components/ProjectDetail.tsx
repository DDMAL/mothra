import { useEffect, useRef, useState } from 'react';
import type { Project } from '../App'
import * as pdfjsLib from 'pdfjs-dist';
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
    'pdfjs-dist/build/pdf.worker.min.mjs',
    import.meta.url,
).href;


interface ProjectDetailProps {
    project: Project;
    onBack: () => void;
    onUpdateProject: (updated: Project) => void;
}

export default function ProjectDetail({ project, onBack, onUpdateProject }: ProjectDetailProps) {
    const [imageMenu, setImageMenu] = useState<{id: string; x: number; y: number } | null>(null);
    const [renameModal, setRenameModal] = useState<{id: string} | null>(null);
    const [renameName, setRenameName] = useState('');
    const [uploadModal, setUploadModal] = useState(false);
    const [dragging, setDragging] = useState(false);
    const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
    const [lastSelectedIdx, setLastSelectedIdx] = useState<number | null>(null);
    const [converting, setConverting] = useState(false);
    const fileInputRef = useRef<HTMLInputElement>(null);
    const folderInputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        const handler = (e: KeyboardEvent) => {
            if (e.key === 'Escape') {
                setImageMenu(null);
                setRenameModal(null);
                setSelectedIds(new Set());
                setLastSelectedIdx(null);
            }
            if (e.key === 'Delete' && selectedIds.size > 0) {
                onUpdateProject({ ...project, images: project.images.filter(img => !selectedIds.has(img.id)) });
                setSelectedIds(new Set());
                setLastSelectedIdx(null);
            }
        };
        window.addEventListener('keydown', handler);
        return () => window.removeEventListener('keydown', handler);
    }, [selectedIds, project, onUpdateProject]);

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

    const pdfToImages = async (file: File): Promise<{ name: string; src: string }[]> => {
        const baseName = file.name.replace(/\.pdf$/i, '');
        const pdf = await pdfjsLib.getDocument({ data: await file.arrayBuffer() }).promise;
        const results: { name: string; src: string }[] = [];
        for (let i = 1; i <= pdf.numPages; i++) {
            const page = await pdf.getPage(i);
            const viewport = page.getViewport({ scale: 1.5 });
            const canvas = document.createElement('canvas');
            canvas.width = viewport.width;
            canvas.height = viewport.height;
            await page.render({ canvasContext: canvas.getContext('2d')!, canvas, viewport }).promise;
            const blob = await new Promise<Blob>(res => canvas.toBlob(b => res(b!), 'image/png'));
            results.push({ name: `${baseName} (page${i}).png`, src: URL.createObjectURL(blob) });
        }
        return results;
    };

    const handleFiles = async (files: FileList | File[]) => {
        const all = Array.from(files);
        const imageFiles = all.filter(f => f.type.startsWith('image/'));
        const pdfFiles = all.filter(f => f.type === 'application/pdf');
        if (imageFiles.length === 0 && pdfFiles.length === 0) return;

        setConverting(true);

        const imageEntries = imageFiles.map(f => ({
            id: crypto.randomUUID(),
            name: f.name,
            src: URL.createObjectURL(f),
        }));
        
        const pdfEntries = (await Promise.all(pdfFiles.map(pdfToImages)))
            .flat()
            .map(({ name, src }) => ({ id: crypto.randomUUID(), name, src }));


        onUpdateProject({
            ...project,
            images: [ ...project.images, ...imageEntries, ...pdfEntries ]
        });
        setConverting(false);
        setUploadModal(false);
        setDragging(false);
    };

    const handleImageClick = (e: React.MouseEvent, id: string, idx: number) => {
        if (e.shiftKey) {
            e.preventDefault();
            setSelectedIds(prev => {
                const next = new Set(prev);
                if (lastSelectedIdx !== null && lastSelectedIdx !== idx) {
                    const lo = Math.min(lastSelectedIdx, idx);
                    const hi = Math.max(lastSelectedIdx, idx);
                    project.images.slice(lo, hi + 1).forEach(img => next.add(img.id));
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
                            ${(uploadModal || !!renameModal || !!imageMenu) ? 'opacity-100' : 'opacity-0'}`} />
            <div className="max-w-5xl mx-auto flex items-center gap-4 mb-8">
                <button
                    onClick={onBack}
                    className="text-white text-2xl hover:opacity-70 transition-opacity cursor-pointer">
                    ←
                </button>
                <h1 className="text-4xl font-bold italic text-white">{project.name}</h1>
                <button
                    onClick={() => setUploadModal(true)}
                    className="ml-4 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer">
                    + new image
                </button>
                {selectedIds.size > 0 && (
                    <>
                        <button
                            className="ml-2 px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20">
                            use {selectedIds.size} image{selectedIds.size > 1 ? 's' : ''}
                        </button>
                        <button
                            onClick={() => {
                                onUpdateProject({ ...project, images: project.images.filter(img => !selectedIds.has(img.id)) });
                                setSelectedIds(new Set());
                                setLastSelectedIdx(null);
                            }}
                            className="px-5 border-2 border-white text-white text-sm rounded-full hover:opacity-90 cursor-pointer bg-white/20">
                            delete {selectedIds.size} image{selectedIds.size > 1 ? 's' : ''}
                        </button>
                    </>
                )}
            </div>

            <div className="max-w-5xl mx-auto">
                <h2 className="text-3xl font-bold italic text-white mb-6">images</h2>
                {project.images.length === 0 ? (
                    <p className="text-white/70 text-sm">no images yet</p>
                ) : (
                    <div
                        className="grid grid-cols-[repeat(auto-fill,minmax(160px,1fr))] gap-4"
                        onMouseDown={(e) => { if (e.shiftKey) e.preventDefault(); }}>
                        {project.images.map((img, idx) => (
                            <div key={img.id} className="flex flex-col gap-2">
                                <div
                                    className={`aspect-square bg-[#C8E6E3]/40 rounded-xl overflow-hidden cursor-pointer
                                                transition-shadow
                                                ${selectedIds.has(img.id) ? 'ring-4 ring-white ring-offset-2 ring-offset-[#4AADAA]' : ''}`}
                                    onClick={(e) => handleImageClick(e, img.id, idx)}>
                                    {img.src && <img src={img.src} alt={img.name} className="w-full h-full object-cover" />}
                                </div>
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

        {uploadModal && (
            <>
                <div className="fixed inset-0 z-40" onClick={() => {setUploadModal(false); setDragging(false); }} />
                {/* dialog */}

                <div className="animate-fade-in fixed z-50 top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2
                                w-full max-w-lg bg-[#C8E6E3] rounded-3xl p-8 flex flex-col gap-6 relative shadow-2xl">

                    <button
                        onClick={() => { if (!converting) {setUploadModal(false); setDragging(false); } }}
                        className="absolute top-4 right-5 text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer">
                        x
                    </button>

                    <h2 className="text-xl text-[#1D3335] text-center">upload image</h2>

                    {/* drop zone */}
                    { converting ? (
                        <div className="flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
                                        border-[#1D3335]/30 bg-white/40 py-12">
                            <p className="text-sm text-[#1D3335] text-center"> converting PDF pages... </p>
                        </div>                    
                    ) : (
                        <div
                        onClick={() => fileInputRef.current?.click()}
                        onDragOver={(e) => {e.preventDefault(); setDragging(true); }}
                        onDragEnter={(e) => {e.preventDefault(); setDragging(true); }}
                        onDragLeave={() => setDragging(false)}
                        onDrop={(e) => {
                            e.preventDefault();
                            handleFiles(e.dataTransfer.files);
                        }}
                        className={`flex flex-col items-center justify-center gap-3 rounded-2xl border-2 border-dashed
                            py-12 cursor-pointer transition-colors
                            ${dragging
                                ? 'border-[#1E6B70] bg-[#1E6B70]/10'
                                : 'border-[#1D3335]/30 bg-white/40 hover:bg-white/60'}`}>
                        <span className="text-3xl">↑</span>
                        <p className="text-sm text-[#1D3335] text-center">
                            drag & drop images, folders, or PDFs here
                        </p>
                        <div className="flex gap-4 text-sm text-[#1D3335]">
                            <button
                                onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click(); }}
                                className="underline hover:opacity-70 cursor-pointer">
                                select files
                            </button>
                            <span className="text-[#1D3335]/40">or</span>
                            <button
                                onClick={(e) => { e.stopPropagation(); folderInputRef.current?.click(); }}
                                className="underline hover:opacity-70 cursor-pointer">
                                select folder
                            </button>
                    </div>
                    </div>

                )}
                    

                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="image/*,application/pdf"
                        multiple
                        className="hidden"
                        onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}/>
                    <input 
                        ref={folderInputRef}
                        type="file"
                        // @ts-expect-error
                        webkitdirectory=""
                        className="hidden"
                        onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }}/>
                </div>
            </>
        )}
        </div>
    );
        
}