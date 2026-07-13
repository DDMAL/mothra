import { useMemo, useState } from "react";
import type { Project } from "../../types";
import FileDropZone from "../shared/FileDropZone";

interface BatchImage {
  id: string;
  name: string;
}

interface BatchTabProps {
  project: Project;
  cantusFolios: string[];
  onUploadImage: (file: File, folio?: string) => Promise<{ id: string; name: string; folio?: string }>;
  onUpdateProject: (p: Project) => void;
  onRunBatch: (imageIds: string[], folios: string[]) => void;
}

export default function BatchTab({
  project, cantusFolios, onUploadImage, onUpdateProject, onRunBatch,
}: BatchTabProps) {
  const folios = cantusFolios;
  const [startFolio, setStartFolio] = useState("");
  const [endFolio, setEndFolio] = useState("");
  const [uploaded, setUploaded] = useState<BatchImage[]>([]);
  const [uploading, setUploading] = useState(false);

  const folioSequence = useMemo(() => {
    if (!startFolio || !endFolio) return [];
    const startIdx = folios.indexOf(startFolio);
    const endIdx = folios.indexOf(endFolio);
    if (startIdx === -1 || endIdx === -1 || startIdx > endIdx) return [];
    return folios.slice(startIdx, endIdx + 1);
  }, [folios, startFolio, endFolio]);

  const countMismatch = uploaded.length > 0 && uploaded.length !== folioSequence.length;

  const handleFiles = async (files: FileList | File[]) => {
    setUploading(true);
    const entries: BatchImage[] = [];
    for (const f of Array.from(files)) {
      const r = await onUploadImage(f);
      entries.push({ id: r.id, name: r.name });
    }
    setUploaded((prev) => [...prev, ...entries]);
    onUpdateProject({ ...project, images: [...project.images, ...entries.map((e) => ({ ...e }))] });
    setUploading(false);
  };

  const move = (i: number, dir: -1 | 1) => {
    setUploaded((prev) => {
      const next = [...prev];
      const j = i + dir;
      if (j < 0 || j >= next.length) return prev;
      [next[i], next[j]] = [next[j], next[i]];
      return next;
    });
  };

  if (folios.length === 0) {
    return <p className="text-white/70 text-sm">load a Cantus source above to set up a batch run</p>;
  }

  return (
    <div className="flex flex-col gap-4 text-white">
      <div className="flex gap-4">
        <label className="flex flex-col gap-1">
          <span className="text-xs text-white/70">start folio</span>
          <select value={startFolio} onChange={(e) => setStartFolio(e.target.value)}
            className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm">
            <option value="">select...</option>
            {folios.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1">
          <span className="text-xs text-white/70">end folio</span>
          <select value={endFolio} onChange={(e) => setEndFolio(e.target.value)}
            className="bg-[#1D3335] border border-white/30 rounded px-2 py-1 text-sm">
            <option value="">select...</option>
            {folios.map((f) => <option key={f} value={f}>{f}</option>)}
          </select>
        </label>
      </div>
      {folioSequence.length > 0 && (
        <p className="text-sm text-white/70">{folioSequence.length} folio(s) in this range</p>
      )}

      <FileDropZone
        dragging={false}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => { e.preventDefault(); handleFiles(e.dataTransfer.files); }}
        onClick={() => document.getElementById("batch-file-input")?.click()}
        label={uploading ? "uploading..." : "drag & drop images here, in manuscript order"}
      >
        <input id="batch-file-input" type="file" multiple accept="image/*" className="hidden"
          onChange={(e) => { if (e.target.files) handleFiles(e.target.files); }} />
      </FileDropZone>

      {uploaded.length > 0 && (
        <ol className="flex flex-col gap-1">
          {uploaded.map((img, i) => (
            <li key={img.id} className="flex items-center gap-2 bg-white/10 rounded px-3 py-1 text-sm">
              <span className="text-white/50 w-6">{i + 1}</span>
              <span className="flex-1 truncate">{img.name}</span>
              <span className="text-white/50">{folioSequence[i] ?? "—"}</span>
              <button onClick={() => move(i, -1)} disabled={i === 0} className="disabled:opacity-30 cursor-pointer">↑</button>
              <button onClick={() => move(i, 1)} disabled={i === uploaded.length - 1} className="disabled:opacity-30 cursor-pointer">↓</button>
            </li>
          ))}
        </ol>
      )}

      {countMismatch && (
        <p className="text-red-200 text-xs">
          {uploaded.length} image(s) uploaded but {folioSequence.length} folio(s) selected — counts must match.
        </p>
      )}

      <button
        onClick={() => onRunBatch(uploaded.map((u) => u.id), folioSequence)}
        disabled={countMismatch || uploaded.length === 0 || folioSequence.length === 0}
        className="px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer w-fit"
      >
        run batch ({folioSequence.length} folios)
      </button>
    </div>
  );
}
