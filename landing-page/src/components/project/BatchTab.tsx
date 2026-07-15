import { useEffect, useState } from "react";
import { AuthImage } from "../shared/AuthImage";
import QuickLookModal from "../shared/QuickLookModal";

interface BatchImage {
  id: string;
  name: string;
}

interface BatchTabProps {
  batchImages: BatchImage[];
  folioSequence: string[];
  onUseBatch: (names: string[]) => void;
  onDiscardBatch: (imageIds: string[]) => void;
}

export default function BatchTab({ batchImages, folioSequence, onUseBatch, onDiscardBatch }: BatchTabProps) {
  const [index, setIndex] = useState(0);
  const [expanded, setExpanded] = useState(false);
  const [confirmDiscard, setConfirmDiscard] = useState(false);

  useEffect(() => {
    setIndex((i) => Math.min(i, Math.max(0, batchImages.length - 1)));
  }, [batchImages.length]);

  useEffect(() => {
    if (!expanded) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "ArrowLeft") setIndex((i) => Math.max(0, i - 1));
      if (e.key === "ArrowRight") setIndex((i) => Math.min(batchImages.length - 1, i + 1));
      if (e.key === "Escape") setExpanded(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [expanded, batchImages.length]);

  const countMismatch = batchImages.length > 0 && batchImages.length !== folioSequence.length;

  if (batchImages.length === 0) {
    return (
      <p className="text-white/70 text-sm">
        select a folio range above, then use "+ new image" to upload
      </p>
    );
  }

  const current = batchImages[index];

  return (
    <div className="flex flex-col gap-4 text-white">
      <div className="relative w-48">
        <button onClick={() => setExpanded(true)} className="block cursor-pointer">
          <AuthImage
            src={`/api/images/${current.id}`}
            alt={current.name}
            className="w-48 h-64 object-cover rounded-lg border border-white/20"
          />
        </button>
        <span className="absolute bottom-1 left-1 bg-[#1D3335]/80 text-white text-[10px] font-mono px-1.5 py-0.5 rounded">
          {index + 1}/{batchImages.length} — {folioSequence[index] ?? "—"}
        </span>
      </div>

      {countMismatch && (
        <p className="text-red-200 text-xs">
          {batchImages.length} image(s) uploaded but {folioSequence.length} folio(s) selected — counts must match.
        </p>
      )}

      {!confirmDiscard ? (
        <div className="flex items-center gap-3">
          <button
            onClick={() => onUseBatch(batchImages.map((b) => b.name))}
            disabled={countMismatch || batchImages.length === 0 || folioSequence.length === 0}
            className="px-5 py-2 bg-white text-[#4AADAA] font-semibold rounded-xl disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer w-fit"
          >
            use batch ({folioSequence.length} folios)
          </button>
          <button
            onClick={() => setConfirmDiscard(true)}
            className="px-5 py-2 border-2 border-red-300/60 text-red-100 font-semibold rounded-xl hover:bg-red-500/10 cursor-pointer w-fit"
          >
            discard batch
          </button>
        </div>
      ) : (
        <div className="flex items-center gap-3 text-sm text-white">
          <span>discard {batchImages.length} uploaded image(s)? this can't be undone.</span>
          <button
            onClick={() => {
              onDiscardBatch(batchImages.map((b) => b.id));
              setConfirmDiscard(false);
            }}
            className="px-3 py-1 bg-white text-[#4AADAA] rounded-lg font-semibold hover:opacity-90 cursor-pointer"
          >
            yes
          </button>
          <button
            onClick={() => setConfirmDiscard(false)}
            className="px-3 py-1 border border-white/40 text-white rounded-lg hover:opacity-90 cursor-pointer"
          >
            no
          </button>
        </div>
      )}

      {expanded && (
        <QuickLookModal onClose={() => setExpanded(false)}>
          <div className="flex items-center gap-4">
            <button
              onClick={() => setIndex((i) => Math.max(0, i - 1))}
              disabled={index === 0}
              className="text-white text-2xl disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
            >
              ‹
            </button>
            <div className="flex flex-col items-center gap-2">
              <AuthImage
                src={`/api/images/${current.id}`}
                alt={current.name}
                className="max-h-[70vh] max-w-full object-contain rounded-lg"
              />
              <p className="text-white/70 text-xs">
                {index + 1} / {batchImages.length} — folio {folioSequence[index] ?? "—"} — {current.name}
              </p>
            </div>
            <button
              onClick={() => setIndex((i) => Math.min(batchImages.length - 1, i + 1))}
              disabled={index === batchImages.length - 1}
              className="text-white text-2xl disabled:opacity-30 cursor-pointer disabled:cursor-not-allowed"
            >
              ›
            </button>
          </div>
        </QuickLookModal>
      )}
    </div>
  );
}
