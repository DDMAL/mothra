import type { MeiFile, StaveSource } from "../../types";
import { downloadBlob } from "../../utils/download";
import TruncatedName from "../shared/TruncatedName";

interface Props {
  file: MeiFile;
  onClose: () => void;
}

// Which of tasks_encode.py's 3-tier fallback produced this MEI's zones --
// real detected geometry vs. a fallback vs. a true placeholder. Badge color
// mirrors StafflineViewerModal's anomaly-flag language (blue = worth a
// second look), reserving the strongest color for the one case that means
// "no real stave geometry at all".
const STAVE_SOURCE_LABELS: Record<
  StaveSource,
  { label: string; className: string }
> = {
  staffline_detection: {
    label: "real staffline detection",
    className: "text-[#1E6B70]",
  },
  yolo_annotation: {
    label: "YOLO annotation geometry",
    className: "text-[#1E6B70]",
  },
  glyph_estimate: {
    label: "estimated from glyphs",
    className: "text-[#1D3335]/60",
  },
  glyph_estimate_unresolved_lines: {
    label: "estimated -- pitch unresolved",
    className: "text-[#2563EB]",
  },
  glyph_estimate_synthetic_lines: {
    label: "estimated -- synthetic lines",
    className: "text-[#2563EB]",
  },
  placeholder_no_glyphs: {
    label: "placeholder (no glyphs found)",
    className: "text-[#FF3B30]",
  },
};

export default function MeiViewerModal({ file, onClose }: Props) {
  const handleExport = () => {
    downloadBlob(
      new Blob([file.xmlContent ?? ""], { type: "application/xml" }),
      file.name,
    );
  };
  const staveSourceInfo = file.staveSource
    ? STAVE_SOURCE_LABELS[file.staveSource]
    : null;

  return (
    <>
      <div
        className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60"
        onClick={onClose}
      />
      <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
        {/* header */}
        <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
          <TruncatedName
            name={file.name}
            className="font-mono text-sm text-[#1D3335] font-semibold flex-1 min-w-0"
          />
          {staveSourceInfo && (
            <span
              className={`text-xs font-semibold whitespace-nowrap ${staveSourceInfo.className}`}
            >
              {staveSourceInfo.label}
            </span>
          )}
          <button
            onClick={handleExport}
            className="px-4 py-1.5 bg-white text-[#1D3335] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
          >
            export
          </button>
          <button
            onClick={onClose}
            className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
          >
            ✕
          </button>
        </div>

        {/* content */}
        <div className="flex-1 min-h-0 overflow-auto p-6">
          <p className="text-[#1D3335]/60 text-xs mb-3 italic">
            Proceed to correction to visualize and correct this file.
          </p>
          <pre className="text-[#1D3335]/80 text-xs font-mono whitespace-pre-wrap">
            {file.xmlContent ?? "(no content)"}
          </pre>
        </div>
      </div>
    </>
  );
}
