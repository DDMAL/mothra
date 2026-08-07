import type { MeiFile } from "../../types";
import { downloadBlob } from "../../utils/download";
import TruncatedName from "../shared/TruncatedName";

interface Props {
  file: MeiFile;
  onClose: () => void;
}

export default function MeiViewerModal({ file, onClose }: Props) {
  const handleExport = () => {
    downloadBlob(
      new Blob([file.xmlContent ?? ""], { type: "application/xml" }),
      file.name,
    );
  };

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
