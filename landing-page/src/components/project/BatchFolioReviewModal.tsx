import Modal from "../shared/Modal";
import type { FolioReviewRow, FolioReviewStatus } from "../../utils/folio";

interface BatchFolioReviewModalProps {
    rows: FolioReviewRow[];
    canUseDetected: boolean;
    onUseDetected: () => void;
    onUsePositional: () => void;
    onCancel: () => void;
}

const STATUS_LABEL: Record<FolioReviewStatus, string> = {
    match: "✓ matches",
    "no-detection": "— no folio in filename",
    mismatch: "⚠ different position",
    "not-in-source": "⚠ not in Cantus source",
    duplicate: "⚠ duplicate folio",
};

const STATUS_COLOR: Record<FolioReviewStatus, string> = {
  match: "text-white/50",
  "no-detection": "text-white/40",
  mismatch: "text-yellow-700",
  "not-in-source": "text-red-700",
  duplicate: "text-red-700",
};

export default function BatchFolioReviewModal({
  rows,
  canUseDetected,
  onUseDetected,
  onUsePositional,
  onCancel,
}: BatchFolioReviewModalProps) {
  return (
    <Modal onClose={onCancel} size="2xl" backdrop="dark">
      <h2 className="text-xl text-[#1D3335] text-center">check folio assignment</h2>
      <p className="text-sm text-[#1D3335]/70 text-center -mt-2">
        some filenames suggest a different folio than upload order would assign —
        review before running annotation/text-finding on this batch.
      </p>
      <div className="max-h-[50vh] overflow-y-auto rounded-xl bg-white/40 divide-y divide-[#1D3335]/10">
        <div className="grid grid-cols-4 gap-2 px-3 py-2 text-xs font-semibold text-[#1D3335]/70 sticky top-0 bg-white/60 backdrop-blur">
          <span>file</span>
          <span>would assign</span>
          <span>detected</span>
          <span>status</span>
        </div>
        {rows.map((row) => (
          <div
            key={row.fileName}
            className="grid grid-cols-4 gap-2 px-3 py-2 text-xs text-[#1D3335] font-mono items-center"
          >
            <span className="truncate" title={row.fileName}>{row.fileName}</span>
            <span>{row.positionalFolio ?? "—"}</span>
            <span>{row.detectedFolio ?? "—"}</span>
            <span className={STATUS_COLOR[row.status]}>{STATUS_LABEL[row.status]}</span>
          </div>
        ))}
      </div>
      <div className="flex items-center justify-center gap-3">
        <button
          onClick={onUseDetected}
          disabled={!canUseDetected}
          className="px-5 py-2 bg-[#4AADAA] text-white font-semibold rounded-xl hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed text-sm"
        >
          use detected folios
        </button>
        <button
          onClick={onUsePositional}
          className="px-5 py-2 border-2 border-[#1D3335]/30 text-[#1D3335] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
        >
          use upload order anyway
        </button>
        <button
          onClick={onCancel}
          className="text-[#1D3335]/60 text-sm hover:text-[#1D3335] cursor-pointer underline"
        >
          cancel
        </button>
      </div>
    </Modal>
  );
}