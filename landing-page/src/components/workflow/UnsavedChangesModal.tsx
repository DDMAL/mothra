import Modal from "../shared/Modal";

interface UnsavedChangesModalProps {
  onConfirm: () => void;
  onCancel: () => void;
}

export default function UnsavedChangesModal({
  onConfirm,
  onCancel,
}: UnsavedChangesModalProps) {
  return (
    <Modal onClose={onCancel} backdrop="dim">
      <h2 className="text-xl text-[#1D3335] font-semibold text-center">
        leave without saving?
      </h2>
      <p className="text-sm text-[#1D3335] text-center leading-relaxed">
        this file has unsaved changes that will be lost if you continue --
        press <span className="font-semibold">Ctrl/Cmd+Enter</span> (or the
        "Mark Done" button) first if you want to keep your edits.
      </p>
      <div className="flex gap-3 justify-center">
        <button
          onClick={onConfirm}
          className="px-6 py-2.5 bg-[#1E6B70] text-white font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
        >
          leave anyway
        </button>
        <button
          onClick={onCancel}
          className="px-6 py-2.5 border-2 border-[#1D3335]/30 text-[#1D3335] font-semibold rounded-xl hover:opacity-70 cursor-pointer text-sm"
        >
          cancel
        </button>
      </div>
    </Modal>
  );
}
