import Modal from "../shared/Modal";
import TruncatedName from "../shared/TruncatedName";

interface LargeImageWarningModalProps {
    oversizedFiles: File[];
    resizing: boolean;
    onResize: () => void;
    onUploadAsIs: () => void;
    onCancel: () => void;
}

const formatBytes = (b: number) => 
    b < 1024
        ? `${b} B`
        : b < 1024 ** 2
            ? `${(b / 1024).toFixed(1)} KB`
            : `${(b / 1024 ** 2).toFixed(2)} MB`;

export default function LargeImageWarningModal({
    oversizedFiles,
    resizing,
    onResize,
    onUploadAsIs,
    onCancel,
}: LargeImageWarningModalProps) {
    const plural = oversizedFiles.length > 1;
    return (
        <Modal onClose={resizing ? undefined : onCancel} size="2xl" backdrop="dark">
        <h2 className="text-xl text-[#1D3335] text-center">
            large image{plural ? "s" : ""} detected
        </h2>
        <p className="text-sm text-[#1D3335]/70 text-center -mt-2">
            {plural
            ? `${oversizedFiles.length} of these images are`
            : "this image is"}{" "}
            larger than 5&nbsp;MB. large images can slow down processing and may
            reduce model performance. you can resize {plural ? "them" : "it"} to
            roughly 2&nbsp;MB before uploading, or upload as-is.
        </p>
        <div className="max-h-[40vh] overflow-y-auto rounded-xl bg-white/40 divide-y divide-[#1D3335]/10">
            <div className="grid grid-cols-2 gap-2 px-3 py-2 text-xs font-semibold text-[#1D3335]/70 sticky top-0 bg-white/60 backdrop-blur">
            <span>file</span>
            <span>size</span>
            </div>
            {oversizedFiles.map((f, i) => (
            <div
                key={`${f.name}-${i}`}
                className="grid grid-cols-2 gap-2 px-3 py-2 text-xs text-[#1D3335] font-mono items-center"
            >
                <TruncatedName name={f.name} className="min-w-0" />
                <span>{formatBytes(f.size)}</span>
            </div>
            ))}
        </div>
        <div className="flex items-center justify-center gap-3">
            <button
            onClick={onResize}
            disabled={resizing}
            className="px-5 py-2 bg-[#4AADAA] text-white font-semibold rounded-xl hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed text-sm"
            >
            {resizing ? "resizing..." : "resize images"}
            </button>
            <button
            onClick={onUploadAsIs}
            disabled={resizing}
            className="px-5 py-2 border-2 border-[#1D3335]/30 text-[#1D3335] font-semibold rounded-xl hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed text-sm"
            >
            upload as-is
            </button>
            <button
            onClick={onCancel}
            disabled={resizing}
            className="text-[#1D3335]/60 text-sm hover:text-[#1D3335] cursor-pointer underline disabled:opacity-40"
            >
            cancel
            </button>
        </div>
        </Modal>
    );
}