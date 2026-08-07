import Modal from "../shared/Modal";
import type { ProjectImage } from "../../types";
import { findFolioConflict } from "../../utils/folio";

interface EditFolioModalProps {
  image: ProjectImage;
  images: ProjectImage[];
  folioOptions: string[];
  value: string;
  onChange: (v: string) => void;
  onSubmit: () => void;
  onClose: () => void;
}

export default function EditFolioModal({
  image,
  images,
  folioOptions,
  value,
  onChange,
  onSubmit,
  onClose,
}: EditFolioModalProps) {
  const conflict = findFolioConflict(images, value, image.id);
  return (
    <Modal onClose={onClose}>
      <h2 className="text-xl text-[#1D3335] text-center">
        edit folio — {image.name}
      </h2>
      <select
        autoFocus
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="bg-white rounded-2xl px-6 py-3 text-center text-[#1D3335] outline-none text-sm"
      >
        <option value="">no folio</option>
        {folioOptions.map((f) => (
          <option key={f} value={f}>
            {f}
          </option>
        ))}
      </select>
      {conflict && (
        <p className="text-red-600 text-xs text-center">
          ⚠ folio "{value}" is already used by {conflict.name}
        </p>
      )}
      <button
        onClick={onSubmit}
        className="bg-[#1E6B70] text-white rounded-xl px-6 py-3 text-sm font-bold self-center hover:opacity-90 transition-opacity cursor-pointer"
      >
        save folio
      </button>
    </Modal>
  );
}
