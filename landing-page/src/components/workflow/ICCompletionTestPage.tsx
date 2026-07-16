import { useRef, useState } from "react";

interface IcCompletionTestPageProps {
  onContinue: () => void;
  onBackToProject: () => void;
  xmlFile: File | null;
  onXmlFileChange: (f: File | null) => void;
  imageFile: File | null;
  onImageFileChange: (f: File | null) => void;
}

export default function IcCompletionTestPage({
  onContinue,
  onBackToProject,
  xmlFile,
  onXmlFileChange,
  imageFile,
  onImageFileChange,
}: IcCompletionTestPageProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [imageDragging, setImageDragging] = useState(false);

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48 relative">
      <div className="flex flex-col items-center gap-6">
        <h1 className="text-5xl font-bold italic text-white">ta-da!</h1>
        <p className="text-xl text-[#1D3335]">
          all images successfully classified!
        </p>
        <p className="text-sm text-white/70 -mt-3">progress saved</p>

        {/* xml upload drop zone */}
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) onXmlFileChange(file);
          }}
          className={`w-full max-w-sm border-2 border-dashed rounded-2xl px-6 py-4 flex flex-col items-center gap-1 cursor-pointer transition-colors
            ${dragging ? "border-white bg-white/10" : "border-white/40 hover:border-white/70"}`}
        >
          <span className="text-white/80 text-sm font-mono">
            {xmlFile ? xmlFile.name : "drop gamera xml here or click to browse"}
          </span>
          {xmlFile && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onXmlFileChange(null);
              }}
              className="text-white/40 text-xs hover:text-white/70 cursor-pointer"
            >
              × clear
            </button>
          )}
          <input
            ref={inputRef}
            type="file"
            accept=".xml"
            className="hidden"
            onChange={(e) => onXmlFileChange(e.target.files?.[0] ?? null)}
          />
        </div>

        {/* image upload drop zone (used to set correct surface bounds in MEI) */}
        <div
          onClick={() => imageInputRef.current?.click()}
          onDragOver={(e) => {
            e.preventDefault();
            setImageDragging(true);
          }}
          onDragLeave={() => setImageDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setImageDragging(false);
            const file = e.dataTransfer.files?.[0];
            if (file) onImageFileChange(file);
          }}
          className={`w-full max-w-sm border-2 border-dashed rounded-2xl px-6 py-4 flex flex-col items-center gap-1 cursor-pointer transition-colors
            ${imageDragging ? "border-white bg-white/10" : "border-white/40 hover:border-white/70"}`}
        >
          <span className="text-white/80 text-sm font-mono">
            {imageFile
              ? imageFile.name
              : "drop source image here (required for correct bounds)"}
          </span>
          {imageFile && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onImageFileChange(null);
              }}
              className="text-white/40 text-xs hover:text-white/70 cursor-pointer"
            >
              × clear
            </button>
          )}
          <input
            ref={imageInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={(e) => onImageFileChange(e.target.files?.[0] ?? null)}
          />
        </div>
        {imageFile && (
          <p className="text-white/50 text-xs -mt-3">
            image dimensions will be used for exact neon bounds
          </p>
        )}

        <div className="flex items-center gap-4">
          <button
            onClick={onContinue}
            disabled={!xmlFile || !imageFile}
            className="px-10 py-4 bg-white text-[#1D3335] font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer
                    disabled:opacity-40 disabled:cursor-not-allowed"
          >
            let's encode
          </button>
          <button
            onClick={onBackToProject}
            className="px-10 py-4 border-2 border-white text-white font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
          >
            back to project
          </button>
        </div>
        {!xmlFile && (
          <p className="text-white/50 text-xs -mt-3">
            no file selected — mock XML data will be used
          </p>
        )}
      </div>
    </div>
  );
}
