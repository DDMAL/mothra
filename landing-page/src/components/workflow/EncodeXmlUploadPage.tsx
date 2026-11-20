import { useRef, useState } from "react";

interface EncodeXmlUploadPageProps {
  onContinue: () => void;
  onBackToProject: () => void;
  xmlFile: File | null;
  onXmlFileChange: (f: File | null) => void;
  imageFile: File | null;
  onImageFileChange: (f: File | null) => void;
}

// Encodes a GameraXML file the user uploads (POST /api/encode-upload), for a
// page classified outside mothra -- every page classified in mothra is
// encoded straight from its IC session instead. Reached from the project
// page's Classifier XML tab.
export default function EncodeXmlUploadPage({
  onContinue,
  onBackToProject,
  xmlFile,
  onXmlFileChange,
  imageFile,
  onImageFileChange,
}: EncodeXmlUploadPageProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const [imageDragging, setImageDragging] = useState(false);

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48 relative">
      <div className="flex flex-col items-center gap-6">
        <h1 className="text-5xl font-bold italic text-[#1D3335] text-center">
          encode a GameraXML file
        </h1>
        <p className="text-xl text-[#1D3335] text-center max-w-xl">
          for a page classified outside mothra: upload its GameraXML and the
          page image, and the encoded MEI file is added to this project
        </p>
        <p className="text-sm text-[#1D3335] text-center max-w-xl -mt-3">
          if the image has the same file name as one of this project's pages,
          that page's detected staves and text are used
        </p>

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
            ${dragging ? "border-[#1D3335] bg-white/20" : "border-[#1D3335]/40 hover:border-[#1D3335]/70"}`}
        >
          <span className="text-[#1D3335] text-sm font-mono">
            {xmlFile ? xmlFile.name : "drop a GameraXML file here, or click to browse"}
          </span>
          {xmlFile && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onXmlFileChange(null);
              }}
              className="text-[#1D3335] text-xs hover:underline cursor-pointer"
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
            ${imageDragging ? "border-[#1D3335] bg-white/20" : "border-[#1D3335]/40 hover:border-[#1D3335]/70"}`}
        >
          <span className="text-[#1D3335] text-sm font-mono">
            {imageFile
              ? imageFile.name
              : "drop the page image here, or click to browse"}
          </span>
          {imageFile && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onImageFileChange(null);
              }}
              className="text-[#1D3335] text-xs hover:underline cursor-pointer"
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
          <p className="text-[#1D3335] text-xs -mt-3">
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
            className="px-10 py-4 border-2 border-[#1D3335] text-[#1D3335] font-semibold text-lg rounded-2xl hover:bg-white/20 cursor-pointer"
          >
            back to project
          </button>
        </div>
      </div>
    </div>
  );
}
