import { useState } from "react";
import type { ProjectImage } from "../../App";

interface InteractiveClassifierProps {
  images: ProjectImage[];
  onProcessAll: () => void;
}

export default function InteractiveClassifier({
  images,
  onProcessAll,
}: InteractiveClassifierProps) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [menuOpen, setMenuOpen] = useState(false);
  const [processedIds, setProcessedIds] = useState<Set<string>>(new Set());

  const markProcessed = () => {
    const next = new Set(processedIds);
    next.add(images[currentIdx].id);
    setProcessedIds(next);
    for (let i = currentIdx + 1; i < images.length; i++) {
      if (!next.has(images[i].id)) { setCurrentIdx(i); return; }
    }
    for (let i = 0; i < currentIdx; i++) {
      if (!next.has(images[i].id)) { setCurrentIdx(i); return; }
    }
  };

  const img = images[currentIdx];

  // show 5 thumbnails at a time, centered on currentIdx
  const VISIBLE = 5;
  const half = Math.floor(VISIBLE / 2);
  const start = Math.max(
    0,
    Math.min(currentIdx - half, images.length - VISIBLE),
  );
  const visibleImages = images.slice(start, start + VISIBLE);

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col pb-6">
      <div className="flex items-center gap-6 px-8 py-5">
        <h1 className="text-4xl font-bold italic text-white">
          interactive classifier
        </h1>
        <button
          onClick={onProcessAll}
          className="px-6 py-2 border-2 border-white text-white rounded-xl hover:opacity-90 cursor-pointer font-semibold"
        >
          process all
        </button>
        <div className="flex-1" />
        <div className="relative">
          <button
            onClick={() => setMenuOpen((o) => !o)}
            className="w-12 h-12 bg-white rounded-full flex flex-col items-center justify-center gap-1 cursor-pointer hover:opacity-90"
          >
            <span className="block w-5 h-0.5 bg-[#1D3335]" />
            <span className="block w-5 h-0.5 bg-[#1D3335]" />
            <span className="block w-5 h-0.5 bg-[#1D3335]" />
          </button>
          {menuOpen && (
            <>
              <div
                className="fixed inset-0 z-40"
                onClick={() => setMenuOpen(false)}
              />
              <div className="absolute right-0 top-14 z-50 bg-white rounded-2xl shadow-lg p-4 flex flex-col gap-1 min-w-[220px] text-center">
                <button className="text-[#1D3335] px-4 py-2 hover:opacity-60 cursor-pointer text-sm">
                  save
                </button>
                <button className="text-[#1D3335] px-4 py-2 hover:opacity-60 cursor-pointer text-sm">
                  about the interactive classifier
                </button>
                <button
                  onClick={() => {
                    setMenuOpen(false);
                    onProcessAll();
                  }}
                  className="text-[#1D3335] px-4 py-2 hover:opacity-60 cursor-pointer text-sm font-bold"
                >
                  process all
                </button>
              </div>
            </>
          )}
        </div>
      </div>

      {/* canvas */}
      <div className="flex-1 bg-[#1D3335] mx-6 rounded-2xl flex flex-col overflow-hidden">
        {/* image area */}
        <div className="flex-1 flex items-start justify-start p-6">
          {images.length === 0 ? (
            <div className="text-white/40 text-sm italic">
              {" "}
              no images selected{" "}
            </div>
          ) : (
            <div className="w-1/2 aspect-[4/3] bg-[#2A4A4D] rounded-xl overflow-hidden flex items-center justify-center">
              {img?.src ? (
                <img
                  src={img.src}
                  alt={img.name}
                  className="w-full h-full object-contain"
                />
              ) : (
                <span className="text-white/50 text-lg">
                  {img?.name ?? "image"} {currentIdx + 1}/{images.length}
                </span>
              )}
            </div>
          )}
        </div>

        {/* filmstrip + mark button */}

        {images.length > 0 && (
          <div className="flex items-center px-6 pb-6 gap-4">
            <button
              onClick={markProcessed}
              disabled={processedIds.has(images[currentIdx].id)}
              className="px-4 py-2 border-2 border-white text-white text-sm rounded-xl hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed whitespace-nowrap flex-shrink-0"
            >
              {processedIds.has(images[currentIdx].id) ? "processed [√]" : "mark image as processed"}
            </button>
            <div className="flex-1 flex items-center justify-center gap-3">
              <button
                onClick={() => setCurrentIdx((i) => i - 1)}
                disabled={currentIdx === 0}
                className="text-white text-xl hover:opacity-70 disabled:opacity-20 cursor-pointer"
              >
                &lt;
              </button>
              {visibleImages.map((thumb, i) => {
                const globalIdx = start + i;
                const active = globalIdx === currentIdx;
                const processed = processedIds.has(thumb.id);
                return (
                  <button
                    key={thumb.id}
                    onClick={() => setCurrentIdx(globalIdx)}
                    className={`relative w-16 aspect-square rounded-lg overflow-hidden flex-shrink-0 cursor-pointer transition-all
                      ${active ? "ring-2 ring-white ring-offset-2 ring-offset-[#1D3335]" : "opacity-50 hover:opacity-80"}`}
                  >
                    {thumb.src ? (
                      <img
                        src={thumb.src}
                        alt={thumb.name}
                        className="w-full h-full object-cover"
                      />
                    ) : (
                      <div className="w-full h-full bg-[#2A4A4D]" />
                    )}
                    {processed && (
                      <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                        <span className="text-white text-lg font-bold">[√]</span>
                      </div>
                    )}
                  </button>
                );
              })}
              <button
                onClick={() => setCurrentIdx((i) => i + 1)}
                disabled={currentIdx === images.length - 1}
                className="text-white text-xl hover:opacity-70 disabled:opacity-20 cursor-pointer"
              >
                &gt;
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
