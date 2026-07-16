import { useState, useRef, useEffect, useCallback } from "react";
import type { ZoneDiff } from "../../utils/meiZoneDiff";
import { apiFetch } from "../../lib/apiFetch";

interface MeiImageDiffViewProps {
  imageId: string | null;
  diff: ZoneDiff;
  imageName: string;
}

export default function MeiImageDiffView({ imageId, diff, imageName }: MeiImageDiffViewProps) {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [naturalSize, setNaturalSize] = useState<{ w: number; h: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!imageId) return;
    let url: string | null = null;
    apiFetch(`/api/images/${imageId}`)
      .then((r) => (r.ok ? r.blob() : Promise.reject("fetch failed")))
      .then((blob) => {
        url = URL.createObjectURL(blob);
        setImageUrl(url);
      })
      .catch(() => setError("Failed to load source image."));
    return () => {
      if (url) URL.revokeObjectURL(url);
    };
  }, [imageId]);

  const drawOverlay = useCallback(() => {
    const canvas = canvasRef.current;
    const img = imgRef.current;
    if (!canvas || !img || !naturalSize) return;
    const dw = img.clientWidth;
    const dh = img.clientHeight;
    canvas.width = dw;
    canvas.height = dh;
    const ctx = canvas.getContext("2d")!;
    ctx.clearRect(0, 0, dw, dh);
    const sx = dw / naturalSize.w;
    const sy = dh / naturalSize.h;

    function drawZone(
      zone: { ulx: number; uly: number; lrx: number; lry: number },
      color: string,
    ) {
      const x = zone.ulx * sx;
      const y = zone.uly * sy;
      const w = (zone.lrx - zone.ulx) * sx;
      const h = (zone.lry - zone.uly) * sy;
      ctx.fillStyle = color + "33";
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
    }

    const hasChanges =
      diff.added.length + diff.removed.length + diff.moved.length > 0;
    if (!hasChanges) {
      diff.unchanged.forEach((z) => drawZone(z, "#4AADAA"));
    } else {
      diff.removed.forEach((z) => drawZone(z, "#EF4444"));
      diff.added.forEach((z) => drawZone(z, "#22C55E"));
      diff.moved.forEach((z) => drawZone(z, "#EAB308"));
    }
  }, [diff, naturalSize]);

  useEffect(() => {
    if (naturalSize) drawOverlay();
  }, [naturalSize, drawOverlay]);

  if (!imageId) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#1D3335]/50 text-sm italic">
        no source image linked to this MEI file
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex-1 flex items-center justify-center text-red-500 text-sm">
        {error}
      </div>
    );
  }

  if (!imageUrl) {
    return (
      <div className="flex-1 flex items-center justify-center text-[#1D3335]/50 text-sm">
        loading image...
      </div>
    );
  }

  const hasChanges =
    diff.added.length + diff.removed.length + diff.moved.length > 0;

  return (
    <div className="flex-1 flex flex-col min-h-0 overflow-auto p-4 gap-3 items-start">
      <div className="relative inline-block max-w-full">
        <img
          ref={imgRef}
          src={imageUrl}
          alt={imageName}
          className="max-w-full block"
          onLoad={(e) => {
            const img = e.currentTarget;
            setNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
          }}
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 pointer-events-none"
          style={{ width: "100%", height: "100%" }}
        />
      </div>

      <div className="text-xs text-[#1D3335]/60 shrink-0">
        {hasChanges ? (
          <span className="flex gap-4 flex-wrap">
            {diff.added.length > 0 && (
              <span>
                <span style={{ color: "#22C55E" }}>●</span> {diff.added.length} added
              </span>
            )}
            {diff.removed.length > 0 && (
              <span>
                <span style={{ color: "#EF4444" }}>●</span> {diff.removed.length} removed
              </span>
            )}
            {diff.moved.length > 0 && (
              <span>
                <span style={{ color: "#EAB308" }}>●</span> {diff.moved.length} moved
              </span>
            )}
          </span>
        ) : (
          <span className="italic">
            all zones displayed — no geometric changes detected (corrections were pitch or note-type only)
          </span>
        )}
      </div>
    </div>
  );
}
