import { useCallback, useEffect, useRef, useState } from "react";
import type { ProjectImage } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { AuthImage } from "../shared/AuthImage";

interface InteractiveClassifierProps {
  images: ProjectImage[];
  projectId: number | null;
  setPendingXmlFile: (f: File | null) => void;
  setPendingImageFile: (f: File | null) => void;
  // Advance to the encoding step (also unlocks step 2). Called once the
  // GameraXML + image have been staged via the setters above.
  onEncode: () => void;
  clefShape: "C" | "F";
  onClefShapeChange: (s: "C" | "F") => void;
  clefLine: number;
  onClefLineChange: (n: number) => void;
}

const stemOf = (name: string) => name.replace(/\.[^.]+$/, "");

export default function InteractiveClassifier({
  images,
  projectId,
  setPendingXmlFile,
  setPendingImageFile,
  onEncode,
  clefShape,
  onClefShapeChange,
  clefLine,
  onClefLineChange
}: InteractiveClassifierProps) {
  const [currentIdx, setCurrentIdx] = useState(0);
  const [icUrl, setIcUrl] = useState<string | null>(null);
  const [icOrigin, setIcOrigin] = useState<string | null>(null);
  // Set only once the user finishes IC's create-session screen (the iframe
  // posts it back); until then there's nothing to encode.
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [status, setStatus] = useState<"idle" | "starting" | "ready" | "error">(
    "idle",
  );
  const [error, setError] = useState<string | null>(null);
  const [encoding, setEncoding] = useState(false);

  const img = images[currentIdx];
  // Guards against a slow /ic/start response landing after the user has
  // already switched pages — only the latest request may set state.
  const startSeq = useRef(0);

  // Stage a fresh page + bboxes in IC whenever the selected page changes.
  useEffect(() => {
    if (!img || projectId == null) return;
    const seq = ++startSeq.current;
    setStatus("starting");
    setError(null);
    setIcUrl(null);
    setIcOrigin(null);
    setSessionId(null);

    fetch(`/api/projects/${projectId}/ic/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ imageName: img.name }),
    })
      .then(async (r) => {
        if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
        return r.json();
      })
      .then((data) => {
        if (seq !== startSeq.current) return; // superseded by a newer page
        setIcUrl(data.ic_url);
        try {
          setIcOrigin(new URL(data.ic_url).origin);
        } catch {
          setIcOrigin(null);
        }
        setStatus("ready");
      })
      .catch((err) => {
        if (seq !== startSeq.current) return;
        setError(String(err.message ?? err));
        setStatus("error");
      });
  }, [img?.name, projectId]);

  // The embedded IC posts its new session id once the user starts the
  // session on the create-session screen. Accept it only from IC's origin.
  useEffect(() => {
    function onMessage(e: MessageEvent) {
      if (icOrigin && e.origin !== icOrigin) return;
      const data = e.data;
      if (data?.type === "ic:session-created" && typeof data.sessionId === "string") {
        setSessionId(data.sessionId);
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [icOrigin]);

  const handleEncode = useCallback(async () => {
    if (!sessionId || !img) return;
    setEncoding(true);
    setError(null);
    try {
      // 1. Finalise the IC session → GameraXML.
      const r = await apiFetch(`/api/ic/${sessionId}/complete`, {
        method: "POST",
      });
      if (!r.ok) throw new Error(await r.text().catch(() => `HTTP ${r.status}`));
      const data = await r.json();
      const xmlBytes = Uint8Array.from(atob(data.xml_base64), (c) =>
        c.charCodeAt(0),
      );
      const xmlFile = new File([xmlBytes], `${stemOf(img.name)}.xml`, {
        type: "application/xml",
      });

      // 2. Fetch the page image bytes so the encoder can read its size.
      const imgResp = await apiFetch(`/api/images/${img.id}`);
      if (!imgResp.ok) throw new Error(`image fetch failed (${imgResp.status})`);
      const blob = await imgResp.blob();
      const imageFile = new File([blob], img.name, {
        type: blob.type || "image/png",
      });

      // 3. Hand both to the existing encode flow and advance.
      setPendingXmlFile(xmlFile);
      setPendingImageFile(imageFile);
      onEncode();
    } catch (err) {
      setError(String((err as Error).message ?? err));
      setEncoding(false);
    }
  }, [sessionId, img, setPendingXmlFile, setPendingImageFile, onEncode]);

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
        {images.length > 1 && (
          <span className="text-white/80 text-sm font-mono">
            page {currentIdx + 1}/{images.length}
            {img ? ` — ${img.name}` : ""}
          </span>
        )}
        <div className="flex items-center gap-2 text-white/80 text-sm">
          <span className="text-white/50 text-xs">clef</span>
          <select
            value={clefShape}
            onChange={e => onClefShapeChange(e.target.value as "C" | "F")}
            className="bg-transparent border border-white/30 rounded px-1 text-sm cursor-pointer text-white"
          >
            <option value="C">C</option>
            <option value="F">F</option>
          </select>
          <input
            type="number" min={1} max={5} value={clefLine}
            onChange={e => onClefLineChange(Number(e.target.value))}
            className="w-10 bg-transparent border border-white/30 rounded px-1 text-sm text-center text-white"
          />
        </div>
        <div className="flex-1" />
        {status === "ready" && !sessionId && (
          <span className="text-white/80 text-sm">
            start the session in the classifier to enable encoding
          </span>
        )}
        <button
          onClick={handleEncode}
          disabled={!sessionId || encoding}
          className="px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {encoding ? "encoding…" : "encode"}
        </button>
      </div>

      {/* canvas */}
      <div className="flex-1 bg-[#1D3335] mx-6 rounded-2xl flex flex-col overflow-hidden">
        {/* IC editor area */}
        <div className="flex-1 flex items-stretch justify-stretch overflow-hidden">
          {images.length === 0 ? (
            <div className="flex-1 flex items-center justify-center text-white/40 text-sm italic">
              no images selected
            </div>
          ) : status === "error" ? (
            <div className="flex-1 flex flex-col items-center justify-center gap-2 text-center px-8">
              <p className="text-red-300 text-sm">
                couldn't start the interactive classifier
              </p>
              <p className="text-white/50 text-xs font-mono max-w-lg break-words">
                {error}
              </p>
              <p className="text-white/40 text-xs">
                is the IC service running on its port? (see CLAUDE.md)
              </p>
            </div>
          ) : icUrl ? (
            <iframe
              key={icUrl}
              src={icUrl}
              title={`Interactive Classifier — ${img?.name ?? ""}`}
              className="flex-1 w-full border-0"
            />
          ) : (
            <div className="flex-1 flex items-center justify-center text-white/50 text-sm">
              starting classifier…
            </div>
          )}
        </div>

        {/* filmstrip for page selection */}
        {images.length > 1 && (
          <div className="flex items-center px-6 pb-6 pt-4 gap-4">
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
                return (
                  <button
                    key={thumb.id}
                    onClick={() => setCurrentIdx(globalIdx)}
                    className={`relative w-16 aspect-square rounded-lg overflow-hidden flex-shrink-0 cursor-pointer transition-all
                      ${active ? "ring-2 ring-white ring-offset-2 ring-offset-[#1D3335]" : "opacity-50 hover:opacity-80"}`}
                  >
                    <AuthImage
                      src={`/api/images/${thumb.id}`}
                      alt={thumb.name}
                      className="w-full h-full object-cover"
                    />
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
