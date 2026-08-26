import { Fragment, useRef, useState } from "react";
import type { IcXmlFile, ProjectImage } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { downloadBlob } from "../../utils/download";
import { sortBySourceThenFolio, sourceGroupLabel } from "../../utils/folio";
import { AuthImage } from "../shared/AuthImage";
import TruncatedName from "../shared/TruncatedName";
import Modal from "../shared/Modal";

interface IcXmlTabProps {
  icXmlFiles: IcXmlFile[];
  images: ProjectImage[];
  projectId: number;
  /** Called after a file is deleted so the project's icXmlFiles drops it. */
  onDeleted: (xmlId: string) => void;
}

const formatBytes = (bytes: number) =>
  bytes < 1024
    ? `${bytes} B`
    : bytes < 1024 * 1024
      ? `${(bytes / 1024).toFixed(0)} KB`
      : `${(bytes / (1024 * 1024)).toFixed(1)} MB`;

// The GameraXML each page was encoded from -- the classifier's output as the
// encoder read it (written by the encode job, see ic_xml_store.py), so it
// belongs beside the MEI it produced. Unlike MEI files, the document body is
// *not* in the project payload (a page's export is megabytes of RLE glyph
// masks), so both "view" and "download" fetch it per file from ic_api.py.
export default function IcXmlTab({
  icXmlFiles,
  images,
  projectId,
  onDeleted,
}: IcXmlTabProps) {
  const [viewFile, setViewFile] = useState<IcXmlFile | null>(null);
  const [viewContent, setViewContent] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  // Identifies the newest "view" click, so a slower earlier one can't write
  // state after it. See handleView.
  const viewSeq = useRef(0);

  if (icXmlFiles.length === 0) {
    return (
      <p className="mt-6 text-white/70 text-sm">
        no classifier XML yet — each page's file is written when the page is
        encoded (step 3), from the XML the encoder read
      </p>
    );
  }

  const fetchXml = async (file: IcXmlFile): Promise<string> => {
    const r = await apiFetch(`/api/projects/${projectId}/ic-xml/${file.id}`);
    if (!r.ok) throw new Error(`couldn't load ${file.name} (${r.status})`);
    const data = await r.json();
    return (data.xmlContent as string) ?? "";
  };

  // Every write after the await is gated on this click still being the
  // newest one. A page's XML is megabytes of RLE glyph masks, so a slow load
  // is easy to abandon -- Modal's backdrop closes on a single click, so
  // "open a page, dismiss, open another" is one stray click away -- and the
  // abandoned response would otherwise land under the newer page's header:
  // right name, right glyph count, wrong document, with nothing to signal
  // the mismatch on a tab whose whole job is showing what the encoder read.
  // The catch and finally need the same gate: a stale failure would close
  // the newer file's modal and report the wrong file's error, and a stale
  // finally would clear the newer file's busy flag while it is still
  // loading.
  const handleView = async (file: IcXmlFile) => {
    const seq = ++viewSeq.current;
    setError(null);
    setViewFile(file);
    setViewContent(null);
    setBusyId(file.id);
    try {
      const xml = await fetchXml(file);
      if (seq === viewSeq.current) setViewContent(xml);
    } catch (e) {
      if (seq === viewSeq.current) {
        setError((e as Error).message);
        setViewFile(null);
      }
    } finally {
      if (seq === viewSeq.current) setBusyId(null);
    }
  };

  // Its own endpoint rather than reusing fetchXml: the download is the one
  // path that never needs the document in JS memory, and a page's export can
  // be several MB of RLE glyph masks.
  const handleDownload = async (file: IcXmlFile) => {
    setError(null);
    setBusyId(file.id);
    try {
      const r = await apiFetch(
        `/api/projects/${projectId}/ic-xml/${file.id}/download`,
      );
      if (!r.ok)
        throw new Error(`couldn't download ${file.name} (${r.status})`);
      downloadBlob(await r.blob(), file.name);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyId(null);
    }
  };

  const handleDelete = async (file: IcXmlFile) => {
    setError(null);
    setBusyId(file.id);
    try {
      const r = await apiFetch(`/api/projects/${projectId}/ic-xml/${file.id}`, {
        method: "DELETE",
      });
      if (!r.ok) throw new Error(`couldn't delete ${file.name} (${r.status})`);
      onDeleted(file.id);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusyId(null);
    }
  };

  const sorted = sortBySourceThenFolio(icXmlFiles, images, (f) => f.imageName);

  return (
    <>
      {viewFile && (
        <Modal onClose={() => setViewFile(null)} size="4xl" backdrop="dark">
          <div>
            <h2 className="text-xl font-bold italic text-[#1D3335]">
              {viewFile.name}
            </h2>
            <p className="mt-1 text-xs text-[#1D3335]/60">
              classifier output (GameraXML)
              {viewFile.glyphCount != null
                ? ` — ${viewFile.glyphCount} glyph${viewFile.glyphCount === 1 ? "" : "s"}`
                : ""}
            </p>
          </div>
          <pre className="h-[60vh] overflow-auto rounded-2xl bg-white p-4 text-[11px] leading-relaxed text-[#1D3335] whitespace-pre-wrap break-all">
            {viewContent ?? "loading…"}
          </pre>
          <button
            onClick={() => handleDownload(viewFile)}
            className="self-end px-5 py-2 bg-[#1D3335] text-white rounded-xl hover:opacity-90 cursor-pointer text-sm font-semibold"
          >
            download ↓
          </button>
        </Modal>
      )}
      {error && (
        <p className="mt-6 text-red-200 text-xs font-mono" title={error}>
          {error}
        </p>
      )}
      <div className="mt-6 grid grid-cols-5 gap-4">
        {sorted.map((file, idx) => {
          const group = sourceGroupLabel(images, file.imageName);
          const prevGroup =
            idx > 0
              ? sourceGroupLabel(images, sorted[idx - 1].imageName)
              : undefined;
          const showHeader = group !== prevGroup;
          const busy = busyId === file.id;
          return (
            <Fragment key={file.id}>
              {showHeader && (
                <div className="col-span-5 text-white/70 text-xs font-mono uppercase tracking-wide mt-4 first:mt-0 pb-1 border-b border-white/20">
                  {group}
                </div>
              )}
              <div className="flex flex-col gap-2">
                <div
                  className="relative aspect-square cursor-pointer"
                  onClick={() => handleView(file)}
                >
                  <div className="absolute inset-0 bg-[#C8E6E3]/50 rounded-xl overflow-hidden flex items-end justify-start p-2">
                    {file.imageSrc && (
                      <AuthImage
                        src={file.imageSrc}
                        alt={file.imageName}
                        className="absolute inset-0 w-full h-full object-cover opacity-60"
                      />
                    )}
                    <span className="relative text-[10px] text-white/80 font-mono z-10">
                      .xml
                    </span>
                    <div className="absolute top-1.5 right-1.5 z-20 flex gap-1">
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDownload(file);
                        }}
                        disabled={busy}
                        title="download this page's classifier XML"
                        className="px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        ↓
                      </button>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleView(file);
                        }}
                        disabled={busy}
                        className="px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                      >
                        view
                      </button>
                    </div>
                    {/* Deletes mothra's copy only -- the IC session it came
                        from is untouched and still resumable, so re-exporting
                        the page brings the file back. */}
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDelete(file);
                      }}
                      disabled={busy}
                      title="delete this copy (the IC session is kept)"
                      className="absolute top-1.5 left-1.5 z-20 px-1.5 py-0.5 bg-black/40 text-white text-[9px] font-mono rounded hover:bg-black/70 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
                    >
                      ✕
                    </button>
                  </div>
                </div>
                <TruncatedName
                  name={file.imageName.replace(/\.[^.]+$/, "")}
                  suffix=".xml"
                  className="text-sm text-white"
                />
                <span className="text-xs text-white/50">
                  {file.glyphCount != null
                    ? `${file.glyphCount} glyph${file.glyphCount === 1 ? "" : "s"}`
                    : "glyph count unknown"}
                  {file.byteSize != null
                    ? ` — ${formatBytes(file.byteSize)}`
                    : ""}
                </span>
              </div>
            </Fragment>
          );
        })}
      </div>
    </>
  );
}
