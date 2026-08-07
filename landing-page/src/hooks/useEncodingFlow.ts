import { useState } from "react";
import type { Project } from "../types";
import { apiFetch } from "../lib/apiFetch";
import { downloadBlob } from "../utils/download";
import type { MeiFile } from "../types";

type SetProjects = React.Dispatch<React.SetStateAction<Project[]>>;

export function useEncodingFlow(
  selectedProjectId: number | null,
  setProjects: SetProjects,
) {
  const [pendingXmlFile, setPendingXmlFile] = useState<File | null>(null);
  const [pendingImageFile, setPendingImageFile] = useState<File | null>(null);
  const [neonManifest, setNeonManifest] = useState<Record<
    string,
    unknown
  > | null>(null);
  const [meiContent, settleMeiContent] = useState<{
    bytes: string;
    stem: string;
  } | null>(null);
  const [pendingBatchPairs, setPendingBatchPairs] = useState<
    { xmlFile: File; imageFile: File }[]
  >([]);
  const [batchResults, setBatchResults] = useState<
    {
      sessionId: string;
      stem: string;
      manifest: Record<string, unknown> | null;
      imageName?: string;
    }[]
  >([]);

  const handleEncodeBatchResult = async (ev: {
    item: number;
    session_id: string;
    mei_base64: string;
    manifest: Record<string, unknown> | null;
    image_name?: string;
    stem?: string;
    stave_source?: string | null;
    logs?: string[];
  }) => {
    const pair = pendingBatchPairs[ev.item];
    const stem =
      ev.stem ??
      pair?.imageFile.name.replace(/\.[^.]+$/, "") ??
      pair?.xmlFile.name.replace(/\.xml$/i, "") ??
      `item-${ev.item}`;
    const imageName = pair?.imageFile.name ?? ev.image_name;
    const xmlBytes = Uint8Array.from(atob(ev.mei_base64), (c) =>
      c.charCodeAt(0),
    );
    const xmlText = new TextDecoder().decode(xmlBytes);
    const newMeiFile: MeiFile = {
      id: crypto.randomUUID(),
      name: `${stem}.mei`,
      xmlContent: xmlText,
      corrected: false,
      imageName,
      staveSource: (ev.stave_source as MeiFile["staveSource"]) ?? null,
    };
    if (selectedProjectId) {
      const r = await apiFetch(`/api/projects/${selectedProjectId}/mei`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newMeiFile.name,
          xmlContent: xmlText,
          imageName: imageName ?? null,
          logs: ev.logs ?? [],
          staveSource: ev.stave_source ?? null,
        }),
      });
      const saved = await r.json();
      newMeiFile.id = saved.id;
    }
    setProjects((prev) =>
      prev.map((p) =>
        p.id === selectedProjectId
          ? { ...p, meiFiles: [...p.meiFiles, newMeiFile] }
          : p,
      ),
    );
    setBatchResults((prev) => [
      ...prev,
      { sessionId: ev.session_id, stem, manifest: ev.manifest, imageName },
    ]);
  };

  const handleEncodeResult = async (ev: {
    session_id: string;
    mei_base64: string;
    manifest: Record<string, unknown> | null;
    stave_source?: string | null;
    logs?: string[];
  }) => {
    setNeonManifest(ev.manifest ?? null);
    const stem =
      pendingImageFile?.name.replace(/\.[^.]+$/, "") ??
      pendingXmlFile?.name.replace(/\.xml$/i, "") ??
      "output";
    settleMeiContent({ bytes: ev.mei_base64, stem });
    const xmlBytes = Uint8Array.from(atob(ev.mei_base64), (c) =>
      c.charCodeAt(0),
    );
    const xmlText = new TextDecoder().decode(xmlBytes);
    const newMeiFile: MeiFile = {
      id: crypto.randomUUID(),
      name: `${stem}.mei`,
      xmlContent: xmlText,
      corrected: false,
      imageName: pendingImageFile?.name ?? undefined,
      staveSource: (ev.stave_source as MeiFile["staveSource"]) ?? null,
    };
    if (selectedProjectId) {
      const r = await apiFetch(`/api/projects/${selectedProjectId}/mei`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: newMeiFile.name,
          xmlContent: xmlText,
          imageName: pendingImageFile?.name ?? null,
          logs: ev.logs ?? [],
          staveSource: ev.stave_source ?? null,
        }),
      });
      const saved = await r.json();
      newMeiFile.id = saved.id;
    }
    setProjects((prev) =>
      prev.map((p) =>
        p.id === selectedProjectId
          ? { ...p, meiFiles: [...p.meiFiles, newMeiFile] }
          : p,
      ),
    );
  };
  // download helpers

  const handleDownloadManifest = () => {
    if (!neonManifest || !meiContent) return;
    downloadBlob(
      new Blob([JSON.stringify(neonManifest, null, 2)], {
        type: "application/ld+json",
      }),
      `${meiContent.stem}_manifest.jsonld`,
    );
  };

  const handleDownloadMei = () => {
    if (!meiContent?.bytes) return;
    const bytes = Uint8Array.from(atob(meiContent.bytes), (c) =>
      c.charCodeAt(0),
    );
    downloadBlob(
      new Blob([bytes], { type: "application/xml" }),
      `${meiContent.stem}.mei`,
    );
  };

  return {
    pendingXmlFile,
    setPendingXmlFile,
    pendingImageFile,
    setPendingImageFile,
    neonManifest,
    meiContent,
    handleDownloadManifest,
    handleDownloadMei,
    handleEncodeResult,
    pendingBatchPairs,
    setPendingBatchPairs,
    batchResults,
    setBatchResults,
    handleEncodeBatchResult,
  };
}
