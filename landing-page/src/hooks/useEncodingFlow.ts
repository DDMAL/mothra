import { useState, useEffect } from "react";
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

  const handleEncodeResult = async (ev: {
    session_id: string;
    mei_base64: string;
    manifest: Record<string, unknown> | null;
  }) => {
    setNeonManifest(ev.manifest ?? null);
    const stem = pendingXmlFile?.name.replace(".xml", "") ?? "output";
    settleMeiContent({ bytes: ev.mei_base64, stem });
    const xmlBytes = Uint8Array.from(atob(ev.mei_base64), (c) => c.charCodeAt(0));
    const xmlText = new TextDecoder().decode(xmlBytes);
    const newMeiFile: MeiFile = {
      id: crypto.randomUUID(),
      name: `${stem}.mei`,
      xmlContent: xmlText,
      corrected: false,
      imageName: pendingImageFile?.name ?? undefined,
    };
    if (selectedProjectId) {
      const r = await apiFetch(`/api/projects/${selectedProjectId}/mei`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        name: newMeiFile.name,
        xmlContent: xmlText,
        imageName: pendingImageFile?.name ?? null,
        logs: [],
      }),
    });
    const saved = await r.json();
    newMeiFile.id = saved.id;
  }
  setProjects((prev) => 
    prev.map((p) => 
      p.id === selectedProjectId ? {...p, meiFiles: [...p.meiFiles, newMeiFile] } : p, 
    )
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
  };
}
