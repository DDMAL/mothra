import { useState, useEffect } from "react";
import type { Project } from "../types";
import { authHeaders } from "./useAuth";
import { downloadBlob } from "../utils/download";
import type { View } from "../types";
import type { MeiFile } from "../types";

type SetProjects = React.Dispatch<React.SetStateAction<Project[]>>;

export function useEncodingFlow(
    view: View,
    selectedProjectId: number | null,
    setProjects: SetProjects,
) {
    const [encodingLogs, setEncodingLogs] = useState<string[]>([]);
    const [pendingXmlFile, setPendingXmlFile] = useState<File | null>(null);
    const [pendingImageFile, setPendingImageFile] = useState<File | null>(null);
    const [neonManifest, setNeonManifest] = useState<Record<string, unknown> | null>(null);
    const [meiContent, settleMeiContent] = useState<{ bytes: string; stem: string } | null>(null);


    // encoding effect
    
    useEffect(() => {
        if (view !== "encoding-processing") return;
        settleMeiContent(null);
        setNeonManifest(null);
    
        if (pendingXmlFile) {
          const form = new FormData();
          form.append("xml_file", pendingXmlFile);
          if (pendingImageFile) {
            form.append("image_file", pendingImageFile);
          }
          fetch("/api/encode-upload", { method: "POST", body: form })
            .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
            .then(async (data) => {
              setNeonManifest(data.manifest ?? null);
              setEncodingLogs(data.logs ?? []);
              settleMeiContent({ bytes: data.mei_base64, stem: pendingXmlFile.name.replace(".xml", "")});
    
              const xmlBytes = Uint8Array.from(atob(data.mei_base64), (c) => c.charCodeAt(0));
              const xmlText = new TextDecoder().decode(xmlBytes);
              const stem = pendingXmlFile.name.replace(".xml", "");
              const newMeiFile: MeiFile = {
                id: crypto.randomUUID(),
                name: `${stem}.mei`,
                xmlContent: xmlText,
                corrected: false,
              };
              if (selectedProjectId) {
                const r = await fetch(`/api/projects/${selectedProjectId}/mei`, {
                  method: "POST",
                  headers: { ...authHeaders(), "Content-Type": "application/json"},
                  body: JSON.stringify({ name: newMeiFile.name, xmlContent: xmlText }),
                });
                const saved = await r.json();
                newMeiFile.id = saved.id;
              }
    
              setProjects((prev) => 
                prev.map((p) => 
                  p.id === selectedProjectId
                    ? { ...p, meiFiles: [...p.meiFiles, newMeiFile]}
                  : p,
                ),
              );
            })
            .catch((err) => console.error("Encoding failed:", err));
        } else {
          // mock fallback
          fetch("/api/encode", { method: "POST"})
            .then((r) => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); })
            .then((data) => {
              setEncodingLogs(data.logs ?? []);
            })
            .catch((err) => console.error("encoding failed:", err));
        }
    }, [view, pendingXmlFile, pendingImageFile, selectedProjectId]);

      // download helpers
    
    const handleDownloadManifest = () => {
        if (!neonManifest || !meiContent) return;
        downloadBlob(
          new Blob([JSON.stringify(neonManifest, null, 2)], { type: "application/ld+json" }),
          `${meiContent.stem}_manifest.jsonld`,
        );
    };
    
    const handleDownloadMei = () => {
        if (!meiContent?.bytes) return;
        const bytes = Uint8Array.from(atob(meiContent.bytes), (c) => c.charCodeAt(0));
        downloadBlob(new Blob([bytes], { type: "application/xml" }), `${meiContent.stem}.mei`);
    };

    return {
        encodingLogs,
        pendingXmlFile, setPendingXmlFile,
        pendingImageFile, setPendingImageFile,
        neonManifest,
        meiContent,
        handleDownloadManifest,
        handleDownloadMei,
    };
}