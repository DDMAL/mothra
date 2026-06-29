import { useState } from "react";
import type { Project, MeiFile } from "../../types";
import { authHeaders } from "../../hooks/useAuth";
import CompletionPage from "./CompletionPage";
import MeiCompareModal from "./MeiCompareModal";

interface Props {
    project: Project;
    originalMeiFiles: MeiFile[];
    onSendToCantus: () => void;
    onBackToProject: () => void;
}

export default function NeonCompletionPage({
    project,
    originalMeiFiles,
    onSendToCantus,
    onBackToProject,
}: Props) {
    const [showCompare, setShowCompare] = useState(false);
    const [correctedFiles, setCorrectedFiles] = useState<MeiFile[]>([]);
    const [loadingCompare, setLoadingCompare] = useState(false);

    async function handleCompare() {
        if (correctedFiles.length > 0) {
            setShowCompare(true);
            return;
        }
        setLoadingCompare(true);
        try {
            const r = await fetch(`/api/projects/${project.id}`, {
                headers: authHeaders(),
            });
            if (r.ok) {
                const data = await r.json();
                const files: MeiFile[] = (data.meiFiles ?? []).map(
                    (f: { id: string; name: string; xmlContent?: string; corrected?: boolean; imageName?: string }) => ({
                        id: String(f.id),
                        name: f.name,
                        xmlContent: f.xmlContent ?? "",
                        corrected: !!f.corrected,
                        imageName: f.imageName,
                    }),
                );
                setCorrectedFiles(files.filter((f) => f.corrected));
            }
        } finally {
            setLoadingCompare(false);
            setShowCompare(true);
        }
    }

    return (
        <>
        <CompletionPage
            description="corrected mei files can now be sent to cantus ultimus and viewed on the project page."
            continueLabel={loadingCompare ? "loading…" : "send to cantus ultimus"}
            onContinue={onSendToCantus}
            onBackToProject={onBackToProject}
            onCompare={originalMeiFiles.length > 0 ? handleCompare : undefined}
        />
        {showCompare && (
            <MeiCompareModal
            originalFiles={originalMeiFiles}
            correctedFiles={correctedFiles}
            onClose={() => setShowCompare(false)}
            projectImages={project.images}
            />
        )}
        </>
    );
}