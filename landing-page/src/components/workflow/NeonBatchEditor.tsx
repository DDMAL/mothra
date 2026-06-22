import { useState, useEffect, useRef } from "react";
import type { Project, MeiFile } from "../../types";
import { authHeaders } from "../../hooks/useAuth";

interface BatchSession {
    session_id: string;
    manifest_id: string;
}

interface NeonBatchEditorProps {
    project: Project;
    meiFiles: MeiFile[];
    onFinish: () => void;
    onBack: () => void;
}

const btn: React.CSSProperties = {
    padding: "6px 14px",
    borderRadius: 6,
    border: "none",
    cursor: "pointer",
    background: "#2d2d4e",
    color: "white",
    fontSize: 13,
    whiteSpace: "nowrap",
};

export default function NeonBatchEditor({ project, meiFiles, onFinish, onBack }: NeonBatchEditorProps) {
    const [sessions, setSessions] = useState<Map<string, BatchSession>>(new Map());
    const [currentIndex, setCurrentIndex] = useState(0);
    const [corrected, setCorrected] = useState<Set<string>>(new Set());
    const [loading, setLoading] = useState(true);
    const iframeRef = useRef<HTMLIFrameElement>(null);

    useEffect(() => {
        async function initSessions() {
            const results = await Promise.all(
                meiFiles.map(async (file) => {
                    const r = await fetch(`/api/projects/${project.id}/mei/${file.id}/edit-session`, {
                        method: "POST",
                        headers: authHeaders(),
                    });
                    if (!r.ok) return [file.id, null] as const;
                    const data: BatchSession = await r.json();
                    return [file.id, data] as const;
                })
            );
            const map = new Map<string, BatchSession>();
            for (const [id, session] of results) {
                if (session) map.set(id, session);
            }
            setSessions(map);
            setLoading(false);
        }
        initSessions();
    }, [project.id, meiFiles]);

    const currentFile = meiFiles[currentIndex];
    const currentSession = currentFile ? sessions.get(currentFile.id) : null;
    const allCorrected = meiFiles.length > 0 && meiFiles.every((f) => corrected.has(f.id));

    function triggerNeonSave() {
        const iframeBody = iframeRef.current?.contentDocument?.body;
        if (iframeBody) {
            iframeBody.dispatchEvent(new KeyboardEvent("keydown", { key: "s", bubbles: true }));
        }
    }

    async function markCurrentDone() {
        if (!currentFile || corrected.has(currentFile.id)) return;
        triggerNeonSave();
        // brief wait for the async PUT inside Neon's updateDatabase() to complete
        await new Promise((r) => setTimeout(r, 800));
        await fetch(`/api/projects/${project.id}/mei/${currentFile.id}`, {
            method: "PATCH",
            headers: { ...authHeaders(), "Content-Type": "application/json" },
            body: JSON.stringify({ corrected: true }),
        });
        setCorrected((prev) => new Set([...prev, currentFile.id]));
    }

    async function handleDoneAndNext() {
        await markCurrentDone();
        if (currentIndex < meiFiles.length - 1) {
            setCurrentIndex((i) => i + 1);
        }
    }

    return (
        <div style={{ display: "flex", flexDirection: "column", height: "100vh", background: "#0f0f1a" }}>
            <div style={{
                display: "flex", alignItems: "center", gap: 8, padding: "8px, 16px",
                background: "#1a1a2e", borderBottom: "2px solid #4AADAA", flexShrink: 0,
            }}>
                <button onClick={onBack} style={btn}>← Back</button>
                <button 
                    onClick={() => setCurrentIndex((i) => Math.max(0, i-1))}
                    disabled={currentIndex === 0}
                    style={{ ...btn, opacity: currentIndex === 0 ? 0.4 : 1}}
                >← Prev</button>

                <div style={{ flex: 1, display: "flex", gap: 6, overflowX: "auto", padding: "2px 0" }}>
                    {meiFiles.map((f, i) => (
                        <button
                            key={f.id}
                            onClick={() => setCurrentIndex(i)}
                            style={{
                                ...btn,
                                background: i === currentIndex ? "#4AADAA" : corrected.has(f.id) ? "#1e4d4b" : "#2d2d4e",
                                border: i === currentIndex ? "none" : "1px solid #4AADAA44",
                                flexShrink: 0,
                            }}
                        >
                            {corrected.has(f.id) ? "✓" : ""}{f.name}
                        </button>
                    ))}
                </div>

                <button
                    onClick={() => setCurrentIndex((i) => 
                        Math.min(meiFiles.length - 1, i + 1))}  
                        disabled = {currentIndex === meiFiles.length - 1}
                        style={{ ...btn, opacity: currentIndex === meiFiles.length - 1 ? 0.4 : 1}}
                >
                    Next →
                </button>

                <button
                    onClick={handleDoneAndNext}
                    style={{ ...btn, background: corrected.has (currentFile?.id ?? "") ? "#1e4d4b" : "#4AADAA", color: "white" }}
                >
                    {corrected.has(currentFile?.id ?? "") ? "✓ Done" : "Mark Done"}
                    {currentIndex < meiFiles.length - 1 ? " & Next" : ""}
                </button>

                <button
                    onClick={allCorrected ? onFinish : undefined}
                    disabled={!allCorrected}
                    style={{ ...btn, background: allCorrected ? "#22c55e" : "#2d2d4e", opacity: allCorrected ? 1 : 0.4, cursor: allCorrected ? "pointer" : "not-allowed" }}
                >Finish All</button>
            </div>

            {loading ? (
                <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#4AADAA" }}>
                    Preparing editor...
                </div>
            ) : currentSession ? (
                <iframe
                    ref={iframeRef}
                    key={currentSession.session_id}
                    src={`/neon/editor.html?manifest=${currentSession.session_id}`}
                    style={{ flex: 1, border: "none", width: "100%" }}
                    title={`Neon editor - ${currentFile?.name ?? ""}`}
                />
            ) : (
                <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center", color: "#ef4444" }}>
                    Failed to load editor for this file.
                </div>
            )}
        </div>
    );
}