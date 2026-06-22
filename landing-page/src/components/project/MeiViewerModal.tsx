import { useState, useRef } from "react";
import type { Project, MeiFile } from "../../types";
import { authHeaders } from "../../hooks/useAuth";
import { downloadBlob } from "../../utils/download";

type Tab = "text" | "score";
type SessionState = 
    | { status: "idle" }
    | { status: "loading" }
    | { status: "loaded"; sessionId: string }
    | { status: "error"; message: string };

interface Props {
    file: MeiFile;
    project: Project;
    onClose: () => void;
}

export default function MeiViewerModal({ file, project, onClose }: Props) {
    const [tab, setTab] = useState<Tab>("text");
    const [session, setSession] = useState<SessionState>({ status: "idle" });
    const fetched = useRef(false);

    const handleExport = () => {
        downloadBlob(
            new Blob([file.xmlContent ?? ""], { type: "application/xml" }),
            file.name,
        );
    }

    const openScoreTab = () => {
        setTab("score");
        if (fetched.current) return;
        fetched.current = true;
        if (!file.imageName) {
            setSession({ status: "error", message: "No image is associated with this MEI file." });
            return;
        }
        setSession({ status: "loading" });
        fetch(`/api/projects/${project.id}/mei/${file.id}/edit-session`, {
            method: "POST",
            headers: authHeaders(),
        })
            .then((r) => (r.ok ? r.json() : Promise.reject(r.status)))
            .then((data) => setSession({ status: "loaded", sessionId: data.session_id }))
            .catch(() => setSession({ status: "error", message: "Failed to load score view." }));
    };

    return (
        <>
            <div className="fixed top-14 inset-x-0 bottom-0 z-40 bg-black/60" onClick={onClose} />
            <div className="fixed z-50 top-[4.5rem] bottom-4 left-1/2 -translate-x-1/2 w-[calc(100vw-2rem)] max-w-5xl bg-[#C8E6E3] rounded-3xl shadow-2xl flex flex-col overflow-hidden animate-fade-in">
                <div className="flex items-center gap-4 px-6 py-3 border-b border-[#1D3335]/20 shrink-0">
                <p className="font-mono text-sm text-[#1D3335] font-semibold truncate flex-1">
                    {file.name}
                </p>
                <div className="flex gap-1 bg-white/40 rounded-xl p-1">
                    <button
                    onClick={() => setTab("text")}
                    className={`px-4 py-1 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                        tab === "text"
                        ? "bg-white text-[#4AADAA]"
                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                    }`}
                    >
                    Text
                    </button>
                    <button
                    onClick={openScoreTab}
                    className={`px-4 py-1 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                        tab === "score"
                        ? "bg-white text-[#4AADAA]"
                        : "text-[#1D3335]/60 hover:text-[#1D3335]"
                    }`}
                    >
                    Score
                    </button>
                </div>
                <button
                    onClick={handleExport}
                    className="px-4 py-1.5 bg-white text-[#1D3335] font-semibold rounded-xl hover:opacity-90 cursor-pointer text-sm"
                >
                    export
                </button>
                <button
                    onClick={onClose}
                    className="text-[#1D3335] text-lg leading-none hover:opacity-60 cursor-pointer ml-2"
                >
                    ✕
                </button>
                </div>

                <div className="flex-1 min-h-0 overflow-auto">
                {tab === "text" ? (
                    <pre className="text-[#1D3335]/80 text-xs font-mono h-full whitespace-pre-wrap p-6">
                    {file.xmlContent ?? "(no content)"}
                    </pre>
                ) : session.status === "loading" ? (
                    <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                    loading score view…
                    </div>
                ) : session.status === "error" ? (
                    <div className="flex items-center justify-center h-full text-[#1D3335]/60 text-sm">
                    {session.message}
                    </div>
                ) : session.status === "loaded" ? (
                    <iframe
                    src={`/neon/editor.html?manifest=${session.sessionId}`}
                    className="w-full h-full border-none"
                    title="MEI score viewer"
                    />
                ) : null}
                </div>
            </div>
            </>
    );
}