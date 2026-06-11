import { useState } from "react";
import DocsHome from "./DocsHome";
import DocsQuickStart from "./DocsQuickStart";
import DocsWalkthrough from "./DocsWalkthrough";

type DocSection = "home" | "quickstart" | "walkthrough";

const QUICK_START_ITEMS = [
    "logging in",
    "creating a project",
    "uploading images",
    "uploading models",
    "what now?",
];

const WALKTHROUGH_ITEMS = [
    "models",
    "annotations",
    "interactive classifier",
    "neon",
    "sending mei files to cantus ultimus",
    "saving and exporting data",
    "user accounts",
];

function toSlug(s: string) {
    return s.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/-$/, "");
}

interface DocumentationProps {
    onHome?: () => void;
}

export default function Documentation({ onHome: _onHome }: DocumentationProps) {
    const [section, setSection] = useState<DocSection>("home");
    const [scrollTarget, setScrollTarget] = useState<string | null>(null);

    function navigate(sec: DocSection, target?: string) {
        setSection(sec);
        setScrollTarget(target ?? null);
        if (!target) {
            window.scrollTo({ top: 0, behavior: "smooth" });
        }
    }

    return (
        <div className="flex-1 flex bg-[#4AADAA]">
            <aside className="w-72 shrink-0 bg-[#c5dfe0] flex flex-col py-6 px-2 sticky top-14 self-start h-[calc(100vh-3.5rem)] overflow-y-auto">
                <nav className="flex flex-col gap-0.5 text-sm text-[#1D3335]">
                    <button
                        onClick={() => navigate("home")}
                        className={`text-left font-bold px-4 py-1.5 rounded-lg transition-colors ${
                            section === "home" ? "bg-[#4AADAA]/40": "hover:bg-[#4AADAA]/20"
                        }`}>
                            home
                    </button>
                    <button
                        onClick={() => navigate("quickstart")}
                        className={`text-left font-bold px-4 py-1.5 rounded-lg mt-2 transition-colors ${
                            section === "quickstart" ? "bg-[#4AADAA]/40": "hover:bg-[#4AADAA]/20"
                        }`}>
                            quick start
                    </button>
                    {QUICK_START_ITEMS.map((item) => (
                        <button
                            key={item}
                            onClick={() => navigate("quickstart", toSlug(item))}
                            className="text-center px-4 py-1 rounded-lg hover:bg-[#4AADAA]/20 transition-colors">
                            {item}
                        </button>
                    ))}
                    <button
                        onClick={() => navigate("walkthrough")}
                        className={`text-left font-bold px-4 py-1.5 rounded-lg mt-2 transition-colors ${
                        section === "walkthrough" ? "bg-[#4AADAA]/20" : "hover:bg-[#4AADAA]/20"
                        }`}
                    >
                        walkthrough:
                    </button>
                    {WALKTHROUGH_ITEMS.map((item) => (
                        <button
                            key={item}
                            onClick={() => navigate("walkthrough", toSlug(item))}
                            className="text-center px-4 py-1 rounded-lg hover:bg-[#4AADAA]/20 transition-colors">
                            {item}
                        </button>
                    ))}
                </nav>
            </aside>

            {/* main content */}
            <main className="flex-1 overflow-y-auto">
                {section === "home" && (
                    <DocsHome
                        onNavigateQuickStart={() => navigate("quickstart")}
                        onNavigateWalkthrough={() => navigate("walkthrough")}
                    />
                )}
                {section === "quickstart" && (
                    <DocsQuickStart
                        scrollTarget={scrollTarget}
                        onNavigateWalkthrough={() => navigate("walkthrough")}
                    />
                )}
                {section === "walkthrough" && <DocsWalkthrough />}
            </main>
        </div>
    );
}