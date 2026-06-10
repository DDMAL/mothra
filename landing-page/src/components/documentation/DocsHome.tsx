interface DocsHomeProps {
    onNavigateQuickStart: () => void;
    onNavigateWalkthrough: () => void;
}

export default function DocsHome({ onNavigateQuickStart, onNavigateWalkthrough: _onNavigateWalkthrough }: DocsHomeProps) {
    return (
        <div className="animate-fade-in px-10 pt-10 pb-16 text-[#1D3335]">
            <h1 className="text-3xl font-bold italic text-white mb-8"> mothra walkthrough </h1>

            <p className="text-center max-w-2xl mx-auto mb-8 leading-relaxed">
                Mothra is a web application that allows users to go from an image of a medieval manuscript
                to accurate MEI files, which can then be uploaded to{" "}
                <a
                href="https://cantus.simssa.ca/"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:opacity-70 transition-opacity"
                >
                Cantus Ultimus
                </a>{" "}
                to allow for music &amp; text searchability. Users can view the GitHub repository{" "}
                <a
                href="https://github.com/DDMAL/mothra"
                target="_blank"
                rel="noopener noreferrer"
                className="underline hover:opacity-70 transition-opacity"
                >
                here.
                </a>
            </p>

            <hr className="border-white/30 max-w-2xl mx-auto mb-8" />

            <h2 className="text-2xl font-bold italic text-white mb-3 text-center">quick start</h2>
            <p className="text-center max-w-xl mx-auto mb-10 leading-relaxed">
                The{" "}
                <button
                onClick={onNavigateQuickStart}
                className="underline hover:opacity-70 transition-opacity"
                >
                quick start
                </button>{" "}
                is intended as a tutorial to walk completely new users through the basics of setting up
                their experience on and using Mothra.
            </p>

            <h2 className="text-2xl font-bold italic text-white mb-3 text-center">walkthrough</h2>
            <p className="text-center max-w-xl mx-auto mb-16 leading-relaxed">
                The walkthrough has been structured as an overview of the user workflow, from image input
                to final export to cantus ultimus. For easier reference, users can go through the index tabs.
            </p>

            <div className="flex justify-end">
                <button
                onClick={onNavigateQuickStart}
                className="text-sm hover:opacity-70 transition-opacity"
                >
                continue to quick start →
                </button>
            </div>
        </div>
    );
}