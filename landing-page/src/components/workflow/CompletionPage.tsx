interface CompletionPageProps {
  onContinue?: () => void;
  continueHref?: string;
  onBackToProject: () => void;
  description?: string;
  continueLabel?: string;
  logsFileName?: string;
  onDownloadMei?: () => void;
  onDownloadManifest?: () => void;
}

export default function CompletionPage({
  onContinue,
  continueHref,
  onBackToProject,
  description = "images have successfully been normalized and initially annotated. you can now view annotations on the project page!",
  continueLabel = "continue to IC",
  logsFileName,
  onDownloadMei,
  onDownloadManifest,
}: CompletionPageProps) {
  const handleDownloadLogs = () => {
    const blob = new Blob([""], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = logsFileName!;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48 relative">
      <div className="flex flex-col items-center gap-6">
        <h1 className="text-5xl font-bold italic text-white">ta-da!</h1>
        <p className="text-xl text-[#1D3335]">{description}</p>
        <p className="text-sm text-white/70 -mt-3">progress saved</p>
        <div className="flex items-center gap-4">
          {continueHref ? (
            <a
              href={continueHref}
              target="_blank"
              rel="noopener noreferrer"
              className="px-10 py-4 bg-white text-[#1D3335] font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
            >
              {continueLabel}
            </a>
          ) : (
            <button
              onClick={onContinue}
              className="px-10 py-4 bg-white text-[#1D3335] font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
            >
              {continueLabel}
            </button>
          )}
          <button
            onClick={onBackToProject}
            className="px-10 py-4 border-2 border-white text-white font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
          >
            back to project
          </button>
        </div>
      </div>
      {logsFileName && (
        <button
          onClick={handleDownloadLogs}
          className="absolute bottom-8 left-8 text-white/60 text-sm hover:text-white cursor-pointer"
        >
          &gt; download {logsFileName}
        </button>
      )}
      {onDownloadMei && (
        <button
          onClick={onDownloadMei}
          className="absolute bottom-8 right-8 text-white/60 text-sm hover:text-white cursor-pointer">
            &gt; download mei file
        </button>
      )}
      {onDownloadManifest && (
        <button
          onClick={onDownloadManifest}
          className="absolute bottom-16 right-8 text-white/60 text-sm hover:text-white cursor-pointer">
            &gt; download neon manifest
        </button>
      )}
    </div>
  );
}
