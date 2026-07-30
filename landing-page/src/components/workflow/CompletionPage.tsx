import { downloadBlob } from "../../utils/download";

interface CompletionPageProps {
  onContinue?: () => void;
  continueHref?: string;
  onBackToProject: () => void;
  description?: string;
  continueLabel?: string;
  continueDisabled?: boolean;
  errorText?: string | null;
  logsFileName?: string;
  logContent?: string;
  onDownloadMei?: () => void;
  onDownloadManifest?: () => void;
  onDownloadAnnotations?: () => void;
  onDownloadAnnotationsJson?: () => void;
  onDownloadZip?: () => void;
  onCompare?: () => void;
  onClassifyMore?: () => void;
  classifyMoreCount?: number;
}

export default function CompletionPage({
  onContinue,
  continueHref,
  onBackToProject,
  description = "images have successfully been normalized and initially annotated. you can now view annotations on the project page!",
  continueLabel = "continue to IC",
  continueDisabled = false,
  errorText,
  logsFileName,
  logContent,
  onDownloadMei,
  onDownloadManifest,
  onDownloadAnnotations,
  onDownloadAnnotationsJson,
  onDownloadZip,
  onCompare,
  onClassifyMore,
  classifyMoreCount,
}: CompletionPageProps) {
  const handleDownloadLogs = () => {
    downloadBlob(new Blob([logContent ?? ""], { type: "text/plain" }), logsFileName!);
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
              disabled={continueDisabled}
              className="px-10 py-4 bg-white text-[#1D3335] font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer disabled:opacity-50 disabled:cursor-default"
            >
              {continueLabel}
            </button>
          )}
          {onClassifyMore && (
            <button
              onClick={onClassifyMore}
              className="px-10 py-4 border-2 border-white text-white font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
            >
              classify more{classifyMoreCount != null ? ` (${classifyMoreCount})` : ""}
            </button>
          )}
          <button
            onClick={onBackToProject}
            className="px-10 py-4 border-2 border-white text-white font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
          >
            back to project
          </button>
        </div>
        {onCompare && (
          <button
            onClick={onCompare}
            className="text-white/70 text-sm hover:text-white cursor-pointer underline"
          >
            compare before &amp; after →
          </button>
        )}
        {errorText && (
          <p className="text-red-100 text-sm">{errorText}</p>
        )}
      </div>
      {logsFileName && (
        <button
          onClick={handleDownloadLogs}
          className="absolute bottom-8 left-8 text-white/60 text-sm hover:text-white cursor-pointer"
        >
          &gt; download {logsFileName}
        </button>
      )}
      {(onDownloadAnnotations || onDownloadAnnotationsJson || onDownloadZip) && (
        <div className="absolute bottom-8 right-8 flex flex-col items-end gap-2">
          {onDownloadAnnotations && (
            <button
              onClick={onDownloadAnnotations}
              className="text-white/60 text-sm hover:text-white cursor-pointer"
            >
              &gt; download annotations (.txt)
            </button>
          )}
          {onDownloadAnnotationsJson && (
            <button
              onClick={onDownloadAnnotationsJson}
              className="text-white/60 text-sm hover:text-white cursor-pointer"
            >
              &gt; download annotations (.json)
            </button>
          )}
          {onDownloadZip && (
            <button
              onClick={onDownloadZip}
              className="text-white/60 text-sm hover:text-white cursor-pointer"
            >
              &gt; download batch (.zip)
            </button>
          )}
        </div>
      )}
      {onDownloadMei && (
        <button
          onClick={onDownloadMei}
          className="absolute bottom-8 right-8 text-white/60 text-sm hover:text-white cursor-pointer"
        >
          &gt; download mei file
        </button>
      )}
      {onDownloadManifest && (
        <button
          onClick={onDownloadManifest}
          className="absolute bottom-16 right-8 text-white/60 text-sm hover:text-white cursor-pointer"
        >
          &gt; download neon manifest
        </button>
      )}
    </div>
  );
}
