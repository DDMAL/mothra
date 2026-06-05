interface CompletionPageProps {
  onContinue: () => void;
}

export default function CompletionPage({ onContinue }: CompletionPageProps) {
  const handleDownloadLogs = () => {
    const blob = new Blob([""], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "logs.txt";
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center justify-center px-12 py-20 pb-48 relative">
      <div className="flex flex-col items-center gap-6">
        <h1 className="text-5xl font-bold italic text-white">ta-da!</h1>
        <p className="text-xl text-[#1D3335]">
          images have successfully been normalized and initially annotated
        </p>
        <button
          onClick={onContinue}
          className="mt-2 px-10 py-4 bg-white text-[#1D3335] font-semibold text-lg rounded-2xl hover:opacity-90 cursor-pointer"
        >
          continue to IC
        </button>
      </div>
      <button
        onClick={handleDownloadLogs}
        className="absolute bottom-8 left-8 text-white/60 text-sm hover:text-white cursor-pointer"
      >
        &gt; download logs.txt
      </button>
    </div>
  );
}
