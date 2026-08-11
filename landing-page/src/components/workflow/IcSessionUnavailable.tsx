interface IcSessionUnavailableProps {
  /** IC's stored page name for the session (a file-name stem), "" if it
   * recorded none. */
  sourceName: string;
  sessionId: string;
  onBack: () => void;
  /** Enter the IC step normally, dropping the resume request. */
  onOpenClassifier: () => void;
}

// Shown in place of the classifier when a saved session was picked in "manage
// IC sessions" but the page it belongs to can't be resolved in this project -
// the image was deleted, or the session predates IC recording mothra's image
// id and its name no longer matches one.
//
// Mounting the classifier anyway is not a harmless fallback: it would open on
// whatever page is first in the list, and queueing from there pairs *this*
// session's GameraXML with that other page's image and filename
// (InteractiveClassifier's handleQueuePage builds the pair from the selected
// page), so the classifications would be encoded against the wrong folio with
// nothing to flag it. Refusing is the only safe option.
export default function IcSessionUnavailable({
  sourceName,
  sessionId,
  onBack,
  onOpenClassifier,
}: IcSessionUnavailableProps) {
  return (
    <div className="animate-fade-in flex-1 min-h-0 bg-[#4AADAA] flex flex-col items-center justify-center gap-4 px-8 text-center">
      <h1 className="text-3xl font-bold italic text-white">
        saved session unavailable
      </h1>
      <p className="max-w-xl text-sm text-white/80">
        {sourceName ? (
          <>
            the page this session was classifying (
            <span className="font-mono">{sourceName}</span>) is no longer one of
            this project's images, so the classifier can't open it against the
            right page.
          </>
        ) : (
          <>
            this session doesn't record which of this project's pages it belongs
            to, so the classifier can't open it against the right page.
          </>
        )}
      </p>
      <p className="max-w-xl text-xs text-white/50">
        opening it on another page would attach these classifications to the
        wrong image when the page is queued for encoding. delete the session
        from "manage IC sessions" on the project page, or carry on without it.
      </p>
      <p className="font-mono text-[11px] text-white/30">session {sessionId}</p>
      <div className="mt-1 flex items-center gap-3">
        <button
          onClick={onBack}
          className="px-6 py-2 bg-white text-[#1D3335] rounded-xl hover:opacity-90 cursor-pointer font-semibold"
        >
          back to project
        </button>
        <button
          onClick={onOpenClassifier}
          className="px-6 py-2 bg-[#1D3335] text-white border border-white/30 rounded-xl hover:opacity-90 cursor-pointer font-semibold"
        >
          open classifier without it
        </button>
      </div>
    </div>
  );
}
