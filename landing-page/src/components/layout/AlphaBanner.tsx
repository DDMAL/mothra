import { useState } from "react";

const DISMISS_KEY = "mothra_alpha_banner_dismissed_v1";

// mothra#290: a persistent notice that Mothra is still an alpha, so bugs
// and rough edges are expected rather than surprising. Dismissible
// per-browser (localStorage, not per-session/per-view) -- re-showing this
// on every single page load once a user has already acknowledged it would
// just be noise. The key is versioned (_v1) so a future, more important
// revision of this notice can still reach someone who dismissed an earlier
// wording, without needing to touch any other app state.
export default function AlphaBanner() {
  const [dismissed, setDismissed] = useState(() => {
    try {
      return localStorage.getItem(DISMISS_KEY) === "1";
    } catch {
      // localStorage can throw in a locked-down browser context (private
      // mode with storage blocked, etc.) -- fail open (show the banner)
      // rather than crash the whole app over a dismiss preference.
      return false;
    }
  });

  if (dismissed) return null;

  const handleDismiss = () => {
    setDismissed(true);
    try {
      localStorage.setItem(DISMISS_KEY, "1");
    } catch {
      // See above -- dismissal for this load still works even if it can't
      // be remembered for next time.
    }
  };

  return (
    <div className="bg-[#FDF3D9] border-b border-[#D97706]/30 px-4 py-2 flex items-center justify-center gap-3 text-sm text-[#7C4A03]">
      <span className="text-center leading-snug">
        <strong className="font-semibold">alpha version</strong> — mothra is
        still under active development and testing. Expect bugs, rough
        edges, and occasional breaking changes.{" "}
        <a
          href="https://github.com/DDMAL/mothra/issues/new"
          target="_blank"
          rel="noopener noreferrer"
          className="underline hover:opacity-70"
        >
          report an issue
        </a>
      </span>
      <button
        onClick={handleDismiss}
        aria-label="dismiss alpha notice"
        className="text-[#7C4A03]/70 hover:text-[#7C4A03] leading-none cursor-pointer shrink-0"
      >
        ✕
      </button>
    </div>
  );
}
