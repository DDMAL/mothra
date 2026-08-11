import { useEffect, useState } from "react";
import Modal from "../shared/Modal";
import { apiFetch } from "../../lib/apiFetch";

export interface IcResumeRequest {
  sessionId: string;
  /** mothra's project_images.id for the session's page — IC stores it when
   * the session is staged, so it maps straight back to a ProjectImage. */
  imageId: string | null;
  sourceName: string;
}

interface IcSessionsModalProps {
  projectId: number;
  onClose: () => void;
  /** Clicking an in-progress session asks the host to open it. Deliberately
   * not handled inside this modal: IC's SessionView on its own is missing the
   * filmstrip, clef controls and encode queue that make the session useful,
   * so the IC step page opens it instead. */
  onResumeSession: (req: IcResumeRequest) => void;
}

// Manage this project's saved Interactive Classifier sessions. The whole
// list/resume/delete UI already lives in IC's own SPA; we just iframe its
// project-scoped management page (see ic_api.py's /ic/manage-url and IC's
// ?manage=1 deep-link) so there's nothing to reimplement here.
export default function IcSessionsModal({
  projectId,
  onClose,
  onResumeSession,
}: IcSessionsModalProps) {
  const [icUrl, setIcUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    apiFetch(`/api/projects/${projectId}/ic/manage-url`)
      .then(async (r) => {
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(
            (d as { detail?: string }).detail || `request failed (${r.status})`,
          );
        }
        return r.json();
      })
      .then((d: { ic_url: string }) => {
        if (!cancelled) setIcUrl(d.ic_url);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  // IC's manage page posts this instead of resuming in place when it's
  // embedded (see the ic submodule's App.tsx). Accept it only from IC's own
  // origin, the same guard InteractiveClassifier uses for its messages.
  useEffect(() => {
    if (!icUrl) return;
    let origin: string | null = null;
    try {
      origin = new URL(icUrl).origin;
    } catch {
      origin = null;
    }
    function onMessage(e: MessageEvent) {
      if (origin && e.origin !== origin) return;
      const data = e.data;
      if (
        data?.type === "ic:resume-session" &&
        typeof data.sessionId === "string"
      ) {
        onResumeSession({
          sessionId: data.sessionId,
          imageId: typeof data.imageId === "string" ? data.imageId : null,
          sourceName:
            typeof data.sourceName === "string" ? data.sourceName : "",
        });
      }
    }
    window.addEventListener("message", onMessage);
    return () => window.removeEventListener("message", onMessage);
  }, [icUrl, onResumeSession]);

  return (
    <Modal onClose={onClose} size="5xl" backdrop="dark">
      <div>
        <h2 className="text-xl font-bold italic text-[#1D3335]">
          saved IC sessions
        </h2>
        <p className="mt-1 text-xs text-[#1D3335]/60">
          Resume or delete the interactive-classifier sessions saved for this
          project's pages.
        </p>
      </div>
      <div className="h-[72vh] w-full overflow-hidden rounded-2xl bg-white">
        {error ? (
          <div className="flex h-full items-center justify-center px-8 text-center text-sm text-red-600">
            {error}
          </div>
        ) : icUrl ? (
          <iframe
            src={icUrl}
            title="Saved IC sessions"
            className="h-full w-full border-0"
          />
        ) : (
          <div className="flex h-full items-center justify-center text-sm text-[#1D3335]/50">
            loading sessions…
          </div>
        )}
      </div>
    </Modal>
  );
}
