import { useEffect, useMemo, useState } from "react";
import type { IcSessionSummary, ProjectImage } from "../../types";
import { apiFetch } from "../../lib/apiFetch";
import { formatRelativeTime } from "../../utils/time";
import { AuthImage } from "../shared/AuthImage";
import Modal from "../shared/Modal";
import type { IcResumeRequest } from "./IcSessionsModal";

interface IcSessionPickerProps {
  projectId: number;
  /** *Every* image in the project, not just the pages still pending -- a
   * session worth reopening usually belongs to a page that has already been
   * encoded, which is exactly what pendingIcImages() filters out. */
  images: ProjectImage[];
  onClose: () => void;
  /** The picked sessions, in list order. The host reopens the IC step with
   * all of their pages on the filmstrip. */
  onOpen: (requests: IcResumeRequest[]) => void;
}

const stemOf = (name: string) => name.replace(/\.[^.]+$/, "");

/** Why a listed session can't be reopened, or null when it can. Mirrors
 * AppRouter's own resolution rule exactly, including its refusal to fall back
 * to a name match when an image id was recorded but no longer resolves: that
 * means the page was deleted, and a stem match could land on a different
 * image that reuses the filename -- pairing this session's classifications
 * with the wrong folio at encode time (see IcSessionUnavailable). */
function blockedReason(
  session: IcSessionSummary,
  image: ProjectImage | null,
): string | null {
  if (session.state === "export")
    return "completed in the classifier — can't be reopened";
  if (!image)
    return session.imageId
      ? "its page is no longer in this project"
      : "doesn't record which page it belongs to";
  return null;
}

// Pick one or more saved IC sessions to reopen together. Distinct from
// IcSessionsModal, which iframes IC's own management page: that one lists and
// deletes, and opens a single session at a time. The filmstrip holds as many
// pages as you like, so reopening a batch of them (a folio range corrected in
// one sitting) shouldn't mean walking back out to the project page per page.
// Resolving each session against the project's own images is also what lets
// this show the actual page thumbnail and refuse the sessions AppRouter would
// have to reject anyway.
export default function IcSessionPicker({
  projectId,
  images,
  onClose,
  onOpen,
}: IcSessionPickerProps) {
  const [sessions, setSessions] = useState<IcSessionSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [selected, setSelected] = useState<Set<string>>(new Set());

  useEffect(() => {
    let cancelled = false;
    apiFetch(`/api/projects/${projectId}/ic/sessions`)
      .then(async (r) => {
        if (!r.ok) {
          const d = await r.json().catch(() => ({}));
          throw new Error(
            (d as { detail?: string }).detail || `request failed (${r.status})`,
          );
        }
        return r.json();
      })
      .then((d: IcSessionSummary[]) => {
        if (!cancelled) setSessions(Array.isArray(d) ? d : []);
      })
      .catch((e) => {
        if (!cancelled) setError(e instanceof Error ? e.message : String(e));
      });
    return () => {
      cancelled = true;
    };
  }, [projectId]);

  const rows = useMemo(
    () =>
      (sessions ?? []).map((session) => {
        const image =
          (session.imageId
            ? images.find((img) => img.id === session.imageId)
            : images.find((img) => stemOf(img.name) === session.sourceName)) ??
          null;
        return { session, image, blocked: blockedReason(session, image) };
      }),
    [sessions, images],
  );
  const openable = rows.filter((r) => !r.blocked);

  const toggle = (sessionId: string) =>
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(sessionId)) next.delete(sessionId);
      else next.add(sessionId);
      return next;
    });

  const handleOpen = () => {
    const requests = openable
      .filter((r) => selected.has(r.session.sessionId))
      .map(({ session, image }) => ({
        sessionId: session.sessionId,
        // Send the resolved image's own id rather than echoing the session's:
        // for a name-matched session (no id recorded) this is the only way
        // the host lands on the same page this list showed.
        imageId: image?.id ?? session.imageId ?? null,
        sourceName: session.sourceName,
      }));
    if (requests.length > 0) onOpen(requests);
  };

  return (
    <Modal onClose={onClose} size="2xl" backdrop="dark">
      <div>
        <h2 className="text-xl font-bold italic text-[#1D3335]">
          reopen saved sessions
        </h2>
        <p className="mt-1 text-xs text-[#1D3335]/60">
          Pick the pages to put back on the filmstrip. Their classifications are
          exactly as you left them — encoding a page doesn't end its session.
        </p>
      </div>

      <div className="max-h-[52vh] overflow-y-auto rounded-2xl bg-white p-2">
        {error ? (
          <p className="px-3 py-8 text-center text-sm text-red-600">{error}</p>
        ) : sessions === null ? (
          <p className="px-3 py-8 text-center text-sm text-[#1D3335]/50">
            loading sessions…
          </p>
        ) : rows.length === 0 ? (
          <p className="px-3 py-8 text-center text-sm text-[#1D3335]/50">
            no saved sessions for this project yet
          </p>
        ) : (
          <ul className="flex flex-col gap-1">
            {rows.map(({ session, image, blocked }) => {
              const label =
                image?.name ?? session.sourceName ?? session.sessionId;
              const isSelected = selected.has(session.sessionId);
              return (
                <li key={session.sessionId}>
                  <label
                    className={`flex items-center gap-3 rounded-xl px-3 py-2 ${
                      blocked
                        ? "opacity-50"
                        : `cursor-pointer hover:bg-[#C8E6E3]/40 ${isSelected ? "bg-[#C8E6E3]/60" : ""}`
                    }`}
                    title={blocked ?? undefined}
                  >
                    <input
                      type="checkbox"
                      disabled={!!blocked}
                      checked={isSelected}
                      onChange={() => toggle(session.sessionId)}
                      className="h-4 w-4 shrink-0 accent-[#4AADAA] disabled:cursor-not-allowed"
                    />
                    <div className="h-10 w-10 shrink-0 overflow-hidden rounded-lg bg-[#C8E6E3]/60">
                      {image && (
                        <AuthImage
                          src={`/api/images/${image.id}`}
                          alt={label}
                          className="h-full w-full object-cover"
                        />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-semibold text-[#1D3335]">
                        {label}
                      </p>
                      <p className="truncate text-xs text-[#1D3335]/60">
                        {blocked ??
                          [
                            session.glyphCount != null
                              ? `${session.glyphCount} glyph${session.glyphCount === 1 ? "" : "s"}`
                              : null,
                            session.updatedAt
                              ? formatRelativeTime(session.updatedAt)
                              : null,
                          ]
                            .filter(Boolean)
                            .join(" · ")}
                      </p>
                    </div>
                  </label>
                </li>
              );
            })}
          </ul>
        )}
      </div>

      <div className="flex items-center justify-between gap-3">
        <button
          onClick={() =>
            setSelected((prev) =>
              prev.size === openable.length
                ? new Set()
                : new Set(openable.map((r) => r.session.sessionId)),
            )
          }
          disabled={openable.length === 0}
          className="text-xs text-[#1D3335]/60 hover:text-[#1D3335] cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
        >
          {selected.size === openable.length && openable.length > 0
            ? "clear selection"
            : "select all reopenable"}
        </button>
        <div className="flex items-center gap-3">
          <button
            onClick={onClose}
            className="px-5 py-2 text-sm font-semibold text-[#1D3335]/70 hover:text-[#1D3335] cursor-pointer"
          >
            cancel
          </button>
          <button
            onClick={handleOpen}
            disabled={selected.size === 0}
            className="px-5 py-2 bg-[#1D3335] text-white rounded-xl hover:opacity-90 cursor-pointer text-sm font-semibold disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {selected.size > 1
              ? `open ${selected.size} pages`
              : selected.size === 1
                ? "open 1 page"
                : "open"}
          </button>
        </div>
      </div>
    </Modal>
  );
}
