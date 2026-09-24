import { useEffect, useMemo, useRef, useState } from "react";
import type {
  CuFolio,
  CuManuscript,
  CuSubmission,
  CuSubmissionStatus,
  MeiFile,
  Project,
} from "../../types";
import { apiFetch, apiFetchOrThrow } from "../../lib/apiFetch";
import { latestMeiPerImage } from "../../utils/mei";
import {
  matchCanonicalFolio,
  sortBySourceThenFolio,
} from "../../utils/folio";
import FolioSelect from "../shared/FolioSelect";
import TruncatedName from "../shared/TruncatedName";

interface Props {
  project: Project;
  onBackToProject: () => void;
  /** The existing zip export, kept as a secondary route — it is still the only
   * way to hand off anything CU will not accept. */
  onDownloadBundle: () => void;
  downloadingBundle?: boolean;
  /** Failure from the zip export, which is driven by AppRouter rather than by
   * this page -- surfaced here because this is the only place it can be
   * triggered from now. */
  bundleError?: string | null;
  /** MEI ids pre-selected on the project page, if the user came from there. */
  initialSelection?: string[];
}

/** Per-page outcome of a submit run. Deliberately local, not persisted: mothra
 * keeps no record of submissions, so anything durable has to be re-read from
 * CU (see refreshStatuses below). */
type RowOutcome =
  | { kind: "idle" }
  | { kind: "sending" }
  | { kind: "sent"; alreadyPending: boolean }
  | { kind: "failed"; message: string };

const STATUS_LABEL: Record<CuSubmissionStatus, string> = {
  PENDING: "pending review",
  PUBLISHING: "publishing…",
  PUBLISHED: "✓ published",
  CORRECTION_REQUESTED: "⚠ correction requested",
  REFUSED: "✕ refused",
  SUPERSEDED: "superseded by a newer submission",
};

const STATUS_COLOR: Record<CuSubmissionStatus, string> = {
  PENDING: "text-white/70",
  PUBLISHING: "text-white/70",
  PUBLISHED: "text-[#C8E6E3]",
  CORRECTION_REQUESTED: "text-yellow-200",
  REFUSED: "text-red-200",
  SUPERSEDED: "text-white/40",
};

export default function CuSubmissionPage({
  project,
  onBackToProject,
  onDownloadBundle,
  downloadingBundle = false,
  bundleError = null,
  initialSelection,
}: Props) {
  const [manuscripts, setManuscripts] = useState<CuManuscript[]>([]);
  const [manuscriptsError, setManuscriptsError] = useState<string | null>(null);
  const [loadingManuscripts, setLoadingManuscripts] = useState(true);
  const [manuscriptFilter, setManuscriptFilter] = useState("");
  const [manuscriptId, setManuscriptId] = useState<string>("");

  const [folios, setFolios] = useState<CuFolio[] | null>(null);
  const [foliosError, setFoliosError] = useState<string | null>(null);

  const [submissions, setSubmissions] = useState<CuSubmission[]>([]);
  const [comment, setComment] = useState("");

  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(initialSelection ?? []),
  );
  const [folioByMei, setFolioByMei] = useState<Record<string, string>>({});
  const [outcomes, setOutcomes] = useState<Record<string, RowOutcome>>({});
  const [running, setRunning] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  // Monotonic guard, as in AppRouter's handleSendToCantus: a folio list or
  // status refresh that resolves after the user has moved to another
  // manuscript must not overwrite the current one.
  const folioReqRef = useRef(0);

  // One page per image, newest revision — the same dedupe the Neon editor
  // uses. mei_files is append-only, so without this a re-encoded page would
  // appear twice and the stale revision could be the one submitted.
  const pages = useMemo(() => {
    const latest = latestMeiPerImage(project.meiFiles ?? []);
    return sortBySourceThenFolio(latest, project.images, (f) => f.imageName);
  }, [project.meiFiles, project.images]);

  const folioOptions = useMemo(
    () => (folios ?? []).map((f) => f.number),
    [folios],
  );
  // CU refuses a deposit for a folio with no image_uri, so those are offered
  // but flagged rather than silently failing mid-run.
  const unmappedFolios = useMemo(
    () =>
      new Set(
        (folios ?? []).filter((f) => !f.image_uri).map((f) => f.number),
      ),
    [folios],
  );

  useEffect(() => {
    let cancelled = false;
    apiFetchOrThrow("/api/cu/manuscripts")
      .then((r) => r.json())
      .then((data: CuManuscript[]) => {
        if (cancelled) return;
        setManuscripts(data);
        setManuscriptsError(null);
      })
      .catch((e: unknown) => {
        if (cancelled) return;
        setManuscripts([]);
        setManuscriptsError(
          e instanceof Error ? e.message : "could not reach cantus ultimus",
        );
      })
      .finally(() => {
        if (!cancelled) setLoadingManuscripts(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  // Each page's folio, defaulted from the image it was encoded from and
  // matched against CU's own spelling — CU canonicalizes "1r" to "001r", so an
  // exact string compare would leave most pages looking unmatched.
  //
  // Derived at render rather than synced into state by an effect: only the
  // user's explicit overrides live in folioByMei, so changing manuscript (and
  // with it the whole folio vocabulary) re-defaults every untouched row for
  // free, with no stale entries to clear.
  const effectiveFolios = useMemo(() => {
    const map = new Map<string, string>();
    for (const page of pages) {
      const override = folioByMei[page.id];
      if (override) {
        map.set(page.id, override);
        continue;
      }
      const image = project.images.find(
        (i) => i.id === page.imageId || i.name === page.imageName,
      );
      const guess = image?.folio
        ? matchCanonicalFolio(folioOptions, image.folio)
        : undefined;
      if (guess) map.set(page.id, guess);
    }
    return map;
  }, [pages, folioByMei, folioOptions, project.images]);

  async function loadManuscript(id: string) {
    setManuscriptId(id);
    setFolios(null);
    setFoliosError(null);
    setSubmissions([]);
    if (!id.trim()) return;
    const req = ++folioReqRef.current;
    try {
      const r = await apiFetchOrThrow(
        `/api/cu/manuscripts/${encodeURIComponent(id.trim())}/folios`,
      );
      const data: CuFolio[] = await r.json();
      if (folioReqRef.current !== req) return;
      setFolios(data);
      if (data.length === 0) {
        setFoliosError(
          "this manuscript has no folios on cantus ultimus yet — its chants " +
            "need importing there first",
        );
      }
    } catch (e) {
      if (folioReqRef.current !== req) return;
      setFolios([]);
      setFoliosError(e instanceof Error ? e.message : "could not load folios");
    }
    refreshStatuses();
  }

  async function refreshStatuses() {
    // Status lives only on CU; a failure here must not block submitting, so it
    // is swallowed rather than surfaced as a run error.
    const r = await apiFetch("/api/cu/submissions");
    if (!r.ok) return;
    setSubmissions(await r.json());
  }

  useEffect(() => {
    refreshStatuses();
  }, []);

  /** CU's latest record for a page, matched on (manuscript, folio). This is
   * the whole cost of keeping no submissions table on mothra's side: there is
   * no id linking a page to its submission, only the folio number. */
  function statusFor(meiId: string): CuSubmission | undefined {
    const folio = effectiveFolios.get(meiId);
    if (!folio || !manuscriptId) return undefined;
    return submissions.find(
      (s) =>
        String(s.manuscript_id) === manuscriptId.trim() &&
        matchCanonicalFolio([s.folio_number], folio) !== undefined,
    );
  }

  async function submitOne(page: MeiFile): Promise<"ok" | "failed" | "stop"> {
    const folio = effectiveFolios.get(page.id);
    if (!folio) {
      setOutcomes((o) => ({
        ...o,
        [page.id]: { kind: "failed", message: "no folio chosen" },
      }));
      return "failed";
    }
    setOutcomes((o) => ({ ...o, [page.id]: { kind: "sending" } }));
    try {
      const r = await apiFetchOrThrow(
        `/api/projects/${project.id}/mei/${page.id}/cu-submit`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            manuscriptId: Number(manuscriptId),
            folioNumber: folio,
            comment: comment.trim() || undefined,
          }),
        },
      );
      const record: CuSubmission = await r.json();
      setOutcomes((o) => ({
        ...o,
        [page.id]: { kind: "sent", alreadyPending: !!record.alreadyPending },
      }));
      return "ok";
    } catch (e) {
      const message = e instanceof Error ? e.message : "submission failed";
      setOutcomes((o) => ({ ...o, [page.id]: { kind: "failed", message } }));
      // A throttle applies to every subsequent page too, so continuing would
      // just convert one real error into forty identical ones. Every other
      // failure is page-specific (wrong folio, unmapped folio, bad MEI) and
      // must not stop the pages after it.
      return /rate-limit|throttl/i.test(message) ? "stop" : "failed";
    }
  }

  async function submit(pagesToSend: MeiFile[]) {
    if (!manuscriptId.trim()) {
      setRunError("choose a manuscript first");
      return;
    }
    if (pagesToSend.length === 0) {
      setRunError("select at least one page");
      return;
    }
    setRunning(true);
    setRunError(null);
    let stopped = false;
    for (const page of pagesToSend) {
      const result = await submitOne(page);
      if (result === "stop") {
        stopped = true;
        setRunError(
          "stopped early — cantus ultimus is rate-limiting deposits. the " +
            "pages already sent are filed; try the rest again later.",
        );
        break;
      }
    }
    setRunning(false);
    if (!stopped) await refreshStatuses();
  }

  const visibleManuscripts = useMemo(() => {
    const q = manuscriptFilter.trim().toLowerCase();
    if (!q) return manuscripts;
    return manuscripts.filter((m) =>
      `${m.name ?? ""} ${m.siglum ?? ""} ${m.id}`.toLowerCase().includes(q),
    );
  }, [manuscripts, manuscriptFilter]);

  const selectedPages = pages.filter((p) => selected.has(p.id));
  const canSubmit = !!manuscriptId.trim() && !running;

  function toggle(id: string) {
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }

  return (
    <div className="animate-fade-in flex-1 bg-[#4AADAA] flex flex-col items-center px-12 py-16">
      <div className="w-full max-w-5xl flex flex-col gap-6">
        <div className="flex flex-col gap-2">
          <h1 className="text-4xl font-bold italic text-white">
            submit to cantus ultimus
          </h1>
          <p className="text-[#1D3335]">
            each page is filed for review by a cantus ultimus admin. nothing
            becomes public until they publish it.
          </p>
        </div>

        {/* manuscript */}
        <div className="flex flex-col gap-2 bg-white/10 rounded-2xl p-5">
          <label className="text-white text-sm font-semibold">manuscript</label>
          {loadingManuscripts ? (
            <p className="text-white/70 text-sm">loading manuscripts…</p>
          ) : (
            <>
              <input
                value={manuscriptFilter}
                onChange={(e) => setManuscriptFilter(e.target.value)}
                placeholder="filter by name or siglum…"
                className="px-3 py-2 rounded-xl bg-white/80 text-[#1D3335] text-sm outline-none"
              />
              <select
                value={manuscriptId}
                onChange={(e) => loadManuscript(e.target.value)}
                className="px-3 py-2 rounded-xl bg-white/80 text-[#1D3335] text-sm outline-none"
              >
                <option value="">select a manuscript…</option>
                {visibleManuscripts.map((m) => (
                  <option key={m.id} value={String(m.id)}>
                    {m.siglum ? `${m.siglum} — ` : ""}
                    {m.name ?? `manuscript ${m.id}`} (id {m.id})
                  </option>
                ))}
              </select>
            </>
          )}
          {/* CU lists only PUBLIC manuscripts, but accepts deposits for
              unpublished ones — which is the normal state of a manuscript
              being OMR'd. Without this escape hatch that case is unreachable. */}
          <details className="text-white/70 text-xs">
            <summary className="cursor-pointer">
              manuscript not listed? enter its id directly
            </summary>
            <p className="mt-2 mb-1 text-white/60">
              cantus ultimus only lists published manuscripts here, but accepts
              submissions for unpublished ones too.
            </p>
            <input
              defaultValue={manuscriptId}
              onBlur={(e) => loadManuscript(e.target.value)}
              placeholder="e.g. 123723"
              className="px-3 py-2 rounded-xl bg-white/80 text-[#1D3335] text-sm outline-none"
            />
          </details>
          {manuscriptsError && (
            <p className="text-red-200 text-xs" title={manuscriptsError}>
              {manuscriptsError}
            </p>
          )}
          {foliosError && (
            <p className="text-red-200 text-xs" title={foliosError}>
              {foliosError}
            </p>
          )}
        </div>

        {/* comment */}
        <div className="flex flex-col gap-2 bg-white/10 rounded-2xl p-5">
          <label className="text-white text-sm font-semibold">
            note for the reviewer <span className="font-normal">(optional)</span>
          </label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            rows={2}
            placeholder="anything the reviewer should know about these pages…"
            className="px-3 py-2 rounded-xl bg-white/80 text-[#1D3335] text-sm outline-none resize-none"
          />
        </div>

        {/* pages */}
        <div className="rounded-2xl bg-white/10 overflow-hidden">
          <div className="grid grid-cols-[2rem_2fr_1fr_2fr] gap-2 px-4 py-2 text-xs font-semibold text-white/70 bg-white/10">
            <span />
            <span>page</span>
            <span>folio on CU</span>
            <span>status</span>
          </div>
          <div className="max-h-[40vh] overflow-y-auto divide-y divide-white/10">
            {pages.length === 0 && (
              <p className="px-4 py-6 text-white/70 text-sm">
                this project has no encoded MEI yet.
              </p>
            )}
            {pages.map((page) => {
              const outcome = outcomes[page.id] ?? { kind: "idle" };
              const cuStatus = statusFor(page.id);
              const folio = effectiveFolios.get(page.id) ?? "";
              return (
                <div
                  key={page.id}
                  className="grid grid-cols-[2rem_2fr_1fr_2fr] gap-2 px-4 py-2 text-xs text-white items-center"
                >
                  <input
                    type="checkbox"
                    checked={selected.has(page.id)}
                    onChange={() => toggle(page.id)}
                  />
                  <TruncatedName
                    name={page.imageName ?? page.name}
                    className="min-w-0"
                  />
                  <FolioSelect
                    value={folio}
                    options={folioOptions}
                    onChange={(v) =>
                      setFolioByMei((m) => ({ ...m, [page.id]: v }))
                    }
                    // Background and text colour both have to be explicit.
                    // Tailwind's Preflight gives every <select> `color: inherit`
                    // and `background-color: transparent`, so inheriting this
                    // row's `text-white` left the native option popup drawing
                    // white text on the browser's white backdrop. Same idiom as
                    // the manuscript picker above; FolioSelect forwards this to
                    // its custom-folio <input> too, which had the same problem.
                    className="text-xs px-2 py-1 rounded-lg bg-white/80 text-[#1D3335] outline-none"
                  />
                  <span className="font-mono">
                    {outcome.kind === "sending" && (
                      <span className="text-white/70">submitting…</span>
                    )}
                    {outcome.kind === "failed" && (
                      <span className="text-red-200" title={outcome.message}>
                        ✕ {outcome.message}
                      </span>
                    )}
                    {outcome.kind === "sent" && !cuStatus && (
                      <span className="text-[#C8E6E3]">
                        {outcome.alreadyPending
                          ? "already pending (unchanged)"
                          : "✓ submitted"}
                      </span>
                    )}
                    {cuStatus && outcome.kind !== "failed" && (
                      <span className={STATUS_COLOR[cuStatus.status]}>
                        {STATUS_LABEL[cuStatus.status]}
                        {cuStatus.review_note && (
                          <span
                            className="block text-white/60 not-italic"
                            title={cuStatus.review_note}
                          >
                            “{cuStatus.review_note}”
                          </span>
                        )}
                      </span>
                    )}
                    {outcome.kind === "idle" &&
                      !cuStatus &&
                      folio &&
                      unmappedFolios.has(folio) && (
                        <span className="text-yellow-200">
                          ⚠ folio not mapped to an image on CU — it will be
                          refused
                        </span>
                      )}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {runError && <p className="text-red-100 text-sm">{runError}</p>}
        {bundleError && <p className="text-red-100 text-sm">{bundleError}</p>}

        <div className="flex items-center gap-4 flex-wrap">
          <button
            onClick={() => submit(selectedPages)}
            disabled={!canSubmit || selectedPages.length === 0}
            className="px-8 py-3 bg-white text-[#1D3335] font-semibold rounded-2xl hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {running
              ? "submitting…"
              : `submit selected (${selectedPages.length})`}
          </button>
          <button
            onClick={() => submit(pages)}
            disabled={!canSubmit || pages.length === 0}
            className="px-8 py-3 border-2 border-white text-white font-semibold rounded-2xl hover:opacity-90 cursor-pointer disabled:opacity-40 disabled:cursor-not-allowed"
          >
            submit all ({pages.length})
          </button>
          <button
            onClick={refreshStatuses}
            disabled={running}
            className="text-white/70 text-sm hover:text-white cursor-pointer underline disabled:opacity-40"
          >
            refresh status
          </button>
          <div className="flex-1" />
          <button
            onClick={onDownloadBundle}
            disabled={downloadingBundle}
            className="text-white/70 text-sm hover:text-white cursor-pointer underline disabled:opacity-40"
          >
            {downloadingBundle ? "preparing zip…" : "download zip instead →"}
          </button>
          <button
            onClick={onBackToProject}
            className="px-8 py-3 border-2 border-white text-white font-semibold rounded-2xl hover:opacity-90 cursor-pointer"
          >
            back to project
          </button>
        </div>
      </div>
    </div>
  );
}
