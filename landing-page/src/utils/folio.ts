import type { ProjectImage } from "../types";

export type FolioReviewStatus =
  | "match"
  | "no-detection"
  | "mismatch"
  | "not-in-source"
  | "duplicate";

export interface FolioReviewRow {
  fileName: string;
  positionalFolio?: string;
  detectedFolio?: string;
  status: FolioReviewStatus;
}

/** [numeric folio, recto(0)/verso(1)] — mirrors mothra-text/steps/nw_chant_allocator.py's _folio_sort_key.
 * The recto/verso letter is optional: some CantusDB sources record a bare folio
 * number with no side marker at all (e.g. "096" rather than "096r"/"096v"). */
export function folioSortKey(folio: string): [number, number] {
  const m = /^0*(\d+)([rv])?/i.exec(folio.trim());
  if (!m) return [Number.MAX_SAFE_INTEGER, 0]; // unparsable - sort last, not first
  return [Number(m[1]), m[2]?.toLowerCase() === "v" ? 1 : 0];
}

export function compareFolios(a?: string, b?: string): number {
  if (!a && !b) return 0;
  if (!a) return 1; // untagged images sort after tagged ones
  if (!b) return -1;
  const [an, as] = folioSortKey(a);
  const [bn, bs] = folioSortKey(b);
  return an !== bn ? an - bn : as - bs;
}

/** The other image (if any) in `images` already tagged with `folio`, excluding `excludeImageId` (so an edit doesn't warn against itself). */
export function findFolioConflict(
  images: ProjectImage[],
  folio: string,
  excludeImageId?: string,
): ProjectImage | undefined {
  if (!folio) return undefined;
  return images.find((img) => img.folio === folio && img.id !== excludeImageId);
}

/** The ProjectImage backing `imageName` — used by tabs (annotations/text/mei) to
 * look up the source/folio of a derived asset for source-sectioned grouping. */
export function findImageByName(
  images: ProjectImage[],
  imageName?: string,
): ProjectImage | undefined {
  return images.find((img) => img.name === imageName);
}

/** Sorts `items` by their backing image's source, then folio, mirroring how
 * the images tab groups/orders by source — for use with items that only
 * carry an `imageName` back-reference (annotations, text-alignments, mei files). */
export function sortBySourceThenFolio<T>(
  items: T[],
  images: ProjectImage[],
  imageNameOf: (item: T) => string | undefined,
): T[] {
  return [...items].sort((a, b) => {
    const imgA = findImageByName(images, imageNameOf(a));
    const imgB = findImageByName(images, imageNameOf(b));
    const bySource = (imgA?.sourceName || "￿").localeCompare(
      imgB?.sourceName || "￿",
    );
    return bySource !== 0 ? bySource : compareFolios(imgA?.folio, imgB?.folio);
  });
}

/** The group-header label for an item, matching the images tab's "no source" fallback. */
export function sourceGroupLabel(
  images: ProjectImage[],
  imageName?: string,
): string {
  return findImageByName(images, imageName)?.sourceName || "no source";
}

function lastFolioMatch(base: string): RegExpExecArray | undefined {
  const re = /(?:^|[^a-z0-9])0*(\d{1,4})[-_\s]?([rv])(?:$|[^a-z0-9])/gi;
  let match: RegExpExecArray | null;
  let last: RegExpExecArray | undefined;
  while ((match = re.exec(base)) !== null) {
    last = match;
    // step back one char so an immediately-adjacent following token can still be found
    re.lastIndex = match.index + match[0].length - 1;
  }
  return last;
}

export function extractFolioFromFilename(filename: string): string | undefined {
  const base = filename.replace(/\.[^.]+$/, "");
  const last = lastFolioMatch(base);
  if (!last) return undefined;
  return `${last[1]}${last[2].toLowerCase()}`;
}

/** Best-effort project-name suggestion from an uploaded image's filename —
 * strips the trailing folio token `extractFolioFromFilename` finds (plus its
 * separator), e.g. "Ch._Fco_002r.jpg" -> "Ch. Fco". Falls back to the bare
 * filename (minus extension) when there's no recognizable folio token, or
 * when nothing is left after stripping it, so this always returns something
 * usable rather than requiring callers to handle an empty/undefined case. */
export function suggestProjectNameFromFilename(filename: string): string {
  const base = filename.replace(/\.[^.]+$/, "");
  const last = lastFolioMatch(base);
  // Only strip the folio token when it ends the filename — otherwise it's
  // a folio-shaped substring in the middle (e.g. "Ch_002r_copy.jpg") and
  // slicing at its index would discard trailing text ("copy") too.
  const prefix =
    last && last.index + last[0].length === base.length
      ? base.slice(0, last.index)
      : base;
  return (
    prefix.replace(/[_]+/g, " ").replace(/\s+/g, " ").trim() || base.trim()
  );
}

/** Resolves `candidate` (as extracted from a filename) to the EXACT canonical
 * string in `folios` that Cantus DB uses for that folio — required because
 * downstream text-alignment matches folios by exact string, not by numeric
 * equivalence (so filename "2r" must resolve to Cantus's "002r", not stay
 * "2r"). Tries an exact (trimmed, case-insensitive) match first, then falls
 * back to numeric equivalence via folioSortKey. Never fuzzy-matches when
 * either side is unparsable by folioSortKey (some real Cantus foliation,
 * e.g. "005bisr", doesn't parse and would otherwise sentinel-collide with
 * every other unparsable folio in the list). */
export function matchCanonicalFolio(
  folios: string[],
  candidate: string,
): string | undefined {
  const trimmed = candidate.trim();
  const exact = folios.find(
    (f) => f.trim().toLowerCase() === trimmed.toLowerCase(),
  );
  if (exact) return exact;
  const [num, side] = folioSortKey(trimmed);
  if (num === Number.MAX_SAFE_INTEGER) return undefined;
  return folios.find((f) => {
    const [fn, fs] = folioSortKey(f);
    if (fn === Number.MAX_SAFE_INTEGER) return false;
    return fn === num && fs === side;
  });
}
