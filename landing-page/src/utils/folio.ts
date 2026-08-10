import type { ProjectImage } from "../types";

export type FolioReviewStatus =
  | "match"
  | "no-detection"
  | "mismatch"
  | "not-in-source"
  | "duplicate"
  | "phantom-continuation";

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

/** [number, side] — mirrors mothra-text/run_chain.py's _parse_folio_id. Unlike
 * folioSortKey, side is undefined (not defaulted to recto) when the string
 * has no r/v marker at all, since contiguity rules differ for bare numbers. */
function parseFolioId(
  folioId: string,
): [number, "r" | "v" | undefined] | [undefined, undefined] {
  const m = /^(\d+)([rv]?)$/i.exec(folioId.trim());
  if (!m) return [undefined, undefined];
  const side = m[2] ? (m[2].toLowerCase() as "r" | "v") : undefined;
  return [Number(m[1]), side];
}

/** Formats (number, side) as a folio label matching `reference`'s digit
 * width and letter case (e.g. reference "002v", num=3, side="r" -> "003r").
 * Needed because extractFolioFromFilename deliberately strips leading
 * zeros from its raw guess (so it can be resolved against the canonical
 * list) - for a phantom-continuation folio there's no canonical entry to
 * resolve against, so the stripped guess must be re-padded by hand to match
 * its zero-padded siblings instead of standing out as e.g. "3r" next to "002v"/"003v". */
function formatFolioLike(
  reference: string,
  num: number,
  side: "r" | "v" | undefined,
): string {
  const m = /^(0*)(\d+)([rv]?)$/i.exec(reference.trim());
  const digitWidth = m ? m[1].length + m[2].length : String(num).length;
  const referenceSide = m?.[3] ?? "";
  const isUpper =
    referenceSide !== "" && referenceSide === referenceSide.toUpperCase();
  const sideChar = side ?? "";
  return `${String(num).padStart(digitWidth, "0")}${isUpper ? sideChar.toUpperCase() : sideChar.toLowerCase()}`;
}

/** Returns true if folio `b` directly follows folio `a` in manuscript
 * sequence (same number r->v, or number+1 v->r / bare-number+1) — port of
 * mothra-text/run_chain.py's _are_contiguous. Used to recognize a genuine
 * single-step numbering gap rather than guessing at an arbitrary jump. */
export function areFoliosContiguous(a: string, b: string): boolean {
  const [numA, sideA] = parseFolioId(a);
  const [numB, sideB] = parseFolioId(b);
  if (numA === undefined || numB === undefined) return false;
  if (sideA === undefined && sideB === undefined) return numB === numA + 1;
  if (sideA === "r" && sideB === "v") return numA === numB;
  if (sideA === "v" && sideB === "r") return numB === numA + 1;
  return false;
}

/** Shape-check for a manually-typed folio label (e.g. "003r") — the same
 * number[+side] pattern folioSortKey/parseFolioId parse. */
export function isValidFolioShape(input: string): boolean {
  return /^\s*0*\d+[rv]?\s*$/i.test(input);
}

export interface EffectiveFolioSequence {
  /** One entry per input guess - undefined where neither a canonical nor a phantom folio could be resolved. */
  sequence: (string | undefined)[];
  /** Parallel to `sequence` - true where that entry is a recognized phantom-continuation folio. */
  isPhantom: boolean[];
  /** How many entries of `canonical` were consumed - callers with more files to place (e.g. PDF pages) continue from this index. */
  canonicalConsumed: number;
}

/** Reconciles the canonical (Cantus DB) folio list against the folios
 * guessed from filenames of files about to be uploaded, producing one output
 * entry per guess.
 *
 * A guess that doesn't resolve to the next canonical folio, but is
 * contiguous with BOTH its already-accepted predecessor and that next
 * canonical entry, is a genuine single-step gap (e.g. a "phantom
 * continuation" folio like 003r that Cantus DB doesn't catalogue because no
 * chant starts there) and gets spliced in without consuming a canonical
 * slot. This is safe specifically because there IS an uploaded file for it -
 * it isn't inventing a folio, only recognizing one the user already has.
 * Anything else falls back to today's plain positional behaviour (assign the
 * next canonical folio regardless of what the filename suggests).
 *
 * A batch that starts exactly at a phantom folio, or ends exactly before its
 * next canonical neighbor is available, has no contiguity to check against
 * and can't be auto-detected here - that's the expected point where manual
 * entry (the "custom folio..." picker) takes over. */
export function buildEffectiveFolioSequence(
  canonical: string[],
  detectedGuesses: (string | undefined)[],
): EffectiveFolioSequence {
  const sequence: (string | undefined)[] = [];
  const isPhantom: boolean[] = [];
  let c = 0;
  let lastAccepted: string | undefined;

  for (const guess of detectedGuesses) {
    const nextCanonical = canonical[c];
    const resolvesToNext =
      guess !== undefined && nextCanonical !== undefined
        ? matchCanonicalFolio([nextCanonical], guess) !== undefined
        : false;

    if (resolvesToNext) {
      sequence.push(nextCanonical);
      isPhantom.push(false);
      lastAccepted = nextCanonical;
      c++;
      continue;
    }

    const isGenuineGap =
      guess !== undefined &&
      lastAccepted !== undefined &&
      nextCanonical !== undefined &&
      areFoliosContiguous(lastAccepted, guess) &&
      areFoliosContiguous(guess, nextCanonical);

    if (isGenuineGap) {
      // guess is defined here (isGenuineGap requires it), but re-derive
      // its (number, side) rather than using the raw string directly -
      // extractFolioFromFilename strips leading zeros, and there's no
      // canonical entry here to resolve that padding against.
      const [num, side] = parseFolioId(guess!);
      const formatted =
        num !== undefined ? formatFolioLike(nextCanonical, num, side) : guess!;
      sequence.push(formatted);
      isPhantom.push(true);
      lastAccepted = formatted;
      // canonical pointer NOT advanced - nextCanonical is still upcoming
      continue;
    }

    // fall back to today's positional behavior
    sequence.push(nextCanonical);
    isPhantom.push(false);
    if (nextCanonical !== undefined) {
      lastAccepted = nextCanonical;
      c++;
    }
  }

  return { sequence, isPhantom, canonicalConsumed: c };
}
