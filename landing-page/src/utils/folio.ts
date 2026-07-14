import type { ProjectImage } from "../types";

export type FolioReviewStatus = "match" | "no-detection" | "mismatch" | "not-in-source" | "duplicate";

export interface FolioReviewRow {
    fileName: string;
    positionalFolio?: string;
    detectedFolio?: string;
    status: FolioReviewStatus;
}

/** [numeric folio, recto(0)/verso(1)] — mirrors mothra-text/steps/nw_chant_allocator.py's _folio_sort_key. */
export function folioSortKey(folio: string): [number, number] {
    const m = /^0*(\d+)([rv])/i.exec(folio.trim());
    if (!m) return [Number.MAX_SAFE_INTEGER, 0]; // unparsable - sort last, not first
    return [Number(m[1]), m[2].toLowerCase() === "r" ? 0 : 1];
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
export function findImageByName(images: ProjectImage[], imageName?: string): ProjectImage | undefined {
    return images.find((img) => img.name === imageName);
}

/** Sorts `items` by their backing image's source, then folio, mirroring how
 * the images tab groups/orders by source — for use with items that only
 * carry an `imageName` back-reference (annotations, text-alignments, mei files). */
export function sortBySourceThenFolio<T>(items: T[], images: ProjectImage[], imageNameOf: (item: T) => string | undefined): T[] {
    return [...items].sort((a, b) => {
        const imgA = findImageByName(images, imageNameOf(a));
        const imgB = findImageByName(images, imageNameOf(b));
        const bySource = (imgA?.sourceName || "￿").localeCompare(imgB?.sourceName || "￿");
        return bySource !== 0 ? bySource : compareFolios(imgA?.folio, imgB?.folio);
    });
}

/** The group-header label for an item, matching the images tab's "no source" fallback. */
export function sourceGroupLabel(images: ProjectImage[], imageName?: string): string {
    return findImageByName(images, imageName)?.sourceName || "no source";
}


/** Best-effort folio token pulled out of an arbitrary filename (not a bare
 * folio string) — e.g. "A-Gu_29_002r.jpeg" -> "002r". Searches for the LAST
 * digit-run + r/v token bounded by non-alphanumeric characters or string
 * start/end (an optional separator like "-"/"_"/space between the digits
 * and the r/v is allowed, e.g. "002-r"), on the theory that the folio is
 * usually the final identifying component before the extension. Returns
 * undefined for names with no such token (e.g. "IMG_0001.jpg", or roman-
 * numeral foliation like "ir") — callers should fall back to positional
 * assignment in that case, exactly as before this function existed. */
export function extractFolioFromFilename(filename: string): string | undefined {
    const base = filename.replace(/\.[^.]+$/, "");
    const re = /(?:^|[^a-z0-9])0*(\d{1,4})[-_\s]?([rv])(?:$|[^a-z0-9])/gi;
    let match: RegExpExecArray | null;
    let last: RegExpExecArray | null = null;
    while ((match = re.exec(base)) !== null) {
        last = match;
        // step back one char so an immediately-adjacent following token can still be found
        re.lastIndex = match.index + match[0].length - 1;
    }
    if (!last) return undefined;
    return `${last[1]}${last[2].toLowerCase()}`;
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
export function matchCanonicalFolio(folios: string[], candidate: string): string | undefined {
    const trimmed = candidate.trim();
    const exact = folios.find((f) => f.trim().toLowerCase() === trimmed.toLowerCase());
    if (exact) return exact;
    const [num, side] = folioSortKey(trimmed);
    if (num === Number.MAX_SAFE_INTEGER) return undefined;
    return folios.find((f) => {
        const [fn, fs] = folioSortKey(f);
        if (fn === Number.MAX_SAFE_INTEGER) return false;
        return fn === num && fs === side;
    });
}