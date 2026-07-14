import type { ProjectImage } from "../types";

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