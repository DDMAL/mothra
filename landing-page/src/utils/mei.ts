import type { MeiFile } from "../types";

/** How a page is identified across MEI revisions: its image id when the row
 * recorded one (duplicate-named uploads each get their own id, mothra#241),
 * else its image name. Mirrors imageStep.ts's matchesImage rule. */
const pageKeyOf = (f: MeiFile) => f.imageId || f.imageName || f.id;

/**
 * One MEI file per page — the newest revision of each.
 *
 * `mei_files` is append-only: every encode of a page INSERTs a new row (the
 * cantus-bundle export relies on that history, picking each image's latest
 * revision server-side). Correcting in Neon updates a row in place rather
 * than adding one, so a page only ever has several rows because it was
 * encoded more than once — which is exactly what reopening a saved IC session
 * and encoding it again does. Handing that raw list to the Neon batch editor
 * showed the same page twice, with the stale revision first.
 *
 * Newest wins by `createdAt`; rows without one (encoded before the column was
 * backfilled) fall back to list order, which both API queries now order by
 * `created_at` and which local state appends to. Non-destructive: the older
 * revisions stay in the project's MEI-files tab, where the history is the
 * point.
 *
 * Two revisions of one page are only recognised as such when they agree on
 * how the page is identified: an `image_id`-less legacy row and a current one
 * key differently and both survive, exactly as they did before this existed.
 * Keying on the name instead would fix that at the cost of collapsing two
 * genuinely distinct duplicate-named uploads into one — which would hide a
 * real page from the editor, a strictly worse failure than showing an extra.
 */
export function latestMeiPerImage(files: MeiFile[]): MeiFile[] {
  // firstIndex is where the page first appeared, seenIndex where the winning
  // revision did: the winner is chosen by recency, but the page keeps the
  // slot it already had, so re-encoding one page doesn't shuffle it to the
  // end of the editor's file list.
  const bestByPage = new Map<
    string,
    { file: MeiFile; firstIndex: number; seenIndex: number }
  >();
  files.forEach((file, index) => {
    const key = pageKeyOf(file);
    const prev = bestByPage.get(key);
    if (prev === undefined) {
      bestByPage.set(key, { file, firstIndex: index, seenIndex: index });
      return;
    }
    const a = file.createdAt ? Date.parse(file.createdAt) : NaN;
    const b = prev.file.createdAt ? Date.parse(prev.file.createdAt) : NaN;
    const newer =
      Number.isNaN(a) || Number.isNaN(b) ? index > prev.seenIndex : a >= b;
    if (newer)
      bestByPage.set(key, {
        file,
        firstIndex: prev.firstIndex,
        seenIndex: index,
      });
  });
  return [...bestByPage.values()]
    .sort((x, y) => x.firstIndex - y.firstIndex)
    .map((entry) => entry.file);
}
