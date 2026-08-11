interface TruncatedNameProps {
  /** The (possibly long) distinguishing part of the name, e.g. a filename or folio label. */
  name: string;
  /** Always-visible text appended after `name`, not counted toward truncation (e.g. "_annotations"). */
  suffix?: string;
  /** How many trailing characters of `name` to always keep visible (default 14) — this is
   * usually where the distinguishing part (a folio number, an extension) lives, so the
   * ellipsis eats into the shared prefix instead of hiding the useful part. */
  tailLength?: number;
  className?: string;
}

// Tailwind's `truncate` ellipsizes from the end, which hides exactly the part
// (a folio number like "_010r") that distinguishes otherwise-identical, long
// manuscript filenames from each other. This instead always keeps the last
// `tailLength` characters of `name` (plus `suffix`, if any) visible, and lets
// only the shared prefix ellipsize when space runs out.
// Below this many characters, ellipsizing the head saves too little space to
// be worth splitting off its own box — and a head that short is exactly what
// triggers the two rendering bugs below, so folding it into `tail` instead
// sidesteps both.
const MIN_HEAD_LENGTH = 3;
// Never collapsed or trimmed by CSS whitespace rules, unlike a regular space.
const NBSP = " ";

export default function TruncatedName({
  name,
  suffix = "",
  tailLength = 14,
  className = "",
}: TruncatedNameProps) {
  const hasHead = name.length > tailLength + MIN_HEAD_LENGTH;
  let head = "";
  let tail = name;

  if (hasHead) {
    head = name.slice(0, name.length - tailLength);
    tail = name.slice(name.length - tailLength);

    // The split point can land on/next to whitespace in `name`. Rendered in
    // separate boxes, that space sits at the very edge of one span's own
    // line box, where the browser's normal whitespace-collapsing trims it
    // away entirely (issue #192) — fusing two words with no visible gap.
    // Trim it from both sides and reinsert a single non-breaking space so
    // exactly one space survives, same as if the string had never been split.
    const trimmedHead = head.replace(/\s+$/, "");
    const hadBoundarySpace = trimmedHead.length !== head.length || /^\s/.test(tail);
    head = trimmedHead;
    tail = tail.replace(/^\s+/, "");
    if (hadBoundarySpace) tail = NBSP + tail;
  }

  return (
    <span
      className={`inline-flex min-w-0 max-w-full overflow-hidden ${className}`}
      title={name + suffix}
    >
      {head && (
        // `min-w-0` alone lets flexbox shrink this span to a few sub-pixel-wide
        // sliver when the head is short and the overflow is only slight (e.g.
        // "Eix_611_029r.jpg" with the default 14-char tail) — too narrow to
        // paint a full "…" glyph, so the browser clips it down to nothing and
        // the head's leftover characters butt straight up against the tail
        // with no visible ellipsis at all (issue #130). A small min-width
        // floor guarantees there's always room for a legible ellipsis; if
        // that means the row no longer fully fits, the outer `overflow-hidden`
        // clips a sliver off the (still fully-legible) tail instead, which is
        // far better than a garbled head.
        <span className="overflow-hidden text-ellipsis whitespace-nowrap min-w-[1.2em]">
          {head}
        </span>
      )}
      {/* No `max-w-full` here (issue #130, round 2) — `max-width` is applied
          as a hard clamp before the flex-shrink algorithm runs, so it was
          silently overriding `shrink-0` the instant this row got narrower
          than the tail's own text: the tail — the one part meant to *always*
          stay fully visible — got squeezed down and re-ellipsized by its own
          `text-ellipsis`, on top of whatever the head was already doing.
          Dropping it lets `shrink-0` actually hold; if there's truly no room
          for both, the outer span's `overflow-hidden` clips a plain sliver
          off the tail's end instead of re-truncating it into a garbled mess. */}
      <span className="min-w-0 overflow-hidden text-ellipsis whitespace-nowrap shrink-0">
        {tail}
        {suffix}
      </span>
    </span>
  );
}
