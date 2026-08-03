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
export default function TruncatedName({
  name,
  suffix = "",
  tailLength = 14,
  className = "",
}: TruncatedNameProps) {
  const hasHead = name.length > tailLength;
  const head = hasHead ? name.slice(0, name.length - tailLength) : "";
  const tail = hasHead ? name.slice(name.length - tailLength) : name;

  return (
    <span
      className={`inline-flex min-w-0 max-w-full overflow-hidden ${className}`}
      title={name + suffix}
    >
      {head && (
        <span className="overflow-hidden text-ellipsis whitespace-nowrap min-w-0">
          {head}
        </span>
      )}
      <span className="min-w-0 max-w-full overflow-hidden text-ellipsis whitespace-nowrap shrink-0">
        {tail}
        {suffix}
      </span>
    </span>
  );
}
