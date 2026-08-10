import { useEffect, useState } from "react";
import { isValidFolioShape } from "../../utils/folio";

interface FolioSelectProps {
  value: string;
  options: string[];
  onChange: (v: string) => void;
  className?: string;
  placeholder?: string;
  autoFocus?: boolean;
}

const CUSTOM_VALUE = "__custom__";

/**
 * Folio picker: a `<select>` of the canonical (Cantus DB) options, plus a
 * "custom folio…" entry that swaps to a free-text input for folios Cantus DB
 * doesn't list (e.g. a phantom-continuation folio like "003r", or any folio
 * a human wants to name by hand). Exposes the same `onChange(v: string)`
 * contract a plain `<select>` would, so callers' existing conflict-detection
 * etc. keeps working unmodified regardless of which control produced the value.
 */
export default function FolioSelect({
  value,
  options,
  onChange,
  className,
  placeholder = "select folio...",
  autoFocus,
}: FolioSelectProps) {
  // Start in custom mode if we're editing a folio that's already off-canonical,
  // so re-opening a picker on a phantom-tagged image shows the typed value,
  // not a blank dropdown that appears to have lost it.
  const [customMode, setCustomMode] = useState(
    () => !!value && !options.includes(value),
  );
  const [customInput, setCustomInput] = useState(() =>
    customMode ? value : "",
  );

  // The above only derives customInput once, at mount. If the parent later
  // pushes a new `value` while we're already in custom mode (e.g. a folio
  // renumbered elsewhere, or the value cleared/reset), keep the free-text
  // input in sync with it rather than silently showing a stale value.
  useEffect(() => {
    if (customMode) setCustomInput(value);
  }, [value, customMode]);

  if (customMode) {
    const invalid = customInput !== "" && !isValidFolioShape(customInput);
    return (
      <div className="flex flex-col gap-1">
        <div className="flex items-center gap-2">
          <input
            autoFocus={autoFocus}
            value={customInput}
            onChange={(e) => {
              const v = e.target.value;
              setCustomInput(v);
              if (v === "" || isValidFolioShape(v)) onChange(v);
            }}
            placeholder="e.g. 003r"
            className={className}
          />
          <button
            type="button"
            onClick={() => {
              setCustomMode(false);
              setCustomInput("");
              onChange("");
            }}
            className="text-xs underline opacity-70 hover:opacity-100 cursor-pointer whitespace-nowrap"
          >
            back to list
          </button>
        </div>
        {invalid && (
          <p className="text-red-400 text-xs">
            should look like "003r" or "003" — digits, optionally followed by
            r/v
          </p>
        )}
      </div>
    );
  }

  return (
    <select
      autoFocus={autoFocus}
      value={value}
      onChange={(e) => {
        if (e.target.value === CUSTOM_VALUE) {
          setCustomMode(true);
          setCustomInput("");
          onChange("");
          return;
        }
        onChange(e.target.value);
      }}
      className={className}
    >
      <option value="">{placeholder}</option>
      {options.map((f) => (
        <option key={f} value={f}>
          {f}
        </option>
      ))}
      <option value={CUSTOM_VALUE}>custom folio…</option>
    </select>
  );
}
