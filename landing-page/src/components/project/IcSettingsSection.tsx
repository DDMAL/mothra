import { useEffect, useState } from "react";
import { apiFetch } from "../../lib/apiFetch";
import type { IcMode, useIcSettings } from "../../hooks/useIcSettings";

interface IcSettingsSectionProps {
  icSettings: ReturnType<typeof useIcSettings>;
}

const MODES: IcMode[] = ["auto", "manual"];

/**
 * Step-1 (interactive classifier) settings, laid out in place on the project
 * page rather than behind a popover on the IC page itself: the mode switch
 * plus the shared training set (built-in presets + uploaded GameraXML) that
 * "auto" classifies every page with, and that "manual" pre-selects on each
 * page in the classifier.
 */
export default function IcSettingsSection({
  icSettings,
}: IcSettingsSectionProps) {
  const {
    mode,
    setMode,
    trainingPresets,
    setTrainingPresets,
    trainingFiles,
    setTrainingFiles,
    notationType,
    setNotationType,
  } = icSettings;
  const [availablePresets, setAvailablePresets] = useState<string[]>([]);

  // Same list IC's own create-session screen offers, so a preset picked here
  // is one the classifier will recognise.
  useEffect(() => {
    let cancelled = false;
    apiFetch("/api/ic/training-presets")
      .then((r) => (r.ok ? r.json() : []))
      .then((list) => {
        if (!cancelled) setAvailablePresets(Array.isArray(list) ? list : []);
      })
      .catch(() => {
        if (!cancelled) setAvailablePresets([]);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const totalTrainingSets = trainingPresets.length + trainingFiles.length;

  return (
    <div data-tutorial-target="ic-settings" className="flex flex-col gap-3">
      <h3 className="text-base text-white font-semibold">
        Glyph Classifier settings
      </h3>

      <div className="flex flex-col gap-1">
        <div className="inline-flex rounded-lg border border-white/30 overflow-hidden w-fit">
          {MODES.map((m) => (
            <button
              key={m}
              onClick={() => setMode(m)}
              className={`px-4 py-1 text-xs cursor-pointer transition-colors ${
                mode === m
                  ? "bg-white text-[#1D3335] font-semibold"
                  : "text-white/70 hover:bg-white/10"
              }`}
            >
              {m}
            </button>
          ))}
        </div>
        <span className="text-white/40 text-xs italic">
          {mode === "auto"
            ? "every page is classified and queued for you"
            : "you classify each page in the classifier"}
        </span>
      </div>

      <div className="flex flex-col gap-1">
        <span className="text-sm font-medium text-white/90">notation</span>
        <select
          value={notationType}
          onChange={(e) =>
            setNotationType(e.target.value as "square" | "hufnagel")
          }
          className="bg-transparent border border-white/30 rounded-lg px-2 py-1 text-xs text-white cursor-pointer w-fit"
        >
          <option value="square">square</option>
          <option value="hufnagel">hufnagel</option>
        </select>
        <span className="text-white/40 text-xs italic">
          which neume shapes encoding maps classifications to — applies to
          every page, in either mode
        </span>
      </div>

      {/* "training data" is the section; the preset picker and the upload are
          its two sub-groups, so they sit a step down in the hierarchy rather
          than reading as siblings of it. */}
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-0.5">
          <span className="text-sm font-medium text-white/90">
            training data
          </span>
          <span className="text-xs text-white/40 italic">
            {mode === "auto"
              ? "applied to every page"
              : "pre-selected on each page in the classifier"}
          </span>
        </div>

        <div className="flex flex-col gap-2 pl-3 border-l border-white/20">
          <div className="flex flex-col gap-1">
            <span className="text-[11px] uppercase tracking-wide text-white/50">
              presets
            </span>
            {availablePresets.length === 0 ? (
              <span className="text-white/50 text-xs">
                no built-in presets available
              </span>
            ) : (
              <div className="flex flex-col gap-1 max-h-40 overflow-y-auto">
                {availablePresets.map((name) => (
                  <label
                    key={name}
                    className="flex items-center gap-2 cursor-pointer text-xs text-white/80"
                  >
                    <input
                      type="checkbox"
                      checked={trainingPresets.includes(name)}
                      // Presets are mutually exclusive here for the same reason
                      // they are in IC's own picker: checking one unchecks the
                      // rest. Kept as a list because the API takes one, and an
                      // uploaded set can still be combined with a preset.
                      onChange={(e) =>
                        setTrainingPresets(e.target.checked ? [name] : [])
                      }
                      className="accent-[#1D3335]"
                    />
                    <span className="truncate">{name}</span>
                  </label>
                ))}
              </div>
            )}
          </div>

          <div className="flex flex-col gap-1">
            <label className="block">
              <span className="mb-1 block text-[11px] uppercase tracking-wide text-white/50">
                upload GameraXML (.xml)
              </span>
              <input
                type="file"
                accept=".xml"
                multiple
                onChange={(e) =>
                  setTrainingFiles(Array.from(e.target.files ?? []))
                }
                className="block w-full text-xs text-white/70 file:mr-2 file:cursor-pointer file:rounded-lg file:border-0 file:bg-[#1D3335] file:px-3 file:py-1.5 file:text-xs file:font-semibold file:text-white hover:file:opacity-90"
              />
            </label>
            {trainingFiles.length > 0 && (
              <div className="flex items-center gap-2">
                <span className="text-xs text-white/60">
                  {trainingFiles.length} file
                  {trainingFiles.length === 1 ? "" : "s"} selected
                </span>
                <button
                  onClick={() => setTrainingFiles([])}
                  className="text-white/50 hover:text-white text-[10px] underline cursor-pointer"
                >
                  clear
                </button>
              </div>
            )}
          </div>
        </div>

        {/* No warning for auto-with-no-training-data here: the Continue button
            on the right already carries that one, greyed out with the reason. */}
        <p className="text-white/40 text-xs italic">
          {totalTrainingSets > 0
            ? `${totalTrainingSets} training set${totalTrainingSets === 1 ? "" : "s"} will classify each page`
            : mode === "auto"
              ? "no training data selected yet"
              : "no training data selected — the classifier opens with nothing pre-selected"}
        </p>
      </div>
    </div>
  );
}
