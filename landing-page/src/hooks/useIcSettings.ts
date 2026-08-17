import { useState } from "react";

export type IcMode = "auto" | "manual";
export type NotationType = "square" | "hufnagel";

export interface IcSettings {
  /** "auto" skips the classifier interface entirely — every page is
   *  classified and queued with the training set below. "manual" opens the
   *  classifier and only pre-selects that training set on each page. */
  mode: IcMode;
  trainingPresets: string[];
  trainingFiles: File[];
  /** Which bundled neume-to-MEI mapping the encoder uses (see
   *  encode_to_mei.py's resolve_neume_mapping) — a project-wide choice made
   *  once here, same as mode/training set above, rather than per-page or
   *  per-encode-run (mothra#210: it used to live disconnected on each IC
   *  view's own local state, so auto mode — the default — had no way to
   *  set it at all). */
  notationType: NotationType;
}

/**
 * IC step settings, picked on the project page (see IcSettingsSection) and
 * consumed by both IC routes. Application-level state, like
 * useInferenceSettings/useTextFindingSettings — not persisted per project
 * (uploaded training files are in-memory File objects, so a DB-backed
 * selection couldn't round-trip them anyway).
 */
export function useIcSettings() {
  // Defaults to "auto" - the hands-off path is the intended default for the
  // IC step. Note this makes training data effectively required up front: the
  // auto pass can't classify without one, so ProjectDetail blocks Continue
  // until the user picks one here or switches to manual.
  const [mode, setMode] = useState<IcMode>("auto");
  const [trainingPresets, setTrainingPresets] = useState<string[]>([]);
  const [trainingFiles, setTrainingFiles] = useState<File[]>([]);
  const [notationType, setNotationType] = useState<NotationType>("square");

  return {
    mode,
    setMode,
    trainingPresets,
    setTrainingPresets,
    trainingFiles,
    setTrainingFiles,
    notationType,
    setNotationType,
    // "the user made a settings choice" — what gates auto mode (classify
    // needs a non-empty training pool) and what decides whether the manual
    // classifier gets a prefill at all.
    hasTrainingSet: trainingPresets.length + trainingFiles.length > 0,
  };
}
