import { useState } from "react";

export type ModelPreset = "medieval" | "printed" | "custom";

export interface DetectorSettings {
  threshold: number;
  device: "cpu" | "cuda" | "mps";
}
export interface InferenceSettings {
  threshold: number;
  device: "cpu" | "cuda" | "mps";
  modelPreset: ModelPreset;
  customModelId: string;
  useSharedDetectorSettings: boolean; // true = both medieval detectors use threshold/device above
  textMusicSettings: DetectorSettings;
  staveSettings: DetectorSettings;
}

export function useInferenceSettings() {
  const [threshold, setThreshold] = useState(0.25); // #242: was 0.5, now matches mothra-text
  const [device, setDevice] = useState<InferenceSettings["device"]>("cpu");
  const [modelPreset, setModelPreset] = useState<ModelPreset>("medieval");
  const [customModelId, setCustomModelId] = useState("");
  const [useSharedDetectorSettings, setUseSharedDetectorSettings] =
    useState(true);
  const [textMusicSettings, setTextMusicSettings] = useState<DetectorSettings>({
    threshold: 0.25, // #242: was 0.5, now matches mothra-text
    device: "cpu",
  });
  // SF-1 fix (primary cause of #213): stave-class confidence needed its own
  // default, decoupled from the shared text/music one (0.5 at the time,
  // now also 0.25 per #242) -- 0.25 matches staff-finding's own proven
  // default (see yolo_inference.DEFAULT_STAVE_CONFIDENCE's comment for the
  // measured before/after). This only affects the pre-filled value shown
  // when a user unchecks "use shared detector settings" and customizes
  // stave threshold explicitly; the default (shared, unchanged) flow
  // already sends `null` for stave_confidence_threshold and picks up the
  // same 0.25 server-side.
  const [staveSettings, setStaveSettings] = useState<DetectorSettings>({
    threshold: 0.25,
    device: "cpu",
  });

  const patch = (p: Partial<InferenceSettings>) => {
    if (p.threshold !== undefined) setThreshold(p.threshold);
    if (p.device !== undefined) setDevice(p.device);
    if (p.modelPreset !== undefined) setModelPreset(p.modelPreset);
    if (p.customModelId !== undefined) setCustomModelId(p.customModelId);
    if (p.useSharedDetectorSettings !== undefined)
      setUseSharedDetectorSettings(p.useSharedDetectorSettings);
    if (p.textMusicSettings !== undefined)
      setTextMusicSettings(p.textMusicSettings);
    if (p.staveSettings !== undefined) setStaveSettings(p.staveSettings);
  };

  return {
    threshold,
    device,
    modelPreset,
    customModelId,
    useSharedDetectorSettings,
    textMusicSettings,
    staveSettings,
    patch,
  };
}
