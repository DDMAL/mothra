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
  const [threshold, setThreshold] = useState(0.5);
  const [device, setDevice] = useState<InferenceSettings["device"]>("cpu");
  const [modelPreset, setModelPreset] = useState<ModelPreset>("medieval");
  const [customModelId, setCustomModelId] = useState("");
  const [useSharedDetectorSettings, setUseSharedDetectorSettings] =
    useState(true);
  const [textMusicSettings, setTextMusicSettings] = useState<DetectorSettings>({
    threshold: 0.5,
    device: "cpu",
  });
  const [staveSettings, setStaveSettings] = useState<DetectorSettings>({
    threshold: 0.5,
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
