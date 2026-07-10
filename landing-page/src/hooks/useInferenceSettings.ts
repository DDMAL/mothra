import { useState } from "react";

export type ModelPreset = "medieval" | "printed" | "custom";

export interface InferenceSettings {
  threshold: number;
  device: "cpu" | "cuda" | "mps";
  modelPreset: ModelPreset;
  customModelId: string;
}

export function useInferenceSettings() {
  const [threshold, setThreshold] = useState(0.5);
  const [device, setDevice] = useState<InferenceSettings["device"]>("cpu");
  const [modelPreset, setModelPreset] = useState<ModelPreset>("medieval");
  const [customModelId, setCustomModelId] = useState("");

  const patch = (p: Partial<InferenceSettings>) => {
    if (p.threshold !== undefined) setThreshold(p.threshold);
    if (p.device !== undefined) setDevice(p.device);
    if (p.modelPreset !== undefined) setModelPreset(p.modelPreset);
    if (p.customModelId !== undefined) setCustomModelId(p.customModelId);
  };

  return { threshold, device, modelPreset, customModelId, patch };
}
