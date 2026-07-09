import { useState } from "react";

export interface InferenceSettings {
  threshold: number;
  device: "cpu" | "cuda" | "mps";
}

export function useInferenceSettings() {
  const [threshold, setThreshold] = useState(0.5);
  const [device, setDevice] = useState<InferenceSettings["device"]>("cpu");

  const patch = (p: Partial<InferenceSettings>) => {
    if (p.threshold !== undefined) setThreshold(p.threshold);
    if (p.device !== undefined) setDevice(p.device);
  };

  return { threshold, device, patch };
}
