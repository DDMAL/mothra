import { useState } from "react";

export interface TextFindingSettings {
  columnCount: "auto" | "1" | "2";
  segmentationModelId: string;
  recognitionModelId: string;
  device: "cpu" | "cuda";
  columnBimodalThreshold: number;
  maskingEnabled: boolean;
  maskPadding: number;
  maskModelId: string;
}

export function useTextFindingSettings() {
  const [columnCount, setColumnCount] = useState<TextFindingSettings["columnCount"]>("auto");
  const [segmentationModelId, setSegmentationModelId] = useState("");
  const [recognitionModelId, setRecognitionModelId] = useState("");
  const [device, setDevice] = useState<TextFindingSettings["device"]>("cpu");
  const [columnBimodalThreshold, setColumnBimodalThreshold] = useState(0.5);

  const [maskingEnabled, setMaskingEnabled] = useState(true);
  const [maskPadding, setMaskPadding] = useState(15);
  const [maskModelId, setMaskModelId] = useState("");

  const patch = (p: Partial<TextFindingSettings>) => {
    if (p.columnCount !== undefined) setColumnCount(p.columnCount);
    if (p.segmentationModelId !== undefined) setSegmentationModelId(p.segmentationModelId);
    if (p.recognitionModelId !== undefined) setRecognitionModelId(p.recognitionModelId);
    if (p.device !== undefined) setDevice(p.device);
    if (p.columnBimodalThreshold !== undefined) setColumnBimodalThreshold(p.columnBimodalThreshold);
    if (p.maskingEnabled !== undefined) setMaskingEnabled(p.maskingEnabled);
    if (p.maskPadding !== undefined) setMaskPadding(p.maskPadding);
    if (p.maskModelId !== undefined) setMaskModelId(p.maskModelId);
  };

  return {
    columnCount, segmentationModelId, recognitionModelId, device, columnBimodalThreshold,
    maskingEnabled, maskPadding, maskModelId,
    patch,
  };
}
