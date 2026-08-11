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
  ocrOnlyMode: boolean;
  sourceId: string;
  folio: string;
  musicOverlapFilterEnabled: boolean;
  debugMode: boolean;
}

export function useTextFindingSettings() {
  const [columnCount, setColumnCount] =
    useState<TextFindingSettings["columnCount"]>("auto");
  const [segmentationModelId, setSegmentationModelId] = useState("");
  const [recognitionModelId, setRecognitionModelId] = useState("");
  const [device, setDevice] = useState<TextFindingSettings["device"]>("cpu");
  const [columnBimodalThreshold, setColumnBimodalThreshold] = useState(0.5);

  const [maskingEnabled, setMaskingEnabled] = useState(true);
  const [maskPadding, setMaskPadding] = useState(15);
  const [maskModelId, setMaskModelId] = useState("");

  const [ocrOnlyMode, setOcrOnlyMode] = useState(false);
  const [sourceId, setSourceId] = useState("");
  const [folio, setFolio] = useState("");

  const [musicOverlapFilterEnabled, setMusicOverlapFilterEnabled] =
    useState(true);
  const [debugMode, setDebugMode] = useState(false);
  const [debugDataByImage, setDebugDataByImage] = useState<
    Record<string, unknown>
  >({});

  const patch = (p: Partial<TextFindingSettings>) => {
    if (p.columnCount !== undefined) setColumnCount(p.columnCount);
    if (p.segmentationModelId !== undefined)
      setSegmentationModelId(p.segmentationModelId);
    if (p.recognitionModelId !== undefined)
      setRecognitionModelId(p.recognitionModelId);
    if (p.device !== undefined) setDevice(p.device);
    if (p.columnBimodalThreshold !== undefined)
      setColumnBimodalThreshold(p.columnBimodalThreshold);
    if (p.maskingEnabled !== undefined) setMaskingEnabled(p.maskingEnabled);
    if (p.maskPadding !== undefined) setMaskPadding(p.maskPadding);
    if (p.maskModelId !== undefined) setMaskModelId(p.maskModelId);
    if (p.ocrOnlyMode !== undefined) setOcrOnlyMode(p.ocrOnlyMode);
    if (p.sourceId !== undefined) setSourceId(p.sourceId);
    if (p.folio !== undefined) setFolio(p.folio);
    if (p.musicOverlapFilterEnabled !== undefined)
      setMusicOverlapFilterEnabled(p.musicOverlapFilterEnabled);
    if (p.debugMode !== undefined) setDebugMode(p.debugMode);
  };

  return {
    columnCount,
    segmentationModelId,
    recognitionModelId,
    device,
    columnBimodalThreshold,
    maskingEnabled,
    musicOverlapFilterEnabled,
    maskPadding,
    maskModelId,
    ocrOnlyMode,
    sourceId,
    folio,
    debugMode,
    debugDataByImage,
    setDebugDataByImage,
    patch,
  };
}
