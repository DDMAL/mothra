export type ModelKind = "yolo" | "segmentation" | "recognition" | "text_mask";

export interface ProjectImage {
  id: string;
  name: string;
  src?: string;
  folio?: string;
  sourceId?: string;
  sourceName?: string;
}
export interface Project {
  id: number;
  name: string;
  user: string;
  images: ProjectImage[];
  models: ProjectModel[];
  annotations: AnnotationSet[];
  meiFiles: MeiFile[];
  stepsUnlocked: number;
  usedImageNames: string[];
  usedModelNames: string[];
  usedAnnotationNames: string[];
  deletedAt?: string;
  lastOpenedAt?: string;
  isPinned?: boolean;
  textAlignments: TextAlignment[];
  cantusSourceId?: string;
}

export interface CantusSource {
  sourceId: string;
  name: string;
  folios: string[];
}

export interface TextAlignment {
  id: string;
  imageName: string;
  imageSrc?: string;
  medianLineSpacing: number;
  syllableCount: number;
}

export interface ProjectModel {
  id: string;
  name: string;
  kind: ModelKind;
  classMap?: Record<string, string> | null;
}

export interface AnnotationSet {
  id: string;
  imageName: string;
  imageSrc?: string;
  jsonName: string;
  txtName: string;
  detectionCount?: number;
  modelLabel?: string | null;
}

export interface MeiFile {
  id: string;
  name: string;
  xmlContent?: string;
  corrected?: boolean;
  imageName?: string;
}

export type View =
  | "landing"
  | "about"
  | "login"
  | "register"
  | "account"
  | "docs"
  | "projects"
  | "project"
  | "processing"
  | "completion"
  | "ic"
  | "ic-completion"
  | "encoding-processing"
  | "encoding-completion"
  | "sending"
  | "send-completion"
  | "neon-editor"
  | "neon-completion"
  | "neon-test";
