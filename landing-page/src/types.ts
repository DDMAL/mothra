export interface ProjectImage {
  id: string;
  name: string;
  src?: string;
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
  deletedAt?: number;
  lastOpenedAt?: string;
  isPinned?: boolean;
}
export interface ProjectModel {
  id: string;
  name: string;
}

export interface AnnotationSet {
  id: string;
  imageName: string;
  imageSrc?: string;
  jsonName: string;
  txtName: string;
  detectionCount?: number;
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
  | "ic-processing"
  | "ic-completion"
  | "encoding-processing"
  | "encoding-completion"
  | "sending"
  | "send-completion"
  | "neon-editor"
  | "neon-test";
