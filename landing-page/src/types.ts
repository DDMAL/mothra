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
  stafflines: StafflineSet[];
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
  siglum?: string;
  medianLineSpacing: number;
  syllableCount: number;
}

export interface StafflineSet {
  id: string;
  imageName: string;
  imageSrc?: string | null;
  staveCount?: number | null;
  modeLinesPerStave?: number | null;
  status?: string;
}

// Field names here are snake_case, matching landing-page/scripts/staffline_stage.py's
// _assemble_jsomr_records verbatim -- jsomr_json is stored/returned as-is (JSONB, not
// reshaped server-side), unlike every other camelCase field on this page.
export interface JsomrLineRecord {
  id: string;
  source: "detected" | "fallback_redetected" | "interpolated";
  bounding_box: { ulx: number; uly: number; lrx: number; lry: number } | null;
  centerline_page: { x_start: number; x_end: number; y_values: number[] };
  scale_unit: number;
  stave_id: number | null;
  rhythm_status: string | null;
  within_stave_index: number | null;
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
  | "send-completion"
  | "neon-editor"
  | "neon-completion"
  | "neon-test";
