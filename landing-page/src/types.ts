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

/** Which tab/sub-tab ProjectDetail should open on mount -- set by App.tsx's
 * job-done toast handler when routing "view" for a succeeded job to the tab
 * that actually holds what it produced (issue #196). Consumed once via
 * ProjectDetail's onInitialTabConsumed and then cleared, so an unrelated
 * later navigation back to the project page doesn't re-apply a stale value. */
export interface ProjectInitialTab {
  tab: "images" | "models" | "generated";
  subTab?: "annotations" | "text" | "stafflines" | "mei files";
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

// Which of tasks_encode.py's 3-tier stave-source fallback produced this
// file's zones -- see auth_api.py's mei_files migration comment for the
// full tier list. Undefined for MEI files encoded before this was tracked.
export type StaveSource =
  | "staffline_detection"
  | "yolo_annotation"
  | "glyph_estimate"
  | "glyph_estimate_unresolved_lines"
  | "glyph_estimate_synthetic_lines"
  | "placeholder_no_glyphs";

export interface MeiFile {
  id: string;
  name: string;
  xmlContent?: string;
  corrected?: boolean;
  imageName?: string;
  staveSource?: StaveSource | null;
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
