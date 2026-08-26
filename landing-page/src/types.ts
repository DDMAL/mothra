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
  // mothra#241 follow-up (CodeRabbit): renamed from usedImageNames and now
  // holds project_images.id values, not names -- a name-keyed list couldn't
  // let two duplicate-named uploads be selected/removed/run independently,
  // which defeats the actual point of allowing duplicates (e.g. testing the
  // pipeline against multiple copies of the same page).
  usedImageIds: string[];
  usedModelNames: string[];
  usedAnnotationNames: string[];
  deletedAt?: string;
  lastOpenedAt?: string;
  isPinned?: boolean;
  textAlignments: TextAlignment[];
  cantusSourceId?: string;
  stafflines: StafflineSet[];
  /** GameraXML exported from the Interactive Classifier, one per page --
   * metadata only. The document itself is fetched on demand from
   * ic_api.py's /ic-xml/{id} endpoints, since a page's export runs to
   * megabytes (one RLE glyph mask each) and would otherwise ride along
   * with every project list request. */
  icXmlFiles: IcXmlFile[];
}

/** Which tab/sub-tab ProjectDetail should open on mount -- set by App.tsx's
 * job-done toast handler when routing "view" for a succeeded job to the tab
 * that actually holds what it produced (issue #196). Consumed once via
 * ProjectDetail's onInitialTabConsumed and then cleared, so an unrelated
 * later navigation back to the project page doesn't re-apply a stale value. */
export interface ProjectInitialTab {
  tab: "images" | "models" | "generated";
  subTab?: "annotations" | "text" | "stafflines" | "ic xml" | "mei files";
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
  hasClassifierImage?: boolean;
  /** True when this detection's medieval-preset predict run fell back to
   * raw-page staffline detection because paco-classifier-service was
   * unreachable/erroring/timed out. Derived from settings_json.source_label
   * server-side, not from classifierError, so pre-existing fallback rows
   * still surface this even without a stored reason. */
  hasClassifierFallback?: boolean;
  /** Short categorized reason for the fallback above (e.g. "timeout: ...",
   * "unreachable: ..."); null/undefined on rows written before this was
   * added, or when hasClassifierFallback is false. */
  classifierError?: string | null;
  /** mothra#286: paco-classifier's OTHER output layer -- the
   * background-only PNG, the sibling to the stafflines-only layer
   * hasClassifierImage already exposes. Never true without
   * hasClassifierImage also being true (both come from the same
   * classify_stafflines call) -- see GET .../background-image. */
  hasBackgroundImage?: boolean;
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

/** One saved Interactive Classifier session, as
 * `GET /api/projects/{id}/ic/sessions` returns it (ic_api.py's `ic_sessions`
 * re-shapes IC's own snake_case payload). Sessions live in IC's store, not
 * mothra's DB, so this is the only way to know a project has any. */
export interface IcSessionSummary {
  sessionId: string;
  /** IC's own lifecycle state -- "classifying" (resumable) or "export"
   * (terminal/read-only; only pre-existing sessions should still be here,
   * see ic_api.py's ic_complete on finalize=false). */
  state?: string | null;
  /** IC's stored page name, a file-name stem -- "" if it recorded none. */
  sourceName: string;
  /** mothra's project_images.id, recorded when the page was staged. Null on
   * sessions saved before IC tracked it, which can then only be matched by
   * `sourceName`. */
  imageId?: string | null;
  glyphCount?: number | null;
  updatedAt?: string | null;
}

/** One page's exported Interactive Classifier GameraXML (metadata only --
 * see Project.icXmlFiles). Written server-side at export time, so it exists
 * for auto-classified pages too, not just ones opened in the classifier. */
export interface IcXmlFile {
  id: string;
  /** File name the download uses -- the page's stem plus ".xml". */
  name: string;
  imageId?: string | null;
  imageName: string;
  imageSrc?: string | null;
  /** `<glyph>` elements in the document; null if it couldn't be counted. */
  glyphCount?: number | null;
  byteSize?: number | null;
  createdAt?: string | null;
}

export interface ProjectModel {
  id: string;
  name: string;
  kind: ModelKind;
  classMap?: Record<string, string> | null;
}

export interface AnnotationSet {
  id: string;
  imageId?: string;
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
  imageId?: string;
  imageName?: string;
  staveSource?: StaveSource | null;
  /** When this revision was encoded. `mei_files` is append-only, so a page
   * encoded twice has two rows; this is how utils/mei.ts's latestMeiPerImage
   * tells them apart. Undefined on a row built client-side straight after an
   * encode (useEncodingFlow), where list order already says it's newest. */
  createdAt?: string | null;
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
  | "ic-auto"
  | "ic-completion"
  | "encoding-processing"
  | "encoding-completion"
  | "send-completion"
  | "neon-editor"
  | "neon-completion";
