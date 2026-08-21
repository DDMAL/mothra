// Converts a raw YOLO-format annotation .txt (one "cls cx cy bw bh" line per
// detection -- see AnnotationViewerModal.tsx's parseYolo) into the same JSON
// shape ProjectDetail.tsx's bulk ".json" download and the single-annotation
// ".json" download in AppRouter.tsx already produce. Shared here so all three
// call sites stay in sync instead of each carrying their own copy.
export function yoloTxtToJson(yoloTxt: string, imageName: string): string {
  // Trim each line and split on any run of whitespace -- matching
  // AnnotationViewerModal.tsx's own parseYolo -- so multi-space/tab-
  // separated fields and CRLF line endings parse the same way in both
  // places, and a trailing/whitespace-only line can't sneak through
  // filter(Boolean) and produce a bogus NaN-filled annotation.
  const annotations = yoloTxt
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
    .map((line) => {
      const [cls, x, y, w, h] = line.split(/\s+/).map(Number);
      return { class: cls, x_center: x, y_center: y, width: w, height: h };
    });
  return JSON.stringify({ imageName, annotations }, null, 2);
}
