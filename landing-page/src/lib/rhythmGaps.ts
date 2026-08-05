// Client-side reconstruction of the "gap distribution" diagnostic that
// staff-finding/scripts/group_staves.py plots server-side (its
// plot_gap_distribution helper, run via run_page.py) -- for stafflines run
// through the landing-page pipeline, group_staves.py's own
// GroupingResult.cut_threshold_px is computed but never persisted onto a
// JsomrLineRecord (see staffline_stage.py's _assemble_jsomr_records), so
// this recomputes an equivalent threshold from what *is* stored
// (centerline_page.y_values, scale_unit, stave_id, rhythm_status) rather
// than requiring a backend change. Multipliers mirror group_staves.py's own
// MIN_GAP_MULTIPLIER / CUT_THRESHOLD_MULTIPLIER constants -- keep in sync if
// those ever change.
import type { JsomrLineRecord } from "../types";

const MIN_GAP_MULTIPLIER = 0.5;
const CUT_THRESHOLD_MULTIPLIER = 1.5;

export interface RhythmGapBar {
  staveId: number;
  gapIndex: number;
  gapPx: number;
  isAnomalous: boolean;
}

export interface RhythmGapSummary {
  bars: RhythmGapBar[];
  scaleUnit: number;
  noiseFloorPx: number;
  cutThresholdPx: number;
  maxInterpGapPx: number;
}

function meanY(record: JsomrLineRecord): number | null {
  const values = record.centerline_page?.y_values;
  if (!values || values.length === 0) return null;
  return values.reduce((sum, y) => sum + y, 0) / values.length;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** Consecutive-line-gap distribution across every stave on the page, plus
 * the reference thresholds group_staves.py would have used to group those
 * same lines -- null if there isn't enough data to compute anything
 * (fewer than two lines with a known stave assignment). */
export function computeRhythmGaps(records: JsomrLineRecord[]): RhythmGapSummary | null {
  const byStave = new Map<number, JsomrLineRecord[]>();
  for (const r of records) {
    if (r.stave_id === null) continue;
    const list = byStave.get(r.stave_id) ?? [];
    list.push(r);
    byStave.set(r.stave_id, list);
  }
  if (byStave.size === 0) return null;

  const scaleUnit = records.find((r) => r.scale_unit)?.scale_unit ?? 0;

  const bars: RhythmGapBar[] = [];
  const allGaps: number[] = [];
  for (const [staveId, lines] of byStave) {
    const sorted = [...lines].sort(
      (a, b) => (a.within_stave_index ?? 0) - (b.within_stave_index ?? 0),
    );
    const isAnomalous = sorted.some((r) => r.rhythm_status);
    const ys = sorted.map(meanY).filter((y): y is number => y !== null);
    for (let i = 1; i < ys.length; i++) {
      const gapPx = Math.abs(ys[i] - ys[i - 1]);
      allGaps.push(gapPx);
      bars.push({ staveId, gapIndex: bars.length, gapPx, isAnomalous });
    }
  }
  if (bars.length === 0) return null;

  const medianGap = median(allGaps);
  const noiseFloorPx = scaleUnit * MIN_GAP_MULTIPLIER;
  const cutThresholdPx = Math.max(medianGap * CUT_THRESHOLD_MULTIPLIER, scaleUnit);
  const maxInterpGapPx = Math.max(...allGaps, cutThresholdPx);

  return { bars, scaleUnit, noiseFloorPx, cutThresholdPx, maxInterpGapPx };
}
