import { type RhythmGapSummary } from "../../lib/rhythmGaps";

// Inline-SVG re-creation of staff-finding/scripts/group_staves.py's
// "Gap Distribution (Consecutive Fits)" diagnostic plot (normally only
// produced by run_page.py, off the landing-page request path -- see
// rhythmGaps.ts's module comment). No charting library: this is one bar
// chart, consistent with the rest of the codebase avoiding new heavy deps
// for a single visual.

const PALETTE = ["#4AADAA", "#FFA500", "#E87BF7", "#F76B6B", "#6BF7A5", "#F7E16B"];
const ANOMALY_COLOR = "#FF3B30";
const NOISE_FLOOR_COLOR = "#1D3335";
const CUT_THRESHOLD_COLOR = "#FF3B30";
const MAX_INTERP_COLOR = "#FFA500";

const CHART_H = 260;
const PAD_L = 44;
const PAD_R = 16;
const PAD_T = 16;
const PAD_B = 28;
const BAR_GAP = 3;
const MIN_BAR_W = 4;

interface Props {
  summary: RhythmGapSummary;
}

export default function RhythmChart({ summary }: Props) {
  const { bars, noiseFloorPx, cutThresholdPx, maxInterpGapPx } = summary;
  const yMax = Math.max(maxInterpGapPx, ...bars.map((b) => b.gapPx)) * 1.08 || 1;
  const barW = Math.max(MIN_BAR_W, 12);
  const chartW = PAD_L + PAD_R + bars.length * (barW + BAR_GAP);
  const plotH = CHART_H - PAD_T - PAD_B;

  const yToPx = (v: number) => PAD_T + plotH - (v / yMax) * plotH;

  return (
    <div className="w-full overflow-x-auto">
      <svg
        viewBox={`0 0 ${chartW} ${CHART_H}`}
        width={chartW}
        height={CHART_H}
        className="min-w-full"
      >
        {/* axes */}
        <line x1={PAD_L} y1={PAD_T} x2={PAD_L} y2={PAD_T + plotH} stroke="#1D3335" strokeWidth={1} />
        <line
          x1={PAD_L}
          y1={PAD_T + plotH}
          x2={chartW - PAD_R}
          y2={PAD_T + plotH}
          stroke="#1D3335"
          strokeWidth={1}
        />
        {[0, 0.25, 0.5, 0.75, 1].map((f) => (
          <text
            key={f}
            x={PAD_L - 6}
            y={yToPx(yMax * f) + 3}
            textAnchor="end"
            fontSize={9}
            fontFamily="monospace"
            fill="#1D3335"
            opacity={0.6}
          >
            {Math.round(yMax * f)}
          </text>
        ))}

        {/* reference lines */}
        <line
          x1={PAD_L}
          y1={yToPx(noiseFloorPx)}
          x2={chartW - PAD_R}
          y2={yToPx(noiseFloorPx)}
          stroke={NOISE_FLOOR_COLOR}
          strokeOpacity={0.35}
          strokeDasharray="2,2"
        />
        <line
          x1={PAD_L}
          y1={yToPx(cutThresholdPx)}
          x2={chartW - PAD_R}
          y2={yToPx(cutThresholdPx)}
          stroke={CUT_THRESHOLD_COLOR}
          strokeDasharray="5,3"
        />
        <line
          x1={PAD_L}
          y1={yToPx(maxInterpGapPx)}
          x2={chartW - PAD_R}
          y2={yToPx(maxInterpGapPx)}
          stroke={MAX_INTERP_COLOR}
          strokeDasharray="5,3"
        />

        {/* bars */}
        {bars.map((b, i) => {
          const x = PAD_L + i * (barW + BAR_GAP);
          const y = yToPx(b.gapPx);
          const h = PAD_T + plotH - y;
          const color = b.isAnomalous ? ANOMALY_COLOR : PALETTE[b.staveId % PALETTE.length];
          return <rect key={i} x={x} y={y} width={barW} height={Math.max(h, 0.5)} fill={color} />;
        })}
      </svg>
      <div className="flex flex-wrap gap-x-4 gap-y-1 mt-2 text-[10px] font-mono text-[#1D3335]/70">
        <span>
          <span style={{ color: NOISE_FLOOR_COLOR }}>┅</span> noise floor ({noiseFloorPx.toFixed(1)}px)
        </span>
        <span>
          <span style={{ color: CUT_THRESHOLD_COLOR }}>┅</span> cut threshold ({cutThresholdPx.toFixed(1)}px)
        </span>
        <span>
          <span style={{ color: MAX_INTERP_COLOR }}>┅</span> max interp. gap ({maxInterpGapPx.toFixed(1)}px)
        </span>
        <span>
          <span style={{ color: ANOMALY_COLOR }}>■</span> flagged stave
        </span>
      </div>
      <p className="mt-2 text-[#1D3335]/50 text-[11px] font-mono">
        approximated client-side from stored per-line fit data — thresholds mirror
        group_staves.py's own multipliers, not its exact valley-detection pass
      </p>
    </div>
  );
}
