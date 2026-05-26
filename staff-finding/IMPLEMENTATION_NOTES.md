## Implementation Summary: group_staves() Function

### What Was Fixed

The `group_staves()` function in [group_staves.py](group_staves.py) was raising `NotImplementedError`. It is now fully implemented following ADR-002 strategy for stave grouping.

### Implementation Details

The implementation follows a **simple, inspectable ratio-based gap analysis** approach (ADR-002 §1-2):

#### Core Algorithm (8 Stages):

1. **Extract y-positions** - Get y-value at horizontal center of each fit's centerline
   - Fits with no y_values are excluded but recorded with flags
   
2. **Sort by y-position** - Order fits top-to-bottom
   
3. **Compute gaps** - Calculate consecutive differences between sorted y-positions
   - All gaps stored in `gap_distribution` for diagnostic evidence
   
4. **Determine cut threshold** - Separate intra-stave from inter-stave gaps
   - Formula: `median_gap × CUT_THRESHOLD_MULTIPLIER` (default 1.5)
   - Floored at `scale_unit` (sanity check per ADR-002 §2.4)
   
5. **Segment into staves** - Group consecutive fits where gaps < cut_threshold
   - Gaps ≥ cut_threshold trigger stave boundaries
   
6. **Assign IDs & indices** - Label each fit with stave_id (0-based, top to bottom) and within_stave_index
   - Preserve original input order in output assignments
   
7. **Analyze line counts** - Compute distribution and flag anomalies
   - Flags: `mode_count_below_typical` if mode < 4 lines
   - Flags: `staves_with_unexpected_count` if any stave differs from mode by >1
   
8. **Generate diagnostics** (optional) - Render page with stave assignments
   - Shows centerlines colored by stave, gap distribution chart, cut threshold marker

#### Helper Functions Implemented:

- **`_y_at_fit_center()`** - Extract representative y-position at horizontal midpoint
- **`_compute_gap_distribution()`** - Calculate consecutive y-gaps (utility)
- **`_determine_cut_threshold()`** - Calculate median-ratio threshold (utility)
- **`_assign_staves()`** - Walk fits and assign stave IDs (utility)
- **`_save_grouping_diagnostic()`** - Render diagnostic visualization

### ADR-002 References in Code

All new code includes inline comments referencing relevant ADR-002 sections:

- §1-2: Simple ratio-based vs. learned methods
- §2.1: Y-position extraction strategy
- §2.3: Median-ratio thresholding
- §2.4: Scale-unit sanity floor
- §2.5: Gap threshold for stave segmentation
- §2.6: Stave ID assignment
- §3: Line count distribution anomaly detection
- §3.2: Unexpected count flagging
- §4: Missing line interpolation (deferred, flagged instead)
- §5: Diagnostic preservation for QA

### Test Results

All unit tests pass:
- ✓ Basic stave grouping (2 staves, 4 lines each)
- ✓ Empty fits list handling
- ✓ Fits with no y_values handling

### Design Philosophy

Following ADR-001 principles:
- **Permissive filtering**: Keeps valid groupings even with imperfect inputs
- **Preserved evidence**: Returns gap_distribution, cut_threshold, and flags for inspection
- **Downstrem robustness**: Robust fitting handles local asymmetries; grouping doesn't need to

### Edge Cases Handled

1. No fits available → flags `no_fits_available`
2. All fits have no y_values → flags `no_fits_with_y_positions`
3. Single fit → one stave with one line
4. All fits same y-position → all in one stave
5. Fits with missing y_values → marked unassigned, flagged

### Usage

```python
from group_staves import group_staves

result = group_staves(
    fits=fit_results,
    scale_unit=h_scale,
    save_path=output_diagnostic_path,
    page_size=(page_width, page_height),
    page_image=bgr_processed_page,
)

print(f"Staves: {result.mode_lines_per_stave} lines/stave")
print(f"Distribution: {result.line_count_distribution}")
print(f"Flags: {result.flags}")
for assignment in result.assignments:
    print(f"  Fit {assignment.fit_index} → Stave {assignment.stave_id}, line {assignment.within_stave_index}")
```

### Notes

- Default behavior: flag missing lines instead of synthesizing (interpolate_missing=False per ADR-002)
- Diagnostic visualization uses matplotlib and requires page_size or page_image
- All parameters documented with examples and edge cases
- Comments link implementation to ADR-002 decision points for maintenance
