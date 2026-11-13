# Embedded Neon update — issue #215

Date: 2026-11-13 (America/Toronto)

The delivery uses official Neon develop including merged #1410, plus the same
reviewed pending-save correction as the previous delivery. A fresh full build
and all 13 local integration tests passed: the original six successful-save and
navigation checks, and seven checks of #1410 inside Mothra's production iframe.
No push, pull request, merge or deployment was performed.

## Exact versions

- Original Mothra base: `7ac370af52ec51da82b95bea633a0904daffc621`.
- Previous Mothra delivery: `9d315cbca7aaad462d900df47b33441e046fbf66`.
- Previous Neon delivery: `1b6e34ab99cfcdbfc444f5315467ebf3844e3bc3`.
- Official develop queried on 2026-11-13:
  `7f6322757a2002efa29d5a9b11793a07f7d08c9a`. It matches the reviewed base,
  with no newer commits. Its merge includes #1410 head `8b89eb18`.
- Selected Neon: `a7b9dc66aff78897f59fb2e5be5b4b33991d328b`, on local branch
  `fix/215-neon-1410-cache-dirty`, with parent `7f632275`.
- Previous versions are preserved by `baseline/215-before-1410` in both clones.
  Mothra continues on `fix/215-update-embedded-neon`.

Official develop still calls `loadData(pageURI, cached.mei)` without preserving
its dirty state. Therefore the pending-save correction was still necessary.
The exact patch from previous delivery `1b6e34ab` was cherry-picked onto the new
base. Original author, author date, coauthor and source `2a46faaf` are preserved,
with an additional source record for `1b6e34ab`. The diff against official develop
is only four added lines and one removed line in `NeonCore.ts`; the patch matches
the previous delivery byte-for-byte. `loadPage` passes `cached.dirty` to `loadData`.

Relative to the previous Neon delivery, the source changes are precisely #1410's
18 changed paths. There are no extra product fixes. Existing Mothra wrapper and
save integration changes remain. No notation/subtype migration, mapping, shared
zone changes, discarded-edit reopening changes or Diva selection fixes are added.

## Build and dependencies

The normal `landing-page` `npm run build` passed under Node 22.16.0, including
Neon webpack, TypeScript and Vite. The generated webpack cache was cleared before
building. The final editor displays `a7b9dc66`. The existing large-chunk warning
remains. The development webpack configuration and local Verovio worker are
unchanged; production-mode loading of an online Verovio URL is not used.

The local Verovio version remains `6.4.0-dev-ce49d38`, SHA256
`7689c9549b49a0f233bca23f3ea2b9e9951f2ad49a2d92b8865a824aace8f722`.
All source/public/dist copies match. Of 94 Neon assets, 91 static assets remain
identical to the previous reviewed candidate; the three JS bundles changed for
#1410 and version metadata. Final editor SHA256:
`f40ff5f5dbdf2a5c720fa26356cc2d2a0ff02cb640e48f9bfe0d86ec23ff70b9`.
Build output, public assets and dist assets match. Details are in
`NEON_215_ASSET_COMPARISON.json`.

## Fresh validation results

| Check | Result |
| --- | --- |
| Original six integration tests | 6 passed, 0 failed, 0 skipped |
| Both conversion directions through direct Save | Correct raw MEI PUT; new session identities and fresh GET XML match saved data |
| Mark Done after conversion | Successful PUT response completes before corrected PATCH |
| Native Hufnagel connection editing, save and reopen | Changed connection state persists; PUT-response ordering checked |
| Initial notation and unsaved-navigation dialog | Existing UI/no-write assertions pass; not a full discarded-data consistency claim |
| Embedded #1410 sizing, Square and Hufnagel | 2 passed; fixed size through 25%, 100%, 400% Neon Zoom |
| Legal staff moves with off-image handles | 3 passed: normal page at 100%/25%, and ten-times-smaller page at 25% |
| Actual glyph/staff-line out-of-image moves | 2 passed; refused before toolkit edit, path restored, MEI content unchanged |

The new spec adapts #1410 `cypress/e2e/edit-handles.cy.ts` and
`cypress/support/neon-mei.ts`, using the existing Square/Hufnagel samples,
scaled-surface fixture technique, CSS measurements, exported-MEI comparison and
drag gestures. It mounts the real Mothra wrapper and uses the actual built iframe,
fonts and toolkit. Size/Zoom/reselection issue no edit action, and exported MEI
music and facsimile zones remain identical apart from regenerated page-break IDs.

### Measured CSS pixels

Measurements use `getBoundingClientRect` inside the iframe. At handle size 6,
Square and Hufnagel both measure the following (rounded to three decimals):

| Neon Zoom | Left circle diameter | Left diamond width × height | Diamond center offset from left circle center |
| --- | --- | --- | --- |
| 25% | 12.000 px | 14.400 × 16.800 px | −24.000, −12.000 px |
| 100% | 12.000 px | 14.400 × 16.800 px | −24.000, −12.000 px |
| 400% | 12.000 px | 14.400 × 16.800 px | −24.000, −12.000 px |

With text bboxes switched on and back off, the slider stays visible and usable.
At size 10 the circle is 20.000 px, diamond 24.000 × 28.000 px and center offset
−40.000, −20.000 px; reselection preserves these values. Raw measurements and
exported staff-zone before/after values are in `NEON_215_HANDLES_MEASUREMENTS.json`.
For the normal-page legal left drag, ulx changes 623 → 81 at 100% and 623 → 74
at 25%; the small-page legal drag changes ulx 62 → 81. Gesture rounding tolerance
is expressed in content units from the actual screen scale, as in upstream tests.

### Initial test failures and their classification

The first run had three failures in the newly adapted tests because Cypress's
DOM-aware `exist` assertion was used for a detached XML node. Using upstream's
identity assertion resolved them; they were test errors, not failed drag edits.
The Mark Done test also encountered an upstream-response callback on a script
request canceled during iframe remount/navigation. The existing observer was
changed to fetch the built script once and serve the observed copy directly,
without an upstream callback on canceled requests. It still forwards original
Worker messages unchanged. No product code was changed for either test issue.
All 13 tests then ran successfully together, without skips or error suppression.

### Limits and untested items

API responses are locally simulated. Results establish successful saves with OK
responses; production database persistence, authentication, save failures,
timeouts and local fallback are untested. Discarded-edit reopening, shared-zone
behavior and Diva second-page selection were not expanded or repaired. The new
#1410 integration checks cover the SingleView image path used by Mothra; they do
not claim an additional Diva or whole-upstream-suite validation. Measurements
cover the left-side representative controls and the stated Zoom levels.

## Reproduce and publish later

Use `landing-page/tests/neon-editor/README.md`. The repository contains all tests
and sample references; no personal-workspace absolute paths are needed.
The suite writes `tests/neon-editor/artifacts/handle-results.json` for measurements.

The new Neon commit is local only. GitHub's commit API returned 422 / No commit
found for this SHA on 2026-11-13. It must become reachable from the configured
submodule remote `https://github.com/DDMAL/Neon.git` before another clone or CI can
fetch the Mothra pin. Publishing Neon first is a future authorized step, followed
by Mothra. This delivery does not push, open a PR, merge or deploy.
